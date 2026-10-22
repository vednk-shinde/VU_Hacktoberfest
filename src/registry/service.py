import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.ingestion.profile import profile_dataset
from src.registry.models import DatasetRecord, DatasetStatus, DatasetVersion, now_utc_iso
from src.registry.store import DatasetRegistryStore
from src.storage.manager import StorageManager, sanitize_filename


def detect_schema_diff(
    old_schema: Dict[str, Any],
    new_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Detects differences in columns and types between two schema profiles."""
    old_cols = set(old_schema.get("columns", []))
    new_cols = set(new_schema.get("columns", []))

    added_cols = sorted(list(new_cols - old_cols))
    removed_cols = sorted(list(old_cols - new_cols))

    old_types = old_schema.get("column_types", {})
    new_types = new_schema.get("column_types", {})

    type_changes = {}
    for col in old_cols.intersection(new_cols):
        ot = old_types.get(col)
        nt = new_types.get(col)
        if ot != nt:
            type_changes[col] = {"from": ot, "to": nt}

    return {
        "added_columns": added_cols,
        "removed_columns": removed_cols,
        "type_changes": type_changes,
        "row_count_delta": new_schema.get("row_count", 0) - old_schema.get("row_count", 0),
    }


class DatasetRegistryService:
    """
    High-level lifecycle manager for enterprise datasets.
    Implements Operation 1 (Add Dataset) and Operation 2 (Replace Dataset) with
    staged validation, SHA-256 integrity, atomic activation, and safe rollback.
    """

    def __init__(
        self,
        store: Optional[DatasetRegistryStore] = None,
        storage_manager: Optional[StorageManager] = None,
    ):
        self.store = store or DatasetRegistryStore()
        self.storage = storage_manager or StorageManager()

    def add_dataset(
        self,
        file_path: Union[str, Path],
        name: str,
        description: str = "",
        dataset_id: Optional[str] = None,
    ) -> DatasetVersion:
        """
        Operation 1: Add a new dataset without affecting existing datasets.
        """
        src_path = Path(file_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {src_path}")

        ext = src_path.suffix.lower()
        if ext not in {".csv", ".json"}:
            raise ValueError("Supported formats: CSV and JSON.")

        # Assign unique dataset ID if not provided
        d_id = dataset_id or f"ds_{uuid.uuid4().hex[:12]}"
        d_id = sanitize_filename(d_id)

        # Check for duplicate dataset ID
        if self.store.get_dataset(d_id) is not None:
            raise ValueError(f"Dataset ID '{d_id}' already exists in registry.")

        now = now_utc_iso()
        version_num = 1
        safe_fname = sanitize_filename(src_path.name)

        # 1. Stage the file
        staged_path, checksum, file_size = self.storage.stage_file(
            source_path=src_path,
            dataset_id=d_id,
            version=version_num,
            original_filename=safe_fname,
        )

        # 2. Profile and validate schema
        try:
            profile = profile_dataset(staged_path)
        except Exception as err:
            self.storage.cleanup_staging(d_id, version_num)
            raise ValueError(f"Dataset validation failed during profiling: {err}") from err

        # 3. Promote staged file to canonical storage
        final_path = self.storage.commit_staged_file(d_id, version_num, safe_fname)

        # 4. Create database records
        dataset_record = DatasetRecord(
            dataset_id=d_id,
            name=name,
            description=description,
            active_version=version_num,
            created_at=now,
            updated_at=now,
        )
        self.store.create_dataset(dataset_record)

        version_record = DatasetVersion(
            dataset_id=d_id,
            version=version_num,
            filename=safe_fname,
            storage_path=str(final_path),
            format=ext.lstrip("."),
            file_size_bytes=file_size,
            checksum_sha256=checksum,
            status=DatasetStatus.ACTIVE,
            schema_profile=profile,
            upload_timestamp=now,
            processing_timestamp=now,
            activation_timestamp=now,
            provenance={"action": "add_dataset", "source_file": src_path.name},
        )
        self.store.save_version(version_record)

        return version_record

    def replace_dataset(
        self,
        dataset_id: str,
        replacement_file_path: Union[str, Path],
        replacement_filename: Optional[str] = None,
    ) -> DatasetVersion:
        """
        Operation 2: Stage, validate, and atomically replace an existing dataset.
        If validation fails, the active version is preserved completely intact (rollback).
        """
        d_id = sanitize_filename(dataset_id)
        current_record = self.store.get_dataset(d_id)
        if not current_record:
            raise KeyError(f"Dataset '{d_id}' not found in registry.")

        src_path = Path(replacement_file_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Replacement dataset file not found: {src_path}")

        ext = src_path.suffix.lower()
        if ext not in {".csv", ".json"}:
            raise ValueError("Supported formats: CSV and JSON.")

        # Determine next candidate version number
        existing_versions = [v.version for v in current_record.versions]
        candidate_version = (max(existing_versions) if existing_versions else 0) + 1

        now = now_utc_iso()
        safe_fname = sanitize_filename(replacement_filename or src_path.name)

        # 1. Stage the candidate version
        try:
            staged_path, checksum, file_size = self.storage.stage_file(
                source_path=src_path,
                dataset_id=d_id,
                version=candidate_version,
                original_filename=safe_fname,
            )
        except Exception as err:
            raise ValueError(f"Failed to stage replacement file: {err}") from err

        # Record candidate version as PROCESSING
        candidate_record = DatasetVersion(
            dataset_id=d_id,
            version=candidate_version,
            filename=safe_fname,
            storage_path=str(staged_path),
            format=ext.lstrip("."),
            file_size_bytes=file_size,
            checksum_sha256=checksum,
            status=DatasetStatus.PROCESSING,
            schema_profile={},
            upload_timestamp=now,
            provenance={"action": "replace_dataset", "source_file": src_path.name},
        )
        self.store.save_version(candidate_record)

        # 2. Validate and profile candidate file
        try:
            profile = profile_dataset(staged_path)
            candidate_record.schema_profile = profile
            candidate_record.processing_timestamp = now_utc_iso()

            # Schema diff analysis with currently active version
            active_ver = self.store.get_active_version(d_id)
            if active_ver and active_ver.schema_profile:
                diff = detect_schema_diff(active_ver.schema_profile, profile)
                candidate_record.provenance["schema_diff"] = diff

        except Exception as val_err:
            # ROLLBACK: Clean up staging, mark candidate version as FAILED, retain active version
            self.storage.cleanup_staging(d_id, candidate_version)
            self.store.update_version_status(
                dataset_id=d_id,
                version=candidate_version,
                status=DatasetStatus.FAILED,
                error_message=str(val_err),
            )
            raise ValueError(
                f"Replacement validation failed for dataset '{d_id}' (v{candidate_version}). "
                f"Active version v{current_record.active_version} retained. Error: {val_err}"
            ) from val_err

        # 3. Promote staged file to canonical storage directory
        final_path = self.storage.commit_staged_file(d_id, candidate_version, safe_fname)
        candidate_record.storage_path = str(final_path)
        candidate_record.status = DatasetStatus.VALIDATED
        self.store.save_version(candidate_record)

        # 4. Atomically switch active version (previous active is marked SUPERSEDED)
        self.store.set_active_version(d_id, candidate_version)
        candidate_record.status = DatasetStatus.ACTIVE
        candidate_record.activation_timestamp = now_utc_iso()

        return candidate_record

    def get_active_dataset(self, dataset_id: str) -> DatasetVersion:
        """Returns the active DatasetVersion for a dataset ID."""
        d_id = sanitize_filename(dataset_id)
        ver = self.store.get_active_version(d_id)
        if not ver:
            raise KeyError(f"No active version found for dataset '{d_id}'.")
        return ver

    def get_version_history(self, dataset_id: str) -> List[DatasetVersion]:
        """Returns the full chronological version history for a dataset."""
        d_id = sanitize_filename(dataset_id)
        ds = self.store.get_dataset(d_id)
        if not ds:
            raise KeyError(f"Dataset '{d_id}' not found.")
        return ds.versions

    def list_datasets(self) -> List[DatasetRecord]:
        """Lists all registered datasets."""
        return self.store.list_datasets()
