import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, Union
from src.registry.models import DatasetRecord, DatasetStatus, DatasetVersion, now_utc_iso


class DatasetRegistryStore:
    """
    SQLite-backed persistent store for dataset lifecycle and metadata registry.
    Strictly standard library sqlite3; ensures data persistence and transactional integrity.
    Explicitly closes connections to avoid Windows file locks.
    """

    def __init__(self, db_path: Union[str, Path] = "storage/registry.db"):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    active_version INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dataset_versions (
                    dataset_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    format TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    schema_profile_json TEXT,
                    upload_timestamp TEXT NOT NULL,
                    processing_timestamp TEXT,
                    activation_timestamp TEXT,
                    error_message TEXT,
                    provenance_json TEXT,
                    PRIMARY KEY (dataset_id, version),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_versions_dataset_id
                ON dataset_versions(dataset_id);
                """
            )

    def create_dataset(self, record: DatasetRecord) -> None:
        """Inserts a new top-level dataset entry."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO datasets (dataset_id, name, description, active_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    record.dataset_id,
                    record.name,
                    record.description,
                    record.active_version,
                    record.created_at,
                    record.updated_at,
                ),
            )

    def save_version(self, ver: DatasetVersion) -> None:
        """Inserts or replaces a dataset version record."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO dataset_versions (
                    dataset_id, version, filename, storage_path, format,
                    file_size_bytes, checksum_sha256, status, schema_profile_json,
                    upload_timestamp, processing_timestamp, activation_timestamp,
                    error_message, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    ver.dataset_id,
                    ver.version,
                    ver.filename,
                    ver.storage_path,
                    ver.format,
                    ver.file_size_bytes,
                    ver.checksum_sha256,
                    ver.status,
                    json.dumps(ver.schema_profile),
                    ver.upload_timestamp,
                    ver.processing_timestamp,
                    ver.activation_timestamp,
                    ver.error_message,
                    json.dumps(ver.provenance) if ver.provenance else None,
                ),
            )

    def set_active_version(self, dataset_id: str, version: int) -> None:
        """
        Atomically switches the active version of a dataset, marking previous active version
        as SUPERSEDED and new active version as ACTIVE.
        """
        now = now_utc_iso()
        with self._connection() as conn:
            # 1. Supersede any existing active version
            conn.execute(
                """
                UPDATE dataset_versions
                SET status = ?
                WHERE dataset_id = ? AND status = ?;
                """,
                (DatasetStatus.SUPERSEDED, dataset_id, DatasetStatus.ACTIVE),
            )

            # 2. Mark target version active
            conn.execute(
                """
                UPDATE dataset_versions
                SET status = ?, activation_timestamp = ?
                WHERE dataset_id = ? AND version = ?;
                """,
                (DatasetStatus.ACTIVE, now, dataset_id, version),
            )

            # 3. Update dataset master record
            conn.execute(
                """
                UPDATE datasets
                SET active_version = ?, updated_at = ?
                WHERE dataset_id = ?;
                """,
                (version, now, dataset_id),
            )

    def update_version_status(
        self,
        dataset_id: str,
        version: int,
        status: str,
        error_message: Optional[str] = None,
        activation_timestamp: Optional[str] = None,
        processing_timestamp: Optional[str] = None,
    ) -> None:
        """Updates the status and timestamps of a specific version."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE dataset_versions
                SET status = ?,
                    error_message = coalesce(?, error_message),
                    activation_timestamp = coalesce(?, activation_timestamp),
                    processing_timestamp = coalesce(?, processing_timestamp)
                WHERE dataset_id = ? AND version = ?;
                """,
                (
                    status,
                    error_message,
                    activation_timestamp,
                    processing_timestamp,
                    dataset_id,
                    version,
                ),
            )

    def get_dataset(self, dataset_id: str) -> Optional[DatasetRecord]:
        """Retrieves dataset record with all associated versions."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?;", (dataset_id,)
            ).fetchone()
            if not row:
                return None

            v_rows = conn.execute(
                "SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY version ASC;",
                (dataset_id,),
            ).fetchall()

            versions = [
                DatasetVersion(
                    dataset_id=r["dataset_id"],
                    version=r["version"],
                    filename=r["filename"],
                    storage_path=r["storage_path"],
                    format=r["format"],
                    file_size_bytes=r["file_size_bytes"],
                    checksum_sha256=r["checksum_sha256"],
                    status=r["status"],
                    schema_profile=json.loads(r["schema_profile_json"] or "{}"),
                    upload_timestamp=r["upload_timestamp"],
                    processing_timestamp=r["processing_timestamp"],
                    activation_timestamp=r["activation_timestamp"],
                    error_message=r["error_message"],
                    provenance=json.loads(r["provenance_json"]) if r["provenance_json"] else None,
                )
                for r in v_rows
            ]

            return DatasetRecord(
                dataset_id=row["dataset_id"],
                name=row["name"],
                description=row["description"],
                active_version=row["active_version"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                versions=versions,
            )

    def list_datasets(self) -> List[DatasetRecord]:
        """Lists all registered datasets with metadata."""
        with self._connection() as conn:
            rows = conn.execute("SELECT dataset_id FROM datasets ORDER BY created_at DESC;").fetchall()

        results: List[DatasetRecord] = []
        for r in rows:
            ds = self.get_dataset(r["dataset_id"])
            if ds:
                results.append(ds)
        return results

    def get_version(self, dataset_id: str, version: int) -> Optional[DatasetVersion]:
        """Retrieves a specific version record."""
        with self._connection() as conn:
            r = conn.execute(
                "SELECT * FROM dataset_versions WHERE dataset_id = ? AND version = ?;",
                (dataset_id, version),
            ).fetchone()
            if not r:
                return None

            return DatasetVersion(
                dataset_id=r["dataset_id"],
                version=r["version"],
                filename=r["filename"],
                storage_path=r["storage_path"],
                format=r["format"],
                file_size_bytes=r["file_size_bytes"],
                checksum_sha256=r["checksum_sha256"],
                status=r["status"],
                schema_profile=json.loads(r["schema_profile_json"] or "{}"),
                upload_timestamp=r["upload_timestamp"],
                processing_timestamp=r["processing_timestamp"],
                activation_timestamp=r["activation_timestamp"],
                error_message=r["error_message"],
                provenance=json.loads(r["provenance_json"]) if r["provenance_json"] else None,
            )

    def get_active_version(self, dataset_id: str) -> Optional[DatasetVersion]:
        """Returns the currently active DatasetVersion for a dataset."""
        ds = self.get_dataset(dataset_id)
        if not ds or ds.active_version is None:
            return None
        return self.get_version(dataset_id, ds.active_version)
