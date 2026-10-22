import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union


INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes an uploaded filename to prevent directory traversal,
    null bytes, and platform-specific control characters.
    """
    if not filename:
        return "dataset.csv"

    # Extract only the base name (strips ../, C:\, etc.)
    base = Path(filename).name
    # Strip dangerous characters
    cleaned = INVALID_FILENAME_CHARS.sub("_", base).strip(" .")

    if not cleaned:
        cleaned = "dataset"

    # Preserve extension if present, otherwise default to .bin
    ext = Path(base).suffix.lower()
    if not Path(cleaned).suffix and ext:
        cleaned = f"{cleaned}{ext}"

    return cleaned


def ensure_within_directory(base_dir: Path, target_path: Path) -> Path:
    """
    Verifies that target_path resolves strictly within base_dir.
    Raises ValueError if a path traversal attempt is detected.
    """
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()

    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as err:
        raise ValueError(
            f"Path traversal detected: '{target_path}' is outside storage root '{base_dir}'."
        ) from err

    return resolved_target


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 cryptographic checksum of a file using buffered reads."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for checksum: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class StorageManager:
    """
    Manages isolated, application-controlled dataset storage and staging.
    Enforces directory boundaries, generates SHA-256 checksums, and coordinates
    atomic commits and rollbacks.
    """

    def __init__(self, base_dir: Union[str, Path] = "storage"):
        self.base_dir = Path(base_dir).resolve()
        self.datasets_dir = self.base_dir / "datasets"
        self.staging_dir = self.base_dir / "staging"

        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset_dir(self, dataset_id: str, version: int) -> Path:
        """Returns the canonical storage path for a specific dataset version."""
        safe_id = sanitize_filename(dataset_id)
        target = self.datasets_dir / safe_id / f"v{version}"
        return ensure_within_directory(self.datasets_dir, target)

    def get_staging_dir(self, dataset_id: str, version: int) -> Path:
        """Returns the temporary staging directory for a pending version."""
        safe_id = sanitize_filename(dataset_id)
        target = self.staging_dir / safe_id / f"v{version}"
        return ensure_within_directory(self.staging_dir, target)

    def stage_file(
        self,
        source_path: Union[str, Path],
        dataset_id: str,
        version: int,
        original_filename: str,
    ) -> Tuple[Path, str, int]:
        """
        Safely copies source file into staging area.
        Returns (staged_path, sha256_checksum, file_size_bytes).
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source dataset file not found: {src}")
        if not src.is_file():
            raise ValueError(f"Source path is not a file: {src}")

        safe_name = sanitize_filename(original_filename or src.name)
        stage_dir = self.get_staging_dir(dataset_id, version)
        stage_dir.mkdir(parents=True, exist_ok=True)

        destination = stage_dir / safe_name
        ensure_within_directory(stage_dir, destination)

        shutil.copy2(src, destination)
        checksum = compute_sha256(destination)
        file_size = destination.stat().st_size

        return destination, checksum, file_size

    def commit_staged_file(
        self,
        dataset_id: str,
        version: int,
        filename: str,
    ) -> Path:
        """
        Atomically promotes a staged file to the canonical dataset storage.
        """
        safe_name = sanitize_filename(filename)
        stage_dir = self.get_staging_dir(dataset_id, version)
        staged_file = stage_dir / safe_name

        if not staged_file.exists():
            raise FileNotFoundError(f"Staged file not found: {staged_file}")

        dest_dir = self.get_dataset_dir(dataset_id, version)
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_dest = dest_dir / safe_name

        # Atomically replace or copy
        shutil.copy2(staged_file, final_dest)

        # Cleanup staging
        self.cleanup_staging(dataset_id, version)

        return final_dest

    def cleanup_staging(self, dataset_id: str, version: int) -> None:
        """Removes the staging directory for a dataset version on rollback or completion."""
        stage_dir = self.get_staging_dir(dataset_id, version)
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
