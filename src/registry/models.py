from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class DatasetStatus:
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    VALIDATED = "validated"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"


def now_utc_iso() -> str:
    """Return current timezone-aware UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SchemaProfile:
    columns: List[str]
    column_types: Dict[str, str]
    row_count: int
    column_count: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    preview: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaProfile":
        return cls(
            columns=data.get("columns", []),
            column_types=data.get("column_types", {}),
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", 0),
            missing_values=data.get("missing_values", {}),
            duplicate_rows=data.get("duplicate_rows", 0),
            preview=data.get("preview", []),
        )


@dataclass
class DatasetVersion:
    dataset_id: str
    version: int
    filename: str
    storage_path: str
    format: str
    file_size_bytes: int
    checksum_sha256: str
    status: str  # One of DatasetStatus
    schema_profile: Dict[str, Any]
    upload_timestamp: str
    processing_timestamp: Optional[str] = None
    activation_timestamp: Optional[str] = None
    error_message: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetVersion":
        return cls(
            dataset_id=data["dataset_id"],
            version=data["version"],
            filename=data["filename"],
            storage_path=data["storage_path"],
            format=data["format"],
            file_size_bytes=data["file_size_bytes"],
            checksum_sha256=data["checksum_sha256"],
            status=data["status"],
            schema_profile=data.get("schema_profile", {}),
            upload_timestamp=data["upload_timestamp"],
            processing_timestamp=data.get("processing_timestamp"),
            activation_timestamp=data.get("activation_timestamp"),
            error_message=data.get("error_message"),
            provenance=data.get("provenance"),
        )


@dataclass
class DatasetRecord:
    dataset_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    active_version: Optional[int] = None
    versions: List[DatasetVersion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active_version": self.active_version,
            "versions": [v.to_dict() for v in self.versions],
        }
