"""CSV/JSON dataset ingestion with deterministic schema and quality metadata."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.ingestion.profile import profile_records, read_dataset_records


def to_number(val: Any) -> Optional[Union[int, float]]:
    """Parse finite numeric values without converting booleans or invalid tokens."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val if math.isfinite(val) else None
    s = str(val).strip()
    if not s:
        return None
    if not any(char in s.lower() for char in (".", "e")):
        try:
            return int(s)
        except ValueError:
            pass
    try:
        number = float(s)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


@dataclass
class Dataset:
    file_path: Path
    name: str
    rows: List[Dict[str, Any]]
    columns: List[str]
    column_types: Dict[str, str]
    profile: Dict[str, Any]
    dataset_id: Optional[str] = None
    dataset_version: Optional[int] = None
    content_fingerprint: Optional[str] = None

    def get_column_values(self, column: str, clean_numeric: bool = False) -> List[Any]:
        """Return values for a column, optionally converted to finite numbers."""
        values = []
        for row in self.rows:
            value = row.get(column)
            if clean_numeric:
                number = to_number(value)
                if number is not None:
                    values.append(number)
            else:
                values.append(value)
        return values


def _fingerprint(columns: List[str], rows: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    for row in rows:
        normalized = {column: row.get(column) for column in columns}
        digest.update(b"\n")
        digest.update(json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), default=str).encode("utf-8"))
    return digest.hexdigest()


def load_dataset(file_path: Union[str, Path]) -> Dataset:
    """Load a CSV or JSON file and retain source values without implicit coercion.

    This low-level loader parses a caller-selected file. Application entry points
    must authorize paths before calling it (QAEngine does so via approved roots).
    """
    path = Path(file_path)
    columns, rows, read_warnings = read_dataset_records(path)
    profile = profile_records(path.name, columns, rows, read_warnings)
    return Dataset(
        file_path=path,
        name=path.name,
        rows=rows,
        columns=columns,
        column_types=profile["column_types"],
        profile=profile,
        content_fingerprint=_fingerprint(columns, rows),
    )
