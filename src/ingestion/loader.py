import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from src.ingestion.profile import profile_dataset


def to_number(val: Any) -> Optional[Union[int, float]]:
    """Convert value to int or float if possible, otherwise return None."""
    if val is None:
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return val
    s = str(val).strip()
    if not s:
        return None
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


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

    def get_column_values(self, column: str, clean_numeric: bool = False) -> List[Any]:
        """Return values for a column, optionally converted to numbers."""
        values = []
        for r in self.rows:
            v = r.get(column)
            if clean_numeric:
                num = to_number(v)
                if num is not None:
                    values.append(num)
            else:
                values.append(v)
        return values


def load_dataset(file_path: Union[str, Path]) -> Dataset:
    """
    Safely load a CSV or JSON dataset using Python's standard library.
    Validates file format, path existence, and structure without arbitrary execution.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in {".csv", ".json"}:
        raise ValueError("Supported formats: CSV and JSON.")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    # Handle empty files (0 bytes) gracefully
    if path.stat().st_size == 0:
        return Dataset(
            file_path=path,
            name=path.name,
            rows=[],
            columns=[],
            column_types={},
            profile={
                "file": path.name,
                "row_count": 0,
                "column_count": 0,
                "columns": [],
                "missing_values": {},
                "column_types": {},
                "duplicate_rows": 0,
                "preview": [],
            },
        )

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON in dataset: {err}") from err

        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("data", [data])
        else:
            raise ValueError("JSON must contain an object or a list of objects.")

        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("Dataset rows must be JSON objects.")

        columns = list(dict.fromkeys(
            key for row in rows for key in row.keys()
        ))

    # Detect high-level column types (numeric vs text vs empty)
    column_types: Dict[str, str] = {}
    for col in columns:
        vals = [r.get(col) for r in rows]
        non_empty = [v for v in vals if v is not None and str(v).strip() != ""]
        if not non_empty:
            column_types[col] = "empty"
        elif all(to_number(v) is not None for v in non_empty):
            column_types[col] = "numeric"
        else:
            column_types[col] = "text"

    # Compute dataset profile using existing profiler or inline for consistency
    try:
        profile = profile_dataset(path)
    except Exception:
        profile = {
            "file": path.name,
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": columns,
            "missing_values": {col: sum(r.get(col) is None or str(r.get(col)).strip() == "" for r in rows) for col in columns},
            "column_types": column_types,
            "duplicate_rows": 0,
            "preview": rows[:5],
        }

    return Dataset(
        file_path=path,
        name=path.name,
        rows=rows,
        columns=columns,
        column_types=column_types,
        profile=profile,
    )
