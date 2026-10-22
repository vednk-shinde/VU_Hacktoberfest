"""Deterministic CSV/JSON schema and quality profiling helpers."""

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# The current application materializes datasets in memory. Keep ingestion bounded
# until a streaming execution path is available.
MAX_DATASET_FILE_BYTES = 64 * 1024 * 1024
MAX_DATASET_ROWS = 100_000


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _number(value: Any):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if not any(char in text.lower() for char in (".", "e")):
        try:
            return int(text)
        except ValueError:
            pass
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _is_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    try:
        if "T" in text or " " in text:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            date.fromisoformat(text)
        return True
    except ValueError:
        return False


def _invalid_numeric(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return not math.isfinite(float(value.strip()))
    except (ValueError, OverflowError):
        return False


def read_dataset_records(file_path) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read supported tabular formats without silently repairing malformed CSV."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".json"}:
        raise ValueError("Supported formats: CSV and JSON.")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if not path.is_file():
        raise ValueError("Dataset path is not a regular file.")
    try:
        file_size = path.stat().st_size
        if file_size > MAX_DATASET_FILE_BYTES:
            raise ValueError(
                f"Dataset exceeds the {MAX_DATASET_FILE_BYTES // (1024 * 1024)} MiB ingestion limit."
            )
        if file_size == 0:
            return [], [], []
        if suffix == ".csv":
            return _read_csv(path)
        return _read_json(path)
    except (PermissionError, OSError) as err:
        raise OSError(f"Could not read dataset file '{path.name}': {err.strerror or type(err).__name__}") from err
    except UnicodeDecodeError as err:
        raise ValueError(f"Dataset '{path.name}' is not valid UTF-8 text.") from err


def _read_csv(path: Path):
    warnings: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file, strict=True)
            headers = next(reader, None)
            if headers is None:
                return [], [], warnings
            if not headers or any(not h.strip() for h in headers):
                raise ValueError("CSV header contains an empty column name.")
            if any(h != h.strip() for h in headers):
                raise ValueError("CSV header names cannot have leading or trailing whitespace.")
            normalized = [h.casefold() for h in headers]
            if len(set(normalized)) != len(headers):
                raise ValueError("CSV header contains duplicate column names (case-insensitive).")
            short_rows = 0
            for row_number, fields in enumerate(reader, start=2):
                if not fields:
                    continue
                if len(rows) >= MAX_DATASET_ROWS:
                    raise ValueError(f"Dataset exceeds the {MAX_DATASET_ROWS}-row ingestion limit.")
                if len(fields) > len(headers):
                    raise ValueError(f"CSV row {row_number} has more fields than the header.")
                if len(fields) < len(headers):
                    short_rows += 1
                rows.append({name: fields[i] if i < len(fields) else None
                             for i, name in enumerate(headers)})
    except csv.Error as err:
        raise ValueError(f"Malformed CSV in dataset '{path.name}': {err}") from err
    if short_rows:
        warnings.append({"code": "short_rows", "count": short_rows,
                          "message": f"{short_rows} row(s) have fewer fields than the header; omitted cells are treated as missing."})
    return headers, rows, warnings


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON in dataset '{path.name}' at line {err.lineno}, column {err.colno}.") from err
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("data", [data])
    else:
        raise ValueError("JSON must contain an object or a list of objects.")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Dataset rows must be a list of JSON objects.")
    if len(rows) > MAX_DATASET_ROWS:
        raise ValueError(f"Dataset exceeds the {MAX_DATASET_ROWS}-row ingestion limit.")
    if any(isinstance(value, (dict, list)) for row in rows for value in row.values()):
        raise ValueError("JSON dataset cells must contain scalar values; nested objects and arrays are unsupported.")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    return columns, rows, []


def profile_records(file_name: str, columns: List[str], rows: List[Dict[str, Any]],
                    read_warnings: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Profile already-loaded rows in one pass; never changes their values."""
    missing_values: Dict[str, int] = {}
    column_types: Dict[str, str] = {}
    type_counts: Dict[str, Dict[str, int]] = {}
    statistics: Dict[str, Dict[str, Any]] = {}
    warnings = list(read_warnings or [])

    for column in columns:
        values = [row.get(column) for row in rows]
        present = [value for value in values if not _missing(value)]
        missing_count = len(values) - len(present)
        missing_values[column] = missing_count
        type_counts[column] = dict(Counter(type(v).__name__ for v in present))
        if not present:
            kind = "empty"
            statistics[column] = {"count": 0}
            warnings.append({"code": "empty_column", "column": column,
                             "message": "Column has no non-missing values."})
        else:
            numeric = [_number(value) for value in present]
            date_flags = [_is_date(value) for value in present]
            kinds = ["numeric" if n is not None else "invalid_numeric" if _invalid_numeric(value) else "date" if is_date else
                     "boolean" if isinstance(value, bool) else "text"
                     for value, n, is_date in zip(present, numeric, date_flags)]
            unique_kinds = set(kinds)
            kind = next(iter(unique_kinds)) if len(unique_kinds) == 1 else "mixed"
            if kind == "numeric":
                nums = [n for n in numeric if n is not None]
                statistics[column] = {"count": len(nums), "min": min(nums), "max": max(nums),
                                      "mean": sum(nums) / len(nums)}
            elif kind == "date":
                dates = [str(value).strip() for value in present]
                statistics[column] = {"count": len(dates), "min": min(dates), "max": max(dates),
                                      "format": "ISO-8601"}
            elif kind == "mixed":
                statistics[column] = {"count": len(present), "distinct_count": len(set(map(str, present)))}
                warnings.append({"code": "mixed_types", "column": column,
                                 "message": "Column contains incompatible value types; numeric aggregation is disabled."})
            elif kind == "invalid_numeric":
                statistics[column] = {"count": 0}
                warnings.append({"code": "invalid_numeric_values", "column": column,
                                 "count": len(present), "message": "Column contains non-finite numeric values."})
            else:
                statistics[column] = {"count": len(present), "distinct_count": len(set(map(str, present)))}
        column_types[column] = kind
        if missing_count:
            warnings.append({"code": "missing_values", "column": column, "count": missing_count,
                             "message": f"Column contains {missing_count} missing value(s)."})

    if not rows:
        warnings.append({"code": "empty_dataset", "message": "Dataset contains no data rows."})
    unique_rows = set()
    duplicate_rows = 0
    for row in rows:
        serialized = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        fingerprint = hashlib.sha256(serialized.encode("utf-8")).digest()
        if fingerprint in unique_rows:
            duplicate_rows += 1
        else:
            unique_rows.add(fingerprint)
    return {
        "file": file_name,
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": list(columns),
        "column_types": column_types,
        "type_counts": type_counts,
        "missing_values": missing_values,
        "empty_columns": [column for column in columns if column_types[column] == "empty"],
        "statistics": statistics,
        "quality_warnings": warnings,
        "duplicate_rows": duplicate_rows,
        "preview": rows[:5],
    }


def profile_dataset(file_path):
    path = Path(file_path)
    columns, rows, warnings = read_dataset_records(path)
    return profile_records(path.name, columns, rows, warnings)
