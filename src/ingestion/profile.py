
import csv
import json
from pathlib import Path
from collections import Counter


def profile_dataset(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in {".csv", ".json"}:
        raise ValueError("Supported formats: CSV and JSON.")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Keep the existing code below this point.


    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames or []
            rows = list(reader)

    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)

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

    else:
        raise ValueError("Supported formats: CSV and JSON.")

    missing = {}
    column_types = {}

    for column in columns:
        values = [row.get(column) for row in rows]
        missing[column] = sum(
            value is None or str(value).strip() == ""
            for value in values
        )

        types = Counter(
            type(value).__name__
            for value in values
            if value is not None
        )
        column_types[column] = dict(types)

    serialized = [
        json.dumps(row, sort_keys=True, default=str)
        for row in rows
    ]

    return {
        "file": path.name,
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": columns,
        "missing_values": missing,
        "column_types": column_types,
        "duplicate_rows": len(serialized) - len(set(serialized)),
        "preview": rows[:5],
    }
