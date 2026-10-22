
from src.ingestion.profile import profile_dataset


result = profile_dataset("data/sample.csv")

print("DATASET PROFILE")
print("================")
print("File:", result["file"])
print("Rows:", result["row_count"])
print("Columns:", result["columns"])
print("Missing values:", result["missing_values"])
print("Duplicate rows:", result["duplicate_rows"])
print("Preview:", result["preview"])

assert result["row_count"] == 5
assert result["column_count"] == 3
assert result["missing_values"]["salary"] == 1
assert result["duplicate_rows"] == 1

print("\nAll checks passed!")
