import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.engine import QAEngine
from src.ingestion.loader import load_dataset
from src.ingestion.profile import profile_dataset


class TestIngestionQuality(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8", newline="")
        return path

    def test_valid_csv_preserves_values_and_infers_schema(self):
        path = self.write_csv("valid.csv", "amount,group\n12.5,A\n7.5,B\n")
        dataset = load_dataset(path)
        self.assertEqual(dataset.rows[0], {"amount": "12.5", "group": "A"})
        self.assertEqual(dataset.column_types, {"amount": "numeric", "group": "text"})
        self.assertEqual(dataset.profile["statistics"]["amount"],
                         {"count": 2, "min": 7.5, "max": 12.5, "mean": 10.0})
        self.assertEqual(len(dataset.content_fingerprint), 64)

    def test_missing_file_and_unreadable_file_raise_actionable_errors(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.root / "missing.csv")
        path = self.write_csv("unreadable.csv", "a\n1\n")
        with patch.object(Path, "open", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(OSError, "Could not read dataset file"):
                load_dataset(path)

    def test_empty_file_and_header_only_dataset_are_profiles(self):
        empty = self.write_csv("empty.csv", "")
        dataset = load_dataset(empty)
        self.assertEqual((dataset.columns, dataset.rows), ([], []))
        self.assertEqual(dataset.profile["row_count"], 0)
        self.assertIn("empty_dataset", [item["code"] for item in dataset.profile["quality_warnings"]])
        header_only = self.write_csv("header.csv", "a,b\n")
        header_dataset = load_dataset(header_only)
        self.assertEqual(header_dataset.columns, ["a", "b"])
        self.assertEqual(header_dataset.profile["empty_columns"], ["a", "b"])

    def test_malformed_csv_and_extra_fields_fail(self):
        malformed = self.write_csv("malformed.csv", 'a,b\n1,"unfinished\n')
        extra = self.write_csv("extra.csv", "a,b\n1,2,3\n")
        for path in (malformed, extra):
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                load_dataset(path)

    def test_duplicate_and_blank_headers_fail_without_renaming(self):
        for name, header in (("duplicate.csv", "Amount,amount\n1,2\n"),
                             ("blank.csv", "amount, \n1,2\n")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "header"):
                load_dataset(self.write_csv(name, header))

    def test_short_rows_are_kept_as_missing_and_warned(self):
        dataset = load_dataset(self.write_csv("short.csv", "a,b\n1\n"))
        self.assertIsNone(dataset.rows[0]["b"])
        self.assertEqual(dataset.profile["missing_values"]["b"], 1)
        codes = [item["code"] for item in dataset.profile["quality_warnings"]]
        self.assertIn("short_rows", codes)
        self.assertIn("missing_values", codes)

    def test_mixed_types_are_reported_and_numeric_aggregation_abstains(self):
        path = self.write_csv("mixed.csv", "amount\n12\nunknown\n15\n")
        dataset = load_dataset(path)
        self.assertEqual(dataset.column_types["amount"], "mixed")
        self.assertIn("mixed_types", [item["code"] for item in dataset.profile["quality_warnings"]])
        engine = QAEngine(approved_file_roots=[self.root])
        result = engine.ask("What is the average amount?", path)
        self.assertEqual(result.status, "abstained")
        self.assertIn("detected type: mixed", result.reason)

    def test_non_finite_numeric_values_are_not_aggregated(self):
        path = self.write_csv("nonfinite.csv", "amount\nNaN\n")
        dataset = load_dataset(path)
        self.assertEqual(dataset.column_types["amount"], "invalid_numeric")
        self.assertIn("invalid_numeric_values", [item["code"] for item in dataset.profile["quality_warnings"]])

    def test_failed_dataset_selection_clears_prior_selection(self):
        valid = self.write_csv("valid.csv", "amount\n10\n")
        engine = QAEngine(approved_file_roots=[self.root])
        engine.load(valid)
        with self.assertRaises(FileNotFoundError):
            engine.load(self.root / "missing.csv")
        result = engine.ask("How many rows?")
        self.assertEqual(result.status, "abstained")
        self.assertIn("No dataset loaded", result.reason)

    def test_date_profile_is_deterministic_and_does_not_coerce_source_values(self):
        path = self.write_csv("dates.csv", "when\n2025-02-01\n2025-01-01\n")
        dataset = load_dataset(path)
        self.assertEqual(dataset.column_types["when"], "date")
        self.assertEqual(dataset.rows[0]["when"], "2025-02-01")
        self.assertEqual(dataset.profile["statistics"]["when"]["min"], "2025-01-01")

    def test_inconsistent_json_rows_are_profiled_without_coercion(self):
        path = self.root / "rows.json"
        path.write_text('[{"value":1},{"other":"x"}]', encoding="utf-8")
        dataset = load_dataset(path)
        self.assertEqual(dataset.columns, ["value", "other"])
        self.assertEqual(dataset.rows[0]["value"], 1)
        self.assertEqual(dataset.profile["missing_values"], {"value": 1, "other": 1})

    def test_nested_json_cells_fail_clearly(self):
        path = self.root / "nested.json"
        path.write_text('[{"value":{"nested":1}}]', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nested objects and arrays are unsupported"):
            load_dataset(path)

    def test_profile_function_includes_basic_quality_and_statistics(self):
        result = profile_dataset(self.write_csv("quality.csv", "n,label\n1,A\n2,A\n"))
        self.assertEqual(result["column_count"], 2)
        self.assertEqual(result["statistics"]["n"]["mean"], 1.5)
        self.assertEqual(result["statistics"]["label"]["distinct_count"], 1)

    def test_file_size_and_row_limits_fail_before_unbounded_ingestion(self):
        oversized = self.write_csv("oversized.csv", "a\n12345\n")
        with patch("src.ingestion.profile.MAX_DATASET_FILE_BYTES", 4):
            with self.assertRaisesRegex(ValueError, "ingestion limit"):
                load_dataset(oversized)
        many_rows = self.write_csv("many.csv", "a\n1\n2\n3\n")
        with patch("src.ingestion.profile.MAX_DATASET_ROWS", 2):
            with self.assertRaisesRegex(ValueError, "row ingestion limit"):
                load_dataset(many_rows)


if __name__ == "__main__":
    unittest.main()
