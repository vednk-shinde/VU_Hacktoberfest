import json
import tempfile
import unittest
from pathlib import Path

from src.engine import QAEngine
from src.ingestion.loader import load_dataset
from src.ingestion.profile import profile_dataset


class TestQAEngine(unittest.TestCase):
    def setUp(self):
        self.engine = QAEngine(approved_file_roots=[Path.cwd() / "data", Path(tempfile.gettempdir())])
        self.csv_path = "data/sample.csv"
        self.json_path = "data/sample.json"
        self.synth_path = "data/synthetic_employees.csv"

    # --- 1. Correct Answers for All Supported Operations ---

    def test_row_count_csv_and_json(self):
        res_csv = self.engine.ask("How many rows are in the dataset?", self.csv_path)
        self.assertEqual(res_csv.status, "answered")
        self.assertEqual(res_csv.answer, 5)

        res_json = self.engine.ask("Total records in dataset", self.json_path)
        self.assertEqual(res_json.status, "answered")
        self.assertEqual(res_json.answer, 3)

    def test_column_names_csv_and_json(self):
        res_csv = self.engine.ask("What are the columns?", self.csv_path)
        self.assertEqual(res_csv.status, "answered")
        self.assertEqual(sorted(res_csv.answer), sorted(["employee", "department", "salary"]))

        res_json = self.engine.ask("List columns", self.json_path)
        self.assertEqual(res_json.status, "answered")
        self.assertEqual(sorted(res_json.answer), sorted(["product", "price", "stock"]))

    def test_missing_values_counts(self):
        # Column specific
        res_col = self.engine.ask("How many missing values in salary?", self.csv_path)
        self.assertEqual(res_col.status, "answered")
        self.assertEqual(res_col.answer, 1)

        # Dataset wide
        res_total = self.engine.ask("Total missing values in dataset", self.synth_path)
        self.assertEqual(res_total.status, "answered")
        self.assertEqual(res_total.answer, 2)

    def test_duplicate_counts(self):
        res_csv = self.engine.ask("How many duplicate rows are there?", self.csv_path)
        self.assertEqual(res_csv.status, "answered")
        self.assertEqual(res_csv.answer, 1)

        res_json = self.engine.ask("Count duplicates in the dataset", self.json_path)
        self.assertEqual(res_json.status, "answered")
        self.assertEqual(res_json.answer, 0)

    def test_numeric_aggregations(self):
        # Average
        res_avg = self.engine.ask("What is the average salary?", self.csv_path)
        self.assertEqual(res_avg.status, "answered")
        self.assertAlmostEqual(res_avg.answer, 63500.0, places=1)

        # Max
        res_max = self.engine.ask("What is the maximum salary?", self.csv_path)
        self.assertEqual(res_max.status, "answered")
        self.assertEqual(res_max.answer, 72000)

        # Min
        res_min = self.engine.ask("What is the minimum salary?", self.csv_path)
        self.assertEqual(res_min.status, "answered")
        self.assertEqual(res_min.answer, 52000)

        # Sum
        res_sum = self.engine.ask("What is the total salary?", self.csv_path)
        self.assertEqual(res_sum.status, "answered")
        self.assertEqual(res_sum.answer, 254000)

    def test_category_counts(self):
        res = self.engine.ask("Count by department", self.csv_path)
        self.assertEqual(res.status, "answered")
        self.assertEqual(res.answer, {"Engineering": 3, "Sales": 1, "Marketing": 1})

    def test_filtering(self):
        res_cat = self.engine.ask("How many employees are in Engineering?", self.csv_path)
        self.assertEqual(res_cat.status, "answered")
        self.assertEqual(res_cat.answer, 3)

        res_num = self.engine.ask("How many products where price > 1000?", self.json_path)
        self.assertEqual(res_num.status, "answered")
        self.assertEqual(res_num.answer, 2)

    def test_calculation_method_and_evidence(self):
        res = self.engine.ask("What is the average salary?", self.csv_path)
        self.assertEqual(res.status, "answered")
        self.assertTrue(isinstance(res.calculation_method, str))
        self.assertIn("salary", res.calculation_method)
        self.assertIsInstance(res.evidence, dict)
        self.assertIn("sample_values", res.evidence)

    # --- 2. Abstention on Missing Columns ---

    def test_missing_column_abstains(self):
        res = self.engine.ask("What is the average bonus?", self.csv_path)
        self.assertEqual(res.status, "abstained")
        self.assertIsNone(res.answer)
        self.assertIn("does not exist in dataset", res.reason)

        res_cat = self.engine.ask("Count by location", self.csv_path)
        self.assertEqual(res_cat.status, "abstained")
        self.assertIn("does not exist in dataset", res_cat.reason)

    # --- 3. Abstention on Incompatible Types ---

    def test_non_numeric_aggregation_abstains(self):
        res = self.engine.ask("What is the sum of department?", self.csv_path)
        self.assertEqual(res.status, "abstained")
        self.assertIn("non-numeric column", res.reason)

    # --- 4. Abstention on Unsupported & Ambiguous Queries ---

    def test_unsupported_questions_abstain(self):
        unsupported = [
            "What is the capital of France?",
            "Who is the CEO of Google?",
            "Write a poem about datasets",
            "Predict tomorrow's stock price",
        ]
        for q in unsupported:
            res = self.engine.ask(q, self.csv_path)
            self.assertEqual(res.status, "abstained", f"Failed to abstain on: {q}")
            self.assertIsNone(res.answer)

    def test_ambiguous_questions_abstain(self):
        ambiguous = [
            "What is the value?",
            "How much?",
            "Tell me about it",
            "",
            "   ",
        ]
        for q in ambiguous:
            res = self.engine.ask(q, self.csv_path)
            self.assertEqual(res.status, "abstained", f"Failed to abstain on: {q}")

    # --- 5. Security & Safety Validations ---

    def test_disallowed_and_mutating_keywords(self):
        unsafe_queries = [
            "DROP TABLE employees",
            "DELETE FROM dataset WHERE salary < 50000",
            "exec('import os; os.system(\"calc\")')",
            "eval('2+2')",
        ]
        for q in unsafe_queries:
            res = self.engine.ask(q, self.csv_path)
            self.assertEqual(res.status, "abstained")
            self.assertIn("disallowed or unsafe keyword", res.reason.lower())

    # --- 6. Empty Datasets ---

    def test_empty_csv_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            res_rows = self.engine.ask("How many rows are in the dataset?", temp_path)
            self.assertEqual(res_rows.status, "answered")
            self.assertEqual(res_rows.answer, 0)

            res_agg = self.engine.ask("What is the average salary?", temp_path)
            self.assertEqual(res_agg.status, "abstained")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_empty_json_list(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("[]")
            temp_path = f.name

        try:
            res_rows = self.engine.ask("How many rows are in the dataset?", temp_path)
            self.assertEqual(res_rows.status, "answered")
            self.assertEqual(res_rows.answer, 0)

            res_agg = self.engine.ask("What is the maximum price?", temp_path)
            self.assertEqual(res_agg.status, "abstained")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    # --- 7. Malformed Inputs ---

    def test_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.engine.ask("How many rows?", "data/non_existent_file.csv")

    def test_unsupported_file_extension(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("some text")
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                self.engine.ask("How many rows?", temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_malformed_json_syntax(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{invalid json content")
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                self.engine.ask("How many rows?", temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
