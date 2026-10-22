import tempfile
import unittest
from pathlib import Path
from src.retrieval.db_executor import CompoundFilter, DatabaseExecutor, FilterCriterion
from src.retrieval.filters import apply_filter_object
from src.ingestion.loader import load_dataset


class TestDatabaseExecutor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_data.sqlite"
        self.executor = DatabaseExecutor(self.db_path)

        self.columns = ["employee", "department", "salary"]
        self.col_types = {"employee": "text", "department": "text", "salary": "numeric"}
        self.sample_csv = "data/sample.csv"

        self.executor.sync_dataset_file(self.sample_csv, self.columns, self.col_types)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_db_row_count(self):
        cnt = self.executor.execute_count(self.columns)
        self.assertEqual(cnt, 5)

    def test_db_aggregations(self):
        # Avg
        avg = self.executor.execute_aggregation("average", "salary", self.columns)
        self.assertAlmostEqual(avg, 63500.0, places=1)

        # Max
        m_val = self.executor.execute_aggregation("max", "salary", self.columns)
        self.assertEqual(m_val, 72000.0)

        # Min
        min_val = self.executor.execute_aggregation("min", "salary", self.columns)
        self.assertEqual(min_val, 52000.0)

        # Sum
        s_val = self.executor.execute_aggregation("sum", "salary", self.columns)
        self.assertEqual(s_val, 254000.0)

    def test_db_category_counts(self):
        counts = self.executor.execute_category_count("department", self.columns)
        self.assertEqual(counts["Engineering"], 3)
        self.assertEqual(counts["Sales"], 1)
        self.assertEqual(counts["Marketing"], 1)

    def test_compound_and_filter(self):
        # Filter: department == 'Engineering' AND salary > 65000
        comp_filter = CompoundFilter(
            combinator="AND",
            criteria=[
                FilterCriterion(column="department", operator="==", value="Engineering"),
                FilterCriterion(column="salary", operator=">", value=65000),
            ],
        )
        # In sample.csv: Meera (72000) is the only one in Engineering with salary > 65000
        cnt = self.executor.execute_count(self.columns, filter_obj=comp_filter)
        self.assertEqual(cnt, 1)

        # Avg salary for that filter
        avg = self.executor.execute_aggregation("average", "salary", self.columns, filter_obj=comp_filter)
        self.assertEqual(avg, 72000.0)

    def test_compound_or_filter(self):
        # Filter: department == 'Sales' OR department == 'Marketing'
        comp_filter = CompoundFilter(
            combinator="OR",
            criteria=[
                FilterCriterion(column="department", operator="==", value="Sales"),
                FilterCriterion(column="department", operator="==", value="Marketing"),
            ],
        )
        # Rohan (Sales) + Kabir (Marketing) = 2
        cnt = self.executor.execute_count(self.columns, filter_obj=comp_filter)
        self.assertEqual(cnt, 2)

    def test_sql_injection_defense(self):
        # Malicious payload in value should be treated strictly as a literal string
        malicious = FilterCriterion(
            column="department",
            operator="==",
            value="Engineering' OR '1'='1",
        )
        cnt = self.executor.execute_count(self.columns, filter_obj=malicious)
        self.assertEqual(cnt, 0)  # No department literally named "Engineering' OR '1'='1"

    def test_quoted_identifier_is_safe(self):
        csv_path = Path(self.temp_dir.name) / "quoted.csv"
        csv_path.write_text('value"name\n1\n2\n')
        self.executor.sync_dataset_file(csv_path, ['value"name'], {'value"name': 'numeric'})
        self.assertEqual(self.executor.execute_aggregation("sum", 'value"name', ['value"name']), 3)

    def test_csv_ingestion_uses_multiple_bounded_batches(self):
        csv_path = Path(self.temp_dir.name) / "many.csv"
        csv_path.write_text("n\n" + "\n".join(str(i) for i in range(2505)))
        self.executor.sync_dataset_file(csv_path, ["n"], {"n": "numeric"})
        self.assertEqual(self.executor.execute_count(["n"]), 2505)

    def test_text_and_null_filter_semantics_match_memory(self):
        csv_path = Path(self.temp_dir.name) / "text.csv"
        csv_path.write_text("category,value\nalpha,1\nALPHA,2\nbeta,3\n,4\n")
        cols = ["category", "value"]
        self.executor.sync_dataset_file(csv_path, cols, {"category": "text", "value": "numeric"})
        dataset_rows = load_dataset(csv_path).rows
        cases = [
            FilterCriterion("category", "==", "alpha"),
            FilterCriterion("category", "contains", "ph"),
            FilterCriterion("category", "in", ["ALPHA", "Beta"]),
            FilterCriterion("category", "!=", "alpha"),
            FilterCriterion("category", "is_null"),
            FilterCriterion("category", "is_not_null"),
        ]
        for criterion in cases:
            with self.subTest(operator=criterion.operator):
                memory_count = len(apply_filter_object(dataset_rows, criterion))
                sql_count = self.executor.execute_count(cols, filter_obj=criterion)
                self.assertEqual(sql_count, memory_count)

    def test_category_count_respects_filter(self):
        criterion = FilterCriterion("salary", ">", 65000)
        counts = self.executor.execute_category_count("department", self.columns, filter_obj=criterion)
        self.assertEqual(counts, {"Engineering": 1})


if __name__ == "__main__":
    unittest.main()
