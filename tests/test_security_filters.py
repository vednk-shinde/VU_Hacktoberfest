import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.engine import QAEngine
from src.security.permissions import UserContext


class TestSecurityAndCompoundFilters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.csv = self.root / "orders.csv"
        self.csv.write_text("region,category,units\nNorth,A,10\nNorth,B,7\nSouth,A,3\nSouth,B,2\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _planner(self, plan):
        mock = MagicMock()
        mock.propose_plan.return_value = plan
        return mock

    def _plan(self, category_column="category", operation="category_count"):
        return {
            "operation": operation, "column": "region" if operation == "category_count" else None, "agg_function": None,
            "filter": {"combinator": "AND", "criteria": [
                {"column": "units", "operator": ">", "value": 2},
                {"combinator": "OR", "criteria": [
                    {"column": category_column, "operator": "==", "value": "A"},
                    {"column": "category", "operator": "==", "value": "B"},
                ]},
            ]},
        }

    def test_direct_file_requires_approved_root(self):
        outside = Path(tempfile.gettempdir()) / "not-approved.csv"
        outside.write_text("x\n1\n")
        try:
            engine = QAEngine()
            with self.assertRaises(PermissionError):
                engine.load(outside)
        finally:
            outside.unlink(missing_ok=True)

    def test_registered_dataset_requires_user_context(self):
        engine = QAEngine(approved_file_roots=[self.root])
        engine.load(self.csv)
        engine._active_dataset_id = "sales"
        denied = engine.ask("How many rows?")
        self.assertEqual(denied.status, "abstained")
        self.assertIn("authorized user context is required", denied.reason)
        default_context = engine.ask("How many rows?", user_context=UserContext("no-grants"))
        self.assertEqual(default_context.status, "abstained")

    def test_nested_filters_work_in_memory_and_sqlite(self):
        user = UserContext("analyst", allowed_datasets=["sales"],
                           allowed_columns={"sales": ["region", "category", "units"]})
        for operation, expected in (("category_count", {"North": 2, "South": 1}),
                                    ("filter_count", 3), ("row_count", 3)):
            answers = []
            for pushdown in (False, True):
                engine = QAEngine(gemma_planner=self._planner(self._plan(operation=operation)),
                                  enable_db_pushdown=pushdown, approved_file_roots=[self.root])
                engine.load(self.csv)
                engine._active_dataset_id = "sales"
                res = engine.ask("Count matching rows", use_gemma=True, user_context=user)
                self.assertEqual(res.status, "answered", res.reason)
                answers.append(res.answer)
            self.assertEqual(answers, [expected, expected])

    def test_nested_column_permission_denied(self):
        engine = QAEngine(gemma_planner=self._planner(self._plan("secret")),
                          approved_file_roots=[self.root])
        engine.load(self.csv)
        engine._active_dataset_id = "sales"
        user = UserContext("analyst", allowed_datasets=["sales"],
                           allowed_columns={"sales": ["region", "category", "units"]})
        res = engine.ask("Count matching regions", use_gemma=True, user_context=user)
        self.assertEqual(res.status, "abstained")
        self.assertIn("secret", res.reason)


if __name__ == "__main__":
    unittest.main()
