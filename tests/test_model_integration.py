import unittest
from pathlib import Path
from unittest.mock import MagicMock
from src.engine import QAEngine
from src.models.ollama_client import OllamaClient
from src.reasoning.intents import AggFunction, Operation
from src.reasoning.llm_planner import GemmaQueryPlanner
from src.security.permissions import UserContext


class TestModelIntegration(unittest.TestCase):
    def setUp(self):
        self.sample_csv = "data/sample.csv"
        self.columns = ["employee", "department", "salary"]
        self.col_types = {"employee": "text", "department": "text", "salary": "numeric"}

    def test_ollama_client_health_check(self):
        client = OllamaClient(base_url="http://localhost:11434")
        self.assertTrue(client.is_healthy(), "Local Ollama service should be online at localhost:11434")

    def test_ollama_client_unreachable_endpoint(self):
        # Point to invalid port
        client = OllamaClient(base_url="http://localhost:59999", timeout_seconds=1.0)
        self.assertFalse(client.is_healthy())
        with self.assertRaises(RuntimeError) as ctx:
            client.generate(prompt="hello")
        self.assertIn("unreachable", str(ctx.exception).lower())

    def test_gemma_planner_proposes_valid_plan(self):
        client = OllamaClient(base_url="http://localhost:11434", model_name="gemma3:4b")
        if not client.is_healthy():
            self.skipTest("Ollama service not online.")

        planner = GemmaQueryPlanner(ollama_client=client)

        # 1. Row count query
        plan_rows = planner.propose_plan(
            question="How many employees are in the dataset?",
            columns=self.columns,
            column_types=self.col_types,
            dataset_name="sample.csv",
        )
        self.assertEqual(plan_rows["operation"], Operation.ROW_COUNT)

        # 2. Aggregation query
        plan_avg = planner.propose_plan(
            question="What is the average salary of employees?",
            columns=self.columns,
            column_types=self.col_types,
            dataset_name="sample.csv",
        )
        self.assertEqual(plan_avg["operation"], Operation.NUMERIC_AGGREGATION)
        self.assertEqual(plan_avg["column"], "salary")
        self.assertEqual(plan_avg["agg_function"], AggFunction.AVERAGE)

    def test_gemma_planner_rejects_nonexistent_column(self):
        client = OllamaClient(base_url="http://localhost:11434", model_name="gemma3:4b")
        if not client.is_healthy():
            self.skipTest("Ollama service not online.")

        planner = GemmaQueryPlanner(ollama_client=client)
        plan = planner.propose_plan(
            question="What is the average bonus?",
            columns=self.columns,
            column_types=self.col_types,
            dataset_name="sample.csv",
        )
        # Should be classified as unsupported / missing column
        self.assertIn(plan["operation"], {Operation.UNSUPPORTED, Operation.AMBIGUOUS})

    def test_gemma_planner_malformed_json_fallback(self):
        # Mock client returning invalid JSON string
        mock_client = MagicMock()
        mock_client.generate.side_effect = ValueError("Model returned invalid JSON syntax")

        planner = GemmaQueryPlanner(ollama_client=mock_client)

        # Engine should catch model failure and fall back safely to deterministic parser
        engine = QAEngine(gemma_planner=planner, approved_file_roots=[Path.cwd() / "data"])
        res = engine.ask(
            question="How many rows are in the dataset?",
            dataset_path=self.sample_csv,
            use_gemma=True,
        )
        self.assertEqual(res.status, "answered")
        self.assertEqual(res.answer, 5)

    def test_gemma_planner_unpermitted_operation_rejected(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "parsed": {
                "operation": "DROP_DATABASE",
                "column": None,
            }
        }
        planner = GemmaQueryPlanner(ollama_client=mock_client)
        with self.assertRaises(ValueError) as ctx:
            planner.propose_plan("Delete all records", self.columns, self.col_types)
        self.assertIn("unpermitted operation", str(ctx.exception))

    def test_user_permission_rejection_dataset_level(self):
        engine = QAEngine(approved_file_roots=[Path.cwd() / "data"])
        engine.load(self.sample_csv)
        engine._active_dataset_id = "restricted_ds"

        # User restricted to other_ds only
        user = UserContext(user_id="alice", allowed_datasets=["other_ds"])

        res = engine.ask("How many rows?", user_context=user)
        self.assertEqual(res.status, "abstained")
        self.assertIn("Access denied", res.reason)

    def test_user_permission_rejection_column_level(self):
        engine = QAEngine(approved_file_roots=[Path.cwd() / "data"])
        engine.load(self.sample_csv)
        engine._active_dataset_id = "salaries_ds"

        # User allowed to see 'employee' and 'department', but NOT 'salary'
        user = UserContext(
            user_id="bob",
            allowed_datasets=["salaries_ds"],
            allowed_columns={"salaries_ds": ["employee", "department"]},
        )

        # Allowed column query
        res_ok = engine.ask("Count by department", user_context=user)
        self.assertEqual(res_ok.status, "answered")

        # Restricted column query
        res_blocked = engine.ask("What is the average salary?", user_context=user)
        self.assertEqual(res_blocked.status, "abstained")
        self.assertIn("Access denied", res_blocked.reason)
        self.assertIn("salary", res_blocked.reason)

    def test_grounded_explanation_generation(self):
        client = OllamaClient(base_url="http://localhost:11434", model_name="gemma3:4b")
        if not client.is_healthy():
            self.skipTest("Ollama service not online.")

        planner = GemmaQueryPlanner(ollama_client=client)
        engine = QAEngine(gemma_planner=planner, approved_file_roots=[Path.cwd() / "data"])
        engine._active_dataset_id = "employee_data"
        engine._active_dataset_version = 1

        res = engine.ask(
            question="What is the average salary?",
            dataset_path=self.sample_csv,
            user_context=UserContext(user_id="model-test", allowed_datasets=["employee_data"]),
            explain_with_gemma=True,
        )
        self.assertEqual(res.status, "answered")
        self.assertIsNotNone(res.explanation)
        self.assertTrue(len(res.explanation) > 10)


if __name__ == "__main__":
    unittest.main()
