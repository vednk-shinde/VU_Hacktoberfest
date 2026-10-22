import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from src.demo_enterprise import DEMO_DIR, PLAN, RELATIONSHIPS, SOURCES
from src.engine import QAEngine


class TestExecutionEvidence(unittest.TestCase):
    def test_single_dataset_result_has_traceable_redacted_execution_record(self):
        engine = QAEngine(approved_file_roots=[Path.cwd() / "data"])
        result = engine.ask("What is the average salary?", "data/sample.csv")
        self.assertEqual(result.status, "answered")
        record = result.evidence["execution"]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["operations"], [{"operation": "numeric_aggregation",
                                                   "backend": "in_memory", "status": "completed"}])
        self.assertEqual(record["intent"]["columns_used"], ["salary"])
        self.assertEqual(record["intent"]["planning_source"], "deterministic_parser")
        self.assertEqual(record["dataset"]["dataset_id"], None)
        self.assertEqual(len(record["dataset"]["content_sha256"]), 64)
        self.assertEqual(len(record["dataset"]["schema_sha256"]), 64)
        self.assertEqual(result.explanation_source, "deterministic_execution_evidence")
        self.assertIn("deterministic executor", result.explanation)
        self.assertEqual(result.to_dict()["answer"], result.answer)
        rendered = json.dumps(record)
        self.assertNotIn("salary\":\"65000", rendered)

    def test_execution_filter_evidence_redacts_value(self):
        engine = QAEngine(approved_file_roots=[Path.cwd() / "data"])
        result = engine.ask("How many employees are in Engineering?", "data/sample.csv")
        self.assertEqual(result.status, "answered")
        summary = result.evidence["execution"]["intent"]["filter"]
        self.assertEqual(summary["column"], "department")
        self.assertEqual(summary["operator"], "==")
        self.assertTrue(summary["value_redacted"])
        self.assertNotIn("value", summary)

    def test_multi_dataset_trace_matches_executed_plan_and_outputs(self):
        engine = QAEngine(approved_file_roots=[DEMO_DIR])
        result = engine.ask_multi_dataset("verified demo", PLAN, SOURCES, RELATIONSHIPS)
        self.assertEqual(result.status, "answered", result.reason)
        record = result.evidence["execution"]
        self.assertEqual(record["status"], "completed")
        self.assertEqual([step["step_id"] for step in record["operations"]],
                         [step["id"] for step in PLAN["steps"]])
        self.assertTrue(all(step["status"] == "completed" for step in record["operations"]))
        self.assertEqual(record["outputs"], [{"step_id": ref, "row_count": len(result.answer[ref])}
                                             for ref in PLAN["outputs"]])
        self.assertEqual(set(result.evidence["datasets"]), set(SOURCES))
        self.assertTrue(all(len(item["schema_sha256"]) == 64 and
                            len(item["checksum_sha256"]) == 64
                            for item in result.evidence["datasets"].values()))
        self.assertEqual(record["result_semantics"], PLAN["result_semantics"])
        self.assertIn("Assumptions and limits", result.explanation)
        self.assertIn("Computed findings", result.explanation)
        self.assertEqual(result.explanation_source, "deterministic_execution_evidence")

    def test_invalid_plan_is_not_reported_as_executed(self):
        engine = QAEngine(approved_file_roots=[DEMO_DIR])
        bad_plan = {"steps": [{"id": "bad", "op": "scan", "dataset": "missing"}]}
        result = engine.ask_multi_dataset("invalid plan", bad_plan, SOURCES, RELATIONSHIPS)
        self.assertEqual(result.status, "abstained")
        self.assertIsNone(result.answer)
        self.assertEqual(result.evidence["execution"]["status"], "not_executed")
        self.assertEqual(result.evidence["execution"]["operations"], [])
        self.assertIn("Unknown dataset", result.reason)
        self.assertIn("Not verified", result.explanation)

    def test_contradictory_model_explanation_is_never_used(self):
        planner = MagicMock()
        planner.explain_result.return_value = "The average salary was 999999 dollars."
        engine = QAEngine(gemma_planner=planner, approved_file_roots=[Path.cwd() / "data"])
        result = engine.ask("What is the average salary?", "data/sample.csv", explain_with_gemma=True)
        self.assertEqual(result.answer, 63500.0)
        self.assertNotIn("999999", result.explanation)
        planner.explain_result.assert_not_called()
        self.assertEqual(result.explanation_source, "deterministic_execution_evidence")

    def test_plan_failure_does_not_contain_claimed_operations_or_dataset_rows(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        data = Path(root.name) / "private.csv"
        data.write_text("name,score\nPrivate Person,2\n", encoding="utf-8")
        engine = QAEngine(approved_file_roots=[root.name])
        bad = {"steps": [{"id": "scan", "op": "scan", "dataset": "unregistered"}]}
        result = engine.ask_multi_dataset("invalid", bad, {"private": data}, [])
        self.assertEqual(result.status, "abstained")
        self.assertNotIn("Private Person", json.dumps(result.evidence))
        self.assertEqual(result.evidence["execution"]["operations"], [])

    def test_runtime_failure_returns_abstention_and_no_success_trace(self):
        engine = QAEngine(approved_file_roots=[Path.cwd() / "data"])
        with patch("src.engine.evaluate_intent", side_effect=RuntimeError("secret row value")):
            result = engine.ask("How many rows?", "data/sample.csv")
        self.assertEqual(result.status, "abstained")
        self.assertIsNone(result.answer)
        self.assertEqual(result.evidence["execution"]["reason_code"], "execution_error")
        self.assertNotIn("secret row value", json.dumps(result.evidence))

    def test_multi_dataset_runtime_failure_is_abstention_with_failed_status(self):
        engine = QAEngine(approved_file_roots=[DEMO_DIR])
        with patch("src.analytics.plan.AnalyticalPlanExecutor.execute",
                   side_effect=RuntimeError("sensitive runtime detail")):
            result = engine.ask_multi_dataset("verified demo", PLAN, SOURCES, RELATIONSHIPS)
        record = result.evidence["execution"]
        self.assertEqual(result.status, "abstained")
        self.assertIsNone(result.answer)
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["operations_known"])
        self.assertNotIn("sensitive runtime detail", json.dumps(result.evidence))
        self.assertNotIn("sensitive runtime detail", result.reason)


if __name__ == "__main__":
    unittest.main()
