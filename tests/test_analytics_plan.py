import copy
import unittest
from pathlib import Path

from src.analytics.plan import AnalyticalPlanExecutor, PlanError
from src.demo_enterprise import DEMO_DIR, PLAN, RELATIONSHIPS, SOURCES, expected_summary
from src.engine import QAEngine
from src.ingestion.loader import load_dataset


class TestMultiDatasetAnalyticsPlan(unittest.TestCase):
    def test_enterprise_demo_matches_independent_ground_truth(self):
        engine = QAEngine(approved_file_roots=[DEMO_DIR])
        result = engine.ask_multi_dataset(
            "Which products are at supply risk?", PLAN, SOURCES, RELATIONSHIPS
        )
        self.assertEqual(result.status, "answered", result.reason)
        expected = expected_summary()
        risks = result.answer["supplier_risk"]
        self.assertEqual([r["product_id"] for r in risks], ["P100"])
        bike = risks[0]
        self.assertEqual(bike["shortage_units"], expected["shortage"])
        self.assertEqual(bike["supplier_name"], expected["supplier"])
        self.assertEqual(int(bike["lead_time_days"]), expected["lead_time_days"])
        self.assertAlmostEqual(bike["scenario_revenue_exposure"], expected["scenario_revenue_exposure"])
        east = next(r for r in result.answer["regional_metrics"] if r["region"] == "East")
        self.assertEqual(east["units_previous"], expected["east_units_previous"])
        self.assertEqual(east["units_current"], expected["east_units_current"])
        self.assertEqual(east["revenue_previous"], expected["east_revenue_previous"])
        self.assertEqual(east["revenue_current"], expected["east_revenue_current"])
        self.assertGreater(east["units_current"], east["units_previous"])
        self.assertLess(east["revenue_current"], east["revenue_previous"])
        self.assertEqual(set(result.evidence["datasets"]), set(SOURCES))
        self.assertTrue(all(len(x["checksum_sha256"]) == 64 for x in result.evidence["datasets"].values()))

    def test_unapproved_join_rejected_before_execution(self):
        datasets = {name: load_dataset(path) for name, path in SOURCES.items()}
        invalid = copy.deepcopy(PLAN)
        join = next(s for s in invalid["steps"] if s["id"] == "product_stock")
        join["on"] = [["region", "product_id"]]
        with self.assertRaises(PlanError):
            AnalyticalPlanExecutor(datasets, RELATIONSHIPS).execute(invalid)

    def test_unknown_column_rejected_before_execution(self):
        datasets = {name: load_dataset(path) for name, path in SOURCES.items()}
        invalid = copy.deepcopy(PLAN)
        scan = next(s for s in invalid["steps"] if s["id"] == "sales")
        scan["columns"].append("customer_ssn")
        with self.assertRaises(PlanError):
            AnalyticalPlanExecutor(datasets, RELATIONSHIPS).execute(invalid)

    def test_resource_limit_rejects_large_intermediate(self):
        datasets = {name: load_dataset(path) for name, path in SOURCES.items()}
        small = AnalyticalPlanExecutor(datasets, RELATIONSHIPS, max_rows=2)
        with self.assertRaises(PlanError):
            small.execute({"steps": [{"id": "sales", "op": "scan", "dataset": "sales"}]})


if __name__ == "__main__":
    unittest.main()
