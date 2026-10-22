import copy
import csv
from collections import defaultdict
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

    def test_rank_and_trend_outputs_match_independent_variant_calculation(self):
        variant = DEMO_DIR.parent / "enterprise_demo_variant" / "sales.csv"
        with variant.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))
        totals = defaultdict(lambda: {"previous": [0, 0], "current": [0, 0]})
        for row in records:
            totals[row["product_id"]][row["period"]][0] += int(row["units"])
            totals[row["product_id"]][row["period"]][1] += int(row["revenue"])
        expected_trends = sorted(
            product for product, periods in totals.items()
            if periods["current"][0] > periods["previous"][0]
            and periods["current"][1] < periods["previous"][1]
        )
        region_totals = defaultdict(lambda: {"previous": 0, "current": 0})
        for row in records:
            region_totals[row["region"]][row["period"]] += int(row["revenue"])
        expected_region = min(
            region_totals,
            key=lambda region: (
                (region_totals[region]["current"] - region_totals[region]["previous"])
                / region_totals[region]["previous"], region
            ),
        )

        datasets = {name: load_dataset(path) for name, path in SOURCES.items()}
        datasets["sales"] = load_dataset(variant)
        result = AnalyticalPlanExecutor(datasets, RELATIONSHIPS).execute(PLAN)
        self.assertEqual(sorted(row["product_id"] for row in result["outputs"]["product_trends"]), expected_trends)
        self.assertEqual(result["outputs"]["regional_decline_ranked"][0]["region"], expected_region)

        datasets["sales"].rows.reverse()
        reordered = AnalyticalPlanExecutor(datasets, RELATIONSHIPS).execute(PLAN)
        self.assertEqual(reordered["outputs"]["regional_decline_ranked"], result["outputs"]["regional_decline_ranked"])
        self.assertEqual(reordered["outputs"]["product_trends"], result["outputs"]["product_trends"])

    def test_rank_rejects_ambiguous_direction_and_non_integer_limit(self):
        datasets = {name: load_dataset(path) for name, path in SOURCES.items()}
        for field, value in (("descending", "false"), ("limit", True)):
            invalid = copy.deepcopy(PLAN)
            rank = next(step for step in invalid["steps"] if step["op"] == "rank")
            rank[field] = value
            with self.subTest(field=field), self.assertRaises(PlanError):
                AnalyticalPlanExecutor(datasets, RELATIONSHIPS).validate(invalid, None)

    def test_rank_ties_are_stable_independent_of_input_row_order(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ties.csv"
            path.write_text("item,score\nZeta,10\nAlpha,10\nBeta,8\n", encoding="utf-8")
            dataset = load_dataset(path)
            plan = {"steps": [
                {"id": "scan", "op": "scan", "dataset": "ties"},
                {"id": "ranked", "op": "rank", "input": "scan", "column": "score",
                 "descending": True, "limit": 2},
            ], "outputs": ["ranked"]}
            first = AnalyticalPlanExecutor({"ties": dataset}, []).execute(plan)["outputs"]["ranked"]
            dataset.rows.reverse()
            second = AnalyticalPlanExecutor({"ties": dataset}, []).execute(plan)["outputs"]["ranked"]
            self.assertEqual(first, second)
            self.assertEqual([row["item"] for row in first], ["Alpha", "Zeta"])

    def test_semantics_must_reference_emitted_results_and_real_metrics(self):
        datasets = {name: load_dataset(path) for name, path in SOURCES.items()}
        invalid = copy.deepcopy(PLAN)
        invalid["result_semantics"]["computed_results"]["not_emitted"] = "Invented result."
        with self.assertRaisesRegex(PlanError, "emitted outputs"):
            AnalyticalPlanExecutor(datasets, RELATIONSHIPS).validate(invalid, None)
        invalid = copy.deepcopy(PLAN)
        invalid["result_semantics"]["assumptions"][0]["metric"] = "invented_metric"
        with self.assertRaisesRegex(PlanError, "not produced"):
            AnalyticalPlanExecutor(datasets, RELATIONSHIPS).validate(invalid, None)

    def test_compare_periods_rejects_duplicate_group_period_rows(self):
        datasets = {"sales": load_dataset(SOURCES["sales"])}
        invalid = {"steps": [
            {"id": "sales", "op": "scan", "dataset": "sales",
             "columns": ["region", "period", "units"]},
            {"id": "compared", "op": "compare_periods", "input": "sales",
             "group_by": ["region"], "period_column": "period",
             "previous": "previous", "current": "current", "metrics": ["units"]},
        ], "outputs": ["compared"]}
        with self.assertRaisesRegex(PlanError, "duplicate period rows"):
            AnalyticalPlanExecutor(datasets, RELATIONSHIPS).execute(invalid)


if __name__ == "__main__":
    unittest.main()
