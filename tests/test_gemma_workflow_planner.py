import json
import unittest
from unittest.mock import MagicMock

from src.reasoning.llm_planner import GemmaQueryPlanner


SCHEMAS = {
    "products": {"product_id": "text", "name": "text", "category": "text"},
    "sales": {"product_id": "text", "region": "text", "period": "text", "units": "numeric", "revenue": "numeric"},
    "inventory": {"product_id": "text", "on_hand": "numeric", "supplier_id": "text"},
    "suppliers": {"supplier_id": "text", "name": "text", "lead_time_days": "numeric", "defect_rate": "numeric"},
}
RELATIONSHIPS = [
    {"left_dataset": "sales", "left_key": "product_id", "right_dataset": "inventory", "right_key": "product_id"},
    {"left_dataset": "sales", "left_key": "product_id", "right_dataset": "products", "right_key": "product_id"},
    {"left_dataset": "inventory", "left_key": "product_id", "right_dataset": "products", "right_key": "product_id"},
    {"left_dataset": "inventory", "left_key": "supplier_id", "right_dataset": "suppliers", "right_key": "supplier_id"},
]


def proposal():
    return {
        "workflow": "inventory_sales_regional_comparison",
        "datasets": {name: name for name in SCHEMAS},
        "columns": {name: list(cols) for name, cols in SCHEMAS.items()},
        "relationships": RELATIONSHIPS,
        "operations": ["scan", "filter", "aggregate", "compare_periods", "derive", "join"],
        "previous_period": "previous", "current_period": "current",
    }


class TestGemmaWorkflowPlanner(unittest.TestCase):
    def test_successful_multi_dataset_proposal(self):
        client = MagicMock()
        client.generate.return_value = {"parsed": proposal(), "raw": "{...}"}
        result = GemmaQueryPlanner(client).propose_analytical_plan("analyze sales and stock", SCHEMAS, RELATIONSHIPS)
        self.assertEqual(result["workflow"], "inventory_sales_regional_comparison")
        self.assertEqual(result["metrics"]["repair_attempts"], 0)

    def test_malformed_json_gets_one_repair_attempt(self):
        client = MagicMock()
        client.generate.side_effect = [ValueError("invalid JSON: {…"), {"parsed": proposal()}]
        result = GemmaQueryPlanner(client).propose_analytical_plan("analyze", SCHEMAS, RELATIONSHIPS)
        self.assertEqual(result["metrics"]["repair_attempts"], 1)
        self.assertEqual(client.generate.call_count, 2)

    def test_operation_step_arrays_are_repaired_to_supported_names(self):
        invalid = proposal()
        invalid["operations"] = [["scan", "sales", "current"], ["join", "sales", "products"]]
        client = MagicMock()
        client.generate.side_effect = [
            {"parsed": invalid, "raw": '{"operations":[["scan","sales","current"]]}'},
            {"parsed": proposal(), "raw": "corrected supported operation names"},
        ]
        result = GemmaQueryPlanner(client).propose_analytical_plan("analyze sales and stock", SCHEMAS, RELATIONSHIPS)
        self.assertEqual(result["metrics"]["repair_attempts"], 1)
        repair_prompt = client.generate.call_args_list[1].kwargs["prompt"]
        self.assertIn("Operation entries must be strings", repair_prompt)
        self.assertIn('["scan", "filter", "aggregate", "compare_periods", "derive", "join"]', repair_prompt)
        self.assertIn('"right_dataset": "suppliers"', repair_prompt)

    def test_compiler_required_column_omission_gets_repaired(self):
        invalid = proposal()
        invalid["columns"]["inventory"].remove("supplier_id")
        client = MagicMock()
        client.generate.side_effect = [
            {"parsed": invalid, "raw": json.dumps(invalid)},
            {"parsed": proposal(), "raw": "corrected complete columns"},
        ]
        result = GemmaQueryPlanner(client).propose_analytical_plan("find suppliers for shortages", SCHEMAS, RELATIONSHIPS)
        self.assertEqual(result["metrics"]["repair_attempts"], 1)
        repair_prompt = client.generate.call_args_list[1].kwargs["prompt"]
        self.assertIn("supplier_id", repair_prompt)
        self.assertIn("omits compiler-required columns", repair_prompt)

    def test_missing_required_operation_rejected(self):
        plan = proposal()
        plan["operations"].remove("filter")
        with self.assertRaisesRegex(ValueError, "requires each supported operation"):
            GemmaQueryPlanner._validate_workflow_proposal(plan, SCHEMAS, RELATIONSHIPS)

    def test_invalid_column_rejected(self):
        plan = proposal()
        plan["columns"]["sales"].append("category")
        with self.assertRaisesRegex(ValueError, "Invalid column"):
            GemmaQueryPlanner._validate_workflow_proposal(plan, SCHEMAS, RELATIONSHIPS)

    def test_unsupported_operation_rejected(self):
        plan = proposal()
        plan["operations"].append("execute_sql")
        with self.assertRaisesRegex(ValueError, "Unsupported operation"):
            GemmaQueryPlanner._validate_workflow_proposal(plan, SCHEMAS, RELATIONSHIPS)

    def test_nested_operation_descriptions_rejected(self):
        plan = proposal()
        plan["operations"] = [["scan", "sales", "current"]]
        with self.assertRaisesRegex(ValueError, "Operation entries must be strings"):
            GemmaQueryPlanner._validate_workflow_proposal(plan, SCHEMAS, RELATIONSHIPS)

    def test_missing_dataset_reference_rejected(self):
        plan = proposal()
        del plan["datasets"]["suppliers"]
        with self.assertRaisesRegex(ValueError, "dataset reference"):
            GemmaQueryPlanner._validate_workflow_proposal(plan, SCHEMAS, RELATIONSHIPS)

    def test_unapproved_or_missing_relationship_rejected(self):
        plan = proposal()
        plan["relationships"] = plan["relationships"][:-1]
        with self.assertRaisesRegex(ValueError, "relationship"):
            GemmaQueryPlanner._validate_workflow_proposal(plan, SCHEMAS, RELATIONSHIPS)


if __name__ == "__main__":
    unittest.main()
