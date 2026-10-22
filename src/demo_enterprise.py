"""Reproducible four-dataset enterprise analytics demonstration."""
import argparse
import copy
import json
import time
import traceback
from pathlib import Path

from src.engine import QAEngine
from src.analytics.plan import AnalyticalPlanExecutor
from src.ingestion.loader import load_dataset
from src.models.ollama_client import OllamaClient
from src.reasoning.llm_planner import GemmaQueryPlanner
from src.security.permissions import UserContext


DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "enterprise_demo"
QUESTION = (
    "Which products have current sales demand above inventory, which suppliers are linked to them, "
    "and did any region increase units while revenue declined? Estimate revenue exposure only as a scenario."
)
SOURCES = {name: DEMO_DIR / f"{name}.csv" for name in ("products", "sales", "inventory", "suppliers")}
RELATIONSHIPS = [
    ("sales", "product_id", "inventory", "product_id"),
    ("sales", "product_id", "products", "product_id"),
    ("inventory", "product_id", "products", "product_id"),
    ("inventory", "supplier_id", "suppliers", "supplier_id"),
]
PLAN = {
    "steps": [
        {"id": "sales", "op": "scan", "dataset": "sales", "columns": ["product_id", "region", "period", "units", "revenue"]},
        {"id": "by_product_period", "op": "aggregate", "input": "sales", "group_by": ["product_id", "period"],
         "metrics": {"units": {"function": "sum", "column": "units"}, "revenue": {"function": "sum", "column": "revenue"}}},
        {"id": "product_comparison", "op": "compare_periods", "input": "by_product_period", "group_by": ["product_id"],
         "period_column": "period", "previous": "previous", "current": "current", "metrics": ["units", "revenue"]},
        {"id": "product_metrics", "op": "derive", "input": "product_comparison", "calculations": {
            "asp_previous": {"op": "divide", "left": "revenue_previous", "right": "units_previous"},
            "asp_current": {"op": "divide", "left": "revenue_current", "right": "units_current"}}},
        {"id": "inventory", "op": "scan", "dataset": "inventory", "columns": ["product_id", "on_hand", "supplier_id"]},
        {"id": "product_stock", "op": "join", "left": "product_metrics", "right": "inventory", "on": [["product_id", "product_id"]]},
        {"id": "stock_metrics", "op": "derive", "input": "product_stock", "calculations": {
            "shortage_units": {"op": "max_zero", "left": "units_current", "right": "on_hand"},
            "scenario_revenue_exposure": {"op": "multiply", "left": "shortage_units", "right": "asp_current"}}},
        {"id": "at_risk", "op": "filter", "input": "stock_metrics",
         "filter": {"column": "shortage_units", "operator": ">", "value": 0}},
        {"id": "products", "op": "scan", "dataset": "products", "columns": ["product_id", "name", "category"], "rename": {"name": "product_name"}},
        {"id": "risk_products", "op": "join", "left": "at_risk", "right": "products", "on": [["product_id", "product_id"]]},
        {"id": "suppliers", "op": "scan", "dataset": "suppliers", "columns": ["supplier_id", "name", "lead_time_days", "defect_rate"], "rename": {"name": "supplier_name"}},
        {"id": "supplier_risk", "op": "join", "left": "risk_products", "right": "suppliers", "on": [["supplier_id", "supplier_id"]]},
        {"id": "regional_sales", "op": "aggregate", "input": "sales", "group_by": ["region", "period"],
         "metrics": {"units": {"function": "sum", "column": "units"}, "revenue": {"function": "sum", "column": "revenue"}}},
        {"id": "regional_comparison", "op": "compare_periods", "input": "regional_sales", "group_by": ["region"],
         "period_column": "period", "previous": "previous", "current": "current", "metrics": ["units", "revenue"]},
        {"id": "regional_metrics", "op": "derive", "input": "regional_comparison", "calculations": {
            "asp_previous": {"op": "divide", "left": "revenue_previous", "right": "units_previous"},
            "asp_current": {"op": "divide", "left": "revenue_current", "right": "units_current"}}},
    ],
    "outputs": ["supplier_risk", "regional_metrics"],
}


def expected_summary():
    """Independent expectations, calculated directly from the synthetic input rows."""
    sales = load_dataset(SOURCES["sales"]).rows
    product = "P100"
    old = [r for r in sales if r["product_id"] == product and r["period"] == "previous"]
    new = [r for r in sales if r["product_id"] == product and r["period"] == "current"]
    old_units, new_units = sum(int(r["units"]) for r in old), sum(int(r["units"]) for r in new)
    old_rev, new_rev = sum(int(r["revenue"]) for r in old), sum(int(r["revenue"]) for r in new)
    inventory = {r["product_id"]: r for r in load_dataset(SOURCES["inventory"]).rows}
    shortage = max(0, new_units - int(inventory[product]["on_hand"]))
    supplier = {r["supplier_id"]: r for r in load_dataset(SOURCES["suppliers"]).rows}[inventory[product]["supplier_id"]]
    region_east = [r for r in sales if r["region"] == "East"]
    east_old = [r for r in region_east if r["period"] == "previous"]
    east_new = [r for r in region_east if r["period"] == "current"]
    return {
        "product": product, "previous_units": old_units, "current_units": new_units,
        "previous_revenue": old_rev, "current_revenue": new_rev, "shortage": shortage,
        "supplier": supplier["name"], "lead_time_days": int(supplier["lead_time_days"]),
        "scenario_revenue_exposure": shortage * (new_rev / new_units),
        "east_units_previous": sum(int(r["units"]) for r in east_old),
        "east_units_current": sum(int(r["units"]) for r in east_new),
        "east_revenue_previous": sum(int(r["revenue"]) for r in east_old),
        "east_revenue_current": sum(int(r["revenue"]) for r in east_new),
    }


def compile_workflow_proposal(proposed, allowed_relationships):
    """Compile the one supported workflow only after checking its complete contract."""
    required_columns = {
        "products": {"product_id", "name", "category"},
        "sales": {"product_id", "region", "period", "units", "revenue"},
        "inventory": {"product_id", "on_hand", "supplier_id"},
        "suppliers": {"supplier_id", "name", "lead_time_days", "defect_rate"},
    }
    if any(not required.issubset(set(proposed["columns"][name]))
           for name, required in required_columns.items()):
        raise ValueError("Model proposal omitted columns required by the supported workflow.")
    required_ops = {"scan", "aggregate", "compare_periods", "derive", "join", "filter"}
    if set(proposed["operations"]) != required_ops:
        raise ValueError("Model proposal did not select the complete supported operation set.")
    rel_keys = ("left_dataset", "left_key", "right_dataset", "right_key")
    required_relationships = {tuple(r[k] for k in rel_keys) for r in allowed_relationships}
    selected_list = [tuple(r[k] for k in rel_keys) for r in proposed["relationships"]]
    if set(selected_list) != required_relationships or len(selected_list) != len(set(selected_list)):
        raise ValueError("Model proposal omitted or changed a relationship used by the workflow.")
    if (proposed["previous_period"], proposed["current_period"]) != ("previous", "current"):
        raise ValueError("Model proposal period labels do not match this dataset.")
    return copy.deepcopy(PLAN)


def run_demo(use_gemma=False):
    engine = QAEngine(approved_file_roots=[DEMO_DIR])
    plan = PLAN
    gemma_metrics = None
    if use_gemma:
        client = OllamaClient(model_name="gemma3:4b", timeout_seconds=90)
        if not client.is_healthy():
            gemma_metrics = {"planning_success": False, "reason": "Local Ollama is unavailable."}
        else:
            planner = GemmaQueryPlanner(client)
            schemas = {name: {c: ds.column_types.get(c, "text") for c in ds.columns}
                       for name, ds in ((n, load_dataset(p)) for n, p in SOURCES.items())}
            allowed = [{"left_dataset": a, "left_key": b, "right_dataset": c, "right_key": d}
                       for a, b, c, d in RELATIONSHIPS]
            started = time.perf_counter()
            try:
                proposed = planner.propose_analytical_plan(QUESTION, schemas, allowed)
                compiled = compile_workflow_proposal(proposed, allowed)
                planning_ms = (time.perf_counter() - started) * 1000
                gemma_metrics = {"planning_success": True, "planning_latency_ms": planning_ms,
                                 "proposed_workflow": proposed, "compiled_step_count": len(compiled["steps"]),
                                 "compiled_plan": compiled, "execution_mode": "gemma_workflow_compiled_deterministic_plan"}
                validation_started = time.perf_counter()
                try:
                    plan_datasets = {name: load_dataset(path) for name, path in SOURCES.items()}
                    AnalyticalPlanExecutor(plan_datasets, RELATIONSHIPS).validate(compiled, None)
                    gemma_metrics["validation_success"] = True
                    plan = compiled
                except Exception as exc:
                    gemma_metrics.update({"validation_success": False, "validation_failure": repr(exc),
                                          "validation_traceback": traceback.format_exc()})
                gemma_metrics["validation_latency_ms"] = (time.perf_counter() - validation_started) * 1000
            except Exception as exc:
                gemma_metrics = {"planning_success": False, "planning_latency_ms": (time.perf_counter() - started) * 1000,
                                 "error": repr(exc), "traceback": traceback.format_exc()}
    result = engine.ask_multi_dataset(QUESTION, plan, SOURCES, RELATIONSHIPS)
    expected = expected_summary()
    correctness = False
    if result.status == "answered":
        outputs = result.answer
        risk = outputs.get("supplier_risk", [])
        regions = outputs.get("regional_metrics", [])
        p100 = next((r for r in risk if r.get("product_id") == expected["product"]), None)
        east = next((r for r in regions if r.get("region") == "East"), None)
        correctness = bool(p100 and east and
                           p100.get("shortage_units") == expected["shortage"] and
                           p100.get("product_name") == "Trail Bike" and p100.get("supplier_name") == expected["supplier"] and int(p100.get("lead_time_days")) == expected["lead_time_days"] and
                           abs(p100.get("scenario_revenue_exposure") - expected["scenario_revenue_exposure"]) < 1e-6 and
                           east.get("units_previous") == expected["east_units_previous"] and
                           east.get("units_current") == expected["east_units_current"] and
                           east.get("revenue_previous") == expected["east_revenue_previous"] and
                           east.get("revenue_current") == expected["east_revenue_current"])
    if gemma_metrics is not None:
        gemma_metrics["baseline_fallback_used"] = False
        gemma_metrics["model_plan_executed"] = plan is not PLAN and gemma_metrics.get("validation_success") is True
        gemma_metrics["execution_success"] = result.status == "answered" and gemma_metrics["model_plan_executed"]
        gemma_metrics["execution_error"] = result.reason if result.status != "answered" else None
        gemma_metrics["execution_correct"] = correctness if gemma_metrics["model_plan_executed"] else False
        gemma_metrics["end_to_end_correct"] = correctness if gemma_metrics["model_plan_executed"] else False
        if plan is PLAN:
            # A rejected model proposal falls back to the deterministic baseline, but is not
            # counted as a successful model-planned end-to-end run.
            gemma_metrics["baseline_fallback_used"] = True
            gemma_metrics["execution_mode"] = "deterministic_fallback"
        elif result.status != "answered" or not correctness:
            result = engine.ask_multi_dataset(QUESTION, PLAN, SOURCES, RELATIONSHIPS)
            _, correctness, _ = run_baseline_assertion(result)
            gemma_metrics["baseline_fallback_used"] = True
            gemma_metrics["execution_mode"] = "deterministic_fallback_after_model_execution_failure"
    print("Question:", QUESTION)
    print("Execution mode:", (gemma_metrics or {}).get("execution_mode", "deterministic_baseline"))
    print("Status:", result.status, "| Ground-truth assertions:", "PASS" if correctness else "FAIL")
    if result.status != "answered":
        print("Error:", result.reason)
    else:
        risk, regions = result.answer["supplier_risk"], result.answer["regional_metrics"]
        print("At-risk products:")
        print(json.dumps(risk, indent=2))
        print("Regional period comparison:")
        print(json.dumps(regions, indent=2))
        print("Evidence:", json.dumps(result.evidence, indent=2))
        print("Interpretation: demand above stock is a supply-risk indicator, not proof of lost sales. Revenue exposure is a scenario using current observed average selling price; it assumes the estimated shortage could be fulfilled.")
    if gemma_metrics is not None:
        print("Gemma acceptance:", json.dumps({key: gemma_metrics.get(key) for key in (
            "planning_success", "validation_success", "model_plan_executed", "execution_correct",
            "end_to_end_correct", "baseline_fallback_used", "execution_mode")}, sort_keys=True))
        print("Gemma metrics:", json.dumps(gemma_metrics, indent=2))
    return result, correctness, gemma_metrics


def run_baseline_assertion(result):
    """Check deterministic outputs against independently computed expected aggregates."""
    expected = expected_summary()
    correctness = False
    if result.status == "answered":
        risks, regions = result.answer.get("supplier_risk", []), result.answer.get("regional_metrics", [])
        bike = next((r for r in risks if r.get("product_id") == expected["product"]), None)
        east = next((r for r in regions if r.get("region") == "East"), None)
        correctness = bool(bike and east and bike.get("shortage_units") == expected["shortage"] and
                           bike.get("product_name") == "Trail Bike" and
                           bike.get("supplier_name") == expected["supplier"] and
                           int(bike.get("lead_time_days")) == expected["lead_time_days"] and
                           abs(bike.get("scenario_revenue_exposure") - expected["scenario_revenue_exposure"]) < 1e-6 and
                           east.get("units_previous") == expected["east_units_previous"] and
                           east.get("units_current") == expected["east_units_current"] and
                           east.get("revenue_previous") == expected["east_revenue_previous"] and
                           east.get("revenue_current") == expected["east_revenue_current"])
    return result, correctness, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma", action="store_true", help="Ask local Gemma to propose the plan before deterministic validation and execution.")
    args = parser.parse_args()
    result, correct, _ = run_demo(args.gemma)
    return 0 if result.status == "answered" and correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
