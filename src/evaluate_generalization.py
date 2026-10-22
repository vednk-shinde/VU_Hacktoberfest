"""Evaluate the fixed four-dataset Gemma workflow on unseen questions and changed data.

This records planning and executor success separately from whether this compiler's
fixed outputs actually answer each question. It never falls back to the baseline.
"""
import csv
import argparse
import hashlib
import json
import time
from pathlib import Path

from src.analytics.plan import AnalyticalPlanExecutor
from src.demo_enterprise import DEMO_DIR, PLAN, RELATIONSHIPS, compile_workflow_proposal
from src.ingestion.loader import load_dataset
from src.models.ollama_client import OllamaClient
from src.reasoning.llm_planner import GemmaQueryPlanner


ROOT = Path(__file__).resolve().parents[1]
VARIANT_DIR = ROOT / "data" / "enterprise_demo_variant"
BASE_SOURCES = {name: DEMO_DIR / f"{name}.csv" for name in ("products", "sales", "inventory", "suppliers")}

CASES = [
    {
        "id": "largest_region_revenue_decline",
        "question": "Which region has the largest percentage revenue decline?",
        "compiler_support": False,
        "limitation": "The compiled outputs include regional percentage changes but no rank/selection output.",
    },
    {
        "id": "products_units_up_revenue_down",
        "question": "Which products have rising units but falling revenue?",
        "compiler_support": False,
        "limitation": "The compiled product output is filtered to stock shortages and omits other trending products.",
        "variant": True,
    },
    {
        "id": "supplier_shortage_associations",
        "question": "Which suppliers are associated with products whose current-period demand exceeds stock?",
        "compiler_support": True,
        "limitation": None,
    },
    {
        "id": "direct_facts_vs_assumptions",
        "question": "Which conclusions are directly calculated, and which require assumptions about lost sales or recovered revenue?",
        "compiler_support": False,
        "limitation": "The executor returns numeric evidence and a scenario field, but does not classify direct findings versus assumptions in its result schema.",
    },
    {
        "id": "supplier_shortage_modified_sales",
        "question": "Which suppliers are associated with products whose current-period demand exceeds stock?",
        "compiler_support": True,
        "limitation": None,
        "variant": True,
    },
]


def _read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def independent_expectations(sources):
    """Compute expected results directly from CSV rows, independent of the plan executor."""
    sales = _read_rows(sources["sales"])
    inventory_rows = _read_rows(sources["inventory"])
    products_rows = _read_rows(sources["products"])
    supplier_rows = _read_rows(sources["suppliers"])

    def grouped(dimensions):
        groups = {}
        for row in sales:
            key = tuple(row[d] for d in dimensions) + (row["period"],)
            values = groups.setdefault(key, {"units": 0, "revenue": 0})
            values["units"] += int(row["units"])
            values["revenue"] += int(row["revenue"])
        return groups

    product_periods = grouped(["product_id"])
    product_changes = {}
    for product in {r["product_id"] for r in sales}:
        old, new = product_periods.get((product, "previous")), product_periods.get((product, "current"))
        if old and new:
            product_changes[product] = {
                "units_change": new["units"] - old["units"],
                "revenue_change": new["revenue"] - old["revenue"],
                "units_current": new["units"], "revenue_current": new["revenue"],
            }
    regional_periods = grouped(["region"])
    regional_changes = {}
    for region in {r["region"] for r in sales}:
        old, new = regional_periods.get((region, "previous")), regional_periods.get((region, "current"))
        if old and new and old["revenue"]:
            regional_changes[region] = {
                "units_change": new["units"] - old["units"],
                "revenue_change": new["revenue"] - old["revenue"],
                "revenue_change_pct": (new["revenue"] - old["revenue"]) / old["revenue"],
            }
    inventory = {r["product_id"]: r for r in inventory_rows}
    suppliers = {r["supplier_id"]: r for r in supplier_rows}
    products = {r["product_id"]: r for r in products_rows}
    shortage_suppliers = []
    for product, values in product_changes.items():
        stock = int(inventory[product]["on_hand"])
        if values["units_current"] > stock:
            supplier = suppliers[inventory[product]["supplier_id"]]
            shortage_suppliers.append({"product_id": product, "product_name": products[product]["name"],
                                       "supplier_name": supplier["name"],
                                       "shortage_units": values["units_current"] - stock})
    return {
        "largest_revenue_decline": min(regional_changes, key=lambda r: regional_changes[r]["revenue_change_pct"]),
        "regional_changes": regional_changes,
        "products_units_up_revenue_down": sorted(
            p for p, v in product_changes.items() if v["units_change"] > 0 and v["revenue_change"] < 0),
        "shortage_suppliers": sorted(shortage_suppliers, key=lambda r: r["product_id"]),
        "product_changes": product_changes,
    }


def run_case(case, planner):
    sources = dict(BASE_SOURCES)
    if case.get("variant"):
        sources["sales"] = VARIANT_DIR / "sales.csv"
    datasets = {name: load_dataset(path) for name, path in sources.items()}
    schemas = {name: {col: datasets[name].column_types.get(col, "text") for col in datasets[name].columns}
               for name in datasets}
    expected = independent_expectations(sources)
    record = {
        "case": case["id"], "question": case["question"], "variant_sales_sha256":
            hashlib.sha256(sources["sales"].read_bytes()).hexdigest() if case.get("variant") else None,
        "compiler_support_expected": case["compiler_support"], "compiler_limitation": case["limitation"],
        "planning_success": False, "proposal_validation_success": False,
        "compiled_plan_validation_success": False, "compiled_plan_executed": False,
        "result_matches_independent_ground_truth": False, "fallback_used": False,
        "planning_latency_ms": None, "validation_latency_ms": None, "execution_latency_ms": None,
        "exact_failure_reason": None, "raw_proposal": None, "independent_ground_truth": expected,
        "observed_relevant_outputs": None,
    }
    started = time.perf_counter()
    try:
        proposal = planner.propose_analytical_plan(case["question"], schemas, [
            {"left_dataset": a, "left_key": b, "right_dataset": c, "right_key": d}
            for a, b, c, d in RELATIONSHIPS])
        record["planning_success"] = True
        record["proposal_validation_success"] = True  # returned only after strict planner validation
        record["raw_proposal"] = proposal.get("metrics", {}).get("raw_response")
        compiled = compile_workflow_proposal(proposal, [
            {"left_dataset": a, "left_key": b, "right_dataset": c, "right_key": d}
            for a, b, c, d in RELATIONSHIPS])
    except Exception as exc:
        record["planning_latency_ms"] = (time.perf_counter() - started) * 1000
        record["exact_failure_reason"] = repr(exc)
        return record
    record["planning_latency_ms"] = (time.perf_counter() - started) * 1000

    executor = AnalyticalPlanExecutor(datasets, RELATIONSHIPS)
    started = time.perf_counter()
    try:
        executor.validate(compiled, None)
        record["compiled_plan_validation_success"] = True
    except Exception as exc:
        record["validation_latency_ms"] = (time.perf_counter() - started) * 1000
        record["exact_failure_reason"] = repr(exc)
        return record
    record["validation_latency_ms"] = (time.perf_counter() - started) * 1000

    provenance = {name: {"source_file": path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for name, path in sources.items()}
    started = time.perf_counter()
    try:
        result = executor.execute(compiled, None, provenance)
        record["compiled_plan_executed"] = True
    except Exception as exc:
        record["execution_latency_ms"] = (time.perf_counter() - started) * 1000
        record["exact_failure_reason"] = repr(exc)
        return record
    record["execution_latency_ms"] = (time.perf_counter() - started) * 1000
    outputs = result["outputs"]
    actual_risks = [{"product_id": row["product_id"], "product_name": row["product_name"],
                     "supplier_name": row["supplier_name"], "shortage_units": row["shortage_units"]}
                    for row in outputs.get("supplier_risk", [])]
    actual_regions = outputs.get("regional_metrics", [])
    record["observed_relevant_outputs"] = {
        "supplier_risk": actual_risks,
        "regional_metrics": actual_regions,
        "output_names": list(outputs),
    }
    case_id = case["id"]
    if case_id.startswith("supplier_shortage"):
        record["result_matches_independent_ground_truth"] = actual_risks == expected["shortage_suppliers"]
    elif case_id == "largest_region_revenue_decline":
        actual_by_region = {r["region"]: r.get("revenue_change_pct") for r in actual_regions}
        expected_by_region = {r: v["revenue_change_pct"] for r, v in expected["regional_changes"].items()}
        evidence_complete = actual_by_region == expected_by_region
        record["computed_evidence_matches_ground_truth"] = evidence_complete
        record["result_matches_independent_ground_truth"] = False  # no rank/selection is emitted
        record["exact_failure_reason"] = "The compiled outputs contain the correct regional percentages but emit no selected maximum/ranked region."
    elif case_id == "products_units_up_revenue_down":
        actual_ids = sorted({r["product_id"] for r in actual_risks})
        record["actual_emitted_product_ids"] = actual_ids
        record["result_matches_independent_ground_truth"] = actual_ids == expected["products_units_up_revenue_down"]
        if not record["result_matches_independent_ground_truth"]:
            record["exact_failure_reason"] = (
                f"Compiled supplier_risk emitted products {actual_ids}; independent trend calculation requires "
                f"{expected['products_units_up_revenue_down']}. The plan's stock filter omits trend-qualified products without shortages."
            )
    elif case_id == "direct_facts_vs_assumptions":
        has_scenario = any("scenario_revenue_exposure" in row for row in actual_risks)
        record["scenario_value_present"] = has_scenario
        record["result_matches_independent_ground_truth"] = False  # no direct-vs-assumption classification returned
        record["exact_failure_reason"] = "Executor returns metrics but no direct-versus-assumption classification or assumption metadata."
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    client = OllamaClient(model_name="gemma3:4b", timeout_seconds=90)
    if not client.is_healthy():
        raise SystemExit("Local Ollama is unavailable; no evaluation cases were run.")
    planner = GemmaQueryPlanner(client)
    cases = [run_case(case, planner) for case in CASES]
    report = {
        "model": "gemma3:4b", "fallback_policy": "none; invalid plans are recorded and not executed",
        "evaluation_count": len(cases), "cases": cases,
        "success_count": sum(c["planning_success"] and c["proposal_validation_success"] and
                              c["compiled_plan_validation_success"] and c["compiled_plan_executed"] and
                              c["result_matches_independent_ground_truth"] and not c["fallback_used"]
                              for c in cases),
        "limitations": [
            "The current planner accepts one fixed workflow proposal shape regardless of requested output intent.",
            "The compiler emits supplier_risk and regional_metrics only; it does not rank regions, list all products by trend, or classify evidence versus assumptions.",
        ],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved complete report: {args.output}")
    for case in cases:
        print(json.dumps({key: case.get(key) for key in (
            "case", "planning_success", "proposal_validation_success", "compiled_plan_validation_success",
            "compiled_plan_executed", "result_matches_independent_ground_truth", "fallback_used",
            "planning_latency_ms", "validation_latency_ms", "execution_latency_ms", "exact_failure_reason")},
            ensure_ascii=False))
    print(f"Overall model-planned matches: {report['success_count']}/{len(cases)}")


if __name__ == "__main__":
    main()
