import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.engine import QAEngine


@dataclass
class BenchmarkItemResult:
    item_id: str
    question: str
    category: str
    expected_status: str
    actual_status: str
    expected_answer: Any
    actual_answer: Any
    passed: bool
    reason: Optional[str] = None


@dataclass
class BenchmarkReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    accuracy: float = 0.0
    category_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    item_results: List[BenchmarkItemResult] = field(default_factory=list)


def compare_answers(expected: Any, actual: Any) -> bool:
    """Compare expected and actual answers handling floats, lists, and dicts."""
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False

    # Numeric comparison
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-4, abs_tol=1e-4)

    # List comparison (case-insensitive for column names)
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return sorted([str(x).lower() for x in expected]) == sorted([str(x).lower() for x in actual])

    # Dict comparison (category counts)
    if isinstance(expected, dict) and isinstance(actual, dict):
        if len(expected) != len(actual):
            return False
        for k, v in expected.items():
            if str(k) not in actual or actual[str(k)] != v:
                return False
        return True

    return expected == actual


def run_benchmark(
    benchmark_file: Union[str, Path] = "data/benchmark_questions.json",
    verbose: bool = True,
) -> BenchmarkReport:
    """
    Executes all benchmark questions and measures accuracy deterministically.
    """
    path = Path(benchmark_file)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    engine = QAEngine(approved_file_roots=[Path.cwd() / "data"])
    results: List[BenchmarkItemResult] = []
    category_counts: Dict[str, Dict[str, int]] = {}

    for item in questions:
        q_id = item["id"]
        dataset_path = item["dataset"]
        question_text = item["question"]
        expected_status = item["expected_status"]
        expected_ans = item["expected_answer"]
        category = item.get("category", "general")

        qa_res = engine.ask(question_text, dataset_path=dataset_path)

        passed = False
        if expected_status == "abstained":
            passed = (qa_res.status == "abstained")
        elif expected_status == "answered":
            passed = (qa_res.status == "answered") and compare_answers(expected_ans, qa_res.answer)

        res_item = BenchmarkItemResult(
            item_id=q_id,
            question=question_text,
            category=category,
            expected_status=expected_status,
            actual_status=qa_res.status,
            expected_answer=expected_ans,
            actual_answer=qa_res.answer,
            passed=passed,
            reason=qa_res.reason,
        )
        results.append(res_item)

        if category not in category_counts:
            category_counts[category] = {"total": 0, "passed": 0}
        category_counts[category]["total"] += 1
        if passed:
            category_counts[category]["passed"] += 1

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count
    accuracy = (passed_count / total * 100.0) if total > 0 else 0.0

    category_metrics = {
        cat: {
            "total": d["total"],
            "passed": d["passed"],
            "accuracy": (d["passed"] / d["total"] * 100.0) if d["total"] > 0 else 0.0,
        }
        for cat, d in category_counts.items()
    }

    report = BenchmarkReport(
        total=total,
        passed=passed_count,
        failed=failed_count,
        accuracy=accuracy,
        category_metrics=category_metrics,
        item_results=results,
    )

    if verbose:
        print_benchmark_report(report)

    return report


def print_benchmark_report(report: BenchmarkReport) -> None:
    print("=" * 72)
    print("             QUESTION-ANSWERING BENCHMARK EVALUATION")
    print("=" * 72)
    print(f"Total Questions Evaluated : {report.total}")
    print(f"Passed                    : {report.passed}")
    print(f"Failed                    : {report.failed}")
    print(f"Overall Measured Accuracy : {report.accuracy:.2f}%")
    print("-" * 72)
    print(f"{'Category':<24} | {'Passed / Total':<16} | {'Accuracy':<10}")
    print("-" * 72)
    for cat, metrics in sorted(report.category_metrics.items()):
        p_t = f"{metrics['passed']} / {metrics['total']}"
        acc_str = f"{metrics['accuracy']:.1f}%"
        print(f"{cat:<24} | {p_t:<16} | {acc_str:<10}")
    print("-" * 72)

    if report.failed > 0:
        print("\nFailures:")
        for r in report.item_results:
            if not r.passed:
                print(f"  [{r.item_id}] {r.question}")
                print(f"     Expected: status={r.expected_status}, answer={r.expected_answer}")
                print(f"     Actual:   status={r.actual_status}, answer={r.actual_answer}, reason={r.reason}")
    else:
        print("\nAll benchmark queries passed successfully!")
    print("=" * 72)


def run_gemma_model_benchmark(
    benchmark_file: Union[str, Path] = "data/benchmark_questions.json",
    sample_size: int = 12,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates Gemma 3 4B structured query-plan accuracy, latency, and end-to-end execution.
    """
    import time
    from src.models.ollama_client import OllamaClient
    from src.reasoning.llm_planner import GemmaQueryPlanner

    client = OllamaClient()
    if not client.is_healthy():
        print("Ollama service offline. Cannot run Gemma benchmark.")
        return {"error": "Ollama service offline"}

    planner = GemmaQueryPlanner(ollama_client=client)
    engine = QAEngine(gemma_planner=planner, approved_file_roots=[Path.cwd() / "data"])

    path = Path(benchmark_file)
    with path.open("r", encoding="utf-8") as f:
        all_questions = json.load(f)

    # Select a diverse representative sample across all categories
    categories_seen = set()
    sample_questions = []
    for q in all_questions:
        cat = q.get("category", "")
        if cat not in categories_seen or len(sample_questions) < sample_size:
            sample_questions.append(q)
            categories_seen.add(cat)
        if len(sample_questions) >= sample_size:
            break

    total = len(sample_questions)
    valid_plans = 0
    correct_intents = 0
    correct_columns = 0
    correct_abstentions = 0
    e2e_correct = 0
    latencies: List[float] = []

    if verbose:
        print("\n" + "=" * 72)
        print("          GEMMA 3 4B STRUCTURED QUERY-PLANNING BENCHMARK")
        print("=" * 72)

    for item in sample_questions:
        q_text = item["question"]
        ds_path = item["dataset"]
        exp_status = item["expected_status"]
        exp_ans = item["expected_answer"]
        exp_cat = item["category"]

        ds = engine.load(ds_path)
        t0 = time.perf_counter()

        try:
            plan = planner.propose_plan(
                question=q_text,
                columns=ds.columns,
                column_types=ds.column_types,
                dataset_name=ds.name,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            # 1. Plan Validity
            is_valid_plan = isinstance(plan, dict) and "operation" in plan
            if is_valid_plan:
                valid_plans += 1

            # 2. Intent matching
            op = plan.get("operation")
            intent_match = False
            if exp_status == "abstained":
                intent_match = op in {"unsupported", "ambiguous"}
                if intent_match:
                    correct_abstentions += 1
            else:
                intent_match = (op == exp_cat or (exp_cat == "numeric_aggregation" and op == "numeric_aggregation"))

            if intent_match:
                correct_intents += 1

            # 3. Column matching
            col = plan.get("column")
            if exp_status == "answered" and exp_cat in {"numeric_aggregation", "missing_values", "category_count"}:
                # check column
                if col and col in ds.columns:
                    correct_columns += 1

            # 4. End-to-end execution with Gemma planning
            qa_res = engine.ask(q_text, dataset_path=ds_path, use_gemma=True)
            passed_e2e = False
            if exp_status == "abstained":
                passed_e2e = (qa_res.status == "abstained")
            elif exp_status == "answered":
                passed_e2e = (qa_res.status == "answered") and compare_answers(exp_ans, qa_res.answer)

            if passed_e2e:
                e2e_correct += 1

            if verbose:
                status_icon = "PASS" if passed_e2e else "FAIL"
                print(f"[{status_icon}] {q_text[:45]:<45} | Plan: {op:<15} | {elapsed_ms:.1f}ms")

        except Exception as err:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            if verbose:
                print(f"[ERR ] {q_text[:45]:<45} | Error: {str(err)[:30]}")

    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    p_validity = (valid_plans / total * 100.0) if total else 0.0
    p_intent = (correct_intents / total * 100.0) if total else 0.0
    p_e2e = (e2e_correct / total * 100.0) if total else 0.0

    metrics = {
        "total_evaluated": total,
        "valid_plans": valid_plans,
        "plan_validity_rate": p_validity,
        "intent_accuracy": p_intent,
        "end_to_end_accuracy": p_e2e,
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }

    if verbose:
        print("-" * 72)
        print(f"Total Evaluated        : {total}")
        print(f"Plan Validity Rate     : {p_validity:.1f}%")
        print(f"Intent Matching Rate   : {p_intent:.1f}%")
        print(f"End-to-End Accuracy    : {p_e2e:.1f}%")
        print(f"Average Inference Time : {avg_latency:.1f} ms")
        print("=" * 72)

    return metrics


if __name__ == "__main__":
    from typing import Union
    report = run_benchmark()
    if report.failed > 0:
        sys.exit(1)
