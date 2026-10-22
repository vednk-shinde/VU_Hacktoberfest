"""Small, privacy-conscious provenance records for verified executions."""

import hashlib
import json
import uuid
from typing import Any, Dict, Optional


def _canonical_digest(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def dataset_provenance(dataset) -> Dict[str, Any]:
    schema = {"columns": list(dataset.columns), "column_types": dict(dataset.column_types)}
    warnings = [{key: warning[key] for key in ("code", "column", "count") if key in warning}
                for warning in dataset.profile.get("quality_warnings", [])]
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "name": dataset.name,
        "source_file": dataset.file_path.name,
        "row_count": len(dataset.rows),
        "schema": schema,
        "schema_sha256": _canonical_digest(schema),
        "content_sha256": dataset.content_fingerprint,
        "quality_warnings": warnings,
    }


def summarize_filter(node: Any) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    if isinstance(node, dict) and "criteria" in node:
        return {"combinator": node.get("combinator", "AND"),
                "criteria": [summarize_filter(child) for child in node["criteria"]]}
    if isinstance(node, dict):
        summary = {key: node[key] for key in ("column", "operator") if key in node}
        if "value" in node:
            summary["value_redacted"] = True
        return summary
    if hasattr(node, "criteria"):
        return {"combinator": node.combinator,
                "criteria": [summarize_filter(child) for child in node.criteria]}
    if hasattr(node, "column") and hasattr(node, "operator"):
        return {"column": node.column, "operator": node.operator, "value_redacted": True}
    return {"available": False}


def single_execution_record(result, dataset, intent, backend: str,
                            planning_source: str = "deterministic_parser") -> Dict[str, Any]:
    columns = []
    for column in (intent.column, intent.filter_column):
        if column and column not in columns:
            columns.append(column)
    if intent.compound_filter is not None:
        # Use validated filter references, while omitting their potentially sensitive values.
        from src.reasoning.validator import filter_columns
        for column in filter_columns(intent.compound_filter):
            if column not in columns:
                columns.append(column)
    if intent.operation in {"column_names", "missing_values", "duplicate_count"} and not columns:
        columns = list(dataset.columns)
    filter_obj = intent.compound_filter
    if filter_obj is None and intent.filter_column and intent.filter_operator:
        filter_obj = {"column": intent.filter_column, "operator": intent.filter_operator,
                      "value": intent.filter_value}

    return {
        "schema_version": 1,
        "execution_id": str(uuid.uuid4()),
        "status": "completed" if result.status == "answered" else "abstained",
        "dataset": dataset_provenance(dataset),
        "intent": {
            "operation": intent.operation,
            "column": intent.column,
            "aggregation": intent.agg_function,
            "filter": summarize_filter(filter_obj),
            "columns_used": columns,
            "planning_source": planning_source,
        },
        "operations": [{"operation": intent.operation, "backend": backend,
                        "status": "completed" if result.status == "answered" else "abstained"}],
        "result_reference": "QAResult.answer",
        "limitations": list(dataset.profile.get("quality_warnings", [])),
    }


def not_executed_record(reason_code: str, dataset=None, operation: Optional[str] = None) -> Dict[str, Any]:
    record = {
        "schema_version": 1,
        "execution_id": str(uuid.uuid4()),
        "status": "not_executed",
        "operations": [],
        "reason_code": reason_code,
        "limitations": ["No answer was computed; inspect QAResult.reason."],
    }
    if dataset is not None:
        record["dataset"] = dataset_provenance(dataset)
    if operation is not None:
        record["proposed_operation"] = operation
    return record


def deterministic_explanation(result, dataset_name: str, dataset_version: Optional[int] = None) -> str:
    """Describe only executor metadata and the already returned answer, never model claims."""
    if result.status != "answered":
        return f"Not verified: {result.reason or 'execution did not produce an answer.'}"
    source = dataset_name
    if dataset_version is not None:
        source += f" (version {dataset_version})"
    method = result.calculation_method or f"deterministic {result.operation} operation"
    if result.operation == "multi_step_analytics" and isinstance(result.answer, dict):
        execution = (result.evidence or {}).get("execution", {})
        outputs = execution.get("outputs", [])
        detail = ", ".join(f"{item['step_id']} ({item['row_count']} rows)" for item in outputs)
        suffix = f" Outputs: {detail}." if detail else " See the structured answer for output rows."
        semantics = execution.get("result_semantics") or {}
        observed = semantics.get("source_facts", [])
        computed = list(semantics.get("computed_results", {}).values())
        assumptions = [item.get("description") for item in semantics.get("assumptions", [])
                       if isinstance(item, dict) and item.get("description")]
        if observed:
            suffix += " Direct source facts: " + " ".join(observed)
        if computed:
            suffix += " Computed findings: " + " ".join(computed)
        if assumptions:
            suffix += " Assumptions and limits: " + " ".join(assumptions)
    elif isinstance(result.answer, (int, float, bool, str)) or result.answer is None:
        suffix = f" Computed value: {result.answer!r}."
    else:
        suffix = f" Returned {len(result.answer)} result item(s); see the structured answer."
    return f"Verified by the deterministic executor on {source}: {method}.{suffix}"
