"""Small, deterministic multi-dataset analytics plan interpreter.

Plans are data, never executable code. All operations, columns, joins, dependencies,
filters, and result sizes are validated before the first step runs.
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ingestion.loader import Dataset, to_number
from src.retrieval.db_executor import CompoundFilter, FilterCriterion
from src.retrieval.filters import apply_filter_object
from src.retrieval.filters import ALLOWED_OPERATORS
from src.security.permissions import UserContext, can_access_columns, can_access_dataset


class PlanError(ValueError):
    pass


class AnalyticalPlanExecutor:
    OPS = {"scan", "filter", "aggregate", "compare_periods", "derive", "join", "rank"}
    AGGS = {"sum", "average", "min", "max", "count"}
    ARITH = {"add", "subtract", "multiply", "divide", "max_zero"}

    def __init__(self, datasets: Dict[str, Dataset], relationships: List[Tuple[str, str, str, str]],
                 max_steps: int = 40, max_rows: int = 100_000):
        self.datasets = datasets
        self.relationships = set(relationships)
        self.max_steps = max_steps
        self.max_rows = max_rows

    @staticmethod
    def _filter_columns(node: Any) -> List[str]:
        if isinstance(node, FilterCriterion):
            if node.operator not in ALLOWED_OPERATORS:
                raise PlanError(f"Filter operator is not allowed: {node.operator}")
            return [node.column]
        if isinstance(node, CompoundFilter):
            if node.combinator.upper() not in {"AND", "OR"} or not node.criteria:
                raise PlanError("Malformed compound filter.")
            return [c for child in node.criteria for c in AnalyticalPlanExecutor._filter_columns(child)]
        if isinstance(node, dict):
            if "criteria" in node:
                if node.get("combinator", "AND").upper() not in {"AND", "OR"} or not isinstance(node.get("criteria"), list) or not node["criteria"]:
                    raise PlanError("Malformed compound filter.")
                return [c for child in node["criteria"] for c in AnalyticalPlanExecutor._filter_columns(child)]
            if set(node) >= {"column", "operator", "value"}:
                if node["operator"] not in ALLOWED_OPERATORS:
                    raise PlanError(f"Filter operator is not allowed: {node['operator']}")
                return [node["column"]]
        raise PlanError("Malformed filter condition.")

    @staticmethod
    def _materialize_filter(node: Any):
        if isinstance(node, (FilterCriterion, CompoundFilter)):
            return node
        if "criteria" in node:
            return CompoundFilter(node.get("combinator", "AND"), [AnalyticalPlanExecutor._materialize_filter(x) for x in node["criteria"]])
        return FilterCriterion(node["column"], node["operator"], node.get("value"))

    def validate(self, plan: Dict[str, Any], user: Optional[UserContext]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
            raise PlanError("Plan must contain a steps list.")
        steps = plan["steps"]
        if not steps or len(steps) > self.max_steps:
            raise PlanError(f"Plan must contain between 1 and {self.max_steps} steps.")
        schemas: Dict[str, Dict[str, Any]] = {}
        row_origins: Dict[str, Set[str]] = {}
        datasets_used: Set[str] = set()
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("id"), str) or step["id"] in schemas:
                raise PlanError("Every step needs a unique string id.")
            sid, op = step["id"], step.get("op")
            if op not in self.OPS:
                raise PlanError(f"Unsupported operation in step {sid}: {op}")
            if op == "scan":
                dname = step.get("dataset")
                if dname not in self.datasets:
                    raise PlanError(f"Unknown dataset: {dname}")
                ds = self.datasets[dname]
                if len(ds.rows) > self.max_rows:
                    raise PlanError(f"Dataset {dname} exceeds the {self.max_rows}-row execution limit.")
                if ds.dataset_id and not can_access_dataset(user, ds.dataset_id):
                    raise PlanError(f"Access denied to dataset {ds.dataset_id}.")
                cols = step.get("columns", ds.columns)
                if not isinstance(cols, list) or not cols or any(c not in ds.columns for c in cols):
                    raise PlanError(f"Invalid selected columns for dataset {dname}.")
                rename = step.get("rename", {})
                if not isinstance(rename, dict) or any(k not in cols or not isinstance(v, str) for k, v in rename.items()):
                    raise PlanError(f"Invalid scan aliases for dataset {dname}.")
                output_cols = [rename.get(c, c) for c in cols]
                if len(set(output_cols)) != len(output_cols):
                    raise PlanError(f"Scan aliases create duplicate output columns for dataset {dname}.")
                filter_obj = step.get("filter")
                if filter_obj is not None:
                    fcols = self._filter_columns(filter_obj)
                    if any(c not in ds.columns for c in fcols):
                        raise PlanError(f"Filter references an unknown column in {dname}.")
                    if not set(fcols).issubset(cols):
                        raise PlanError(f"Scan filter columns must also be selected in {dname}.")
                    cols_for_auth = list(set(cols + fcols))
                else:
                    cols_for_auth = cols
                if ds.dataset_id:
                    ok, denied = can_access_columns(user, ds.dataset_id, cols_for_auth)
                    if not ok:
                        raise PlanError(f"Access denied to column(s): {denied}")
                types = {rename.get(c, c): ds.column_types.get(c, "unknown") for c in cols}
                schemas[sid] = {"columns": set(output_cols), "origins": {dname}, "types": types}
                row_origins[sid] = {dname}
                datasets_used.add(dname)
            else:
                ref_fields = ["input"] if op in {"filter", "aggregate", "compare_periods", "derive", "rank"} else ["left", "right"]
                refs = [step.get(f) for f in ref_fields]
                if any(r not in schemas for r in refs):
                    raise PlanError(f"Step {sid} has a missing or forward dependency.")
                origins = set().union(*(row_origins[r] for r in refs))
                cols = set(schemas[refs[0]]["columns"])
                types = dict(schemas[refs[0]]["types"])
                if op == "filter":
                    if not set(self._filter_columns(step.get("filter"))).issubset(cols):
                        raise PlanError(f"Filter in {sid} references an unknown column.")
                elif op == "aggregate":
                    groups, metrics = step.get("group_by"), step.get("metrics")
                    if not isinstance(groups, list) or not set(groups).issubset(cols) or not isinstance(metrics, dict) or not metrics:
                        raise PlanError(f"Invalid group_by or metrics in {sid}.")
                    out = set(groups)
                    out_types = {c: types.get(c, "unknown") for c in groups}
                    for alias, spec in metrics.items():
                        if not isinstance(spec, dict) or spec.get("function") not in self.AGGS or spec.get("column") not in cols:
                            raise PlanError(f"Invalid metric {alias} in {sid}.")
                        if spec["function"] != "count" and types.get(spec["column"]) != "numeric":
                            raise PlanError(f"Metric {alias} requires a numeric source column.")
                        out.add(alias)
                        out_types[alias] = "numeric"
                    cols = out
                    types = out_types
                elif op == "compare_periods":
                    period = step.get("period_column")
                    groups, metrics = step.get("group_by"), step.get("metrics")
                    if period not in cols or not isinstance(groups, list) or not set(groups).issubset(cols) or period in groups:
                        raise PlanError(f"Invalid period/group columns in {sid}.")
                    if not isinstance(metrics, list) or not metrics or not set(metrics).issubset(cols):
                        raise PlanError(f"Invalid comparison metrics in {sid}.")
                    if any(types.get(m) != "numeric" for m in metrics):
                        raise PlanError(f"Comparison metrics in {sid} must be numeric.")
                    if step.get("current") == step.get("previous"):
                        raise PlanError("Current and previous period labels must differ.")
                    cols = set(groups)
                    types = {c: types.get(c, "unknown") for c in groups}
                    for metric in metrics:
                        cols.update({f"{metric}_previous", f"{metric}_current", f"{metric}_change", f"{metric}_change_pct"})
                        types.update({f"{metric}_previous": "numeric", f"{metric}_current": "numeric",
                                      f"{metric}_change": "numeric", f"{metric}_change_pct": "numeric"})
                elif op == "derive":
                    calculations = step.get("calculations")
                    if not isinstance(calculations, dict) or not calculations:
                        raise PlanError(f"Invalid calculations in {sid}.")
                    for alias, expr in calculations.items():
                        if not isinstance(expr, dict) or expr.get("op") not in self.ARITH:
                            raise PlanError(f"Invalid derived operation for {alias}.")
                        if expr.get("op") == "max_zero":
                            operands = [expr.get("value", expr.get("left")), expr.get("right")]
                        else:
                            operands = [expr.get("left"), expr.get("right")]
                        if any(isinstance(x, str) and x not in cols for x in operands):
                            raise PlanError(f"Derived expression {alias} references an unknown column.")
                        if any(isinstance(x, str) and types.get(x) != "numeric" for x in operands):
                            raise PlanError(f"Derived expression {alias} requires numeric inputs.")
                        cols.add(alias)
                        types[alias] = "numeric"
                elif op == "join":
                    right_cols = set(schemas[refs[1]]["columns"])
                    right_types = schemas[refs[1]]["types"]
                    pairs = step.get("on")
                    if not isinstance(pairs, list) or not pairs:
                        raise PlanError(f"Join {sid} requires explicit key pairs.")
                    for pair in pairs:
                        if not isinstance(pair, list) or len(pair) != 2 or pair[0] not in cols or pair[1] not in right_cols:
                            raise PlanError(f"Invalid join key in {sid}.")
                        left_type, right_type = types.get(pair[0]), right_types.get(pair[1])
                        if left_type not in {None, "unknown"} and right_type not in {None, "unknown"} and left_type != right_type:
                            raise PlanError(f"Join key types do not match in {sid}.")
                    # Every cross-source join must match an explicitly approved relationship.
                    for left_key, right_key in pairs:
                        approved = any(
                            (a, left_key, b, right_key) in self.relationships or
                            (b, right_key, a, left_key) in self.relationships
                            for a in row_origins[refs[0]] for b in row_origins[refs[1]]
                        )
                        if len(row_origins[refs[0]] | row_origins[refs[1]]) > 1 and not approved:
                            raise PlanError(f"Join relationship is not approved for keys {left_key} and {right_key}.")
                    collisions = (cols & right_cols) - {a for a, b in pairs if a == b}
                    if collisions:
                        raise PlanError(f"Join {sid} has ambiguous duplicate columns: {sorted(collisions)}")
                    cols.update(right_cols - {b for a, b in pairs if a == b})
                    types.update({c: t for c, t in right_types.items() if not any(c == b and a == b for a, b in pairs)})
                elif op == "rank":
                    if step.get("column") not in cols or step.get("limit", 0) <= 0 or step.get("limit", 0) > self.max_rows:
                        raise PlanError(f"Invalid rank column or limit in {sid}.")
                schemas[sid] = {"columns": cols, "origins": origins, "types": types}
                row_origins[sid] = origins
        result_refs = plan.get("outputs", [steps[-1]["id"]])
        if not isinstance(result_refs, list) or not result_refs or any(ref not in schemas for ref in result_refs):
            raise PlanError("Plan outputs must reference existing steps.")
        return schemas

    def execute(self, plan: Dict[str, Any], user: Optional[UserContext] = None,
                provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        schemas = self.validate(plan, user)
        results: Dict[str, List[Dict[str, Any]]] = {}
        comparison_gaps: Dict[str, int] = {}
        for step in plan["steps"]:
            sid, op = step["id"], step["op"]
            if op == "scan":
                ds = self.datasets[step["dataset"]]
                rows = [dict(row) for row in ds.rows]
                if step.get("filter") is not None:
                    rows = apply_filter_object(rows, self._materialize_filter(step["filter"]))
                selected = step.get("columns", ds.columns)
                rows = [{c: row.get(c) for c in selected} for row in rows]
                rename = step.get("rename", {})
                if rename:
                    rows = [{rename.get(k, k): v for k, v in row.items()} for row in rows]
            elif op == "filter":
                rows = apply_filter_object(results[step["input"]], self._materialize_filter(step["filter"]))
            elif op == "aggregate":
                source = results[step["input"]]
                buckets = defaultdict(list)
                for row in source:
                    buckets[tuple(row[c] for c in step["group_by"])].append(row)
                rows = []
                for key, members in buckets.items():
                    out = dict(zip(step["group_by"], key))
                    for alias, spec in step["metrics"].items():
                        vals = [to_number(x.get(spec["column"])) for x in members]
                        vals = [v for v in vals if v is not None]
                        fn = spec["function"]
                        out[alias] = (len(members) if fn == "count" else None) if not vals and fn != "count" else (
                            sum(vals) if fn == "sum" else sum(vals) / len(vals) if fn == "average" else
                            min(vals) if fn == "min" else max(vals) if fn == "max" else len(members))
                    rows.append(out)
            elif op == "compare_periods":
                source = results[step["input"]]
                groups = defaultdict(dict)
                for row in source:
                    groups[tuple(row[c] for c in step["group_by"])][row[step["period_column"]]] = row
                rows = []
                missing = 0
                for key, periods in groups.items():
                    old, new = periods.get(step["previous"]), periods.get(step["current"])
                    if old is None or new is None:
                        missing += 1
                        continue
                    out = dict(zip(step["group_by"], key))
                    for metric in step["metrics"]:
                        a, b = to_number(old.get(metric)), to_number(new.get(metric))
                        out[f"{metric}_previous"], out[f"{metric}_current"] = a, b
                        out[f"{metric}_change"] = None if a is None or b is None else b - a
                        out[f"{metric}_change_pct"] = None if a in (None, 0) or b is None else (b - a) / a
                    rows.append(out)
                comparison_gaps[sid] = missing
            elif op == "derive":
                rows = [dict(row) for row in results[step["input"]]]
                for row in rows:
                    for alias, expr in step["calculations"].items():
                        fn = expr["op"]
                        if fn == "max_zero":
                            value = self._value(row, expr.get("value", expr.get("left")))
                            right = self._value(row, expr.get("right")) if expr.get("right") is not None else 0
                            row[alias] = max(0, (value or 0) - (right or 0))
                            continue
                        left, right = self._value(row, expr["left"]), self._value(row, expr["right"])
                        if left is None or right is None or (fn == "divide" and right == 0):
                            row[alias] = None
                        elif fn == "add": row[alias] = left + right
                        elif fn == "subtract": row[alias] = left - right
                        elif fn == "multiply": row[alias] = left * right
                        else: row[alias] = left / right
            elif op == "join":
                left_rows, right_rows = results[step["left"]], results[step["right"]]
                indexes = defaultdict(list)
                for row in right_rows:
                    indexes[tuple(row[r] for l, r in step["on"])].append(row)
                rows = []
                for left in left_rows:
                    for right in indexes.get(tuple(left[l] for l, r in step["on"]), []):
                        merged = dict(left)
                        merged.update({k: v for k, v in right.items() if not any(k == r and l == r for l, r in step["on"])})
                        rows.append(merged)
                        if len(rows) > self.max_rows:
                            raise PlanError(f"Join step {sid} exceeded the {self.max_rows}-row execution limit.")
            elif op == "rank":
                source = results[step["input"]]
                present = [r for r in source if r.get(step["column"]) is not None]
                missing = [r for r in source if r.get(step["column"]) is None]
                present.sort(key=lambda r: r[step["column"]], reverse=step.get("descending", True))
                rows = (present + missing)[:step["limit"]]
            if len(rows) > self.max_rows:
                raise PlanError(f"Step {sid} exceeded the {self.max_rows}-row execution limit.")
            results[sid] = rows
        outputs = {ref: results[ref] for ref in plan.get("outputs", [plan["steps"][-1]["id"]])}
        return {"outputs": outputs, "steps": results,
                "evidence": {"plan_step_count": len(plan["steps"]), "row_counts": {k: len(v) for k, v in results.items()},
                             "comparison_groups_missing_periods": comparison_gaps,
                             "datasets": provenance or {name: {"dataset_name": ds.name} for name, ds in self.datasets.items()}}}

    @staticmethod
    def _value(row, value):
        if isinstance(value, str):
            return to_number(row.get(value))
        if isinstance(value, (int, float)):
            return value
        return None
