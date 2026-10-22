import json
import re
from typing import Any, Dict, List, Optional
from src.models.ollama_client import OllamaClient
from src.reasoning.intents import AggFunction, Operation
from src.reasoning.parser import extract_candidate_column
from src.verification.evaluator import QAResult


VALID_OPERATIONS = {
    Operation.ROW_COUNT,
    Operation.COLUMN_NAMES,
    Operation.MISSING_VALUES,
    Operation.DUPLICATE_COUNT,
    Operation.CATEGORY_COUNT,
    Operation.NUMERIC_AGGREGATION,
    Operation.FILTER_COUNT,
    Operation.FILTER_ROWS,
    Operation.UNSUPPORTED,
    Operation.AMBIGUOUS,
}

VALID_AGG_FUNCTIONS = {
    AggFunction.SUM,
    AggFunction.AVERAGE,
    AggFunction.MIN,
    AggFunction.MAX,
}

REQUIRED_WORKFLOW_COLUMNS = {
    "products": {"product_id", "name", "category"},
    "sales": {"product_id", "region", "period", "units", "revenue"},
    "inventory": {"product_id", "on_hand", "supplier_id"},
    "suppliers": {"supplier_id", "name", "lead_time_days", "defect_rate"},
}


def build_system_prompt(columns: List[str], column_types: Dict[str, str], dataset_name: str) -> str:
    cols_desc = ", ".join([f"'{c}' ({column_types.get(c, 'text')})" for c in columns])
    return f"""You are a privacy-first, local enterprise data query planner.
Your job is to translate natural language questions about the dataset '{dataset_name}' into a strictly structured JSON query plan.
Never execute arbitrary code, shell commands, or SQL. Only plan operations from the allowlist.

Available Columns & Types:
{cols_desc}

CRITICAL GROUNDING RULES:
1. ONLY use column names that appear in Available Columns.
2. If the user asks about an entity, metric, or column that does NOT exist in Available Columns (e.g. asking for 'bonus', 'profit', or 'location' when not listed), you MUST set "operation": "unsupported", "column": null, and state in "reason" that the requested column does not exist.
3. NEVER substitute or guess an available column (like substituting 'salary' for 'bonus').

Allowlisted Operations:
- "row_count": Total records in dataset
- "column_names": List of column headers
- "missing_values": Count null/empty cells (set "column" to specific column or null for total)
- "duplicate_count": Count duplicate rows
- "category_count": Value breakdown of a column (requires "column")
- "numeric_aggregation": Numeric math (requires "column" and "agg_function" in ["sum", "average", "min", "max"])
- "filter_count": Count records satisfying filter
- "filter_rows": Return records satisfying filter
- "unsupported": General knowledge, code generation, mutations, or questions about columns not in the schema
- "ambiguous": Underspecified questions where the target column or intent is completely unclear

Output Format (strictly JSON):
{{
  "operation": "row_count" | "column_names" | "missing_values" | "duplicate_count" | "category_count" | "numeric_aggregation" | "filter_count" | "filter_rows" | "unsupported" | "ambiguous",
  "column": "<column_name>" or null,
  "agg_function": "sum" | "average" | "min" | "max" or null,
  "filter": {{
    "combinator": "AND" | "OR",
    "criteria": [
      {{"column": "<col>", "operator": "==" | "!=" | ">" | ">=" | "<" | "<=" | "contains", "value": <val>}}
    ]
  }} or null,
  "reason": "<short explanation of your plan or reason for abstention>"
}}"""


class GemmaQueryPlanner:
    """
    Coordinates Gemma 3 4B through Ollama to interpret natural language queries
    into strictly validated, constrained JSON query plans.
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    def propose_plan(
        self,
        question: str,
        columns: List[str],
        column_types: Dict[str, str],
        dataset_name: str = "dataset",
    ) -> Dict[str, Any]:
        """
        Requests Gemma 3 4B to propose a structured JSON query plan.
        Validates the structure and allowlisted operations before returning.
        """
        system_prompt = build_system_prompt(columns, column_types, dataset_name)
        prompt = f"Question: {question.strip()}"

        result = self.client.generate(
            prompt=prompt,
            system=system_prompt,
            format_type="json",
            temperature=0.0,
        )

        parsed = result.get("parsed")
        if not isinstance(parsed, dict):
            raise ValueError("Model failed to return a valid JSON object.")

        # Validate mandatory plan fields
        op = parsed.get("operation")
        if op not in VALID_OPERATIONS:
            raise ValueError(f"Model proposed unpermitted operation: '{op}'")

        col = parsed.get("column")
        if col is not None and col not in set(columns):
            # Model proposed a column that does not exist in schema
            return {
                "operation": Operation.UNSUPPORTED,
                "column": col,
                "agg_function": parsed.get("agg_function"),
                "filter": None,
                "reason": f"Column '{col}' does not exist in dataset schema. Available: {columns}",
                "metrics": {
                    "eval_count": result.get("eval_count", 0),
                    "total_duration_ns": result.get("total_duration_ns", 0),
                },
            }

        # Guard against hallucinated column substitution
        # (e.g. user asked for 'bonus', but model substituted 'salary')
        if col is not None:
            col_norm = col.lower().replace("_", " ")
            q_lower = question.lower()
            if col.lower() not in q_lower and col_norm not in q_lower:
                candidate = extract_candidate_column(question, columns)
                if candidate and candidate.lower() != col.lower():
                    return {
                        "operation": Operation.UNSUPPORTED,
                        "column": candidate,
                        "agg_function": parsed.get("agg_function"),
                        "filter": None,
                        "reason": f"Column '{candidate}' does not exist in dataset. Ungrounded substitution with '{col}' rejected.",
                        "metrics": {
                            "eval_count": result.get("eval_count", 0),
                            "total_duration_ns": result.get("total_duration_ns", 0),
                        },
                    }

        agg_func = parsed.get("agg_function")
        if agg_func is not None and agg_func not in VALID_AGG_FUNCTIONS:
            raise ValueError(f"Model proposed unpermitted aggregation function: '{agg_func}'")

        parsed["metrics"] = {
            "eval_count": result.get("eval_count", 0),
            "total_duration_ns": result.get("total_duration_ns", 0),
        }
        return parsed

    def propose_analytical_plan(self, question: str, schemas: Dict[str, Dict[str, str]],
                                relationships: List[Dict[str, str]]) -> Dict[str, Any]:
        """Ask Gemma for a closed, schema-grounded workflow proposal; invalid output fails closed."""
        allowed_ops = ["scan", "filter", "aggregate", "compare_periods", "derive", "join"]
        system = f"""Return only one JSON object with exactly these keys and value types:
{{"workflow":"inventory_sales_regional_comparison","datasets":{{"products":"products","sales":"sales","inventory":"inventory","suppliers":"suppliers"}},"columns":{{"products":["column_name"],"sales":["column_name"],"inventory":["column_name"],"suppliers":["column_name"]}},"relationships":[{{"left_dataset":"...","left_key":"...","right_dataset":"...","right_key":"..."}}],"operations":["scan","filter","aggregate","compare_periods","derive","join"],"previous_period":"previous","current_period":"current"}}
The `operations` value MUST be a flat JSON array of operation-name strings exactly from this allowlist: {json.dumps(allowed_ops)}. Do not put step descriptions, argument lists, arrays, objects, or tuples inside `operations`. Do not add aliases to operation names.
The `columns` value MUST map each dataset to a flat array of column-name strings (no nested arrays), and must include all columns required by the workflow: {json.dumps({k: sorted(v) for k, v in REQUIRED_WORKFLOW_COLUMNS.items()})}. Use only dataset and column names provided. The only supported workflow is inventory_sales_regional_comparison. Include every supplied approved relationship exactly once. This is a proposal, not an answer: do not add results, steps, SQL, explanations, or other fields. Period labels must be exactly previous and current. If unsupported, return {{"workflow":"unsupported"}}."""
        prompt = json.dumps({"question": question, "actual_dataset_schemas": schemas,
                             "approved_relationships": relationships,
                             "period_labels": ["previous", "current"]}, ensure_ascii=False)
        last_error = None
        failed_proposals = []
        for attempt in range(2):  # initial generation plus one bounded repair
            response = None
            parsed = None
            repair = ""
            if last_error:
                repair = (
                    "\nREPAIR REQUIRED. The exact validation error was: " + last_error +
                    ". Allowed operations, spelled exactly: " + json.dumps(allowed_ops) +
                    ". Use no other operation names. Preserve these exact dataset references: " +
                    json.dumps({name: name for name in sorted(schemas)}) +
                    ". Preserve only these approved relationships: " + json.dumps(relationships) +
                    ". `operations` must contain every allowed operation exactly once. `columns` must contain each dataset's required columns listed in the system instruction, as flat string arrays with no nested lists. Period labels must be exactly previous and current. Use only the actual columns in the schemas in the original request. Return the complete corrected JSON proposal only."
                )
            try:
                response = self.client.generate(prompt=prompt + repair, system=system,
                                                format_type="json", temperature=0.0)
                parsed = response.get("parsed")
                self._validate_workflow_proposal(parsed, schemas, relationships)
                parsed["metrics"] = {"eval_count": response.get("eval_count", 0),
                                     "total_duration_ns": response.get("total_duration_ns", 0),
                                     "repair_attempts": attempt, "raw_response": response.get("raw")}
                return parsed
            except (ValueError, TypeError) as exc:
                last_error = str(exc)
                failed_proposals.append({"raw": response.get("raw") if response else None,
                                         "parsed": parsed,
                                         "validation_error": last_error})
        raise ValueError(f"Gemma workflow proposal invalid after one repair: {last_error}; "
                         f"raw proposals: {json.dumps(failed_proposals, ensure_ascii=False, default=str)}")

    @staticmethod
    def _validate_workflow_proposal(plan, schemas, relationships):
        if not isinstance(plan, dict):
            raise ValueError("Model did not return a JSON workflow object.")
        required = {"workflow", "datasets", "columns", "relationships", "operations", "previous_period", "current_period"}
        if set(plan) != required:
            raise ValueError(f"Workflow fields must be exactly {sorted(required)}.")
        if plan["workflow"] != "inventory_sales_regional_comparison":
            raise ValueError("Unsupported workflow.")
        expected = {"products", "sales", "inventory", "suppliers"}
        if plan["datasets"] != {n: n for n in expected}:
            raise ValueError("Missing or invalid dataset reference.")
        if not isinstance(plan["columns"], dict) or set(plan["columns"]) != expected:
            raise ValueError("Column selections must reference all four datasets.")
        for name, cols in plan["columns"].items():
            if not isinstance(cols, list) or not cols or any(not isinstance(c, str) or c not in schemas[name] for c in cols):
                raise ValueError(f"Invalid column selection for dataset {name}.")
            missing = REQUIRED_WORKFLOW_COLUMNS.get(name, set()) - set(cols)
            if missing:
                raise ValueError(f"Column selection for {name} omits compiler-required columns: {sorted(missing)}.")
        allowed_ops = {"scan", "filter", "aggregate", "compare_periods", "derive", "join"}
        if not isinstance(plan["operations"], list) or not plan["operations"]:
            raise ValueError(f"Operations must be a non-empty list; allowed operations: {sorted(allowed_ops)}.")
        non_string_ops = [op for op in plan["operations"] if not isinstance(op, str)]
        if non_string_ops:
            raise ValueError(f"Operation entries must be strings, not step descriptions or nested values: {non_string_ops!r}; allowed operations: {sorted(allowed_ops)}.")
        unsupported = [op for op in plan["operations"] if op not in allowed_ops]
        if unsupported:
            raise ValueError(f"Unsupported operation(s) {unsupported!r}; allowed operations: {sorted(allowed_ops)}.")
        missing_ops = allowed_ops - set(plan["operations"])
        duplicates = sorted({op for op in plan["operations"] if plan["operations"].count(op) > 1})
        if missing_ops or duplicates:
            raise ValueError(f"Workflow requires each supported operation exactly once; missing: {sorted(missing_ops)}, duplicates: {duplicates}.")
        keys = ("left_dataset", "left_key", "right_dataset", "right_key")
        approved = {tuple(r.get(k) for k in keys) for r in relationships if isinstance(r, dict)}
        selected = plan["relationships"]
        if not isinstance(selected, list) or not selected or any(
            not isinstance(r, dict) or any(not isinstance(r.get(k), str) for k in keys) or
            tuple(r.get(k) for k in keys) not in approved for r in selected
        ):
            raise ValueError("Missing or unapproved dataset relationship.")
        selected_keys = [tuple(r.get(k) for k in keys) for r in selected]
        if set(selected_keys) != approved or len(selected_keys) != len(set(selected_keys)):
            raise ValueError("Workflow proposal is missing an approved relationship required by this workflow.")
        if (plan["previous_period"], plan["current_period"]) != ("previous", "current"):
            raise ValueError("Period references must be exactly previous and current for this workflow.")

    def explain_result(
        self,
        question: str,
        result: QAResult,
        dataset_name: str,
        dataset_version: Optional[int] = None,
    ) -> str:
        """
        Requests Gemma 3 4B to generate a grounded natural language summary
        using ONLY verified calculation results and evidence.
        """
        if result.status == "abstained":
            return f"I cannot answer this question because: {result.reason}"

        system_prompt = (
            "You are an enterprise AI assistant explaining verified calculations.\n"
            "Explain the answer clearly and concisely.\n"
            "RULES:\n"
            "1. You MUST ONLY use the provided verified answer, calculation method, and evidence.\n"
            "2. Never hallucinate, extrapolate, or invent figures not in the evidence.\n"
            "3. Mention the dataset name and dataset version used."
        )

        prompt = (
            f"Question: {question}\n"
            f"Dataset: {dataset_name} (Version: {dataset_version or 1})\n"
            f"Verified Answer: {result.answer}\n"
            f"Calculation Method: {result.calculation_method}\n"
            f"Evidence: {json.dumps(result.evidence, default=str)}\n\n"
            "Please provide a factual, 1-2 sentence response explaining this result to the user."
        )

        res = self.client.generate(
            prompt=prompt,
            system=system_prompt,
            format_type=None,
            temperature=0.1,
        )
        return res.get("raw", "").strip()
