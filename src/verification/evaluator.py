import json
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src.ingestion.loader import Dataset, to_number
from src.reasoning.intents import AggFunction, Operation, QueryIntent
from src.retrieval.filters import apply_filter, apply_filter_object, group_and_count


@dataclass
class QAResult:
    question: str
    status: str  # "answered" or "abstained"
    answer: Any
    operation: Optional[str] = None
    calculation_method: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status,
            "answer": self.answer,
            "operation": self.operation,
            "calculation_method": self.calculation_method,
            "evidence": self.evidence,
            "reason": self.reason,
            "explanation": self.explanation,
        }


def evaluate_intent(intent: QueryIntent, dataset: Dataset) -> QAResult:
    """
    Deterministically computes the answer and calculation evidence for a validated QueryIntent.
    Strictly standard library; zero arbitrary code or SQL execution.
    """
    op = intent.operation

    filtered_rows = dataset.rows
    if intent.compound_filter is not None:
        filtered_rows = apply_filter_object(dataset.rows, intent.compound_filter)
    elif intent.filter_column and intent.filter_operator:
        filtered_rows = apply_filter(dataset.rows, intent.filter_column, intent.filter_operator, intent.filter_value)

    # 1. Row count
    if op == Operation.ROW_COUNT:
        count = len(filtered_rows)
        return QAResult(
            question=intent.raw_question,
            status="answered",
            answer=count,
            operation=op,
            calculation_method=f"Counted total records loaded from {dataset.name}.",
            evidence={"row_count": count, "file": dataset.name},
        )

    # 2. Column names
    if op == Operation.COLUMN_NAMES:
        cols = list(dataset.columns)
        return QAResult(
            question=intent.raw_question,
            status="answered",
            answer=cols,
            operation=op,
            calculation_method="Extracted column header list from dataset schema.",
            evidence={"columns": cols, "column_count": len(cols)},
        )

    # 3. Missing values
    if op == Operation.MISSING_VALUES:
        if intent.column is not None:
            # Column-specific missing values
            col = intent.column
            missing = sum(
                r.get(col) is None or str(r.get(col)).strip() == ""
                for r in filtered_rows
            )
            return QAResult(
                question=intent.raw_question,
                status="answered",
                answer=missing,
                operation=op,
                calculation_method=f"Counted empty or null cells for column '{col}' across all {len(dataset.rows)} rows.",
                evidence={"column": col, "missing_count": missing, "total_rows": len(dataset.rows)},
            )
        else:
            # Entire dataset missing values
            total_missing = sum(
                sum(r.get(col) is None or str(r.get(col)).strip() == "" for col in dataset.columns)
                for r in filtered_rows
            )
            return QAResult(
                question=intent.raw_question,
                status="answered",
                answer=total_missing,
                operation=op,
                calculation_method="Summed all null or empty string cells across all columns and rows in dataset.",
                evidence={"total_missing": total_missing, "columns": dataset.columns, "total_rows": len(filtered_rows)},
            )

    # 4. Duplicate counts
    if op == Operation.DUPLICATE_COUNT:
        serialized = [
            json.dumps(row, sort_keys=True, default=str)
            for row in filtered_rows
        ]
        dup_count = len(serialized) - len(set(serialized))
        return QAResult(
            question=intent.raw_question,
            status="answered",
            answer=dup_count,
            operation=op,
            calculation_method="Serialized all rows to JSON and calculated (total_rows - unique_rows).",
            evidence={
                "duplicate_rows": dup_count,
                "total_rows": len(dataset.rows),
                "unique_rows": len(dataset.rows) - dup_count,
            },
        )

    # 5. Category count
    if op == Operation.CATEGORY_COUNT:
        counts = group_and_count(filtered_rows, intent.column)
        return QAResult(
            question=intent.raw_question,
            status="answered",
            answer=counts,
            operation=op,
            calculation_method=f"Grouped records by '{intent.column}' and tallied occurrences of each distinct value.",
            evidence={"column": intent.column, "distinct_categories": len(counts), "distribution": counts},
        )

    # 6. Numeric aggregation
    if op == Operation.NUMERIC_AGGREGATION:
        target_rows = filtered_rows
        filter_applied = intent.compound_filter is not None or bool(intent.filter_column and intent.filter_operator)

        # Extract numeric values
        numbers: List[float] = []
        for r in target_rows:
            num = to_number(r.get(intent.column))
            if num is not None:
                numbers.append(float(num) if isinstance(num, float) else num)

        if not numbers:
            return QAResult(
                question=intent.raw_question,
                status="abstained",
                answer=None,
                operation=op,
                reason=f"No numeric values found in column '{intent.column}' for aggregation.",
            )

        func = intent.agg_function
        if func == AggFunction.SUM:
            val = round(sum(numbers), 4) if any(isinstance(x, float) for x in numbers) else sum(numbers)
        elif func == AggFunction.AVERAGE:
            val = round(sum(numbers) / len(numbers), 4)
        elif func == AggFunction.MIN:
            val = min(numbers)
        elif func == AggFunction.MAX:
            val = max(numbers)
        else:
            return QAResult(
                question=intent.raw_question,
                status="abstained",
                answer=None,
                operation=op,
                reason=f"Unsupported aggregation function '{func}'.",
            )

        calc_str = f"Computed {func} across {len(numbers)} numeric values from column '{intent.column}'."
        if filter_applied:
            calc_str += f" Filtered where {intent.filter_column} {intent.filter_operator} '{intent.filter_value}'."

        evidence = {
            "column": intent.column,
            "agg_function": func,
            "count": len(numbers),
            "sample_values": numbers[:5],
            "filter_applied": filter_applied,
        }
        if filter_applied:
            evidence["filter_column"] = intent.filter_column
            evidence["filter_operator"] = intent.filter_operator
            evidence["filter_value"] = intent.filter_value

        return QAResult(
            question=intent.raw_question,
            status="answered",
            answer=val,
            operation=op,
            calculation_method=calc_str,
            evidence=evidence,
        )

    # 7. Filter count
    if op == Operation.FILTER_COUNT:
        matched = filtered_rows
        return QAResult(
            question=intent.raw_question,
            status="answered",
            answer=len(matched),
            operation=op,
            calculation_method=(
                f"Filtered rows where '{intent.filter_column}' {intent.filter_operator} "
                f"'{intent.filter_value}' and counted matching records."
            ),
            evidence={
                "filter_column": intent.filter_column,
                "operator": intent.filter_operator,
                "filter_value": intent.filter_value,
                "matched_count": len(matched),
                "total_rows": len(dataset.rows),
                "sample_matches": matched[:3],
            },
        )

    # 8. Filter rows
    if op == Operation.FILTER_ROWS:
        matched = filtered_rows
        return QAResult(
            question=intent.raw_question,
            status="answered",
            answer=matched,
            operation=op,
            calculation_method=(
                f"Retrieved rows where '{intent.filter_column}' {intent.filter_operator} "
                f"'{intent.filter_value}'."
            ),
            evidence={
                "filter_column": intent.filter_column,
                "operator": intent.filter_operator,
                "filter_value": intent.filter_value,
                "matched_count": len(matched),
                "total_rows": len(dataset.rows),
            },
        )

    # Fallback abstention
    return QAResult(
        question=intent.raw_question,
        status="abstained",
        answer=None,
        operation=op,
        reason="Operation could not be evaluated.",
    )
