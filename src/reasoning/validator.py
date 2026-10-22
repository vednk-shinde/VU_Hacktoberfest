from dataclasses import dataclass
from typing import Optional
from src.ingestion.loader import Dataset
from src.reasoning.intents import Operation, QueryIntent
from src.retrieval.filters import ALLOWED_OPERATORS
from src.retrieval.db_executor import CompoundFilter, FilterCriterion


def filter_columns(filter_obj):
    if filter_obj is None:
        return []
    if isinstance(filter_obj, FilterCriterion):
        return [filter_obj.column]
    if isinstance(filter_obj, CompoundFilter):
        if filter_obj.combinator.upper() not in {"AND", "OR"}:
            raise ValueError("Filter combinator must be AND or OR.")
        return [c for item in filter_obj.criteria for c in filter_columns(item)]
    raise ValueError("Malformed filter condition.")


@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None


def validate_intent(intent: QueryIntent, dataset: Dataset) -> ValidationResult:
    """
    Validates a parsed query intent against the dataset schema and safety rules.
    Prevents execution on missing columns, non-numeric aggregations, or invalid operators.
    """
    # 1. Check for abstention operations
    if intent.operation in {Operation.UNSUPPORTED, Operation.AMBIGUOUS}:
        return ValidationResult(
            is_valid=False,
            reason=intent.abstain_reason or "Question is unsupported or ambiguous.",
        )

    # 2. Validate primary target column if specified
    if intent.column is not None:
        if intent.column not in dataset.columns:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"Column '{intent.column}' does not exist in dataset. "
                    f"Available columns: {dataset.columns}"
                ),
            )

    # 3. Validate numeric aggregation compatibility
    if intent.operation == Operation.NUMERIC_AGGREGATION and intent.column is not None:
        col_type = dataset.column_types.get(intent.column, "unknown")
        if col_type == "text":
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"Cannot compute {intent.agg_function} on non-numeric column '{intent.column}' "
                    f"(detected type: text)."
                ),
            )

    # 4. Validate filter conditions if present
    filter_obj = intent.compound_filter
    if filter_obj is None and intent.filter_column is not None:
        filter_obj = FilterCriterion(intent.filter_column, intent.filter_operator, intent.filter_value)
    try:
        filter_cols = filter_columns(filter_obj)
    except ValueError as err:
        return ValidationResult(False, str(err))
    for filter_col in filter_cols:
        if filter_col not in dataset.columns:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"Filter column '{filter_col}' does not exist in dataset. "
                    f"Available columns: {dataset.columns}"
                ),
            )
        if filter_obj is not None and isinstance(filter_obj, FilterCriterion) and filter_obj.operator not in ALLOWED_OPERATORS:
            return ValidationResult(False, f"Filter operator '{filter_obj.operator}' is not permitted.")
    def validate_operators(item):
        if isinstance(item, FilterCriterion):
            return item.operator in ALLOWED_OPERATORS
        if isinstance(item, CompoundFilter):
            return item.combinator.upper() in {"AND", "OR"} and all(validate_operators(x) for x in item.criteria)
        return False
    if filter_obj is not None and not validate_operators(filter_obj):
        return ValidationResult(False, "Filter contains a disallowed operator or malformed nested condition.")

    return ValidationResult(is_valid=True, reason=None)
