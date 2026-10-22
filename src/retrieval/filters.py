from typing import Any, Dict, List
from collections import Counter
from src.ingestion.loader import to_number
from src.retrieval.db_executor import CompoundFilter, FilterCriterion

ALLOWED_OPERATORS = {"==", "!=", ">", ">=", "<", "<=", "contains", "in", "is_null", "is_not_null"}


def compare_values(actual: Any, target: Any, operator: str) -> bool:
    """Deterministically compare two values using a safe, whitelisted operator."""
    if operator not in ALLOWED_OPERATORS:
        raise ValueError(f"Operator '{operator}' is not permitted.")

    # Null / empty value handling
    if operator == "is_null":
        return actual is None or str(actual).strip() == ""
    if operator == "is_not_null":
        return actual is not None and str(actual).strip() != ""
    if actual is None or str(actual).strip() == "":
        return operator == "!=" and target is not None

    act_num = to_number(actual)
    tgt_num = to_number(target)

    # Numeric comparison when both sides are convertible to numbers
    if act_num is not None and tgt_num is not None:
        if operator == "==":
            return act_num == tgt_num
        elif operator == "!=":
            return act_num != tgt_num
        elif operator == ">":
            return act_num > tgt_num
        elif operator == ">=":
            return act_num >= tgt_num
        elif operator == "<":
            return act_num < tgt_num
        elif operator == "<=":
            return act_num <= tgt_num

    # String comparison (case-insensitive, normalized)
    act_str = str(actual).strip().lower()
    tgt_str = str(target).strip().lower()

    if operator == "==":
        return act_str == tgt_str
    elif operator == "!=":
        return act_str != tgt_str
    elif operator == "contains":
        return tgt_str in act_str
    elif operator == "in":
        if isinstance(target, (list, tuple, set)):
            target_set = {str(item).strip().lower() for item in target}
            return act_str in target_set
        return act_str in tgt_str

    # Unsupported comparison between non-numbers
    return False


def apply_filter(
    rows: List[Dict[str, Any]],
    column: str,
    operator: str,
    target_value: Any,
) -> List[Dict[str, Any]]:
    """
    Filter rows where row[column] satisfies operator target_value.
    Deterministic, memory-safe, and pure Python stdlib.
    """
    matched: List[Dict[str, Any]] = []
    for row in rows:
        val = row.get(column)
        if compare_values(val, target_value, operator):
            matched.append(row)
    return matched


def apply_filter_object(rows: List[Dict[str, Any]], filter_obj: Any) -> List[Dict[str, Any]]:
    """Apply validated simple or nested AND/OR filters deterministically."""
    if filter_obj is None:
        return list(rows)
    def matches(row, node):
        if isinstance(node, FilterCriterion):
            return compare_values(row.get(node.column), node.value, node.operator)
        if isinstance(node, CompoundFilter):
            values = [matches(row, child) for child in node.criteria]
            return all(values) if node.combinator.upper() == "AND" else any(values)
        raise ValueError("Malformed filter condition.")
    return [row for row in rows if matches(row, filter_obj)]


def group_and_count(
    rows: List[Dict[str, Any]],
    column: str,
) -> Dict[str, int]:
    """
    Count occurrences of each unique value in a given column.
    Empty or null values are labeled as '(missing)'.
    """
    counts = Counter()
    for row in rows:
        val = row.get(column)
        if val is None or str(val).strip() == "":
            counts["(missing)"] += 1
        else:
            counts[str(val).strip()] += 1
    return dict(counts)


def extract_column(
    rows: List[Dict[str, Any]],
    column: str,
) -> List[Any]:
    """Safely extract all values for a column across rows."""
    return [row.get(column) for row in rows]
