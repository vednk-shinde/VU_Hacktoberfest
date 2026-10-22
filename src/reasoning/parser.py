import re
from typing import Any, Dict, List, Optional, Tuple
from src.reasoning.intents import AggFunction, Operation, QueryIntent


UNSAFE_KEYWORDS = {
    "drop", "delete", "insert", "update", "truncate", "alter", "create",
    "exec", "eval", "import", "os.", "sys.", "subprocess", "__import__",
}


STOP_WORDS = {
    "the", "a", "an", "this", "each", "every", "all", "dataset", "data",
    "table", "row", "rows", "record", "records", "file", "csv", "json",
    "missing", "null", "empty", "na", "values", "value", "count", "counts",
    "number", "numbers", "total", "totals", "average", "mean", "min", "max",
}


def normalize_token(token: str) -> str:
    """Normalize a word token by lowercasing and stripping common punctuation."""
    return re.sub(r"[^\w]", "", token.lower())


def match_column_in_text(text: str, available_columns: List[str]) -> Optional[str]:
    """
    Search for a column name in text, prioritizing exact matches and longer names.
    Supports pluralization and snake_case variations.
    """
    cleaned = text.lower()
    # Sort columns by length descending to match 'years_experience' before 'experience'
    sorted_cols = sorted(available_columns, key=len, reverse=True)

    for col in sorted_cols:
        col_lower = col.lower()
        col_norm = col_lower.replace("_", " ")

        # Match exact snake_case or space-separated form
        pattern = rf"\b({re.escape(col_lower)}|{re.escape(col_norm)})\b"
        if re.search(pattern, cleaned):
            return col

        # Match simple plural forms (e.g. salaries -> salary, employees -> employee)
        if col_lower.endswith("y"):
            plural = col_lower[:-1] + "ies"
        else:
            plural = col_lower + "s"

        if re.search(rf"\b{re.escape(plural)}\b", cleaned):
            return col

    return None


def extract_candidate_column(text: str, available_columns: List[str]) -> Optional[str]:
    """
    Find column from available columns, or extract the unknown subject word
    following phrases like 'average', 'max', 'min', 'sum of', 'in', 'for'.
    """
    matched = match_column_in_text(text, available_columns)
    if matched:
        return matched

    # Look for candidate unknown column name after keywords
    patterns = [
        r"\b(?:average|avg|mean|max|maximum|min|minimum|sum|total|count of)\s+(?:of\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
        r"\b(?:count by|breakdown by|group by|by)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r"\b(?:in|for|of|column|field)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:column|field)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            candidate = m.group(1).strip()
            if candidate not in STOP_WORDS:
                return candidate

    return None


def extract_filter_condition(
    text: str,
    available_columns: List[str],
) -> Tuple[Optional[str], Optional[str], Optional[Any]]:
    """
    Extract a simple filter condition from questions such as:
    - 'where department is Engineering'
    - 'where price > 1000'
    - 'in Engineering'
    - 'with salary greater than 60000'
    """
    cleaned = text.lower()

    # Pattern 1: explicit where / with clause with operators
    op_patterns = [
        (r"\b(?:where|with)\s+([a-zA-Z0-9_]+)\s*(==|=|is)\s*['\"]?([a-zA-Z0-9_ ]+?)['\"]?(?:\s|$|\?)", "=="),
        (r"\b(?:where|with)\s+([a-zA-Z0-9_]+)\s*(!=|is not)\s*['\"]?([a-zA-Z0-9_ ]+?)['\"]?(?:\s|$|\?)", "!="),
        (r"\b(?:where|with)\s+([a-zA-Z0-9_]+)\s*(?:>=|greater than or equal to)\s*([0-9.]+)", ">="),
        (r"\b(?:where|with)\s+([a-zA-Z0-9_]+)\s*(?:<=|less than or equal to)\s*([0-9.]+)", "<="),
        (r"\b(?:where|with)\s+([a-zA-Z0-9_]+)\s*(?:>|greater than|more than|above)\s*([0-9.]+)", ">"),
        (r"\b(?:where|with)\s+([a-zA-Z0-9_]+)\s*(?:<|less than|fewer than|below)\s*([0-9.]+)", "<"),
        (r"\b(?:where|with)\s+([a-zA-Z0-9_]+)\s*(?:contains|has)\s*['\"]?([a-zA-Z0-9_]+)['\"]?", "contains"),
    ]

    for pat, op in op_patterns:
        m = re.search(pat, cleaned)
        if m:
            col_cand = m.group(1).strip()
            val_cand = m.group(2 if op in {">", "<", ">=", "<="} else 2).strip()
            # If the regex had 3 groups (e.g. for ==, !=), value is group(2) or group(3)
            if len(m.groups()) >= 3 and op in {"==", "!="}:
                val_cand = m.group(3).strip()

            actual_col = match_column_in_text(col_cand, available_columns) or col_cand
            return actual_col, op, val_cand

    # Pattern 2: 'in <value>' where value matches known categorical patterns like 'in Engineering'
    # E.g. 'how many employees in Engineering'
    in_match = re.search(r"\b(?:in|from)\s+([A-Z][a-zA-Z0-9_]*)\b", text)
    if in_match:
        val = in_match.group(1)
        # Search which column could match department / category / etc.
        cat_col = None
        for col in available_columns:
            if col.lower() in {"department", "category", "dept", "group", "status", "role"}:
                cat_col = col
                break
        if cat_col:
            return cat_col, "==", val

    return None, None, None


def parse_question(question: str, available_columns: List[str]) -> QueryIntent:
    """
    Deterministically parses a natural language question into a QueryIntent.
    Strictly standard library regex and token matching.
    """
    raw = question.strip()
    if not raw:
        return QueryIntent(
            operation=Operation.AMBIGUOUS,
            raw_question=raw,
            abstain_reason="Question is empty or whitespace only.",
        )

    lower = raw.lower()

    # Security check for prohibited modification/execution tokens
    tokens = [normalize_token(t) for t in lower.split()]
    for unsafe in UNSAFE_KEYWORDS:
        if unsafe in tokens or unsafe in lower:
            return QueryIntent(
                operation=Operation.UNSUPPORTED,
                raw_question=raw,
                abstain_reason=f"Query contains disallowed or unsafe keyword: '{unsafe}'.",
            )

    # 1. Column names
    if re.search(r"\b(what are the columns|list (all )?columns|show (the )?columns|column names|what columns (exist|are in)|list fields|field names)\b", lower):
        return QueryIntent(operation=Operation.COLUMN_NAMES, raw_question=raw)

    # 2. Duplicate counts
    if re.search(r"\b(how many duplicate(s| rows)?|duplicate count|number of duplicate(s| rows)?|count duplicates)\b", lower):
        return QueryIntent(operation=Operation.DUPLICATE_COUNT, raw_question=raw)

    # 3. Row count (placed before general numeric aggregation to catch 'total records', 'total rows')
    if re.search(r"\b(how many rows|row count|number of rows|count rows|total rows|how many records|number of records|total records|how many entries|how many total items)\b", lower):
        return QueryIntent(operation=Operation.ROW_COUNT, raw_question=raw)

    if re.search(r"\b(total|number of|count of)\s+(rows|records|entries)\b", lower):
        return QueryIntent(operation=Operation.ROW_COUNT, raw_question=raw)

    # 4. Missing values
    if re.search(r"\b(missing|null|empty|na)\b", lower):
        # Check if dataset-wide missing values
        if re.search(r"\b(total|how many|count of|number of|all)?\s*(missing|null|empty)\s*(values|entries)?\s*(in dataset|in the dataset)?\b", lower) and not re.search(r"\b(?:in|for|of|column)\s+([a-zA-Z0-9_]+)\b", lower):
            return QueryIntent(
                operation=Operation.MISSING_VALUES,
                column=None,
                raw_question=raw,
            )
        col = extract_candidate_column(lower, available_columns)
        if col:
            return QueryIntent(
                operation=Operation.MISSING_VALUES,
                column=col,
                raw_question=raw,
            )
        return QueryIntent(
            operation=Operation.MISSING_VALUES,
            column=None,
            raw_question=raw,
        )

    # 5. Category breakdown / Group count
    if re.search(r"\b(count by|breakdown by|per|distribution of|group by)\b", lower):
        col = extract_candidate_column(lower, available_columns)
        if col:
            return QueryIntent(
                operation=Operation.CATEGORY_COUNT,
                column=col,
                raw_question=raw,
            )
        return QueryIntent(
            operation=Operation.AMBIGUOUS,
            raw_question=raw,
            abstain_reason="Category count requested but target column was not specified.",
        )

    # 6. Filter count or Filter rows (with explicit where/with conditions)
    f_col, f_op, f_val = extract_filter_condition(raw, available_columns)
    if f_col and f_op and f_val is not None and not any(k in lower for k in ["average", "avg", "mean", "max", "min", "sum"]):
        if re.search(r"\b(show|list|filter|get|find)\b", lower):
            return QueryIntent(
                operation=Operation.FILTER_ROWS,
                filter_column=f_col,
                filter_operator=f_op,
                filter_value=f_val,
                raw_question=raw,
            )
        else:
            return QueryIntent(
                operation=Operation.FILTER_COUNT,
                filter_column=f_col,
                filter_operator=f_op,
                filter_value=f_val,
                raw_question=raw,
            )

    # 7. Numeric aggregations (average, min, max, sum)
    agg_mapping = [
        (r"\b(average|avg|mean)\b", AggFunction.AVERAGE),
        (r"\b(maximum|max|highest|greatest|largest)\b", AggFunction.MAX),
        (r"\b(minimum|min|lowest|smallest|least)\b", AggFunction.MIN),
        (r"\b(sum of|sum|total)\b", AggFunction.SUM),
    ]

    for pat, agg_func in agg_mapping:
        if re.search(pat, lower):
            col = extract_candidate_column(lower, available_columns)
            # Check for optional filter condition (e.g. 'average salary in Engineering')
            f_col, f_op, f_val = extract_filter_condition(raw, available_columns)

            if col:
                return QueryIntent(
                    operation=Operation.NUMERIC_AGGREGATION,
                    column=col,
                    agg_function=agg_func,
                    filter_column=f_col,
                    filter_operator=f_op,
                    filter_value=f_val,
                    raw_question=raw,
                )
            else:
                return QueryIntent(
                    operation=Operation.AMBIGUOUS,
                    raw_question=raw,
                    abstain_reason=f"Aggregation '{agg_func}' requested but target column was not specified.",
                )

    # 7. Dataset row count
    if re.search(r"\b(how many rows|row count|number of rows|count rows|total rows|how many records|number of records|total records|how many entries|how many total items)\b", lower):
        return QueryIntent(operation=Operation.ROW_COUNT, raw_question=raw)

    # General row count on primary subject (e.g. 'how many employees are there?', 'how many products are there?')
    if re.search(r"\bhow many\s+([a-zA-Z0-9_]+)\s+(?:are there|in the dataset|total)\b", lower):
        return QueryIntent(operation=Operation.ROW_COUNT, raw_question=raw)

    # Check for ambiguous questions
    if len(lower.split()) <= 2 and not any(k in lower for k in ["row", "column", "count"]):
        return QueryIntent(
            operation=Operation.AMBIGUOUS,
            raw_question=raw,
            abstain_reason="Question is too ambiguous or underspecified to identify target operation.",
        )

    # Unsupported question
    return QueryIntent(
        operation=Operation.UNSUPPORTED,
        raw_question=raw,
        abstain_reason="Question intent cannot be mapped to a supported analytical operation.",
    )
