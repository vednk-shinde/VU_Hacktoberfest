from dataclasses import dataclass
from typing import Any, Optional


class Operation:
    ROW_COUNT = "row_count"
    COLUMN_NAMES = "column_names"
    MISSING_VALUES = "missing_values"
    DUPLICATE_COUNT = "duplicate_count"
    CATEGORY_COUNT = "category_count"
    NUMERIC_AGGREGATION = "numeric_aggregation"
    FILTER_COUNT = "filter_count"
    FILTER_ROWS = "filter_rows"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


class AggFunction:
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"


@dataclass
class QueryIntent:
    operation: str
    column: Optional[str] = None
    agg_function: Optional[str] = None
    filter_column: Optional[str] = None
    filter_operator: Optional[str] = None
    filter_value: Optional[Any] = None
    raw_question: str = ""
    abstain_reason: Optional[str] = None
    compound_filter: Optional[Any] = None
