import csv
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from src.ingestion.loader import to_number

ALLOWED_SQL_OPERATORS = {
    "==": "=",
    "=": "=",
    "!=": "!=",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
    "contains": "LIKE",
    "in": "IN",
    "is_null": "IS NULL",
    "is_not_null": "IS NOT NULL",
}


def _quote_identifier(value: str) -> str:
    """Quote a SQLite identifier, escaping embedded quotes safely."""
    return '"' + value.replace('"', '""') + '"'


@dataclass
class FilterCriterion:
    column: str
    operator: str
    value: Any = None


@dataclass
class CompoundFilter:
    combinator: str = "AND"  # "AND" or "OR"
    criteria: List[Union[FilterCriterion, "CompoundFilter"]] = field(default_factory=list)


class DatabaseExecutor:
    """
    SQLite-backed query execution engine.
    Ingests dataset records into a local SQLite table for query pushdown and bounded memory usage.
    Never accepts or constructs unrestricted raw SQL; strictly parameterized queries.
    """

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def sync_dataset_file(
        self,
        file_path: Union[str, Path],
        columns: List[str],
        column_types: Dict[str, str],
    ) -> None:
        """
        Creates or refreshes the 'records' table from a CSV or JSON file.
        Detects numeric vs text types and populates rows with parameterized batch inserts.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        # Sanitize column names for SQL safety (alphanumeric + underscore only)
        col_defs = []
        for col in columns:
            t = "REAL" if column_types.get(col) == "numeric" else "TEXT"
            col_defs.append(f'{_quote_identifier(col)} {t}')

        schema_sql = f'CREATE TABLE IF NOT EXISTS records ({", ".join(col_defs)});'

        with self._connection() as conn:
            conn.execute("DROP TABLE IF EXISTS records;")
            if columns:
                conn.execute(schema_sql)

        # Batch insert data
        if not path.exists() or path.stat().st_size == 0 or not columns:
            return

        batch_size = 1000
        rows_to_insert: List[List[Any]] = []

        placeholders = ", ".join(["?"] * len(columns))
        quoted_cols = ", ".join(_quote_identifier(col) for col in columns)
        insert_sql = f'INSERT INTO records ({quoted_cols}) VALUES ({placeholders});'

        def flush_batch():
            if rows_to_insert:
                with self._connection() as conn:
                    conn.executemany(insert_sql, rows_to_insert)
                rows_to_insert.clear()

        def append_row(row):
            row_vals = []
            for col in columns:
                v = row.get(col)
                if v is None or str(v).strip() == "":
                    row_vals.append(None)
                elif column_types.get(col) == "numeric":
                    num = to_number(v)
                    row_vals.append(num if num is not None else None)
                else:
                    row_vals.append(str(v).strip())
            rows_to_insert.append(row_vals)
            if len(rows_to_insert) >= batch_size:
                flush_batch()

        if ext == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    append_row(r)

        elif ext == ".json":
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            raw_rows = data if isinstance(data, list) else data.get("data", [data])
            for r in raw_rows:
                if not isinstance(r, dict):
                    continue
                row_vals = []
                for col in columns:
                    v = r.get(col)
                    if v is None or str(v).strip() == "":
                        row_vals.append(None)
                    elif column_types.get(col) == "numeric":
                        num = to_number(v)
                        row_vals.append(num if num is not None else None)
                    else:
                        row_vals.append(str(v).strip())
                rows_to_insert.append(row_vals)
                if len(rows_to_insert) >= batch_size:
                    flush_batch()

        flush_batch()

    def _build_where_clause(
        self,
        filter_obj: Optional[Union[FilterCriterion, CompoundFilter]],
        available_columns: List[str],
    ) -> Tuple[str, List[Any]]:
        """
        Recursively translates a validated FilterCriterion or CompoundFilter
        into a parameterized SQL WHERE clause.
        """
        if not filter_obj:
            return "", []

        params: List[Any] = []
        valid_cols = set(available_columns)

        if isinstance(filter_obj, FilterCriterion):
            if filter_obj.column not in valid_cols:
                raise ValueError(f"Invalid column '{filter_obj.column}' in filter condition.")

            op = filter_obj.operator.lower()
            sql_op = ALLOWED_SQL_OPERATORS.get(op)
            if not sql_op:
                raise ValueError(f"Operator '{filter_obj.operator}' is not permitted.")

            col_name = _quote_identifier(filter_obj.column)

            if sql_op == "IS NULL":
                return f"({col_name} IS NULL OR trim(CAST({col_name} AS TEXT)) = '')", []
            elif sql_op == "IS NOT NULL":
                return f"({col_name} IS NOT NULL AND trim(CAST({col_name} AS TEXT)) != '')", []
            elif sql_op == "LIKE":
                escaped = str(filter_obj.value).replace("!", "!!").replace("%", "!%").replace("_", "!_")
                return f"{col_name} LIKE ? ESCAPE '!'", [f"%{escaped}%"]
            elif sql_op == "IN":
                if not isinstance(filter_obj.value, (list, tuple, set)):
                    raise ValueError("The 'in' filter requires a list of values.")
                values = list(filter_obj.value)
                if not values:
                    return "0", []
                marks = ", ".join("?" for _ in values)
                return f"{col_name} COLLATE NOCASE IN ({marks})", [str(v).strip() for v in values]
            else:
                val = filter_obj.value
                num_val = to_number(val)
                if num_val is not None and not isinstance(val, bool):
                    return f"{col_name} {sql_op} ?", [num_val]
                if sql_op == "=":
                    return f"{col_name} COLLATE NOCASE = ?", [str(val).strip()]
                if sql_op == "!=":
                    return f"({col_name} IS NULL OR {col_name} COLLATE NOCASE != ?)", [str(val).strip()]
                return f"{col_name} {sql_op} ?", [str(val).strip()]

        elif isinstance(filter_obj, CompoundFilter):
            comb = filter_obj.combinator.upper()
            if comb not in {"AND", "OR"}:
                raise ValueError(f"Combinator '{filter_obj.combinator}' must be 'AND' or 'OR'.")

            clauses = []
            for item in filter_obj.criteria:
                sub_clause, sub_params = self._build_where_clause(item, available_columns)
                if sub_clause:
                    clauses.append(sub_clause)
                    params.extend(sub_params)

            if not clauses:
                return "", []

            joined = f" {comb} ".join(f"({c})" for c in clauses)
            return joined, params

        return "", []

    def execute_aggregation(
        self,
        agg_function: str,
        column: str,
        available_columns: List[str],
        filter_obj: Optional[Union[FilterCriterion, CompoundFilter]] = None,
    ) -> Optional[Union[int, float]]:
        """
        Pushes aggregation down to SQLite using parameterized SQL.
        agg_function: 'sum', 'average', 'min', 'max'
        """
        if column not in set(available_columns):
            raise ValueError(f"Column '{column}' does not exist in dataset.")

        func_map = {
            "sum": "SUM",
            "average": "AVG",
            "min": "MIN",
            "max": "MAX",
        }
        sql_func = func_map.get(agg_function.lower())
        if not sql_func:
            raise ValueError(f"Unsupported aggregation function: '{agg_function}'")

        where_clause, params = self._build_where_clause(filter_obj, available_columns)
        where_sql = f" WHERE {where_clause}" if where_clause else ""

        query = f'SELECT {sql_func}({_quote_identifier(column)}) AS result FROM records{where_sql};'

        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()
            if not row or row["result"] is None:
                return None
            res = row["result"]
            if sql_func == "AVG":
                return round(float(res), 4)
            return res

    def execute_count(
        self,
        available_columns: List[str],
        filter_obj: Optional[Union[FilterCriterion, CompoundFilter]] = None,
    ) -> int:
        """Pushes row counting down to SQLite."""
        where_clause, params = self._build_where_clause(filter_obj, available_columns)
        where_sql = f" WHERE {where_clause}" if where_clause else ""

        query = f"SELECT COUNT(*) AS total FROM records{where_sql};"

        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()
            return int(row["total"]) if row else 0

    def execute_category_count(
        self,
        column: str,
        available_columns: List[str],
        filter_obj: Optional[Union[FilterCriterion, CompoundFilter]] = None,
    ) -> Dict[str, int]:
        """Pushes GROUP BY category counting down to SQLite."""
        if column not in set(available_columns):
            raise ValueError(f"Column '{column}' does not exist in dataset.")

        where_clause, params = self._build_where_clause(filter_obj, available_columns)
        where_sql = f" WHERE {where_clause}" if where_clause else ""
        quoted_column = _quote_identifier(column)
        category_expr = f"CASE WHEN {quoted_column} IS NULL OR trim(CAST({quoted_column} AS TEXT)) = '' THEN '(missing)' ELSE CAST({quoted_column} AS TEXT) END"
        query = f"""
            SELECT {category_expr} AS cat, COUNT(*) AS cnt
            FROM records{where_sql}
            GROUP BY {category_expr}
            ORDER BY cnt DESC;
        """
        results: Dict[str, int] = {}
        with self._connection() as conn:
            for row in conn.execute(query, params).fetchall():
                results[str(row["cat"])] = int(row["cnt"])
        return results

    def execute_filter_rows(
        self,
        available_columns: List[str],
        filter_obj: Optional[Union[FilterCriterion, CompoundFilter]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieves matched rows with parameterized filtering."""
        where_clause, params = self._build_where_clause(filter_obj, available_columns)
        where_sql = f" WHERE {where_clause}" if where_clause else ""

        cols_sql = ", ".join(_quote_identifier(c) for c in available_columns) if available_columns else "*"
        query = f"SELECT {cols_sql} FROM records{where_sql} LIMIT ?;"
        params.append(limit)

        results = []
        with self._connection() as conn:
            for row in conn.execute(query, params).fetchall():
                results.append(dict(row))
        return results
