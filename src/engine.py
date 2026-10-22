from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.ingestion.loader import Dataset, load_dataset
from src.reasoning.intents import AggFunction, Operation, QueryIntent
from src.reasoning.parser import parse_question
from src.reasoning.validator import filter_columns, validate_intent
from src.retrieval.db_executor import CompoundFilter, DatabaseExecutor, FilterCriterion
from src.security.permissions import UserContext, can_access_columns, can_access_dataset
from src.verification.evaluator import QAResult, evaluate_intent
from src.verification.evidence import deterministic_explanation, not_executed_record, single_execution_record


def _dict_to_filter(f_dict: Optional[Dict[str, Any]]) -> Optional[Union[FilterCriterion, CompoundFilter]]:
    """Converts a dictionary representation of a filter into a FilterCriterion or CompoundFilter."""
    if f_dict is None:
        return None
    if not isinstance(f_dict, dict):
        raise ValueError("Filter must be an object or null.")

    # Check if compound
    if "criteria" in f_dict and "combinator" in f_dict:
        sub_criteria = []
        if not isinstance(f_dict["criteria"], list) or not f_dict["criteria"]:
            raise ValueError("Compound filter criteria must be a non-empty list.")
        for c in f_dict["criteria"]:
            sub_criteria.append(_dict_to_filter(c))
        return CompoundFilter(combinator=f_dict["combinator"], criteria=sub_criteria)

    # Simple criterion
    if "column" in f_dict and "operator" in f_dict:
        return FilterCriterion(
            column=f_dict["column"],
            operator=f_dict["operator"],
            value=f_dict.get("value"),
        )

    raise ValueError("Malformed filter condition.")


class QAEngine:
    """
    Deterministic Question-Answering Engine using Python's standard library.
    Evaluates dataset queries locally without external APIs, ML models, or arbitrary code execution.
    Can query standalone datasets or versioned datasets tracked by the DatasetRegistryService.
    Supports local Gemma 3 4B query planning and explanation with permission guardrails.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path, Dataset, None] = None,
        registry_service: Optional[Any] = None,
        gemma_planner: Optional[Any] = None,
        enable_db_pushdown: bool = False,
        approved_file_roots: Optional[List[Union[str, Path]]] = None,
    ):
        self._dataset: Union[Dataset, None] = None
        self._active_dataset_id: Optional[str] = None
        self._active_dataset_version: Optional[int] = None
        self.registry = registry_service
        self.gemma_planner = gemma_planner
        self.enable_db_pushdown = enable_db_pushdown
        self.approved_file_roots = [Path(p).resolve() for p in (approved_file_roots or [])]
        self._db_executor: Optional[DatabaseExecutor] = None

        if dataset_path is not None:
            self.load(dataset_path)

    def load(
        self,
        dataset: Union[str, Path, Dataset],
        version: Optional[int] = None,
        user_context: Optional[UserContext] = None,
    ) -> Dataset:
        """
        Load and cache a dataset for subsequent queries.
        Supports direct file paths, in-memory Dataset objects, or registered dataset IDs.
        """
        # A failed selection must not leave a prior dataset queryable by accident.
        self._dataset = None
        self._active_dataset_id = None
        self._active_dataset_version = None
        self._db_executor = None

        if isinstance(dataset, Dataset):
            self._validate_direct_file_path(dataset.file_path)
            self._dataset = dataset
            return self._dataset

        # Direct paths are allowed only under explicitly configured roots.
        str_id = str(dataset).strip()
        candidate_path = Path(str_id)
        is_path = candidate_path.exists()
        if is_path:
            self._validate_direct_file_path(candidate_path)
            if self.registry is not None:
                managed_root = Path(self.registry.storage.datasets_dir).resolve()
                if self._is_within(candidate_path.resolve(), managed_root):
                    raise PermissionError("Use the registered dataset ID to access managed dataset files.")

        # Registry errors other than a genuine not-found condition must propagate.
        if self.registry is not None:
            ver_rec = None
            if not is_path:
                if version is not None:
                    ver_rec = self.registry.store.get_version(str_id, version)
                    if ver_rec is None:
                        raise KeyError(f"Dataset '{str_id}' version {version} was not found.")
                else:
                    ver_rec = self.registry.get_active_dataset(str_id)
            if ver_rec is not None:
                if not can_access_dataset(user_context, ver_rec.dataset_id):
                    raise PermissionError(f"Access denied to dataset '{ver_rec.dataset_id}'.")
                storage_path = Path(ver_rec.storage_path).resolve()
                storage_root = Path(self.registry.storage.datasets_dir).resolve()
                if not self._is_within(storage_path, storage_root):
                    raise PermissionError("Registered dataset path is outside approved registry storage.")
                if not storage_path.is_file():
                    raise FileNotFoundError(f"Registered dataset content is missing: {storage_path}")
                self._active_dataset_id = ver_rec.dataset_id
                self._active_dataset_version = ver_rec.version
                self._dataset = load_dataset(storage_path)
                if self.enable_db_pushdown:
                    db_p = storage_path.parent / "records.sqlite"
                    self._db_executor = DatabaseExecutor(db_p)
                    self._db_executor.sync_dataset_file(storage_path, self._dataset.columns, self._dataset.column_types)
                return self._dataset

        # Standard direct file path loading
        if not is_path:
            raise FileNotFoundError(f"Dataset file not found or dataset ID is unregistered: {dataset}")
        self._dataset = load_dataset(candidate_path)
        if self.enable_db_pushdown:
            db_p = candidate_path.parent / f"{candidate_path.stem}.sqlite"
            self._db_executor = DatabaseExecutor(db_p)
            self._db_executor.sync_dataset_file(candidate_path, self._dataset.columns, self._dataset.column_types)

        return self._dataset

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _validate_direct_file_path(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Dataset file not found: {resolved}")
        if not any(self._is_within(resolved, root) for root in self.approved_file_roots):
            raise PermissionError(f"Direct dataset path is outside configured approved roots: {resolved}")

    def ask(
        self,
        question: str,
        dataset_path: Union[str, Path, Dataset, None] = None,
        version: Optional[int] = None,
        user_context: Optional[UserContext] = None,
        use_gemma: bool = False,
        explain_with_gemma: bool = False,
    ) -> QAResult:
        """
        Answer a natural language question against the dataset.
        1. Checks dataset-level permissions.
        2. Proposes query plan (via Gemma 3 4B if enabled, or rule-based parser fallback).
        3. Validates column permissions and schema allowlist.
        4. Executes calculation deterministically.
        5. Attaches evidence, dataset provenance, and optional grounded explanation.
        """
        ds = self._dataset
        if dataset_path is not None:
            try:
                ds = self.load(dataset_path, version=version, user_context=user_context)
            except PermissionError as err:
                return QAResult(question=question, status="abstained", answer=None, reason=str(err),
                                evidence={"execution": not_executed_record("dataset_permission_denied")})

        if ds is None:
            return QAResult(
                question=question,
                status="abstained",
                answer=None,
                reason="No dataset loaded. Provide a dataset path or ID before asking questions.",
            )

        # 1. Dataset-level permission check
        if self._active_dataset_id and not can_access_dataset(user_context, self._active_dataset_id):
            return QAResult(
                question=question,
                status="abstained",
                answer=None,
                reason=f"Access denied: An authorized user context is required for dataset '{self._active_dataset_id}'.",
                evidence={"execution": not_executed_record("dataset_permission_denied")},
            )

        # 2. Query planning (Gemma 3 4B or rule-based parser)
        intent = None
        compound_filter = None
        planning_source = "deterministic_parser"

        if use_gemma and self.gemma_planner is not None:
            try:
                plan = self.gemma_planner.propose_plan(
                    question=question,
                    columns=ds.columns,
                    column_types=ds.column_types,
                    dataset_name=self._active_dataset_id or ds.name,
                )
                op = plan.get("operation")
                col = plan.get("column")
                agg_func = plan.get("agg_function")
                f_dict = plan.get("filter")
                try:
                    compound_filter = _dict_to_filter(f_dict)
                except ValueError as err:
                    return QAResult(question=question, status="abstained", answer=None,
                                    operation=op, reason=f"Invalid model filter: {err}",
                                    evidence={"execution": not_executed_record("invalid_model_filter", ds, op)})

                # Convert to QueryIntent
                intent = QueryIntent(
                    operation=op,
                    column=col,
                    agg_function=agg_func,
                    filter_column=compound_filter.column if isinstance(compound_filter, FilterCriterion) else None,
                    filter_operator=compound_filter.operator if isinstance(compound_filter, FilterCriterion) else None,
                    filter_value=compound_filter.value if isinstance(compound_filter, FilterCriterion) else None,
                    raw_question=question,
                    abstain_reason=plan.get("reason"),
                    compound_filter=compound_filter,
                )
                planning_source = "gemma_validated_intent"
            except Exception as err:
                # Fallback to rule-based parser on model failure or unreachable Ollama
                intent = parse_question(question, ds.columns)
                planning_source = "deterministic_parser_fallback"

        if intent is None:
            intent = parse_question(question, ds.columns)

        # 3. Column-level permission checks
        cols_to_check = []
        if intent.column:
            cols_to_check.append(intent.column)
        cols_to_check.extend(filter_columns(compound_filter))
        if intent.filter_column and compound_filter is None:
            cols_to_check.append(intent.filter_column)

        if self._active_dataset_id and cols_to_check:
            ok, unauthorized = can_access_columns(user_context, self._active_dataset_id, cols_to_check)
            if not ok:
                return QAResult(
                    question=question,
                    status="abstained",
                    answer=None,
                    operation=intent.operation,
                    reason=f"Access denied: User lacks permission for column(s) {unauthorized}.",
                    evidence={"execution": not_executed_record("column_permission_denied")},
                )

        # 4. Schema validation
        validation = validate_intent(intent, ds)
        if not validation.is_valid:
            return QAResult(
                question=question,
                status="abstained",
                answer=None,
                operation=intent.operation,
                reason=validation.reason,
                evidence={"execution": not_executed_record("schema_validation_failed", ds, intent.operation)},
            )

        # Handle empty datasets
        if len(ds.rows) == 0 and intent.operation not in {Operation.ROW_COUNT, Operation.COLUMN_NAMES}:
            return QAResult(
                question=question,
                status="abstained",
                answer=None,
                operation=intent.operation,
                reason="Dataset contains 0 rows; cannot perform requested calculation.",
                evidence={"execution": not_executed_record("empty_dataset", ds, intent.operation)},
            )

        # 5. Deterministic Execution (Database pushdown or in-memory)
        db_supported = intent.operation in {Operation.ROW_COUNT, Operation.FILTER_COUNT,
                                            Operation.NUMERIC_AGGREGATION, Operation.CATEGORY_COUNT}
        backend = ("sqlite" if db_supported and self._db_executor is not None and
                   (compound_filter is not None or self.enable_db_pushdown) else "in_memory")
        execution_failed = False
        try:
            if backend == "sqlite":
                result = self._execute_via_db(intent, compound_filter, ds)
            else:
                result = evaluate_intent(intent, ds)
        except Exception as err:
            execution_failed = True
            failure_record = not_executed_record("execution_error", ds, intent.operation)
            failure_record["status"] = "failed"
            result = QAResult(question=question, status="abstained", answer=None,
                              operation=intent.operation,
                              reason=f"Execution failed ({type(err).__name__}); no result was verified.",
                              evidence={"execution": failure_record})

        # 6. Attach dataset provenance
        result.evidence = result.evidence or {}
        if not execution_failed:
            result.evidence["execution"] = single_execution_record(
                result, ds, intent, backend, planning_source=planning_source
            )
        if self._active_dataset_id is not None:
            result.evidence["dataset_id"] = self._active_dataset_id
            result.evidence["dataset_version"] = self._active_dataset_version

        # 7. Grounded explanation from verified result (Gemma 3 4B)
        # Explanations stay deterministic even when the legacy explain_with_gemma flag is set.
        result.explanation = deterministic_explanation(
            result, self._active_dataset_id or ds.name, self._active_dataset_version
        )
        result.explanation_source = "deterministic_execution_evidence"

        return result

    def _execute_via_db(
        self,
        intent: QueryIntent,
        compound_filter: Optional[Union[FilterCriterion, CompoundFilter]],
        dataset: Dataset,
    ) -> QAResult:
        """Executes query using SQLite query pushdown."""
        executor = self._db_executor
        assert executor is not None

        op = intent.operation
        filter_obj = compound_filter
        if filter_obj is None and intent.filter_column and intent.filter_operator:
            filter_obj = FilterCriterion(
                column=intent.filter_column,
                operator=intent.filter_operator,
                value=intent.filter_value,
            )

        if op == Operation.ROW_COUNT:
            cnt = executor.execute_count(dataset.columns, filter_obj=filter_obj)
            return QAResult(
                question=intent.raw_question,
                status="answered",
                answer=cnt,
                operation=op,
                calculation_method=f"Counted rows via SQLite pushdown from {dataset.name}.",
                evidence={"row_count": cnt, "execution_backend": "sqlite"},
            )

        elif op == Operation.FILTER_COUNT:
            cnt = executor.execute_count(dataset.columns, filter_obj=filter_obj)
            return QAResult(
                question=intent.raw_question,
                status="answered",
                answer=cnt,
                operation=op,
                calculation_method="Filtered and counted records via SQLite pushdown.",
                evidence={"matched_count": cnt, "execution_backend": "sqlite"},
            )

        elif op == Operation.NUMERIC_AGGREGATION and intent.column and intent.agg_function:
            ans = executor.execute_aggregation(
                agg_function=intent.agg_function,
                column=intent.column,
                available_columns=dataset.columns,
                filter_obj=filter_obj,
            )
            if ans is None:
                return QAResult(
                    question=intent.raw_question,
                    status="abstained",
                    answer=None,
                    operation=op,
                    reason=f"No numeric values available to compute {intent.agg_function}.",
                )
            return QAResult(
                question=intent.raw_question,
                status="answered",
                answer=ans,
                operation=op,
                calculation_method=f"Computed {intent.agg_function}('{intent.column}') via SQLite pushdown.",
                evidence={"column": intent.column, "agg_function": intent.agg_function, "execution_backend": "sqlite"},
            )

        elif op == Operation.CATEGORY_COUNT and intent.column:
            counts = executor.execute_category_count(intent.column, dataset.columns, filter_obj=filter_obj)
            return QAResult(
                question=intent.raw_question,
                status="answered",
                answer=counts,
                operation=op,
                calculation_method=f"Computed GROUP BY count('{intent.column}') via SQLite pushdown.",
                evidence={"column": intent.column, "distribution": counts, "execution_backend": "sqlite"},
            )

        # Fallback to in-memory evaluator for non-DB operations (like duplicate_count, column_names)
        return evaluate_intent(intent, dataset)

    def ask_multi_dataset(
        self, question: str, plan: Dict[str, Any], dataset_sources: Dict[str, Union[str, Path]],
        relationships: List[tuple], user_context: Optional[UserContext] = None,
        versions: Optional[Dict[str, int]] = None,
    ) -> QAResult:
        """Validate and execute a bounded multi-dataset analytical plan."""
        from src.analytics.plan import AnalyticalPlanExecutor
        from src.storage.manager import compute_sha256
        from src.verification.evidence import dataset_provenance
        loaded, provenance = {}, {}
        plan_validated = False
        try:
            for name, source in dataset_sources.items():
                reader = QAEngine(registry_service=self.registry, approved_file_roots=self.approved_file_roots)
                ds = reader.load(source, version=(versions or {}).get(name), user_context=user_context)
                ds.dataset_id, ds.dataset_version = reader._active_dataset_id, reader._active_dataset_version
                loaded[name] = ds
                provenance[name] = {**dataset_provenance(ds),
                                    "checksum_sha256": compute_sha256(ds.file_path)}
            executor = AnalyticalPlanExecutor(loaded, relationships)
            executor.validate(plan, user_context)
            plan_validated = True
            output = executor.execute(plan, user_context, provenance)
        except Exception as err:
            failure_record = not_executed_record("plan_validation_or_execution_failed",
                                                  operation="multi_step_analytics")
            if plan_validated:
                failure_record["status"] = "failed"
                failure_record["operations_known"] = False
                failure_record["limitations"] = ["The validated plan failed during execution; a complete step trace is unavailable."]
            reason = (f"Execution failed ({type(err).__name__}); no result was verified."
                      if plan_validated else str(err))
            result = QAResult(question=question, status="abstained", answer=None, reason=reason,
                              operation="multi_step_analytics",
                              evidence={"execution": failure_record})
            result.explanation = deterministic_explanation(result, "requested datasets")
            result.explanation_source = "deterministic_execution_evidence"
            return result
        result = QAResult(question=question, status="answered", answer=output["outputs"],
                          operation="multi_step_analytics", calculation_method="Executed validated deterministic analytical plan.",
                          evidence=output["evidence"])
        result.explanation = deterministic_explanation(result, "multiple datasets")
        result.explanation_source = "deterministic_execution_evidence"
        return result
