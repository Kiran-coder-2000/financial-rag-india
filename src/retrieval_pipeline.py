"""Coordinate validated Gemini query plans with Chroma evidence retrieval."""

from dataclasses import dataclass, field

from . import query_planner
from .retrieval import ChromaRetriever, RetrievalResult


class RetrievalPipelineError(RuntimeError):
    """Raised when a planner result cannot safely drive retrieval."""


@dataclass
class TaskRetrievalResult:
    task_id: str
    query: str
    filters: dict[str, str]
    results: list[RetrievalResult] = field(default_factory=list)
    skipped_reason: str | None = None
    error: str | None = None


@dataclass
class FinancialRetrievalResult:
    original_question: str
    plan: query_planner.QueryPlan
    task_results: list[TaskRetrievalResult]
    warnings: list[str]
    status: str


class FinancialRetrievalPipeline:
    """Use the planner's atomic queries and constraints to retrieve evidence."""

    def __init__(self, planner, retriever: ChromaRetriever):
        self.planner = planner
        self.retriever = retriever

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}.")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    @staticmethod
    def _validate_explicit_filters(filters: dict[str, str | None]) -> dict[str, str]:
        validated = {}
        for name, value in filters.items():
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Explicit {name} filter must be a non-empty string.")
            validated[name] = value
        return validated

    @staticmethod
    def _task_filters(task, explicit_filters: dict[str, str]) -> dict[str, str]:
        filters = {}
        if task.company is not None:
            filters["company"] = task.company
        if task.financial_year is not None:
            try:
                filters["financial_year"] = query_planner.metadata_financial_year(
                    task.financial_year
                )
            except (TypeError, ValueError, IndexError) as error:
                raise RetrievalPipelineError(
                    f"Task {task.task_id!r} has an invalid planned financial year "
                    f"{task.financial_year!r}."
                ) from error
        if task.chunk_type is not query_planner.ChunkType.any:
            filters["chunk_type"] = task.chunk_type.value

        for name, value in explicit_filters.items():
            planned_value = filters.get(name)
            if planned_value is not None and planned_value != value:
                raise ValueError(
                    f"Explicit {name} filter {value!r} conflicts with task "
                    f"{task.task_id!r} planner filter {planned_value!r}."
                )
            filters[name] = value
        return filters

    def run(
        self,
        user_query: str,
        top_k: int = 5,
        company: str | None = None,
        financial_year: str | None = None,
        chunk_type: str | None = None,
    ) -> FinancialRetrievalResult:
        """Plan once, then retrieve each usable task without merging its results.

        Optional explicit filters act as additional constraints. They never
        replace a conflicting company, year, or chunk type chosen by the plan.
        Explicit financial years must use the exact metadata form stored by
        Chroma, for example ``2025-2026``.
        """
        self._validate_top_k(top_k)
        if not isinstance(user_query, str) or not user_query.strip():
            raise query_planner.PlannerInputError("Query must be a non-empty string.")
        explicit_filters = self._validate_explicit_filters({
            "company": company,
            "financial_year": financial_year,
            "chunk_type": chunk_type,
        })

        plan = self.planner.plan(user_query)
        if plan.planner_status != "ok":
            raise RetrievalPipelineError(
                f"Planner returned status {plan.planner_status!r}; unplanned retrieval is disabled."
            )

        warnings = list(plan.warnings)
        if plan.needs_clarification:
            task_results = []
            for task in plan.tasks:
                if task.is_unresolved_company:
                    reason = (
                        f"Company {task.company_mention!r} is unsupported; clarification is required."
                    )
                else:
                    reason = "Clarification is required before retrieval."
                task_results.append(TaskRetrievalResult(
                    task_id=task.task_id,
                    query=task.query,
                    filters={},
                    skipped_reason=reason,
                ))
            return FinancialRetrievalResult(
                user_query, plan, task_results, warnings, "needs_clarification"
            )
        if query_planner.Intent.out_of_scope in plan.intents:
            return FinancialRetrievalResult(user_query, plan, [], warnings, "out_of_scope")
        if not plan.tasks:
            warnings.append("The planner returned no usable retrieval tasks.")
            return FinancialRetrievalResult(user_query, plan, [], warnings, "no_tasks")

        task_results = []
        for task in plan.tasks:
            if task.is_unresolved_company:
                reason = (
                    f"Company {task.company_mention!r} is unsupported; this task was not retrieved."
                )
                warnings.append(reason)
                task_results.append(TaskRetrievalResult(
                    task_id=task.task_id,
                    query=task.query,
                    filters={},
                    skipped_reason=reason,
                ))
                continue

            filters = self._task_filters(task, explicit_filters)
            try:
                results = self.retriever.search(
                    query=task.query,
                    top_k=top_k,
                    company=filters.get("company"),
                    financial_year=filters.get("financial_year"),
                    chunk_type=filters.get("chunk_type"),
                )
            except Exception as error:
                message = f"Task {task.task_id!r} retrieval failed: {type(error).__name__}: {error}"
                warnings.append(message)
                task_results.append(TaskRetrievalResult(
                    task_id=task.task_id,
                    query=task.query,
                    filters=filters,
                    error=message,
                ))
                continue

            task_results.append(TaskRetrievalResult(
                task_id=task.task_id,
                query=task.query,
                filters=filters,
                results=results,
            ))

        status = (
            "partial"
            if any(result.error or result.skipped_reason for result in task_results)
            else "completed"
        )
        return FinancialRetrievalResult(
            user_query,
            plan,
            task_results,
            warnings,
            status,
        )