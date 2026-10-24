import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import query_planner, retrieval, retrieval_pipeline

MANUAL_SCRIPT_PATH = Path(__file__).with_name("test_retrieval_manual.py")
MANUAL_SCRIPT_SPEC = importlib.util.spec_from_file_location("retrieval_manual", MANUAL_SCRIPT_PATH)
retrieval_manual = importlib.util.module_from_spec(MANUAL_SCRIPT_SPEC)
MANUAL_SCRIPT_SPEC.loader.exec_module(retrieval_manual)


def _task(
    task_id="t1",
    query="Infosys revenue FY2026",
    company="Infosys",
    company_mention="Infosys",
    financial_year="FY2026",
    chunk_type=query_planner.ChunkType.table,
):
    return query_planner.RetrievalTask(
        task_id=task_id,
        query=query,
        company=company,
        company_mention=company_mention,
        financial_year=financial_year,
        metric="revenue",
        chunk_type=chunk_type,
    )


def _plan(
    tasks=None,
    intents=None,
    needs_clarification=False,
    clarification_question=None,
    warnings=None,
    planner_status="ok",
):
    return query_planner.QueryPlan(
        original_query="Original question",
        normalized_query="Normalized question",
        intents=intents or [query_planner.Intent.lookup],
        tasks=tasks if tasks is not None else [_task()],
        calculations=[],
        comparison=None,
        visualization_required=False,
        visualization=None,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        warnings=warnings or [],
        planner_status=planner_status,
    )


class FakePlanner:
    def __init__(self, plan=None, error=None):
        self.plan_result = plan or _plan()
        self.error = error
        self.queries = []

    def plan(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.plan_result


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if isinstance(self.results, list) and self.results and isinstance(self.results[0], list):
            return self.results[len(self.calls) - 1]
        return self.results


def _result(chunk_id, **metadata):
    chunk = {
        "chunk_id": chunk_id,
        "text": f"evidence for {chunk_id}",
        "company": "Infosys",
        "document": "Annual Report",
        "financial_year": "2025-2026",
        "page": 33,
        "section": "Board report",
        "chunk_type": "table",
        **metadata,
    }
    return retrieval.RetrievalResult(chunk_id, 0.91, 1, chunk)


def _pipeline(plan=None, results=None, planner_error=None, retriever_error=None):
    planner = FakePlanner(plan=plan, error=planner_error)
    retriever = FakeRetriever(results=results, error=retriever_error)
    return retrieval_pipeline.FinancialRetrievalPipeline(planner, retriever), planner, retriever


def test_original_question_reaches_planner_unchanged_once():
    pipeline, planner, _ = _pipeline()
    original = "  What was Infosys's revenue in FY2026?  "

    result = pipeline.run(original)

    assert planner.queries == [original]
    assert result.original_question == original
    assert result.plan is planner.plan_result


def test_rewritten_task_query_and_resolved_filters_drive_retrieval(monkeypatch):
    pipeline, _, retriever = _pipeline()
    conversions = []
    original_converter = query_planner.metadata_financial_year

    def track_conversion(label):
        conversions.append(label)
        return original_converter(label)

    monkeypatch.setattr(retrieval_pipeline.query_planner, "metadata_financial_year", track_conversion)
    result = pipeline.run("What was Infosys's revenue in FY2026?", top_k=4)

    assert result.task_results[0].query == "Infosys revenue FY2026"
    assert retriever.calls == [{
        "query": "Infosys revenue FY2026",
        "top_k": 4,
        "company": "Infosys",
        "financial_year": "2025-2026",
        "chunk_type": "table",
    }]
    assert conversions == ["FY2026"]


def test_chunk_type_any_maps_to_no_filter():
    plan = _plan(tasks=[_task(chunk_type=query_planner.ChunkType.any)])
    pipeline, _, retriever = _pipeline(plan)

    pipeline.run("question")

    assert retriever.calls[0]["chunk_type"] is None


def test_missing_planner_company_and_year_filters_are_not_invented():
    task = _task(
        company=None,
        company_mention=None,
        financial_year=None,
        chunk_type=query_planner.ChunkType.any,
    )
    pipeline, _, retriever = _pipeline(_plan(tasks=[task]))

    pipeline.run("general revenue question")

    assert retriever.calls[0]["company"] is None
    assert retriever.calls[0]["financial_year"] is None
    assert retriever.calls[0]["chunk_type"] is None


def test_multiple_tasks_keep_separate_results_and_provenance():
    plan = _plan(tasks=[
        _task(task_id="revenue", query="Infosys revenue", financial_year="FY2026"),
        _task(task_id="prior-year", query="Infosys prior-year revenue", financial_year="FY2025"),
    ])
    first, second = _result("current-year"), _result("prior-year", financial_year="2024-2025")
    pipeline, _, retriever = _pipeline(plan, results=[[first], [second]])

    result = pipeline.run("compare revenue")

    assert [item.task_id for item in result.task_results] == ["revenue", "prior-year"]
    assert result.task_results[0].results == [first]
    assert result.task_results[1].results == [second]
    assert [call["financial_year"] for call in retriever.calls] == ["2025-2026", "2024-2025"]
    assert result.task_results[0].results[0].chunk["page"] == 33


def test_clarification_plan_performs_no_retrieval():
    plan = _plan(needs_clarification=True, clarification_question="Which company?")
    pipeline, _, retriever = _pipeline(plan)

    result = pipeline.run("ambiguous financial question")

    assert result.status == "needs_clarification"
    assert result.plan.clarification_question == "Which company?"
    assert result.task_results[0].skipped_reason == "Clarification is required before retrieval."
    assert retriever.calls == []


def test_out_of_scope_plan_performs_no_retrieval():
    plan = _plan(tasks=[], intents=[query_planner.Intent.out_of_scope])
    pipeline, _, retriever = _pipeline(plan)

    result = pipeline.run("weather today")

    assert result.status == "out_of_scope"
    assert result.task_results == []
    assert retriever.calls == []


def test_unresolved_company_is_skipped_without_removing_safety_boundary():
    plan = _plan(tasks=[
        _task(task_id="supported", company="Infosys"),
        _task(task_id="unresolved", company=None, company_mention="Unknown Bank"),
    ])
    pipeline, _, retriever = _pipeline(plan, results=[[_result("supported-result")]])

    result = pipeline.run("compare Infosys and Unknown Bank")

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["company"] == "Infosys"
    assert result.status == "partial"
    assert result.task_results[0].results[0].chunk_id == "supported-result"
    assert result.task_results[1].skipped_reason
    assert "Unknown Bank" in result.task_results[1].skipped_reason
    assert any("Unknown Bank" in warning for warning in result.warnings)


def test_planner_failure_and_fallback_never_trigger_unfiltered_retrieval():
    pipeline, _, retriever = _pipeline(planner_error=RuntimeError("planner unavailable"))

    with pytest.raises(RuntimeError, match="planner unavailable"):
        pipeline.run("Infosys revenue")
    assert retriever.calls == []

    fallback_pipeline, _, fallback_retriever = _pipeline(
        _plan(planner_status="fallback", tasks=[_task(company=None, financial_year=None,
                                                       chunk_type=query_planner.ChunkType.any)])
    )
    with pytest.raises(retrieval_pipeline.RetrievalPipelineError, match="unplanned retrieval is disabled"):
        fallback_pipeline.run("Infosys revenue")
    assert fallback_retriever.calls == []


def test_no_task_plan_returns_clear_result():
    plan = _plan(tasks=[], needs_clarification=True, clarification_question="Please clarify.")
    pipeline, _, retriever = _pipeline(plan)

    result = pipeline.run("question")

    assert result.status == "needs_clarification"
    assert result.task_results == []
    assert retriever.calls == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_rejected_before_planning(top_k):
    pipeline, planner, retriever = _pipeline()

    with pytest.raises(ValueError):
        pipeline.run("question", top_k=top_k)

    assert planner.queries == []
    assert retriever.calls == []


def test_retrieval_errors_are_attached_to_the_task():
    pipeline, _, retriever = _pipeline(retriever_error=ValueError("invalid collection filter"))

    result = pipeline.run("Infosys revenue")

    assert result.status == "partial"
    assert result.task_results[0].task_id == "t1"
    assert "Task 't1' retrieval failed" in result.task_results[0].error
    assert "invalid collection filter" in result.task_results[0].error
    assert retriever.calls[0]["company"] == "Infosys"


def test_baseline_strategy_remains_default_and_uses_existing_retriever():
    pipeline, _, retriever = _pipeline()

    result = pipeline.run("Infosys revenue")

    assert result.retrieval_strategy == "baseline"
    assert "strategy" not in retriever.calls[0]


def test_experimental_strategy_requires_and_uses_hybrid_retriever():
    planner = FakePlanner()
    baseline = FakeRetriever()

    with pytest.raises(retrieval_pipeline.RetrievalPipelineError, match="requires an experimental retriever"):
        retrieval_pipeline.FinancialRetrievalPipeline(
            planner, baseline, retrieval_strategy="hybrid_all"
        )

    class FakeHybridRetriever:
        def __init__(self):
            self.calls = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return [_result("hybrid-hit")]

    hybrid = FakeHybridRetriever()
    pipeline = retrieval_pipeline.FinancialRetrievalPipeline(
        planner,
        baseline,
        retrieval_strategy="hybrid_chunk_keyword",
        experimental_retriever=hybrid,
    )
    result = pipeline.run("Infosys revenue")

    assert result.retrieval_strategy == "hybrid_chunk_keyword"
    assert hybrid.calls[0]["strategy"] == "hybrid_chunk_keyword"
    assert hybrid.calls[0]["top_k"] == 10
    assert baseline.calls == []


def test_explicit_filters_are_additional_constraints_and_conflicts_fail():
    pipeline, _, retriever = _pipeline()

    result = pipeline.run(
        "Infosys revenue",
        company="Infosys",
        financial_year="2025-2026",
        chunk_type="table",
    )

    assert result.task_results[0].filters == {
        "company": "Infosys",
        "financial_year": "2025-2026",
        "chunk_type": "table",
    }
    assert retriever.calls[0] == {
        "query": "Infosys revenue FY2026",
        "top_k": 5,
        "company": "Infosys",
        "financial_year": "2025-2026",
        "chunk_type": "table",
    }

    call_count = len(retriever.calls)
    for conflicting_filter in (
        {"company": "RelianceIndustries"},
        {"financial_year": "2024-2025"},
        {"chunk_type": "text"},
    ):
        with pytest.raises(ValueError, match="conflicts with task"):
            pipeline.run("Infosys revenue", **conflicting_filter)
    assert len(retriever.calls) == call_count


def test_explicit_filters_are_passed_verbatim_when_plan_has_no_constraints():
    task = _task(
        company=None,
        company_mention=None,
        financial_year=None,
        chunk_type=query_planner.ChunkType.any,
    )
    pipeline, _, retriever = _pipeline(_plan(tasks=[task]))
    explicit_filters = {
        "company": "RelianceIndustries",
        "financial_year": "2025-2026",
        "chunk_type": "table",
    }

    result = pipeline.run("company financial question", **explicit_filters)

    assert result.task_results[0].filters == explicit_filters
    assert {name: retriever.calls[0][name] for name in explicit_filters} == explicit_filters


def test_manual_result_printer_displays_complete_chunk_text(capsys):
    result = _result("long-chunk")
    full_text = "x" * 1500
    result.chunk["text"] = full_text

    retrieval_manual._print_result(result)

    assert full_text in capsys.readouterr().out


def test_manual_script_prompts_when_query_flag_is_omitted(monkeypatch):
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return "  What was TCS revenue?  "

    monkeypatch.setattr("builtins.input", fake_input)

    assert retrieval_manual._read_query(None) == "What was TCS revenue?"
    assert prompts == ["Question: "]


def test_manual_script_uses_explicit_query_without_prompt(monkeypatch):
    def fail_if_prompted(prompt):
        raise AssertionError("Explicit --query must not prompt.")

    monkeypatch.setattr("builtins.input", fail_if_prompted)

    assert retrieval_manual._read_query("Infosys revenue?") == "Infosys revenue?"