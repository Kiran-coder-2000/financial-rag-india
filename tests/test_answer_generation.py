import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import answer_generation, query_planner, retrieval, retrieval_pipeline
from src.llm.base import (
    GenerationResponse,
    LLMProviderCapabilityError,
    LLMProviderResponseError,
)
from src.llm.validation import validate_json_output, validate_response_schema


def _task(task_id="t1", query="Infosys revenue", company="Infosys"):
    return query_planner.RetrievalTask(
        task_id=task_id,
        query=query,
        company_mention=company,
        company=company,
        financial_year="FY2025",
        metric="revenue",
        chunk_type=query_planner.ChunkType.table,
    )


def _plan(
    tasks=None,
    intents=None,
    needs_clarification=False,
    clarification_question=None,
    planner_status="ok",
):
    return query_planner.QueryPlan(
        original_query="Original plan question",
        normalized_query="Normalized plan question",
        intents=intents or [query_planner.Intent.lookup],
        tasks=tasks if tasks is not None else [_task()],
        calculations=[],
        comparison=None,
        visualization_required=False,
        visualization=None,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        planner_status=planner_status,
    )


def _chunk(chunk_id="chunk-1", **overrides):
    return {
        "chunk_id": chunk_id,
        "text": f"Original report text for {chunk_id}.",
        "company": "Infosys",
        "document": "Annual Report",
        "financial_year": "2024-2025",
        "page": 33,
        "section": "Financial statements",
        "chunk_type": "text",
        **overrides,
    }


def _hit(chunk_id="chunk-1", rank=1, score=0.92, chunk=None):
    return retrieval.RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        chunk=chunk if chunk is not None else _chunk(chunk_id),
    )


def _task_result(task_id="t1", query="Infosys revenue", results=None, **overrides):
    return retrieval_pipeline.TaskRetrievalResult(
        task_id=task_id,
        query=query,
        filters={"company": "Infosys"},
        results=results if results is not None else [],
        **overrides,
    )


def _pipeline_result(
    question="What was Infosys revenue?",
    plan=None,
    task_results=None,
    status="completed",
    retrieval_strategy="baseline",
):
    return retrieval_pipeline.FinancialRetrievalResult(
        original_question=question,
        plan=plan if plan is not None else _plan(),
        task_results=task_results if task_results is not None else [],
        warnings=[],
        status=status,
        retrieval_strategy=retrieval_strategy,
    )


def _claim(chunk_id="chunk-1", claim_id="c1", text="Revenue was reported."):
    return {
        "claim_id": claim_id,
        "text": text,
        "claim_type": "qualitative_fact",
        "citation_chunk_ids": [chunk_id],
    }


def _answer(
    mode="corpus_grounded",
    status="answered",
    claims=None,
    limitations=None,
    text="The report states the requested information.",
):
    return {
        "answer_mode": mode,
        "status": status,
        "answer": text,
        "claims": claims if claims is not None else [_claim()],
        "limitations": limitations if limitations is not None else [],
    }


class FakeProvider:
    provider_name = "fake"
    model = "fake-model"

    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        text = self.value if isinstance(self.value, str) else json.dumps(self.value)
        return GenerationResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
        )


def _generator(value=None, error=None):
    provider = FakeProvider(value=value, error=error)
    return answer_generation.FinancialAnswerGenerator(provider), provider


def test_response_schema_is_valid_draft_2020_12_and_forbids_extra_fields():
    validate_response_schema(answer_generation.ANSWER_RESPONSE_SCHEMA)
    for mode in answer_generation.ANSWER_MODES:
        validate_response_schema(answer_generation._request_schema(mode))


def test_corpus_grounded_question_uses_provider_and_returns_claim_citation():
    provider_answer = _answer()
    generator, provider = _generator(provider_answer)
    result = _pipeline_result(
        task_results=[_task_result(results=[_hit()])]
    )

    output = generator.generate(result)

    assert output["answer_mode"] == "corpus_grounded"
    assert output["status"] == "answered"
    assert output["claims"][0]["citation_chunk_ids"] == ["chunk-1"]
    assert output["answer"] == output["claims"][0]["text"]
    assert output["answer"] != provider_answer["answer"]
    assert output["citations"][0]["chunk_id"] == "chunk-1"
    assert len(provider.requests) == 1
    assert provider.requests[0].json_mode is True
    assert provider.requests[0].response_schema is not None
    assert json.loads(provider.requests[0].user_prompt)["original_user_question"] == (
        "What was Infosys revenue?"
    )
    assert "score" not in provider.requests[0].user_prompt
    assert "retrieval_rank" not in provider.requests[0].user_prompt


def test_corpus_answer_cannot_add_facts_outside_validated_claims():
    provider_answer = _answer(
        claims=[_claim(text="Infosys revenue increased by 12%.")],
        text=(
            "Infosys revenue increased by 12%. Its operating margin improved "
            "due to better cost management."
        ),
    )
    generator, _ = _generator(provider_answer)

    output = generator.generate(_pipeline_result(
        task_results=[_task_result(results=[_hit()])]
    ))

    assert output["answer"] == "Infosys revenue increased by 12%."
    assert "operating margin" not in output["answer"]
    assert len(output["claims"]) == 1
    assert output["claims"][0]["citation_chunk_ids"] == ["chunk-1"]


def test_general_knowledge_question_routes_out_of_scope_planner_result_to_provider():
    provider_answer = _answer(
        mode="general_knowledge",
        claims=[{
            "claim_id": "c1",
            "text": "An LLM generates text from learned patterns.",
            "claim_type": "qualitative_fact",
            "citation_chunk_ids": [],
        }],
        text="An LLM generates text from patterns learned during training.",
    )
    generator, provider = _generator(provider_answer)
    result = _pipeline_result(
        question="How does an LLM work?",
        plan=_plan(tasks=[], intents=[query_planner.Intent.out_of_scope]),
        status="out_of_scope",
    )

    output = generator.generate(result)

    assert output["answer_mode"] == "general_knowledge"
    assert output["citations"] == []
    assert provider.requests
    assert "financial-report corpus" in provider.requests[0].system_instruction
    assert "tasks" not in json.loads(provider.requests[0].user_prompt)


def test_general_knowledge_is_not_blocked_by_empty_financial_evidence():
    answer = _answer(
        mode="general_knowledge",
        claims=[{
            "claim_id": "c1",
            "text": "Artificial intelligence describes computer systems performing tasks.",
            "claim_type": "qualitative_fact",
            "citation_chunk_ids": [],
        }],
    )
    generator, provider = _generator(answer)
    result = _pipeline_result(
        question="What is artificial intelligence?",
        plan=_plan(tasks=[], intents=[query_planner.Intent.out_of_scope]),
        status="out_of_scope",
    )

    output = generator.generate(result)

    assert output["answer_mode"] == "general_knowledge"
    assert output["status"] == "answered"
    assert output["answer"] == output["claims"][0]["text"]
    assert len(provider.requests) == 1


def test_current_information_request_returns_limitation_without_provider_call():
    generator, provider = _generator(
        _answer(mode="corpus_grounded")
    )
    result = _pipeline_result(
        question="What is TCS's current share price?",
        task_results=[_task_result(results=[_hit()])],
    )

    output = generator.generate(result)

    assert output["answer_mode"] == "current_information_unavailable"
    assert output["status"] == "abstained"
    assert output["claims"] == []
    assert output["citations"] == []
    assert "no live information source" in output["answer"]
    assert provider.requests == []


def test_current_market_event_does_not_use_historical_report_evidence():
    generator, provider = _generator(_answer())
    result = _pipeline_result(
        question="Why did TCS stock price crash today?",
        task_results=[_task_result(results=[_hit()])],
    )

    output = generator.generate(result)

    assert output["answer_mode"] == "current_information_unavailable"
    assert "historical reports" in output["answer"]
    assert provider.requests == []


@pytest.mark.parametrize(
    "question",
    [
        "Why did TCS stock price crash today?",
        "Why is TCS stock falling?",
        "Why are Infosys shares declining this morning?",
        "What is Reliance's current share price?",
        "What happened in the stock market this week?",
    ],
)
def test_current_market_questions_do_not_use_report_evidence(question):
    generator, provider = _generator(_answer())

    output = generator.generate(_pipeline_result(
        question=question,
        task_results=[_task_result(results=[_hit()])],
    ))

    assert output["answer_mode"] == "current_information_unavailable"
    assert output["status"] == "abstained"
    assert provider.requests == []


@pytest.mark.parametrize(
    "question",
    [
        "Why did TCS stock price fall in 2020?",
        "What was Infosys' share price in FY2025?",
        "What is TCS's latest share price in FY2020?",
        "Explain Reliance's stock performance during 2019–2020.",
        "What caused the market decline in March 2020?",
    ],
)
def test_explicit_historical_periods_take_precedence(question):
    generator, provider = _generator(_answer())

    output = generator.generate(_pipeline_result(
        question=question,
        task_results=[_task_result(results=[_hit()])],
    ))

    assert output["answer_mode"] == "corpus_grounded"
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "question",
    [
        "Why did Infosys shares decline?",
        "Why did TCS stock fall?",
    ],
)
def test_unqualified_market_movements_request_a_period_clarification(question):
    generator, provider = _generator(_answer())

    output = generator.generate(_pipeline_result(
        question=question,
        task_results=[_task_result(results=[_hit()])],
    ))

    assert output["answer_mode"] == "corpus_grounded"
    assert output["status"] == "abstained"
    assert output["claims"] == []
    assert "current market movement or a specific historical period" in output["answer"]
    assert provider.requests == []


@pytest.mark.parametrize(
    "question",
    [
        "What is artificial intelligence?",
        "How does an LLM work?",
    ],
)
def test_ordinary_general_questions_do_not_route_to_current_information(question):
    response = _answer(
        mode="general_knowledge",
        claims=[{
            "claim_id": "c1",
            "text": "Artificial intelligence is a broad area of computing.",
            "claim_type": "qualitative_fact",
            "citation_chunk_ids": [],
        }],
    )
    generator, provider = _generator(response)

    output = generator.generate(_pipeline_result(
        question=question,
        plan=_plan(tasks=[], intents=[query_planner.Intent.out_of_scope]),
        status="out_of_scope",
    ))

    assert output["answer_mode"] == "general_knowledge"
    assert output["citations"] == []
    assert output["answer"] == output["claims"][0]["text"]
    assert len(provider.requests) == 1


def test_unqualified_share_price_request_is_treated_as_current_information():
    generator, provider = _generator(_answer())

    output = generator.generate(_pipeline_result(
        question="What is TCS's share price?",
        task_results=[_task_result(results=[_hit()])],
    ))

    assert output["answer_mode"] == "current_information_unavailable"
    assert provider.requests == []


def test_historical_share_price_request_can_use_report_evidence():
    generator, provider = _generator(_answer())

    output = generator.generate(_pipeline_result(
        question="What was TCS's share price in FY2025?",
        task_results=[_task_result(results=[_hit()])],
    ))

    assert output["answer_mode"] == "corpus_grounded"
    assert len(provider.requests) == 1


def test_financial_question_with_empty_evidence_abstains_without_generation():
    generator, provider = _generator(_answer())

    output = generator.generate(_pipeline_result(
        task_results=[_task_result(results=[])],
    ))

    assert output["answer_mode"] == "corpus_grounded"
    assert output["status"] == "abstained"
    assert output["claims"] == []
    assert output["citations"] == []
    assert provider.requests == []


def test_all_failed_or_skipped_tasks_abstain_without_generation():
    generator, provider = _generator(_answer())
    result = _pipeline_result(
        task_results=[
            _task_result(
                task_id="t1",
                results=[],
                error="retrieval failed",
            ),
            _task_result(
                task_id="t2",
                query="Infosys operating margin",
                results=[],
                skipped_reason="company clarification required",
            ),
        ],
        plan=_plan(tasks=[
            _task(),
            _task(task_id="t2", query="Infosys operating margin"),
        ]),
        status="partial",
    )

    output = generator.generate(result)

    assert output["status"] == "abstained"
    assert provider.requests == []


def test_planner_clarification_abstains_and_keeps_clarification_question():
    generator, provider = _generator(_answer())
    result = _pipeline_result(
        plan=_plan(
            needs_clarification=True,
            clarification_question="Which company do you mean?",
        ),
        task_results=[],
        status="needs_clarification",
    )

    output = generator.generate(result)

    assert output["answer_mode"] == "corpus_grounded"
    assert output["status"] == "abstained"
    assert output["answer"] == "Which company do you mean?"
    assert provider.requests == []


def test_unvalidated_planner_status_abstains_without_generation():
    generator, provider = _generator(_answer())
    result = _pipeline_result(
        plan=_plan(planner_status="fallback"),
        task_results=[_task_result(results=[_hit()])],
    )

    output = generator.generate(result)

    assert output["status"] == "abstained"
    assert provider.requests == []


def test_evidence_adapter_preserves_text_table_and_optional_metadata():
    table_chunk = _chunk(
        "table-1",
        text="Revenue | 100",
        chunk_type="table",
        table_id="revenue-table",
        table_heading="Revenue",
        row_start=0,
        row_end=1,
    )
    no_metadata_chunk = {"chunk_id": "minimal", "text": "One source passage."}
    provider_answer = _answer(
        claims=[_claim("table-1"), _claim("minimal", "c2")]
    )
    generator, provider = _generator(provider_answer)
    result = _pipeline_result(
        task_results=[_task_result(results=[
            _hit("table-1", chunk=table_chunk),
            _hit("minimal", rank=2, chunk=no_metadata_chunk),
        ])]
    )

    output = generator.generate(result)

    prompt = json.loads(provider.requests[0].user_prompt)
    evidence = prompt["tasks"][0]["evidence"]
    assert evidence[0]["source_metadata"]["table_heading"] == "Revenue"
    assert evidence[0]["source_metadata"]["chunk_type"] == "table"
    assert evidence[1]["source_metadata"] == {}
    assert output["citations"][1]["chunk_id"] == "minimal"
    assert "page" not in output["citations"][1]


def test_hybrid_provenance_is_backend_only_and_enriched_in_citation():
    hybrid_chunk = _chunk(
        "hybrid-1",
        retrieval_score_type="rrf",
        retrieval_path_evidence={
            "chunk_semantic": {"rank": 1, "score": 0.8},
            "keyword_semantic": {"rank": 2, "score": 0.4},
        },
    )
    generator, provider = _generator(_answer(claims=[_claim("hybrid-1")]))
    result = _pipeline_result(
        task_results=[_task_result(results=[
            _hit("hybrid-1", chunk=hybrid_chunk)
        ])],
        retrieval_strategy="hybrid_chunk_keyword",
    )

    output = generator.generate(result)

    prompt = provider.requests[0].user_prompt
    assert "retrieval_path_evidence" not in prompt
    citation = output["citations"][0]
    assert citation["retrieval_provenance"][0]["retrieval_strategy"] == "hybrid_chunk_keyword"
    assert citation["retrieval_provenance"][0]["retrieval_path_evidence"] == (
        hybrid_chunk["retrieval_path_evidence"]
    )
    assert citation["retrieval_provenance"][0]["retrieval_rank"] == 1


def test_same_chunk_under_multiple_tasks_preserves_associations():
    shared = _hit("shared", rank=1)
    generator, provider = _generator(_answer(claims=[_claim("shared")]))
    result = _pipeline_result(
        plan=_plan(tasks=[
            _task("t1", "Infosys revenue", "Infosys"),
            _task("t2", "Infosys income", "Infosys"),
        ]),
        task_results=[
            _task_result("t1", "Infosys revenue", [shared]),
            _task_result("t2", "Infosys income", [shared]),
        ],
    )

    output = generator.generate(result)

    prompt = json.loads(provider.requests[0].user_prompt)
    assert [task["task_id"] for task in prompt["tasks"]] == ["t1", "t2"]
    assert all(len(task["evidence"]) == 1 for task in prompt["tasks"])
    provenance = output["citations"][0]["retrieval_provenance"]
    assert [item["task_id"] for item in provenance] == ["t1", "t2"]


def test_inconsistent_duplicate_source_records_are_rejected_before_generation():
    generator, provider = _generator(_answer())
    first = _hit("shared", chunk=_chunk("shared"))
    altered = _hit("shared", chunk=_chunk("shared", page=99))
    result = _pipeline_result(
        plan=_plan(tasks=[
            _task("t1", "Infosys revenue"),
            _task("t2", "Infosys income"),
        ]),
        task_results=[
            _task_result("t1", "Infosys revenue", [first]),
            _task_result("t2", "Infosys income", [altered]),
        ],
    )

    with pytest.raises(answer_generation.AnswerGenerationError, match="inconsistent"):
        generator.generate(result)

    assert provider.requests == []


def test_partial_response_is_validated_and_contains_limitations():
    partial = _answer(
        status="partial",
        claims=[_claim()],
        limitations=["No evidence was retrieved for the requested prior period."],
    )
    generator, provider = _generator(partial)
    result = _pipeline_result(
        plan=_plan(tasks=[
            _task("t1", "Infosys revenue"),
            _task("t2", "Infosys revenue prior year"),
        ]),
        task_results=[
            _task_result("t1", "Infosys revenue", [_hit()]),
            _task_result("t2", "Infosys revenue prior year", []),
        ],
    )

    output = generator.generate(result)

    assert output["status"] == "partial"
    assert output["limitations"]
    assert len(provider.requests) == 1


def test_incomplete_retrieval_cannot_be_marked_answered():
    generator, _ = _generator(_answer())
    result = _pipeline_result(
        plan=_plan(tasks=[
            _task("t1", "Infosys revenue"),
            _task("t2", "Infosys operating margin"),
        ]),
        task_results=[
            _task_result("t1", "Infosys revenue", [_hit()]),
            _task_result("t2", "Infosys operating margin", []),
        ],
    )

    with pytest.raises(LLMProviderResponseError, match="incomplete retrieval"):
        generator.generate(result)


def test_general_answer_without_citations_is_valid():
    answer = _answer(
        mode="general_knowledge",
        claims=[{
            "claim_id": "c1",
            "text": "Machine learning is a field of artificial intelligence.",
            "claim_type": "qualitative_fact",
            "citation_chunk_ids": [],
        }],
    )
    generator, _ = _generator(answer)
    result = _pipeline_result(
        question="What is machine learning?",
        plan=_plan(tasks=[], intents=[query_planner.Intent.out_of_scope]),
        status="out_of_scope",
    )

    output = generator.generate(result)

    assert output["status"] == "answered"
    assert output["claims"][0]["citation_chunk_ids"] == []
    assert output["citations"] == []


def test_abstained_provider_answer_must_have_no_claims_and_a_limitation():
    response = _answer(
        status="abstained",
        claims=[],
        limitations=["The requested figure was not found."],
        text="I cannot establish the requested figure.",
    )
    generator, _ = _generator(response)

    output = generator.generate(_pipeline_result(
        task_results=[_task_result(results=[_hit()])]
    ))

    assert output["status"] == "abstained"
    assert output["claims"] == []
    assert output["citations"] == []


def test_provider_errors_propagate_without_unstructured_fallback():
    generator, provider = _generator(
        error=LLMProviderCapabilityError("structured output is unavailable")
    )

    with pytest.raises(LLMProviderCapabilityError):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))

    assert len(provider.requests) == 1


def test_invalid_structured_json_is_rejected():
    generator, _ = _generator("not-json")

    with pytest.raises(LLMProviderResponseError, match="malformed JSON"):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))


def test_unknown_citation_id_is_rejected():
    generator, _ = _generator(_answer(claims=[_claim("not-retrieved")]))

    with pytest.raises(LLMProviderResponseError, match="not supplied"):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))


def test_duplicate_claim_ids_are_rejected():
    claims = [_claim(claim_id="duplicate"), _claim(claim_id="duplicate")]
    generator, _ = _generator(_answer(claims=claims))

    with pytest.raises(LLMProviderResponseError, match="duplicate claim"):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))


def test_duplicate_citation_ids_are_rejected_by_schema():
    generator, _ = _generator(_answer(claims=[{
        **_claim(),
        "citation_chunk_ids": ["chunk-1", "chunk-1"],
    }]))

    with pytest.raises(LLMProviderResponseError, match="does not match"):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))


def test_corpus_claim_without_citations_is_rejected_by_schema():
    generator, _ = _generator(_answer(claims=[{
        **_claim(),
        "citation_chunk_ids": [],
    }]))

    with pytest.raises(LLMProviderResponseError, match="does not match"):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))


def test_model_cannot_supply_fabricated_citation_metadata():
    claim = _claim()
    claim["company"] = "Fabricated Company"
    generator, _ = _generator(_answer(claims=[claim]))

    with pytest.raises(LLMProviderResponseError, match="does not match"):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))


def test_model_cannot_supply_top_level_citation_metadata():
    response = _answer()
    response["citations"] = [{"chunk_id": "forged"}]
    generator, _ = _generator(response)

    with pytest.raises(LLMProviderResponseError, match="does not match"):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))


def test_citation_metadata_comes_only_from_original_retrieval_result():
    generator, _ = _generator(_answer(claims=[_claim()]))
    result = _pipeline_result(
        task_results=[_task_result(results=[
            _hit(chunk=_chunk(page=33, company="Original Company"))
        ])]
    )

    output = generator.generate(result)

    citation = output["citations"][0]
    assert citation["company"] == "Original Company"
    assert citation["page"] == 33
    assert citation["document"] == "Annual Report"
    assert citation["financial_year"] == "2024-2025"


@pytest.mark.parametrize(
    ("mode", "status", "claims", "limitations"),
    [
        ("corpus_grounded", "abstained", [_claim()], ["No support."]),
        ("corpus_grounded", "partial", [_claim()], []),
        ("general_knowledge", "partial", [_claim()], []),
    ],
)
def test_invalid_status_mode_combinations_are_rejected(mode, status, claims, limitations):
    answer = _answer(
        mode=mode,
        status=status,
        claims=claims,
        limitations=limitations,
    )
    if mode == "general_knowledge":
        answer["claims"] = [{
            **claims[0],
            "citation_chunk_ids": [],
        }]
    generator, _ = _generator(answer)

    with pytest.raises(LLMProviderResponseError):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))


@pytest.mark.parametrize(
    ("mode", "status"),
    [
        ("corpus_grounded", "answered"),
        ("corpus_grounded", "partial"),
        ("corpus_grounded", "abstained"),
        ("general_knowledge", "answered"),
        ("general_knowledge", "partial"),
        ("general_knowledge", "abstained"),
        ("current_information_unavailable", "abstained"),
    ],
)
def test_every_supported_mode_status_combination_is_schema_valid(mode, status):
    claim = _claim() if mode == "corpus_grounded" else {
        "claim_id": "c1",
        "text": "A general-knowledge fact.",
        "claim_type": "qualitative_fact",
        "citation_chunk_ids": [],
    }
    answer = _answer(
        mode=mode,
        status=status,
        claims=[] if status == "abstained" else [claim],
        limitations=["A limitation."] if status != "answered" else [],
    )

    validate_json_output(
        json.dumps(answer),
        json_mode=True,
        response_schema=answer_generation.ANSWER_RESPONSE_SCHEMA,
    )


@pytest.mark.parametrize(
    "answer",
    [
        _answer(mode="current_information_unavailable", status="answered"),
        _answer(mode="current_information_unavailable", status="partial"),
        _answer(
            mode="corpus_grounded",
            status="partial",
            limitations=[],
        ),
        _answer(
            mode="corpus_grounded",
            status="abstained",
            claims=[],
            limitations=[],
        ),
        _answer(
            mode="general_knowledge",
            status="partial",
            claims=[{
                "claim_id": "c1",
                "text": "A general fact.",
                "claim_type": "qualitative_fact",
                "citation_chunk_ids": [],
            }],
            limitations=[],
        ),
        _answer(
            mode="general_knowledge",
            status="answered",
            claims=[_claim()],
        ),
    ],
)
def test_unsupported_mode_status_and_missing_limitation_responses_fail_schema(answer):
    with pytest.raises(LLMProviderResponseError, match="does not match"):
        validate_json_output(
            json.dumps(answer),
            json_mode=True,
            response_schema=answer_generation.ANSWER_RESPONSE_SCHEMA,
        )


def test_empty_answer_and_empty_answered_claims_fail_schema():
    for answer in (
        _answer(text=""),
        _answer(claims=[]),
        _answer(claims=[_claim(text="")]),
    ):
        with pytest.raises(LLMProviderResponseError, match="does not match"):
            validate_json_output(
                json.dumps(answer),
                json_mode=True,
                response_schema=answer_generation.ANSWER_RESPONSE_SCHEMA,
            )


def test_provider_mode_mismatch_is_rejected():
    generator, _ = _generator(_answer(mode="general_knowledge"))

    with pytest.raises(LLMProviderResponseError):
        generator.generate(_pipeline_result(
            task_results=[_task_result(results=[_hit()])]
        ))


def test_source_id_mismatch_and_missing_source_text_are_rejected():
    generator, provider = _generator(_answer())
    bad_id_result = _pipeline_result(
        task_results=[_task_result(results=[
            _hit("chunk-1", chunk=_chunk("different-id"))
        ])]
    )
    with pytest.raises(answer_generation.AnswerGenerationError, match="does not match"):
        generator.generate(bad_id_result)
    assert provider.requests == []

    missing_text_result = _pipeline_result(
        task_results=[_task_result(results=[
            _hit("chunk-1", chunk={"chunk_id": "chunk-1"})
        ])]
    )
    with pytest.raises(answer_generation.AnswerGenerationError, match="no original text"):
        generator.generate(missing_text_result)


def test_query_planner_clarification_without_financial_intent_routes_general():
    generator, provider = _generator(_answer(
        mode="general_knowledge",
        claims=[{
            "claim_id": "c1",
            "text": "Hello! I can help with general questions.",
            "claim_type": "qualitative_fact",
            "citation_chunk_ids": [],
        }],
        text="Hello! I can help with general questions.",
    ))
    result = _pipeline_result(
        question="Hello",
        plan=_plan(
            tasks=[],
            intents=[query_planner.Intent.out_of_scope],
            needs_clarification=True,
            clarification_question="Could you clarify?",
        ),
        status="out_of_scope",
    )

    output = generator.generate(result)

    assert output["answer_mode"] == "general_knowledge"
    assert provider.requests
