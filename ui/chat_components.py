"""Reusable presentation and validation helpers for research chat."""

import html
from collections.abc import Callable
from typing import Any

import streamlit as st

from src import answer_generation, query_planner, retrieval_pipeline
from src.llm.base import LLMProviderError
from src.llm.settings import LLMConfigurationError
from src.retrieval import ChromaRetrievalError


_SUGGESTED_QUESTIONS = (
    "What was {company}'s revenue growth over the reported financial years?",
    "How has {company}'s operating margin changed?",
    "What major business risks does {company} disclose in its annual reports?",
    "What do {company}'s annual reports say about capital expenditure?",
    "How has {company}'s debt position changed?",
)


def suggested_questions(company: str | None) -> list[str]:
    company_label = company or "the company"
    return [question.format(company=company_label) for question in _SUGGESTED_QUESTIONS]


def evidence_for_citations(
    result: Any,
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    retrieved_chunks = {
        item.chunk_id: item.chunk
        for task_result in result.task_results
        for item in task_result.results
    }
    return [
        {
            "citation": citation,
            "text": retrieved_chunks.get(citation.get("chunk_id"), {}).get("text", ""),
        }
        for citation in citations
    ]


def friendly_chat_error(error: Exception) -> str:
    if isinstance(error, query_planner.PlannerConfigError):
        return "The query planner is not configured. Set up Gemini credentials in your local environment."
    if isinstance(error, LLMConfigurationError):
        return "The answer-generation provider is not configured. Set up its credentials in your local environment."
    if isinstance(error, query_planner.PlannerInputError):
        return "This question could not be accepted. Keep it within the 1,000-character limit."
    if isinstance(error, query_planner.QueryPlannerError):
        return "The query planner could not complete this request. Check its configuration and try again."
    if isinstance(error, (ChromaRetrievalError, retrieval_pipeline.RetrievalPipelineError)):
        return "Report evidence could not be retrieved. Check the local retrieval index and try again."
    if isinstance(error, (answer_generation.AnswerGenerationError, LLMProviderError)):
        return "Answer generation could not complete. Check the selected provider and try again."
    if isinstance(error, OSError):
        return "A required local corpus or retrieval-index file is unavailable."
    if isinstance(error, ValueError):
        return "The request could not be validated. Check the question and try again."
    return "An unexpected application error occurred. Check the application logs."


def _metadata_value(value: Any) -> str:
    if value is None or not str(value).strip():
        return "Not available"
    return str(value)


def _status_tone(value: Any) -> str:
    status = str(value).casefold()
    if status in {"completed", "answered"}:
        return "success"
    if status in {"partial", "abstained", "needs_clarification", "no_tasks"}:
        return "warning"
    if status == "failed":
        return "error"
    return "neutral"


def render_citation_card(item: dict[str, Any], index: int) -> None:
    citation = item["citation"]
    company = _metadata_value(citation.get("company"))
    document = _metadata_value(citation.get("document"))
    financial_year = _metadata_value(citation.get("financial_year"))
    page = _metadata_value(citation.get("page"))
    chunk_id = _metadata_value(citation.get("chunk_id"))

    with st.container(border=True):
        st.markdown(f"**Source {index}**")
        company_column, document_column, year_column, page_column = st.columns(
            [1, 2, 1, 0.7]
        )
        with company_column:
            st.caption("COMPANY")
            st.write(company)
        with document_column:
            st.caption("DOCUMENT")
            st.write(document)
        with year_column:
            st.caption("FINANCIAL YEAR")
            st.write(financial_year)
        with page_column:
            st.caption("PAGE")
            st.write(page)
        st.caption(f"Chunk ID · `{chunk_id}`")
        with st.expander("View retrieved passage"):
            passage = item.get("text")
            if isinstance(passage, str) and passage:
                st.markdown(passage)
            else:
                st.info("The retrieved passage is not available in this session.")


def render_chat_message(message: dict[str, Any]) -> None:
    role = message["role"]
    with st.chat_message(role):
        if role == "user":
            st.markdown(message["content"])
            return

        if message.get("error"):
            st.error(message["content"])
            return

        st.markdown("#### Answer")
        st.markdown(message["content"])

        retrieval_status = html.escape(
            _metadata_value(message.get("retrieval_status"))
        )
        answer_status = html.escape(
            _metadata_value(message.get("answer_status"))
        )
        retrieval_tone = _status_tone(message.get("retrieval_status"))
        answer_tone = _status_tone(message.get("answer_status"))
        st.markdown(
            (
                '<div class="chat-status-row">'
                f'<span class="chat-status status-{retrieval_tone}">'
                f"Retrieval · {retrieval_status}</span>"
                f'<span class="chat-status status-{answer_tone}">'
                f"Answer · {answer_status}</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        warnings = message.get("warnings", [])
        if warnings:
            with st.expander(f"Retrieval warnings ({len(warnings)})"):
                for warning in warnings:
                    st.warning(warning)

        evidence = message.get("evidence", [])
        with st.expander(f"Sources & retrieved evidence ({len(evidence)})"):
            if evidence:
                for index, item in enumerate(evidence, start=1):
                    render_citation_card(item, index)
            else:
                st.info("No citations were returned for this answer.")


def render_suggested_questions(
    company: str | None,
    on_submit: Callable[[str], None],
) -> None:
    st.markdown("### Suggested research questions")
    st.caption("Examples only — answers depend on retrieved report evidence.")
    suggestions = suggested_questions(company)
    for row_start in range(0, len(suggestions), 2):
        columns = st.columns(2)
        for offset, (column, question) in enumerate(
            zip(columns, suggestions[row_start : row_start + 2])
        ):
            with column:
                if st.button(
                    question,
                    key=f"suggested_question_{row_start + offset}",
                    use_container_width=True,
                ):
                    on_submit(question)
                    st.rerun()
