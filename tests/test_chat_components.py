from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from src import query_planner
from ui import dashboard
from ui.chat_components import (
    _metadata_value,
    evidence_for_citations,
    friendly_chat_error,
    suggested_questions,
)


def test_suggested_questions_use_selected_company():
    questions = suggested_questions("Infosys")

    assert len(questions) == 5
    assert all("Infosys" in question for question in questions)
    assert all("the company" not in question for question in questions)


def test_suggested_questions_without_company_stay_generic():
    assert all(
        "the company" in question
        for question in suggested_questions(None)
    )


def test_evidence_for_citations_matches_only_backend_citation_chunk_ids():
    hit = SimpleNamespace(
        chunk_id="chunk-1",
        chunk={"text": "Retrieved report passage."},
    )
    result = SimpleNamespace(
        task_results=[SimpleNamespace(results=[hit])]
    )
    citations = [
        {"chunk_id": "chunk-1", "document": "Annual Report"},
        {"chunk_id": "missing-chunk"},
    ]

    evidence = evidence_for_citations(result, citations)

    assert evidence == [
        {
            "citation": citations[0],
            "text": "Retrieved report passage.",
        },
        {"citation": citations[1], "text": ""},
    ]


def test_metadata_without_a_value_is_not_available():
    assert _metadata_value(None) == "Not available"
    assert _metadata_value("  ") == "Not available"


def test_planner_configuration_error_is_user_friendly_and_does_not_expose_details():
    error = query_planner.PlannerConfigError("sensitive provider detail")

    message = friendly_chat_error(error)

    assert "Gemini credentials" in message
    assert "sensitive provider detail" not in message


def test_unexpected_error_is_logged_elsewhere_without_echoing_details_to_ui():
    error = RuntimeError("sensitive provider detail")

    message = friendly_chat_error(error)

    assert "application error" in message
    assert "sensitive provider detail" not in message


def test_suggested_question_button_submits_selected_company_question():
    app = """
import streamlit as st
from ui.chat_components import render_suggested_questions
render_suggested_questions(
    "TCS",
    lambda question: st.session_state.update({
        "submitted_question": question,
        "submission_count": st.session_state.get("submission_count", 0) + 1,
    }),
)
"""
    at = AppTest.from_string(app).run()

    at.button[0].click().run()

    assert at.session_state["submitted_question"] == (
        "What was TCS's revenue growth over the reported financial years?"
    )
    assert at.session_state["submission_count"] == 1


def test_suggested_question_runs_chat_workflow_and_renders_associated_evidence(
    monkeypatch,
):
    retrieved_text = "Retrieved passage for the selected company."
    hit = SimpleNamespace(
        chunk_id="chunk-42",
        chunk={"text": retrieved_text},
    )
    pipeline_result = SimpleNamespace(
        task_results=[SimpleNamespace(results=[hit])],
        warnings=["A retrieval warning."],
        status="partial",
    )

    class FakePipeline:
        def __init__(self):
            self.calls = []

        def run(self, question, *, company):
            self.calls.append((question, company))
            return pipeline_result

    class FakeAnswerGenerator:
        def generate(self, result):
            assert result is pipeline_result
            return {
                "answer": "Generated response for the suggested question.",
                "citations": [{"chunk_id": "chunk-42", "document": "Annual Report"}],
                "status": "answered",
            }

    pipeline = FakePipeline()
    monkeypatch.setattr(
        dashboard,
        "_cached_chat_services",
        lambda: (pipeline, FakeAnswerGenerator()),
    )
    app = """
import streamlit as st
from ui import dashboard
dashboard.configure_session_state()
st.session_state.selected_company = "TCS"
dashboard.render_chat()
"""
    at = AppTest.from_string(app).run()

    at.button[0].click().run()

    question = "What was TCS's revenue growth over the reported financial years?"
    assert not at.exception
    assert pipeline.calls == [(question, "TataConsultancyServices")]
    history = at.session_state["chat_history"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[0]["content"] == question
    assert history[1]["content"] == "Generated response for the suggested question."
    assert history[1]["warnings"] == ["A retrieval warning."]
    assert history[1]["retrieval_status"] == "partial"
    assert history[1]["answer_status"] == "answered"
    assert history[1]["evidence"] == [{
        "citation": {"chunk_id": "chunk-42", "document": "Annual Report"},
        "text": retrieved_text,
    }]
    assert "Generated response for the suggested question." in [
        item.value for item in at.markdown
    ]


def test_assistant_message_renders_backend_status_warnings_and_citations():
    app = """
from ui.chat_components import render_chat_message
render_chat_message({
    "role": "assistant",
    "content": "**Revenue** was reported.",
    "retrieval_status": "partial",
    "answer_status": "abstained",
    "warnings": ["Task retrieval failed."],
    "evidence": [{
        "citation": {
            "chunk_id": "chunk-42",
            "company": "Infosys",
            "document": "Annual Report",
            "financial_year": "2024-2025",
            "page": 33,
        },
        "text": "Exact retrieved annual-report passage.",
    }],
})
"""
    at = AppTest.from_string(app).run()

    assert not at.exception
    rendered = [element.value for element in at.markdown]
    assert "**Revenue** was reported." in rendered
    assert any("Retrieval" in item and "partial" in item for item in rendered)
    assert any("Answer" in item and "abstained" in item for item in rendered)
    assert "Exact retrieved annual-report passage." in rendered
    assert "chunk-42" in " ".join(element.value for element in at.caption)
    assert [element.value for element in at.warning] == ["Task retrieval failed."]


def test_submit_question_keeps_backend_status_and_answer_citations(monkeypatch):
    pipeline_result = SimpleNamespace(
        task_results=[],
        warnings=["Task retrieval failed."],
        status="partial",
    )

    class FakePipeline:
        def run(self, question, *, company):
            assert question == "What did Infosys disclose?"
            assert company == "Infosys"
            return pipeline_result

    class FakeAnswerGenerator:
        def generate(self, result):
            assert result is pipeline_result
            return {
                "answer": "The answer is unavailable from the retrieved evidence.",
                "citations": [{"chunk_id": "chunk-1"}],
                "status": "abstained",
            }

    monkeypatch.setattr(
        dashboard,
        "_cached_chat_services",
        lambda: (FakePipeline(), FakeAnswerGenerator()),
    )
    app = """
import streamlit as st
from ui.dashboard import _submit_research_question
st.session_state.setdefault("chat_history", [])
_submit_research_question("What did Infosys disclose?", "Infosys")
"""
    at = AppTest.from_string(app).run()

    assert not at.exception
    history = at.session_state["chat_history"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[1]["retrieval_status"] == "partial"
    assert history[1]["answer_status"] == "abstained"
    assert history[1]["warnings"] == ["Task retrieval failed."]
    assert history[1]["citations"] == [{"chunk_id": "chunk-1"}]
    assert history[1]["evidence"] == [{"citation": {"chunk_id": "chunk-1"}, "text": ""}]


def test_submit_question_shows_safe_error_when_planner_credentials_are_missing(monkeypatch):
    def missing_planner_credentials():
        raise query_planner.PlannerConfigError("sensitive provider detail")

    monkeypatch.setattr(
        dashboard,
        "_cached_chat_services",
        missing_planner_credentials,
    )
    app = """
import streamlit as st
from ui.dashboard import _submit_research_question
st.session_state.setdefault("chat_history", [])
_submit_research_question("What did Infosys disclose?", "Infosys")
"""
    at = AppTest.from_string(app).run()

    assert not at.exception
    assistant_message = at.session_state["chat_history"][-1]
    assert assistant_message["error"] is True
    assert "Gemini credentials" in assistant_message["content"]
    assert "sensitive provider detail" not in assistant_message["content"]


def test_clear_conversation_requires_confirmation():
    app = """
import streamlit as st
from ui.dashboard import _render_clear_conversation
st.session_state.setdefault("chat_history", [{"role": "user", "content": "Question"}])
_render_clear_conversation()
"""
    at = AppTest.from_string(app).run()

    at.button[0].click().run()
    assert at.session_state["confirm_clear_chat"] is True
    assert at.session_state["chat_history"]

    at.button[0].click().run()
    assert at.session_state["confirm_clear_chat"] is False
    assert at.session_state["chat_history"] == []
