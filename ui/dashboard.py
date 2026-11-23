"""Pages and shared components for the Financial RAG Streamlit dashboard."""

import html
import logging
from pathlib import Path

import streamlit as st

from src import answer_generation, query_planner, retrieval_pipeline
from src.llm.settings import LLMConfigurationError, load_settings
from src.retrieval import ChromaRetriever

from . import chat_components, company_analysis, evidence_explorer
from . import evaluation_dashboard
from .dashboard_data import (
    CHUNKS_FILE,
    COMPANIES,
    PROJECT_ROOT,
    OverviewData,
    load_overview_data,
)

logger = logging.getLogger(__name__)

COMPANY_OPTIONS = ("All companies", *(display for display, _ in COMPANIES))
COMPANY_CANONICAL_NAMES = {
    display: canonical for display, canonical in COMPANIES
}

_STYLE = """
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1220; color: #e7edf5; }
[data-testid="stAppViewContainer"] { background: #0b1220; }
[data-testid="stSidebar"] {
    background: #0d1625;
    border-right: 1px solid #1d2b3b;
}
[data-testid="stHeader"] {
    background: #0b1220;
}
[data-testid="stToolbar"] {
    background: #0b1220;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #93a4b8;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] {
    border-radius: 8px;
}
.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 { color: #edf3fa; letter-spacing: -0.02em; }
[data-testid="stMetric"] {
    background: #111d2c;
    border: 1px solid #1c2b3b;
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 124px;
}
[data-testid="stMetricLabel"] p { color: #9aabbe; font-size: 0.88rem; }
[data-testid="stMetricValue"] { color: #eaf3fa; }
[data-testid="stMetricDelta"] { color: #51d6c2; }
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #101b2a;
    border-color: #1d2b3b;
    border-radius: 12px;
}
div[data-testid="stForm"] {
    background: #101b2a;
    border: 1px solid #1d2b3b;
    border-radius: 12px;
    padding: 1rem;
}
.stTextArea textarea,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #101b2a !important;
    color: #e7edf5 !important;
    border-color: #293b4e !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #101b2a;
}
[data-baseweb="menu"] li {
    color: #e7edf5;
}
button[kind="primary"] {
    background: #168f84;
    border-color: #168f84;
}
.brand-mark {
    color: #56d8c6;
    font-weight: 750;
    font-size: 1.25rem;
    letter-spacing: 0.02em;
}
.brand-subtitle {
    color: #74879c;
    font-size: 0.68rem;
    letter-spacing: 0.13em;
    margin: 0.2rem 0 1.2rem;
}
.eyebrow {
    color: #59d5c3;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.page-description { color: #9aabbe; margin-top: -0.65rem; }
.company-card {
    height: 100%;
    background: #111d2c;
    border: 1px solid #1c2b3b;
    border-top: 2px solid #31b9aa;
    border-radius: 12px;
    padding: 1.15rem;
}
.company-card h3 { margin: 0.35rem 0; font-size: 1.06rem; }
.company-card p { color: #9aabbe; margin: 0.35rem 0 0; }
.company-count {
    color: #59d5c3;
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 1.1rem;
}
.sidebar-footer {
    border-top: 1px solid #1d2b3b;
    margin-top: 1.5rem;
    padding-top: 0.8rem;
    color: #718399;
    font-size: 0.72rem;
}
.subtle { color: #91a2b5; font-size: 0.88rem; }
[data-testid="stChatMessage"] {
    background: #101b2a;
    border: 1px solid #1d2b3b;
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-bottom: 1rem;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    line-height: 1.7;
}
[data-testid="stChatInput"] {
    background: #101b2a;
    border: 1px solid #2a4053;
    border-radius: 14px;
}
[data-testid="stChatInput"] textarea {
    color: #e7edf5;
}
.chat-status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.8rem 0 0.5rem;
}
.chat-status {
    display: inline-block;
    color: #b6c7d8;
    background: #162536;
    border: 1px solid #2a4053;
    border-radius: 999px;
    padding: 0.25rem 0.65rem;
    font-size: 0.76rem;
}
.chat-status.status-success { color: #69decf; border-color: #267c72; }
.chat-status.status-warning { color: #f0c46b; border-color: #80652d; }
.chat-status.status-error { color: #ff8e8e; border-color: #854444; }
.citation-label {
    color: #61d4c3;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
}
[data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: #0d1724;
}
@media (max-width: 700px) {
    .block-container { padding: 1rem 1rem 2rem; }
    [data-testid="stMetric"] { min-height: 110px; padding: 14px; }
    [data-testid="stChatMessage"] { padding: 0.75rem; }
}
@media (max-width: 1100px) {
    h1 { font-size: 2rem !important; }
}
</style>
"""


def configure_session_state() -> None:
    if "selected_company" not in st.session_state:
        st.session_state.selected_company = "All companies"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def inject_styles() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)


def render_page_header(
    title: str,
    description: str,
    *,
    company_selector_key: str | None = None,
) -> str | None:
    title_column, selector_column = (
        st.columns([3, 1])
        if company_selector_key is not None
        else (st.container(), None)
    )
    with title_column:
        st.markdown(f'<div class="eyebrow">Investor research workspace</div>', unsafe_allow_html=True)
        st.title(title)
        st.markdown(f'<p class="page-description">{html.escape(description)}</p>', unsafe_allow_html=True)

    if selector_column is None:
        return None
    if company_selector_key not in st.session_state:
        st.session_state[company_selector_key] = st.session_state.selected_company
    with selector_column:
        selected = st.selectbox(
            "Company",
            COMPANY_OPTIONS,
            key=company_selector_key,
        )
    st.session_state.selected_company = selected
    return COMPANY_CANONICAL_NAMES.get(selected)


def _display_value(value: int | None) -> str:
    return f"{value:,}" if value is not None else "Unavailable"


@st.cache_data(show_spinner=False)
def _cached_overview_data(project_root: str) -> OverviewData:
    return load_overview_data(Path(project_root))


@st.cache_data(show_spinner=False)
def _cached_company_analysis_data(
    project_root: str,
    company: str,
) -> company_analysis.CompanyAnalysisData:
    return company_analysis.load_company_analysis(Path(project_root), company)


@st.cache_data(show_spinner=False)
def _cached_evidence_corpus(project_root: str) -> evidence_explorer.EvidenceCorpus:
    return evidence_explorer.load_evidence_corpus(Path(project_root))


@st.cache_data(show_spinner=False)
def _cached_evaluation_data(project_root: str) -> evaluation_dashboard.EvaluationData:
    return evaluation_dashboard.load_evaluation_data(Path(project_root))


def _render_structured_evidence_table(rows: list[list[str]]) -> None:
    if not rows:
        return
    displayed_rows = rows[:101]
    header, *body = displayed_rows
    table_html = ["<table><thead><tr>"]
    table_html.extend(f"<th>{html.escape(str(cell))}</th>" for cell in header)
    table_html.append("</tr></thead><tbody>")
    for row in body:
        table_html.append("<tr>")
        table_html.extend(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        table_html.append("</tr>")
    table_html.append("</tbody></table>")
    st.markdown(
        '<div style="overflow-x:auto">' + "".join(table_html) + "</div>",
        unsafe_allow_html=True,
    )
    if len(rows) > len(displayed_rows):
        st.caption(
            f"Showing the first {len(body)} data rows of {len(rows) - 1}."
        )


def _render_kpis(data: OverviewData) -> None:
    columns = st.columns(4)
    items = (
        ("Companies covered", data.companies_covered, "From available annual-report sources"),
        ("Benchmark questions", data.benchmark_questions, "Loaded from the project benchmark"),
        ("Retrieval strategies", data.retrieval_strategies, "Configured evaluation strategies"),
        ("Corpus documents", data.corpus_documents, f"{_display_value(data.corpus_chunks)} indexed chunks"),
    )
    for column, (label, value, caption) in zip(columns, items):
        with column:
            st.metric(label, _display_value(value))
            st.caption(caption)


def _render_company_cards(data: OverviewData) -> None:
    st.subheader("Company coverage")
    columns = st.columns(3)
    for column, (display_name, canonical_name) in zip(columns, COMPANIES):
        document_count = (
            data.company_documents.get(canonical_name)
            if data.company_documents is not None
            else None
        )
        count_label = (
            f"{document_count:,} source documents"
            if document_count is not None
            else "Document count unavailable"
        )
        with column:
            st.markdown(
                (
                    '<div class="company-card">'
                    '<div class="eyebrow">Annual report corpus</div>'
                    f"<h3>{html.escape(display_name)}</h3>"
                    '<p>Financial disclosures available for research</p>'
                    f'<div class="company-count">{html.escape(count_label)}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_overview() -> None:
    render_page_header(
        "Overview",
        "A grounded workspace for research across Indian corporate disclosures.",
    )
    data = _cached_overview_data(str(PROJECT_ROOT))
    _render_kpis(data)
    st.divider()
    _render_company_cards(data)

    st.divider()
    activity_column, status_column = st.columns([1.2, 1])
    with activity_column:
        st.subheader("Recent research activity")
        st.info("Research activity is not persisted by this project. Chat history is kept for this browser session.")
    with status_column:
        st.subheader("Research platform")
        st.markdown(
            "Search annual-report disclosures, review retrieved evidence, and explore "
            "evaluation outputs. Answers are designed to remain grounded in cited sources."
        )
        index_status = "Available" if data.corpus_chunks is not None else "Unavailable"
        st.caption(
            f"Retrieval index: {index_status} · "
            f"{_display_value(data.corpus_chunks)} chunks"
        )


def _get_chat_services() -> tuple[
    retrieval_pipeline.FinancialRetrievalPipeline,
    answer_generation.FinancialAnswerGenerator,
]:
    registry = query_planner.registry_from_chunks_file(CHUNKS_FILE)
    planner = query_planner.QueryPlanner(registry)
    retriever = ChromaRetriever()
    pipeline = retrieval_pipeline.FinancialRetrievalPipeline(planner, retriever)
    return pipeline, answer_generation.FinancialAnswerGenerator()


@st.cache_resource(show_spinner=False)
def _cached_chat_services() -> tuple[
    retrieval_pipeline.FinancialRetrievalPipeline,
    answer_generation.FinancialAnswerGenerator,
]:
    return _get_chat_services()


def _submit_research_question(question: str, selected_company: str | None) -> None:
    history = st.session_state.chat_history
    if st.session_state.get("chat_request_in_progress"):
        return
    st.session_state.chat_request_in_progress = True
    history.append({"role": "user", "content": question})
    try:
        pipeline, answer_generator = _cached_chat_services()
        with st.spinner("Planning the search and reviewing report evidence..."):
            result = pipeline.run(question, company=selected_company)
            answer = answer_generator.generate(result)
    except Exception as error:
        logger.exception("Financial research request failed")
        history.append({
            "role": "assistant",
            "content": chat_components.friendly_chat_error(error),
            "error": True,
        })
    else:
        citations = answer.get("citations", [])
        history.append({
            "role": "assistant",
            "content": answer["answer"],
            "citations": citations,
            "evidence": chat_components.evidence_for_citations(result, citations),
            "warnings": result.warnings,
            "retrieval_status": result.status,
            "answer_status": answer.get("status", "unavailable"),
        })
    finally:
        st.session_state.chat_request_in_progress = False


def _render_clear_conversation() -> None:
    if not st.session_state.chat_history:
        return
    if st.session_state.get("confirm_clear_chat"):
        st.warning("Clear all messages and cited evidence from this browser session?")
        confirm_column, cancel_column = st.columns(2)
        with confirm_column:
            if st.button(
                "Confirm clear",
                type="primary",
                key="confirm_clear_chat_button",
                use_container_width=True,
            ):
                st.session_state.chat_history.clear()
                st.session_state.confirm_clear_chat = False
                st.rerun()
        with cancel_column:
            if st.button(
                "Keep conversation",
                key="cancel_clear_chat_button",
                use_container_width=True,
            ):
                st.session_state.confirm_clear_chat = False
                st.rerun()
    elif st.button("Clear conversation", key="clear_chat_button"):
        st.session_state.confirm_clear_chat = True
        st.rerun()


def render_chat() -> None:
    selected_company = render_page_header(
        "Financial Research Chat",
        "Ask about company disclosures and review the evidence behind every citation.",
        company_selector_key="chat_company_selector",
    )
    st.caption(
        "Responses use the existing planner, retrieval pipeline, and answer generator. "
        "Provider credentials are required; no mock answers are shown."
    )
    if not st.session_state.chat_history:
        chat_components.render_suggested_questions(
            st.session_state.chat_company_selector
            if selected_company
            else None,
            lambda question: _submit_research_question(question, selected_company),
        )
    for message in st.session_state.chat_history:
        chat_components.render_chat_message(message)

    _render_clear_conversation()

    question = st.chat_input(
        "Ask a financial question about TCS, Infosys, or Reliance Industries...",
        key="research_chat_input",
        max_chars=1000,
        disabled=st.session_state.get("chat_request_in_progress", False),
    )
    if question is not None:
        if not question.strip():
            st.warning("Enter a question before submitting.")
        else:
            _submit_research_question(question.strip(), selected_company)
            st.rerun()


def render_company_analysis() -> None:
    canonical_name = render_page_header(
        "Company Analysis",
        "A focused view of report coverage and company-level financial disclosures.",
        company_selector_key="analysis_company_selector",
    )
    if canonical_name is None:
        st.info("Select TCS, Infosys, or Reliance Industries to view report analysis.")
        return

    with st.spinner("Loading annual-report tables and cited evidence..."):
        data = _cached_company_analysis_data(str(PROJECT_ROOT), canonical_name)
    if not data.available_years:
        st.info(data.unavailable_reason or "No annual-report data is available.")
        return

    company_label = next(
        display for display, canonical in COMPANIES if canonical == canonical_name
    )
    company_slug = "".join(
        character.lower()
        for character in canonical_name
        if character.isalnum()
    )
    year_key = f"analysis_financial_year_{company_slug}"
    saved_years = st.session_state.setdefault("analysis_financial_years", {})
    if year_key not in st.session_state:
        saved_year = saved_years.get(company_slug)
        st.session_state[year_key] = (
            saved_year if saved_year in data.available_years else data.available_years[0]
        )
    selected_year = st.selectbox(
        "Financial year",
        data.available_years,
        key=year_key,
    )
    saved_years[company_slug] = selected_year
    report = data.report_for_year(selected_year)
    if report is None:
        st.info("No processed annual report is available for the selected year.")
        return

    st.subheader(company_label)
    st.caption(f"Financial year {report.financial_year}")
    report_columns = st.columns(3)
    with report_columns[0]:
        st.metric("Annual report", report.source_file or "Not available")
    with report_columns[1]:
        st.metric(
            "Report pages",
            _display_value(report.number_of_pages),
        )
    with report_columns[2]:
        st.metric("Extracted tables", f"{report.table_count:,}")

    operating_metric = (
        report.metrics.get("operating_profit")
        or report.metrics.get("operating_margin")
    )
    kpis = (
        ("revenue", report.metrics.get("revenue")),
        ("net_profit", report.metrics.get("net_profit")),
        ("operating", operating_metric),
        ("total_assets", report.metrics.get("total_assets")),
        ("total_borrowings", report.metrics.get("total_borrowings")),
    )
    st.subheader("Financial snapshot")
    st.caption(
        "Values are shown only when an exact consolidated-statement row, unit, "
        "report year, and matching corpus evidence chunk are available."
    )
    kpi_columns = st.columns(len(kpis))
    for column, (key, metric) in zip(kpi_columns, kpis):
        label = (
            metric.label
            if key == "operating" and metric is not None
            else (
                "Operating profit / margin"
                if key == "operating"
                else company_analysis.METRIC_DEFINITIONS[key]["label"]
            )
        )
        with column:
            with st.container(border=True):
                if metric is None:
                    st.metric(label, "Data unavailable")
                    st.caption(f"FY {report.financial_year} · no reliable sourced value")
                else:
                    st.metric(label, metric.value)
                    st.caption(
                        f"{metric.unit} · FY {metric.financial_year} · "
                        f"page {metric.source.page}"
                    )
                    st.caption(f"Chunk ID · `{metric.source.chunk_id}`")
                    with st.expander("Source and extracted evidence"):
                        chat_components.render_citation_card(
                            {
                                "citation": {
                                    "company": metric.source.company,
                                    "document": metric.source.document,
                                    "financial_year": metric.source.financial_year,
                                    "page": metric.source.page,
                                    "chunk_id": metric.source.chunk_id,
                                },
                                "text": metric.source.text,
                            },
                            1,
                        )

    st.divider()
    st.subheader("Historical financial trends")
    st.caption(
        "Only directly reported values with traceable evidence are plotted. "
        "Missing years are omitted; values are not estimated or converted."
    )
    chart_definitions = (
        ("revenue", "Revenue trend"),
        ("net_profit", "Net profit trend"),
        ("operating_margin", "Operating margin trend"),
        ("total_borrowings", "Debt trend"),
    )
    chart_columns = st.columns(2)
    for index, (metric_key, chart_title) in enumerate(chart_definitions):
        with chart_columns[index % 2]:
            st.markdown(f"#### {chart_title}")
            observations = [
                (annual_report, annual_report.metrics.get(metric_key))
                for annual_report in data.reports
                if annual_report.metrics.get(metric_key) is not None
            ]
            observed_units = {
                metric.unit for _, metric in observations if metric is not None
            }
            if not observations:
                st.info(f"No reliable {chart_title.lower()} data is available.")
                continue
            if len(observed_units) != 1:
                st.info(
                    "The available reports use different or unspecified units; "
                    "a continuous trend is not shown."
                )
                continue

            unit = next(iter(observed_units))
            points = company_analysis.chart_values(data.reports, metric_key)
            st.markdown(
                company_analysis.line_chart_svg(points, chart_title, unit),
                unsafe_allow_html=True,
            )
            for _, metric in observations:
                if metric is None:
                    continue
                with st.expander(
                    f"{metric.financial_year} · {metric.value} {metric.unit} · "
                    f"page {metric.source.page} · {metric.source.chunk_id}"
                ):
                    chat_components.render_citation_card(
                        {
                            "citation": {
                                "company": metric.source.company,
                                "document": metric.source.document,
                                "financial_year": metric.source.financial_year,
                                "page": metric.source.page,
                                "chunk_id": metric.source.chunk_id,
                            },
                            "text": metric.source.text,
                        },
                        1,
                    )


def _render_evidence_record(
    chunk: dict,
    corpus: evidence_explorer.EvidenceCorpus,
    *,
    query: str = "",
    exact_lookup: bool = False,
) -> None:
    company = evidence_explorer.display_company_name(chunk.get("company"))
    document = chunk.get("document") or "Not available"
    financial_year = chunk.get("financial_year") or "Not available"
    page = chunk.get("page")
    page_label = str(page) if isinstance(page, int) and not isinstance(page, bool) else "Not available"
    chunk_type = chunk.get("chunk_type")
    type_label = {
        "text": "Text chunk",
        "table": "Table chunk",
    }.get(chunk_type, str(chunk_type or "Not available"))
    chunk_id = chunk.get("chunk_id")
    chunk_id_label = chunk_id if isinstance(chunk_id, str) and chunk_id else "Not available"
    source = evidence_explorer.source_for_chunk(corpus, chunk)
    heading = chunk.get("table_heading") or chunk.get("section")
    excerpt = evidence_explorer.result_excerpt(chunk, query)

    with st.container(border=True):
        if exact_lookup:
            st.markdown("**Exact chunk match**")
        columns = st.columns([1.1, 1.5, 1, 0.65, 0.8])
        for column, label, value in zip(
            columns,
            ("COMPANY", "DOCUMENT", "FINANCIAL YEAR", "PAGE", "TYPE"),
            (company, document, financial_year, page_label, type_label),
        ):
            with column:
                st.caption(label)
                st.write(value)
        st.caption(f"Chunk ID · `{chunk_id_label}`")
        if isinstance(heading, str) and heading.strip():
            st.markdown(f"**{html.escape(heading)}**")
        if excerpt:
            st.write(excerpt)
        with st.expander("Inspect evidence and provenance"):
            metadata_columns = st.columns(3)
            details = (
                ("Company", company),
                ("Document", document),
                ("Financial year", financial_year),
                ("Page", page_label),
                ("Chunk type", type_label),
                ("Chunk ID", chunk_id_label),
            )
            for index, (label, value) in enumerate(details):
                with metadata_columns[index % len(metadata_columns)]:
                    st.caption(label)
                    st.write(value)
            if isinstance(chunk.get("table_heading"), str) and chunk["table_heading"]:
                st.caption("Table heading")
                st.write(chunk["table_heading"])
            if isinstance(chunk.get("context"), str) and chunk["context"]:
                st.caption("Table context")
                st.write(chunk["context"])
            if source is not None:
                st.caption(f"Source report · {source.source_file or document}")
                if source.pdf_path is not None:
                    st.caption(f"Source PDF is available locally: {source.pdf_path.name}")
                table_rows = evidence_explorer.read_structured_table(source)
                if chunk_type == "table" and table_rows:
                    st.markdown("**Structured extracted table**")
                    _render_structured_evidence_table(table_rows)
                elif chunk_type == "table":
                    st.info(
                        "A matching structured CSV is unavailable; the original "
                        "extracted passage is shown below."
                    )
            elif chunk_type == "table":
                st.info(
                    "Structured source metadata is unavailable; the original "
                    "extracted passage is shown below."
                )
            st.caption("Full extracted passage")
            passage = chunk.get("text")
            st.text(passage if isinstance(passage, str) else "Passage not available.")
            other_metadata = {
                key: value for key, value in chunk.items() if key != "text"
            }
            with st.expander("Other original chunk metadata"):
                st.json(other_metadata)


def render_evidence_explorer() -> None:
    selected_company = render_page_header(
        "Evidence Explorer",
        "Search, inspect and verify evidence from company annual reports.",
        company_selector_key="evidence_company_selector",
    )
    with st.spinner("Loading annual-report evidence..."):
        corpus = _cached_evidence_corpus(str(PROJECT_ROOT))
    if corpus.unavailable_reason and not corpus.chunks:
        st.warning(corpus.unavailable_reason)
        return

    overview = _cached_overview_data(str(PROJECT_ROOT))
    stats = st.columns(4)
    with stats[0]:
        st.metric("Indexed chunks", _display_value(overview.corpus_chunks))
    with stats[1]:
        st.metric("Companies", len(corpus.companies))
    with stats[2]:
        st.metric("Financial years", len(corpus.financial_years))
    with stats[3]:
        st.metric("Annual reports", corpus.available_reports)
    st.caption(
        f"Searchable records in the corpus file: {len(corpus.chunks):,} · "
        f"Available years: {', '.join(corpus.financial_years) or 'Not available'}"
    )
    if corpus.malformed_records:
        st.warning(
            f"{corpus.malformed_records:,} malformed corpus record(s) were skipped."
        )

    query = st.text_input(
        "Search annual-report evidence",
        key="evidence_search_query",
        placeholder="Try: Revenue from operations, total borrowings, or risk management",
    )
    st.caption(
        "Search mode: case-insensitive lexical matching against original passage text, "
        "table headings, context, and section labels. Every entered term must match; "
        "no LLM or semantic retrieval is used."
    )

    filter_columns = st.columns([1.1, 1.1, 1.2, 0.8])
    year_options = ("All available years", *corpus.financial_years)
    document_options = ("All document types", *corpus.documents)
    type_options = ("All evidence types", "Text chunks", "Table chunks")
    with filter_columns[0]:
        selected_year = st.selectbox(
            "Financial year",
            year_options,
            key="evidence_financial_year",
        )
    with filter_columns[1]:
        selected_document = st.selectbox(
            "Document type",
            document_options,
            key="evidence_document_type",
        )
    with filter_columns[2]:
        selected_type = st.selectbox(
            "Evidence type",
            type_options,
            key="evidence_chunk_type",
        )
    max_page = max(
        (
            chunk["page"]
            for chunk in corpus.chunks
            if isinstance(chunk.get("page"), int)
            and not isinstance(chunk.get("page"), bool)
            and chunk["page"] > 0
        ),
        default=1,
    )
    with filter_columns[3]:
        selected_page = st.number_input(
            "Page (0 = all)",
            min_value=0,
            max_value=max_page,
            step=1,
            key="evidence_page_filter",
        )
    results = evidence_explorer.search_chunks(
        corpus.chunks,
        company=selected_company,
        financial_year=(
            selected_year if selected_year != "All available years" else None
        ),
        document=(
            selected_document if selected_document != "All document types" else None
        ),
        chunk_type={
            "Text chunks": "text",
            "Table chunks": "table",
        }.get(selected_type),
        page=selected_page or None,
        query=query,
    )

    st.divider()
    st.subheader("Exact chunk ID lookup")
    exact_id = st.text_input(
        "Find an exact corpus chunk ID",
        key="evidence_exact_chunk_id",
        placeholder="Paste a chunk ID from a research citation",
    ).strip()
    if exact_id:
        exact_chunk = evidence_explorer.exact_chunk_lookup(corpus.chunks, exact_id)
        if exact_chunk is None:
            st.info("No corpus chunk has that exact ID.")
        else:
            _render_evidence_record(exact_chunk, corpus, exact_lookup=True)

    st.divider()
    st.subheader("Search results")
    st.caption(f"{len(results):,} matching evidence chunk(s)")
    if not results:
        st.info("No evidence matches the current search and filters.")
        return

    page_count = max(
        1,
        (len(results) + evidence_explorer.RESULTS_PER_PAGE - 1)
        // evidence_explorer.RESULTS_PER_PAGE,
    )
    page_key = "evidence_results_page"
    current_page = st.session_state.get(page_key, 1)
    if not isinstance(current_page, int) or isinstance(current_page, bool):
        current_page = 1
    st.session_state[page_key] = min(max(current_page, 1), page_count)
    page_number = st.number_input(
        "Results page",
        min_value=1,
        max_value=page_count,
        step=1,
        key=page_key,
    )
    page_results, _ = evidence_explorer.paginate_results(
        results,
        page_number,
    )
    start = (page_number - 1) * evidence_explorer.RESULTS_PER_PAGE + 1
    end = min(start + len(page_results) - 1, len(results))
    st.caption(f"Showing {start:,}–{end:,} of {len(results):,} results")
    for chunk in page_results:
        _render_evidence_record(chunk, corpus, query=query)


def render_evaluation() -> None:
    render_page_header(
        "Evaluation & Performance",
        "Compare saved retrieval results, operational performance, and evaluation-data reliability.",
    )
    data = _cached_evaluation_data(str(PROJECT_ROOT))
    evaluation_dashboard.render_evaluation_dashboard(data)


def render_settings() -> None:
    render_page_header(
        "Settings",
        "Review local model configuration status. Secret values are never displayed.",
    )
    planner_column, answer_column = st.columns(2)
    with planner_column:
        st.subheader("Query planning")
        try:
            query_planner.load_settings()
        except query_planner.PlannerConfigError as error:
            st.warning(f"Needs configuration: {error}")
        else:
            st.success("Gemini query planner is configured.")
    with answer_column:
        st.subheader("Answer generation")
        try:
            settings = load_settings()
        except LLMConfigurationError as error:
            st.warning(f"Needs configuration: {error}")
        else:
            st.success(f"{settings.provider.title()} is configured · model `{settings.model}`")
    st.caption(
        "Configuration is loaded from the local environment and `.env` file. "
        "The existing query planner uses Gemini; answer generation uses the selected provider."
    )
