import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui import dashboard
from ui.dashboard_data import OverviewData
from ui.evidence_explorer import (
    EvidenceCorpus,
    TableSource,
    exact_chunk_lookup,
    load_evidence_corpus,
    paginate_results,
    read_structured_table,
    search_chunks,
    source_for_chunk,
)


def _chunk(
    chunk_id="chunk-1",
    *,
    company="TataConsultancyServices",
    document="Annual Report",
    financial_year="2024-2025",
    page=10,
    chunk_type="text",
    text="Revenue from operations increased.",
    **metadata,
):
    return {
        "chunk_id": chunk_id,
        "company": company,
        "document": document,
        "financial_year": financial_year,
        "page": page,
        "chunk_type": chunk_type,
        "text": text,
        **metadata,
    }


def _write_corpus(root: Path, records: list[dict]) -> None:
    chunks_path = root / "data" / "chunks" / "all_chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_table_source(root: Path) -> tuple[dict, Path]:
    report_directory = root / "data" / "processed" / "tataconsultancyservices" / "2025"
    tables_directory = report_directory / "tables"
    tables_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / "document_metadata.json").write_text(
        json.dumps(
            {
                "company": "TataConsultancyServices",
                "document": "Annual Report",
                "financial_year": "2024-2025",
                "source_file": "TataConsultancyServices_AnnualReport_2024-2025.pdf",
            }
        ),
        encoding="utf-8",
    )
    (report_directory / "table_metadata.json").write_text(
        json.dumps(
            [
                {
                    "table_id": "tataconsultancyservices_2025_p010_t001",
                    "company": "TataConsultancyServices",
                    "document": "Annual Report",
                    "financial_year": "2024-2025",
                    "page": 10,
                    "table_heading": "Consolidated statement",
                    "context": "Amounts in crore",
                    "filename": "page_010_table_001.csv",
                }
            ]
        ),
        encoding="utf-8",
    )
    csv_path = tables_directory / "page_010_table_001.csv"
    csv_path.write_text("Particulars,Value\nRevenue,100\n", encoding="utf-8")
    raw_directory = root / "data" / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    (raw_directory / "TataConsultancyServices_AnnualReport_2024-2025.pdf").write_bytes(
        b"test-pdf-placeholder"
    )
    chunk = _chunk(
        chunk_id="annualreport_tataconsultancyservices_2025_p010_t001",
        chunk_type="table",
        text="Table Heading: Consolidated statement\n| Particulars | Value |",
        table_id="tataconsultancyservices_2025_p010_t001",
        table_heading="Consolidated statement",
        context="Amounts in crore",
    )
    return chunk, csv_path


def test_loads_valid_jsonl_and_skips_malformed_records(tmp_path):
    _write_corpus(
        tmp_path,
        [
            _chunk(),
            _chunk("chunk-2", company="Infosys"),
            {"chunk_id": "sparse", "company": [], "document": {}, "financial_year": []},
        ],
    )
    path = tmp_path / "data" / "chunks" / "all_chunks.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write("{invalid json}\n[]\n")

    corpus = load_evidence_corpus(tmp_path)

    assert len(corpus.chunks) == 3
    assert corpus.malformed_records == 2
    assert corpus.companies == ("Infosys", "TataConsultancyServices")
    assert corpus.financial_years == ("2024-2025",)
    assert corpus.available_reports == 2


def test_missing_corpus_returns_clear_unavailable_state(tmp_path):
    corpus = load_evidence_corpus(tmp_path)

    assert corpus.chunks == ()
    assert corpus.unavailable_reason == "The corpus chunk file could not be read."


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        ({"company": "TataConsultancyServices"}, ["chunk-1", "chunk-2", "chunk-4"]),
        ({"financial_year": "2023-2024"}, ["chunk-2"]),
        ({"document": "Other filing"}, ["chunk-3"]),
        ({"chunk_type": "table"}, ["chunk-4"]),
        ({"page": 12}, ["chunk-2", "chunk-4"]),
        (
            {
                "company": "TataConsultancyServices",
                "financial_year": "2024-2025",
                "document": "Annual Report",
                "chunk_type": "table",
                "page": 12,
            },
            [],
        ),
    ],
)
def test_filters_can_be_combined(tmp_path, filters, expected_ids):
    chunks = [
        _chunk(),
        _chunk("chunk-2", financial_year="2023-2024", page=12),
        _chunk("chunk-3", company="Infosys", document="Other filing"),
        _chunk(
            "chunk-4",
            chunk_type="table",
            page=12,
            financial_year="2022-2023",
        ),
    ]

    matches = search_chunks(chunks, **filters)

    assert [chunk["chunk_id"] for chunk in matches] == expected_ids


def test_lexical_search_matches_all_terms_in_source_content():
    chunks = [
        _chunk("revenue-profit", text="Consolidated revenue and profit"),
        _chunk("revenue-only", text="Revenue from operations"),
        _chunk(
            "metadata-match",
            text="No matching words in passage",
            table_heading="Capital expenditure",
        ),
    ]

    assert [
        chunk["chunk_id"]
        for chunk in search_chunks(chunks, query="CONSOLIDATED, revenue.")
    ] == ["revenue-profit"]
    assert [
        chunk["chunk_id"]
        for chunk in search_chunks(chunks, query="capital expenditure")
    ] == ["metadata-match"]


def test_exact_lookup_does_not_return_similar_ids():
    chunks = [_chunk("annualreport_infosys_2024"), _chunk("annualreport_infosys_2025")]

    assert exact_chunk_lookup(chunks, "annualreport_infosys_2025") == chunks[1]
    assert exact_chunk_lookup(chunks, "annualreport_infosys_202") is None


def test_missing_metadata_is_retained_and_source_is_not_guessed(tmp_path):
    chunk = {"chunk_id": "sparse", "text": "Evidence without provenance fields"}
    _write_corpus(tmp_path, [chunk])

    corpus = load_evidence_corpus(tmp_path)

    assert corpus.chunks == (chunk,)
    assert search_chunks(corpus.chunks, query="evidence")
    assert exact_chunk_lookup(corpus.chunks, "sparse") == chunk
    assert source_for_chunk(corpus, chunk) is None


def test_missing_exact_id_has_no_fallback():
    assert exact_chunk_lookup([_chunk()], "chunk-10") is None


def test_structured_table_is_loaded_from_original_csv_and_checked_against_metadata(tmp_path):
    chunk, csv_path = _write_table_source(tmp_path)
    corpus = load_evidence_corpus(tmp_path)
    source = source_for_chunk(corpus, chunk)

    assert source is not None
    assert source.csv_path == csv_path
    assert source.pdf_path is not None
    assert read_structured_table(source) == [
        ["Particulars", "Value"],
        ["Revenue", "100"],
    ]

    incorrect_provenance = {**chunk, "page": 11}
    assert source_for_chunk(corpus, incorrect_provenance) is None


def test_table_file_rejects_traversal_and_missing_csv(tmp_path):
    chunk, _ = _write_table_source(tmp_path)
    report_metadata = json.loads(
        (
            tmp_path
            / "data"
            / "processed"
            / "tataconsultancyservices"
            / "2025"
            / "table_metadata.json"
        ).read_text(encoding="utf-8")
    )
    report_metadata[0]["filename"] = "..\\outside.csv"
    (tmp_path / "data" / "processed" / "tataconsultancyservices" / "2025" / "table_metadata.json").write_text(
        json.dumps(report_metadata),
        encoding="utf-8",
    )
    corpus = load_evidence_corpus(tmp_path)

    assert source_for_chunk(corpus, chunk).csv_path is None
    assert read_structured_table(source_for_chunk(corpus, chunk)) is None


def test_pagination_limits_results_and_clamps_invalid_page():
    results = [_chunk(f"chunk-{index}") for index in range(45)]

    first_page, page_count = paginate_results(results, 1, page_size=20)
    last_page, _ = paginate_results(results, 99, page_size=20)

    assert len(first_page) == 20
    assert page_count == 3
    assert [chunk["chunk_id"] for chunk in last_page] == ["chunk-40", "chunk-41", "chunk-42", "chunk-43", "chunk-44"]


def test_evidence_table_renderer_uses_escaped_structured_csv():
    app = """
from ui.dashboard import _render_structured_evidence_table
_render_structured_evidence_table([["Metric", "Value"], ["<script>", "100"]])
"""
    at = AppTest.from_string(app).run()

    assert not at.exception
    markup = next(element.value for element in at.markdown if "<table>" in element.value)
    assert "&lt;script&gt;" in markup
    assert "<script>" not in markup


def test_evidence_explorer_renders_search_and_empty_state(monkeypatch):
    corpus = EvidenceCorpus(
        chunks=(_chunk(),),
        table_sources={},
    )
    monkeypatch.setattr(dashboard, "_cached_evidence_corpus", lambda root: corpus)
    monkeypatch.setattr(
        dashboard,
        "_cached_overview_data",
        lambda root: OverviewData(3, 60, 7, 36, 48_063, {}),
    )
    app = """
from ui.dashboard import configure_session_state, render_evidence_explorer
configure_session_state()
render_evidence_explorer()
"""
    at = AppTest.from_string(app).run()

    assert not at.exception
    assert any("1 matching evidence chunk" in element.value for element in at.caption)
    assert any("Revenue from operations increased." in element.value for element in at.markdown)

    at.text_input[0].set_value("unmatched phrase").run()

    assert not at.exception
    assert any("No evidence matches" in element.value for element in at.info)


def test_evidence_explorer_company_filter_and_exact_id_are_independent(monkeypatch):
    chunks = (
        _chunk(),
        _chunk("infosys-1", company="Infosys", text="Revenue from operations at Infosys."),
    )
    corpus = EvidenceCorpus(chunks=chunks, table_sources={})
    monkeypatch.setattr(dashboard, "_cached_evidence_corpus", lambda root: corpus)
    monkeypatch.setattr(
        dashboard,
        "_cached_overview_data",
        lambda root: OverviewData(2, 60, 7, 2, 2, {}),
    )
    app = """
from ui.dashboard import configure_session_state, render_evidence_explorer
configure_session_state()
render_evidence_explorer()
"""
    at = AppTest.from_string(app).run()
    at.selectbox[0].select("Infosys").run()

    assert not at.exception
    assert any("1 matching evidence chunk" in element.value for element in at.caption)
    at.text_input[1].set_value("chunk-10").run()

    assert not at.exception
    assert any("No corpus chunk has that exact ID" in element.value for element in at.info)
    assert any("1 matching evidence chunk" in element.value for element in at.caption)
    at.text_input[1].set_value("chunk-1").run()

    assert not at.exception
    assert any("Exact chunk match" in element.value for element in at.markdown)


def test_evidence_filters_remain_selected_as_other_filters_change(monkeypatch):
    chunks = (
        _chunk(
            "tcs-table",
            chunk_type="table",
            table_heading="Revenue",
        ),
        _chunk(
            "infosys-quarterly",
            company="Infosys",
            document="Quarterly Report",
            financial_year="2023-2024",
        ),
    )
    corpus = EvidenceCorpus(chunks=chunks, table_sources={})
    monkeypatch.setattr(dashboard, "_cached_evidence_corpus", lambda root: corpus)
    monkeypatch.setattr(
        dashboard,
        "_cached_overview_data",
        lambda root: OverviewData(2, 60, 7, 2, 2, {}),
    )
    app = """
from ui.dashboard import configure_session_state, render_evidence_explorer
configure_session_state()
render_evidence_explorer()
"""
    at = AppTest.from_string(app).run()
    at.selectbox[0].select("TCS").run()
    at.selectbox[1].select("2024-2025").run()
    at.selectbox[2].select("Annual Report").run()
    at.selectbox[3].select("Table chunks").run()
    at.number_input[0].set_value(10).run()

    assert not at.exception
    assert [select.value for select in at.selectbox] == [
        "TCS",
        "2024-2025",
        "Annual Report",
        "Table chunks",
    ]
    assert at.number_input[0].value == 10
    assert any("1 matching evidence chunk" in element.value for element in at.caption)
