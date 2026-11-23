import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from ui import dashboard
from ui.company_analysis import (
    METRIC_DEFINITIONS,
    AnnualReportAnalysis,
    CompanyAnalysisData,
    EvidenceSource,
    MetricObservation,
    _parse_numeric_value,
    chart_values,
    line_chart_svg,
    load_company_analysis,
)


def _write_report(
    project_root: Path,
    *,
    company_folder: str = "infosys",
    company: str = "Infosys",
    report_end_year: int = 2025,
    unit_context: str | None = "₹ crore",
    evidence: bool = True,
    revenue: str = "10,000",
    profit: str = "1,000",
    include_operating_margin: bool = True,
    borrowings_rows: tuple[tuple[str, str], ...] = (("Total borrowings", "100"),),
) -> None:
    report_year = f"{report_end_year - 1}-{report_end_year}"
    report_directory = (
        project_root / "data" / "processed" / company_folder / str(report_end_year)
    )
    table_directory = report_directory / "tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / "document_metadata.json").write_text(
        json.dumps(
            {
                "company": company,
                "document": "Annual Report",
                "financial_year": report_year,
                "source_file": f"{company}_AnnualReport_{report_year}.pdf",
                "number_of_pages": 200,
            }
        ),
        encoding="utf-8",
    )

    profit_loss_name = "page_010_table_001.csv"
    profit_loss_rows = [
        ["Particulars", f"Year ended March 31, {report_end_year}", f"Year ended March 31, {report_end_year - 1}"],
        ["Revenue from operations", revenue, "9,000"],
        ["Profit for the year", profit, "800"],
    ]
    if include_operating_margin:
        profit_loss_rows.append(["Operating margin", "20.5%", "19.0%"])
    with (table_directory / profit_loss_name).open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(profit_loss_rows)

    balance_sheet_name = "page_008_table_002.csv"
    balance_sheet_rows = [
        ["Particulars", f"As at March 31, {report_end_year}", f"As at March 31, {report_end_year - 1}"],
        ["Total assets", "25,000", "20,000"],
        *[[label, value, "90"] for label, value in borrowings_rows],
    ]
    with (table_directory / balance_sheet_name).open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(balance_sheet_rows)

    table_metadata = [
        {
            "table_id": f"{company_folder}_{report_end_year}_p010_t001",
            "table_number": 1,
            "page": 10,
            "filename": profit_loss_name,
            "table_heading": "Consolidated Statement of Profit and Loss",
            "context": unit_context,
        },
        {
            "table_id": f"{company_folder}_{report_end_year}_p008_t002",
            "table_number": 2,
            "page": 8,
            "filename": balance_sheet_name,
            "table_heading": "Consolidated Balance Sheet",
            "context": unit_context,
        },
    ]
    (report_directory / "table_metadata.json").write_text(
        json.dumps(table_metadata),
        encoding="utf-8",
    )

    chunks_directory = project_root / "data" / "chunks"
    chunks_directory.mkdir(parents=True, exist_ok=True)
    chunks = []
    if evidence:
        chunks = [
            {
                "company": company,
                "document": "Annual Report",
                "financial_year": report_year,
                "page": 10,
                "chunk_type": "table",
                "chunk_id": f"annualreport_{company_folder}_{report_end_year}_p010_t001",
                "text": (
                    "Table Heading: Consolidated Statement of Profit and Loss "
                    f"Revenue from operations {revenue} Profit for the year {profit} "
                    "Operating margin 20.5%"
                ),
            },
            {
                "company": company,
                "document": "Annual Report",
                "financial_year": report_year,
                "page": 8,
                "chunk_type": "table",
                "chunk_id": f"annualreport_{company_folder}_{report_end_year}_p008_t002",
                "text": "Total assets 25,000 Total borrowings 100",
            },
        ]
    with (chunks_directory / f"{company_folder}_{report_end_year}.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            file.write(json.dumps(chunk) + "\n")


def _metric(
    key: str,
    year: str,
    value: str,
    number: float,
    unit: str,
    page: int,
) -> MetricObservation:
    source = EvidenceSource(
        company="Infosys",
        document="Annual Report",
        financial_year=year,
        page=page,
        chunk_id=f"annualreport_infosys_{year[-4:]}_p{page:03d}_t001",
        text=f"Source passage for {key} {value}",
    )
    return MetricObservation(
        key=key,
        label=METRIC_DEFINITIONS[key]["label"],
        value=value,
        numeric_value=number,
        unit=unit,
        financial_year=year,
        row_label=METRIC_DEFINITIONS[key]["aliases"][0],
        source=source,
    )


def test_company_year_and_kpis_are_selected_from_report_metadata(tmp_path):
    _write_report(tmp_path, report_end_year=2025)
    _write_report(tmp_path, report_end_year=2026, revenue="11,000", profit="1,100")

    data = load_company_analysis(tmp_path, "Infosys")

    assert data.available_years == ("2025-2026", "2024-2025")
    report = data.report_for_year("2024-2025")
    assert report is not None
    assert report.source_file == "Infosys_AnnualReport_2024-2025.pdf"
    assert report.metrics["revenue"] is not None
    assert report.metrics["revenue"].value == "10,000"
    assert report.metrics["revenue"].unit == "₹ crore"
    assert report.metrics["revenue"].source.page == 10
    assert report.metrics["revenue"].source.chunk_id == (
        "annualreport_infosys_2025_p010_t001"
    )
    assert report.metrics["net_profit"].value == "1,000"
    assert report.metrics["operating_margin"].unit == "%"
    assert report.metrics["total_assets"].value == "25,000"
    assert report.metrics["total_borrowings"].value == "100"


def test_chart_values_include_only_sourced_years_and_are_chronological(tmp_path):
    _write_report(tmp_path, report_end_year=2025)
    _write_report(tmp_path, report_end_year=2026, evidence=False)
    data = load_company_analysis(tmp_path, "Infosys")

    assert chart_values(data.reports, "revenue") == [
        {"Financial year": "2024-2025", "Value": 10000.0}
    ]
    svg = line_chart_svg(chart_values(data.reports, "revenue"), "Revenue trend", "₹ crore")
    assert 'aria-label="Revenue trend in ₹ crore"' in svg
    assert "2024-2025" in svg
    assert "10,000.00" in svg


def test_unavailable_unit_malformed_value_and_missing_chunk_are_not_shown(tmp_path):
    _write_report(tmp_path, unit_context=None, evidence=True)
    data_without_unit = load_company_analysis(tmp_path, "Infosys")
    report_without_unit = data_without_unit.report_for_year("2024-2025")
    assert report_without_unit is not None
    assert report_without_unit.metrics["revenue"] is None

    _write_report(tmp_path, unit_context="₹ crore", evidence=False, revenue="49, 454")
    data_without_evidence = load_company_analysis(tmp_path, "Infosys")
    report_without_evidence = data_without_evidence.report_for_year("2024-2025")
    assert report_without_evidence is not None
    assert report_without_evidence.metrics["revenue"] is None


def test_ambiguous_borrowing_rows_are_not_summed_or_guessed(tmp_path):
    _write_report(
        tmp_path,
        borrowings_rows=(
            ("Borrowings", "1,000"),
            ("Borrowings", "2,000"),
        ),
    )

    report = load_company_analysis(tmp_path, "Infosys").report_for_year("2024-2025")

    assert report is not None
    assert report.metrics["total_borrowings"] is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0.0),
        ("1,78,650", 178650.0),
        ("(1,234)", -1234.0),
        ("(1,234", None),
        ("1,234)", None),
        ("49, 454", None),
        ("1,2,345", None),
        ("-", None),
        ("20.5%", None),
    ],
)
def test_financial_value_parser_requires_unambiguous_numeric_cells(raw, expected):
    assert _parse_numeric_value(raw) == expected


def test_financial_year_selection_and_kpi_rendering(monkeypatch):
    year = "2024-2025"
    metrics = {key: None for key in METRIC_DEFINITIONS}
    metrics["revenue"] = _metric("revenue", year, "10,000", 10000, "₹ crore", 10)
    selected_report = AnnualReportAnalysis(
        financial_year=year,
        source_file="Infosys_AnnualReport_2024-2025.pdf",
        number_of_pages=200,
        table_count=12,
        metrics=metrics,
    )
    latest_report = AnnualReportAnalysis(
        financial_year="2025-2026",
        source_file="Infosys_AnnualReport_2025-2026.pdf",
        number_of_pages=210,
        table_count=10,
        metrics={key: None for key in METRIC_DEFINITIONS},
    )
    data = CompanyAnalysisData(
        company="Infosys",
        reports=(latest_report, selected_report),
    )
    requested_companies = []

    def fake_company_data(project_root, company):
        requested_companies.append(company)
        return data

    monkeypatch.setattr(dashboard, "_cached_company_analysis_data", fake_company_data)
    app = """
import streamlit as st
from ui.dashboard import configure_session_state, render_company_analysis, render_page_header
configure_session_state()
if st.session_state.get("active_page", "analysis") == "analysis":
    render_company_analysis()
else:
    render_page_header("Financial Research Chat", "Chat", company_selector_key="chat_company_selector")
"""
    at = AppTest.from_string(app).run()
    at.selectbox[0].select("Infosys").run()
    assert at.selectbox[1].options == ["2025-2026", year]
    at.selectbox[1].select(year).run()

    assert not at.exception
    assert requested_companies == ["Infosys", "Infosys"]
    assert at.session_state["analysis_financial_year_infosys"] == year
    metric_values = [metric.value for metric in at.metric]
    assert "Data unavailable" in metric_values
    assert "10,000" in metric_values
    assert any("annualreport_infosys_2025_p010_t001" in item.value for item in at.caption)
    assert any("<svg" in element.value for element in at.markdown)

    at.session_state["active_page"] = "chat"
    at.run()
    at.session_state["active_page"] = "analysis"
    at.run()

    assert at.selectbox[0].value == "Infosys"
    assert at.selectbox[1].value == year
