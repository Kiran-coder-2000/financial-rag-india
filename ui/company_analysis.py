"""Conservative, citation-backed reads of already-extracted annual-report tables."""

import csv
import html
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dashboard_data import COMPANIES

logger = logging.getLogger(__name__)

METRIC_DEFINITIONS = {
    "revenue": {
        "label": "Consolidated revenue",
        "statement": "profit and loss",
        "aliases": (
            "value of sales services revenue",
            "revenue from operations",
            "revenues",
        ),
    },
    "net_profit": {
        "label": "Net profit",
        "statement": "profit and loss",
        "aliases": (
            "profit for the year",
            "profit after tax and share of profit loss of associates and joint ventures",
            "profit after tax",
            "net profit",
        ),
    },
    "operating_profit": {
        "label": "Operating profit",
        "statement": "profit and loss",
        "aliases": ("operating profit",),
    },
    "operating_margin": {
        "label": "Operating margin",
        "statement": "profit and loss",
        "aliases": ("operating margin", "operating profit margin"),
    },
    "total_assets": {
        "label": "Total assets",
        "statement": "balance sheet",
        "aliases": ("total assets",),
    },
    "total_borrowings": {
        "label": "Total borrowings",
        "statement": "balance sheet",
        "aliases": ("total borrowings", "borrowings"),
    },
}

@dataclass(frozen=True)
class EvidenceSource:
    company: str
    document: str
    financial_year: str
    page: int
    chunk_id: str
    text: str


@dataclass(frozen=True)
class MetricObservation:
    key: str
    label: str
    value: str
    numeric_value: float
    unit: str
    financial_year: str
    row_label: str
    source: EvidenceSource


@dataclass(frozen=True)
class AnnualReportAnalysis:
    financial_year: str
    source_file: str | None
    number_of_pages: int | None
    table_count: int
    metrics: dict[str, MetricObservation | None]


@dataclass(frozen=True)
class CompanyAnalysisData:
    company: str
    reports: tuple[AnnualReportAnalysis, ...]
    unavailable_reason: str | None = None

    @property
    def available_years(self) -> tuple[str, ...]:
        return tuple(report.financial_year for report in self.reports)

    def report_for_year(self, financial_year: str) -> AnnualReportAnalysis | None:
        return next(
            (report for report in self.reports if report.financial_year == financial_year),
            None,
        )


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _company_folder(company: str) -> str | None:
    for display_name, canonical_name in COMPANIES:
        if company in (display_name, canonical_name):
            return re.sub(r"[^a-z0-9]", "", canonical_name.casefold())
    return None


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read company-analysis data from %s: %s", path, error)
        return None


def _financial_year(value: Any) -> str | None:
    match = re.fullmatch(r"\s*(\d{4})\s*[-/]\s*(\d{4})\s*", str(value or ""))
    if not match or int(match.group(2)) != int(match.group(1)) + 1:
        return None
    return f"{match.group(1)}-{match.group(2)}"


def _financial_year_end(financial_year: str) -> int:
    return int(financial_year[-4:])


def _current_year_column(headers: list[str], financial_year: str) -> int | None:
    start_year, end_year = (int(part) for part in financial_year.split("-"))
    short_end = f"{end_year % 100:02d}"
    candidates: list[int] = []
    for index, header in enumerate(headers[1:], start=1):
        normalized_header = str(header)
        has_full_end_year = re.search(
            rf"(?<!\d){end_year}(?!\d)", normalized_header
        ) is not None
        has_current_fiscal_label = re.search(
            rf"(?<!\d){start_year}\s*[-/]\s*{short_end}(?!\d)",
            normalized_header,
        ) is not None
        if has_full_end_year or has_current_fiscal_label:
            candidates.append(index)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _parse_numeric_value(value: str, *, is_percentage: bool = False) -> float | None:
    text = str(value).strip()
    if is_percentage and text.endswith("%"):
        text = text[:-1].strip()
    elif "%" in text:
        return None
    number_pattern = (
        r"(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?"
    )
    if text.startswith("(") or text.endswith(")"):
        if not re.fullmatch(rf"\(\s*{number_pattern}\s*\)", text):
            return None
        text = "-" + text[1:-1].strip()
    elif not re.fullmatch(rf"-?{number_pattern}", text):
        return None
    return float(text.replace(",", ""))


def _unit(context: str | None, headers: list[str], row_label: str) -> str | None:
    unit_text = " ".join((context or "", *map(str, headers), row_label)).casefold()
    if "margin" in _normalize(row_label) or "%" in row_label or "percent" in row_label:
        return "%"
    if re.search(r"\b(crore|crores)\b", unit_text):
        if "₹" in unit_text:
            return "₹ crore"
        if re.search(r"\b(?:inr|rs\.?)\b", unit_text):
            return "INR crore"
        return "crore (currency not explicit)"
    if re.search(r"\b(million|millions)\b", unit_text):
        if "₹" in unit_text:
            return "₹ million"
        if re.search(r"\b(?:inr|rs\.?)\b", unit_text):
            return "INR million"
        return "million (currency not explicit)"
    if re.search(r"\b(billion|billions)\b", unit_text):
        if "₹" in unit_text:
            return "₹ billion"
        if re.search(r"\b(?:inr|rs\.?)\b", unit_text):
            return "INR billion"
        return "billion (currency not explicit)"
    return None


def _table_rows(report_directory: Path, table: dict[str, Any]) -> list[list[str]] | None:
    filename = table.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        logger.warning("Ignoring table with invalid filename metadata in %s", report_directory)
        return None
    path = report_directory / "tables" / filename
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            return list(csv.reader(file))
    except (OSError, UnicodeError, csv.Error) as error:
        logger.warning("Could not read extracted table %s: %s", path, error)
        return None


def _statement_kind(table: dict[str, Any]) -> str | None:
    heading = _normalize(table.get("table_heading"))
    if "consolidated" not in heading:
        return None
    if "consolidated statement of profit and loss" in heading:
        return "profit and loss"
    if "consolidated balance sheet" in heading:
        return "balance sheet"
    if "consolidated statement of changes in equity" in heading:
        return "changes in equity"
    if "consolidated statement of cash flows" in heading:
        return "cash flows"
    return None


def _main_statement_tables(
    tables: list[dict[str, Any]],
    statement: str,
) -> list[dict[str, Any]]:
    statement_tables = [
        table for table in tables if _statement_kind(table) == statement
    ]
    valid_pages = [
        table.get("page")
        for table in statement_tables
        if isinstance(table.get("page"), int) and table["page"] > 0
    ]
    if not valid_pages:
        return []
    start_page = min(valid_pages)
    next_statement_pages = [
        table["page"]
        for table in tables
        if _statement_kind(table) not in (None, statement)
        and isinstance(table.get("page"), int)
        and table["page"] > start_page
    ]
    end_page = min(next_statement_pages, default=float("inf"))
    return [
        table
        for table in statement_tables
        if isinstance(table.get("page"), int)
        and start_page <= table["page"] < end_page
    ]


def _read_evidence_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as error:
                    logger.warning(
                        "Skipping invalid chunk record in %s at line %s: %s",
                        path,
                        line_number,
                        error,
                    )
                    continue
                if isinstance(chunk, dict):
                    chunks.append(chunk)
    except OSError as error:
        logger.warning("Could not read evidence chunks from %s: %s", path, error)
    return chunks


def _evidence_for_table_row(
    chunks: list[dict[str, Any]],
    table: dict[str, Any],
    financial_year: str,
    row_label: str,
    raw_value: str,
) -> EvidenceSource | None:
    table_id = str(table.get("table_id") or "")
    suffix_match = re.search(r"_p(\d+)_t(\d+)$", table_id)
    page = table.get("page")
    if not suffix_match or not isinstance(page, int) or page <= 0:
        return None
    table_suffix = f"p{int(suffix_match.group(1)):03d}_t{int(suffix_match.group(2)):03d}"
    normalized_label = _normalize(row_label)
    normalized_value = re.sub(r"\D", "", raw_value)
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        if (
            chunk.get("chunk_type") != "table"
            or chunk.get("financial_year") != financial_year
            or chunk.get("page") != page
            or not isinstance(chunk_id, str)
            or not re.search(rf"{re.escape(table_suffix)}(?:_|$)", chunk_id)
            or not isinstance(text, str)
        ):
            continue
        normalized_text = _normalize(text)
        text_digits = re.sub(r"\D", "", text)
        if normalized_label not in normalized_text or normalized_value not in text_digits:
            continue
        source_company = chunk.get("company")
        source_document = chunk.get("document")
        if not isinstance(source_company, str) or not isinstance(source_document, str):
            continue
        return EvidenceSource(
            company=source_company,
            document=source_document,
            financial_year=financial_year,
            page=page,
            chunk_id=chunk_id,
            text=text,
        )
    return None


def _find_metric(
    key: str,
    definition: dict[str, Any],
    tables: list[dict[str, Any]],
    report_directory: Path,
    chunks: list[dict[str, Any]],
    company: str,
    financial_year: str,
) -> MetricObservation | None:
    candidate_tables = _main_statement_tables(tables, definition["statement"])
    aliases = tuple(_normalize(alias) for alias in definition["aliases"])
    for alias in aliases:
        matching_rows: list[
            tuple[dict[str, Any], list[str], list[str], int, str, float, str]
        ] = []
        for table in candidate_tables:
            rows = _table_rows(report_directory, table)
            if not rows:
                continue
            headers = rows[0]
            value_column = _current_year_column(headers, financial_year)
            if value_column is None:
                continue
            for row in rows[1:]:
                row_label = row[0].strip() if row else ""
                if _normalize(row_label) != alias or value_column >= len(row):
                    continue
                raw_value = row[value_column].strip()
                is_percentage = key == "operating_margin"
                numeric_value = _parse_numeric_value(
                    raw_value,
                    is_percentage=is_percentage,
                )
                unit = _unit(table.get("context"), headers, row_label)
                if numeric_value is None or unit is None:
                    continue
                matching_rows.append(
                    (table, rows, headers, value_column, row_label, numeric_value, raw_value)
                )
        if len(matching_rows) != 1:
            continue

        table, _, headers, _, row_label, numeric_value, raw_value = matching_rows[0]
        source = _evidence_for_table_row(
            chunks,
            table,
            financial_year,
            row_label,
            raw_value,
        )
        if source is None:
            logger.warning(
                "No matching evidence chunk for %s %s %s on page %s",
                company,
                financial_year,
                row_label,
                table.get("page"),
            )
            return None
        unit = _unit(table.get("context"), headers, row_label)
        if unit is None:
            return None
        return MetricObservation(
            key=key,
            label=definition["label"],
            value=raw_value,
            numeric_value=numeric_value,
            unit=unit,
            financial_year=financial_year,
            row_label=row_label,
            source=source,
        )
    return None


def _load_report(
    company: str,
    company_folder: str,
    report_directory: Path,
    project_root: Path,
) -> AnnualReportAnalysis | None:
    document_metadata = _read_json(report_directory / "document_metadata.json")
    tables = _read_json(report_directory / "table_metadata.json")
    if not isinstance(document_metadata, dict) or not isinstance(tables, list):
        return None
    financial_year = _financial_year(document_metadata.get("financial_year"))
    if financial_year is None:
        logger.warning("Invalid financial-year metadata in %s", report_directory)
        return None
    end_year = _financial_year_end(financial_year)
    chunk_path = project_root / "data" / "chunks" / f"{company_folder}_{end_year}.jsonl"
    if not chunk_path.is_file():
        logger.warning("Report has no matching corpus chunk file: %s", chunk_path)
        return None
    chunks = _read_evidence_chunks(chunk_path)
    metrics = {
        key: _find_metric(
            key,
            definition,
            tables,
            report_directory,
            chunks,
            company,
            financial_year,
        )
        for key, definition in METRIC_DEFINITIONS.items()
    }
    source_file = document_metadata.get("source_file")
    number_of_pages = document_metadata.get("number_of_pages")
    return AnnualReportAnalysis(
        financial_year=financial_year,
        source_file=source_file if isinstance(source_file, str) else None,
        number_of_pages=number_of_pages if isinstance(number_of_pages, int) else None,
        table_count=len(tables),
        metrics=metrics,
    )


def load_company_analysis(
    project_root: Path,
    company: str,
) -> CompanyAnalysisData:
    """Load exact, sourced metrics from existing processed report tables."""
    company_folder = _company_folder(company)
    if company_folder is None:
        raise ValueError(f"Unsupported company for analysis: {company}")
    processed_directory = project_root / "data" / "processed" / company_folder
    if not processed_directory.is_dir():
        logger.warning("Processed reports are unavailable for %s", company)
        return CompanyAnalysisData(
            company=company,
            reports=(),
            unavailable_reason="Processed annual-report data is not available for this company.",
        )

    reports = [
        report
        for report_directory in processed_directory.iterdir()
        if report_directory.is_dir()
        if (
            report := _load_report(
                company,
                company_folder,
                report_directory,
                project_root,
            )
        )
        is not None
    ]
    reports.sort(
        key=lambda report: _financial_year_end(report.financial_year),
        reverse=True,
    )
    return CompanyAnalysisData(
        company=company,
        reports=tuple(reports),
        unavailable_reason=(
            None
            if reports
            else "No processed annual-report years with matching corpus chunks are available."
        ),
    )


def chart_values(
    reports: tuple[AnnualReportAnalysis, ...],
    metric_key: str,
) -> list[dict[str, str | float]]:
    """Return only sourced observations, in ascending financial-year order."""
    points = [
        {
            "Financial year": report.financial_year,
            "Value": observation.numeric_value,
        }
        for report in reports
        if (observation := report.metrics.get(metric_key)) is not None
    ]
    return sorted(points, key=lambda point: str(point["Financial year"]))


def line_chart_svg(
    points: list[dict[str, str | float]],
    title: str,
    unit: str,
) -> str:
    """Render a dependency-free chart from sourced numeric points."""
    if not points:
        return ""
    values = [float(point["Value"]) for point in points]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Chart values must be finite numbers.")

    width, height = 700, 290
    left, right, top, bottom = 82, 18, 20, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        padding = abs(minimum) * 0.05 or 1
        minimum -= padding
        maximum += padding
    else:
        padding = (maximum - minimum) * 0.08
        minimum -= padding
        maximum += padding

    def x_position(index: int) -> float:
        if len(points) == 1:
            return left + plot_width / 2
        return left + index * plot_width / (len(points) - 1)

    def y_position(value: float) -> float:
        return top + (maximum - value) * plot_height / (maximum - minimum)

    lines = []
    for tick in range(5):
        value = maximum - tick * (maximum - minimum) / 4
        y = top + tick * plot_height / 4
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" '
            f'y2="{y:.2f}" stroke="#26364a" stroke-width="1" />'
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            'fill="#9aabbe" font-size="11">'
            f"{html.escape(format(value, ',.2f'))}</text>"
        )

    coordinates = [
        (x_position(index), y_position(value))
        for index, value in enumerate(values)
    ]
    path = " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(coordinates)
    )
    line = (
        f'<path d="{path}" fill="none" stroke="#52d8c6" '
        'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
    )
    marks = []
    for (x, y), point, value in zip(coordinates, points, values):
        financial_year = html.escape(str(point["Financial year"]))
        value_label = html.escape(format(value, ",.2f"))
        marks.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#52d8c6" '
            'stroke="#0b1220" stroke-width="2">'
            f"<title>{financial_year}: {value_label} {html.escape(unit)}</title>"
            "</circle>"
            f'<text x="{x:.2f}" y="{height - 28}" text-anchor="end" '
            f'transform="rotate(-32 {x:.2f} {height - 28})" '
            'fill="#9aabbe" font-size="11">'
            f"{financial_year}</text>"
        )

    return (
        '<div style="width:100%; overflow-x:auto">'
        f'<svg role="img" aria-label="{html.escape(title)} in {html.escape(unit)}" '
        f'viewBox="0 0 {width} {height}" width="100%" '
        'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">'
        + "".join(lines)
        + line
        + "".join(marks)
        + f'<text x="18" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_height / 2:.2f})" '
        'fill="#9aabbe" font-size="11">'
        + html.escape(unit)
        + "</text></svg></div>"
    )
