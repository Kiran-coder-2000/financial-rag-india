"""Read-only loading, lexical search, and source resolution for corpus evidence."""

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dashboard_data import COMPANIES

logger = logging.getLogger(__name__)

RESULTS_PER_PAGE = 20
ALL_FILTER = "All"


@dataclass(frozen=True)
class TableSource:
    company: str | None
    document: str | None
    financial_year: str | None
    page: int | None
    source_file: str | None
    table_heading: str | None
    context: str | None
    table_metadata: dict[str, Any]
    csv_path: Path | None
    pdf_path: Path | None


@dataclass(frozen=True)
class EvidenceCorpus:
    chunks: tuple[dict[str, Any], ...]
    table_sources: dict[str, TableSource]
    malformed_records: int = 0
    unavailable_reason: str | None = None

    @property
    def chunk_by_id(self) -> dict[str, dict[str, Any]]:
        return {
            chunk_id: chunk
            for chunk in self.chunks
            if isinstance((chunk_id := chunk.get("chunk_id")), str) and chunk_id
        }

    @property
    def companies(self) -> tuple[str, ...]:
        return tuple(sorted({
            company
            for chunk in self.chunks
            if isinstance((company := chunk.get("company")), str) and company
        }))

    @property
    def financial_years(self) -> tuple[str, ...]:
        years = {
            year
            for chunk in self.chunks
            if isinstance((year := chunk.get("financial_year")), str) and year
        }
        return tuple(sorted(years, key=_year_sort_key, reverse=True))

    @property
    def documents(self) -> tuple[str, ...]:
        return tuple(sorted({
            document
            for chunk in self.chunks
            if isinstance((document := chunk.get("document")), str) and document
        }))

    @property
    def chunk_types(self) -> tuple[str, ...]:
        return tuple(sorted({
            chunk_type
            for chunk in self.chunks
            if isinstance((chunk_type := chunk.get("chunk_type")), str) and chunk_type
        }))

    @property
    def available_reports(self) -> int:
        return len({
            (chunk.get("company"), chunk.get("document"), chunk.get("financial_year"))
            for chunk in self.chunks
            if all(
                isinstance(chunk.get(field), str) and chunk.get(field)
                for field in ("company", "document", "financial_year")
            )
        })


def _year_sort_key(year: str) -> tuple[int, str]:
    match = re.search(r"(\d{4})$", year)
    return (int(match.group(1)) if match else 0, year)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        logger.warning("Could not read evidence metadata from %s: %s", path, error)
        return None


def _safe_child(directory: Path, filename: Any) -> Path | None:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        return None
    path = directory / filename
    try:
        if path.resolve().parent != directory.resolve():
            return None
    except OSError:
        return None
    return path if path.is_file() else None


def _load_table_sources(project_root: Path) -> dict[str, TableSource]:
    processed_directory = project_root / "data" / "processed"
    raw_directory = project_root / "data" / "raw"
    if not processed_directory.is_dir():
        return {}

    sources: dict[str, TableSource] = {}
    for company_directory in processed_directory.iterdir():
        if not company_directory.is_dir():
            continue
        for report_directory in company_directory.iterdir():
            if not report_directory.is_dir():
                continue
            document_metadata = _read_json(report_directory / "document_metadata.json")
            tables = _read_json(report_directory / "table_metadata.json")
            if not isinstance(document_metadata, dict) or not isinstance(tables, list):
                continue
            source_file_value = document_metadata.get("source_file")
            source_file = (
                source_file_value
                if isinstance(source_file_value, str) and Path(source_file_value).name == source_file_value
                else None
            )
            pdf_path = _safe_child(raw_directory, source_file)
            for table in tables:
                if not isinstance(table, dict):
                    continue
                table_id = table.get("table_id")
                if not isinstance(table_id, str) or not table_id:
                    continue
                csv_path = _safe_child(report_directory / "tables", table.get("filename"))
                page_value = table.get("page")
                sources[table_id] = TableSource(
                    company=(
                        table.get("company")
                        if isinstance(table.get("company"), str)
                        else document_metadata.get("company")
                        if isinstance(document_metadata.get("company"), str)
                        else None
                    ),
                    document=(
                        table.get("document")
                        if isinstance(table.get("document"), str)
                        else document_metadata.get("document")
                        if isinstance(document_metadata.get("document"), str)
                        else None
                    ),
                    financial_year=(
                        table.get("financial_year")
                        if isinstance(table.get("financial_year"), str)
                        else document_metadata.get("financial_year")
                        if isinstance(document_metadata.get("financial_year"), str)
                        else None
                    ),
                    page=page_value if isinstance(page_value, int) and not isinstance(page_value, bool) else None,
                    source_file=source_file,
                    table_heading=(
                        table.get("table_heading")
                        if isinstance(table.get("table_heading"), str)
                        else None
                    ),
                    context=table.get("context") if isinstance(table.get("context"), str) else None,
                    table_metadata=table,
                    csv_path=csv_path,
                    pdf_path=pdf_path,
                )
    return sources


def load_evidence_corpus(project_root: Path) -> EvidenceCorpus:
    """Load valid JSONL records without changing or reranking their source data."""
    chunks_path = project_root / "data" / "chunks" / "all_chunks.jsonl"
    chunks: list[dict[str, Any]] = []
    malformed_records = 0
    try:
        with chunks_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    malformed_records += 1
                    logger.warning(
                        "Skipping malformed corpus record in %s at line %s: %s",
                        chunks_path,
                        line_number,
                        error,
                    )
                    continue
                if not isinstance(record, dict):
                    malformed_records += 1
                    logger.warning(
                        "Skipping non-object corpus record in %s at line %s",
                        chunks_path,
                        line_number,
                    )
                    continue
                chunks.append(record)
    except (OSError, UnicodeError) as error:
        logger.warning("Could not read evidence corpus from %s: %s", chunks_path, error)
        return EvidenceCorpus(
            chunks=(),
            table_sources=_load_table_sources(project_root),
            malformed_records=malformed_records,
            unavailable_reason="The corpus chunk file could not be read.",
        )

    return EvidenceCorpus(
        chunks=tuple(chunks),
        table_sources=_load_table_sources(project_root),
        malformed_records=malformed_records,
        unavailable_reason=None if chunks else "No valid corpus records are available.",
    )


def source_for_chunk(
    corpus: EvidenceCorpus,
    chunk: dict[str, Any],
) -> TableSource | None:
    table_id = chunk.get("table_id")
    if not isinstance(table_id, str):
        return None
    source = corpus.table_sources.get(table_id)
    if source is None:
        return None
    if (
        source.company != chunk.get("company")
        or source.document != chunk.get("document")
        or source.financial_year != chunk.get("financial_year")
        or source.page != chunk.get("page")
    ):
        return None
    return source


def _query_terms(query: str) -> list[str]:
    return re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)


def search_chunks(
    chunks: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    company: str | None = None,
    financial_year: str | None = None,
    document: str | None = None,
    chunk_type: str | None = None,
    page: int | None = None,
    query: str = "",
) -> list[dict[str, Any]]:
    """Filter exact corpus records, then match every query term lexically."""
    terms = _query_terms(query)
    results = []
    for chunk in chunks:
        if company is not None and chunk.get("company") != company:
            continue
        if financial_year is not None and chunk.get("financial_year") != financial_year:
            continue
        if document is not None and chunk.get("document") != document:
            continue
        if chunk_type is not None and chunk.get("chunk_type") != chunk_type:
            continue
        if page is not None and chunk.get("page") != page:
            continue
        if terms:
            searchable_text = " ".join(
                str(chunk.get(field) or "")
                for field in ("text", "table_heading", "context", "section")
            ).casefold()
            if not all(term in searchable_text for term in terms):
                continue
        results.append(chunk)
    return results


def exact_chunk_lookup(
    chunks: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    chunk_id: str,
) -> dict[str, Any] | None:
    """Return only an exact chunk ID match; never perform a fuzzy fallback."""
    return next(
        (
            chunk
            for chunk in chunks
            if isinstance(chunk.get("chunk_id"), str) and chunk["chunk_id"] == chunk_id
        ),
        None,
    )


def result_excerpt(chunk: dict[str, Any], query: str, max_length: int = 320) -> str:
    text = " ".join(
        str(chunk.get(field) or "")
        for field in ("table_heading", "section", "text", "context")
    ).strip()
    if len(text) <= max_length:
        return text
    first_term = next(
        (term for term in _query_terms(query) if term in text.casefold()),
        "",
    )
    position = text.casefold().find(first_term.casefold()) if first_term else 0
    start = max(0, position - max_length // 3)
    end = min(len(text), start + max_length)
    excerpt = text[start:end]
    if start:
        excerpt = "… " + excerpt
    if end < len(text):
        excerpt += " …"
    return excerpt


def paginate_results(
    results: list[dict[str, Any]],
    page_number: int,
    page_size: int = RESULTS_PER_PAGE,
) -> tuple[list[dict[str, Any]], int]:
    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    page_count = max(1, (len(results) + page_size - 1) // page_size)
    page_number = min(max(page_number, 1), page_count)
    start = (page_number - 1) * page_size
    return results[start : start + page_size], page_count


def read_structured_table(source: TableSource) -> list[list[str]] | None:
    if source.csv_path is None:
        return None
    try:
        with source.csv_path.open(encoding="utf-8-sig", newline="") as file:
            return list(csv.reader(file))
    except (OSError, UnicodeError, csv.Error) as error:
        logger.warning("Could not read structured evidence table %s: %s", source.csv_path, error)
        return None


def display_company_name(canonical_name: Any) -> str:
    for display_name, canonical in COMPANIES:
        if canonical_name == canonical:
            return display_name
    return str(canonical_name) if canonical_name else "Not available"
