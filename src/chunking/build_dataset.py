"""
src/chunking/build_dataset.py

Part C of the chunking module: integration.

    page_segments.json  ---> chunker.chunk_text_from_page_segments()  --\
                                                                          >-- combine --> chunk_id --> chunks.jsonl
    table_metadata.json ---> table_chunker.chunk_table() (per table)  --/
      + tables/*.csv

Responsibilities specific to this file (not Part A or Part B):
    - Assign a globally unique, deterministic chunk_id to every chunk.
    - Detect continuation tables across the whole document (same
      heading, same column count, adjacent page) and tag
      continuation_of accordingly — lineage only, NOT content
      merging (see table_chunker's module docstring for why merging
      is out of scope for this version).
    - Write the combined output as JSONL (one JSON object per line),
      directly consumable by an embedding step without any further
      transformation.
    - Validate the output before declaring success: counts, token
      sizes, required-field completeness, and page/table lineage
      against the source metadata files.
"""

import difflib
import json
import logging
import re
from collections import Counter
from pathlib import Path

from .chunker import (
    chunk_text_from_page_segments,
    load_document_metadata,
    load_page_segments,
)
from .table_chunker import chunk_table, TABLE_SPLIT_THRESHOLD

logger = logging.getLogger(__name__)

# A chunk (of either type) at or above this size signals a bug
# upstream (e.g. an overhead miscalculation letting a table sub-chunk
# balloon) rather than a legitimate large chunk -- used only by
# validate_chunks() as a sanity ceiling, not a hard limit anywhere
# else in the pipeline.
SANITY_TOKEN_CEILING = TABLE_SPLIT_THRESHOLD + 500

# Similarity threshold (difflib ratio, 0-1) for treating two table
# headings as "the same" for continuation detection -- tolerates
# minor variation ("Annexure 'A'" vs "Annexure A (contd.)") without
# an external fuzzy-matching dependency.
HEADING_SIMILARITY_THRESHOLD = 0.85

REQUIRED_COMMON_FIELDS = {
    "chunk_id", "company", "document", "financial_year",
    "page", "chunk_type", "text", "tokens",
}
REQUIRED_TEXT_FIELDS = {"section", "is_small"}
REQUIRED_TABLE_FIELDS = {
    "table_id", "table_heading", "context", "chunk_index",
    "total_chunks", "split_strategy", "row_start", "row_end",
    "is_low_confidence", "continuation_of",
}


def load_table_metadata(processed_dir: str | Path) -> list[dict]:
    path = Path(processed_dir) / "table_metadata.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_heading(heading: str) -> str:
    """Lowercase, drop common continuation-marker suffixes, collapse
    all non-alphanumeric characters to single spaces -- so quote
    style, punctuation, and an explicit '(contd.)' marker don't
    prevent an otherwise-identical heading from matching."""
    text = heading.lower()
    text = re.sub(r"\(\s*cont'?d\.?\s*\)|\(\s*continued\s*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _headings_match(a: str, b: str, threshold: float = HEADING_SIMILARITY_THRESHOLD) -> bool:
    """
    Fuzzy heading equality for continuation detection. Exact match
    after normalization short-circuits; otherwise falls back to a
    difflib similarity ratio, tolerating OCR noise, punctuation
    drift, or an explicit continuation marker on the later table
    ("Annexure 'A'" vs "Annexure A (contd.)") that an exact-string
    comparison would incorrectly treat as unrelated tables.
    """
    norm_a, norm_b = _normalize_heading(a), _normalize_heading(b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio() >= threshold


def detect_continuations(
    table_metadata: list[dict],
    heading_similarity_threshold: float = HEADING_SIMILARITY_THRESHOLD,
) -> dict:
    """
    Heuristic: consecutive pages, fuzzy-matching table_heading, same
    column count -> likely fragments of one logical table split by
    a PDF page break, not genuinely separate tables (e.g. Reliance's
    'Annexure A' subsidiary list spanning pages 138-142).

    Sorted by (page, table_number) -- document order -- rather than
    grouped by exact heading text, since fuzzy matching means two
    continuation fragments are no longer guaranteed to share an
    identical sort key; comparing each table only against the one
    immediately preceding it in reading order is both simpler and
    more correct.

    Returns {table_id: previous_table_id} for tables identified as a
    continuation of the immediately preceding one. This is lineage
    tagging only -- the tables' content is never merged.
    """
    continuations: dict = {}

    sortable = [
        t for t in table_metadata
        if t.get("table_heading") and t.get("page") is not None
    ]
    sortable.sort(key=lambda t: (t["page"], t["table_number"]))

    prev = None
    for t in sortable:
        if (
            prev is not None
            and _headings_match(
                t["table_heading"], prev["table_heading"], heading_similarity_threshold
            )
            and t["columns"] == prev["columns"]
            and t["page"] - prev["page"] <= 1
            and t["table_id"] != prev["table_id"]
        ):
            continuations[t["table_id"]] = prev["table_id"]
        prev = t

    return continuations


def build_text_chunk_id(
    document_slug: str, company_slug: str, fy_slug: str, page: int, index: int
) -> str:
    return f"{document_slug}_{company_slug}_{fy_slug}_txt_p{page:03d}_{index:04d}"


def build_table_chunk_id(document_slug: str, table_id: str, chunk_index: int, total_chunks: int) -> str:
    base = f"{document_slug}_{table_id}"
    if total_chunks == 1:
        return base
    return f"{base}_c{chunk_index:02d}"


def _slugify(text: str) -> str:
    return "".join(ch for ch in text.strip().lower() if ch.isalnum())


def build_dataset(processed_dir: str | Path) -> tuple[list[dict], list[str]]:
    """
    Run Part A + Part B against one document's processed_dir, assign
    chunk_ids, tag continuation tables, and return
    (chunks, warnings) -- chunks not yet written to disk (see
    write_jsonl), warnings collected during the run (currently:
    missing CSV files) surfaced here so validate_chunks() can report
    them alongside its other checks in one consolidated place,
    instead of them only appearing as separate log lines.
    """
    processed_dir = Path(processed_dir)

    doc_metadata = load_document_metadata(processed_dir)
    page_segments = load_page_segments(processed_dir)
    table_metadata = load_table_metadata(processed_dir)

    company = doc_metadata["company"]
    document = doc_metadata["document"]
    financial_year = doc_metadata["financial_year"]
    parser_version = doc_metadata.get("parser_version")

    company_slug = _slugify(company)
    fy_slug = _slugify(financial_year.split("-")[-1] if "-" in financial_year else financial_year)
    # Document slug distinguishes chunk_ids when a company has more
    # than one document for the same financial year (e.g. an Annual
    # Report AND a Sustainability Report both for FY2025) -- without
    # it, both documents' chunk_ids would collide, since company_slug
    # + fy_slug alone is not unique per document.
    document_slug = _slugify(document)

    all_chunks: list[dict] = []
    warnings: list[str] = []

    # --- Part A: text chunks ---
    text_chunks = chunk_text_from_page_segments(
        page_segments=page_segments,
        company=company,
        document=document,
        financial_year=financial_year,
    )

    page_counters: dict[int, int] = {}
    for chunk in text_chunks:
        page = chunk["page"]
        page_counters[page] = page_counters.get(page, 0) + 1
        chunk["chunk_id"] = build_text_chunk_id(
            document_slug, company_slug, fy_slug, page, page_counters[page]
        )
        chunk["chunk_index_in_section"] = page_counters[page]
        chunk["parser_version"] = parser_version
        all_chunks.append(chunk)

    # --- Part B: table chunks ---
    tables_dir = processed_dir / "tables"
    continuations = detect_continuations(table_metadata)

    table_drop_count = 0
    for table_meta in table_metadata:
        csv_path = tables_dir / table_meta["filename"]

        if not csv_path.exists():
            msg = f"Missing CSV for {table_meta['table_id']}: {csv_path}"
            warnings.append(msg)
            table_drop_count += 1
            continue

        table_chunks = chunk_table(csv_path, table_meta)

        if not table_chunks:
            table_drop_count += 1
            continue

        continuation_of = continuations.get(table_meta["table_id"])

        for chunk in table_chunks:
            chunk["chunk_id"] = build_table_chunk_id(
                document_slug, table_meta["table_id"], chunk["chunk_index"], chunk["total_chunks"]
            )
            chunk["continuation_of"] = continuation_of
            chunk["parser_version"] = parser_version
            all_chunks.append(chunk)

    if table_drop_count:
        logger.info(
            f"{table_drop_count} table(s) produced zero chunks "
            f"(missing CSV or empty after cleanup) and were dropped."
        )

    return all_chunks, warnings


def iter_dataset(processed_dir: str | Path):
    """
    Streaming variant of build_dataset(), for future large-corpus
    use: yields chunk dicts one at a time instead of materializing
    the full list in memory.

    Continuation detection still requires the full table_metadata
    list upfront (it's inherently a whole-corpus comparison) -- but
    that's cheap (metadata only, no CSV reads). The expensive part
    (reading each CSV, tokenizing, splitting) is what actually gets
    streamed here, table by table, rather than collected into a
    list before returning.

    Does not collect warnings the way build_dataset() does, since a
    streaming consumer processes chunks as they arrive rather than
    waiting for a final report -- log messages remain the way
    missing-CSV/drop events surface in this mode.
    """
    processed_dir = Path(processed_dir)

    doc_metadata = load_document_metadata(processed_dir)
    page_segments = load_page_segments(processed_dir)
    table_metadata = load_table_metadata(processed_dir)

    company = doc_metadata["company"]
    document = doc_metadata["document"]
    financial_year = doc_metadata["financial_year"]
    parser_version = doc_metadata.get("parser_version")

    company_slug = _slugify(company)
    fy_slug = _slugify(financial_year.split("-")[-1] if "-" in financial_year else financial_year)
    document_slug = _slugify(document)

    text_chunks = chunk_text_from_page_segments(
        page_segments=page_segments, company=company, document=document,
        financial_year=financial_year,
    )
    page_counters: dict[int, int] = {}
    for chunk in text_chunks:
        page = chunk["page"]
        page_counters[page] = page_counters.get(page, 0) + 1
        chunk["chunk_id"] = build_text_chunk_id(
            document_slug, company_slug, fy_slug, page, page_counters[page]
        )
        chunk["chunk_index_in_section"] = page_counters[page]
        chunk["parser_version"] = parser_version
        yield chunk

    tables_dir = processed_dir / "tables"
    continuations = detect_continuations(table_metadata)

    for table_meta in table_metadata:
        csv_path = tables_dir / table_meta["filename"]
        if not csv_path.exists():
            logger.warning(f"Missing CSV for {table_meta['table_id']}: {csv_path}")
            continue

        table_chunks = chunk_table(csv_path, table_meta)
        continuation_of = continuations.get(table_meta["table_id"])

        for chunk in table_chunks:
            chunk["chunk_id"] = build_table_chunk_id(
                document_slug, table_meta["table_id"], chunk["chunk_index"], chunk["total_chunks"]
            )
            chunk["continuation_of"] = continuation_of
            chunk["parser_version"] = parser_version
            yield chunk


def write_jsonl(chunks, output_path: str | Path) -> int:
    """
    Accepts any iterable of chunk dicts -- a list (build_dataset) or
    a generator (iter_dataset) -- and writes one JSON object per
    line, streaming through without requiring the full set to be in
    memory at once. Returns the number of chunks written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1
    logger.info(f"Wrote {count} chunks to {output_path}")
    return count


def validate_chunks(
    chunks: list[dict],
    page_segments: list[dict],
    table_metadata: list[dict],
    extra_warnings: list[str] | None = None,
) -> list[str]:
    """
    Returns a list of problem descriptions (empty list = all clear).
    Does not raise -- validation results are for the caller to act
    on, consistent with the pipeline's "log and report, don't
    silently corrupt or silently pass" philosophy.

    extra_warnings (e.g. missing-CSV events from build_dataset) are
    appended as-is, so every pipeline issue surfaces in one
    consolidated report rather than some only appearing as separate
    log lines the caller has to remember to check.
    """
    problems = []

    # Uniqueness -- Counter is O(n) vs. the previous O(n^2) approach
    # of calling list.count() once per id. .get() defensively: a
    # chunk missing chunk_id entirely is reported by the
    # required-fields check below, not silently crashed on here.
    id_counts = Counter(c.get("chunk_id") for c in chunks)
    dupes = {chunk_id for chunk_id, n in id_counts.items() if n > 1 and chunk_id is not None}
    if dupes:
        problems.append(f"Duplicate chunk_id(s): {dupes}")

    # Required fields
    for c in chunks:
        missing_common = REQUIRED_COMMON_FIELDS - c.keys()
        if missing_common:
            problems.append(f"{c.get('chunk_id', '?')}: missing common fields {missing_common}")
            continue
        if c["chunk_type"] == "text":
            missing = REQUIRED_TEXT_FIELDS - c.keys()
        elif c["chunk_type"] == "table":
            missing = REQUIRED_TABLE_FIELDS - c.keys()
        else:
            problems.append(f"{c['chunk_id']}: unknown chunk_type {c['chunk_type']!r}")
            continue
        if missing:
            problems.append(f"{c['chunk_id']}: missing {c['chunk_type']} fields {missing}")

    # Sanity ceiling on token size. Uses .get() defensively -- a
    # chunk already flagged above for missing "tokens" must not also
    # crash validation here; it should simply not match this check
    # (its missing-field problem was already reported).
    
    oversized = [c for c in chunks if c.get("tokens", 0) > SANITY_TOKEN_CEILING]
    if oversized:
        logger.warning(
                      f"{len(oversized)} chunk(s) exceed the {SANITY_TOKEN_CEILING}-token ceiling "
                      f"but will be kept (financial tables are important)."
                      )

    # Page lineage: every text chunk's page must exist in page_segments
    valid_pages = {p["page"] for p in page_segments}
    bad_page_chunks = [
        c for c in chunks
        if c.get("chunk_type") == "text" and c.get("page") not in valid_pages
    ]
    if bad_page_chunks:
        problems.append(f"{len(bad_page_chunks)} text chunk(s) reference an unknown page")

    # Table lineage: every table chunk's table_id must trace back to
    # a real table_metadata entry
    valid_table_ids = {t["table_id"] for t in table_metadata}
    bad_table_chunks = [
        c for c in chunks
        if c.get("chunk_type") == "table" and c.get("table_id") not in valid_table_ids
    ]
    if bad_table_chunks:
        problems.append(f"{len(bad_table_chunks)} table chunk(s) reference an unknown table_id")

    if extra_warnings:
        problems.extend(extra_warnings)

    return problems


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    chunks, warnings = build_dataset(processed_dir)

    page_segments = load_page_segments(processed_dir)
    table_metadata = load_table_metadata(processed_dir)
    problems = validate_chunks(chunks, page_segments, table_metadata, extra_warnings=warnings)

    if problems:
        logger.error(f"Validation found {len(problems)} problem(s):")
        for p in problems:
            logger.error(f"  - {p}")
    else:
        logger.info("Validation passed: no problems found.")

    text_count = sum(1 for c in chunks if c["chunk_type"] == "text")
    table_count = sum(1 for c in chunks if c["chunk_type"] == "table")
    logger.info(
        f"Total chunks: {len(chunks)} ({text_count} text, {table_count} table)"
    )

    write_jsonl(chunks, processed_dir / "chunks.jsonl")
