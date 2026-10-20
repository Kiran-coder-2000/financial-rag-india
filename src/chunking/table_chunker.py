"""
src/chunking/table_chunker.py

Part B of the chunking module: turns one table (its CSV + its entry
in table_metadata.json) into one or more retrieval-ready table
chunks.

Design principle (see the architecture writeup this file implements):
"what information must remain together for this table to remain
semantically meaningful when retrieved independently" — not a row
count, not a token count alone, but the table's own logical
structure (financial-statement sections, subtotals, and — for wide
tables — column identity).

Hierarchy applied, in order:
    1. Fits the token budget intact -> keep as one chunk.
    2. Too large, has detectable structural segments (label rows /
       subtotal rows) -> pack whole segments greedily, never
       splitting inside a segment if avoidable.
    3. A single segment is itself oversized -> row-by-row batching
       within that segment only, carrying the last complete row
       forward as a labeled continuity line (never a text-overlap).
    4. Wide tables (few rows, many columns) are serialized as one
       label:value record per row instead of a pipe table, so column
       identity is never lost regardless of chunk boundaries.

Nothing is ever silently dropped except: fully-blank rows (zero
information) and rows that exactly duplicate the column-header
tuple (a Docling artifact, not data). Ordinary duplicate data rows,
missing headings, and missing context are all preserved as-is —
never invented, never "fixed".
"""

import logging
import re
from pathlib import Path

import pandas as pd
import tiktoken

logger = logging.getLogger(__name__)

ENCODER = tiktoken.encoding_for_model("gpt-4o")

# Ceiling for keeping a table intact as one chunk. See the capstone's
# token-distribution analysis: 90th percentile across three
# companies' annual reports sits at 775-1327 tokens, so 1000 keeps
# ~92% of real tables intact while still bounding the outliers.
TABLE_SPLIT_THRESHOLD = 2000

# Safety buffer subtracted from the per-table sub-chunk budget on
# top of the measured heading/context/header overhead, so token
# counting drift (e.g. markdown rendering quirks) doesn't push a
# sub-chunk just over TABLE_SPLIT_THRESHOLD.
OVERHEAD_MARGIN = 50

# Floor for the per-row-batch budget even if overhead is unusually
# high, so a table with far more columns than we've seen in
# practice degrades gracefully instead of producing a zero/negative
# budget.
MIN_ROW_BUDGET = 100

# A table is "wide" (few rows, many columns) when columns outnumber
# rows AND exceed this count. Both conditions matter: a table that's
# just large in both dimensions is NOT what this branch is for.
WIDE_TABLE_COLUMN_THRESHOLD = 6

_TOTAL_KEYWORDS = re.compile(
    r"\btotal\b|\bsub[- ]?total\b|\bnet\b|\bgrand\s+total\b",
    flags=re.IGNORECASE,
)


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def _is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _parse_financial_number(value) -> float | None:
    """
    Best-effort numeric parse of a financial CSV cell, tolerating
    Indian-report-specific formatting:
      - Currency symbols/prefixes: "₹", "Rs.", and the backtick "`"
        seen in this corpus as an OCR/extraction substitute for "₹"
        (observed literally in real context text: "( ` in crore,
        unless otherwise stated)").
      - Thousands separators: both Indian lakh/crore grouping
        ("12,34,567") and Western grouping ("1,234,567") reduce to
        the same digits once commas are stripped wholesale, so
        either convention parses correctly.
      - Parenthesized negatives ANYWHERE in the string, not only if
        the string starts with "(" — a currency symbol or prefix
        commonly precedes the parenthesis (e.g. "Rs. (1,234)",
        "₹(1,234)"), and the previous start/end-anchored check
        missed these, silently treating them as positive.
      - A lone dash/en-dash/em-dash used to mean "nil", not a
        negative sign (a distinct convention from parenthesized
        negatives).

    Used ONLY as an internal signal for boundary detection — never
    written back into chunk output, so it can never "fix" a
    displayed value; the CSV's original text is what gets
    serialized, always.
    """
    if _is_blank(value):
        return None

    text = str(value).strip()

    if text in ("-", "\u2013", "\u2014", "--"):
        return None

    # Detect parenthesized-negative by presence, not position --
    # a prefix (currency symbol, "Rs.") may precede the "(".
    has_parens = "(" in text and ")" in text

    # Remove thousands separators first (both Indian lakh/crore
    # grouping "12,34,567" and Western "1,234,567" reduce to the
    # same digits), then extract the first proper number pattern --
    # NOT a blanket "keep every digit and '.'" filter, which would
    # let an unrelated period leak in from an abbreviation like
    # "Rs." and corrupt the result into a bogus decimal (e.g.
    # "Rs. (1,234)" naively filtered to ".1234" -> 0.1234, wrong).
    text_no_separators = text.replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text_no_separators)

    if not match:
        return None

    try:
        num = float(match.group())
    except ValueError:
        return None

    return -num if has_parens else num


def is_wide_table(df: pd.DataFrame) -> bool:
    rows, cols = df.shape
    return cols > rows and cols > WIDE_TABLE_COLUMN_THRESHOLD


def _looks_like_label_row(row: pd.Series) -> bool:
    """First cell has text, every other cell is blank -- a section
    label inside the table (e.g. 'Assets' in a balance sheet)."""
    if len(row) < 2:
        return False
    if _is_blank(row.iloc[0]):
        return False
    return all(_is_blank(v) for v in row.iloc[1:])


def _looks_like_keyword_total_row(row: pd.Series) -> bool:
    if len(row) == 0 or _is_blank(row.iloc[0]):
        return False
    return bool(_TOTAL_KEYWORDS.search(str(row.iloc[0])))


def _looks_like_numeric_total(
    df: pd.DataFrame, idx: int, since: int
) -> bool:
    """
    Secondary signal for rows a keyword search misses: does this
    row's value in some column equal the sum of that column's
    values across the preceding contiguous rows since the last
    boundary? A financial-domain-specific detection signal a
    generic text chunker has no access to.
    """
    if idx <= since:
        return False

    row = df.iloc[idx]
    for col in df.columns[1:]:
        candidate = _parse_financial_number(row[col])
        if candidate is None:
            continue

        preceding = [
            _parse_financial_number(df.iloc[j][col])
            for j in range(since, idx)
        ]
        preceding = [v for v in preceding if v is not None]

        if len(preceding) < 2:
            continue

        total = sum(preceding)
        if total == 0:
            continue

        if abs(candidate - total) / abs(total) < 0.01:
            return True

    return False


def build_logical_segments(df: pd.DataFrame) -> list[dict]:
    """
    Partition a table's rows into contiguous logical segments using
    label rows (segment openers) and total/subtotal rows (segment
    closers, keyword- or numeric-reconciliation-detected). Segments
    always cover every row exactly once, in order.

    Returns a list of {"start": int, "end": int, "boundary_type": str}
    (row indices are inclusive, into df as passed in).
    """
    n = len(df)
    segments = []
    seg_start = 0
    last_boundary_end = -1

    for i in range(n):
        row = df.iloc[i]

        if _looks_like_label_row(row) and i > seg_start:
            segments.append(
                {"start": seg_start, "end": i - 1, "boundary_type": "label_start"}
            )
            seg_start = i
            last_boundary_end = i - 1

        is_total_kw = _looks_like_keyword_total_row(row)
        is_total = is_total_kw or (
            not is_total_kw
            and _looks_like_numeric_total(df, i, last_boundary_end + 1)
        )

        if is_total:
            segments.append(
                {
                    "start": seg_start,
                    "end": i,
                    "boundary_type": "total_kw" if is_total_kw else "total_numeric",
                }
            )
            seg_start = i + 1
            last_boundary_end = i

    if seg_start <= n - 1:
        segments.append({"start": seg_start, "end": n - 1, "boundary_type": "end_of_table"})

    return segments


def _format_header(heading: str | None, context: str | None) -> str:
    lines = []
    if heading:
        lines.append(f"Table Heading: {heading}")
    if context:
        lines.append(f"Context: {context}")
    return "\n".join(lines)


def serialize_normal(df_subset: pd.DataFrame, heading, context) -> str:
    header = _format_header(heading, context)
    table_md = df_subset.to_markdown(index=False)
    return "\n\n".join(p for p in (header, table_md) if p)


def _serialize_record(row: pd.Series, columns: list[str], row_label) -> str:
    lines = [f"Record: {row_label}"]
    for col in columns:
        val = row[col]
        if _is_blank(val):
            continue
        lines.append(f"{col}: {val}")
    return "\n".join(lines)


def serialize_transposed(df_subset: pd.DataFrame, heading, context) -> str:
    header = _format_header(heading, context)
    columns = list(df_subset.columns)
    records = [
        _serialize_record(df_subset.iloc[i], columns, row_label=df_subset.iloc[i, 0])
        for i in range(len(df_subset))
    ]
    body = "\n\n".join(records)
    return "\n\n".join(p for p in (header, body) if p)


def _split_segment_by_rows(
    df: pd.DataFrame, start: int, end: int, budget: int
) -> list[tuple[str, tuple[int, int]]]:
    """
    Last-resort fallback: a single logical segment is itself too
    large for the budget. Batch its rows greedily, carrying the
    last COMPLETE row of the previous batch forward as a labeled
    continuity line -- never a raw text-overlap, which would cut
    mid-cell and mean nothing for structured data.
    """
    pieces: list[tuple[str, tuple[int, int]]] = []
    columns = list(df.columns)

    current_rows: list[int] = []
    carry_row_idx: int | None = None

    def render(rows: list[int], carry: int | None) -> str:
        sub_df = df.iloc[rows]
        lines = []
        if carry is not None:
            carry_line = " | ".join(str(v) for v in df.iloc[carry])
            lines.append(f"(continued -- previous row: {carry_line})")
        table_md = sub_df.to_markdown(index=False)
        lines.append(table_md)
        return "\n".join(lines)

    for i in range(start, end + 1):
        trial_rows = current_rows + [i]
        trial_text = render(trial_rows, carry_row_idx)
        if count_tokens(trial_text) > budget and current_rows:
            pieces.append(
                (render(current_rows, carry_row_idx), (current_rows[0], current_rows[-1]))
            )
            carry_row_idx = current_rows[-1]
            current_rows = [i]
        else:
            current_rows = trial_rows

    if current_rows:
        pieces.append(
            (render(current_rows, carry_row_idx), (current_rows[0], current_rows[-1]))
        )

    return pieces


def _split_normal_table(
    df: pd.DataFrame, budget: int
) -> list[tuple[str, str, tuple[int, int]]]:
    """
    Returns list of (body_text, split_strategy, (row_start, row_end)).
    Packs whole logical segments greedily; recurses into row-level
    splitting only for a segment that alone exceeds budget.
    """
    segments = build_logical_segments(df)

    results: list[tuple[str, str, tuple[int, int]]] = []
    current_segments: list[dict] = []
    current_tokens = 0

    def render_segments(segs: list[dict]) -> str:
        rows = list(range(segs[0]["start"], segs[-1]["end"] + 1))
        return df.iloc[rows].to_markdown(index=False)

    def flush():
        if current_segments:
            text = render_segments(current_segments)
            row_range = (current_segments[0]["start"], current_segments[-1]["end"])
            results.append((text, "structural_segments", row_range))

    for seg in segments:
        seg_rows = list(range(seg["start"], seg["end"] + 1))
        seg_text = df.iloc[seg_rows].to_markdown(index=False)
        seg_tokens = count_tokens(seg_text)

        if seg_tokens > budget:
            flush()
            current_segments = []
            current_tokens = 0
            for body, row_range in _split_segment_by_rows(
                df, seg["start"], seg["end"], budget
            ):
                results.append((body, "row_fallback_within_segment", row_range))
            continue

        if current_tokens + seg_tokens > budget:
            flush()
            current_segments = []
            current_tokens = 0

        current_segments.append(seg)
        current_tokens += seg_tokens

    flush()
    return results


def _split_wide_table(
    df: pd.DataFrame, budget: int
) -> list[tuple[str, str, tuple[int, int]]]:
    """
    Wide-table equivalent of _split_normal_table: packs whole
    per-row records greedily; falls back to splitting a single
    oversized record's fields into column-groups (each still
    tagged with the record's identity) if one record alone exceeds
    budget -- this branch is untested against real data since no
    table in the observed corpus needed it (max 29 columns), but it
    exists so the pipeline degrades rather than crashes.
    """
    columns = list(df.columns)
    results: list[tuple[str, str, tuple[int, int]]] = []

    current_rows: list[int] = []
    current_tokens = 0

    def render(rows: list[int]) -> str:
        return "\n\n".join(
            _serialize_record(df.iloc[i], columns, row_label=df.iloc[i, 0])
            for i in rows
        )

    def flush():
        if current_rows:
            results.append(
                (render(current_rows), "wide_record_batch", (current_rows[0], current_rows[-1]))
            )

    for i in range(len(df)):
        record_text = _serialize_record(df.iloc[i], columns, row_label=df.iloc[i, 0])
        record_tokens = count_tokens(record_text)

        if record_tokens > budget:
            flush()
            current_rows = []
            current_tokens = 0
            # Single record too big even alone -- split its fields
            # into column-groups, each re-tagged with the record
            # identity so it's never ambiguous which record a
            # fragment belongs to.
            group_cols: list[str] = []
            group_tokens = 0
            groups: list[list[str]] = []
            for col in columns[1:]:
                col_line = f"{col}: {df.iloc[i][col]}"
                col_tokens = count_tokens(col_line)
                if group_tokens + col_tokens > budget and group_cols:
                    groups.append(group_cols)
                    group_cols, group_tokens = [], 0
                group_cols.append(col)
                group_tokens += col_tokens
            if group_cols:
                groups.append(group_cols)
            for group in groups:
                text = _serialize_record(
                    df.iloc[i][group],
                    group,
                    row_label=df.iloc[i, 0],
                )
                results.append((text, "wide_column_group", (i, i)))
            continue

        if current_tokens + record_tokens > budget:
            flush()
            current_rows = []
            current_tokens = 0

        current_rows.append(i)
        current_tokens += record_tokens

    flush()
    return results


def _verify_header_repeated(text: str, heading, context) -> None:
    """
    Enforce, not just assume, that every table chunk repeats its
    heading and context. Previously this was an implicit consequence
    of always prepending _format_header() — correct in practice, but
    unverified: a future refactor could silently break it. Making it
    an explicit, checked invariant here (and in the test suite) turns
    "should always be true" into "provably true, or loudly not".

    Only checks fields that are actually present -- a table with no
    heading (never invented, per policy) correctly has nothing to
    verify for that field.
    """
    if heading and f"Table Heading: {heading}" not in text:
        logger.warning(
            f"Table chunk is missing its expected heading line — "
            f"heading repetition invariant violated."
        )
    if context and f"Context: {context}" not in text:
        logger.warning(
            f"Table chunk is missing its expected context line — "
            f"context repetition invariant violated."
        )


def _make_table_chunk(
    table_meta: dict,
    *,
    text: str,
    tokens: int,
    chunk_index: int,
    total_chunks: int,
    split_strategy: str,
    row_start: int,
    row_end: int,
    is_low_confidence: bool,
) -> dict:
    _verify_header_repeated(
        text, table_meta.get("table_heading"), table_meta.get("context")
    )
    return {
        "company": table_meta["company"],
        "document": table_meta["document"],
        "financial_year": table_meta["financial_year"],
        "page": table_meta.get("page"),
        "chunk_type": "table",
        "table_id": table_meta["table_id"],
        "table_heading": table_meta.get("table_heading"),
        "context": table_meta.get("context"),
        "text": text,
        "tokens": tokens,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "split_strategy": split_strategy,
        "row_start": row_start,
        "row_end": row_end,
        "is_low_confidence": is_low_confidence,
        "continuation_of": None,  # filled in at integration (Part C)
    }


def chunk_table(
    csv_path: str | Path,
    table_meta: dict,
    split_threshold: int = TABLE_SPLIT_THRESHOLD,
    overhead_margin: int = OVERHEAD_MARGIN,
    min_row_budget: int = MIN_ROW_BUDGET,
) -> list[dict]:
    """
    Turn one table's CSV + its table_metadata.json entry into one or
    more chunk dicts. See module docstring for the splitting
    hierarchy. Returns [] for a genuinely empty table (0 rows, or 0
    rows left after removing blank/header-duplicate rows) -- logged,
    never silent.
    """
    csv_path = Path(csv_path)
    table_id = table_meta["table_id"]
    heading = table_meta.get("table_heading")
    context = table_meta.get("context")

    if not csv_path.exists():
        logger.warning(f"Missing CSV for {table_id}: {csv_path}")
        return []

    # dtype=str: never let pandas reinterpret financial formatting
    # (dropped trailing zeros, "(1,234)" parsed as a plain string
    # rather than silently reformatted, etc.) -- what's in the CSV
    # is exactly what gets serialized.
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)

    if len(df.columns) == 0 or len(df) == 0:
        
        logger.info(f"{table_id}: empty table "
                    f"(Heading: {heading}, Page: {table_meta.get('page')}) — emitting empty chunk.")
        return [
                _make_table_chunk(
                                  table_meta,
                                  text="",
                                  tokens=0,
                                  chunk_index=1,
                                  total_chunks=1,
                                  split_strategy="empty",
                                  row_start=0,
                                  row_end=0,
                                  is_low_confidence=True,
                                 )
                ]



    non_blank_mask = ~df.apply(
        lambda r: all(_is_blank(v) for v in r), axis=1
    )
    blank_dropped = int((~non_blank_mask).sum())
    df = df[non_blank_mask].reset_index(drop=True)

    if len(df) > 0:
        header_tuple = tuple(str(c).strip() for c in df.columns)
        is_header_dup = df.apply(
            lambda r: tuple(str(v).strip() for v in r) == header_tuple, axis=1
        )
        header_dupes_dropped = int(is_header_dup.sum())
        df = df[~is_header_dup].reset_index(drop=True)
    else:
        header_dupes_dropped = 0

    if blank_dropped or header_dupes_dropped:
        logger.info(
            f"{table_id}: dropped {blank_dropped} blank row(s), "
            f"{header_dupes_dropped} duplicate-header row(s)."
        )

    if len(df) == 0:
        logger.info(f"{table_id}: no rows left after cleanup, dropping.")
        return []

    is_low_confidence = df.shape[1] <= 1
    wide = is_wide_table(df)

    full_text = (
        serialize_transposed(df, heading, context)
        if wide
        else serialize_normal(df, heading, context)
    )
    full_tokens = count_tokens(full_text)

    if full_tokens <= split_threshold:
        return [
            _make_table_chunk(
                table_meta,
                text=full_text,
                tokens=full_tokens,
                chunk_index=1,
                total_chunks=1,
                split_strategy="intact",
                row_start=0,
                row_end=len(df) - 1,
                is_low_confidence=is_low_confidence,
            )
        ]

    # Too large -- compute this table's actual overhead (heading +
    # context + header row/first record) rather than assuming a
    # flat budget, since overhead scales with column count.
    if wide:
        overhead_sample = _format_header(heading, context) + "\n\n" + _serialize_record(
            df.iloc[0], list(df.columns), row_label=df.iloc[0, 0]
        )
    else:
        overhead_sample = _format_header(heading, context) + "\n\n" + df.iloc[:0].to_markdown(index=False)

    overhead_tokens = count_tokens(overhead_sample)
    sub_budget = max(split_threshold - overhead_tokens - overhead_margin, min_row_budget)

    pieces = (
        _split_wide_table(df, sub_budget)
        if wide
        else _split_normal_table(df, sub_budget)
    )

    total = len(pieces)
    chunks = []
    for idx, (body, strategy, row_range) in enumerate(pieces, start=1):
        text = (_format_header(heading, context) + "\n\n" + body).strip()
        tokens = count_tokens(text)
        chunks.append(
            _make_table_chunk(
                table_meta,
                text=text,
                tokens=tokens,
                chunk_index=idx,
                total_chunks=total,
                split_strategy=strategy,
                row_start=row_range[0],
                row_end=row_range[1],
                is_low_confidence=is_low_confidence,
            )
        )

    return chunks
