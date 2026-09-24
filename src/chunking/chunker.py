"""
src/chunking/chunker.py

Chunking module for the financial RAG pipeline.

Builds retrieval-ready chunks from a single document's processed
output (produced by DocumentProcessor):

    <output_dir>/
        page_segments.json      -> [{"page": int, "text": str}, ...]
        table_metadata.json     -> [{"table_id", "table_number", "page",
                                      "filename", "company", "document",
                                      "financial_year", "table_heading",
                                      "context", "rows", "columns"}, ...]
        document_metadata.json  -> {"company", "document",
                                      "financial_year", "parser_version",
                                      ...}
        tables/*.csv

Built incrementally, in three parts:
    Part A — text chunking     (this file, so far)
    Part B — table chunking    (implemented in table_chunker file)
    Part C — integration       (implemented in build_dataset file)

Part A responsibility: turn page_segments.json into page- and
section-aware text chunks.

Docling renders tables inline as pipe-delimited markdown rows within
the page text, so those rows are stripped before splitting — tables
are handled entirely by Part B, reading directly from the CSVs.
Leaving them in here would (a) duplicate every table's data as a
second, corrupted copy in the text chunks, since a naive splitter
has no concept of "don't cut through a table row", and (b) pollute
retrieval with garbled fragments.
"""

import json
import logging
import re
from pathlib import Path
import sys

import tiktoken

logger = logging.getLogger(__name__)

# Same tokenizer used for the table token-budget analysis (see the
# capstone's table-token-count script), so a "400 token" text chunk
# and a "1000 token" table-split threshold are measured on the same
# scale rather than two different tokenizers silently disagreeing.
ENCODER = tiktoken.encoding_for_model("gpt-4o")

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60

# A chunk this small after table-stripping carries little retrievable
# value — e.g. a bare "## Balance Sheet" heading with its table rows
# removed leaves only the heading and a date line. Dropped rather
# than kept as noise; count is logged so nothing vanishes silently.
MIN_CHUNK_TOKENS = 10

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def count_tokens(text: str) -> int:
    """Token count using the same tokenizer as the rest of the
    pipeline's token-budget decisions."""
    return len(ENCODER.encode(text))


def strip_markdown_tables(text: str) -> str:
    """
    Remove Docling's inline markdown table rows from page text.

    Matches lines that are pipe-delimited table rows, allowing
    leading whitespace (some nested/continuation table rows are
    indented — a plain `^\\|` anchor misses those). Collapses the
    resulting run of blank lines left behind afterward, since
    re.sub removes each matched line's content but not its newline.
    """
    without_tables = re.sub(pattern = r"^[ \t]*\|.*$", repl = "", string = text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", without_tables).strip()


_ATX_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", flags=re.MULTILINE)
# A line that is ENTIRELY bold markdown (e.g. "**Balance Sheet**")
# on its own line -- a heading convention Docling sometimes emits
# instead of an ATX '#'.
_BOLD_LINE_PATTERN = re.compile(r"^\*\*(.+?)\*\*\s*$", flags=re.MULTILINE)
# Setext-style headings: a text line immediately followed by a line
# of only '=' (level 1) or '---' (level 2, 3+ dashes required so a
# 1-2 dash em-dash-like artifact doesn't false-trigger). The
# candidate heading line is capped at 10 words -- real headings are
# short; without this cap, a '---' used as a plain horizontal-rule
# divider after an ordinary sentence would be misread as a heading.
_SETEXT_PATTERN = re.compile(r"^(.{1,80})\n(=+|-{3,})[ \t]*$", flags=re.MULTILINE)


def _find_all_headers(text: str) -> list[tuple[int, int, str]]:
    """
    Collect header matches from all supported conventions (ATX,
    bold-line, Setext), each as (start, end, header_text) where
    `end` is the index immediately after the header (before its
    content begins). Overlapping matches are resolved by keeping
    whichever was found first when scanning left to right.
    """
    candidates: list[tuple[int, int, str]] = []

    for m in _ATX_PATTERN.finditer(text):
        candidates.append((m.start(), m.end(), m.group(2).strip()))

    for m in _BOLD_LINE_PATTERN.finditer(text):
        candidates.append((m.start(), m.end(), m.group(1).strip()))

    for m in _SETEXT_PATTERN.finditer(text):
        heading_text = m.group(1).strip()
        if 0 < len(heading_text.split()) <= 10:
            candidates.append((m.start(), m.end(), heading_text))

    candidates.sort(key=lambda c: c[0])

    resolved: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, header_text in candidates:
        if start < last_end:
            continue  # overlaps a match already kept
        resolved.append((start, end, header_text))
        last_end = end

    return resolved


def split_on_headers(text: str) -> list[dict]:
    """
    Split page text into (section, content) blocks on headings,
    recognizing ATX ('#'-'######'), bold-line-only ('**Heading**'),
    and Setext ('Heading\\n===' / 'Heading\\n---') conventions —
    Docling's own section headers are typically ATX, but the other
    two are included since heading style can vary by document/
    version. Content appearing before the first heading on the page
    gets section=None, per spec.

    Returns a list of {"section": str | None, "content": str}, in
    document order. Header-scoped to a single page's text only —
    this function is called once per page, so a section never spans
    a page boundary.
    """
    matches = _find_all_headers(text)

    blocks = []

    if not matches:
        content = text.strip()
        if content:
            blocks.append({"section": None, "content": content})
        return blocks

    preamble = text[: matches[0][0]].strip()
    if preamble:
        blocks.append({"section": None, "content": preamble})

    for i, (start, end, header_text) in enumerate(matches):
        content_start = end
        content_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        if content:
            blocks.append({"section": header_text, "content": content})

    return blocks


def recursive_split(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Token-aware recursive splitter: tries the coarsest separator
    first (paragraph breaks), falling back to finer separators
    (line breaks, sentence-ish periods, spaces, then raw token
    count as a last resort) for any piece still over chunk_size.

    Overlap is applied exactly once, on the final flat list of
    pieces, AFTER all splitting (including any nested recursion
    into an oversized piece) is complete — applying it at each
    recursion level as well would double-count the overlap on any
    chunk that came from a nested split, inflating it well past
    chunk_size + chunk_overlap.
    """
    raw_pieces = _split_no_overlap(text, chunk_size, _SEPARATORS)
    return _apply_overlap(raw_pieces, chunk_overlap)


def _split_no_overlap(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    """
    Pure recursive splitting — never applies overlap, at any
    recursion depth. See recursive_split's docstring for why.
    """
    if count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
        return _split_by_tokens(text, chunk_size)

    pieces = [p for p in text.split(separator) if p.strip()]

    if len(pieces) == 1:
        # This separator didn't actually divide the text (e.g. one
        # giant paragraph with no blank lines) — fall through to a
        # finer separator instead of looping on the same split.
        if remaining_separators:
            return _split_no_overlap(text, chunk_size, remaining_separators)
        return _split_by_tokens(text, chunk_size)

    return _merge_pieces(pieces, separator, chunk_size, remaining_separators)


def _merge_pieces(
    pieces: list[str],
    separator: str,
    chunk_size: int,
    remaining_separators: list[str],
) -> list[str]:
    """
    Greedily pack pieces back together up to chunk_size tokens,
    recursing into any single piece that alone exceeds chunk_size
    using the next-finer separator. No overlap applied here.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append(separator.join(current).strip())
        current = []
        current_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece)

        if piece_tokens > chunk_size:
            flush()
            chunks.extend(
                _split_no_overlap(piece, chunk_size, remaining_separators)
            )
            continue

        if current_tokens + piece_tokens > chunk_size:
            flush()

        current.append(piece)
        current_tokens += piece_tokens

    flush()
    return chunks


def _word_safe_tail(text: str, chunk_overlap: int) -> str:
    """
    Build a word-boundary-safe suffix of `text` totaling
    approximately chunk_overlap tokens.

    Slicing raw token IDs (the previous approach) can cut a word
    mid-token when that word spans multiple BPE tokens, producing an
    overlap that starts with a broken sub-word fragment (e.g. "ing
    the report" instead of "reporting the"). Building the tail from
    whole words instead guarantees every overlap starts cleanly,
    at the cost of the token count being approximate rather than
    exact — an acceptable trade for never emitting a fragment.
    """
    words = text.split()
    tail_words: list[str] = []
    tail_tokens = 0

    for word in reversed(words):
        word_tokens = count_tokens(word)
        if tail_tokens + word_tokens > chunk_overlap and tail_words:
            break
        tail_words.insert(0, word)
        tail_tokens += word_tokens

    return " ".join(tail_words)


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    """
    Prefix each chunk (after the first) with a word-safe overlap
    tail taken from the end of the previous chunk, so context
    carries across a hard chunk boundary without ever starting on a
    broken word fragment.
    """
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]

    for i in range(1, len(chunks)):
        tail_text = _word_safe_tail(chunks[i - 1], chunk_overlap)
        if tail_text:
            overlapped.append((tail_text + " " + chunks[i]).strip())
        else:
            overlapped.append(chunks[i])

    return overlapped


def _split_by_tokens(text: str, chunk_size: int) -> list[str]:
    """
    Last resort: split by raw token count, no overlap (overlap is
    applied once, later, by recursive_split). Used only when no
    separator divides a piece and it's still over chunk_size.
    """
    tokens = ENCODER.encode(text)

    pieces = []
    for start in range(0, len(tokens), chunk_size):
        pieces.append(ENCODER.decode(tokens[start : start + chunk_size]))

    return pieces


def chunk_text_from_page_segments(
    page_segments: list[dict],
    company: str,
    document: str,
    financial_year: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
) -> list[dict]:
    """
    Part A: turn page_segments (as loaded from page_segments.json)
    into text chunks.

    For each page: strip inline markdown table rows, split on
    headers (section=None for content preceding the first header on
    that page), then recursively split each section block to
    chunk_size/chunk_overlap tokens.

    chunk_id is deliberately NOT assigned here — that's Part C's
    job, once text and table chunks are combined and can share one
    global ID scheme.

    company/document/financial_year are passed in rather than read
    from document_metadata.json inside this function, so the core
    chunking logic stays testable without needing that file on
    disk — the __main__ block below shows how to load and pass them.

    Returns a list of dicts:
        {company, document, financial_year, page, section,
         chunk_type: "text", text, tokens, is_small}

    is_small=True flags a chunk under min_chunk_tokens (e.g. a bare
    leftover heading once its table rows are stripped) — flagged,
    not discarded: a financial document's short chunks can still be
    meaningful (a standalone figure, a one-line note), so silently
    dropping them risked losing real content on the strength of a
    length heuristic alone. Downstream retrieval/evaluation code can
    filter on is_small if it wants to exclude them, but the decision
    is no longer made unconditionally inside the chunker.
    """

    chunks: list[dict] = []
    flagged_small_count = 0

    for segment in page_segments:
        page = segment["page"]
        page_text = strip_markdown_tables(segment["text"])

        if not page_text:
            continue

        for block in split_on_headers(page_text):
            section = block["section"]
            content = block["content"]

            for piece in recursive_split(content, chunk_size, chunk_overlap):
                tokens = count_tokens(piece)
                is_small = tokens < min_chunk_tokens

                if is_small:
                    flagged_small_count += 1

                chunks.append(
                    {
                        "company": company,
                        "document": document,
                        "financial_year": financial_year,
                        "page": page,
                        "section": section,
                        "chunk_type": "text",
                        "text": piece,
                        "tokens": tokens,
                        "is_small": is_small,
                    }
                )

    if flagged_small_count:
        logger.info(
            f"Flagged {flagged_small_count} text chunk(s) under "
            f"{min_chunk_tokens} tokens as is_small=True (kept, not "
            f"discarded — often a leftover heading after "
            f"table-stripping, but not assumed meaningless)."
        )

    return chunks


def load_page_segments(processed_dir: str | Path) -> list[dict]:
    """Load page_segments.json from a processed document folder."""
    path = Path(processed_dir) / "page_segments.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_document_metadata(processed_dir: str | Path) -> dict:
    """Load document_metadata.json from a processed document folder."""
    path = Path(processed_dir) / "document_metadata.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    doc_metadata = load_document_metadata(processed_dir)
    page_segments = load_page_segments(processed_dir)

    text_chunks = chunk_text_from_page_segments(
        page_segments=page_segments,
        company=doc_metadata["company"],
        document=doc_metadata["document"],
        financial_year=doc_metadata["financial_year"],
    )

    logger.info(
        f"Generated {len(text_chunks)} text chunks from "
        f"{len(page_segments)} pages."
    )
