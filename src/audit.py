"""
src/audit.py

Read-only dataset quality audit over the chunk JSONL corpus produced
by build.py. Never modifies chunks, never re-runs chunking, never
creates embeddings, and never imports chunker.py/table_chunker.py/
build_dataset.py/build.py -- it only reads the plain JSONL files
those modules already produced, using nothing beyond the standard
library.

    python -m src.audit
    python -m src.audit --chunks-root data/chunks --output-root data/audit

Excludes all_chunks.jsonl from the per-document audit, to avoid
double-counting every chunk (each chunk already appears once across
the per-document files). A separate cross-check against
all_chunks.jsonl can be layered on later without touching this
module's per-document logic.
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------
# Schema knowledge: read-only awareness of the EXISTING chunk
# schema (per build_dataset.py's/table_chunker.py's documented
# output fields), not a redefinition of it. Deliberately hardcoded
# here rather than imported, so this module has zero dependency on
# the chunking pipeline's code.
# --------------------------------------------------------------

COMMON_REQUIRED_FIELDS = [
    "chunk_id", "company", "document", "financial_year",
    "page", "chunk_type", "text", "tokens",
]
TABLE_REQUIRED_FIELDS = [
    "table_id", "table_heading", "split_strategy", "row_start", "row_end",
]
# Fields whose PRESENCE is still required, but whose value is
# allowed to legitimately be None by the existing chunking design.
NULLABLE_FIELDS = {"section", "context", "table_heading", "continuation_of"}

NEAR_EMPTY_TOKEN_THRESHOLD = 5
SMALL_TEXT_TOKEN_THRESHOLD = 10
LARGE_CHUNK_TOKEN_THRESHOLD = 400      # informational only
VERY_LARGE_CHUNK_TOKEN_THRESHOLD = 1000  # informational only

# Cap on example chunk_ids embedded per finding/category, so the
# report stays readable for a 5,000+ chunk corpus rather than
# listing every single ID.
SAMPLE_CAP = 50

_FINANCIAL_YEAR_PATTERN = re.compile(r"^\d{4}-\d{2}(\d{2})?$")


@dataclass
class Finding:
    severity: str  # "INFO" | "WARNING" | "ERROR"
    category: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class LoadedRecord:
    chunk: dict
    source_file: str
    line_no: int


def _sample(ids: list) -> dict:
    """Cap a list of IDs for embedding in the report, without
    silently hiding how many there really were."""
    ids = sorted(str(i) for i in ids)
    return {"count": len(ids), "sample": ids[:SAMPLE_CAP], "truncated": len(ids) > SAMPLE_CAP}


# ================================================================
# Loading
# ================================================================

def discover_document_jsonl_files(chunks_root: str | Path) -> list[Path]:
    """
    Every *.jsonl file directly under chunks_root, EXCLUDING
    all_chunks.jsonl -- sorted by filename for deterministic order.
    """
    chunks_root = Path(chunks_root)
    if not chunks_root.is_dir():
        return []
    return sorted(p for p in chunks_root.glob("*.jsonl") if p.name != "all_chunks.jsonl")


def load_jsonl(path: Path) -> tuple[list[LoadedRecord], list[Finding]]:
    """
    Parse one JSONL file. A malformed line is NEVER silently
    skipped: it's reported as an ERROR finding (naming the file and
    line number) and excluded from further analysis, while parsing
    continues for the remaining lines in the file.
    """
    records: list[LoadedRecord] = []
    findings: list[Finding] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                findings.append(Finding(
                    severity="ERROR", category="malformed_jsonl",
                    message=f"Invalid JSON on line {line_no} of {path.name}: {e}",
                    details={"file": path.name, "line_no": line_no},
                ))
                continue
            if not isinstance(obj, dict):
                findings.append(Finding(
                    severity="ERROR", category="malformed_jsonl",
                    message=f"Line {line_no} of {path.name} is not a JSON object.",
                    details={"file": path.name, "line_no": line_no},
                ))
                continue
            records.append(LoadedRecord(chunk=obj, source_file=path.name, line_no=line_no))

    return records, findings


# ================================================================
# 1. Inventory
# ================================================================

def compute_inventory(records: list[LoadedRecord]) -> dict:
    total = len(records)
    text_count = sum(1 for r in records if r.chunk.get("chunk_type") == "text")
    table_count = sum(1 for r in records if r.chunk.get("chunk_type") == "table")

    by_document = Counter(r.source_file for r in records)
    by_company = Counter(r.chunk.get("company", "<missing>") for r in records)

    return {
        "total_chunks": total,
        "text_chunks": text_count,
        "table_chunks": table_count,
        "other_chunk_type_count": total - text_count - table_count,
        "by_document": dict(sorted(by_document.items())),
        "by_company": dict(sorted(by_company.items())),
    }


# ================================================================
# 2. Token statistics
# ================================================================

def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (pct / 100) * (len(sorted_values) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def compute_token_statistics(records: list[LoadedRecord]) -> dict:
    """
    Uses the existing "tokens" field only -- never recomputes token
    counts, never re-tokenizes text. Records without a valid numeric
    "tokens" field are excluded here (already reported separately by
    validate_metadata) and counted as excluded_count.
    """
    values = []
    excluded = 0
    for r in records:
        tokens = r.chunk.get("tokens")
        if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
            values.append(float(tokens))
        else:
            excluded += 1

    if not values:
        return {
            "count": 0, "excluded_count": excluded,
            "min": None, "max": None, "mean": None, "median": None,
            "p25": None, "p75": None, "p90": None, "p95": None,
            "over_400_tokens": 0, "over_1000_tokens": 0,
        }

    values_sorted = sorted(values)

    return {
        "count": len(values),
        "excluded_count": excluded,
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(values_sorted),
        "p25": _percentile(values_sorted, 25),
        "p75": _percentile(values_sorted, 75),
        "p90": _percentile(values_sorted, 90),
        "p95": _percentile(values_sorted, 95),
        "over_400_tokens": sum(1 for v in values if v > LARGE_CHUNK_TOKEN_THRESHOLD),
        "over_1000_tokens": sum(1 for v in values if v > VERY_LARGE_CHUNK_TOKEN_THRESHOLD),
    }


# ================================================================
# 3. Metadata validation
# ================================================================

def validate_metadata(records: list[LoadedRecord]) -> list[Finding]:
    """
    Distinguishes an ABSENT field (key not in the chunk at all) from
    a field that is PRESENT but None. A None value is only accepted
    without complaint for fields in NULLABLE_FIELDS -- for every
    other required field, None is treated the same as missing.
    """
    findings: list[Finding] = []

    for r in records:
        chunk = r.chunk
        cid = chunk.get("chunk_id", f"<no id, {r.source_file}:{r.line_no}>")
        loc = {"chunk_id": cid, "source_file": r.source_file, "line_no": r.line_no}

        for required in COMMON_REQUIRED_FIELDS:
            if required not in chunk:
                findings.append(Finding("ERROR", "missing_field",
                    f"Missing required field '{required}'", dict(loc)))
            elif chunk[required] is None and required not in NULLABLE_FIELDS:
                findings.append(Finding("ERROR", "invalid_null_field",
                    f"Field '{required}' is None but must have a value", dict(loc)))

        chunk_type = chunk.get("chunk_type")
        if chunk_type not in ("text", "table"):
            findings.append(Finding("ERROR", "invalid_chunk_type",
                f"chunk_type is {chunk_type!r}, expected 'text' or 'table'", dict(loc)))

        if chunk_type == "table":
            for required in TABLE_REQUIRED_FIELDS:
                if required not in chunk:
                    findings.append(Finding("ERROR", "missing_table_field",
                        f"Table chunk missing required field '{required}'", dict(loc)))
                elif chunk[required] is None and required not in NULLABLE_FIELDS:
                    findings.append(Finding("ERROR", "invalid_null_table_field",
                        f"Table field '{required}' is None but must have a value", dict(loc)))

    return findings


# ================================================================
# 4. Duplicate detection
# ================================================================

def find_duplicate_chunk_ids(records: list[LoadedRecord]) -> list[Finding]:
    """Duplicate chunk_id -- ERROR. Global uniqueness across the
    whole corpus is a correctness guarantee of build_dataset.py's
    ID scheme; a duplicate here means that guarantee was violated
    somewhere upstream, not a cosmetic issue."""
    by_id: dict[str, list[LoadedRecord]] = {}
    for r in records:
        cid = r.chunk.get("chunk_id")
        if not isinstance(cid, str):
            continue  # non-string/missing chunk_id already reported elsewhere
        by_id.setdefault(cid, []).append(r)

    findings = []
    for cid, occurrences in sorted(by_id.items()):
        if len(occurrences) > 1:
            findings.append(Finding(
                "ERROR", "duplicate_chunk_id",
                f"chunk_id '{cid}' appears {len(occurrences)} times",
                {
                    "chunk_id": cid,
                    "occurrences": [
                        {"source_file": o.source_file, "line_no": o.line_no, "page": o.chunk.get("page")}
                        for o in occurrences
                    ],
                },
            ))
    return findings


def find_duplicate_text(records: list[LoadedRecord]) -> list[Finding]:
    """Duplicate exact TEXT content -- WARNING, not an error: could
    be a genuine repeated line item (e.g. a duplicated financial
    statement row, which table_chunker.py deliberately preserves
    rather than deduplicating) or a real extraction artifact --
    worth a human look, not automatically wrong."""
    by_text: dict[str, list[LoadedRecord]] = {}
    for r in records:
        text = r.chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            continue  # empty/invalid text is handled by find_small_and_empty_chunks
        by_text.setdefault(text, []).append(r)

    findings = []
    for occurrences in by_text.values():
        if len(occurrences) > 1:
            chunk_ids = sorted(str(o.chunk.get("chunk_id")) for o in occurrences)
            findings.append(Finding(
                "WARNING", "duplicate_text",
                f"Identical text content appears in {len(occurrences)} chunks",
                {
                    # Capped the same way as elsewhere in this module
                    # (see _sample) -- a corpus-wide duplicate (e.g.
                    # hundreds of tables sharing boilerplate text)
                    # would otherwise embed every single chunk_id in
                    # one finding, bloating the report for no benefit
                    # over knowing the true count plus a sample.
                    "chunk_ids": _sample(chunk_ids),
                    "documents": sorted({o.source_file for o in occurrences}),
                    "pages": _sample([o.chunk.get("page") for o in occurrences if isinstance(o.chunk.get("page"), int)]),
                    "first_chunk_id": chunk_ids[0],  # stable sort key, always present even when capped
                },
            ))

    findings.sort(key=lambda f: f.details["first_chunk_id"])
    return findings


# ================================================================
# 5. Small and empty chunks -- report only, never modify or drop
# ================================================================

def find_small_and_empty_chunks(records: list[LoadedRecord]) -> dict:
    empty_ids, near_empty_ids, small_text_ids, flagged_is_small_ids = [], [], [], []

    for r in records:
        chunk = r.chunk
        cid = chunk.get("chunk_id", f"<no id, {r.source_file}:{r.line_no}>")
        text = chunk.get("text")
        tokens = chunk.get("tokens")
        numeric_tokens = tokens if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) else None
        is_empty_text = not isinstance(text, str) or not text.strip()

        if is_empty_text or numeric_tokens == 0:
            empty_ids.append(cid)
        if numeric_tokens is not None and numeric_tokens <= NEAR_EMPTY_TOKEN_THRESHOLD:
            near_empty_ids.append(cid)
        if chunk.get("chunk_type") == "text" and numeric_tokens is not None and numeric_tokens < SMALL_TEXT_TOKEN_THRESHOLD:
            small_text_ids.append(cid)
        if chunk.get("is_small") is True:
            flagged_is_small_ids.append(cid)

    return {
        "empty_chunks": _sample(empty_ids),
        "near_empty_chunks_le_5_tokens": _sample(near_empty_ids),
        "text_chunks_under_10_tokens": _sample(small_text_ids),
        "flagged_is_small": _sample(flagged_is_small_ids),
    }


# ================================================================
# 6. Table inspection
# ================================================================

def inspect_tables(records: list[LoadedRecord]) -> tuple[dict, list[Finding]]:
    """
    Table-specific inspection. A large table chunk (>400 tokens) is
    deliberately NEVER flagged here as an error or warning -- it's
    an expected, correct outcome of the existing structure-aware
    table-splitting design (see table_chunker.py's split-threshold
    logic), not a defect this audit should report on.
    """
    table_records = [r for r in records if r.chunk.get("chunk_type") == "table"]

    split_strategy_counts = Counter(r.chunk.get("split_strategy", "<missing>") for r in table_records)
    low_confidence_count = sum(1 for r in table_records if r.chunk.get("is_low_confidence") is True)
    continuation_count = sum(1 for r in table_records if r.chunk.get("continuation_of") is not None)

    findings: list[Finding] = []
    for r in table_records:
        chunk = r.chunk
        cid = chunk.get("chunk_id", f"<no id, {r.source_file}:{r.line_no}>")
        loc = {"chunk_id": cid, "source_file": r.source_file, "line_no": r.line_no}

        table_id = chunk.get("table_id")
        if not isinstance(table_id, str) or not table_id.strip():
            findings.append(Finding("ERROR", "invalid_table_provenance",
                "table_id is missing, empty, or not a string", dict(loc)))

        row_start, row_end = chunk.get("row_start"), chunk.get("row_end")
        rows_are_ints = (
            isinstance(row_start, int) and not isinstance(row_start, bool)
            and isinstance(row_end, int) and not isinstance(row_end, bool)
        )
        if not rows_are_ints:
            findings.append(Finding("ERROR", "invalid_table_provenance",
                "row_start/row_end missing or not integers",
                {**loc, "row_start": row_start, "row_end": row_end}))
        elif row_start > row_end:
            findings.append(Finding("ERROR", "invalid_table_provenance",
                f"row_start ({row_start}) is greater than row_end ({row_end})", dict(loc)))

    info = {
        "table_chunk_count": len(table_records),
        "split_strategy_counts": dict(sorted(split_strategy_counts.items())),
        "low_confidence_count": low_confidence_count,
        "continuation_chunk_count": continuation_count,
    }
    return info, findings


# ================================================================
# 7. Provenance validation (identifying/traceability fields, all chunks)
# ================================================================

def validate_provenance(records: list[LoadedRecord]) -> list[Finding]:
    findings: list[Finding] = []

    for r in records:
        chunk = r.chunk
        cid = chunk.get("chunk_id", f"<no id, {r.source_file}:{r.line_no}>")
        loc = {"chunk_id": cid, "source_file": r.source_file, "line_no": r.line_no}

        for str_field in ("company", "document", "chunk_id"):
            value = chunk.get(str_field)
            if not isinstance(value, str) or not value.strip():
                findings.append(Finding("ERROR", "invalid_provenance",
                    f"'{str_field}' is missing, empty, or not a string", dict(loc)))

        fy = chunk.get("financial_year")
        if isinstance(fy, str) and not _FINANCIAL_YEAR_PATTERN.match(fy):
            findings.append(Finding("WARNING", "unusual_financial_year_format",
                f"financial_year {fy!r} does not match 'YYYY-YY' or 'YYYY-YYYY'", dict(loc)))

        page = chunk.get("page")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            findings.append(Finding("ERROR", "invalid_provenance",
                f"'page' is not a positive integer (got {page!r})", dict(loc)))

    return findings


# ================================================================
# Orchestration
# ================================================================

def run_audit(chunks_root: str | Path = "data/chunks", output_root: str | Path = "data/audit") -> dict:
    chunks_root = Path(chunks_root)
    output_root = Path(output_root)

    files = discover_document_jsonl_files(chunks_root)

    all_records: list[LoadedRecord] = []
    all_findings: list[Finding] = []

    for path in files:
        records, load_findings = load_jsonl(path)
        all_records.extend(records)
        all_findings.extend(load_findings)

    inventory = compute_inventory(all_records)
    token_stats = compute_token_statistics(all_records)
    small_empty = find_small_and_empty_chunks(all_records)
    table_info, table_findings = inspect_tables(all_records)

    all_findings.extend(validate_metadata(all_records))
    all_findings.extend(find_duplicate_chunk_ids(all_records))
    all_findings.extend(find_duplicate_text(all_records))
    all_findings.extend(table_findings)
    all_findings.extend(validate_provenance(all_records))

    severity_counts = Counter(f.severity for f in all_findings)
    if severity_counts.get("ERROR", 0) > 0:
        overall_status = "FAIL"
    elif severity_counts.get("WARNING", 0) > 0:
        overall_status = "PASS_WITH_WARNINGS"
    else:
        overall_status = "PASS"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunks_root": str(chunks_root),
        "documents_audited": [p.name for p in files],
        "inventory": inventory,
        "token_statistics": token_stats,
        "small_and_empty_chunks": small_empty,
        "table_inspection": table_info,
        "findings": [asdict(f) for f in all_findings],
        "summary": {
            "total_findings": len(all_findings),
            "errors": severity_counts.get("ERROR", 0),
            "warnings": severity_counts.get("WARNING", 0),
            "info": severity_counts.get("INFO", 0),
            "overall_status": overall_status,
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)

    _print_terminal_summary(report, report_path)

    return report


def _print_terminal_summary(report: dict, report_path: Path) -> None:
    inv = report["inventory"]
    stats = report["token_statistics"]
    summary = report["summary"]

    print()
    print("=" * 60)
    print("DATASET QUALITY AUDIT")
    print("=" * 60)
    print(f"Documents audited     : {len(report['documents_audited'])}")
    print(f"Total chunks          : {inv['total_chunks']}")
    print(f"Text chunks           : {inv['text_chunks']}")
    print(f"Table chunks          : {inv['table_chunks']}")
    print()
    if stats["count"]:
        print(f"Token mean / median   : {stats['mean']:.1f} / {stats['median']:.1f}")
        print(f"Token min / max       : {stats['min']:.0f} / {stats['max']:.0f}")
        print(f"Chunks > 400 tokens   : {stats['over_400_tokens']}")
        print(f"Chunks > 1000 tokens  : {stats['over_1000_tokens']}")
    print()
    print(f"Findings              : {summary['total_findings']} "
          f"({summary['errors']} errors, {summary['warnings']} warnings)")
    print(f"Report written to     : {report_path}")
    print()
    print(f"AUDIT STATUS: {summary['overall_status']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read-only dataset quality audit over data/chunks/*.jsonl."
    )
    parser.add_argument("--chunks-root", default="data/chunks", help="Default: data/chunks")
    parser.add_argument("--output-root", default="data/audit", help="Default: data/audit")
    args = parser.parse_args()

    report = run_audit(args.chunks_root, args.output_root)

    sys.exit(1 if report["summary"]["overall_status"] == "FAIL" else 0)