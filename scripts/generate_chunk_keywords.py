"""Generate resumable LLM keyword metadata for existing report chunks.

Start with a deterministic stratified pilot, e.g. ``--limit 200 --dry-run``
then remove ``--dry-run`` after reviewing the selection. No input chunks or
embedding artifacts are changed by this script.
"""

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import query_planner
from src.keyword_generator import (
    GeminiKeywordProvider,
    KeywordGenerator,
    generate_keyword_annotations,
    load_chunks_jsonl,
)

DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "keyword_metadata" / "chunk_keywords.jsonl"
logger = logging.getLogger(__name__)


def select_pilot_chunks(chunks: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return list(chunks)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer when provided.")
    if limit >= len(chunks):
        return list(chunks)

    groups_by_company = defaultdict(lambda: defaultdict(list))
    for chunk in chunks:
        company = str(chunk.get("company") or "")
        stratum = (
            str(chunk.get("financial_year") or ""),
            str(chunk.get("chunk_type") or ""),
        )
        groups_by_company[company][stratum].append(chunk)
    for company_groups in groups_by_company.values():
        for group in company_groups.values():
            group.sort(key=lambda chunk: chunk["chunk_id"])

    company_keys = sorted(groups_by_company)
    stratum_keys = {
        company: sorted(groups_by_company[company])
        for company in company_keys
    }
    offsets = {
        (company, stratum): 0
        for company in company_keys
        for stratum in stratum_keys[company]
    }
    cursors = {company: 0 for company in company_keys}

    selected = []
    while len(selected) < limit:
        added_this_round = False
        for company in company_keys:
            strata = stratum_keys[company]
            if not strata:
                continue
            for step in range(len(strata)):
                stratum_index = (cursors[company] + step) % len(strata)
                stratum = strata[stratum_index]
                group = groups_by_company[company][stratum]
                offset_key = (company, stratum)
                offset = offsets[offset_key]
                if offset < len(group):
                    selected.append(group[offset])
                    offsets[offset_key] += 1
                    cursors[company] = (stratum_index + 1) % len(strata)
                    added_this_round = True
                    break
            if len(selected) == limit:
                break
        if not added_this_round:
            break
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate keyword annotations for existing report chunks.")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", choices=("gemini",), default="gemini")
    parser.add_argument("--model", help="Override GEMINI_MODEL for annotations only.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="Deterministically sample this many chunks across metadata strata.")
    parser.add_argument("--dry-run", action="store_true", help="Show the selected pilot without calling Gemini or writing output.")
    parser.add_argument("--force", action="store_true", help="Regenerate even annotations already marked successful.")
    args = parser.parse_args(argv)

    try:
        chunks = load_chunks_jsonl(args.chunks_file)
        selected = select_pilot_chunks(chunks, args.limit)
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive.")
        if args.retries < 0:
            raise ValueError("retries must be non-negative.")
        if args.dry_run:
            strata = defaultdict(int)
            for chunk in selected:
                strata[(chunk.get("company"), chunk.get("financial_year"), chunk.get("chunk_type"))] += 1
            print(f"Dry run: {len(selected)} of {len(chunks)} chunks selected; no API calls or files written.")
            for key, count in sorted(strata.items(), key=lambda item: tuple(str(value) for value in item[0])):
                print(f"{key}: {count}")
            return 0

        settings = query_planner.load_settings()
        if args.model:
            settings = replace(settings, model=args.model)
        provider = GeminiKeywordProvider(
            settings=settings,
            max_retries=args.retries,
            backoff_s=args.backoff_seconds,
        )
        generator = KeywordGenerator(provider)
        summary = generate_keyword_annotations(
            selected,
            args.output,
            generator,
            batch_size=args.batch_size,
            force=args.force,
            source_chunk_ids=[chunk["chunk_id"] for chunk in chunks],
        )
    except (OSError, ValueError, query_planner.QueryPlannerError) as error:
        print(f"Keyword generation failed: {error}", file=sys.stderr)
        return 1

    logger.info("Keyword annotation provider=%s model=%s", provider.provider_name, provider.model)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"output: {args.output}")
    if summary["failed_count"]:
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())