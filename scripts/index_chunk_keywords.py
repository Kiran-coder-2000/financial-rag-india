"""Build the separate keyword-vector and lexical indexes from annotations."""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.keyword_generator import load_chunks_jsonl
from src.keyword_store import (
    DEFAULT_KEYWORD_CHROMA_PATH,
    DEFAULT_KEYWORD_COLLECTION,
    DEFAULT_LEXICAL_INDEX_PATH,
    KeywordLexicalIndex,
    KeywordIndexError,
    index_keyword_vectors,
    load_successful_annotations,
)

DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "keyword_metadata" / "chunk_keywords.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the separate keyword vector and lexical indexes.")
    parser.add_argument("--chunks-file", type=Path, default=PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--keyword-chroma-path", type=Path, default=DEFAULT_KEYWORD_CHROMA_PATH)
    parser.add_argument("--keyword-collection", default=DEFAULT_KEYWORD_COLLECTION)
    parser.add_argument("--lexical-index", type=Path, default=DEFAULT_LEXICAL_INDEX_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-vectors", action="store_true")
    parser.add_argument("--skip-lexical", action="store_true")
    args = parser.parse_args(argv)

    if args.skip_vectors and args.skip_lexical:
        parser.error("At least one index must be selected.")

    try:
        chunks = load_chunks_jsonl(args.chunks_file)
        source_chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
        annotations, annotation_status = load_successful_annotations(args.annotations, source_chunks)
        summary = {
            "annotation_success_count": annotation_status["success"],
            "annotation_failure_count": annotation_status["failed"],
            "annotation_skipped_count": annotation_status["skipped"],
        }
        if not args.skip_lexical:
            lexical = KeywordLexicalIndex(args.lexical_index, source_chunks)
            summary["lexical"] = lexical.index_annotations(annotations)
        if not args.skip_vectors:
            summary["keyword_vectors"] = index_keyword_vectors(
                annotations,
                source_chunks,
                persist_directory=args.keyword_chroma_path,
                collection_name=args.keyword_collection,
                batch_size=args.batch_size,
            )
    except (OSError, ValueError, RuntimeError) as error:
        logger.exception("Keyword index build failed")
        print(f"Keyword index build failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 1 if annotation_status["failed"] else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())