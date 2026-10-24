"""
test_retrieval_manual.py

Manual, opt-in inspection tool for the persistent Chroma retriever.
This file is not part of the automated test suite and never mutates
the collection.

Run from the project root:

    .venv/Scripts/python.exe tests/test_retrieval_manual.py

When --query is omitted, the script prompts for a question. Use
--query for non-interactive runs and --company,
--financial-year, or --chunk-type for exact stored metadata filters.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import query_planner
from src.embeddings import load_chunks
from src.hybrid_retrieval import (
    STRATEGY_PATHS,
    HybridRetriever,
    KeywordLexicalRetriever,
    KeywordSemanticRetriever,
)
from src.keyword_store import DEFAULT_KEYWORD_CHROMA_PATH, DEFAULT_LEXICAL_INDEX_PATH, KeywordLexicalIndex
from src.retrieval import ChromaRetriever, RetrievalProcessor
from src.retrieval_pipeline import FinancialRetrievalPipeline

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks" / "all_chunks.jsonl"


def _read_query(query: str | None) -> str:
    if query is not None:
        return query
    return input("Question: ").strip()


def _print_result(result) -> None:
    chunk = result.chunk
    print(f"Rank {result.rank}")
    print(f"Score: {result.score:.4f}")
    print(f"Chunk ID: {result.chunk_id}")
    print(f"Company: {chunk.get('company')}")
    print(f"Document: {chunk.get('document')}")
    print(f"Financial Year: {chunk.get('financial_year')}")
    print(f"Page: {chunk.get('page')}")
    print(f"Section: {chunk.get('section')}")
    print(f"Chunk Type: {chunk.get('chunk_type')}")
    table_metadata = {
        key: chunk[key]
        for key in ("table_id", "table_heading", "row_start", "row_end")
        if key in chunk
    }
    if table_metadata:
        print(f"Table Metadata: {json.dumps(table_metadata, ensure_ascii=False)}")
    path_evidence = chunk.get("retrieval_path_evidence")
    if path_evidence:
        print(f"Retrieval Paths: {json.dumps(path_evidence, ensure_ascii=False)}")
    print()
    print("Text:")
    print(chunk.get("text", ""))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the existing persistent Chroma collection.")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--company")
    parser.add_argument("--financial-year")
    parser.add_argument("--chunk-type")
    parser.add_argument(
        "--strategy",
        choices=("baseline", *sorted(STRATEGY_PATHS)),
        default="baseline",
        help="baseline is the default; experimental strategies require their separate indexes.",
    )
    args = parser.parse_args()
    user_query = _read_query(args.query)

    try:
        registry = query_planner.registry_from_chunks_file(CHUNKS_FILE)
        planner = query_planner.QueryPlanner(registry=registry)
        processor = RetrievalProcessor()
        retriever = ChromaRetriever(processor=processor)
        experimental_retriever = None
        if args.strategy != "baseline":
            chunks = load_chunks(CHUNKS_FILE)
            source_chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
            needs_keyword_semantic = "keyword_semantic" in STRATEGY_PATHS[args.strategy]
            needs_keyword_lexical = "keyword_lexical" in STRATEGY_PATHS[args.strategy]
            keyword_semantic = None
            keyword_lexical = None
            if needs_keyword_semantic:
                keyword_semantic = KeywordSemanticRetriever(
                    source_chunks,
                    processor=processor,
                    persist_directory=DEFAULT_KEYWORD_CHROMA_PATH,
                )
            if needs_keyword_lexical:
                if not DEFAULT_LEXICAL_INDEX_PATH.is_file():
                    raise FileNotFoundError(
                        f"Lexical keyword index is missing at {DEFAULT_LEXICAL_INDEX_PATH}. "
                        "Run scripts/index_chunk_keywords.py first."
                    )
                keyword_lexical = KeywordLexicalRetriever(
                    KeywordLexicalIndex(DEFAULT_LEXICAL_INDEX_PATH, source_chunks)
                )
            experimental_retriever = HybridRetriever(
                chunk_semantic=retriever,
                keyword_semantic=keyword_semantic,
                keyword_lexical=keyword_lexical,
            )
        pipeline = FinancialRetrievalPipeline(
            planner=planner,
            retriever=retriever,
            retrieval_strategy=args.strategy,
            experimental_retriever=experimental_retriever,
        )
        result = pipeline.run(
            user_query,
            top_k=args.top_k,
            company=args.company,
            financial_year=args.financial_year,
            chunk_type=args.chunk_type,
        )
    except (query_planner.QueryPlannerError, ValueError, RuntimeError) as error:
        print(f"Retrieval pipeline failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"Original question: {result.original_question}")
    print("Validated query plan:")
    print(json.dumps(result.plan.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(f"\nPipeline status: {result.status}; strategy: {result.retrieval_strategy}")
    for warning in result.warnings:
        print(f"Warning: {warning}")

    for task_result in result.task_results:
        print("\n" + "=" * 72)
        print(f"Task ID: {task_result.task_id}")
        print(f"Query: {task_result.query}")
        print(f"Effective metadata filters: {json.dumps(task_result.filters, ensure_ascii=False)}")
        if task_result.skipped_reason:
            print(f"Skipped: {task_result.skipped_reason}")
            continue
        if task_result.error:
            print(f"Retrieval error: {task_result.error}")
            continue
        if not task_result.results:
            print("No results for this task.")
            continue
        for retrieved in task_result.results:
            print("-" * 72)
            print(f"Task ID: {task_result.task_id}")
            _print_result(retrieved)
    if result.task_results:
        print("=" * 72)

    if result.status == "partial":
        raise SystemExit(1)


if __name__ == "__main__":
    main()