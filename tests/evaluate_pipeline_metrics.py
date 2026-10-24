"""Offline retrieval evaluation for fixed query/gold-chunk benchmark data.

Example benchmark shape::

    {
      "cases": [{
        "id": "infosys_revenue_fy2026",
        "question": "What was Infosys's consolidated revenue in FY2026?",
        "rewritten_query": "Infosys consolidated revenues in reported currency terms for fiscal 2026",
        "filters": {"company": "Infosys", "financial_year": "2025-2026"},
        "relevant_chunk_ids": ["annualreport_infosys_2026_txt_p077_0005"]
      }],
      "variants": [
        {"name": "original/table/legacy", "query_source": "question",
         "chunk_type": "table", "collection": "legacy_collection"},
        {"name": "rewritten/text/serialized", "query_source": "rewritten_query",
         "chunk_type": "text", "collection": "serialized_collection"},
        {"name": "original/unfiltered/table-boost", "query_source": "question",
         "chunk_type": "any", "table_score_boost": 0.05}
      ]
    }

Query rewrites are stored in the benchmark file; this script never calls Gemini.
Collections are opened read-only through ChromaRetriever and are never created
or modified. Metrics are macro averages across benchmark cases.
"""

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import ChromaRetriever, RetrievalProcessor
from src.hybrid_retrieval import (
    DEFAULT_PATH_CANDIDATE_LIMITS,
    HybridRetriever,
    KeywordLexicalRetriever,
    KeywordSemanticRetriever,
    STRATEGY_PATHS,
)
from src.keyword_generator import load_chunks_jsonl
from src.keyword_store import (
    DEFAULT_KEYWORD_CHROMA_PATH,
    DEFAULT_KEYWORD_COLLECTION,
    DEFAULT_LEXICAL_INDEX_PATH,
    KeywordLexicalIndex,
)

DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "benchmark" / "retrieval_validation.json"
DEFAULT_PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_COLLECTION = "financial_reports_filtered"
DEFAULT_CANDIDATE_K = 100
DEFAULT_GOLD_DATASET = PROJECT_ROOT / "data" / "benchmark" / "retrieval_gold_dataset.json"
QUERY_SOURCES = {"question", "rewritten_query"}
FILTER_FIELDS = {"company", "financial_year"}
FINANCIAL_YEAR_PATTERN = re.compile(r"^\d{4}-\d{4}$")
SUPPORTED_STRATEGIES = {"baseline", *STRATEGY_PATHS}

DEFAULT_GOLD_VARIANTS = [
    {"name": "baseline", "strategy": "baseline", "chunk_type": "any"},
    {"name": "chunk_semantic", "strategy": "chunk_semantic", "chunk_type": "any"},
    {"name": "keyword_semantic", "strategy": "keyword_semantic", "chunk_type": "any"},
    {"name": "keyword_lexical", "strategy": "keyword_lexical", "chunk_type": "any"},
    {"name": "hybrid_chunk_keyword", "strategy": "hybrid_chunk_keyword", "chunk_type": "any"},
    {"name": "hybrid_chunk_lexical", "strategy": "hybrid_chunk_lexical", "chunk_type": "any"},
    {"name": "hybrid_all", "strategy": "hybrid_all", "chunk_type": "any"},
]


class BenchmarkError(ValueError):
    """Raised when a benchmark file is missing required evaluation data."""


@dataclass(frozen=True)
class CaseDiagnostics:
    case_id: str
    relevant_chunk_count: int
    relevant_ids_retrieved: list[str]
    missing_relevant_ids: list[str]
    first_relevant_rank: int | None
    precision_at_5: float
    candidate_counts_by_path: dict[str, int]
    candidate_overlaps_by_path: dict[str, int]
    unique_candidate_count: int
    latency_by_path_seconds: dict[str, float]
    total_latency_seconds: float
    final_candidate_ids: list[str]


@dataclass(frozen=True)
class VariantMetrics:
    name: str
    case_count: int
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    mean_latency_seconds: float
    case_diagnostics: list[CaseDiagnostics]


def load_benchmark(path: str | Path) -> dict:
    path = Path(path)
    try:
        benchmark = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"Could not read benchmark JSON at {path}: {error}") from error
    if not isinstance(benchmark, dict):
        raise BenchmarkError("Benchmark root must be a JSON object.")

    return validate_benchmark(benchmark)


def validate_benchmark(benchmark: dict) -> dict:
    if not isinstance(benchmark, dict):
        raise BenchmarkError("Benchmark root must be a JSON object.")
    cases = benchmark.get("cases")
    variants = benchmark.get("variants")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("Benchmark must contain a non-empty 'cases' array.")
    if not isinstance(variants, list) or not variants:
        raise BenchmarkError("Benchmark must contain a non-empty 'variants' array.")

    case_ids = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BenchmarkError(f"Case at index {index} must be an object.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise BenchmarkError(f"Case at index {index} needs a unique non-empty 'id'.")
        case_ids.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise BenchmarkError(f"Case {case_id!r} needs a non-empty 'question'.")
        relevant_ids = case.get("relevant_chunk_ids")
        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in relevant_ids)
        ):
            raise BenchmarkError(
                f"Case {case_id!r} needs a non-empty 'relevant_chunk_ids' string array."
            )
        if len(set(relevant_ids)) != len(relevant_ids):
            raise BenchmarkError(f"Case {case_id!r} contains duplicate relevant chunk IDs.")
        filters = case.get("filters", {})
        if not isinstance(filters, dict) or set(filters) - FILTER_FIELDS:
            raise BenchmarkError(
                f"Case {case_id!r} filters may contain only company and financial_year."
            )
        for field, value in filters.items():
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise BenchmarkError(
                    f"Case {case_id!r} filter {field!r} must be an exact non-empty string value."
                )
            if field == "financial_year" and not FINANCIAL_YEAR_PATTERN.fullmatch(value):
                raise BenchmarkError(
                    f"Case {case_id!r} financial_year must use the stored YYYY-YYYY form."
                )

    variant_names = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise BenchmarkError(f"Variant at index {index} must be an object.")
        name = variant.get("name")
        if not isinstance(name, str) or not name.strip() or name in variant_names:
            raise BenchmarkError(f"Variant at index {index} needs a unique non-empty 'name'.")
        variant_names.add(name)
        strategy = variant.get("strategy", "baseline")
        if strategy not in SUPPORTED_STRATEGIES:
            raise BenchmarkError(f"Variant {name!r} has unsupported strategy {strategy!r}.")
        if variant.get("query_source", "question") not in QUERY_SOURCES:
            raise BenchmarkError(
                f"Variant {name!r} query_source must be 'question' or 'rewritten_query'."
            )
        if variant.get("chunk_type") not in (None, "any", "table", "text"):
            raise BenchmarkError(
                f"Variant {name!r} chunk_type must be 'table', 'text', 'any', or null."
            )
        collection = variant.get("collection")
        if collection is not None and (not isinstance(collection, str) or not collection.strip()):
            raise BenchmarkError(f"Variant {name!r} collection must be a non-empty string.")
        boost = variant.get("table_score_boost", 0.0)
        if isinstance(boost, bool) or not isinstance(boost, (int, float)):
            raise BenchmarkError(f"Variant {name!r} table_score_boost must be a finite number.")
        if not math.isfinite(boost) or boost < 0:
            raise BenchmarkError(f"Variant {name!r} table_score_boost must be finite and non-negative.")
        path_limits = variant.get("path_candidate_limits", {})
        if not isinstance(path_limits, dict) or set(path_limits) - set(DEFAULT_PATH_CANDIDATE_LIMITS):
            raise BenchmarkError(f"Variant {name!r} contains unsupported path candidate limits.")
        if any(isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 for limit in path_limits.values()):
            raise BenchmarkError(f"Variant {name!r} path candidate limits must be positive integers.")
        if variant.get("query_source", "question") == "rewritten_query":
            invalid = [
                case["id"]
                for case in cases
                if not isinstance(case.get("rewritten_query"), str)
                or not case["rewritten_query"].strip()
            ]
            if invalid:
                raise BenchmarkError(
                    f"Variant {name!r} needs a non-empty rewritten_query for every case; invalid in {invalid}."
                )

    return benchmark


def load_verified_gold_dataset(path: str | Path) -> tuple[dict, int]:
    """Convert only human-verified cases; candidate IDs are never treated as labels."""
    path = Path(path)
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"Could not read gold dataset at {path}: {error}") from error
    if not isinstance(source, dict) or not isinstance(source.get("cases"), list):
        raise BenchmarkError("Gold dataset must contain a cases array.")
    verified = [case for case in source["cases"] if case.get("verification_status") == "verified"]
    pending_count = sum(case.get("verification_status") == "pending" for case in source["cases"])
    if not verified:
        raise BenchmarkError(
            f"Gold dataset has no verified cases ({pending_count} pending); "
            "candidate_relevant_chunk_ids are not ground truth."
        )

    cases = []
    for case in verified:
        relevant_ids = case.get("relevant_chunk_ids")
        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise BenchmarkError(f"Verified case {case.get('id')!r} needs relevant_chunk_ids.")
        cases.append({
            "id": case["id"],
            "question": case["question"],
            "filters": case.get("filters", {}),
            "relevant_chunk_ids": relevant_ids,
        })
    benchmark = {"cases": cases, "variants": DEFAULT_GOLD_VARIANTS}
    return benchmark, pending_count


def _ranked_results(results, table_score_boost: float):
    if table_score_boost == 0:
        return list(results)
    ranked = sorted(
        results,
        key=lambda result: (
            -(
                result.score * (1.0 + table_score_boost)
                if result.chunk.get("chunk_type") == "table"
                else result.score
            ),
            result.rank,
            result.chunk_id,
        ),
    )
    return ranked


def score_ranked_ids(ranked_ids: list[str], relevant_ids: list[str]) -> tuple[float, float, float]:
    """Recall@k = relevant retrieved / all relevant IDs; MRR = reciprocal first-hit rank."""
    relevant = set(relevant_ids)
    if not relevant:
        raise BenchmarkError("Ground-truth relevant IDs cannot be empty.")
    recall_5 = len(set(ranked_ids[:5]) & relevant) / len(relevant)
    recall_10 = len(set(ranked_ids[:10]) & relevant) / len(relevant)
    reciprocal_rank = next(
        (1.0 / rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in relevant),
        0.0,
    )
    return recall_5, recall_10, reciprocal_rank


def evaluate_variant(
    cases: list[dict],
    variant: dict,
    retriever,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> VariantMetrics:
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 10:
        raise ValueError("candidate_k must be an integer of at least 10.")

    query_source = variant.get("query_source", "question")
    strategy = variant.get("strategy", "baseline")
    chunk_type = variant.get("chunk_type")
    if chunk_type == "any":
        chunk_type = None
    table_score_boost = float(variant.get("table_score_boost", 0.0))
    case_scores = []
    case_diagnostics = []
    case_latencies = []

    for case in cases:
        query = case.get(query_source)
        if not isinstance(query, str) or not query.strip():
            raise BenchmarkError(
                f"Case {case.get('id')!r} has no usable query for source {query_source!r}."
            )
        filters = case.get("filters", {})
        started_at = time.perf_counter()
        search_args = {
            "query": query,
            "top_k": candidate_k,
            "company": filters.get("company"),
            "financial_year": filters.get("financial_year"),
            "chunk_type": chunk_type,
        }
        if strategy in {"baseline", "chunk_semantic"}:
            results = retriever.search(**search_args)
        else:
            results = retriever.search(**search_args, strategy=strategy)
        measured_latency = time.perf_counter() - started_at
        ranked_results = _ranked_results(results, table_score_boost)
        ranked_ids = [result.chunk_id for result in ranked_results]
        case_scores.append(score_ranked_ids(ranked_ids, case["relevant_chunk_ids"]))
        relevant_ids = set(case["relevant_chunk_ids"])
        relevant_retrieved = [
            result.chunk_id for result in ranked_results if result.chunk_id in relevant_ids
        ]
        relevant_ranks = [
            rank
            for rank, result in enumerate(ranked_results, start=1)
            if result.chunk_id in relevant_ids
        ]
        case_diagnostics.append(CaseDiagnostics(
            case_id=case["id"],
            relevant_chunk_count=len(relevant_ids),
            relevant_ids_retrieved=relevant_retrieved,
            missing_relevant_ids=[chunk_id for chunk_id in case["relevant_chunk_ids"] if chunk_id not in relevant_ids.intersection(ranked_ids)],
            first_relevant_rank=min(relevant_ranks) if relevant_ranks else None,
            precision_at_5=len(set(ranked_ids[:5]) & relevant_ids) / 5,
            candidate_counts_by_path=(
                dict(retriever.last_search_diagnostics.get("path_candidate_counts", {}))
                if strategy not in {"baseline", "chunk_semantic"}
                else {"chunk_semantic": len(results)}
            ),
            candidate_overlaps_by_path=(
                dict(retriever.last_search_diagnostics.get("path_overlap_counts", {}))
                if strategy not in {"baseline", "chunk_semantic"}
                else {}
            ),
            unique_candidate_count=(
                retriever.last_search_diagnostics.get("unique_candidate_count", len(results))
                if strategy not in {"baseline", "chunk_semantic"}
                else len(results)
            ),
            latency_by_path_seconds=(
                dict(retriever.last_search_diagnostics.get("path_latencies_seconds", {}))
                if strategy not in {"baseline", "chunk_semantic"}
                else {"chunk_semantic": measured_latency}
            ),
            total_latency_seconds=(
                retriever.last_search_diagnostics.get("total_latency_seconds", measured_latency)
                if strategy not in {"baseline", "chunk_semantic"}
                else measured_latency
            ),
            final_candidate_ids=ranked_ids[:candidate_k],
        ))
        case_latencies.append(case_diagnostics[-1].total_latency_seconds)

    case_count = len(case_scores)
    return VariantMetrics(
        name=variant["name"],
        case_count=case_count,
        recall_at_5=sum(score[0] for score in case_scores) / case_count,
        recall_at_10=sum(score[1] for score in case_scores) / case_count,
        precision_at_5=sum(item.precision_at_5 for item in case_diagnostics) / case_count,
        mrr=sum(score[2] for score in case_scores) / case_count,
        mean_latency_seconds=sum(case_latencies) / case_count,
        case_diagnostics=case_diagnostics,
    )


def evaluate_benchmark(
    benchmark: dict,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    default_collection: str = DEFAULT_COLLECTION,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    retriever_factory=None,
    hybrid_retriever_factory=None,
    chunks_file: str | Path = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl",
    keyword_chroma_path: str | Path = DEFAULT_KEYWORD_CHROMA_PATH,
    keyword_collection: str = DEFAULT_KEYWORD_COLLECTION,
    lexical_index_path: str | Path = DEFAULT_LEXICAL_INDEX_PATH,
) -> list[VariantMetrics]:
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 10:
        raise ValueError("candidate_k must be an integer of at least 10.")

    if retriever_factory is None:
        processor = RetrievalProcessor()
        retrievers = {}

        def retriever_factory(collection_name):
            if collection_name not in retrievers:
                retrievers[collection_name] = ChromaRetriever(
                    processor=processor,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                )
            return retrievers[collection_name]

    if hybrid_retriever_factory is None:
        hybrid_retriever_factory = make_hybrid_retriever_factory(
            persist_directory=persist_directory,
            candidate_k=candidate_k,
            chunks_file=chunks_file,
            keyword_chroma_path=keyword_chroma_path,
            keyword_collection=keyword_collection,
            lexical_index_path=lexical_index_path,
        )

    if hybrid_retriever_factory is None:
        source_chunks = None
        hybrid_retrievers = {}

        def hybrid_retriever_factory(strategy, variant, collection_name):
            nonlocal source_chunks
            path_limits = variant.get("path_candidate_limits", {})
            selected_paths = STRATEGY_PATHS[strategy]
            effective_limits = {
                path_name: path_limits.get(path_name, candidate_k)
                for path_name in selected_paths
            }
            cache_key = (strategy, collection_name, tuple(sorted(effective_limits.items())))
            if cache_key not in hybrid_retrievers:
                if source_chunks is None:
                    source_chunks = load_chunks_jsonl(chunks_file)
                source_by_id = {chunk["chunk_id"]: chunk for chunk in source_chunks}
                processor = RetrievalProcessor()
                base = ChromaRetriever(
                    processor=processor,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                )
                needs_keyword_semantic = "keyword_semantic" in selected_paths
                needs_keyword_lexical = "keyword_lexical" in selected_paths
                keyword_semantic = None
                keyword_lexical = None
                if needs_keyword_semantic:
                    keyword_semantic = KeywordSemanticRetriever(
                        source_by_id,
                        processor=processor,
                        persist_directory=keyword_chroma_path,
                        collection_name=keyword_collection,
                    )
                if needs_keyword_lexical:
                    lexical_path = Path(lexical_index_path)
                    if not lexical_path.is_file():
                        raise BenchmarkError(
                            f"Lexical keyword index missing at {lexical_path}; build indexes before evaluating {strategy!r}."
                        )
                    keyword_lexical = KeywordLexicalRetriever(
                        KeywordLexicalIndex(lexical_path, source_by_id)
                    )
                hybrid_retrievers[cache_key] = HybridRetriever(
                    chunk_semantic=base,
                    keyword_semantic=keyword_semantic,
                    keyword_lexical=keyword_lexical,
                    path_candidate_limits=effective_limits,
                )
            return hybrid_retrievers[cache_key]

    metrics = []
    for variant in benchmark["variants"]:
        collection_name = variant.get("collection", default_collection)
        strategy = variant.get("strategy", "baseline")
        if strategy in {"baseline", "chunk_semantic"}:
            retriever = retriever_factory(collection_name)
        else:
            retriever = hybrid_retriever_factory(strategy, variant, collection_name)
        metrics.append(evaluate_variant(benchmark["cases"], variant, retriever, candidate_k))
    return metrics


def make_hybrid_retriever_factory(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    chunks_file: str | Path = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl",
    keyword_chroma_path: str | Path = DEFAULT_KEYWORD_CHROMA_PATH,
    keyword_collection: str = DEFAULT_KEYWORD_COLLECTION,
    lexical_index_path: str | Path = DEFAULT_LEXICAL_INDEX_PATH,
):
    """Create a cached factory; evaluation candidate budgets match per path."""
    source_chunks = None
    cache = {}

    def factory(strategy: str, variant: dict, collection_name: str):
        nonlocal source_chunks
        selected_paths = STRATEGY_PATHS[strategy]
        path_limits = variant.get("path_candidate_limits", {})
        limits = {name: path_limits.get(name, candidate_k) for name in selected_paths}
        key = (strategy, collection_name, tuple(sorted(limits.items())))
        if key in cache:
            return cache[key]

        if source_chunks is None and any(name != "chunk_semantic" for name in selected_paths):
            source_chunks = load_chunks_jsonl(chunks_file)
        source_by_id = (
            {chunk["chunk_id"]: chunk for chunk in source_chunks}
            if source_chunks is not None else None
        )
        processor = RetrievalProcessor()
        baseline = ChromaRetriever(
            processor=processor,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        keyword_semantic = None
        keyword_lexical = None
        if "keyword_semantic" in selected_paths:
            keyword_semantic = KeywordSemanticRetriever(
                source_by_id,
                processor=processor,
                persist_directory=keyword_chroma_path,
                collection_name=keyword_collection,
            )
        if "keyword_lexical" in selected_paths:
            if not Path(lexical_index_path).is_file():
                raise BenchmarkError(
                    f"Lexical index not found at {lexical_index_path}; build keyword indexes before evaluation."
                )
            keyword_lexical = KeywordLexicalRetriever(
                KeywordLexicalIndex(lexical_index_path, source_by_id)
            )
        cache[key] = HybridRetriever(
            chunk_semantic=baseline,
            keyword_semantic=keyword_semantic,
            keyword_lexical=keyword_lexical,
            path_candidate_limits=limits,
        )
        return cache[key]

    return factory


def load_verified_gold_dataset(path: str | Path) -> tuple[dict, int]:
    """Adapt only explicitly verified cases; pending candidate IDs are never labels."""
    path = Path(path)
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"Could not read gold dataset at {path}: {error}") from error
    if not isinstance(source, dict) or not isinstance(source.get("cases"), list):
        raise BenchmarkError("Gold dataset must contain a cases array.")

    verified = [case for case in source["cases"] if case.get("verification_status") == "verified"]
    pending = sum(case.get("verification_status") == "pending" for case in source["cases"])
    if not verified:
        raise BenchmarkError(
            f"Gold dataset contains no verified cases ({pending} pending); "
            "candidate_relevant_chunk_ids are never used as ground truth."
        )

    cases = []
    for case in verified:
        relevant_ids = case.get("relevant_chunk_ids")
        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise BenchmarkError(f"Verified case {case.get('id')!r} needs relevant_chunk_ids.")
        cases.append({
            "id": case.get("id"),
            "question": case.get("question"),
            "filters": case.get("filters", {}),
            "relevant_chunk_ids": relevant_ids,
        })
    benchmark = {"cases": cases, "variants": DEFAULT_GOLD_VARIANTS}
    return validate_benchmark(benchmark), pending


def format_matrix(metrics: list[VariantMetrics]) -> str:
    headers = ("Variant", "Cases", "Recall@5", "Recall@10", "Precision@5", "MRR", "Mean latency (s)")
    rows = [
        (metric.name, str(metric.case_count), f"{metric.recall_at_5:.4f}",
         f"{metric.recall_at_10:.4f}", f"{metric.precision_at_5:.4f}",
         f"{metric.mrr:.4f}", f"{metric.mean_latency_seconds:.4f}")
        for metric in metrics
    ]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    line = "-+-".join("-" * width for width in widths)
    render = lambda row: " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
    return "\n".join((render(headers), line, *(render(row) for row in rows)))


def format_case_diagnostics(metrics: list[VariantMetrics]) -> str:
    headers = (
        "Variant", "Case", "Relevant", "Retrieved relevant IDs", "Missing relevant IDs",
        "First relevant rank", "P@5", "Path counts", "Path overlap", "Unique candidates",
        "Path latency (s)", "Total latency (s)", "Final candidate IDs",
    )
    rows = []
    for metric in metrics:
        for case in metric.case_diagnostics:
            retrieved = ", ".join(case.relevant_ids_retrieved) or "-"
            missing = ", ".join(case.missing_relevant_ids) or "-"
            first_rank = str(case.first_relevant_rank) if case.first_relevant_rank is not None else "not retrieved"
            rows.append((
                metric.name,
                case.case_id,
                str(case.relevant_chunk_count),
                retrieved,
                missing,
                first_rank,
                f"{case.precision_at_5:.4f}",
                json.dumps(case.candidate_counts_by_path, sort_keys=True),
                json.dumps(case.candidate_overlaps_by_path, sort_keys=True),
                str(case.unique_candidate_count),
                json.dumps(case.latency_by_path_seconds, sort_keys=True),
                f"{case.total_latency_seconds:.4f}",
                ", ".join(case.final_candidate_ids) or "-",
            ))
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    line = "-+-".join("-" * width for width in widths)
    render = lambda row: " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
    return "\n".join((render(headers), line, *(render(row) for row in rows)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate offline dense retrieval variants.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--gold-dataset", type=Path, help="Use only manually verified cases from a gold dataset.")
    parser.add_argument("--persist-directory", type=Path, default=DEFAULT_PERSIST_DIRECTORY)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--keyword-chroma-path", type=Path, default=DEFAULT_KEYWORD_CHROMA_PATH)
    parser.add_argument("--keyword-collection", default=DEFAULT_KEYWORD_COLLECTION)
    parser.add_argument("--lexical-index", type=Path, default=DEFAULT_LEXICAL_INDEX_PATH)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    args = parser.parse_args(argv)

    try:
        if args.gold_dataset:
            benchmark, pending_cases = load_verified_gold_dataset(args.gold_dataset)
        else:
            benchmark = load_benchmark(args.benchmark)
            pending_cases = 0
        hybrid_factory = make_hybrid_retriever_factory(
            persist_directory=args.persist_directory,
            candidate_k=args.candidate_k,
            keyword_chroma_path=args.keyword_chroma_path,
            keyword_collection=args.keyword_collection,
            lexical_index_path=args.lexical_index,
        )
        metrics = evaluate_benchmark(
            benchmark,
            persist_directory=args.persist_directory,
            default_collection=args.collection,
            candidate_k=args.candidate_k,
            hybrid_retriever_factory=hybrid_factory,
        )
    except (BenchmarkError, ValueError, RuntimeError, OSError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(format_matrix(metrics))
    print("\nPer-case diagnostics:")
    print(format_case_diagnostics(metrics))
    print(f"\nCandidate pool per path: {args.candidate_k}")
    print(f"Gold cases pending and excluded: {pending_cases}")
    print(f"Benchmark: {args.gold_dataset or args.benchmark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())