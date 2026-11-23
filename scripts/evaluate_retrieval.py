"""Run the retrieval-only evaluation against the existing benchmark and indexes.

Run from the repository root:
    .venv\\Scripts\\python.exe scripts\\evaluate_retrieval.py
"""

import argparse
import csv
import hashlib
import html
import json
import math
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_config import (  # noqa: E402
    ANNOTATION_PATHS,
    BENCHMARK_PATH,
    CHROMA_COLLECTION,
    CHROMA_PATH,
    CHUNKS_PATH,
    KEYWORD_CHROMA_PILOT_COLLECTION,
    KEYWORD_CHROMA_PILOT_PATH,
    KEYWORD_LEXICAL_PILOT_PATH,
    K_VALUES,
    RESULTS_PATH,
    STRATEGIES,
    STRATEGY_DESCRIPTIONS,
    TOP_K,
)
from src.evaluation_metrics import ranking_metrics  # noqa: E402
from src.embeddings import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    load_chunks,
)
from src.hybrid_retrieval import (  # noqa: E402
    DEFAULT_PATH_CANDIDATE_LIMITS,
    HybridRetriever,
    KeywordLexicalRetriever,
    KeywordSemanticRetriever,
)
from src.keyword_store import (  # noqa: E402
    DEFAULT_KEYWORD_CHROMA_PATH,
    DEFAULT_KEYWORD_COLLECTION,
    DEFAULT_LEXICAL_INDEX_PATH,
    KeywordLexicalIndex,
)
from src.retrieval import ChromaRetriever, RetrievalProcessor  # noqa: E402


class ReadOnlyKeywordLexicalIndex(KeywordLexicalIndex):
    """Use the existing FTS search implementation without initializing or writing its DB."""

    def __init__(self, path, source_chunks_by_id):
        self.path = Path(path).resolve()
        self.source_chunks_by_id = source_chunks_by_id

    def _connect(self):
        uri = f"{self.path.as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_inputs():
    benchmark = load_json(BENCHMARK_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    source_chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(source_chunks) != len(chunks):
        raise ValueError("The source corpus contains duplicate chunk IDs.")

    benchmark_cases = benchmark.get("cases")
    if not isinstance(benchmark_cases, list) or len(benchmark_cases) != 60:
        raise ValueError("The refined benchmark must contain exactly 60 cases.")
    benchmark_by_id = {case["id"]: case for case in benchmark_cases}
    if len(benchmark_by_id) != len(benchmark_cases):
        raise ValueError("The benchmark contains duplicate question IDs.")

    drafts_by_id = {}
    draft_hashes = {}
    for company, path in ANNOTATION_PATHS.items():
        draft = load_json(path)
        if (
            draft.get("status") != "proposed_pending_human_review"
            or draft.get("company") != company
            or draft.get("case_count") != 20
            or len(draft.get("cases", [])) != 20
        ):
            raise ValueError(f"Annotation draft {path.name} has an unexpected status or shape.")
        if draft.get("source_benchmark_sha256") != sha256_file(BENCHMARK_PATH):
            raise ValueError(f"Annotation draft {path.name} references a different benchmark hash.")
        draft_hashes[path.name] = sha256_file(path)
        for draft_case in draft["cases"]:
            case_id = draft_case["id"]
            if case_id in drafts_by_id:
                raise ValueError(f"Duplicate draft case ID: {case_id}")
            benchmark_case = benchmark_by_id.get(case_id)
            if benchmark_case is None:
                raise ValueError(f"Draft case {case_id} is absent from the benchmark.")
            if draft_case.get("question") != benchmark_case.get("question"):
                raise ValueError(f"Draft wording differs from benchmark question {case_id}.")
            if draft_case.get("candidate_expected_action_status") != "unverified":
                raise ValueError(f"Candidate action for {case_id} is not marked unverified.")
            drafts_by_id[case_id] = draft_case

    if set(drafts_by_id) != set(benchmark_by_id):
        raise ValueError("The three annotation drafts do not cover the benchmark exactly once.")

    for case_id, draft_case in drafts_by_id.items():
        case = benchmark_by_id[case_id]
        allowed_companies = set(case.get("filters", {}).get("companies") or [case["company"]])
        for assertion in draft_case["proposed_gold"].get("expected_assertions", []):
            for evidence in assertion.get("evidence", []):
                chunk_id = evidence.get("chunk_id")
                chunk = source_chunks.get(chunk_id)
                if chunk is None:
                    raise ValueError(f"{case_id} cites missing corpus chunk {chunk_id!r}.")
                if chunk.get("company") not in allowed_companies:
                    raise ValueError(
                        f"{case_id} cites {chunk_id!r} from unexpected company "
                        f"{chunk.get('company')!r}."
                    )
                if evidence.get("verification_status") != "proposed_unverified":
                    raise ValueError(
                        f"{case_id} evidence {chunk_id!r} is not marked proposed_unverified."
                    )
                if evidence.get("verified_by") is not None:
                    raise ValueError(f"{case_id} assigns an evidence reviewer.")

    corpus_hash = sha256_file(CHUNKS_PATH)
    expected_corpus_hash = benchmark.get("source", {}).get("source_chunk_corpus_sha256")
    if corpus_hash != expected_corpus_hash:
        raise ValueError("The chunk corpus hash does not match the benchmark's recorded source hash.")

    input_hashes = {
        BENCHMARK_PATH.name: sha256_file(BENCHMARK_PATH),
        CHUNKS_PATH.name: corpus_hash,
        **draft_hashes,
    }
    return benchmark, benchmark_cases, drafts_by_id, chunks, source_chunks, input_hashes


def validate_lexical_pilot(index: ReadOnlyKeywordLexicalIndex, source_chunks: dict) -> int:
    if not index.path.is_file():
        raise FileNotFoundError(f"Keyword lexical pilot index is missing: {index.path}")
    with index._connect() as connection:
        record_count = connection.execute("SELECT COUNT(*) FROM keyword_records").fetchone()[0]
        search_count = connection.execute(
            "SELECT COUNT(DISTINCT chunk_id) FROM keyword_search"
        ).fetchone()[0]
        records = connection.execute(
            "SELECT chunk_id, company, financial_year, chunk_type FROM keyword_records"
        ).fetchall()
    if record_count != search_count:
        raise ValueError(
            f"Keyword lexical pilot has {record_count} records but {search_count} FTS entries."
        )
    for record in records:
        chunk = source_chunks.get(record["chunk_id"])
        if chunk is None:
            raise ValueError(
                f"Keyword lexical pilot references unknown corpus chunk {record['chunk_id']!r}."
            )
        for field in ("company", "financial_year", "chunk_type"):
            if (record[field] or None) != (chunk.get(field) or None):
                raise ValueError(
                    f"Keyword lexical pilot metadata differs from corpus for "
                    f"{record['chunk_id']!r} ({field})."
                )
    return record_count


def chroma_collection_exists(persist_directory: Path, collection_name: str) -> bool:
    database_path = persist_directory / "chroma.sqlite3"
    if not database_path.is_file():
        return False
    connection = sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        row = connection.execute(
            "SELECT 1 FROM collections WHERE name = ? LIMIT 1",
            (collection_name,),
        ).fetchone()
        return row is not None
    except sqlite3.Error as error:
        raise RuntimeError(
            f"Could not inspect Chroma collection {collection_name!r} read-only: {error}"
        ) from error
    finally:
        connection.close()


def _retrieval_hit(hit) -> dict:
    chunk = hit.chunk
    fields = (
        "company",
        "document",
        "financial_year",
        "page",
        "section",
        "chunk_type",
        "table_id",
        "table_heading",
        "row_start",
        "row_end",
        "retrieval_score_type",
        "retrieval_path_evidence",
    )
    return {
        "chunk_id": hit.chunk_id,
        "rank": hit.rank,
        "score": float(hit.score),
        "text": chunk.get("text"),
        "provenance": {key: chunk[key] for key in fields if key in chunk},
    }


def _relevant_chunk_ids(draft_case: dict) -> list[str]:
    relevant = []
    for assertion in draft_case["proposed_gold"].get("expected_assertions", []):
        for evidence in assertion.get("evidence", []):
            if evidence.get("relation") == "supports":
                relevant.append(evidence["chunk_id"])
    return list(dict.fromkeys(relevant))


def _metadata_filters(case: dict) -> dict:
    filters = case.get("filters", {})
    return {
        "company": filters.get("company"),
        "financial_year": filters.get("financial_year"),
        "chunk_type": filters.get("chunk_type"),
    }


def _build_retrievers(
    source_chunks: dict,
    warmup_question: str,
    warmup_question_id: str,
) -> tuple[dict, dict, dict]:
    processors = RetrievalProcessor()
    warmup_started = time.perf_counter()
    processors.embed_query(warmup_question)
    embedding_warmup_seconds = time.perf_counter() - warmup_started
    retrievers = {}
    availability = {}
    index_details = {
        "embedding_warmup_seconds": embedding_warmup_seconds,
        "warmup_question_id": warmup_question_id,
    }

    try:
        chunk_retriever = ChromaRetriever(
            processor=processors,
            persist_directory=CHROMA_PATH,
            collection_name=CHROMA_COLLECTION,
        )
        retrievers["chunk_semantic"] = chunk_retriever
        availability["chunk_semantic"] = {
            "status": "available",
            "index_path": str(CHROMA_PATH),
            "collection": CHROMA_COLLECTION,
            "indexed_chunks": chunk_retriever.count,
        }
    except Exception as error:
        chunk_retriever = None
        availability["chunk_semantic"] = {
            "status": "unavailable",
            "error": f"{type(error).__name__}: {error}",
        }

    try:
        keyword_semantic = KeywordSemanticRetriever(
            source_chunks,
            processor=processors,
            persist_directory=KEYWORD_CHROMA_PILOT_PATH,
            collection_name=KEYWORD_CHROMA_PILOT_COLLECTION,
        )
        retrievers["keyword_semantic"] = keyword_semantic
        availability["keyword_semantic"] = {
            "status": "available",
            "index_path": str(KEYWORD_CHROMA_PILOT_PATH),
            "collection": KEYWORD_CHROMA_PILOT_COLLECTION,
            "indexed_chunks": keyword_semantic._retriever.count,
            "coverage_note": "Pilot index; not the configured production keyword-vector path.",
        }
    except Exception as error:
        keyword_semantic = None
        availability["keyword_semantic"] = {
            "status": "unavailable",
            "error": f"{type(error).__name__}: {error}",
        }

    try:
        lexical_index = ReadOnlyKeywordLexicalIndex(
            KEYWORD_LEXICAL_PILOT_PATH,
            source_chunks,
        )
        lexical_count = validate_lexical_pilot(lexical_index, source_chunks)
        lexical_retriever = KeywordLexicalRetriever(lexical_index)
        retrievers["keyword_lexical"] = lexical_retriever
        availability["keyword_lexical"] = {
            "status": "available",
            "index_path": str(KEYWORD_LEXICAL_PILOT_PATH),
            "indexed_chunks": lexical_count,
            "coverage_note": "Pilot index; not the configured production lexical-index path.",
        }
    except Exception as error:
        lexical_retriever = None
        availability["keyword_lexical"] = {
            "status": "unavailable",
            "error": f"{type(error).__name__}: {error}",
        }

    for strategy in ("baseline", "chunk_semantic"):
        if chunk_retriever is not None:
            availability[strategy] = {
                **availability["chunk_semantic"],
                "alias_of": "chunk_semantic",
            }
        else:
            availability[strategy] = availability["chunk_semantic"]

    hybrid = None
    if chunk_retriever is not None:
        hybrid = HybridRetriever(
            chunk_semantic=chunk_retriever,
            keyword_semantic=keyword_semantic,
            keyword_lexical=lexical_retriever,
        )
    for strategy, required_paths in {
        "hybrid_chunk_keyword": ("chunk_semantic", "keyword_semantic"),
        "hybrid_chunk_lexical": ("chunk_semantic", "keyword_lexical"),
        "hybrid_all": ("chunk_semantic", "keyword_semantic", "keyword_lexical"),
    }.items():
        missing = [path for path in required_paths if path not in retrievers]
        availability[strategy] = {
            "status": "available" if not missing else "unavailable",
            "required_paths": list(required_paths),
            **({"missing_paths": missing} if missing else {}),
        }

    index_details["default_keyword_semantic_path"] = {
        "status": (
            "available"
            if chroma_collection_exists(
                DEFAULT_KEYWORD_CHROMA_PATH,
                DEFAULT_KEYWORD_COLLECTION,
            )
            else "missing"
        ),
        "expected_path": str(DEFAULT_KEYWORD_CHROMA_PATH),
        "collection": DEFAULT_KEYWORD_COLLECTION,
        "available_pilot_path": str(KEYWORD_CHROMA_PILOT_PATH),
    }
    index_details["default_keyword_lexical_path"] = {
        "status": "present" if DEFAULT_LEXICAL_INDEX_PATH.is_file() else "missing",
        "expected_path": str(DEFAULT_LEXICAL_INDEX_PATH),
        "available_pilot_path": str(KEYWORD_LEXICAL_PILOT_PATH),
    }
    return retrievers, availability, {"hybrid": hybrid, **index_details}


def _run_strategy(
    strategy: str,
    query: str,
    filters: dict,
    retrievers: dict,
    availability: dict,
    details: dict,
) -> tuple[list, dict | None]:
    if availability[strategy]["status"] != "available":
        return [], None
    if strategy in {"baseline", "chunk_semantic"}:
        return (
            retrievers["chunk_semantic"].search(
                query=query,
                top_k=TOP_K,
                **filters,
            ),
            None,
        )
    hybrid = details["hybrid"]
    results = hybrid.search(
        query=query,
        top_k=TOP_K,
        strategy=strategy,
        **filters,
    )
    diagnostics = dict(hybrid.last_search_diagnostics)
    return results, diagnostics


def evaluate_cases(benchmark_cases, drafts_by_id, retrievers, availability, details, limit=None):
    cases = benchmark_cases if limit is None else benchmark_cases[:limit]
    raw_cases = []
    metric_rows = []
    for case in cases:
        draft = drafts_by_id[case["id"]]
        proposed_gold = draft["proposed_gold"]
        relevant_ids = _relevant_chunk_ids(draft)
        filters = _metadata_filters(case)
        case_results = {
            "question_id": case["id"],
            "question": case["question"],
            "company": case["company"],
            "category": case["category"],
            "financial_periods": case["financial_periods"],
            "benchmark_filters": case.get("filters", {}),
            "metadata_filters": filters,
            "expected_action": proposed_gold["expected_action"],
            "annotation_status": "proposed_unverified",
            "relevant_chunk_ids": relevant_ids,
            "relevance_label_status": (
                "proposed_unverified" if relevant_ids else "no_supporting_chunk_labels"
            ),
            "strategies": {},
        }
        for strategy in STRATEGIES:
            started = time.perf_counter()
            status = "completed"
            error = None
            hits = []
            diagnostics = None
            if availability[strategy]["status"] != "available":
                status = "unavailable"
                error = availability[strategy].get("error") or (
                    "Required index path(s) are unavailable: "
                    + ", ".join(availability[strategy].get("missing_paths", []))
                )
            else:
                try:
                    retrieved, diagnostics = _run_strategy(
                        strategy,
                        case["question"],
                        filters,
                        retrievers,
                        availability,
                        details,
                    )
                    hits = [_retrieval_hit(hit) for hit in retrieved]
                except Exception as retrieval_error:
                    status = "failed"
                    error = f"{type(retrieval_error).__name__}: {retrieval_error}"
            latency = time.perf_counter() - started
            ids = [hit["chunk_id"] for hit in hits]
            metrics = ranking_metrics(ids, relevant_ids, K_VALUES) if status == "completed" else None
            result = {
                "status": status,
                "latency_seconds": latency,
                "retrieved_count": len(hits),
                "retrieved": hits,
                "metrics": metrics,
                "diagnostics": diagnostics,
            }
            if error is not None:
                result["error"] = error
            case_results["strategies"][strategy] = result
            metric_row = {
                "question_id": case["id"],
                "company": case["company"],
                "category": case["category"],
                "strategy": strategy,
                "retrieval_status": status,
                "metric_status": (
                    "scored"
                    if status == "completed" and relevant_ids
                    else "not_evaluable"
                    if status == "completed"
                    else status
                ),
                "relevant_count": len(relevant_ids),
                "retrieved_count": len(hits),
                "empty_retrieval": status == "completed" and not hits,
                "latency_seconds": latency,
            }
            if metrics:
                metric_row.update(metrics)
            else:
                metric_row["mrr"] = None
                for k in K_VALUES:
                    metric_row[f"hits@{k}"] = 0
                    for name in ("precision", "recall", "f1", "ndcg"):
                        metric_row[f"{name}@{k}"] = None
            metric_rows.append(metric_row)
        raw_cases.append(case_results)

    common_case_ids = {
        case["question_id"]
        for case in raw_cases
        if all(
            result["status"] == "completed"
            for result in case["strategies"].values()
        )
    }
    for case in raw_cases:
        case["common_subset"] = case["question_id"] in common_case_ids
    for row in metric_rows:
        row["common_subset"] = row["question_id"] in common_case_ids
    return raw_cases, metric_rows


def _mean(values):
    return statistics.fmean(values) if values else None


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def aggregate_rows(rows, dimensions):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[dimension] for dimension in dimensions)
        grouped[key].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        successful = [row for row in group if row["retrieval_status"] == "completed"]
        scored = [row for row in successful if row["metric_status"] == "scored"]
        latencies = [row["latency_seconds"] for row in group]
        aggregate = {dimension: value for dimension, value in zip(dimensions, key)}
        query_count = len(group)
        empty_count = sum(row["empty_retrieval"] for row in successful)
        aggregate.update({
            "query_count": query_count,
            "successful_queries": len(successful),
            "failed_queries": sum(row["retrieval_status"] == "failed" for row in group),
            "unavailable_queries": sum(row["retrieval_status"] == "unavailable" for row in group),
            "scored_queries": len(scored),
            "successful_retrieval_rate": len(successful) / query_count if query_count else None,
            "failed_retrieval_rate": (
                sum(row["retrieval_status"] == "failed" for row in group) / query_count
                if query_count else None
            ),
            "unavailable_retrieval_rate": (
                sum(row["retrieval_status"] == "unavailable" for row in group) / query_count
                if query_count else None
            ),
            "empty_retrieval_count": empty_count,
            "empty_retrieval_rate": empty_count / len(successful) if successful else None,
            "mean_latency_seconds": _mean(latencies),
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "p95_latency_seconds": _percentile(latencies, 0.95),
            "min_latency_seconds": min(latencies) if latencies else None,
            "max_latency_seconds": max(latencies) if latencies else None,
        })
        aggregate["macro_mrr"] = _mean(
            [row["mrr"] for row in scored if row.get("mrr") is not None]
        )
        for k in K_VALUES:
            for metric in ("precision", "recall", "f1"):
                aggregate[f"macro_{metric}@{k}"] = _mean(
                    [row[f"{metric}@{k}"] for row in scored if row.get(f"{metric}@{k}") is not None]
                )
                aggregate[f"micro_{metric}@{k}"] = None
            aggregate[f"macro_ndcg@{k}"] = _mean(
                [row[f"ndcg@{k}"] for row in scored if row.get(f"ndcg@{k}") is not None]
            )
            hit_count = sum(row[f"hits@{k}"] for row in scored)
            precision_denominator = k * len(scored)
            recall_denominator = sum(row["relevant_count"] for row in scored)
            micro_precision = (
                hit_count / precision_denominator if precision_denominator else None
            )
            micro_recall = hit_count / recall_denominator if recall_denominator else None
            aggregate[f"micro_precision@{k}"] = micro_precision
            aggregate[f"micro_recall@{k}"] = micro_recall
            aggregate[f"micro_f1@{k}"] = (
                2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision is not None
                and micro_recall is not None
                and micro_precision + micro_recall
                else 0.0
                if micro_precision is not None and micro_recall is not None
                else None
            )
        result.append(aggregate)
    return result


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _strategy_counts(raw_cases):
    counts = {}
    for strategy in STRATEGIES:
        rows = [case["strategies"][strategy] for case in raw_cases]
        counts[strategy] = {
            "completed": sum(row["status"] == "completed" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "unavailable": sum(row["status"] == "unavailable" for row in rows),
            "empty": sum(
                row["status"] == "completed" and row["retrieved_count"] == 0
                for row in rows
            ),
        }
    return counts


def evidence_reference_counts(drafts_by_id):
    counts = defaultdict(int)
    for draft in drafts_by_id.values():
        for assertion in draft["proposed_gold"].get("expected_assertions", []):
            for evidence in assertion.get("evidence", []):
                counts[evidence["verification_status"]] += 1
    return dict(sorted(counts.items()))


def write_svg_chart(path: Path, rows: list[dict]):
    width = 980
    row_height = 54
    height = 100 + row_height * len(rows)
    x0 = 250
    max_width = 250
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#222}"
        ".title{font-size:20px;font-weight:600}.label{font-size:13px}"
        ".value{font-size:12px}.axis{stroke:#aaa;stroke-width:1}</style>",
        '<text class="title" x="20" y="30">Provisional retrieval metrics (macro average)</text>',
        '<text class="label" x="250" y="58">0</text>',
        '<text class="label" x="490" y="58">1.0</text>',
        '<text class="label" x="545" y="58">Recall@5</text>',
        '<text class="label" x="690" y="58">MRR</text>',
        f'<line class="axis" x1="{x0}" y1="64" x2="{x0}" y2="{height - 15}"/>',
    ]
    for index, row in enumerate(rows):
        y = 85 + index * row_height
        label = html.escape(str(row["strategy"]))
        parts.append(f'<text class="label" x="20" y="{y + 15}">{label}</text>')
        for metric, x, color, value_x in (
            ("macro_recall@5", x0, "#2c7fb8", 510),
            ("macro_mrr", 645, "#41ab5d", 895),
        ):
            value = row.get(metric)
            if value is None:
                parts.append(
                    f'<text class="value" x="{x}" y="{y + 15}">not available</text>'
                )
                continue
            bar_width = max_width * max(0.0, min(1.0, value))
            parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_width:.2f}" height="18" fill="{color}"/>'
            )
            parts.append(
                f'<text class="value" x="{value_x}" y="{y + 14}">{value:.3f}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_summary(
    cases,
    source_chunks,
    input_hashes,
    availability,
    index_details,
    overall_rows,
    common_rows,
    raw_cases,
    limit,
    evidence_counts,
):
    annotation_action_counts = defaultdict(int)
    labelable_cases = 0
    for case in cases:
        draft = case["_draft"]
        annotation_action_counts[draft["proposed_gold"]["expected_action"]] += 1
        relevant = _relevant_chunk_ids(draft)
        labelable_cases += bool(relevant)
    out = [
        "# Retrieval Evaluation Summary",
        "",
        f"- Questions evaluated: {len(raw_cases)} of {len(cases)}.",
        f"- Chunk corpus: {len(source_chunks):,} chunks; SHA256 `{input_hashes[CHUNKS_PATH.name]}`.",
        f"- Benchmark SHA256: `{input_hashes[BENCHMARK_PATH.name]}`.",
        f"- Retrieval top-K: {TOP_K}; reported cutoffs: {', '.join(map(str, K_VALUES))}.",
        "- The retrieval pipeline's baseline default is top-5; this evaluation requests top-10 to calculate @10 as well as @1/@3/@5.",
        f"- Query embedding model: `{DEFAULT_MODEL_NAME}` at revision `{DEFAULT_MODEL_REVISION}`.",
        f"- Embedding warm-up: {index_details['embedding_warmup_seconds']:.4f}s using the first original question (`{index_details['warmup_question_id']}`); excluded from per-query strategy latency.",
        "- Queries: exact original benchmark wording; no planner, rewriting, query expansion, or answer generation.",
        "- Relevance labels: `supports` references from proposed drafts only; `context_only` and other relations are excluded.",
        f"- Annotation status: {sum(evidence_counts.values())} evidence references are "
        f"`proposed_unverified`; none has independent human verification. These are provisional-label results, not verified-gold performance.",
        "- Human evidence verification has not been performed by this evaluation. Do not present these scores as final research findings until an independent reviewer checks the cited source content and labels.",
        f"- Supporting-evidence labels: {labelable_cases} cases evaluable; {len(cases) - labelable_cases} cases have no supporting chunk label and are excluded from ranking-metric macro averages.",
        f"- Proposed action counts: {dict(sorted(annotation_action_counts.items()))}.",
        "",
        "## Metric definitions",
        "",
        "- Macro P/R/F1@K: arithmetic mean of the per-question metric over successfully retrieved questions with at least one proposed supporting evidence label. Per-question precision uses K as its denominator, even when fewer than K results are returned.",
        "- Micro P@K: total relevant hits across scored questions divided by K times the number of scored questions.",
        "- Micro R@K: total relevant hits across scored questions divided by the total number of proposed relevant chunk IDs for those questions.",
        "- Micro F1@K: harmonic mean of micro P@K and micro R@K. MRR and nDCG are reported as macro means only.",
        "- Operational rates use all questions in the group as denominator, except empty-retrieval rate, which is empty successful searches divided by successful searches. Failed, unavailable, and successful-empty outcomes are distinct.",
        "- Latency summaries use all measured strategy calls in the group, including failed or unavailable calls; P95 uses the nearest-rank definition (rank = ceil(0.95 * N)).",
        "",
        "## Strategy results",
        "",
        "| Strategy | Scored | Macro P@5 | Macro R@5 | Macro F1@5 | Micro P@5 | Micro R@5 | Micro F1@5 | Macro MRR | Macro nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    counts = _strategy_counts(raw_cases)
    failed_case_ids = sorted({
        case["question_id"]
        for case in raw_cases
        if any(result["status"] == "failed" for result in case["strategies"].values())
    })
    labeled_noncommon_ids = sorted(
        case["question_id"]
        for case in raw_cases
        if case["relevant_chunk_ids"] and not case["common_subset"]
    )
    for row in overall_rows:
        name = row["strategy"]
        values = [
            row.get("macro_precision@5"),
            row.get("macro_recall@5"),
            row.get("macro_f1@5"),
            row.get("micro_precision@5"),
            row.get("micro_recall@5"),
            row.get("micro_f1@5"),
            row.get("macro_mrr"),
            row.get("macro_ndcg@10"),
        ]
        formatted = [f"{value:.4f}" if value is not None else "n/a" for value in values]
        out.append(
            f"| {name} | {row['scored_queries']} | " + " | ".join(formatted) + " |"
        )
    out.extend([
        "",
        "## Common evaluation subset",
        "",
        f"- {sum(case['common_subset'] for case in raw_cases)} questions were successfully retrieved by every configured strategy; all strategy comparisons below use exactly this subset.",
        f"- {sum(case['common_subset'] and case['relevant_chunk_ids'] != [] for case in raw_cases)} common-subset questions have proposed supporting evidence labels and contribute to ranking metrics.",
        (
            "- Proposed-label questions excluded from the common subset due to retrieval failures: "
            + ", ".join(f"`{case_id}`" for case_id in labeled_noncommon_ids)
            + "."
            if labeled_noncommon_ids
            else "- No labeled questions were excluded from the common subset."
        ),
        "",
        "| Strategy | Common scored | Macro P@5 | Macro R@5 | Macro F1@5 | Micro P@5 | Micro R@5 | Micro F1@5 | Macro MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in common_rows:
        values = [
            row.get("macro_precision@5"),
            row.get("macro_recall@5"),
            row.get("macro_f1@5"),
            row.get("micro_precision@5"),
            row.get("micro_recall@5"),
            row.get("micro_f1@5"),
            row.get("macro_mrr"),
        ]
        formatted = [f"{value:.4f}" if value is not None else "n/a" for value in values]
        out.append(
            f"| {row['strategy']} | {row['scored_queries']} | "
            + " | ".join(formatted)
            + " |"
        )
    out.extend([
        "",
        "## Operational metrics",
        "",
        "| Strategy | Successful | Failed | Unavailable | Empty (count/rate) | Success rate | Failure rate | Mean / median / P95 latency (s) | Min / max latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in overall_rows:
        empty_rate = (
            f"{row['empty_retrieval_count']} / {row['empty_retrieval_rate']:.4f}"
            if row["empty_retrieval_rate"] is not None
            else f"{row['empty_retrieval_count']} / n/a"
        )
        latencies = " / ".join(
            f"{row[key]:.4f}" if row[key] is not None else "n/a"
            for key in ("mean_latency_seconds", "median_latency_seconds", "p95_latency_seconds")
        )
        min_max = " / ".join(
            f"{row[key]:.4f}" if row[key] is not None else "n/a"
            for key in ("min_latency_seconds", "max_latency_seconds")
        )
        out.append(
            f"| {row['strategy']} | {row['successful_queries']} | {row['failed_queries']} | "
            f"{row['unavailable_queries']} | {empty_rate} | "
            f"{row['successful_retrieval_rate']:.4f} | {row['failed_retrieval_rate']:.4f} | "
            f"{latencies} | {min_max} |"
        )
    out.extend([
        "",
        "## Retrieval failures",
        "",
        (
            "- No strategy calls failed."
            if not failed_case_ids
            else f"- Failed question IDs: {', '.join(f'`{case_id}`' for case_id in failed_case_ids)}. "
            "Chroma-backed strategies reject the requested financial-year filters for these out-of-index periods; "
            "the exact per-strategy errors are retained in `raw_retrieval_results.json`. No unfiltered fallback was used."
        ),
        "",
        "## Index availability and limitations",
        "",
        f"- Default chunk-semantic collection: `{CHROMA_COLLECTION}` at `{CHROMA_PATH}`; "
        f"{availability.get('chunk_semantic', {}).get('indexed_chunks', 'unavailable')} indexed chunks.",
        "- `baseline` and `chunk_semantic` use the same direct Chroma chunk-semantic path; both are listed to make the configured default explicit.",
        f"- Keyword semantic path: {availability.get('keyword_semantic', {}).get('status')}; "
        f"{availability.get('keyword_semantic', {}).get('indexed_chunks', 0)} pilot chunks. "
        f"Configured default path `{index_details['default_keyword_semantic_path']['expected_path']}` "
        f"is {index_details['default_keyword_semantic_path']['status']}; pilot results are not full-corpus production results.",
        f"- Keyword lexical path: {availability.get('keyword_lexical', {}).get('status')}; "
        f"{availability.get('keyword_lexical', {}).get('indexed_chunks', 0)} pilot chunks. "
        f"Configured default path `{index_details['default_keyword_lexical_path']['expected_path']}` "
        f"is {index_details['default_keyword_lexical_path']['status']}; pilot results are not full-corpus production results.",
        "- Hybrid paths use the existing RRF implementation and its default per-path candidate limit of "
        f"{DEFAULT_PATH_CANDIDATE_LIMITS}; no retrieval implementation or index was rebuilt.",
        "- Metadata filtering uses the benchmark's exact `company`, `financial_year`, and `chunk_type` fields. The existing retriever accepts only one company filter; multi-company cases retain the benchmark's primary `company` filter.",
        "- When an index rejects a filter or a query fails, the error is preserved in the raw results; no unfiltered fallback is used.",
        "",
        "## Files and reproducibility",
        "",
        "- `raw_retrieval_results.json`: per-question rankings, scores, provenance, diagnostics, failures, and provisional relevance labels.",
        "- `metrics_overall.csv`, `metrics_by_company.csv`, `metrics_by_category.csv`, `per_query_metrics.csv`: macro ranking metrics and retrieval coverage.",
        "- `metrics_common_subset.csv`: macro and micro metrics evaluated on the same successful question set for every strategy.",
        "- `common_evaluation_subset.csv`: exact question IDs and inclusion flags used for the fair comparison.",
        "- `strategy_metrics.svg`: comparison chart.",
        f"- Evaluated limit: {limit if limit is not None else 'all 60 questions'}.",
        "- Input file SHA256 values:",
    ])
    for filename, digest in input_hashes.items():
        out.append(f"  - `{filename}`: `{digest}`")
    out.extend([
        "",
        "The benchmark and annotation files are inputs only. No claims about answer correctness are made, and metrics must be recomputed after human review changes the evidence labels.",
        "",
    ])
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval against provisional evidence labels.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_PATH)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases (for a smoke run).")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer.")

    benchmark, benchmark_cases, drafts_by_id, chunks, source_chunks, input_hashes = load_inputs()
    retrievers, availability, details = _build_retrievers(
        source_chunks,
        benchmark_cases[0]["question"],
        benchmark_cases[0]["id"],
    )
    raw_cases, metric_rows = evaluate_cases(
        benchmark_cases,
        drafts_by_id,
        retrievers,
        availability,
        details,
        args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = aggregate_rows(metric_rows, ("strategy",))
    company_rows = aggregate_rows(metric_rows, ("strategy", "company"))
    category_rows = aggregate_rows(metric_rows, ("strategy", "category"))
    common_rows = aggregate_rows(
        [row for row in metric_rows if row["common_subset"]],
        ("strategy",),
    )
    evidence_counts = evidence_reference_counts(drafts_by_id)
    common_case_rows = [
        {
            "question_id": case["question_id"],
            "company": case["company"],
            "category": case["category"],
            "common_subset": case["common_subset"],
            "supporting_evidence_label_count": len(case["relevant_chunk_ids"]),
        }
        for case in raw_cases
    ]
    write_csv(args.output_dir / "per_query_metrics.csv", metric_rows)
    write_csv(args.output_dir / "metrics_overall.csv", overall_rows)
    write_csv(args.output_dir / "metrics_by_company.csv", company_rows)
    write_csv(args.output_dir / "metrics_by_category.csv", category_rows)
    write_csv(args.output_dir / "metrics_common_subset.csv", common_rows)
    write_csv(args.output_dir / "common_evaluation_subset.csv", common_case_rows)

    metadata = {
        "evaluation_type": "retrieval_only",
        "query_policy": "exact_original_benchmark_question",
        "top_k": TOP_K,
        "metric_cutoffs": list(K_VALUES),
        "relevance_policy": "proposed_gold evidence with relation=supports only",
        "annotation_status": "proposed_unverified",
        "evidence_reference_status_counts": evidence_counts,
        "human_verified_evidence_reference_count": evidence_counts.get("human_verified", 0),
        "human_evidence_verification_performed": False,
        "input_sha256": input_hashes,
        "source_corpus_chunk_count": len(chunks),
        "benchmark_case_count": len(benchmark_cases),
        "evaluated_case_count": len(raw_cases),
        "annotation_action_counts": dict(sorted(
            (action, sum(
                draft["proposed_gold"]["expected_action"] == action
                for draft in drafts_by_id.values()
            ))
            for action in {
                draft["proposed_gold"]["expected_action"]
                for draft in drafts_by_id.values()
            }
        )),
        "strategy_availability": availability,
        "strategy_descriptions": STRATEGY_DESCRIPTIONS,
        "index_details": {
            key: value for key, value in details.items() if key != "hybrid"
        },
        "hybrid_path_candidate_limits": DEFAULT_PATH_CANDIDATE_LIMITS,
    }
    raw_output = {
        "metadata": metadata,
        "cases": raw_cases,
    }
    (args.output_dir / "raw_retrieval_results.json").write_text(
        json.dumps(raw_output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_svg_chart(args.output_dir / "strategy_metrics.svg", overall_rows)

    summary_cases = [
        {**case, "_draft": drafts_by_id[case["id"]]}
        for case in benchmark_cases
    ]
    summary = build_summary(
        summary_cases,
        source_chunks,
        input_hashes,
        availability,
        details,
        overall_rows,
        common_rows,
        raw_cases,
        args.limit,
        evidence_counts,
    )
    (args.output_dir / "evaluation_summary.md").write_text(summary, encoding="utf-8")
    print(f"Evaluated {len(raw_cases)} questions across {len(STRATEGIES)} strategy labels.")
    print(f"Results: {args.output_dir}")
    for strategy, counts in _strategy_counts(raw_cases).items():
        print(
            f"{strategy}: completed={counts['completed']}, failed={counts['failed']}, "
            f"unavailable={counts['unavailable']}"
        )


if __name__ == "__main__":
    main()
