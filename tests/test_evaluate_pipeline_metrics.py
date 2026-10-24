import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / "tests" / "evaluate_pipeline_metrics.py"
MODULE_SPEC = importlib.util.spec_from_file_location("evaluate_pipeline_metrics", MODULE_PATH)
evaluation = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = evaluation
MODULE_SPEC.loader.exec_module(evaluation)


def _benchmark(variants=None, cases=None):
    return {
        "cases": cases or [{
            "id": "revenue-2026",
            "question": "What was Infosys's consolidated revenue in FY2026?",
            "rewritten_query": "Infosys consolidated revenues in reported currency terms for fiscal 2026",
            "filters": {"company": "Infosys", "financial_year": "2025-2026"},
            "relevant_chunk_ids": ["gold-text", "gold-table"],
        }],
        "variants": variants or [{
            "name": "original-table",
            "query_source": "question",
            "chunk_type": "table",
            "collection": "legacy_collection",
        }],
    }


def _result(chunk_id, score, rank, chunk_type):
    from src.retrieval import RetrievalResult

    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        chunk={"chunk_id": chunk_id, "text": chunk_id, "chunk_type": chunk_type},
    )


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def test_metrics_use_recall_at_5_recall_at_10_and_first_relevant_rank():
    ranked_ids = [f"other-{index}" for index in range(1, 8)] + ["gold-a", "gold-b", "other-10"]

    recall_5, recall_10, mrr = evaluation.score_ranked_ids(ranked_ids, ["gold-a", "gold-b"])

    assert recall_5 == 0
    assert recall_10 == 1
    assert mrr == pytest.approx(1 / 8)


def test_benchmark_requires_fixed_cases_and_variants(tmp_path):
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        '{"cases": [], "variants": [{"name": "v"}]}',
        encoding="utf-8",
    )
    with pytest.raises(evaluation.BenchmarkError, match="non-empty 'cases'"):
        evaluation.load_benchmark(benchmark_path)


def test_variant_uses_prewritten_query_and_exact_filter_values():
    variant = {
        "name": "rewritten-table",
        "query_source": "rewritten_query",
        "chunk_type": "table",
        "collection": "serialized_collection",
    }
    case = _benchmark([variant])["cases"][0]
    retriever = FakeRetriever([_result("gold-table", 0.8, 1, "table")])

    metrics = evaluation.evaluate_variant([case], variant, retriever, candidate_k=20)

    assert metrics.recall_at_5 == 0.5
    assert retriever.calls == [{
        "query": case["rewritten_query"],
        "top_k": 20,
        "company": "Infosys",
        "financial_year": "2025-2026",
        "chunk_type": "table",
    }]


def test_text_only_variant_uses_text_chunk_filter():
    variant = {"name": "text-only", "chunk_type": "text"}
    case = _benchmark([variant])["cases"][0]
    retriever = FakeRetriever([_result("gold-text", 0.8, 1, "text")])

    evaluation.evaluate_variant([case], variant, retriever, candidate_k=20)

    assert retriever.calls[0]["chunk_type"] == "text"


def test_unfiltered_variant_can_boost_table_hits_before_metrics():
    variant = {
        "name": "unfiltered-table-boost",
        "query_source": "question",
        "chunk_type": "any",
        "table_score_boost": 0.1,
    }
    case = _benchmark([variant])["cases"][0]
    retriever = FakeRetriever([
        _result("irrelevant-text", 0.82, 1, "text"),
        _result("gold-table", 0.75, 2, "table"),
    ])

    metrics = evaluation.evaluate_variant([case], variant, retriever, candidate_k=20)

    assert metrics.mrr == pytest.approx(1.0)
    assert retriever.calls[0]["chunk_type"] is None


def test_multiplicative_boost_changes_close_ranking():
    results = [
        _result("text-hit", 0.80, 1, "text"),
        _result("table-hit", 0.75, 2, "table"),
    ]

    ranked = evaluation._ranked_results(results, table_score_boost=0.1)

    assert [result.chunk_id for result in ranked] == ["table-hit", "text-hit"]
    assert 0.75 * 1.1 > 0.80


def test_multiplicative_boost_does_not_change_ranking_when_gap_is_too_large():
    results = [
        _result("text-hit", 0.80, 1, "text"),
        _result("table-hit", 0.70, 2, "table"),
    ]

    ranked = evaluation._ranked_results(results, table_score_boost=0.1)

    assert [result.chunk_id for result in ranked] == ["text-hit", "table-hit"]
    assert 0.70 * 1.1 < 0.80


def test_zero_boost_preserves_original_retrieval_order():
    results = [
        _result("first", 0.80, 1, "text"),
        _result("second", 0.75, 2, "table"),
    ]

    assert evaluation._ranked_results(results, table_score_boost=0) == results


def test_variant_metrics_include_candidate_pool_diagnostics():
    variant = {"name": "unfiltered", "chunk_type": "any"}
    case = _benchmark([variant])["cases"][0]
    retriever = FakeRetriever([
        _result("other", 0.9, 1, "text"),
        _result("gold-table", 0.8, 2, "table"),
    ])

    metrics = evaluation.evaluate_variant([case], variant, retriever, candidate_k=10)

    assert metrics.recall_at_5 == pytest.approx(0.5)
    assert metrics.recall_at_10 == pytest.approx(0.5)
    assert metrics.mrr == pytest.approx(0.5)
    diagnostic = metrics.case_diagnostics[0]
    assert diagnostic.case_id == "revenue-2026"
    assert diagnostic.relevant_chunk_count == 2
    assert diagnostic.relevant_ids_retrieved == ["gold-table"]
    assert diagnostic.missing_relevant_ids == ["gold-text"]
    assert diagnostic.first_relevant_rank == 2
    assert diagnostic.precision_at_5 == pytest.approx(0.2)
    assert diagnostic.candidate_counts_by_path == {"chunk_semantic": 2}
    assert diagnostic.candidate_overlaps_by_path == {}
    assert diagnostic.unique_candidate_count == 2
    assert diagnostic.final_candidate_ids == ["other", "gold-table"]
    assert diagnostic.total_latency_seconds >= 0
    assert "gold-table" in evaluation.format_case_diagnostics([metrics])
    assert "not retrieved" not in evaluation.format_case_diagnostics([metrics])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda benchmark: benchmark["cases"][0].update(
            relevant_chunk_ids=["gold", "gold"]
        ),
        lambda benchmark: benchmark["cases"][0].update(
            filters={"company": " Infosys"}
        ),
        lambda benchmark: benchmark["cases"][0].update(
            filters={"company": "Infosys", "financial_year": "FY2026"}
        ),
    ],
)
def test_invalid_relevant_ids_and_filter_values_are_rejected(tmp_path, mutate):
    benchmark = _benchmark()
    mutate(benchmark)
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(benchmark), encoding="utf-8")

    with pytest.raises(evaluation.BenchmarkError):
        evaluation.load_benchmark(path)


def test_whitespace_only_rewritten_query_is_rejected(tmp_path):
    benchmark = _benchmark([{"name": "rewritten", "query_source": "rewritten_query"}])
    benchmark["cases"][0]["rewritten_query"] = " \t "
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(benchmark), encoding="utf-8")

    with pytest.raises(evaluation.BenchmarkError, match="non-empty rewritten_query"):
        evaluation.load_benchmark(path)


def test_collection_variants_are_evaluated_separately():
    benchmark = _benchmark([
        {"name": "legacy", "collection": "legacy_collection", "chunk_type": "table"},
        {"name": "serialized", "collection": "serialized_collection", "chunk_type": "table"},
    ])
    collections = []

    def factory(collection_name):
        collections.append(collection_name)
        return FakeRetriever([_result("gold-table", 0.8, 1, "table")])

    metrics = evaluation.evaluate_benchmark(benchmark, retriever_factory=factory, candidate_k=20)

    assert [metric.name for metric in metrics] == ["legacy", "serialized"]
    assert collections == ["legacy_collection", "serialized_collection"]


def test_experimental_strategy_is_passed_to_hybrid_retriever():
    variant = {
        "name": "hybrid_all",
        "strategy": "hybrid_all",
        "chunk_type": "any",
    }
    case = _benchmark([variant])["cases"][0]

    class FakeHybrid(FakeRetriever):
        def __init__(self):
            super().__init__([_result("gold-text", 0.8, 1, "text")])
            self.last_search_diagnostics = {
                "path_candidate_counts": {"chunk_semantic": 1, "keyword_semantic": 1},
                "path_overlap_counts": {"chunk_semantic|keyword_semantic": 1},
                "unique_candidate_count": 1,
                "path_latencies_seconds": {"chunk_semantic": 0.01, "keyword_semantic": 0.02},
                "total_latency_seconds": 0.03,
                "final_candidate_ids": ["gold-text"],
            }

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return self.results

    retriever = FakeHybrid()
    metrics = evaluation.evaluate_variant([case], variant, retriever, candidate_k=20)

    assert retriever.calls[0]["strategy"] == "hybrid_all"
    assert metrics.case_diagnostics[0].candidate_counts_by_path["chunk_semantic"] == 1
    assert metrics.case_diagnostics[0].candidate_overlaps_by_path[
        "chunk_semantic|keyword_semantic"
    ] == 1
    assert metrics.case_diagnostics[0].final_candidate_ids == ["gold-text"]


def test_verified_gold_loader_excludes_pending_candidate_ids(tmp_path):
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps({
        "cases": [
            {
                "id": "pending",
                "question": "question pending",
                "verification_status": "pending",
                "relevant_chunk_ids": [],
                "candidate_relevant_chunk_ids": ["candidate-not-gold"],
            },
            {
                "id": "verified",
                "question": "verified question",
                "verification_status": "verified",
                "filters": {"company": "Infosys", "financial_year": "2025-2026"},
                "relevant_chunk_ids": ["confirmed-id"],
                "candidate_relevant_chunk_ids": ["unconfirmed-id"],
            },
        ]
    }), encoding="utf-8")

    benchmark, pending_count = evaluation.load_verified_gold_dataset(gold_path)

    assert pending_count == 1
    assert [case["id"] for case in benchmark["cases"]] == ["verified"]
    assert benchmark["cases"][0]["relevant_chunk_ids"] == ["confirmed-id"]
    assert len(benchmark["variants"]) == 7


def test_verified_gold_loader_rejects_pending_only_dataset(tmp_path):
    gold_path = tmp_path / "pending.json"
    gold_path.write_text(json.dumps({
        "cases": [{
            "id": "pending",
            "question": "question",
            "verification_status": "pending",
            "relevant_chunk_ids": [],
            "candidate_relevant_chunk_ids": ["suggestion"],
        }]
    }), encoding="utf-8")

    with pytest.raises(evaluation.BenchmarkError, match="no verified cases"):
        evaluation.load_verified_gold_dataset(gold_path)


@pytest.mark.parametrize(
    "variant",
    [
        {"name": "bad-query", "query_source": "normalized_query"},
        {"name": "bad-type", "chunk_type": "both"},
        {"name": "negative-boost", "table_score_boost": -0.1},
    ],
)
def test_invalid_variant_configuration_rejected(tmp_path, variant):
    benchmark = _benchmark([variant])
    path = tmp_path / "benchmark.json"
    import json

    path.write_text(json.dumps(benchmark), encoding="utf-8")

    with pytest.raises(evaluation.BenchmarkError):
        evaluation.load_benchmark(path)