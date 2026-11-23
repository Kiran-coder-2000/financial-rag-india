from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation_metrics import ranking_metrics
from scripts.evaluate_retrieval import aggregate_rows


def test_ranking_metrics_compute_binary_scores_and_fixed_precision_denominator():
    metrics = ranking_metrics(["a", "x", "b"], ["b", "c"], (1, 3))

    assert metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["precision@1"] == 0
    assert metrics["recall@1"] == 0
    assert metrics["precision@3"] == pytest.approx(1 / 3)
    assert metrics["recall@3"] == pytest.approx(1 / 2)
    assert metrics["f1@3"] == pytest.approx(0.4)
    assert metrics["ndcg@3"] == pytest.approx((1 / 2) / (1 + 1 / 1.584962500721156))
    assert metrics["hits@3"] == 1


def test_ranking_metrics_deduplicate_retrieved_ids():
    metrics = ranking_metrics(["a", "a", "b"], ["a", "b"], (2,))

    assert metrics["precision@2"] == 1
    assert metrics["recall@2"] == 1
    assert metrics["mrr"] == 1


def test_ranking_metrics_are_not_scored_without_supporting_labels():
    metrics = ranking_metrics(["a"], [], (1,))

    assert metrics["relevant_count"] == 0
    assert metrics["mrr"] is None
    assert metrics["precision@1"] is None
    assert metrics["hits@1"] == 0


def test_aggregate_rows_reports_macro_micro_and_operational_metrics():
    rows = []
    for question_id, relevant_count, hits, precision, recall, latency, empty in (
        ("q1", 1, 1, 1.0, 1.0, 1.0, False),
        ("q2", 9, 0, 0.0, 0.0, 3.0, True),
    ):
        row = {
            "question_id": question_id,
            "strategy": "test",
            "retrieval_status": "completed",
            "metric_status": "scored",
            "relevant_count": relevant_count,
            "retrieved_count": 0 if empty else 1,
            "empty_retrieval": empty,
            "latency_seconds": latency,
            "mrr": precision,
        }
        for k in (1, 3, 5, 10):
            row[f"hits@{k}"] = hits
            row[f"precision@{k}"] = precision if k == 1 else hits / k
            row[f"recall@{k}"] = recall
            row[f"f1@{k}"] = (
                2 * row[f"precision@{k}"] * recall
                / (row[f"precision@{k}"] + recall)
                if row[f"precision@{k}"] + recall
                else 0
            )
            row[f"ndcg@{k}"] = precision
        rows.append(row)

    aggregate = aggregate_rows(rows, ("strategy",))[0]

    assert aggregate["macro_precision@1"] == pytest.approx(0.5)
    assert aggregate["micro_precision@1"] == pytest.approx(0.5)
    assert aggregate["macro_recall@1"] == pytest.approx(0.5)
    assert aggregate["micro_recall@1"] == pytest.approx(0.1)
    assert aggregate["micro_f1@1"] == pytest.approx(1 / 6)
    assert aggregate["successful_retrieval_rate"] == 1
    assert aggregate["failed_retrieval_rate"] == 0
    assert aggregate["empty_retrieval_count"] == 1
    assert aggregate["empty_retrieval_rate"] == pytest.approx(0.5)
    assert aggregate["p95_latency_seconds"] == 3
    assert aggregate["min_latency_seconds"] == 1
    assert aggregate["max_latency_seconds"] == 3


def test_aggregate_rows_separates_empty_failed_and_unavailable_retrievals():
    rows = []
    for question_id, status, empty in (
        ("q1", "completed", True),
        ("q2", "failed", False),
        ("q3", "unavailable", False),
    ):
        row = {
            "question_id": question_id,
            "strategy": "test",
            "retrieval_status": status,
            "metric_status": "not_evaluable" if status == "completed" else status,
            "relevant_count": 0,
            "retrieved_count": 0,
            "empty_retrieval": empty,
            "latency_seconds": 0.1,
            "mrr": None,
        }
        for k in (1, 3, 5, 10):
            row[f"hits@{k}"] = 0
            for name in ("precision", "recall", "f1", "ndcg"):
                row[f"{name}@{k}"] = None
        rows.append(row)

    aggregate = aggregate_rows(rows, ("strategy",))[0]

    assert aggregate["successful_queries"] == 1
    assert aggregate["failed_queries"] == 1
    assert aggregate["unavailable_queries"] == 1
    assert aggregate["successful_retrieval_rate"] == pytest.approx(1 / 3)
    assert aggregate["failed_retrieval_rate"] == pytest.approx(1 / 3)
    assert aggregate["unavailable_retrieval_rate"] == pytest.approx(1 / 3)
    assert aggregate["empty_retrieval_count"] == 1
    assert aggregate["empty_retrieval_rate"] == 1


@pytest.mark.parametrize("cutoffs", [(0,), (True,), (1, 1)])
def test_ranking_metrics_reject_invalid_cutoffs(cutoffs):
    with pytest.raises(ValueError):
        ranking_metrics([], ["a"], cutoffs)
