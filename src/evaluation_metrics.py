"""Ranking metrics for retrieval evaluation."""

from collections.abc import Iterable
from math import log2


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def ranking_metrics(
    retrieved_ids: Iterable[str],
    relevant_ids: Iterable[str],
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> dict[str, float | int | None]:
    """Calculate binary-ranking metrics; metrics are null without relevance labels."""
    ranked = _unique(retrieved_ids)
    relevant = set(relevant_ids)
    cutoffs = tuple(k_values)
    if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in cutoffs):
        raise ValueError("Each cutoff must be a positive integer.")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Metric cutoffs must be unique.")

    metrics: dict[str, float | int | None] = {"relevant_count": len(relevant)}
    if not relevant:
        metrics["mrr"] = None
        for k in cutoffs:
            metrics[f"hits@{k}"] = 0
            metrics[f"precision@{k}"] = None
            metrics[f"recall@{k}"] = None
            metrics[f"f1@{k}"] = None
            metrics[f"ndcg@{k}"] = None
        return metrics

    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(ranked, start=1)
        if chunk_id in relevant
    ]
    metrics["mrr"] = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    for k in cutoffs:
        hits = sum(chunk_id in relevant for chunk_id in ranked[:k])
        metrics[f"hits@{k}"] = hits
        precision = hits / k
        recall = hits / len(relevant)
        metrics[f"precision@{k}"] = precision
        metrics[f"recall@{k}"] = recall
        metrics[f"f1@{k}"] = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        dcg = sum(
            1.0 / log2(rank + 1)
            for rank, chunk_id in enumerate(ranked[:k], start=1)
            if chunk_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
        metrics[f"ndcg@{k}"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return metrics
