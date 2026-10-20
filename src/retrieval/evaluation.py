"""Ranking metrics and evaluation helpers for retrieval experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ranking_metrics(
    rankings: Sequence[Sequence[int]],
    relevant_ids: Sequence[set[int]],
    *,
    precision_k: int = 5,
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> dict[str, float | int]:
    """Compute macro P@k, MRR@k and nDCG@k for binary relevance."""
    if len(rankings) != len(relevant_ids):
        raise ValueError("Each ranking must have exactly one relevance set")
    if not rankings:
        raise ValueError("At least one ranking is required")
    if precision_k < 1 or mrr_k < 1 or ndcg_k < 1:
        raise ValueError("Metric cutoffs must be positive")

    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for ranking, relevant in zip(rankings, relevant_ids):
        ranked = list(ranking)
        top_precision = ranked[:precision_k]
        precisions.append(sum(item in relevant for item in top_precision) / precision_k)

        first_relevant_rank = next(
            (rank for rank, item in enumerate(ranked[:mrr_k], start=1) if item in relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

        gains = [1.0 if item in relevant else 0.0 for item in ranked[:ndcg_k]]
        dcg = sum(gain / np.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
        ideal_hits = min(len(relevant), ndcg_k)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    return {
        "n_queries": len(rankings),
        f"precision@{precision_k}": float(np.mean(precisions)),
        f"mrr@{mrr_k}": float(np.mean(reciprocal_ranks)),
        f"ndcg@{ndcg_k}": float(np.mean(ndcgs)),
    }


def evaluate_same_label_with_bank(
    query_labels: Sequence[int],
    rankings: Sequence[Sequence[dict]],
    bank_labels: dict[int, int],
    *,
    precision_k: int = 5,
    ndcg_k: int = 10,
) -> dict[str, float | int]:
    """Evaluate top-k rankings using a complete train-bank idx→label lookup."""
    if len(query_labels) != len(rankings):
        raise ValueError("Each query label must have exactly one ranking")
    indices_by_label: dict[int, set[int]] = {}
    for idx, label in bank_labels.items():
        indices_by_label.setdefault(int(label), set()).add(int(idx))
    query_relevance = [
        indices_by_label.get(int(query_label), set())
        for query_label in query_labels
    ]
    return ranking_metrics(
        [[int(result["idx"]) for result in ranking] for ranking in rankings],
        query_relevance,
        precision_k=precision_k,
        ndcg_k=ndcg_k,
    )


def evaluate_manual_judgments(
    query_rankings: dict[str, Sequence[dict]],
    judgments: dict[str, set[int]],
    *,
    precision_k: int = 5,
    ndcg_k: int = 10,
) -> dict[str, float | int]:
    """Evaluate only the supplied manually judged questions."""
    missing = set(judgments).difference(query_rankings)
    if missing:
        raise ValueError(f"Judgments have no matching query rankings: {sorted(missing)}")
    query_ids = list(judgments)
    return ranking_metrics(
        [
            [int(result["idx"]) for result in query_rankings[query_id]]
            for query_id in query_ids
        ],
        [judgments[query_id] for query_id in query_ids],
        precision_k=precision_k,
        ndcg_k=ndcg_k,
    )
