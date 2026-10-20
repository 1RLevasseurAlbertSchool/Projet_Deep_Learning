"""BM25 and reciprocal-rank fusion for clause retrieval."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.retrieval.dense import DenseRetriever, _validate_bank

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, retaining numeric terms such as ``30``."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """BM25 search over a clause bank, returning the shared retrieval result contract."""

    def __init__(self, bank: pd.DataFrame) -> None:
        _validate_bank(bank)
        self.bank = bank.loc[:, ["idx", "text", "label", "label_name"]].reset_index(drop=True).copy()
        self._index = BM25Okapi([tokenize(text) for text in self.bank["text"]])

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")

        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        matching_rows = (scores > 0).nonzero()[0]
        if not len(matching_rows):
            return []
        candidate_rows = matching_rows[np.argsort(-scores[matching_rows], kind="stable")[:k]]
        return [
            {
                "idx": int(self.bank.iloc[row]["idx"]),
                "text": self.bank.iloc[row]["text"],
                "label": int(self.bank.iloc[row]["label"]),
                "label_name": self.bank.iloc[row]["label_name"],
                "score": float(scores[row]),
            }
            for row in candidate_rows
        ]

    def search_many(self, queries: list[str], k: int = 5) -> list[list[dict]]:
        """Search multiple queries while sharing the already-built BM25 index."""
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if not queries:
            return []
        return [self.search(query, k=k) for query in queries]


class HybridRetriever:
    """Fuse dense cosine and BM25 rankings with reciprocal rank fusion (RRF)."""

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever) -> None:
        if not dense_retriever.bank["idx"].equals(bm25_retriever.bank["idx"]):
            raise ValueError("Dense and BM25 retrievers must use the same ordered clause bank")
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.bank = dense_retriever.bank

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        predicted_label: int | None = None,
        candidate_k: int = 100,
        rrf_constant: int = 60,
    ) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if (
            not isinstance(candidate_k, int)
            or isinstance(candidate_k, bool)
            or candidate_k < 1
            or not isinstance(rrf_constant, int)
            or isinstance(rrf_constant, bool)
            or rrf_constant < 1
        ):
            raise ValueError("candidate_k and rrf_constant must be positive")

        bank_size = len(self.bank)
        fetch_k = min(candidate_k, bank_size)
        while True:
            dense_results = self.dense.search(query, k=fetch_k)
            bm25_results = self.bm25.search(query, k=fetch_k)
            fused = self._fuse(dense_results, bm25_results, k, predicted_label, rrf_constant)
            if len(fused) >= k or fetch_k >= bank_size:
                return fused[:k]
            fetch_k = min(fetch_k * 2, bank_size)

    def search_many(
        self,
        queries: list[str],
        k: int = 5,
        *,
        candidate_k: int = 100,
        rrf_constant: int = 60,
        predicted_labels: list[int] | None = None,
    ) -> list[list[dict]]:
        """Batch dense encoding and fuse rankings for many queries."""
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if (
            not isinstance(candidate_k, int)
            or isinstance(candidate_k, bool)
            or candidate_k < 1
            or not isinstance(rrf_constant, int)
            or isinstance(rrf_constant, bool)
            or rrf_constant < 1
        ):
            raise ValueError("candidate_k and rrf_constant must be positive")
        if not queries:
            return []
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("every query must be a non-empty string")
        if predicted_labels is not None:
            if len(predicted_labels) != len(queries):
                raise ValueError("predicted_labels must match the number of queries")
            return [
                self.search(
                    query,
                    k=k,
                    predicted_label=label,
                    candidate_k=candidate_k,
                    rrf_constant=rrf_constant,
                )
                for query, label in zip(queries, predicted_labels)
            ]

        fetch_k = min(max(candidate_k, k), len(self.bank))
        dense_rankings = self.dense.search_many(queries, k=fetch_k)
        bm25_rankings = self.bm25.search_many(queries, k=fetch_k)
        return [
            self._fuse(dense, bm25, k, None, rrf_constant)[:k]
            for dense, bm25 in zip(dense_rankings, bm25_rankings)
        ]

    @staticmethod
    def _fuse(
        dense_results: list[dict],
        bm25_results: list[dict],
        k: int,
        predicted_label: int | None,
        rrf_constant: int,
    ) -> list[dict]:
        dense_by_id = {int(item["idx"]): item for item in dense_results}
        bm25_by_id = {int(item["idx"]): item for item in bm25_results}
        dense_ranks = {idx: rank for rank, idx in enumerate(dense_by_id, start=1)}
        bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_by_id, start=1)}
        candidate_ids = set(dense_by_id).union(bm25_by_id)
        fused = []
        for idx in candidate_ids:
            item = dense_by_id.get(idx) or bm25_by_id[idx]
            if predicted_label is not None and int(item["label"]) != int(predicted_label):
                continue
            rrf_score = 0.0
            if idx in dense_ranks:
                rrf_score += 1.0 / (rrf_constant + dense_ranks[idx])
            if idx in bm25_ranks:
                rrf_score += 1.0 / (rrf_constant + bm25_ranks[idx])
            fused.append({
                **item,
                "score": rrf_score,
                "dense_score": dense_by_id[idx]["score"] if idx in dense_by_id else None,
                "bm25_score": bm25_by_id[idx]["score"] if idx in bm25_by_id else None,
            })
        fused.sort(key=lambda item: (-item["score"], int(item["idx"])))
        return fused[:k] if predicted_label is None else fused
