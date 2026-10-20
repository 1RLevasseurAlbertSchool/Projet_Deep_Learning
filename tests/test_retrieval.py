"""CPU-only tests for dense sentence-embedding retrieval."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.retrieval.dense import DenseRetriever, build_dense_index
from src.retrieval.evaluation import (
    evaluate_manual_judgments,
    evaluate_same_label_with_bank,
    ranking_metrics,
)
from src.retrieval.hybrid import BM25Retriever, HybridRetriever


class FakeEncoder:
    def encode(self, sentences, **kwargs):
        vectors = {
            "query": [1.0, 0.0],
            "parallel clause": [0.9, 0.1],
            "unrelated clause": [0.0, 1.0],
            "same-score clause": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in sentences], dtype=np.float32)


@pytest.fixture
def bank():
    return pd.DataFrame({
        "idx": [17, 4, 28],
        "text": ["parallel clause", "unrelated clause", "same-score clause"],
        "label": [2, 1, 2],
        "label_name": ["Termination", "Payment", "Termination"],
    })


def test_search_returns_cosine_ranked_rows_with_shared_contract(bank):
    retriever = DenseRetriever(
        bank,
        np.asarray([[9.0, 1.0], [0.0, 3.0], [1.0, 0.0]], dtype=np.float32),
        FakeEncoder(),
    )

    results = retriever.search("query", k=2)

    assert [row["idx"] for row in results] == [28, 17]
    assert all(set(row) == {"idx", "text", "label", "label_name", "score"} for row in results)
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.9 / np.sqrt(0.82))


def test_search_is_stable_for_equal_scores_and_caps_k_to_bank_size(bank):
    retriever = DenseRetriever(
        bank,
        np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        FakeEncoder(),
    )

    results = retriever.search("query", k=10)

    assert [row["idx"] for row in results] == [17, 28, 4]


def test_search_many_matches_single_query_results(bank):
    retriever = DenseRetriever(
        bank,
        np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        FakeEncoder(),
    )
    many = retriever.search_many(["query", "query"], k=2)
    assert many == [retriever.search("query", k=2), retriever.search("query", k=2)]


@pytest.mark.parametrize("query,k", [("", 5), ("  ", 5), ("query", 0), ("query", -1), ("query", True)])
def test_search_rejects_invalid_query_or_k(bank, query, k):
    retriever = DenseRetriever(
        bank,
        np.ones((len(bank), 2), dtype=np.float32),
        FakeEncoder(),
    )
    with pytest.raises(ValueError):
        retriever.search(query, k=k)


def test_constructor_rejects_duplicate_ids_and_zero_embeddings(bank):
    duplicate_bank = bank.copy()
    duplicate_bank.loc[1, "idx"] = duplicate_bank.loc[0, "idx"]
    with pytest.raises(ValueError, match="unique"):
        DenseRetriever(duplicate_bank, np.ones((3, 2)), FakeEncoder())

    with pytest.raises(ValueError, match="zero vector"):
        DenseRetriever(bank, np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), FakeEncoder())


def test_build_dense_index_reuses_matching_cache(bank, tmp_path, monkeypatch):
    encoders = []

    class CacheTestEncoder:
        def __init__(self, model_name, **kwargs):
            self.calls = 0
            encoders.append(self)

        def encode(self, sentences, **kwargs):
            self.calls += 1
            return np.tile([1.0, 0.0], (len(sentences), 1))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=CacheTestEncoder),
    )

    first = build_dense_index(bank, model_name="test-model", cache_dir=tmp_path)
    second = build_dense_index(bank, model_name="test-model", cache_dir=tmp_path)

    assert not first.cache_hit
    assert second.cache_hit
    assert [encoder.calls for encoder in encoders] == [1, 0]


def test_ranking_metrics_and_same_label_relevance():
    rankings = [
        [10, 11, 12],
        [11, 10, 12],
    ]
    result = ranking_metrics(rankings, [{10, 12}, {12}], precision_k=2, ndcg_k=3)
    assert result["n_queries"] == 2
    assert result["precision@2"] == pytest.approx(0.25)
    assert result["mrr@10"] == pytest.approx(2 / 3)
    assert 0 <= result["ndcg@3"] <= 1

    same_label = evaluate_same_label_with_bank(
        [1],
        [[{"idx": 10, "label": 1}, {"idx": 11, "label": 0}]],
        {10: 1, 11: 0, 12: 1},
    )
    assert same_label["precision@5"] == pytest.approx(0.2)
    assert same_label["mrr@10"] == pytest.approx(1.0)


def test_manual_query_evaluation_requires_matching_query_ids():
    results = {"q1": [{"idx": 4}, {"idx": 8}]}
    metrics = evaluate_manual_judgments(results, {"q1": {8}}, precision_k=2, ndcg_k=2)
    assert metrics["precision@2"] == pytest.approx(0.5)
    assert metrics["mrr@10"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no matching query"):
        evaluate_manual_judgments(results, {"missing": {8}})


def test_bm25_and_hybrid_return_shared_contract_and_filter_label(bank):
    dense = DenseRetriever(
        bank,
        np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        FakeEncoder(),
    )
    bm25 = BM25Retriever(bank)
    hybrid = HybridRetriever(dense, bm25)

    keyword_results = bm25.search("parallel clause", k=2)
    assert keyword_results[0]["idx"] == 17
    assert set(keyword_results[0]) == {"idx", "text", "label", "label_name", "score"}

    unfiltered = hybrid.search("parallel clause", k=3)
    filtered = hybrid.search("parallel clause", k=2, predicted_label=2)
    assert len(unfiltered) == 3
    assert len(filtered) == 2
    assert all(result["label"] == 2 for result in filtered)
    assert all("dense_score" in result and "bm25_score" in result for result in unfiltered)

    filtered_after_expansion = hybrid.search(
        "unrelated clause",
        k=1,
        predicted_label=2,
        candidate_k=1,
    )
    assert len(filtered_after_expansion) == 1
    assert filtered_after_expansion[0]["label"] == 2

    batch = hybrid.search_many(["parallel clause", "unrelated clause"], k=2)
    assert len(batch) == 2
    assert all(len(ranking) == 2 for ranking in batch)
