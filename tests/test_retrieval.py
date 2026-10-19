"""CPU-only tests for dense sentence-embedding retrieval."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.retrieval.dense import DenseRetriever, build_dense_index


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
