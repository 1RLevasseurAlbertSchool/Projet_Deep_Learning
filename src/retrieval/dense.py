"""Normalised sentence-embedding retrieval over the LEDGAR training clauses."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import INDEX_DIR

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_REQUIRED_COLUMNS = ("idx", "text", "label", "label_name")


def _validate_bank(bank: pd.DataFrame) -> None:
    missing = set(_REQUIRED_COLUMNS).difference(bank.columns)
    if missing:
        raise ValueError(f"Clause bank is missing columns: {sorted(missing)}")
    if bank.empty:
        raise ValueError("Clause bank must contain at least one clause")
    if bank["idx"].duplicated().any():
        raise ValueError("Clause bank idx values must be unique")
    if not bank["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every clause must have non-empty text")


class DenseRetriever:
    """Search a clause bank using cosine similarity of normalised embeddings."""

    def __init__(
        self,
        bank: pd.DataFrame,
        embeddings: np.ndarray,
        encoder: Any,
        *,
        cache_hit: bool = False,
    ) -> None:
        _validate_bank(bank)

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(bank) or vectors.shape[1] == 0:
            raise ValueError("Embedding matrix must have one non-empty vector per clause")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding matrix contains non-finite values")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding matrix contains a zero vector")

        self.bank = bank.loc[:, _REQUIRED_COLUMNS].reset_index(drop=True).copy()
        self.embeddings = vectors / norms
        self.encoder = encoder
        self.cache_hit = cache_hit

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k clauses with ``idx``, text, label, label_name and cosine score."""
        return self.search_many([query], k=k)[0]

    def search_many(self, queries: list[str], k: int = 5) -> list[list[dict]]:
        """Search multiple queries in one encoder and matrix-multiplication batch."""
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if not queries:
            return []
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("every query must be a non-empty string")

        encoded = self.encoder.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(queries) > 1,
        )
        query_vectors = np.array(encoded, dtype=np.float32, copy=True)
        if query_vectors.ndim != 2 or query_vectors.shape != (len(queries), self.embeddings.shape[1]):
            raise ValueError("Query embedding dimension does not match the clause bank")
        if not np.isfinite(query_vectors).all():
            raise ValueError("Query embedding contains non-finite values")
        query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        if np.any(query_norms == 0):
            raise ValueError("Query embedding contains a zero vector")

        query_vectors /= query_norms
        n_results = min(k, len(self.bank))
        results = []
        for start in range(0, len(queries), 128):
            batch_scores = query_vectors[start : start + 128] @ self.embeddings.T
            for scores in batch_scores:
                if n_results == len(self.bank):
                    candidate_rows = np.arange(len(self.bank))
                else:
                    candidate_rows = np.argpartition(-scores, n_results - 1)[:n_results]
                order = np.lexsort((candidate_rows, -scores[candidate_rows]))
                ranked_rows = candidate_rows[order]
                query_results = []
                for row in ranked_rows:
                    clause = self.bank.iloc[row]
                    query_results.append({
                        "idx": int(clause["idx"]),
                        "text": clause["text"],
                        "label": int(clause["label"]),
                        "label_name": clause["label_name"],
                        "score": float(scores[row]),
                    })
                results.append(query_results)
        return results


def _bank_fingerprint(bank: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for idx, text in bank[["idx", "text"]].itertuples(index=False, name=None):
        for value in (str(idx).encode("utf-8"), text.encode("utf-8")):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def build_dense_index(
    bank: pd.DataFrame,
    model_name: str = MODEL_NAME,
    cache_dir: str | Path = INDEX_DIR,
    batch_size: int = 64,
    device: str | None = None,
) -> DenseRetriever:
    """Load cached bank embeddings or encode the bank and cache normalised vectors."""
    _validate_bank(bank)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from sentence_transformers import SentenceTransformer

    model_kwargs = {"device": device} if device is not None else {}
    encoder = SentenceTransformer(model_name, **model_kwargs)
    fingerprint = _bank_fingerprint(bank)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    cache_path = cache_dir / f"{safe_model_name}.npz"
    cache_hit = False
    embeddings = None

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            if metadata == {
                "model_name": model_name,
                "bank_fingerprint": fingerprint,
                "rows": len(bank),
            }:
                embeddings = cached["embeddings"]
                cache_hit = True

    if embeddings is None:
        embeddings = encoder.encode(
            bank["text"].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        metadata = {
            "model_name": model_name,
            "bank_fingerprint": fingerprint,
            "rows": len(bank),
        }
        temporary_path = cache_path.with_name(f".{cache_path.name}.tmp.npz")
        np.savez(
            temporary_path,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            metadata=json.dumps(metadata),
        )
        temporary_path.replace(cache_path)

    return DenseRetriever(bank, embeddings, encoder, cache_hit=cache_hit)
