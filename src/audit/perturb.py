"""Counterfactual and out-of-distribution robustness audit helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

Predictor = Callable[[list[str]], Sequence[dict[str, Any]]]

_ROLE_PAIRS = (
    ("borrower_lender", "borrower", "lender"),
    ("employer_employee", "employer", "employee"),
    ("landlord_tenant", "landlord", "tenant"),
    ("buyer_seller", "buyer", "seller"),
)
_PRONOUN_PAIR = ("he", "she")
_JURISDICTIONS = (
    ("New York", "Luxembourg"),
    ("New York", "Delaware"),
    ("New York", "England"),
)
_CURRENCIES = (
    (r"(?<![A-Za-z])(?:U\.S\.\s*dollars|US\s*dollars|USD)(?![A-Za-z])", "EUR"),
    (r"\$", "€"),
)


def _case_like(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_pair(text: str, left: str, right: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group()
        counterpart = right if token.lower() == left.lower() else left
        return _case_like(token, counterpart)

    return re.sub(rf"\b(?:{re.escape(left)}|{re.escape(right)})\b", replace, text, flags=re.I)


def _make_pair(idx: int, category: str, original: str, perturbed: str) -> dict[str, Any]:
    return {
        "pair_id": f"test-{idx}-{category}",
        "idx": int(idx),
        "category": category,
        "original_text": original,
        "perturbed_text": perturbed,
    }


def generate_counterfactual_pairs(
    examples: pd.DataFrame,
    *,
    text_column: str = "text",
    idx_column: str = "idx",
) -> list[dict[str, Any]]:
    """Generate one-factor perturbations only when the source text contains the target.

    Templates are transformations of actual supplied clauses, not standalone
    synthetic examples. Each output pair changes only its named role, pronoun,
    jurisdiction, or currency feature.
    """
    if not {text_column, idx_column}.issubset(examples.columns):
        raise ValueError(f"examples must contain {text_column!r} and {idx_column!r}")
    pairs = []
    for idx, original in examples[[idx_column, text_column]].itertuples(index=False, name=None):
        if not isinstance(original, str) or not original.strip():
            continue
        for category, left, right in _ROLE_PAIRS:
            if re.search(rf"\b{left}\b", original, flags=re.I) and re.search(
                rf"\b{right}\b", original, flags=re.I
            ):
                perturbed = _replace_pair(original, left, right)
                if perturbed != original:
                    pairs.append(_make_pair(idx, f"role_swap_{category}", original, perturbed))

        left, right = _PRONOUN_PAIR
        if re.search(rf"\b(?:{left}|{right})\b", original, flags=re.I):
            perturbed = _replace_pair(original, left, right)
            if perturbed != original:
                pairs.append(_make_pair(idx, "pronoun_he_she", original, perturbed))

        for source, target in _JURISDICTIONS:
            pattern = rf"\b{re.escape(source)}\b"
            if re.search(pattern, original, flags=re.I):
                perturbed = re.sub(
                    pattern,
                    lambda match: _case_like(match.group(), target),
                    original,
                    flags=re.I,
                )
                pairs.append(_make_pair(idx, f"jurisdiction_{source.lower().replace(' ', '_')}_to_{target.lower().replace(' ', '_')}", original, perturbed))

        for pattern, target in _CURRENCIES:
            if re.search(pattern, original, flags=re.I):
                perturbed = re.sub(
                    pattern,
                    lambda match: _case_like(match.group(), target),
                    original,
                    flags=re.I,
                )
                pairs.append(_make_pair(idx, "currency_usd_to_eur", original, perturbed))
                break

    return pairs


def _validate_prediction_outputs(
    predictor: Predictor,
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    outputs = list(predictor(texts))
    if len(outputs) != len(texts):
        raise ValueError("predictor must return one result per input text")
    try:
        labels = np.asarray([int(output["label"]) for output in outputs], dtype=int)
        confidences = np.asarray([float(output["confidence"]) for output in outputs], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each prediction must contain integer label and numeric confidence") from error
    if not np.isfinite(confidences).all() or np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("Prediction confidences must be finite probabilities in [0, 1]")
    return labels, confidences


def audit_counterfactuals(
    pairs: Sequence[dict[str, Any]],
    predictor: Predictor,
) -> pd.DataFrame:
    """Report prediction flip and top-label confidence change per generated pair."""
    if not pairs:
        return pd.DataFrame(columns=[
            "pair_id",
            "idx",
            "category",
            "original_label",
            "perturbed_label",
            "original_confidence",
            "perturbed_confidence",
            "confidence_delta",
            "prediction_flipped",
        ])
    original_texts = [str(pair["original_text"]) for pair in pairs]
    perturbed_texts = [str(pair["perturbed_text"]) for pair in pairs]
    original_labels, original_confidences = _validate_prediction_outputs(predictor, original_texts)
    perturbed_labels, perturbed_confidences = _validate_prediction_outputs(predictor, perturbed_texts)
    rows = []
    for position, pair in enumerate(pairs):
        rows.append({
            "pair_id": pair["pair_id"],
            "idx": int(pair["idx"]),
            "category": pair["category"],
            "original_label": int(original_labels[position]),
            "perturbed_label": int(perturbed_labels[position]),
            "original_confidence": float(original_confidences[position]),
            "perturbed_confidence": float(perturbed_confidences[position]),
            "confidence_delta": float(perturbed_confidences[position] - original_confidences[position]),
            "prediction_flipped": bool(original_labels[position] != perturbed_labels[position]),
        })
    return pd.DataFrame(rows)


def summarize_counterfactuals(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate flip rate, confidence change, and support by perturbation category."""
    if results.empty:
        return pd.DataFrame(columns=["category", "n_pairs", "flip_rate", "mean_confidence_delta"])
    return (
        results.groupby("category", as_index=False)
        .agg(
            n_pairs=("pair_id", "count"),
            flip_rate=("prediction_flipped", "mean"),
            mean_confidence_delta=("confidence_delta", "mean"),
        )
        .sort_values("category")
        .reset_index(drop=True)
    )


def audit_ood_examples(
    examples: pd.DataFrame,
    predictor: Predictor,
    *,
    text_column: str = "text",
) -> pd.DataFrame:
    """Return predictions/confidence for OOD examples without claiming accuracy."""
    if text_column not in examples or "id" not in examples:
        raise ValueError(f"examples must contain id and {text_column!r}")
    texts = examples[text_column].astype(str).tolist()
    labels, confidences = _validate_prediction_outputs(predictor, texts)
    output = examples.copy()
    output["predicted_label"] = labels
    output["predicted_confidence"] = confidences
    return output


def audit_retrieval_counterfactuals(
    pairs: Sequence[dict[str, Any]],
    retriever: Any,
    *,
    k: int = 5,
) -> pd.DataFrame:
    """Measure top-k retrieval overlap under the same text perturbations."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    columns = [
        "pair_id",
        "idx",
        "category",
        "top1_same",
        "topk_jaccard",
        "original_top1_rank_after_perturbation",
    ]
    if not pairs:
        return pd.DataFrame(columns=columns)
    original_texts = [str(pair["original_text"]) for pair in pairs]
    perturbed_texts = [str(pair["perturbed_text"]) for pair in pairs]
    if hasattr(retriever, "search_many"):
        original_rankings = retriever.search_many(original_texts, k=k)
        perturbed_rankings = retriever.search_many(perturbed_texts, k=k)
    else:
        original_rankings = [retriever.search(text, k=k) for text in original_texts]
        perturbed_rankings = [retriever.search(text, k=k) for text in perturbed_texts]

    rows = []
    for pair, original, perturbed in zip(pairs, original_rankings, perturbed_rankings):
        original_ids = [int(result["idx"]) for result in original]
        perturbed_ids = [int(result["idx"]) for result in perturbed]
        original_set, perturbed_set = set(original_ids), set(perturbed_ids)
        union = original_set | perturbed_set
        top1 = original_ids[0] if original_ids else None
        rows.append({
            "pair_id": pair["pair_id"],
            "idx": int(pair["idx"]),
            "category": pair["category"],
            "top1_same": bool(top1 is not None and perturbed_ids and top1 == perturbed_ids[0]),
            "topk_jaccard": len(original_set & perturbed_set) / len(union) if union else 1.0,
            "original_top1_rank_after_perturbation": (
                perturbed_ids.index(top1) + 1 if top1 in perturbed_ids else None
            ),
        })
    return pd.DataFrame(rows, columns=columns)
