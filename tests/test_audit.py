"""Unit tests for the Stream C classification and robustness audit helpers."""

import numpy as np
import pandas as pd
import pytest

from src.audit.perturb import (
    audit_counterfactuals,
    audit_ood_examples,
    audit_retrieval_counterfactuals,
    generate_counterfactual_pairs,
    summarize_counterfactuals,
)
from src.audit.slices import build_audit_frame, slice_performance


def test_build_audit_frame_creates_frequency_length_and_party_slices():
    train = pd.DataFrame({
        "idx": range(12),
        "text": [f"training clause {idx}" for idx in range(12)],
        "label": [0] * 6 + [1] * 4 + [2] * 2,
    })
    evaluation = pd.DataFrame({
        "idx": [0, 1, 2, 3],
        "text": [
            "The Borrower shall pay the Lender.",
            "The Employee shall notify the Employer.",
            "The Tenant shall pay the Landlord.",
            "An ordinary short clause.",
        ],
        "label": [0, 1, 2, 2],
    })
    predictions = pd.DataFrame({
        "idx": [0, 1, 2, 3],
        "label": [0, 1, 2, 2],
        "pred": [0, 2, 2, 1],
        "proba_pred": [0.9, 0.6, 0.8, 0.7],
    })

    audit = build_audit_frame(evaluation, predictions, train)
    summaries = slice_performance(audit)

    assert list(audit["idx"]) == [0, 1, 2, 3]
    assert set(audit["frequency_group"]) <= {"head", "middle", "tail"}
    assert audit.loc[0, "borrower_lender"]
    assert audit.loc[1, "employer_employee"]
    assert audit.loc[2, "landlord_tenant"]
    assert set(summaries["slice"]) >= {
        "frequency_group",
        "length_group",
        "party_type",
        "borrower_lender",
        "employer_employee",
        "landlord_tenant",
    }
    assert summaries["support"].sum() >= len(evaluation)


def test_build_audit_frame_rejects_incomplete_predictions():
    evaluation = pd.DataFrame({"idx": [0, 1], "text": ["a", "b"], "label": [0, 1]})
    train = pd.DataFrame({"idx": [0, 1], "text": ["a", "b"], "label": [0, 1]})
    predictions = pd.DataFrame({
        "idx": [0],
        "label": [0],
        "pred": [0],
        "proba_pred": [0.9],
    })
    with pytest.raises(ValueError, match="same idx values"):
        build_audit_frame(evaluation, predictions, train)


def test_generate_counterfactual_pairs_changes_one_targeted_feature():
    examples = pd.DataFrame({
        "idx": [4, 5, 6, 7, 8],
        "text": [
            "The Borrower shall pay the Lender in USD.",
            "He shall provide notice in New York.",
            "The contract is governed by New York law.",
            "The amount is $100.",
            "A clause with none of the targeted phrases.",
        ],
    })

    pairs = generate_counterfactual_pairs(examples)
    by_category = {pair["category"]: pair for pair in pairs}

    assert by_category["role_swap_borrower_lender"]["perturbed_text"] == (
        "The Lender shall pay the Borrower in USD."
    )
    assert by_category["pronoun_he_she"]["perturbed_text"] == (
        "She shall provide notice in New York."
    )
    assert by_category["jurisdiction_new_york_to_luxembourg"]["perturbed_text"] == (
        "The contract is governed by Luxembourg law."
    )
    assert by_category["currency_usd_to_eur"]["perturbed_text"] == (
        "The amount is €100."
    )
    assert all(pair["original_text"] != pair["perturbed_text"] for pair in pairs)


def test_counterfactual_prediction_metrics_and_summary():
    pairs = [
        {
            "pair_id": "p1",
            "idx": 2,
            "category": "jurisdiction",
            "original_text": "original",
            "perturbed_text": "changed",
        }
    ]

    def predictor(texts):
        return [
            {"label": 1, "confidence": 0.8}
            if text == "original"
            else {"label": 2, "confidence": 0.6}
            for text in texts
        ]

    results = audit_counterfactuals(pairs, predictor)
    summary = summarize_counterfactuals(results)

    assert results.loc[0, "prediction_flipped"]
    assert results.loc[0, "confidence_delta"] == pytest.approx(-0.2)
    assert summary.loc[0, "flip_rate"] == pytest.approx(1.0)


def test_ood_predictions_do_not_claim_accuracy():
    examples = pd.DataFrame({
        "id": ["en", "fr"],
        "language": ["en", "fr"],
        "text": ["English clause", "Clause française"],
    })
    results = audit_ood_examples(
        examples,
        lambda texts: [
            {"label": 3, "confidence": 0.7},
            {"label": 8, "confidence": 0.4},
        ],
    )
    assert results["predicted_label"].tolist() == [3, 8]
    assert results["predicted_confidence"].tolist() == pytest.approx([0.7, 0.4])
    assert "accuracy" not in results.columns


def test_retrieval_robustness_reports_overlap_and_top1_rank():
    class FakeRetriever:
        def search_many(self, queries, k):
            if queries[0] == "original":
                return [[{"idx": 1}, {"idx": 2}, {"idx": 3}]]
            return [[{"idx": 2}, {"idx": 1}, {"idx": 4}]]

    pairs = [{
        "pair_id": "p1",
        "idx": 9,
        "category": "gender",
        "original_text": "original",
        "perturbed_text": "perturbed",
    }]
    results = audit_retrieval_counterfactuals(pairs, FakeRetriever(), k=3)
    assert not results.loc[0, "top1_same"]
    assert results.loc[0, "topk_jaccard"] == pytest.approx(0.5)
    assert results.loc[0, "original_top1_rank_after_perturbation"] == 2
