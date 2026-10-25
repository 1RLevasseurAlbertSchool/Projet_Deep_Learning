"""Tests of the error-analysis helpers (tasks B7 / B9) — synthetic labels only."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import error_rate_by_length, per_class_comparison, sample_errors, top_confusions
from src.data import frequency_groups

NAMES = ["a", "b", "c", "d"]


def test_per_class_comparison_delta_is_a_minus_b():
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pred_a = labels.copy()
    pred_b = np.array([0, 1, 1, 1, 2, 2, 3, 3])
    df = per_class_comparison(labels, pred_a, pred_b, frequency_groups([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], 4), NAMES,
                              num_labels=4)
    assert df["f1_bert"].tolist() == [1.0] * 4
    assert df.loc[0, "delta_f1"] == pytest.approx(1 - 2 / 3)
    assert df.loc[2, "delta_f1"] == 0.0
    assert df["test_support"].tolist() == [2, 2, 2, 2]
    assert df["train_count"].tolist() == [1, 2, 3, 4]


def test_top_confusions_counts_and_other_model_overlap():
    labels = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([1, 1, 0, 2, 1, 2])
    other = np.array([1, 0, 0, 2, 2, 2])
    pairs = top_confusions(labels, pred, NAMES, other_pred=other, k=5, num_labels=4)
    assert list(zip(pairs["true"], pairs["pred"], pairs["count"])) == [(0, 1, 2), (1, 2, 1)]
    assert pairs.loc[0, "share_of_true"] == pytest.approx(2 / 3)
    assert pairs["other_same_clauses"].tolist() == [1, 1]
    assert pairs["other_pair_count"].tolist() == [1, 2]


def test_error_rate_by_length_splits_on_max_length():
    labels = np.array([0, 0, 0, 0])
    out = error_rate_by_length(labels, {"m": np.array([0, 1, 1, 1])}, n_tokens=[10, 256, 257, 900], max_length=256)
    assert out["n"].tolist() == [2, 2]
    assert out["error_rate"].tolist() == [0.5, 1.0]


def test_sample_errors_is_stratified_and_reproducible():
    rng = np.random.default_rng(0)
    preds = pd.DataFrame({"idx": range(200), "label": 0, "pred": rng.integers(0, 2, 200), "proba_pred": rng.random(200)})
    first, second = sample_errors(preds, 0.5, n_per_group=5), sample_errors(preds, 0.5, n_per_group=5)
    pd.testing.assert_frame_equal(first, second)
    assert (first["pred"] != first["label"]).all()
    assert first["confidence_group"].value_counts().to_dict() == {"high": 5, "low": 5}
    assert (first.loc[first["confidence_group"] == "high", "proba_pred"] >= 0.5).all()
    with pytest.raises(ValueError):
        sample_errors(preds, 0.5, n_per_group=500)
