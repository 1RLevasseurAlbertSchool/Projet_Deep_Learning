"""Tests of the paired comparison of two runs (stream A, task A8) — synthetic prediction files only."""

import numpy as np
import pandas as pd
import pytest

import src.comparison
from src.comparison import compare_runs, paired_comparison, update_results_csv


def _labels_and_preds(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, n)
    return labels, np.where(rng.random(n) < 0.8, labels, (labels + 1) % 5)


def _write_predictions(directory, run_id, labels, preds, idx=None):
    pd.DataFrame({
        "idx": np.arange(len(labels)) if idx is None else idx,
        "label": labels,
        "pred": preds,
    }).to_parquet(directory / f"{run_id}_test.parquet", index=False)


def test_identical_models_give_zero_delta_and_p_value_one():
    labels, preds = _labels_and_preds()
    r = paired_comparison(labels, preds, preds.copy(), n_boot=50, num_labels=5)
    assert r["delta_macro_f1"] == 0.0
    assert r["ci95_low"] == r["ci95_high"] == 0.0
    assert not r["significant"]
    assert (r["b"], r["c"]) == (0, 0)
    assert r["mcnemar_p_value"] == 1.0
    assert r["macro_f1_a"] == r["macro_f1_b"] and r["accuracy_a"] == r["accuracy_b"]


def test_misaligned_idx_or_label_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(src.comparison, "PREDICTIONS_DIR", tmp_path)
    labels, preds = _labels_and_preds()
    _write_predictions(tmp_path, "a", labels, preds)
    _write_predictions(tmp_path, "shuffled", labels, preds, idx=np.roll(np.arange(len(labels)), 1))
    _write_predictions(tmp_path, "relabelled", (labels + 1) % 5, preds)
    with pytest.raises(ValueError, match="idx"):
        compare_runs("a", "shuffled")
    with pytest.raises(ValueError, match="label"):
        compare_runs("a", "relabelled")
    with pytest.raises(FileNotFoundError, match="absent"):
        compare_runs("a", "absent")


def test_deduplicated_variant_and_csv_update(tmp_path, monkeypatch):
    monkeypatch.setattr(src.comparison, "PREDICTIONS_DIR", tmp_path)
    labels, preds = _labels_and_preds()
    _, other = _labels_and_preds(seed=1)
    _write_predictions(tmp_path, "a", labels, preds)
    _write_predictions(tmp_path, "b", labels, other)
    (tmp_path / "dups.txt").write_text("0\n1\n2\n")

    rows = compare_runs("a", "b", exclude_idx_file=tmp_path / "dups.txt", n_boot=20, num_labels=5)
    assert [(row["variant"], row["n"]) for row in rows] == [("all", 500), ("deduplicated", 497)]

    csv = tmp_path / "paired_tests.csv"
    update_results_csv([{**rows[0], "model_b": "c"}], csv)  # another pair: kept
    update_results_csv(rows, csv)
    df = update_results_csv(rows, csv)  # same key again: replaced, not duplicated
    assert len(df) == 3
    assert sorted(df["model_b"]) == ["b", "b", "c"]

    (tmp_path / "wrong_split.txt").write_text("0\n9999\n")
    with pytest.raises(ValueError, match="wrong split"):
        compare_runs("a", "b", exclude_idx_file=tmp_path / "wrong_split.txt", n_boot=20, num_labels=5)
