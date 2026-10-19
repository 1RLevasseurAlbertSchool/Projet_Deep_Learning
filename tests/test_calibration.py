"""Tests of the calibration and human-review threshold helpers (stream A, task A9) — synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    calibration_bins,
    calibration_report,
    coverage_precision_curve,
    coverage_precision_figure,
    expected_calibration_error,
    operating_point,
    per_class_operating_point,
    per_class_precision_figure,
    reliability_diagram,
    select_threshold,
)


def calibrated_sample(n: int = 200_000, seed: int = 0):
    """Confidences uniform on [0, 1], each prediction correct with probability equal to its confidence."""
    rng = np.random.default_rng(seed)
    confidences = rng.uniform(0, 1, n)
    return confidences, rng.uniform(0, 1, n) < confidences


def test_ece_close_to_zero_on_perfectly_calibrated_data():
    assert expected_calibration_error(*calibrated_sample()) < 0.01


def test_ece_exact_values():
    # one bin, accuracy = confidence -> 0; over-confident by 0.3 everywhere -> 0.3
    assert expected_calibration_error(np.full(10, 0.8), np.arange(10) < 8) == pytest.approx(0.0)
    assert expected_calibration_error(np.full(10, 0.9), np.arange(10) < 6) == pytest.approx(0.3)


def test_bins_are_right_closed_and_cover_0_and_1():
    bins = calibration_bins([0.0, 0.5, 0.5001, 1.0], [True, True, False, False], n_bins=2)
    assert bins["count"].tolist() == [2, 2]  # 0 and 0.5 in (0, 0.5]; 0.5001 and 1 in (0.5, 1]
    assert bins["accuracy"].tolist() == [1.0, 0.0]


def test_threshold_zero_covers_everything():
    confidences, correct = calibrated_sample(1_000)
    curve = coverage_precision_curve(confidences, correct, [0.0, 0.5, 1.01])
    assert curve["coverage"].iloc[0] == 1.0
    assert curve["precision"].iloc[0] == pytest.approx(correct.mean())
    assert curve["share_of_errors_reviewed"].iloc[0] == 0.0
    assert np.isnan(curve["precision"].iloc[2])  # nothing above 1.01
    assert curve["share_of_errors_reviewed"].iloc[2] == 1.0


def test_select_threshold_is_smallest_reaching_target():
    confidences = np.array([0.2, 0.4, 0.6, 0.7, 0.8, 0.9])
    correct = np.array([False, False, True, False, True, True])
    # precision at t: <=0.2 -> 3/6, (0.2, 0.4] -> 3/5, (0.4, 0.6] -> 3/4, (0.6, 0.7] -> 2/3, (0.7, 0.8] -> 1
    assert select_threshold(confidences, correct, 0.75) == pytest.approx(0.401)
    assert select_threshold(confidences, correct, 1.0) == pytest.approx(0.701)
    assert select_threshold(confidences, correct, 0.5) == 0.0
    assert select_threshold(np.full(4, 0.9), np.zeros(4, bool), 0.5) is None


def test_operating_point_counts_and_interval():
    point = operating_point([0.3, 0.6, 0.9, 0.95], [False, False, True, True], 0.6)
    assert point["n_auto"] == 3 and point["n_reviewed"] == 1
    assert point["precision"] == pytest.approx(2 / 3)
    assert point["precision_ci95"][0] < 2 / 3 < point["precision_ci95"][1]
    assert point["share_of_errors_reviewed"] == pytest.approx(0.5)


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 1.2], [True, False])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 0.6], [True])


def _predictions(confidences, correct):
    labels = np.zeros(len(confidences), dtype=int)
    return pd.DataFrame({"label": labels, "pred": np.where(correct, 0, 1), "proba_pred": confidences})


def test_report_chooses_threshold_on_validation_and_reads_test_at_it(tmp_path):
    val = _predictions(*calibrated_sample(5_000, seed=1))
    test = _predictions(*calibrated_sample(5_000, seed=2))
    report = calibration_report(val, test, [0.95, 0.90])
    assert [p["target_precision"] for p in report["operating_points"]] == [0.90, 0.95]
    for point in report["operating_points"]:
        assert point["validation"]["precision"] >= point["target_precision"]
        expected = operating_point(test["proba_pred"], test["pred"] == test["label"], point["threshold"])
        assert point["test"] == expected
    assert report["validation"]["ece"] < 0.03
    assert sum(b["count"] for b in report["test"]["bins"]) == len(test)

    arrays = {name: (df["proba_pred"], df["pred"] == df["label"])
              for name, df in {"validation": val, "test": test}.items()}
    reliability_diagram(arrays, tmp_path / "reliability.png")
    coverage_precision_figure(arrays, report["operating_points"], tmp_path / "coverage.png")
    assert (tmp_path / "reliability.png").stat().st_size > 0
    assert (tmp_path / "coverage.png").stat().st_size > 0


def test_per_class_counts_by_true_and_predicted_class():
    labels = [0, 0, 1, 1, 2, 2]
    preds = [0, 1, 1, 1, 1, 2]
    confidences = [0.9, 0.95, 0.8, 0.3, 0.99, 0.5]  # automatic at 0.7: clauses 0, 1, 2 and 4
    table = per_class_operating_point(labels, preds, confidences, 0.7, num_labels=4, min_auto=2)
    assert table["support"].tolist() == [2, 2, 2, 0]
    assert table["n_auto"].tolist() == [2, 1, 1, 0]
    assert table["share_auto"].tolist()[:3] == [1.0, 0.5, 0.5] and np.isnan(table["share_auto"].iloc[3])
    assert table["n_auto_pred"].tolist() == [1, 3, 0, 0]  # clauses 1, 2 and 4 labelled 1 automatically
    assert table["n_auto_correct"].tolist() == [1, 1, 0, 0]
    assert table["precision_auto"].iloc[:2].tolist() == pytest.approx([1.0, 1 / 3])
    assert table["precision_auto"].iloc[2:].isna().all() and table["precision_ci95_low"].iloc[2:].isna().all()
    assert table["precision_ci95_low"].iloc[1] < 1 / 3 < table["precision_ci95_high"].iloc[1]
    assert table["reliable"].tolist() == [False, True, False, False]


def test_per_class_counts_add_up_to_global_operating_point(tmp_path):
    rng = np.random.default_rng(0)
    labels, confidences = rng.integers(0, 10, 2_000), rng.uniform(0, 1, 2_000)
    preds = np.where(rng.uniform(0, 1, 2_000) < confidences, labels, rng.integers(0, 10, 2_000))
    table = per_class_operating_point(labels, preds, confidences, 0.6, num_labels=10)
    point = operating_point(confidences, labels == preds, 0.6)
    assert table["n_auto"].sum() == table["n_auto_pred"].sum() == point["n_auto"]
    assert table["n_auto_correct"].sum() / table["n_auto_pred"].sum() == pytest.approx(point["precision"])

    table = table.assign(train_count=np.arange(10) * 50 + 20, label_name=[f"class {i}" for i in range(10)],
                         frequency_group=pd.qcut(np.arange(10), 3, labels=["tail", "middle", "head"]))
    per_class_precision_figure(table, tmp_path / "per_class.png", overall_precision=point["precision"])
    assert (tmp_path / "per_class.png").stat().st_size > 0
