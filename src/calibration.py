"""Calibration of the classifier's confidence and human-review threshold (stream A, task A9).

Top-label calibration: the confidence of a prediction is the probability of the predicted class (`proba_pred` in the
common prediction format), compared with whether that prediction is correct. A threshold splits the clauses into those
handled automatically (confidence >= threshold) and those sent to a lawyer; its precision is the accuracy on the
clauses handled automatically. The threshold is chosen on the validation set only, then applied as is to the test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

THRESHOLD_GRID = np.round(np.linspace(0.0, 1.0, 1001), 3)  # candidate thresholds, step 0.001

# Light-mode chart colours (validated categorical slots 1-2, neutral ink / gridline / baseline)
SERIES_COLORS = ["#2a78d6", "#eb6834"]
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _as_arrays(confidences, correct) -> tuple[np.ndarray, np.ndarray]:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidences.ndim != 1 or confidences.shape != correct.shape:
        raise ValueError(f"confidences {confidences.shape} and correct {correct.shape} must be 1-D of equal length")
    if ((confidences < 0) | (confidences > 1)).any():
        raise ValueError("confidences must lie in [0, 1]")
    return confidences, correct


def calibration_bins(confidences, correct, n_bins: int = 15) -> pd.DataFrame:
    """Equal-width confidence bins (low, high] over [0, 1] (the first one also holds 0).

    One row per bin: count, mean confidence and accuracy (NaN for an empty bin).
    """
    confidences, correct = _as_arrays(confidences, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, confidences, side="left") - 1, 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({
            "bin_low": edges[:-1],
            "bin_high": edges[1:],
            "count": counts,
            "mean_confidence": np.bincount(bins, weights=confidences, minlength=n_bins) / counts,
            "accuracy": np.bincount(bins, weights=correct, minlength=n_bins) / counts,
        })


def expected_calibration_error(confidences, correct, n_bins: int = 15) -> float:
    """ECE: sum over the bins of (share of examples in the bin) x |accuracy - mean confidence| (Guo et al., 2017)."""
    bins = calibration_bins(confidences, correct, n_bins).query("count > 0")
    gaps = (bins["accuracy"] - bins["mean_confidence"]).abs()
    return float((bins["count"] * gaps).sum() / bins["count"].sum())


def coverage_precision_curve(confidences, correct, thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    """For each threshold, the clauses handled automatically are those with confidence >= threshold.

    Columns: `threshold`, `n_auto`, `n_auto_correct`, `coverage` (share handled automatically), `precision` (accuracy
    on them, NaN when none), `share_of_errors_reviewed` (share of the wrong predictions sent to human review, i.e.
    below the threshold).
    """
    confidences, correct = _as_arrays(confidences, correct)
    thresholds = np.asarray(thresholds, dtype=float)
    auto = confidences[None, :] >= thresholds[:, None]
    n_auto, n_auto_correct = auto.sum(axis=1), (auto & correct).sum(axis=1)
    n_errors = int((~correct).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({
            "threshold": thresholds,
            "n_auto": n_auto,
            "n_auto_correct": n_auto_correct,
            "coverage": n_auto / len(confidences),
            "precision": n_auto_correct / n_auto,
            "share_of_errors_reviewed": (~auto & ~correct).sum(axis=1) / n_errors if n_errors else np.nan,
        })


def select_threshold(confidences, correct, target_precision: float, thresholds=THRESHOLD_GRID) -> float | None:
    """Smallest threshold whose precision reaches `target_precision` (largest coverage); None if none does.

    To be called on the validation set only.
    """
    curve = coverage_precision_curve(confidences, correct, thresholds)
    reached = curve.loc[curve["precision"] >= target_precision, "threshold"]
    return None if reached.empty else float(reached.min())


def operating_point(confidences, correct, threshold: float) -> dict:
    """Coverage and precision at a fixed threshold, with the 95 % Wilson interval of the precision."""
    row = coverage_precision_curve(confidences, correct, [threshold]).iloc[0]
    n_auto = int(row["n_auto"])
    ci = stats.binomtest(int(row["n_auto_correct"]), n_auto).proportion_ci(method="wilson") if n_auto else None
    return {
        "n_auto": n_auto,
        "n_reviewed": len(np.asarray(confidences)) - n_auto,
        "coverage": float(row["coverage"]),
        "precision": float(row["precision"]) if n_auto else None,
        "precision_ci95": [float(ci.low), float(ci.high)] if n_auto else None,
        "share_of_errors_reviewed": float(row["share_of_errors_reviewed"]),
    }


def calibration_summary(confidences, correct, n_bins: int = 15) -> dict:
    """Accuracy, mean confidence, their gap (> 0 = over-confident on average), ECE and bins of one split."""
    confidences, correct = _as_arrays(confidences, correct)
    bins = calibration_bins(confidences, correct, n_bins)
    return {
        "n": len(confidences),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidences.mean()),
        "confidence_minus_accuracy": float(confidences.mean() - correct.mean()),
        "ece": expected_calibration_error(confidences, correct, n_bins),
        "bins": bins.astype(object).where(bins.notna(), None).to_dict("records"),  # empty bin: None, not NaN
    }


def calibration_report(validation: pd.DataFrame, test: pd.DataFrame, target_precisions, n_bins: int = 15) -> dict:
    """Calibration of both splits, and one operating point per target precision.

    `validation` / `test`: prediction files in the common format (`label`, `pred`, `proba_pred`). Each threshold is
    chosen on validation; the test coverage and precision are read at that fixed threshold.
    """
    splits = {"validation": validation, "test": test}
    data = {name: (df["proba_pred"].to_numpy(), (df["pred"] == df["label"]).to_numpy()) for name, df in splits.items()}
    points = []
    for target in sorted(target_precisions):
        threshold = select_threshold(*data["validation"], target)
        points.append({
            "target_precision": target,
            "threshold": threshold,
            **{name: None if threshold is None else operating_point(*arrays, threshold)
               for name, arrays in data.items()},
        })
    return {
        "n_bins": n_bins,
        "threshold_grid_step": float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        **{name: calibration_summary(*arrays, n_bins) for name, arrays in data.items()},
        "operating_points": points,
    }


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)


def reliability_diagram(splits: dict, path, n_bins: int = 15, title: str | None = None) -> None:
    """Reliability diagram (accuracy vs mean confidence per bin) and clauses per bin (log scale), saved as a PNG.

    `splits`: {name: (confidences, correct)}, one series per split (at most two colours defined).
    """
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, height_ratios=[3, 1], facecolor=SURFACE)
    top.plot([0, 1], [0, 1], linestyle="--", color=BASELINE, linewidth=1.2, label="perfect calibration")
    width = 1 / n_bins / (len(splits) + 1)
    for i, (name, (confidences, correct)) in enumerate(splits.items()):
        bins = calibration_bins(confidences, correct, n_bins)
        filled = bins[bins["count"] > 0]
        ece = expected_calibration_error(confidences, correct, n_bins)
        top.plot(filled["mean_confidence"], filled["accuracy"], marker="o", markersize=5, linewidth=2,
                 color=SERIES_COLORS[i], label=f"{name} (ECE {ece:.3f})")
        centres = bins["bin_low"] + 1 / n_bins / 2 + (i - (len(splits) - 1) / 2) * width
        bottom.bar(centres, bins["count"], width=width * 0.9, color=SERIES_COLORS[i], label=name)
    top.set(xlim=(0, 1), ylim=(0, 1))
    top.set_ylabel("accuracy in the bin", color=INK_SECONDARY)
    top.legend(frameon=False, labelcolor=INK, loc="upper left")
    bottom.set_yscale("log")  # the low-confidence bins hold few clauses: their accuracy is noisy
    bottom.set_ylabel("clauses (log)", color=INK_SECONDARY)
    bottom.set_xlabel("confidence = probability of the predicted class (bin)", color=INK_SECONDARY)
    for ax in (top, bottom):
        _style_axes(ax)
    if title:
        top.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def coverage_precision_figure(splits: dict, operating_points: list[dict], path, title: str | None = None) -> None:
    """Precision vs coverage over the threshold grid, one curve per split, with the selected thresholds marked.

    `splits`: {name: (confidences, correct)}; `operating_points`: `calibration_report(...)["operating_points"]`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    low = 1.0
    for i, (name, (confidences, correct)) in enumerate(splits.items()):
        curve = coverage_precision_curve(confidences, correct).dropna(subset=["precision"])
        ax.plot(curve["coverage"], curve["precision"], linewidth=2, color=SERIES_COLORS[i], label=name)
        low = min(low, curve["precision"].min())
        for point in operating_points:
            if point["threshold"] is None:
                continue
            x, y = point[name]["coverage"], point[name]["precision"]
            ax.plot(x, y, marker="o", markersize=7, color=SERIES_COLORS[i], markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3)
            if name == "test":
                ax.annotate(f"threshold {point['threshold']:.3f}\n(target {point['target_precision']:.2f} on val.)",
                            (x, y), xytext=(-12, -6), textcoords="offset points", ha="right", va="top",
                            fontsize=8, color=INK_SECONDARY)
    ax.set(xlim=(0, 1.02), ylim=(np.floor(low * 50) / 50, 1.005))
    ax.set_xlabel("coverage = share of clauses handled automatically (confidence ≥ threshold)", color=INK_SECONDARY)
    ax.set_ylabel("precision = accuracy on those clauses", color=INK_SECONDARY)
    ax.legend(frameon=False, labelcolor=INK, loc="lower left")
    _style_axes(ax)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
