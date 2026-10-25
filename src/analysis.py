"""Error analysis of the final classifier against the TF-IDF baseline (tasks B7 / B9, notebook 03).

Usage:
    from src.analysis import per_class_comparison, top_confusions, error_rate_by_length, sample_errors
    per_class = per_class_comparison(labels, pred_bert, pred_tfidf, groups, label_names)
    pairs = top_confusions(labels, pred_bert, label_names, other_pred=pred_tfidf, k=10)
    by_length = error_rate_by_length(labels, {"legal-bert": pred_bert, "tfidf": pred_tfidf}, n_tokens, 256)
    errors = sample_errors(preds_bert, threshold=0.892, n_per_group=10)

All inputs are aligned arrays over the same split (common prediction format, `idx` = position in the split).
Nothing here selects or tunes anything: these are descriptive views of a single test evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import LABEL_COL, NUM_LABELS, SEED
from src.data import FREQUENCY_GROUPS
from src.metrics import classification_metrics

# Categorical slots 1-3 of the reference data-viz palette (validated all-pairs for 3 series), one per frequency group
GROUP_COLORS = {"head": "#2a78d6", "middle": "#eb6834", "tail": "#1baf7a"}
GROUP_MARKERS = {"head": "o", "middle": "s", "tail": "^"}


def per_class_comparison(labels, pred_a, pred_b, groups: pd.DataFrame, label_names: Sequence[str],
                         name_a: str = "bert", name_b: str = "tfidf", num_labels: int = NUM_LABELS) -> pd.DataFrame:
    """Per-class F1 of two models on the same clauses, with training count and frequency group.

    `groups` is `src.data.frequency_groups(train_labels)`. `delta_f1` = F1(A) − F1(B), positive when A is better.
    """
    labels = np.asarray(labels)
    f1_a = classification_metrics(labels, pred_a, num_labels=num_labels)["per_class_f1"]
    f1_b = classification_metrics(labels, pred_b, num_labels=num_labels)["per_class_f1"]
    df = groups.copy()
    df.insert(1, "label_name", [label_names[i] for i in df[LABEL_COL]])
    df["test_support"] = np.bincount(labels, minlength=num_labels)
    df[f"f1_{name_a}"] = f1_a
    df[f"f1_{name_b}"] = f1_b
    df["delta_f1"] = df[f"f1_{name_a}"] - df[f"f1_{name_b}"]
    return df


def top_confusions(labels, pred, label_names: Sequence[str], other_pred=None, k: int = 10,
                   num_labels: int = NUM_LABELS) -> pd.DataFrame:
    """The `k` most frequent (true → predicted) errors of `pred`, ties broken by label ids.

    `share_of_true` = count / test support of the true class. With `other_pred` (e.g. TF-IDF), two views of whether
    the other model makes the same confusion: `other_same_clauses` = how many of these exact clauses it also assigns
    to the same wrong class, `other_pair_count` = its own count for the pair over the whole split.
    """
    labels, pred = np.asarray(labels), np.asarray(pred)
    ids = np.arange(num_labels)
    cm = confusion_matrix(labels, pred, labels=ids)
    np.fill_diagonal(cm, 0)
    support = np.bincount(labels, minlength=num_labels)
    true_ids, pred_ids = np.nonzero(cm)
    pairs = pd.DataFrame({"true": true_ids, "pred": pred_ids, "count": cm[true_ids, pred_ids]})
    pairs = pairs.sort_values(["count", "true", "pred"], ascending=[False, True, True]).head(k).reset_index(drop=True)
    pairs.insert(2, "true_name", [label_names[i] for i in pairs["true"]])
    pairs.insert(3, "pred_name", [label_names[i] for i in pairs["pred"]])
    pairs["share_of_true"] = pairs["count"] / support[pairs["true"]]
    if other_pred is not None:
        other_pred = np.asarray(other_pred)
        other_cm = confusion_matrix(labels, other_pred, labels=ids)
        pairs["other_same_clauses"] = [int(np.sum((labels == t) & (pred == p) & (other_pred == p)))
                                       for t, p in zip(pairs["true"], pairs["pred"])]
        pairs["other_pair_count"] = other_cm[pairs["true"], pairs["pred"]]
    return pairs


def error_rate_by_length(labels, preds: Mapping[str, np.ndarray], n_tokens, max_length: int) -> pd.DataFrame:
    """Error rate of each model on clauses with ≤ `max_length` tokens vs > `max_length` (i.e. truncated).

    A model that never truncates (TF-IDF) is a useful control: if it also errs more on long clauses, the gap is
    partly length / topic difficulty, not truncation alone.
    """
    labels, n_tokens = np.asarray(labels), np.asarray(n_tokens)
    long_ = n_tokens > max_length
    rows = []
    for model, pred in preds.items():
        wrong = np.asarray(pred) != labels
        for group, mask in ((f"<= {max_length} tokens", ~long_), (f"> {max_length} tokens", long_)):
            rows.append({"model": model, "length_group": group, "n": int(mask.sum()),
                         "n_errors": int(wrong[mask].sum()), "error_rate": float(wrong[mask].mean())})
    return pd.DataFrame(rows)


def sample_errors(preds: pd.DataFrame, threshold: float, n_per_group: int = 10, seed: int = SEED) -> pd.DataFrame:
    """Random errors of a model, stratified by confidence: `n_per_group` with `proba_pred` ≥ `threshold` ("high")
    and `n_per_group` below it ("low"), reproducible with `seed`. Sorted by group, then confidence."""
    errors = preds[preds["pred"] != preds["label"]].copy()
    errors["confidence_group"] = np.where(errors["proba_pred"] >= threshold, "high", "low")
    rng = np.random.default_rng(seed)
    picked = []
    for group in ("high", "low"):
        pool = errors[errors["confidence_group"] == group]
        if len(pool) < n_per_group:
            raise ValueError(f"only {len(pool)} {group}-confidence errors, {n_per_group} requested")
        picked.append(pool.iloc[np.sort(rng.choice(len(pool), n_per_group, replace=False))])
    return (pd.concat(picked).sort_values(["confidence_group", "proba_pred"], ascending=[True, False])
            .reset_index(drop=True))


def plot_delta_vs_frequency(per_class: pd.DataFrame, ax=None, n_labels: int = 3):
    """Scatter of per-class ΔF1 against training count (log x), one colour + marker per frequency group.

    The `n_labels` largest gains and losses are labelled by class name.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(0, color="#52514e", linewidth=1)
    for group in reversed(FREQUENCY_GROUPS):
        sub = per_class[per_class["frequency_group"] == group]
        ax.scatter(sub["train_count"], sub["delta_f1"], s=36, color=GROUP_COLORS[group], marker=GROUP_MARKERS[group],
                   edgecolor="white", linewidth=0.8, label=f"{group} ({len(sub)} classes, median ΔF1 "
                   f"{sub['delta_f1'].median():+.3f})", zorder=3)
    # Gains labelled on the right of their point, losses on the left, so neighbouring labels do not collide
    for extremes, dx, ha in ((per_class.nlargest(n_labels, "delta_f1"), 5, "left"),
                             (per_class.nsmallest(n_labels, "delta_f1"), -5, "right")):
        for row in extremes.itertuples():
            ax.annotate(row.label_name, (row.train_count, row.delta_f1), xytext=(dx, 0), textcoords="offset points",
                        fontsize=8, color="#52514e", va="center", ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel("Training clauses of the class (log scale)")
    ax.set_ylabel("ΔF1 per class (Legal-BERT − TF-IDF), test")
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    return ax


def plot_confusion_subset(labels, pred, classes: Sequence[int], label_names: Sequence[str], ax=None,
                          num_labels: int = NUM_LABELS):
    """Confusion matrix restricted to `classes` (rows = true, columns = predicted), counts annotated.

    Colour = share of the row's test clauses (row-normalised over the whole split, so rows do not sum to 1 when
    errors go to classes outside the subset).
    """
    import matplotlib.pyplot as plt

    labels, pred = np.asarray(labels), np.asarray(pred)
    classes = list(classes)
    cm = confusion_matrix(labels, pred, labels=np.arange(num_labels))
    support = cm.sum(axis=1)
    sub = cm[np.ix_(classes, classes)]
    share = sub / np.maximum(support[classes], 1)[:, None]
    if ax is None:
        size = 0.55 * len(classes) + 3
        _, ax = plt.subplots(figsize=(size + 1.5, size))
    image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if sub[i, j]:
                ax.text(j, i, sub[i, j], ha="center", va="center", fontsize=7,
                        color="white" if share[i, j] > 0.5 else "#0b0b0b")
    names = [label_names[c] for c in classes]
    ax.set_xticks(range(len(classes)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.figure.colorbar(image, ax=ax, fraction=0.04, label="Share of the true class's test clauses")
    return ax
