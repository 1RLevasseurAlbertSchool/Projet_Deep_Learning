"""Paired comparison of two classification runs on the same split (stream A, task A8).

Protocol pre-registered in README.md: paired bootstrap on macro-F1 (1,000 resamples, seed 42, 95 % CI) and McNemar on
accuracy (exact binomial if b + c < 25), both from `src.metrics`. The two runs must have predicted the same clauses in
the same order: their prediction files (common format, results/predictions/<run_id>_<split>.parquet) must hold
identical `idx` and `label` columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NUM_LABELS, PREDICTIONS_DIR, SEED
from src.metrics import classification_metrics, mcnemar_test, paired_bootstrap_f1

N_BOOT = 1_000  # pre-registered


def prediction_path(run_id: str, split: str) -> Path:
    return PREDICTIONS_DIR / f"{run_id}_{split}.parquet"


def load_aligned_predictions(run_a: str, run_b: str, split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prediction files of both runs, after checking that they cover the same clauses in the same order."""
    missing = [str(path) for path in (prediction_path(run_a, split), prediction_path(run_b, split))
               if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction file(s): {', '.join(missing)}")
    preds_a, preds_b = pd.read_parquet(prediction_path(run_a, split)), pd.read_parquet(prediction_path(run_b, split))
    for col in ("idx", "label"):
        if len(preds_a) != len(preds_b) or not np.array_equal(preds_a[col].to_numpy(), preds_b[col].to_numpy()):
            raise ValueError(f"{run_a} and {run_b} ({split}): column `{col}` differs (length or order); "
                             "a paired test needs the same clauses in the same order")
    return preds_a, preds_b


def read_exclude_idx(path) -> np.ndarray:
    """idx to drop: a CSV with an `idx` column, or a text file with one idx per line."""
    df = pd.read_csv(path)
    if "idx" not in df.columns:
        df = pd.read_csv(path, header=None, names=["idx"])
    return df["idx"].astype(int).to_numpy()


def paired_comparison(labels, pred_a, pred_b, n_boot: int = N_BOOT, seed: int = SEED,
                      num_labels: int = NUM_LABELS) -> dict:
    """Macro-F1 and accuracy of A and B, paired bootstrap of ΔF1 = F1(A) − F1(B), McNemar on accuracy."""
    metrics_a = classification_metrics(labels, pred_a, num_labels=num_labels)
    metrics_b = classification_metrics(labels, pred_b, num_labels=num_labels)
    boot = paired_bootstrap_f1(labels, pred_a, pred_b, n_boot=n_boot, seed=seed, num_labels=num_labels)
    mcnemar = mcnemar_test(labels, pred_a, pred_b)
    return {
        "n": len(labels),
        "macro_f1_a": metrics_a["macro_f1"],
        "macro_f1_b": metrics_b["macro_f1"],
        "delta_macro_f1": boot["delta_macro_f1"],
        "ci95_low": boot["ci95_low"],
        "ci95_high": boot["ci95_high"],
        "significant": boot["significant"],
        "p_a_not_better": boot["p_a_not_better"],
        "n_boot": n_boot,
        "seed": seed,
        "accuracy_a": metrics_a["accuracy"],
        "accuracy_b": metrics_b["accuracy"],
        "b": mcnemar["b"],
        "c": mcnemar["c"],
        "mcnemar_p_value": mcnemar["p_value"],
        "mcnemar_method": mcnemar["method"],
    }


def compare_runs(run_a: str, run_b: str, split: str = "test", exclude_idx_file=None, **kwargs) -> list[dict]:
    """One row on the whole split ("all"), plus one without the idx of `exclude_idx_file` ("deduplicated")."""
    preds_a, preds_b = load_aligned_predictions(run_a, run_b, split)
    keep = {"all": np.ones(len(preds_a), dtype=bool)}
    if exclude_idx_file is not None:
        exclude = read_exclude_idx(exclude_idx_file)
        unknown = np.setdiff1d(exclude, preds_a["idx"])
        if len(unknown):
            raise ValueError(f"{len(unknown)} idx of {exclude_idx_file} are not in the {split} predictions, "
                             f"e.g. {unknown[:5].tolist()}: wrong split?")
        keep["deduplicated"] = ~preds_a["idx"].isin(exclude).to_numpy()
    rows = []
    for variant, mask in keep.items():
        rows.append({
            "model_a": run_a, "model_b": run_b, "split": split, "variant": variant,
            "exclude_idx_file": Path(exclude_idx_file).name if variant != "all" else "",
            **paired_comparison(preds_a["label"].to_numpy()[mask], preds_a["pred"].to_numpy()[mask],
                                preds_b["pred"].to_numpy()[mask], **kwargs),
        })
    return rows


def update_results_csv(rows: list[dict], path) -> pd.DataFrame:
    """Write `rows` to `path`, replacing earlier rows of the same pair, split and variant; other rows are kept."""
    key = ["model_a", "model_b", "split", "variant"]
    new = pd.DataFrame(rows)
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        replaced = old[key].astype(str).apply(tuple, axis=1).isin(set(new[key].astype(str).apply(tuple, axis=1)))
        new = pd.concat([old[~replaced], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new
