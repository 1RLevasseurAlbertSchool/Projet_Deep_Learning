"""Unified comparison of the classification models (stream B, task B8).

Reads results/runs/*/config.json and metrics.json (smoke runs and runs re-evaluated under another run_id are
skipped), results/lora_tradeoff.csv (cross-check of the LoRA row) and results/paired_tests.csv (test set, whole
split). Writes results/summary_classification.csv (one row per model, cells formatted for display: 3 decimals for
F1 and accuracy, "not run" / "not recorded" where no number exists), three figures in results/figures/, and prints
the markdown table used in the README.

Paired tests are read, not recomputed: ΔF1 = F1(Legal-BERT final) − F1(model), paired bootstrap 95 % CI and McNemar
p as written by scripts/paired_tests.py.

Example:
    python scripts/summary_classification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIGURES_DIR, NUM_LABELS, RESULTS_DIR  # noqa: E402
from src.utils import list_runs, load_run, markdown_table  # noqa: E402

OUT_PATH = RESULTS_DIR / "summary_classification.csv"
FINAL_RUN = "legal-bert_final_20261019-1523"

# Models of the comparison, in table order: run_id -> (display name, role).
MODELS = {
    "tfidf-logreg_baseline_20261023": ("TF-IDF + LogReg (C = 1.0, not tuned)", "baseline (main comparison, Model B)"),
    "distilbert_full_20261019-0838": ("DistilBERT (full fine-tuning)", "screening, validation only"),
    FINAL_RUN: ("Legal-BERT (full fine-tuning)", "final model (Model A)"),
    "legal-bert_lora-r8_20261020-0940": ("Legal-BERT (LoRA, r=8)", "secondary comparison"),
}

# Numbers measured but not stored in the run files: value and where it was recorded.
RECORDED_ELSEWHERE = {
    "tfidf-logreg_baseline_20261023": {
        "val_macro_f1": (0.766555, "validation macro-F1 from the output of notebooks/01_eda_baseline.ipynb "
                                   "(the run's metrics.json has no validation block)"),
    },
    FINAL_RUN: {
        "weights_bytes": (438_260_096, "model.safetensors, measured on a same-architecture file (PROGRESS.md, A7)"),
    },
    "legal-bert_lora-r8_20261020-0940": {
        "weights_bytes": (1_494_128, "adapter_model.safetensors only; serving also needs the pretrained Legal-BERT "
                                     "(PROGRESS.md, A7)"),
    },
}

NOT_RUN, NOT_RECORDED = "not run", "not recorded"

# Reference categorical palette, slots 1-3 (validated all-pairs, light surface); colour follows the model.
COLORS = {FINAL_RUN: "#2a78d6", "tfidf-logreg_baseline_20261023": "#eb6834",
          "legal-bert_lora-r8_20261020-0940": "#1baf7a"}
SHORT = {FINAL_RUN: "Legal-BERT", "tfidf-logreg_baseline_20261023": "TF-IDF + LogReg",
         "legal-bert_lora-r8_20261020-0940": "LoRA r=8"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def selected_runs() -> list[str]:
    """run_ids of the comparison; fails if a non-smoke run is neither listed in MODELS nor re-evaluated elsewhere."""
    runs = [run_id for run_id in list_runs() if "smoke" not in load_run(run_id)[0]["variant"]]
    re_evaluated = {load_run(run_id)[0].get("source_run") for run_id in runs}
    unknown = [run_id for run_id in runs if run_id not in MODELS and run_id not in re_evaluated]
    if unknown:
        raise ValueError(f"Run(s) not in MODELS: {unknown}; add them to the comparison or skip them explicitly")
    missing = [run_id for run_id in MODELS if run_id not in runs]
    if missing:
        raise FileNotFoundError(f"Run(s) of MODELS not found in results/runs/: {missing}")
    return list(MODELS)


def trained_params(config: dict, metrics: dict) -> tuple[int, str]:
    """Trained parameters and a note; for TF-IDF + LogReg, the LogReg weights (one row of coefficients + bias per
    class), derived from `n_features` in config.json."""
    if metrics.get("n_trainable_params") is not None:
        return metrics["n_trainable_params"], ""
    return NUM_LABELS * (config["n_features"] + 1), (f"trained parameters = {NUM_LABELS} × ({config['n_features']:,} "
                                                     "TF-IDF features + 1 bias), LogReg weights from config.json")


def collect() -> pd.DataFrame:
    """One row per model with raw numbers (NaN where nothing exists) and a `notes` column."""
    paired = pd.read_csv(RESULTS_DIR / "paired_tests.csv")
    paired = paired[(paired["model_a"] == FINAL_RUN) & (paired["split"] == "test") & (paired["variant"] == "all")]
    rows = []
    for run_id in selected_runs():
        config, metrics = load_run(run_id)
        extra = RECORDED_ELSEWHERE.get(run_id, {})
        params, params_note = trained_params(config, metrics)
        val = metrics.get("validation") or {}
        test = metrics.get("test") or {}
        pair = paired[paired["model_b"] == run_id]
        if len(pair) > 1:
            raise ValueError(f"Several paired-test rows for {FINAL_RUN} vs {run_id}")
        pair = pair.iloc[0] if len(pair) else None
        if pair is not None and not np.isclose(pair["macro_f1_b"], test["macro_f1"]):
            raise ValueError(f"{run_id}: macro-F1 in paired_tests.csv differs from its metrics.json")
        rows.append({
            "run_id": run_id,
            "trained_params": params,
            "weights_mb": extra["weights_bytes"][0] / 1e6 if "weights_bytes" in extra else np.nan,
            "train_time_s": metrics.get("train_time_s", np.nan),
            "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb", np.nan),
            "on_gpu": config.get("device") == "cuda",
            "val_macro_f1": val.get("macro_f1", extra.get("val_macro_f1", (np.nan,))[0]),
            "test_macro_f1": test.get("macro_f1", np.nan),
            "test_accuracy": test.get("accuracy", np.nan),
            "delta": np.nan if pair is None else pair["delta_macro_f1"],
            "ci_low": np.nan if pair is None else pair["ci95_low"],
            "ci_high": np.nan if pair is None else pair["ci95_high"],
            "mcnemar_p": np.nan if pair is None else pair["mcnemar_p_value"],
            "per_class_test_f1": test.get("per_class_f1"),
            "notes": "; ".join([n for n in [params_note, *(note for _, note in extra.values())] if n]),
        })
    df = pd.DataFrame(rows)
    full = df.loc[df["run_id"] == FINAL_RUN, "trained_params"].item()
    df["pct_of_full"] = 100 * df["trained_params"] / full
    check_lora_tradeoff(df)
    return df


def check_lora_tradeoff(df: pd.DataFrame) -> None:
    """The LoRA trade-off table (A7) and the run files must give the same numbers."""
    tradeoff = pd.read_csv(RESULTS_DIR / "lora_tradeoff.csv").set_index("run_id")
    for col, ref in [("trained_params", "trainable_params"), ("pct_of_full", "pct_of_full"),
                     ("train_time_s", "train_time_s"), ("test_macro_f1", "test_macro_f1")]:
        for run_id, value in df.set_index("run_id")[col].items():
            if run_id in tradeoff.index and not np.isclose(value, tradeoff.loc[run_id, ref]):
                raise ValueError(f"{run_id}: {col} = {value} but lora_tradeoff.csv has {tradeoff.loc[run_id, ref]}")


def _fmt(value, spec: str, missing: str = NOT_RECORDED) -> str:
    return missing if value is None or value != value else format(value, spec)  # NaN != NaN


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cells as written to the CSV and the README."""
    out = []
    for row in df.itertuples(index=False):
        name, role = MODELS[row.run_id]
        reference = row.run_id == FINAL_RUN
        no_test = row.test_macro_f1 != row.test_macro_f1
        out.append({
            "model": name,
            "run_id": row.run_id,
            "role": role,
            "trained_params": f"{row.trained_params:,d}",
            "pct_of_full": f"{row.pct_of_full:.2f}",
            "weights_on_disk_mb": _fmt(row.weights_mb, ",.2f"),
            "train_time_s": _fmt(row.train_time_s, ",.1f"),
            "peak_gpu_memory_mb": _fmt(row.peak_gpu_memory_mb, ",.1f") if row.on_gpu else "not applicable (CPU)",
            "val_macro_f1": _fmt(row.val_macro_f1, ".3f"),
            "test_macro_f1": _fmt(row.test_macro_f1, ".3f", NOT_RUN),
            "test_accuracy": _fmt(row.test_accuracy, ".3f", NOT_RUN),
            "delta_macro_f1_final_minus_model": "reference" if reference else _fmt(row.delta, "+.3f", NOT_RUN),
            "ci95_delta": ("reference" if reference else NOT_RUN if row.delta != row.delta
                           else f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]"),
            "mcnemar_p": "reference" if reference else _fmt(row.mcnemar_p, ".2e", NOT_RUN),
            "notes": row.notes if not no_test else "; ".join(
                n for n in ["selected out on validation; test never evaluated (single test evaluation reserved for "
                            "the chosen encoder)", row.notes] if n),
        })
    return pd.DataFrame(out)


def readme_table(table: pd.DataFrame) -> str:
    shown = table.drop(columns=["run_id", "role", "notes"]).rename(columns={
        "model": "Model", "trained_params": "Trained parameters", "pct_of_full": "% of full",
        "weights_on_disk_mb": "Weights on disk (MB)", "train_time_s": "Train time (s)",
        "peak_gpu_memory_mb": "Peak GPU memory (MB)", "val_macro_f1": "Val macro-F1",
        "test_macro_f1": "Test macro-F1", "test_accuracy": "Test accuracy",
        "delta_macro_f1_final_minus_model": "ΔF1 (Legal-BERT − model)", "ci95_delta": "95 % CI of ΔF1",
        "mcnemar_p": "McNemar p"})
    return markdown_table(shown)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_test_scores(df: pd.DataFrame, path: Path) -> None:
    """(a) Test macro-F1 and accuracy per model, with the paired-bootstrap CI of ΔF1 to Legal-BERT."""
    tested = df[df["test_macro_f1"].notna()]
    order = [FINAL_RUN] + [r for r in tested["run_id"] if r != FINAL_RUN]
    tested = tested.set_index("run_id").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, facecolor=SURFACE)
    for ax, col, title in [(axes[0], "test_macro_f1", "Test macro-F1"), (axes[1], "test_accuracy", "Test accuracy")]:
        _style(ax)
        x = np.arange(len(tested))
        ax.bar(x, tested[col], width=0.6, color=[COLORS[r] for r in tested.index], edgecolor=SURFACE, linewidth=2)
        for xi, (run_id, value) in zip(x, tested[col].items()):
            ax.text(xi, value + 0.015, f"{value:.3f}", ha="center", va="bottom", color=INK, fontsize=10)
            if col == "test_macro_f1" and run_id != FINAL_RUN:
                row = tested.loc[run_id]
                ax.text(xi, value + 0.075, f"Δ {row['delta']:+.3f}\n[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
                        ha="center", va="bottom", color=INK_2, fontsize=8.5)
        ax.set_xticks(x, [SHORT[r] for r in tested.index], color=INK)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, color=INK, loc="left", fontsize=12)
    axes[0].set_ylabel("score (test set, 10,000 clauses)", color=INK_2)
    no_test = ", ".join(MODELS[r][0] for r in df.loc[df["test_macro_f1"].isna(), "run_id"])
    fig.text(0.01, 0.01, "Δ = F1(Legal-BERT) − F1(model), paired bootstrap 95 % CI (1,000 resamples, seed 42), from "
             f"results/paired_tests.csv.\nNot shown (test not run): {no_test}.", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_per_class(df: pd.DataFrame, path: Path) -> None:
    """(b) Per-class test F1 of TF-IDF, Legal-BERT and LoRA, classes sorted by Legal-BERT F1."""
    per_class = {r: np.asarray(f1) for r, f1 in zip(df["run_id"], df["per_class_test_f1"]) if f1 is not None}
    order = np.argsort(-per_class[FINAL_RUN], kind="stable")
    rank = np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    _style(ax)
    ax.plot(rank, per_class[FINAL_RUN][order], color=COLORS[FINAL_RUN], linewidth=2, label=SHORT[FINAL_RUN],
            zorder=3)
    for run_id, marker in [("tfidf-logreg_baseline_20261023", "o"), ("legal-bert_lora-r8_20261020-0940", "^")]:
        ax.scatter(rank, per_class[run_id][order], s=22, marker=marker, color=COLORS[run_id], edgecolor=SURFACE,
                   linewidth=0.8, label=SHORT[run_id], zorder=2)
    ax.set_xlim(0, len(order) + 1)
    ax.set_ylim(-0.02, 1.02)  # classes at F1 = 0 stay visible
    ax.set_xlabel("class rank by Legal-BERT test F1 (1 = best of the 100 LEDGAR classes)", color=INK_2)
    ax.set_ylabel("test F1 per class", color=INK_2)
    ax.set_title("Per-class test F1, sorted by Legal-BERT", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, path: Path) -> None:
    """(c) Trained parameters (log scale) vs test macro-F1."""
    tested = df[df["test_macro_f1"].notna()]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for row in tested.itertuples(index=False):
        ax.scatter(row.trained_params, row.test_macro_f1, s=90, color=COLORS[row.run_id], edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(f"{SHORT[row.run_id]}\n{row.trained_params:,d} params, F1 {row.test_macro_f1:.3f}",
                    (row.trained_params, row.test_macro_f1), xytext=(10, -4), textcoords="offset points",
                    color=INK, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e9)
    low, high = tested["test_macro_f1"].min(), tested["test_macro_f1"].max()
    ax.set_ylim(low - 0.02, high + 0.015)
    ax.set_xlabel("trained parameters (log scale)", color=INK_2)
    ax.set_ylabel("test macro-F1", color=INK_2)
    ax.set_title("Trained parameters vs test macro-F1", color=INK, loc="left", fontsize=12)
    no_test = ", ".join(MODELS[r][0] for r in df.loc[df["test_macro_f1"].isna(), "run_id"])
    fig.text(0.01, 0.01, f"TF-IDF + LogReg: LogReg weights only (100 × (features + 1)). Not shown (test not run): "
             f"{no_test}.\nThe y-axis does not start at 0.", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    df = collect()
    table = display_table(df)
    table.to_csv(OUT_PATH, index=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures = {"summary_test_scores.png": plot_test_scores, "summary_per_class_f1.png": plot_per_class,
               "summary_tradeoff.png": plot_tradeoff}
    for name, plot in figures.items():
        plot(df, FIGURES_DIR / name)
    print(readme_table(table))
    print(f"\nWritten to {OUT_PATH.relative_to(RESULTS_DIR.parent)} and results/figures/: {', '.join(figures)}")


if __name__ == "__main__":
    main()
