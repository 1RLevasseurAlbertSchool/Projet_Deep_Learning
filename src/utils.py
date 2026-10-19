"""Utilities: seed, saving runs and predictions in the common format (see CLAUDE.md)."""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime

import numpy as np

from src.config import PREDICTIONS_DIR, RUNS_DIR, SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_run_id(model: str, variant: str = "base") -> str:
    return f"{model}_{variant}_{datetime.now():%Y%m%d-%H%M}"


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "sentence_transformers", "peft", "sklearn", "datasets"):
        try:
            out[pkg] = __import__(pkg).__version__
        except ImportError:
            pass
    return out


def save_run(run_id: str, config: dict, metrics: dict) -> None:
    """Write results/runs/<run_id>/config.json (+ versions) and metrics.json."""
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**config, "seed": config.get("seed", SEED), "versions": _versions()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def load_run(run_id: str) -> tuple[dict, dict]:
    """(config, metrics) of results/runs/<run_id>/."""
    run_dir = RUNS_DIR / run_id
    return (json.loads((run_dir / "config.json").read_text()),
            json.loads((run_dir / "metrics.json").read_text()))


def list_runs() -> list[str]:
    """run_id of every results/runs/<run_id>/ with a config.json, sorted."""
    return sorted(path.parent.name for path in RUNS_DIR.glob("*/config.json"))


def markdown_table(df) -> str:
    """DataFrame as a markdown table (for PROGRESS.md / README), cells as formatted by the caller."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def runs_with_test(source_run: str) -> list[str]:
    """Runs that already predicted the test set with the model of `source_run`.

    That is `source_run` itself (trained with --eval-test) or any run whose config.json records it as
    `source_run` (scripts/evaluate_checkpoint.py), when its metrics.json has a "test" block.
    """
    found = []
    for metrics_path in sorted(RUNS_DIR.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        config = json.loads((run_dir / "config.json").read_text())
        if source_run in (run_dir.name, config.get("source_run")) and "test" in json.loads(metrics_path.read_text()):
            found.append(run_dir.name)
    return found


def save_predictions(run_id: str, split: str, labels, preds, probas) -> None:
    """Write results/predictions/<run_id>_<split>.parquet in the common format.

    `probas`: (n, 100) array of probabilities; proba_pred and top5 are derived from it.
    """
    import pandas as pd

    probas = np.asarray(probas)
    preds = np.asarray(preds)
    top5 = np.argsort(-probas, axis=1)[:, :5]
    df = pd.DataFrame({
        "idx": np.arange(len(preds)),
        "label": np.asarray(labels),
        "pred": preds,
        "proba_pred": probas[np.arange(len(preds)), preds],
        "top5": [row.tolist() for row in top5],
    })
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PREDICTIONS_DIR / f"{run_id}_{split}.parquet", index=False)
