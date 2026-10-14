"""Utilitaires : graine, sauvegarde des runs et des prédictions au format commun (voir CLAUDE.md)."""

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
    """Écrit results/runs/<run_id>/config.json (+ versions) et metrics.json."""
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**config, "seed": config.get("seed", SEED), "versions": _versions()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def save_predictions(run_id: str, split: str, labels, preds, probas) -> None:
    """Écrit results/predictions/<run_id>_<split>.parquet au format commun.

    `probas` : tableau (n, 100) de probabilités ; on en déduit proba_pred et top5.
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
