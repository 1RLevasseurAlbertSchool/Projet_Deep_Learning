"""End-to-end clause review: classification + human-review decision + similar clauses (bonus task T4).

No training happens here: the pipeline only loads what earlier steps produced.

- **Classifier**: the best checkpoint of `legal-bert_full_20261019-0916`, i.e. the model evaluated once on the test
  set as `legal-bert_final_20261019-1523`, copied to `models/legal-bert_final/` (gitignored), with its own tokenizer
  and the training `max_length` (256).
- **Human-review threshold**: the recommended operating point of the calibration report (A9), chosen on validation
  (confidence >= threshold → automatic; below → sent to a lawyer).
- **Retrieval**: dense index over the training bank (`src/retrieval/dense.py`), with the encoder that ranks first on
  P@5 in `results/retrieval_summary.csv` (MiniLM if that file is missing). Built and cached on first use.

Usage:
    from src.pipeline import ClauseReviewer
    reviewer = ClauseReviewer.from_local()
    reviewer.review("This Agreement shall be governed by the laws of ...")
    reviewer.search("termination clauses with a 30-day notice period")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODELS_DIR, RESULTS_DIR, ROOT
from src.retrieval.dense import MODEL_NAME as DEFAULT_ENCODER

FINAL_RUN_ID = "legal-bert_final_20261019-1523"
SOURCE_RUN_ID = "legal-bert_full_20261019-0916"
DEFAULT_CHECKPOINT_DIR = MODELS_DIR / "legal-bert_final"
DRIVE_CHECKPOINT = f"MyDrive/ledgar_checkpoints/{SOURCE_RUN_ID}/checkpoint-11250"
CHECKPOINT_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
CALIBRATION_PATH = RESULTS_DIR / f"calibration_{FINAL_RUN_ID}.json"
RETRIEVAL_SUMMARY_PATH = RESULTS_DIR / "retrieval_summary.csv"
MAX_LENGTH = 256  # max_length of the training run (A3)
AUTOMATIC = "automatic"
SEND_TO_LAWYER = "send to a lawyer"
SNIPPET_CHARS = 200


def check_checkpoint_dir(checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    """Return the checkpoint folder, or raise a FileNotFoundError saying which files to download from Drive."""
    checkpoint_dir = Path(checkpoint_dir)
    missing = [name for name in CHECKPOINT_FILES if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Classifier checkpoint incomplete in {checkpoint_dir} (missing: {', '.join(missing)}).\n"
            f"Download these {len(CHECKPOINT_FILES)} files from the Google Drive folder {DRIVE_CHECKPOINT}/ "
            f"(best checkpoint of {SOURCE_RUN_ID}, evaluated as {FINAL_RUN_ID}) into {checkpoint_dir}/:\n"
            + "\n".join(f"  - {name}" for name in CHECKPOINT_FILES)
        )
    return checkpoint_dir


def load_review_threshold(calibration_path: str | Path = CALIBRATION_PATH) -> float:
    """Threshold of the recommended operating point of the calibration report (chosen on validation)."""
    report = json.loads(Path(calibration_path).read_text())
    target = report["recommended_target_precision"]
    for point in report["operating_points"]:
        if point["target_precision"] == target:
            return float(point["threshold"])
    raise ValueError(f"No operating point for the recommended target precision {target} in {calibration_path}")


def best_retrieval_encoder(summary_path: str | Path = RETRIEVAL_SUMMARY_PATH, metric: str = "precision@5") -> str:
    """Dense encoder with the best `metric` in the retrieval benchmark; MiniLM if the summary is missing.

    Only Hugging Face model names (`org/name`) are candidates: BM25 and the hybrid are not dense encoders.
    """
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        return DEFAULT_ENCODER
    summary = pd.read_csv(summary_path)
    dense = summary[summary["method"].str.fullmatch(r"[\w.-]+/[\w.-]+")]
    if dense.empty:
        return DEFAULT_ENCODER
    return str(dense.loc[dense[metric].idxmax(), "method"])


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClauseReviewer:
    """Classify a clause, decide whether a lawyer must check it, and show the closest training clauses."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        retriever: Any,
        threshold: float,
        *,
        max_length: int = MAX_LENGTH,
        device: str = "cpu",
        encoder_name: str | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a probability in [0, 1]")
        import torch

        self.torch = torch
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.threshold = threshold
        self.max_length = max_length
        self.device = device
        self.encoder_name = encoder_name
        self.label_names = [model.config.id2label[i] for i in range(model.config.num_labels)]

    @classmethod
    def from_local(
        cls,
        checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
        *,
        calibration_path: str | Path = CALIBRATION_PATH,
        encoder_name: str | None = None,
        device: str | None = None,
        verbose: bool = True,
    ) -> ClauseReviewer:
        """Load the checkpoint, the threshold and the retrieval index (built and cached over train if missing)."""
        checkpoint_dir = check_checkpoint_dir(checkpoint_dir)
        threshold = load_review_threshold(calibration_path)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.data import load_ledgar, to_pandas
        from src.retrieval.dense import build_dense_index

        device = device or default_device()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

        encoder_name = encoder_name or best_retrieval_encoder()
        bank = to_pandas(load_ledgar(), "train")
        started = time.perf_counter()
        retriever = build_dense_index(bank, model_name=encoder_name, device=device)
        elapsed_s = time.perf_counter() - started
        if verbose:
            how = "loaded from cache" if retriever.cache_hit else "built and cached"
            shown = checkpoint_dir.resolve()
            shown = shown.relative_to(ROOT) if shown.is_relative_to(ROOT) else shown
            print(f"Classifier: {shown} on {device} | review threshold {threshold}")
            print(f"Retrieval index ({encoder_name}, {len(bank):,} training clauses) {how} in {elapsed_s:.1f} s")
        return cls(model, tokenizer, retriever, threshold, device=device, encoder_name=encoder_name)

    def classify(self, text: str, top: int = 3) -> list[dict]:
        """The `top` most probable clause types: `label`, `label_name`, `probability`."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(**encoded).logits[0], dim=-1)
        values, labels = probabilities.topk(min(top, len(self.label_names)))
        return [
            {"label": int(label), "label_name": self.label_names[int(label)], "probability": float(value)}
            for value, label in zip(values.cpu(), labels.cpu())
        ]

    def search(self, query: str, k: int = 5) -> list[dict]:
        """The k closest training clauses: `idx`, `label`, `label_name`, `score`, `snippet` (first 200 chars)."""
        return [
            {
                "idx": hit["idx"],
                "label": hit["label"],
                "label_name": hit["label_name"],
                "score": hit["score"],
                "snippet": hit["text"][:SNIPPET_CHARS],
            }
            for hit in self.retriever.search(query, k=k)
        ]

    def review(self, text: str, k: int = 5) -> dict:
        """Top-3 types, the human-review decision and the k most similar training clauses."""
        top3 = self.classify(text, top=3)
        confidence = top3[0]["probability"]
        return {
            "top3": top3,
            "confidence": confidence,
            "threshold": self.threshold,
            "decision": AUTOMATIC if confidence >= self.threshold else SEND_TO_LAWYER,
            "similar": self.search(text, k=k),
        }


def format_review(result: dict) -> str:
    """Plain-text rendering of `ClauseReviewer.review` for the command line."""
    lines = ["Clause types:"]
    lines += [f"  {i}. {t['label_name']:<35} {t['probability']:.3f}" for i, t in enumerate(result["top3"], 1)]
    lines.append(
        f"Decision: {result['decision'].upper()} "
        f"(confidence {result['confidence']:.3f} vs threshold {result['threshold']:.3f})"
    )
    lines += ["", format_hits(result["similar"])]
    return "\n".join(lines)


def format_hits(hits: list[dict]) -> str:
    """Plain-text rendering of `ClauseReviewer.search` results."""
    lines = ["Most similar training clauses:"]
    for rank, hit in enumerate(hits, 1):
        lines.append(f"  {rank}. [{hit['score']:.3f}] {hit['label_name']} (train idx {hit['idx']})")
        lines.append(f"     {hit['snippet']}")
    return "\n".join(lines)
