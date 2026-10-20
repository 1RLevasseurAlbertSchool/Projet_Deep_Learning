"""Lightweight inference adapter for auditing a saved Hugging Face classifier."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class HuggingFaceTextPredictor:
    """Batch predictor returning the common top-label and confidence audit fields."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_path: str,
        *,
        revision: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
        device: str | None = None,
    ) -> HuggingFaceTextPredictor:
        """Load the model checkpoint and matching base tokenizer."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        return cls(
            model,
            tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )

    def __call__(self, texts: Sequence[str]) -> list[dict[str, float | int]]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("every input text must be a non-empty string")
        if not texts:
            return []

        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = self.torch.softmax(logits, dim=-1)
                confidences, labels = probabilities.max(dim=-1)
            outputs.extend(
                {
                    "label": int(label),
                    "confidence": float(confidence),
                }
                for label, confidence in zip(labels.cpu(), confidences.cpu())
            )
        return outputs


def prediction_summary(
    predictions: Sequence[dict[str, float | int]],
) -> dict[str, float | int]:
    """Compact confidence summary for a prediction-only OOD set."""
    if not predictions:
        raise ValueError("at least one prediction is required")
    confidences = np.asarray([float(item["confidence"]) for item in predictions])
    labels = np.asarray([int(item["label"]) for item in predictions])
    if not np.isfinite(confidences).all() or np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("Prediction confidences must be finite probabilities in [0, 1]")
    return {
        "n_examples": int(len(predictions)),
        "n_predicted_classes": int(len(np.unique(labels))),
        "mean_confidence": float(confidences.mean()),
        "median_confidence": float(np.median(confidences)),
    }
