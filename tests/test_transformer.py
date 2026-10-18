"""Tests of the stream A fine-tuning helpers — CPU only, no model download."""

import numpy as np
import pandas as pd
import pytest

import src.utils
from src.config import NUM_LABELS
from src.models.transformer import (
    logits_to_predictions,
    save_split_predictions,
    short_model_name,
    token_length_stats,
)


@pytest.fixture
def tiny_tokenizer_dir(tmp_path):
    """Local WordPiece tokenizer: one token per word "w", plus [CLS] and [SEP]."""
    from transformers import BertTokenizerFast

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "w"]))
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(tmp_path / "tok")
    return str(tmp_path / "tok")


def test_token_length_stats_counts_special_tokens_without_truncation(tiny_tokenizer_dir):
    n_words = [1, 6, 200, 300, 600]  # -> 3, 8, 202, 302, 602 tokens with [CLS] and [SEP]
    stats = token_length_stats(tiny_tokenizer_dir, [" ".join(["w"] * n) for n in n_words])
    assert stats["n"] == 5
    assert stats["p50"] == 202
    assert stats["max"] == 602  # no truncation at 512
    assert stats["p99"] == pytest.approx(np.percentile([3, 8, 202, 302, 602], 99))
    assert stats["share_gt_128"] == pytest.approx(3 / 5)
    assert stats["share_gt_256"] == pytest.approx(2 / 5)
    assert stats["share_gt_512"] == pytest.approx(1 / 5)


def test_prediction_parquet_common_format(tmp_path, monkeypatch):
    monkeypatch.setattr(src.utils, "PREDICTIONS_DIR", tmp_path)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6, NUM_LABELS))
    labels = np.array([3, 7, 7, 0, 99, 42])
    logits[np.arange(6), labels] += 10  # every example predicted correctly
    logits[5, 42], logits[5, 41] = 0, 20  # except the last one

    metrics = save_split_predictions("fake_run", "validation", labels, logits)

    df = pd.read_parquet(tmp_path / "fake_run_validation.parquet")
    assert list(df.columns) == ["idx", "label", "pred", "proba_pred", "top5"]
    assert df["idx"].tolist() == list(range(6))
    assert df["label"].tolist() == labels.tolist()
    assert df["pred"].tolist() == [3, 7, 7, 0, 99, 41]
    _, probas = logits_to_predictions(logits)
    np.testing.assert_allclose(probas.sum(axis=1), 1)
    np.testing.assert_allclose(df["proba_pred"], probas.max(axis=1))
    assert all(len(t) == 5 and t[0] == p for t, p in zip(df["top5"], df["pred"]))
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert len(metrics["per_class_f1"]) == NUM_LABELS


@pytest.mark.parametrize("name, short", [
    ("distilbert-base-uncased", "distilbert"),
    ("bert-base-uncased", "bert"),
    ("nlpaueb/legal-bert-base-uncased", "legal-bert"),
    ("microsoft/deberta-v3-base", "deberta-v3"),
])
def test_short_model_name(name, short):
    assert short_model_name(name) == short
