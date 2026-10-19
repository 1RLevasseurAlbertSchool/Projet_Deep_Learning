"""Tests of the stream A fine-tuning helpers — CPU only, no model download."""

import json

import numpy as np
import pandas as pd
import pytest

import src.utils
from src.config import NUM_LABELS
from src.models.transformer import (
    RepeatedTestEvaluationError,
    check_single_test_evaluation,
    evaluate_checkpoint,
    logits_to_predictions,
    save_split_predictions,
    short_model_name,
    token_length_stats,
)
from src.utils import runs_with_test

SOURCE_RUN = "distilbert_full_20261020-1200"


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


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(src.utils, "RUNS_DIR", tmp_path)
    return tmp_path


def write_run(runs_dir, run_id, splits=("validation",), **config):
    """Minimal results/runs/<run_id>/: config.json + metrics.json with one block per evaluated split."""
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"run_id": run_id, **config}))
    (run_dir / "metrics.json").write_text(json.dumps({split: {"macro_f1": 0.5} for split in splits}))


def test_guard_allows_first_test_evaluation(runs_dir):
    write_run(runs_dir, SOURCE_RUN, best_checkpoint="models/x/checkpoint-1")
    write_run(runs_dir, "distilbert_final_20261021-0900", source_run=SOURCE_RUN)  # crashed before test
    write_run(runs_dir, "bert_final_20261021-0900", ("validation", "test"), source_run="bert_full_20261020-1200")
    assert runs_with_test(SOURCE_RUN) == []
    check_single_test_evaluation(SOURCE_RUN)  # does not raise


def test_guard_refuses_second_test_evaluation_before_loading_anything(runs_dir):
    write_run(runs_dir, SOURCE_RUN, best_checkpoint="models/x/checkpoint-1")
    write_run(runs_dir, "distilbert_final_20261021-0900", ("validation", "test"), source_run=SOURCE_RUN)
    with pytest.raises(RepeatedTestEvaluationError, match="distilbert_final_20261021-0900"):
        evaluate_checkpoint(SOURCE_RUN)
    assert len(list(runs_dir.iterdir())) == 2  # no new run written


def test_guard_counts_a_source_run_trained_with_eval_test(runs_dir):
    write_run(runs_dir, SOURCE_RUN, ("validation", "test"), best_checkpoint="models/x/checkpoint-1")
    assert runs_with_test(SOURCE_RUN) == [SOURCE_RUN]
    with pytest.raises(RepeatedTestEvaluationError, match="only once"):
        check_single_test_evaluation(SOURCE_RUN)


def test_guard_force_warns_then_goes_on(runs_dir, capsys):
    write_run(runs_dir, SOURCE_RUN, best_checkpoint="missing/checkpoint-1")
    write_run(runs_dir, "distilbert_final_20261021-0900", ("validation", "test"), source_run=SOURCE_RUN)
    # Past the guard, the missing checkpoint stops it before any model loading or download
    with pytest.raises(FileNotFoundError, match="missing/checkpoint-1"):
        evaluate_checkpoint(SOURCE_RUN, force=True)
    out = capsys.readouterr().out
    assert "--force" in out and "only once" in out and "distilbert_final_20261021-0900" in out
