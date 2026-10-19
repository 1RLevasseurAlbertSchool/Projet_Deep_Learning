"""Tests of the stream A fine-tuning helpers — CPU only, no model download."""

import json

import numpy as np
import pandas as pd
import pytest

import src.utils
from src.config import NUM_LABELS
from src.models.transformer import (
    RepeatedTestEvaluationError,
    apply_lora,
    check_single_test_evaluation,
    count_parameters,
    evaluate_checkpoint,
    logits_to_predictions,
    lora_settings,
    lora_tradeoff_table,
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


def tiny_model(model_type: str, d: int = 32, n_layers: int = 2):
    """Randomly initialised classifier with hidden size d (no download)."""
    import transformers as tf

    if model_type == "bert":
        config = tf.BertConfig(vocab_size=50, hidden_size=d, num_hidden_layers=n_layers, num_attention_heads=2,
                               intermediate_size=2 * d, num_labels=NUM_LABELS)
        return tf.BertForSequenceClassification(config)
    config = tf.DistilBertConfig(vocab_size=50, dim=d, n_layers=n_layers, n_heads=2, hidden_dim=2 * d,
                                 num_labels=NUM_LABELS)
    return tf.DistilBertForSequenceClassification(config)


@pytest.mark.parametrize("model_type, targets, head, head_params", [
    ("bert", ["query", "value"], ["classifier"], 32 * NUM_LABELS + NUM_LABELS),
    # DistilBERT's pre_classifier is new too (not in the pretrained checkpoint): trained in full
    ("distilbert", ["q_lin", "v_lin"], ["pre_classifier", "classifier"], 32 * 32 + 32 + 32 * NUM_LABELS + NUM_LABELS),
])
def test_lora_trains_only_the_lora_matrices_and_the_head(model_type, targets, head, head_params):
    import torch

    torch.manual_seed(0)
    d, n_layers, r = 32, 2, 8
    model = tiny_model(model_type, d, n_layers).eval()
    n_full = sum(p.numel() for p in model.parameters())
    input_ids = torch.randint(5, 50, (3, 7))
    with torch.no_grad():
        logits_before = model(input_ids=input_ids).logits

    lora = lora_settings(model, r=r)
    expected = {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": targets, "modules_to_save": head}
    assert lora == expected
    peft_model = apply_lora(model, lora).eval()
    assert lora == expected  # the settings recorded in config.json are not modified by peft

    # r * (d_in + d_out) per targeted matrix (A: r x d_in, B: d_out x r), 2 matrices per layer, plus the head
    n_lora = n_layers * len(targets) * r * (d + d)
    assert count_parameters(peft_model) == (n_lora + head_params, n_full + n_lora)
    trainable = [name for name, p in peft_model.named_parameters() if p.requires_grad]
    assert all(".lora_A." in name or ".lora_B." in name or ".modules_to_save." in name for name in trainable)
    assert sum(".lora_" in name for name in trainable) == 2 * n_layers * len(targets)
    # B starts at 0 and the head copy equals the original: at step 0 the output is the base model's (S06)
    with torch.no_grad():
        torch.testing.assert_close(peft_model(input_ids=input_ids).logits, logits_before)


def test_lora_settings_refuses_an_unknown_architecture():
    import transformers as tf

    config = tf.GPT2Config(n_embd=32, n_layer=1, n_head=2, vocab_size=50, num_labels=NUM_LABELS)
    with pytest.raises(ValueError, match="gpt2"):
        lora_settings(tf.GPT2ForSequenceClassification(config), r=8)


RECIPE = {"max_length": 256, "epochs": 3, "batch_size": 16, "warmup": 0.1, "weight_decay": 0.01, "seed": 42}


def write_full_run(runs_dir, run_id, model="nlpaueb/legal-bert-base-uncased", splits=("validation", "test"),
                   n_trainable=1000, lora=None, subset=None, lr=3e-5, **recipe):
    """results/runs/<run_id>/ with the config and metrics keys read by lora_tradeoff_table."""
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    config = {"run_id": run_id, "model_name": model, "lora": lora, "subset": subset, "lr": lr, **RECIPE, **recipe}
    metrics = {"train_time_s": 10.0, "n_trainable_params": n_trainable, "n_total_params": 1000,
               "peak_gpu_memory_mb": 500.0,
               **{split: {"macro_f1": 0.7, "accuracy": 0.8} for split in splits}}
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_lora_tradeoff_table_rows_and_filters(runs_dir):
    write_full_run(runs_dir, "legal-bert_final_1")
    write_full_run(runs_dir, "legal-bert_lora-r16_1", n_trainable=40, lora={"r": 16}, lr=5e-4)
    write_full_run(runs_dir, "legal-bert_lora-r8_1", n_trainable=20, lora={"r": 8}, lr=5e-4, splits=("validation",))
    write_full_run(runs_dir, "legal-bert_smoke-lora-r8_1", n_trainable=20, lora={"r": 8}, subset=200)  # smoke
    write_full_run(runs_dir, "distilbert_lora-r8_1", model="distilbert-base-uncased", lora={"r": 8})  # other encoder

    df = lora_tradeoff_table("legal-bert_final_1")

    assert df["run_id"].tolist() == ["legal-bert_final_1", "legal-bert_lora-r8_1", "legal-bert_lora-r16_1"]
    assert df["method"].tolist() == ["full fine-tuning", "LoRA r=8", "LoRA r=16"]
    assert df["pct_of_full"].tolist() == pytest.approx([100, 2, 4])
    assert pd.isna(df.loc[1, "test_macro_f1"]) and df.loc[2, "test_macro_f1"] == 0.7  # r=8 without --eval-test


def test_lora_tradeoff_table_warns_on_a_different_recipe(runs_dir):
    write_full_run(runs_dir, "legal-bert_final_1")
    write_full_run(runs_dir, "legal-bert_lora-r8_1", lora={"r": 8}, lr=5e-4, max_length=512)
    with pytest.warns(UserWarning, match="max_length"):
        lora_tradeoff_table("legal-bert_final_1")
