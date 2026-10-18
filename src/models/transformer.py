"""Hugging Face fine-tuning of a BERT-class encoder on LEDGAR (stream A).

Usage:
    from src.models.transformer import token_length_stats, train_and_evaluate
    stats = token_length_stats("distilbert-base-uncased", texts)    # choice of max_length (A3)
    out = train_and_evaluate({**DEFAULT_CONFIG, "model_name": "distilbert-base-uncased", "variant": "smoke"})

Every run writes results/runs/<run_id>/ (config.json, metrics.json) and
results/predictions/<run_id>_<split>.parquet through src.utils (interface contracts of CLAUDE.md).
The test set is predicted only when config["eval_test"] is True: once per final model.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.config import MODELS_DIR, NUM_LABELS, ROOT, SEED, TEXT_COL
from src.data import get_label_names, load_ledgar
from src.metrics import classification_metrics
from src.utils import make_run_id, save_predictions, save_run, set_seed

DEFAULT_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "variant": "base",
    "max_length": 256,
    "epochs": 3,
    "lr": 5e-5,
    "batch_size": 32,
    "warmup": 0.1,          # ratio of the total training steps (transformers 5: warmup_steps < 1)
    "weight_decay": 0.01,
    "subset": None,         # first N examples of each split: smoke tests only
    "eval_test": False,     # single test evaluation, final models only
    "output_dir": str(MODELS_DIR),
    "seed": SEED,
    "lora": None,           # hook for task A7 (peft LoRA); None = full fine-tuning
}

LENGTH_THRESHOLDS = (128, 256, 512)


def token_length_stats(model_name: str, texts) -> dict:
    """Token-length distribution with the model's own tokenizer, without truncation.

    Lengths include the special tokens ([CLS], [SEP]...), i.e. what `max_length` truncates.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # The "sequence longer than model_max_length" warning is expected here: we measure, we do not feed the model
    tokenizer.model_max_length = int(1e9)
    lengths = np.array([len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]])
    stats = {"n": int(len(lengths)), "mean": float(lengths.mean())}
    for q in (50, 90, 95, 99):
        stats[f"p{q}"] = float(np.percentile(lengths, q))
    stats["max"] = int(lengths.max())
    for t in LENGTH_THRESHOLDS:
        stats[f"share_gt_{t}"] = float((lengths > t).mean())
    return stats


def short_model_name(model_name: str) -> str:
    """Run-id prefix: `nlpaueb/legal-bert-base-uncased` -> `legal-bert`, `microsoft/deberta-v3-base` -> `deberta-v3`."""
    parts = model_name.rstrip("/").split("/")[-1].split("-")
    return "-".join(p for p in parts if p not in {"base", "uncased", "cased"}) or parts[0]


def model_revision(model_name: str) -> str | None:
    """Hub commit of the loaded weights (snapshot folder of the HF cache); None for a local folder."""
    from transformers.utils import cached_file

    path = Path(cached_file(model_name, "config.json"))
    return path.parent.name if path.parent.parent.name == "snapshots" else None


def _repo_relative(path) -> str | None:
    """Path relative to the repository root when inside it, so versioned configs stay machine-independent."""
    if path is None:
        return None
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def logits_to_predictions(logits) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class (argmax) and softmax probabilities (n, num_labels)."""
    from scipy.special import softmax

    probas = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return probas.argmax(axis=1), probas


def save_split_predictions(run_id: str, split: str, labels, logits) -> dict:
    """Write the prediction file of a split in the common format and return its metrics."""
    preds, probas = logits_to_predictions(logits)
    save_predictions(run_id, split, labels, preds, probas)
    return classification_metrics(labels, preds, num_labels=NUM_LABELS)


def _compute_metrics(eval_pred) -> dict:
    """Scalar metrics logged by the Trainer at each evaluation (macro-F1 selects the best epoch)."""
    logits, labels = eval_pred
    m = classification_metrics(labels, np.argmax(logits, axis=1), num_labels=NUM_LABELS)
    return {k: m[k] for k in ("macro_f1", "micro_f1", "accuracy")}


def _history(log_history: list[dict]) -> dict:
    """Training-loss and validation curves from the Trainer logs (for the loss / macro-F1 figures)."""
    train = [{"step": h["step"], "epoch": h["epoch"], "loss": h["loss"]} for h in log_history if "loss" in h]
    evals = [{"step": h["step"], "epoch": h["epoch"], "loss": h["eval_loss"], "macro_f1": h["eval_macro_f1"],
              "accuracy": h["eval_accuracy"]} for h in log_history if "eval_macro_f1" in h]
    return {"train": train, "validation": evals}


def train_and_evaluate(config: dict) -> dict:
    """Fine-tune `config["model_name"]` on LEDGAR, evaluate on validation (and test if `eval_test`).

    Returns {"run_id", "config", "metrics"}; the same content is written to results/runs/<run_id>/.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    config = {**DEFAULT_CONFIG, **config}
    if config["lora"] is not None:
        raise NotImplementedError("LoRA is task A7: config['lora'] must be None for full fine-tuning")

    set_seed(config["seed"])
    model_name = config["model_name"]
    run_id = make_run_id(short_model_name(model_name), config["variant"])

    # Data: the model's own tokenizer, truncation at max_length, padding per batch (collator)
    ds = load_ledgar(subset=config["subset"])
    label_names = get_label_names(ds)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = ds.map(
        lambda batch: tokenizer(batch[TEXT_COL], truncation=True, max_length=config["max_length"]),
        batched=True,
        remove_columns=[TEXT_COL],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=dict(enumerate(label_names)),
        label2id={name: i for i, name in enumerate(label_names)},
    )

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "mps" if torch.backends.mps.is_available() else "cpu"
    run_dir = Path(config["output_dir"]) / run_id
    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=config["epochs"],
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=2 * config["batch_size"],
        warmup_steps=config["warmup"],
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=0.02,  # ratio of the total steps: ~50 points on the loss curve
        fp16=use_cuda,
        dataloader_pin_memory=use_cuda,  # pinned memory is not supported on MPS
        report_to="none",
        seed=config["seed"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    train_time_s = time.perf_counter() - start

    # Best epoch (validation macro-F1) is loaded back: predictions always on validation, test only on request
    metrics = {
        "train_time_s": round(train_time_s, 1),
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_total_params": sum(p.numel() for p in model.parameters()),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if use_cuda else None,
    }
    splits = ["validation", "test"] if config["eval_test"] else ["validation"]
    for split in splits:
        out = trainer.predict(tokenized[split])
        metrics[split] = save_split_predictions(run_id, split, out.label_ids, out.predictions)
    metrics["history"] = _history(trainer.state.log_history)

    config = {
        **config,
        "run_id": run_id,
        "output_dir": _repo_relative(config["output_dir"]),
        "model_revision": model_revision(model_name),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if use_cuda else None,
        "fp16": use_cuda,
        "n_examples": {split: len(ds[split]) for split in ds},
        "best_epoch": next((h["epoch"] for h in trainer.state.log_history
                            if h.get("eval_macro_f1") == trainer.state.best_metric), None),
        "best_checkpoint": _repo_relative(trainer.state.best_model_checkpoint),
    }
    save_run(run_id, config, metrics)
    return {"run_id": run_id, "config": config, "metrics": metrics}
