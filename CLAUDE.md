# CLAUDE.md

This file guides Claude Code (and the team members) when working in this repository.
Read it at the start of every session, together with [`PROGRESS.md`](PROGRESS.md) and [`tasks/lessons.md`](tasks/lessons.md).

## Project

Graded final project of the course **Deep Learning for Sequential Data and NLP** (Albert School, G. Saint-Cirgue).
Scenario: AI engineers at a European legal-tech start-up. We build, evaluate and audit the
**NLP core of a contract-review assistant**:

1. **Classification** of every clause into one of the 100 LEDGAR types (fine-tuning of a BERT-class encoder);
2. **Retrieval** of similar clauses from a clause bank (sentence-transformer index), given a clause or a lawyer's
   question;
3. **Evidence** that it works (overall, per clause type, per kind of contract / party) + **bias audit** + a clear
   statement of prohibited uses.

- **Deliverables**: source code on GitHub on **23 October 2026** + PowerPoint presentation (**10 min + 5 min Q&A**)
  in front of the "CTO" (the jury).
- **Team**: Robin (stream A), Amandine (stream B), Brian (stream C) — see [`STRATEGY.md`](STRATEGY.md).
- **Language**: **English everywhere** — code, comments, docstrings, documentation, commit messages
  (Brian does not speak French).
- **Course materials**: `docs/ressources_methodologie/` (Sessions 1 to 6, PDF) — **gitignored**, never push them.
  Sessions 7 (seq2seq + bias audit) and 8 are not available yet.

## Data

**LEDGAR** (LexGLUE version), Hugging Face `coastalcph/lex_glue`, configuration `ledgar`.

| Split | Clauses | Years | Use |
|---|---:|---|---|
| `train` | 60,000 | 2016–2017 | training; **clause bank** for retrieval |
| `validation` | 10,000 | 2018 | **every** decision: hyperparameters, early stopping, model choice, thresholds |
| `test` | 10,000 | 2019 | **a single evaluation** per final model; retrieval queries |

- Columns: `text` (str), `label` (int 0-99); class names via `src/data.py::get_label_names()`.
- Single-label, **100 highly imbalanced classes** → the reference metric is **macro-F1** (S01, S04).
- **Chronological** split: test comes after train → temporal drift already included.
- To check during EDA: **near-duplicates** between train and test (identical boilerplate clauses) → measure their
  share and also report macro-F1 on the de-duplicated test set.
- `data/raw/` and `data/processed/` are gitignored. Default Hugging Face cache.

## Interface contracts (to be respected by all three streams)

These contracts let the three streams move forward in parallel. Any change is discussed as a team and logged in
`PROGRESS.md`.

1. **Data**: `from src.data import load_ledgar, get_label_names` — same splits, same label ids for everyone.
   Never re-split train/test.
2. **Metrics**: `from src.metrics import classification_metrics, paired_bootstrap_f1, mcnemar_test` — nobody
   re-implements their own macro-F1.
3. **Predictions**: every classification model writes
   `results/predictions/<run_id>_<split>.parquet` (gitignored) with the columns:
   `idx` (position in the split), `label`, `pred`, `proba_pred` (probability of the predicted class),
   `top5` (list of the 5 most probable classes). This is the input of the audit (stream C) and of the paired tests.
4. **Runs**: every run writes `results/runs/<run_id>/` with `config.json` (model, hyperparameters, seed, versions)
   and `metrics.json`:
   - one block per evaluated split: `"validation"` (always) and `"test"` (only for the single test evaluation of
     a final model), each with `macro_f1`, `micro_f1`, `accuracy`, `per_class_f1` (100 values, in label-id order);
   - at the top level: `train_time_s`, `n_trainable_params`, `n_total_params`, `peak_gpu_memory_mb` (`null`
     without CUDA) and `history` (`train`: training loss per logged step; `validation`: loss, macro-F1 and
     accuracy per evaluation). Keys that do not apply to a model are `null`.

   Example (`distilbert_smoke_20261018-1823`, abridged and rounded; a run with `--eval-test` adds a `"test"` block
   with the same keys as `"validation"`):

   ```json
   {
     "train_time_s": 59.2,
     "n_trainable_params": 67030372,
     "n_total_params": 67030372,
     "peak_gpu_memory_mb": null,
     "validation": {"macro_f1": 0.0272, "micro_f1": 0.1845, "accuracy": 0.1845, "per_class_f1": [0.0, 0.0, 0.0]},
     "history": {
       "train": [{"step": 2, "epoch": 0.03, "loss": 4.623}],
       "validation": [{"step": 63, "epoch": 1.0, "loss": 4.085, "macro_f1": 0.0272, "accuracy": 0.1845}]
     }
   }
   ```

   `run_id` = `<model>_<variant>_<YYYYMMDD-HHMM>`, e.g. `tfidf_logreg_base_20261015-1030`.
5. **Retrieval**: `search(query: str, k: int = 5) -> list[dict]` with `idx`, `text`, `label`, `label_name`, `score`.

## Methodological rules (from the course)

- **Three sets**: train to learn, validation to decide, test **only once**. Any decision taken on the test set makes
  the score optimistic (S01).
- **Macro-F1** as the main metric (imbalanced classes), plus **per-class** F1 and confusion matrix (S01).
- **Vocabulary / TF-IDF / normalisation** fitted on the **training set only** (S04).
- **Padding**: pads must never count — `pack_padded_sequence` or masked pooling for the LSTM, `attention_mask` for
  transformers, `ignore_index=-100` for token labels (S04, S05).
- **Always the pretrained model's own tokenizer** (`AutoTokenizer.from_pretrained(name)`) (S03).
- **Fine-tuning**: lr 2e-5 to 5e-5, 6-10 % warmup, linear decay, 2-4 epochs, weight decay 0.01, `max_length` chosen
  from the token-length distribution (EDA). Beware of catastrophic forgetting (S06).
- **LoRA**: lr 1e-4 to 1e-3, α = 2r, targets `query`/`value` (BERT) or `q_lin`/`v_lin` (DistilBERT); the
  classification head is fully trained (S06).
- **transformers 5**: `eval_strategy`, `processing_class`, `warmup_steps=0.1` (a ratio when < 1) (S06).
- **Comparing two models = paired test** on the same test set: McNemar (accuracy), paired bootstrap (macro-F1).
  Never two scores side by side without a test (S06).
- **Main comparison pre-registered** in the README: do not change it afterwards.
- **Sentence embeddings** normalised → cosine = dot product (S03).

## Reproducibility

- `src.utils.set_seed(42)` at the top of every notebook / script.
- Record the exact model versions (HF name + revision if possible) and library versions in `config.json`.
- Notebooks must run top to bottom on a fresh clone; paths go through `src/config.py`.
- No secrets in the repository: an optional `HF_TOKEN` goes in `.env` (gitignored), template in `.env.example`.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements.txt
jupyter lab
```

- Heavy training (BERT, LoRA, full LSTM): **Google Colab GPU** (T4 / L4 / A100).
- EDA, TF-IDF, retrieval on a subset, audit: locally.
- In Colab: `!git clone`, `%cd`, `!pip install -r requirements.txt`, then `import src`.

## Git

- `main` branch = always working. Work on `feat/<stream>-<topic>` branches (e.g. `feat/b-tfidf-baseline`),
  merged through a Pull Request reviewed by another member.
- Commits in English, present tense, short (`Add TF-IDF baseline and macro-F1 report`).
- **Milestone tags** required by the course: `m1`, `m2`, `m3`, `m4`, then `v1.0` at delivery.
- Never commit: data, checkpoints (`*.pt`, `*.safetensors`, `*.bin`), indexes, predictions, `.env`, course
  materials. **Metrics** (`metrics.json`, summary CSVs) and **figures** are versioned.
- Notebooks: clear large outputs before committing; keep the key tables and figures.

## Workflow

1. **Plan first**: any non-trivial task (3+ steps) → plan in `tasks/todo.md` before implementing.
2. **Verify**: a result is written in `PROGRESS.md` / README only if it was actually obtained.
   Never replace a result with an assumption.
3. **Learn**: after any correction, add a line to `tasks/lessons.md`
   (`[date] | what went wrong | rule to avoid it`).
4. **Update `PROGRESS.md`** after every significant step (what was done, numeric result, problem, solution,
   next action).
5. **Standard**: "would a staff engineer approve this?" Simplicity first, root causes, no hacks.

## Principles

- DRY: repeated logic goes into `src/`; notebooks tell the story and call `src/`.
- Never assume: check paths, APIs, parameter names (transformers 5 renamed several arguments).
- Always report limitations: a documented failure is worth more than an unverifiable success.
