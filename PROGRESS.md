# Progress — Clause Review Assistant (LEDGAR)

> This file is the **operational log** of the project. It is updated after every significant task.
> `CLAUDE.md` holds the durable rules and decisions; `STRATEGY.md` the plan; `PROGRESS.md` the actual state,
> the results, the problems met and the next actions.

## 1. Overall status

**Current status:** 🟡 Repository initialised and pushed to GitHub (structure, documentation, pre-registered main
comparison, shared modules `src/config.py`, `src/data.py`, `src/metrics.py`, `src/utils.py`), documentation
translated into English. `load_ledgar()` checked on the real data (60,000 / 10,000 / 10,000, 100 labels).
Stream A: generic fine-tuning script ready, DistilBERT smoke test passed locally, `max_length` = 256 decided
(branch `feat/a-hf-finetuning`).
**Next main step: run in parallel EDA + TF-IDF (B), full DistilBERT on Colab (A4) and MiniLM index (C).**

**Deadlines:**
- code freeze: **Thursday 22 October 2026, 23:59** (internal margin);
- GitHub delivery: **Friday 23 October 2026**;
- presentation: 10 min + 5 min Q&A (date to be confirmed).

## 2. Progress table

### Setup
- [x] Brief analysed, course (Sessions 1 to 6) read, milestones `m1` to `m4` identified.
- [x] Dataset identified: LEDGAR (LexGLUE), 100 classes, chronological split 60k / 10k / 10k.
- [x] Structure and documentation (`README`, `CLAUDE`, `PROGRESS`, `STRATEGY`, `tasks/`, `docs/model_card.md`).
- [x] Main comparison pre-registered in the README (14/10/2026).
- [x] Shared modules: `src/config.py`, `src/data.py`, `src/metrics.py`, `src/utils.py`.
- [x] GitHub repository created and first push (`1RLevasseurAlbertSchool/Projet_Deep_Learning`).
- [x] Documentation translated into English (18/10/2026).
- [ ] Team environments installed (local + Colab GPU).

### m1 — Data and baseline (Amandine)
- [ ] `load_ledgar()` validated on the real data.
- [ ] Class distribution, years, lengths (words / tokens).
- [ ] Train/test near-duplicates measured.
- [ ] TF-IDF + LogReg: validation and test macro-F1.
- [ ] Tag `m1`.

### m2 — Retrieval index (Brian)
- [ ] `all-MiniLM-L6-v2` index over the clause bank.
- [ ] 10 queries checked by hand.
- [ ] Tag `m2`.

### m3 — LSTM and error analysis (Amandine)
- [ ] LSTM / BiLSTM with packing and GloVe.
- [ ] Per-class F1, most confused classes.
- [ ] Tag `m3`.

### m4 — Transformers (Robin)
- [x] Generic fine-tuning script (`src/models/transformer.py`, `scripts/train_transformer.py`) + Colab notebook 04.
- [x] `max_length` chosen from the token lengths (256).
- [x] DistilBERT smoke test (local, MPS).
- [ ] Encoder screening on the validation set.
- [ ] Final full fine-tuning run + single test evaluation.
- [ ] LoRA runs + trade-off table.
- [ ] Paired tests (main comparison + secondary ones).
- [ ] Tag `m4`.

### Retrieval and audit (Brian)
- [ ] Automatic retrieval evaluation + lawyer question set.
- [ ] Hybrid BM25 + dense.
- [ ] Per-slice audit + counterfactuals + out-of-distribution.
- [ ] Model card.

### Delivery
- [ ] README results section filled with real results.
- [ ] Notebooks run on a fresh clone.
- [ ] Slides + rehearsal.
- [ ] Tag `v1.0`.

## 3. Results observed so far

### Token lengths of the training clauses (A3, 18/10)

`python scripts/token_lengths.py --model distilbert-base-uncased` — 60,000 train clauses, DistilBERT's own
uncased WordPiece tokenizer (same vocabulary as `bert-base-uncased`), special tokens included, no truncation:

| mean | p50 | p90 | p95 | p99 | max | > 128 | > 256 | > 512 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 141.9 | 106 | 291 | 381 | 589 | 1,771 | 41.07 % | 13.51 % | 1.80 % |

### DistilBERT smoke test (A2, 18/10) — sanity check, not a reportable score

Run `distilbert_smoke_20261018-1823`: first 2,000 examples of each split, 1 epoch (63 steps), `max_length` 128,
batch 32, lr 5e-5, local MacBook Pro M1 (MPS), 59 s of training.
Training loss 4.62 at step 2 (≈ ln 100 = 4.61, uniform guess) → 4.16 on average over the last 5 logged points;
validation loss 4.085, validation accuracy 0.1845, macro-F1 0.0272. **10 distinct classes predicted** on the
validation subset (98 classes present; the top 3 predicted classes take 36 %, 31 % and 17 % of the predictions):
the pipeline learns, the model is far from converged, as expected after 63 steps. Re-running with seed 42 gave
identical numbers.

Published reference points (LexGLUE, LEDGAR test set — not to be confused with our results):

```text
Method          µ-F1   m-F1
TF-IDF SVM      87.0   81.4
BERT            87.6   81.8
RoBERTa         87.9   82.3
DeBERTa         88.2   83.1
Legal-BERT      88.2   83.0
```

## 4. Problems met and solutions

### `python -m src.data` fails outside the project environment (18/10)

**Symptom:** in the anaconda base environment, `python -m src.data` started downloading the `case_hold`
configuration, then crashed on an SSL error.
**Cause:** that environment had `datasets` 2.12. The Hub repository `coastalcph/lex_glue` is now parquet-only, with
its configurations declared in the README YAML (`configs:`), which `datasets` < 2.14 does not read: the `ledgar`
config name was ignored. The SSL error was a flaky network on top.
**Solution:** project `.venv` created from `requirements.txt` (`datasets` 5.1.0, `transformers` 5.20.1,
`torch` 2.14.1): 60,000 / 10,000 / 10,000 and 100 labels. No change to `src/data.py`.
**Status:** ✅ solved.

### `src/models/` excluded by `.gitignore` (14/10)

**Symptom:** `src/models/__init__.py` was missing from the first commit.
**Cause:** the unanchored pattern `models/` matched every folder named `models`.
**Solution:** root-level patterns anchored (`/models/`, `/checkpoints/`, `/outputs/`, `/out/`), commit amended
before the first push.
**Status:** ✅ solved.

## 5. Next immediate action

1. Robin: add Amandine and Brian as collaborators on GitHub (if not done yet).
2. Robin: push `feat/a-hf-finetuning`, open the PR, then run A4 (full DistilBERT, validation only) with
   `notebooks/04_transformer_finetuning.ipynb` on Colab.
3. Team: confirm the `metrics.json` layout per split (see the 18/10 log entry) before B5 writes its first run.
4. Parallel work: B2-B5 (EDA, TF-IDF), C1-C3 (index, queries).

## 6. Rule for updating this file

After every significant step, state: what was done; the result obtained; the numbers or metrics; the problems met;
the solution adopted; the next action.

Never replace a real result with an assumption. A task stays `[ ]` until it has actually been run or confirmed.
Any deviation from the pre-registered comparison is logged here.

## 7. Chronological log

### 18 October 2026 — stream A: fine-tuning script, token lengths, smoke test (Robin)

- **Environment check**: `git` clean, `main` level with `origin/main`; `python -m pytest -q` → 4 passed;
  `python -m src.data` → 60,000 / 10,000 / 10,000 and 100 labels, in the project `.venv` only (see section 4).
- **A1**: `src/models/transformer.py` (`token_length_stats`, `train_and_evaluate`), `scripts/train_transformer.py`,
  `scripts/token_lengths.py`, `notebooks/04_transformer_finetuning.ipynb` (Colab: Drive, clone with the
  `GITHUB_TOKEN` secret, one cell per run), `tests/test_transformer.py` (10 tests in total pass in 8 s, CPU,
  no model download). transformers 5 API: `eval_strategy`, `processing_class`, `warmup_steps=0.1` (ratio),
  dynamic padding with `DataCollatorWithPadding`, best epoch on validation macro-F1, test predicted only with
  `--eval-test`.
- **A2**: smoke test passed (numbers in section 3): the loss goes down, 10 distinct classes predicted.
- **A3 — decision: `max_length` = 256** (default of the script). Justification from the token lengths
  (section 3): 256 keeps 86.5 % of the training clauses whole (p50 = 106, p90 = 291), 512 would keep 98.2 %.
  With batches of 32, a batch has a 1 % chance of containing no clause above 256 tokens (0.865^32), so at
  `max_length` 256 almost every batch is padded to 256; at 512, 44 % of the batches would reach 512 tokens and
  attention cost grows quadratically with length. Roughly doubling the per-step cost to keep 11.7 % more clauses
  whole is not worth it within our Colab budget (several encoders + LoRA). The truncated clauses keep their
  first ~254 word pieces. Limits: other encoders have other vocabularies (re-check with `scripts/token_lengths.py`,
  cell in notebook 04); a 512 run of the final encoder, compared on validation only, is optional if GPU time
  remains.
- **Interface note (to confirm with the team)**: a run evaluated on several splits writes the contract metrics
  (`macro_f1`, `micro_f1`, `accuracy`, `per_class_f1`) under one key per split (`metrics["validation"]`,
  `metrics["test"]`), next to `train_time_s`, `n_trainable_params`, `n_total_params`, `peak_gpu_memory_mb`
  (None without CUDA) and `history` (loss / macro-F1 curves). Contract #4 of `CLAUDE.md` updated accordingly,
  with a JSON example.
- Revision of the pretrained weights recorded in `config.json` from the HF cache snapshot
  (`transformers` 5.20 no longer sets `config._commit_hash`).
- Next: A4 on Colab (full DistilBERT, validation only).

### 18 October 2026 — documentation translated into English (Robin)

- One team member (Brian) does not speak French: the whole repository is now in English — documentation, code
  comments, docstrings, tests.
- Files renamed: `AVANCEMENT.md` → `PROGRESS.md`, `STRATEGIE.md` → `STRATEGY.md`; all links updated.
- `CLAUDE.md` rule changed: English everywhere (code, docs, commits).
- No change to the plan, the interface contracts or the pre-registered comparison.

### 14 October 2026 — project initialisation (Robin)

- Brief and course materials (Sessions 1 to 6) analysed. Milestones expected by the instructor:
  `m1` (TF-IDF + LogReg, macro-F1), `m2` (sentence-embedding index), `m3` (LSTM + error analysis),
  `m4` (fine-tuned BERT, LoRA, paired test of a comparison written in advance). Bias audit announced for Session 7.
- Dataset: LEDGAR / LexGLUE (`coastalcph/lex_glue`, config `ledgar`) — 60k / 10k / 10k, 100 classes,
  train 2016-2017, validation 2018, test 2019.
- Decision: LEDGAR being in English, CamemBERT is not used; English BERT-class candidates (DistilBERT,
  BERT, Legal-BERT, DeBERTa-v3), final choice on the validation set.
- Pre-registered main comparison: best fine-tuned encoder vs TF-IDF + LogReg, test macro-F1,
  paired bootstrap + McNemar.
- Organisation in three streams: A (Robin, transformers), B (Amandine, data / baselines / LSTM),
  C (Brian, retrieval / audit). Schedule in `STRATEGY.md`, tasks in `tasks/todo.md`.
- Course materials moved to `docs/ressources_methodologie/` (gitignored, not redistributed).
- Open point: the data Colab notebook provided by the instructor could not be read automatically;
  check that it loads the same source (`coastalcph/lex_glue`, `ledgar`) and the same splits.
- First commit `957be2e` pushed to `origin/main`.
