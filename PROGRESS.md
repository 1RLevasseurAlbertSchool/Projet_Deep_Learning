# Progress — Clause Review Assistant (LEDGAR)

> This file is the **operational log** of the project. It is updated after every significant task.
> `CLAUDE.md` holds the durable rules and decisions; `STRATEGY.md` the plan; `PROGRESS.md` the actual state,
> the results, the problems met and the next actions.

## 1. Overall status

**Current status:** 🟡 Repository initialised and pushed to GitHub (structure, documentation, pre-registered main
comparison, shared modules `src/config.py`, `src/data.py`, `src/metrics.py`, `src/utils.py`), documentation
translated into English. `load_ledgar()` checked on the real data (60,000 / 10,000 / 10,000, 100 labels).
Stream A: full DistilBERT (A4) and Legal-BERT (A5) fine-tuned on Colab, validation macro-F1 0.7863 and 0.7930;
`legal-bert_full_20261019-0916` selected on validation and evaluated once on test (A6): **test macro-F1 0.7905,
accuracy 0.8717**. Team now works directly on `main` (no feature branches).
**Next main step: TF-IDF + LogReg predictions (B5) for the pre-registered main comparison (A8); LoRA (A7) and
MiniLM index (C) in parallel.**

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
- [x] Single test evaluation from the best checkpoint (`scripts/evaluate_checkpoint.py`, single-use guard).
- [x] Encoder screening on the validation set (DistilBERT vs Legal-BERT, 19/10).
- [x] Final full fine-tuning run + single test evaluation (`legal-bert_final_20261019-1523`, 19/10).
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

### Encoder screening on validation (A4, A5, 19/10)

Colab, Tesla T4, fp16, full training set (60,000), `max_length` 256, 3 epochs, 10 % warmup, weight decay 0.01,
seed 42, validation = 10,000 clauses (2018). Numbers read from `results/runs/<run_id>/metrics.json` and
`config.json`.

| Run | Encoder | lr / batch | Steps | Val macro-F1 | Val accuracy | Best epoch | Train time | Trainable / total params | Peak GPU memory |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `distilbert_full_20261019-0838` | `distilbert-base-uncased` | 5e-5 / 32 | 5,625 | 0.7863 | 0.8660 | 3 of 3 | 1,227.5 s (20.5 min) | 67,030,372 / 67,030,372 | 2,536.7 MB |
| `legal-bert_full_20261019-0916` | `nlpaueb/legal-bert-base-uncased` | 3e-5 / 16 | 11,250 | **0.7930** | **0.8729** | 3 of 3 | 2,688.2 s (44.8 min) | 109,559,140 / 109,559,140 | 3,106.2 MB |
| Legal-BERT − DistilBERT | | | | +0.0067 | +0.0069 | | × 2.19 | × 1.63 | +569.5 MB |

Validation curve per epoch (evaluation at the end of each epoch):

| Epoch | DistilBERT loss | DistilBERT macro-F1 | DistilBERT accuracy | Legal-BERT loss | Legal-BERT macro-F1 | Legal-BERT accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.622 | 0.7197 | 0.8355 | 0.608 | 0.7251 | 0.8395 |
| 2 | 0.534 | 0.7566 | 0.8563 | 0.524 | 0.7720 | 0.8613 |
| 3 | 0.507 | 0.7863 | 0.8660 | 0.501 | 0.7930 | 0.8729 |

- Both runs still improve at the last epoch (macro-F1 +0.0296 and +0.0209 from epoch 2 to 3, validation loss still
  going down): best epoch = last epoch, the models are not converged.
- Per class, Legal-BERT is better on 57 classes, worse on 30, equal on 13. Two classes have F1 = 0 for both
  models: 8 `Assigns` (3 validation clauses) and 14 `Books` (**no** validation clause: its F1 is 0 by construction,
  `zero_division=0` over the 100 labels, which caps every validation macro-F1 at 0.99 and does not change the
  ranking).
- The two runs differ by encoder **and** by recipe (lr, batch size): the screening compares the two set-ups as run,
  not the encoders alone. The +0.0067 gap is small and was not tested (the validation predictions are on Drive);
  the pre-registered selection rule does not require a test (best validation macro-F1).
- `microsoft/deberta-v3-base` (notebook 04, section 8) was not run.

### Single test evaluation of the selected encoder (A6, 19/10)

Run `legal-bert_final_20261019-1523` = `scripts/evaluate_checkpoint.py --source-run legal-bert_full_20261019-0916`
(best checkpoint `checkpoint-11250`, epoch 3, on Drive), Tesla T4, fp16, `max_length` 256, no retraining.

| Split | Clauses | Macro-F1 | Micro-F1 = accuracy | Classes with F1 = 0 |
|---|---:|---:|---:|---|
| Validation (2018), re-predicted | 10,000 | 0.7930 | 0.8729 | 8 `Assigns`, 14 `Books` |
| **Test (2019), evaluated once** | 10,000 | **0.7905** | **0.8717** | 8 `Assigns` (4 clauses), 14 `Books` (2 clauses) |

- Re-predicted validation = source run **exactly**: macro-F1, micro-F1, accuracy and the 100 per-class F1 are
  identical (largest difference 0.0) → the loaded weights and tokenisation are the ones selected on validation.
- Test − validation: macro-F1 −0.0024, accuracy −0.0012 (one year of drift, 2018 → 2019).
- Gap to the LexGLUE reference (Legal-BERT, LEDGAR test): **−3.95 pt macro-F1** (79.05 vs 83.0), −1.03 pt micro-F1
  (87.17 vs 88.2). Not a paired comparison: published numbers, other runs. Likely causes:
  1. `max_length` 256 instead of 512: 13.5 % of the training clauses exceed 256 tokens with the uncased BERT
     WordPiece vocabulary (A3; Legal-BERT has its own vocabulary, its token lengths were not recorded) and lose
     their end;
  2. 3 epochs while still improving: validation macro-F1 +0.0209 between epochs 2 and 3, best epoch = last one.

  The macro-F1 gap being about 4 times the micro-F1 gap suggests that the shortfall sits mostly on the rare
  classes.

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
2. Everyone: work on `main`, `git pull` before starting (see `CLAUDE.md`, Git section).
3. Amandine: B5 (TF-IDF + LogReg, predictions in the common format) — it unblocks the pre-registered main
   comparison A8 (Model A = `legal-bert_final_20261019-1523`).
4. Robin: copy `results/predictions/legal-bert_final_20261019-1523_{validation,test}.parquet` from the Drive clone
   (gitignored, absent from local checkouts) to the machines running A8, A9 and C8.
5. Robin: A7 LoRA on `nlpaueb/legal-bert-base-uncased` with the same recipe (`max_length` 256, 3 epochs) so that
   full vs LoRA stays comparable.
6. Team: confirm the `metrics.json` layout per split (see the 18/10 log entry) before B5 writes its first run.
7. Parallel work: B2-B5 (EDA, TF-IDF), C1-C3 (index, queries).

## 6. Rule for updating this file

After every significant step, state: what was done; the result obtained; the numbers or metrics; the problems met;
the solution adopted; the next action.

Never replace a real result with an assumption. A task stays `[ ]` until it has actually been run or confirmed.
Any deviation from the pre-registered comparison is logged here.

## 7. Chronological log

### 19 October 2026 — Git workflow: work directly on `main` (Robin)

- **Decision**: no new branches. `git pull` on `main` before starting, small commits, push after review
  (`CLAUDE.md`, Git section). The documented workflow (one feature branch + reviewed PR per task) was not the one
  actually followed: `feat/a-eval-checkpoint` was merged into `main` locally.
- **Notebook 04, section 2**: the Drive clone is a read-only mirror, updated with `checkout -f <BRANCH>` +
  `reset --hard origin/<BRANCH>` instead of `checkout` + `pull --ff-only`. Colab writes `results/runs/<run_id>/`
  untracked in the Drive clone; once those JSON files are committed on `main` from a laptop, `pull --ff-only`
  aborts ("untracked working tree files would be overwritten by merge"). Reproduced on a throwaway repository:
  `pull --ff-only` aborts; `checkout -f` + `reset --hard` succeeds, replaces the colliding files with the committed
  copies, keeps the other untracked runs and discards local edits to tracked files. `results/predictions/`
  (gitignored) is not touched.

### 19 October 2026 — stream A: selection on validation and single test evaluation (A6, Robin)

- **Selection** by the pre-registered rule (README, Model A: selected on the validation set only; criterion: best
  validation macro-F1, nothing else): `legal-bert_full_20261019-0916` (0.7930) over `distilbert_full_20261019-0838`
  (0.7863). Neither screening run has test metrics (`eval_test: false`).
- **Run** `legal-bert_final_20261019-1523` (Colab T4, fp16) =
  `scripts/evaluate_checkpoint.py --source-run legal-bert_full_20261019-0916`, no retraining. `config.json`:
  `source_run` = the selected run, `best_checkpoint` = `.../checkpoint-11250` (epoch 3), `max_length` 256,
  `eval_test: true`.
- **Test (2019, 10,000 clauses, evaluated once)**: macro-F1 **0.7905**, micro-F1 = accuracy **0.8717**; F1 = 0 for
  classes 8 `Assigns` and 14 `Books`.
- **Validation re-predicted = source run exactly**: macro-F1 0.7930, micro-F1 = accuracy 0.8729, and the 100
  per-class F1 identical (largest difference 0.0). Training cost and curves copied from the source run (same
  weights): 2,688.2 s, 109,559,140 trainable / total parameters, peak GPU memory 3,106.2 MB.
- **Gap to LexGLUE** (Legal-BERT, 83.0 test macro-F1, 88.2 micro-F1): −3.95 pt macro-F1, −1.03 pt micro-F1. Likely
  causes: `max_length` 256 instead of 512 (13.5 % of the training clauses truncated, A3) and 3 epochs while
  validation macro-F1 was still improving (+0.0209 between epochs 2 and 3). Details in section 3.
- Predictions `results/predictions/legal-bert_final_20261019-1523_{validation,test}.parquet` are in the Drive clone
  (gitignored), not in this checkout.
- **Consequence**: the test set is now used for this model. A later run (512 tokens, more epochs) can only be
  compared on validation; Model A of the main comparison stays `legal-bert_final_20261019-1523`.
- The main comparison (A8) waits for stream B's TF-IDF + LogReg predictions (B5); the pre-registered README block
  is unchanged.

### 19 October 2026 — stream A: Legal-BERT screening on validation (A5, Robin)

- Run `legal-bert_full_20261019-0916`: `nlpaueb/legal-bert-base-uncased` (revision `15b570cb`), full training set,
  `max_length` 256, 3 epochs, lr 3e-5, batch 16, 10 % warmup, weight decay 0.01, seed 42, Colab T4, fp16,
  11,250 steps.
- Validation per epoch: macro-F1 0.7251 → 0.7720 → **0.7930**; accuracy 0.8395 → 0.8613 → 0.8729; loss
  0.608 → 0.524 → 0.501. Training loss 4.611 (step 225) → 0.329 (step 11,250). Best epoch 3 (`checkpoint-11250`).
- Cost: 2,688.2 s (44.8 min), 109,559,140 trainable / total parameters, peak GPU memory 3,106.2 MB.
- vs DistilBERT (A4): +0.0067 macro-F1, +0.0069 accuracy, for 2.19 × the training time, 1.63 × the parameters and
  +569.5 MB of GPU memory. Screening table in section 3.
- The token lengths with the Legal-BERT tokenizer (notebook 04, section 8) are not recorded in the repository.
  `microsoft/deberta-v3-base` was not run.

### 19 October 2026 — stream A: full DistilBERT on Colab (A4, Robin)

- Colab smoke check first: `distilbert_smoke_20261019-0836` (T4, fp16, 2,000 examples per split, 1 epoch,
  `max_length` 128): 18.9 s, validation macro-F1 0.0317, accuracy 0.185, peak GPU memory 1,712.3 MB — same order
  as the local MPS run (0.0272 / 0.1845, fp32). The Colab pipeline works; not a reportable score.
- Run `distilbert_full_20261019-0838`: `distilbert-base-uncased` (revision `12040acc`), 60,000 training clauses,
  `max_length` 256, 3 epochs, lr 5e-5, batch 32, 10 % warmup, weight decay 0.01, seed 42, Colab T4, fp16,
  5,625 steps.
- Validation per epoch: macro-F1 0.7197 → 0.7566 → **0.7863**; accuracy 0.8355 → 0.8563 → 0.8660; loss
  0.622 → 0.534 → 0.507. Training loss 4.548 (step 113) → 0.343 (step 5,537). Best epoch 3 (`checkpoint-5625`).
- Cost: 1,227.5 s (20.5 min), 67,030,372 trainable / total parameters, peak GPU memory 2,536.7 MB.
- Colab library versions for all runs of 19/10 (in each `config.json`): Python 3.13.15, torch 2.11.0+cu130,
  transformers 5.18.0, datasets 4.8.5 — not the local `.venv` ones (torch 2.14.1, transformers 5.20.1,
  datasets 5.1.0).

### 19 October 2026 — stream A: single test evaluation from the best checkpoint (Robin)

- **Problem**: A6 was planned as a retrain of the chosen encoder with `--variant final --eval-test`. That costs
  one more full GPU run, and the retrained weights are not exactly the ones selected on validation (GPU
  non-determinism, fp16): the test score would belong to a different model than the one chosen.
- **Solution**: `scripts/evaluate_checkpoint.py --source-run <run_id>` (logic in
  `src/models/transformer.py::evaluate_checkpoint`). It reads the source `config.json`, reloads the tokenizer and
  the model from `best_checkpoint` (repo-relative path, or absolute for the Drive checkpoints folder), predicts
  validation and test with the same tokenisation (truncation at the source `max_length`, dynamic padding) and the
  same precision as the evaluations made during training, and writes a new run `<model>_final_<timestamp>`:
  `config.json` = source config + `source_run` (`best_checkpoint` included); `metrics.json` = training cost and
  curves of the source run (same weights) + fresh `validation` and `test` blocks.
- **Single-use guard** (`src/utils.py::runs_with_test`): refuses when the source run itself (trained with
  `--eval-test`) or any run with `source_run` = it already has test metrics; `--force` overrides it with a
  warning explaining why the test set is evaluated only once.
- Shared code moved to `src/models/transformer.py` and used by both scripts: `tokenize_ledgar`, `print_metrics`,
  the test-set warning.
- Notebook 04, section 9: `FINAL_RUN = ""` + the same `assert` guard + a call to the script (no retraining).
- **Tests**: 4 guard tests (CPU, no model download); `python -m pytest -q` → 14 passed.
- **End-to-end check** (local, MPS) on the smoke checkpoint of `distilbert_smoke_20261018-1823`, outputs in a
  temporary folder and the test split replaced by the validation subset (the real test set was not touched):
  validation macro-F1 0.0272 = source run; the 2,000 validation predictions are identical to the smoke run's
  (largest probability difference 0.0); a second call is refused by the guard.
- Next: A4 / A5 on validation only, then A6 = `evaluate_checkpoint.py --source-run <run chosen on validation>`.

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
