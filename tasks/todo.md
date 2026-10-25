# Task list

> One task = an ID, an owner, a target date, a verifiable deliverable.
> Tick `[x]` **only** when the deliverable exists in the repository (or the result in `results/`).
> Every task has a final status: **done**, **dropped** / **cancelled** (with date and reason) or **open** (what is
> left before delivery). Initial plan and rationale: [`../STRATEGY.md`](../STRATEGY.md); what was actually done:
> [`../PROGRESS.md`](../PROGRESS.md).

## 0. Setup (everyone) — Wed 14/10

- [x] **S0.1** Repository structure, `README.md`, `CLAUDE.md`, `PROGRESS.md`, `STRATEGY.md` (Robin) — done 14/10
- [x] **S0.2** Main comparison pre-registered in the README (Robin) — done 14/10
- [x] **S0.3** GitHub repository created, first push (Robin) — done 14/10
- [x] **S0.4** Amandine and Brian added as collaborators (Robin) — done (both listed as repository collaborators)
- [x] **S0.5** Repository documentation translated into English (Robin) — done 18/10
- [x] **S0.6** Everyone clones the repo, creates their `.venv`, `pip install -r requirements-dev.txt` (everyone) —
  closed 25/10: the environments that produced the deliverables (Robin, Brian) are set up; not checked member by
  member
- [x] **S0.7** Colab GPU access + shared Drive folder for checkpoints (everyone) — closed 25/10: every GPU run was made
  from notebook 04 on Robin's Colab, checkpoints and prediction files in its Drive clone

## Stream A — Robin: Transformers, training, statistical comparison

| ID | Task | Deliverable | Date | Status |
|---|---|---|---|---|
| [x] A1 | Generic HF script: tokenisation (model's tokenizer), `Trainer`, `compute_metrics` (macro-F1), saving run + predictions in the common format | `src/models/transformer.py`, `scripts/train_transformer.py` | Thu 15/10 | done 18/10 |
| [x] A2 | DistilBERT smoke test on 2,000 examples (loss goes down, not a single predicted class) | `results/runs/distilbert_smoke_*` | Thu 15/10 | done 18/10 |
| [x] A3 | Set `max_length` (256 / 512) from the token lengths | decision in `PROGRESS.md` (256) | Thu 15/10 | done 18/10 |
| [x] A4 | DistilBERT on the full training set, **validation** evaluation | `results/runs/distilbert_full_20261019-0838/` | Fri 16/10 | done 19/10 |
| [x] A5 | Screening `nlpaueb/legal-bert-base-uncased` and/or `microsoft/deberta-v3-base` (validation) | screening table in `PROGRESS.md` | Sat 17/10 | done 19/10 (Legal-BERT; DeBERTa not run) |
| [x] A6 | Choice of the final model **on validation**, final run, **single test evaluation** | `results/runs/legal-bert_final_20261019-1523/`, test predictions | Sun 18/10 | done 19/10 |
| [x] A7 | LoRA (r = 4, 8, 16; α = 2r; query/value) on the same encoder | trade-off table: `results/lora_tradeoff.csv` | Mon 19/10 | done 20/10 with a stated deviation: **r = 8 only** (`legal-bert_lora-r8_20261020-0940`); r = 4 / 16 not run (GPU time) |
| [x] A8 | Paired tests: BERT vs TF-IDF (**main**), full vs LoRA, BERT vs LSTM | `scripts/paired_tests.py`, `results/paired_tests.csv` | Mon 19/10 | done: full vs LoRA r=8 (20/10); main pair by stream C (25/10, ΔF1 = +0.0281, **caveat: TF-IDF baseline not tuned on validation**); BERT vs LSTM dropped 25/10 (no LSTM) |
| [x] A9 | Calibration (ECE, reliability diagram) + "human review" confidence threshold (coverage vs precision); per-class precision at the recommended threshold (test, head / middle / tail by training-frequency terciles) | `results/calibration_*`, reliability / coverage / per-class figures | Tue 20/10 | done 19/10 (no temperature scaling: stated limitation) |
| [x] A10 | README results section, PR reviews, tag `v1.0`, test on a fresh clone | final README | Thu 22/10 | README done 25/10 (final consistency review); PR reviews dropped 19/10 (team works on `main`, no PRs); end-to-end check 25/10 (with T3): demo script, notebook 07 and 71 tests re-run locally, notebooks run on a fresh clone by Brian earlier, 01 / 02 / 05 not re-executed locally, 04 on Colab; **tag `v1.0` still to be created** |

## Stream B — Amandine: Data, baselines, RNN

| ID | Task | Deliverable | Date | Status |
|---|---|---|---|---|
| [x] B1 | Validate `src/data.py` on the real data (sizes 60k/10k/10k, 100 labels, names) — **priority, everyone depends on it** | `load_ledgar()` tested | Thu 15/10 | done 18/10 (`python -m src.data`, `tests/test_data.py`) |
| [x] B2 | EDA: class distribution (log scale), split by year, examples per class | `notebooks/01_eda_baseline.ipynb` | Thu 15/10 | done 23/10: class imbalance and exact-text duplicates; split by year not possible (the shared loader exposes no row-level years) |
| [x] B3 | Lengths in words and WordPiece tokens (p50, p95, p99) → sent to A for `max_length` | numbers in `PROGRESS.md` | Thu 15/10 | done 18/10 by Robin (A3, `scripts/token_lengths.py`, WordPiece tokens); word counts not recorded |
| ~~B4~~ | Train/test near-duplicates (exact normalised text + TF-IDF similarity > 0.95): share of the test set affected | number + list of de-duplicated `idx` | Fri 16/10 | **cancelled 25/10 (out of scope)**: not done before the code freeze; notebook 01 checks exact duplicates only; no de-duplicated test score, listed as a limitation in the model card |
| [x] B5 | **TF-IDF + LogReg** baseline (1-2 n-grams, C tuned on validation), val/test macro-F1, predictions in the common format → **tag m1** | `results/runs/tfidf-logreg_baseline_20261023/` + predictions | Fri 16/10 | done 23/10 with a stated deviation: **C = 1.0 fixed, not tuned** (grid not run, ≈ 187 s and 5.4 GB RAM per C value); val 0.7666 / test 0.7625 macro-F1 |
| ~~B6~~ | **LSTM / BiLSTM**: train vocabulary (`min_freq`), `<pad>`=0 `<unk>`=1, packing, GloVe, PyTorch loop, clipping, early stopping on validation | `src/models/lstm.py`, `notebooks/03_lstm_classifier.ipynb` | Sat 17/10 | **dropped 25/10 (team decision)**: not delivered in time; notebook slot 03 reused for the error analysis |
| [x] B7 | Error analysis: per-class F1 vs support, top 10 confused pairs, 20 errors read and commented → **tag m3** | `notebooks/03_error_analysis.ipynb`, `src/analysis.py` | Sun 18/10 | done 25/10 by Robin, without the LSTM, on Legal-BERT vs TF-IDF (together with B9); the 20 error tags were reviewed by Robin |
| [x] B8 | Unified comparison table of all models + figures (macro-F1, per-class F1) | `results/summary_classification.csv`, `results/figures/summary_*.png` | Mon 19/10 | done 25/10 by Robin: TF-IDF, DistilBERT (validation only), Legal-BERT final, LoRA r=8 |
| [x] B9 | Error analysis of the final model (stream A): which classes does BERT fix compared with TF-IDF? | `results/error_analysis_*.csv`, `results/figures/error_analysis_*.png`, findings in notebook 03 | Tue 20/10 | done 25/10 by Robin |

## Stream C — Brian: Retrieval, audit, model card

| ID | Task | Deliverable | Date | Status |
|---|---|---|---|---|
| [x] C1 | `all-MiniLM-L6-v2` index over the bank (train), normalised and cached embeddings, `search(query, k)` | `src/retrieval/dense.py` | Thu 15/10 | done 19/10 |
| [x] C2 | 10 queries (pasted clauses + questions) checked by hand → **tag m2** | `notebooks/02_retrieval_index.ipynb` | Fri 16/10 | done: notebook run 23/10, the 10 manual checks completed by Brian |
| [x] C3 | Draft 28 lawyer questions; retain as illustrative future queries (no expert judgments, not evaluated) | `eval/retrieval_queries.jsonl` | Fri 16/10 | done 20/10 (drafts only, not evaluated) |
| [x] C4 | Compare MiniLM and BGE-small on a stratified 5% test-query sample, using same-label relevance and P@5 / MRR@10 / nDCG@10 | `results/retrieval_summary.csv` | Sat 17/10 | done 23/10 |
| [x] C5 | Evaluate BM25 and MiniLM+BM25 RRF on the same sampled test queries | `src/retrieval/hybrid.py`, `notebooks/05_retrieval_eval.ipynb` | Sun 18/10 | done 23/10 |
| [x] C6 | Classifier slice metrics: training-frequency groups, clause length, and explicitly labelled rule-detected party-role proxies | `src/audit/slices.py`, `notebooks/06_bias_audit.ipynb` | Sun 18/10 | done 25/10 |
| [x] C7 | One-feature counterfactual stress tests; bounded to at most 1,000 reproducibly sampled test clauses | `eval/counterfactuals/`, `src/audit/perturb.py` | Mon 19/10 | done 25/10 |
| [x] C8 | Focused audit on final classifier: per-slice F1 and counterfactual prediction/confidence changes; no retrieval or synthetic OOD audit claims | `results/audit/`, `notebooks/06_bias_audit.ipynb` | Tue 20/10 | done 25/10 |
| [x] C9 | Model card: intended / prohibited uses, measured results, sample sizes, and limitations | `docs/model_card.md` | Wed 21/10 | done 25/10 |

## Shared

- [ ] **T1** Slides (each stream provides its 2-3 slides) — Wed 21/10 — **open**
- [ ] **T2** Timed 10-min rehearsal + likely CTO questions — Thu 22/10 — **open**
- [x] **T3** Code freeze, notebooks run on a fresh clone, tag `v1.0` — Thu 22/10 23:59 — documentation frozen 25/10
  (final consistency review); end-to-end check 25/10 (Robin): `scripts/demo.py` (clause and `--question`),
  notebook 07 top to bottom and the test suite (71 passed) re-run locally without error; notebooks run on a fresh
  clone by Brian earlier; not re-run: 01, 02, 05 (not re-executed locally), 04 (runs on Colab), 06 (executed by
  Brian); **tags `m1`-`m4` and `v1.0` still to be created**
- [x] **T4** (Bonus) Demo "clause → type + confidence + 5 closest clauses" — **done 25/10** (Robin): `src/pipeline.py`
  (`ClauseReviewer`), `scripts/demo.py`, `notebooks/07_demo.ipynb`, `tests/test_pipeline.py`; no retraining
- ~~**T5**~~ Integrate the content of Session 7 (bias audit) into the C6-C9 plan — **cancelled 25/10**: the Session 7
  material was not available before the code freeze; the audit follows the brief
