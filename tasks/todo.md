# Task list

> One task = an ID, an owner, a target date, a verifiable deliverable.
> Tick `[x]` **only** when the deliverable exists in the repository (or the result in `results/`).
> Strategy and rationale: [`../STRATEGY.md`](../STRATEGY.md).

## 0. Setup (everyone) — Wed 14/10

- [x] **S0.1** Repository structure, `README.md`, `CLAUDE.md`, `PROGRESS.md`, `STRATEGY.md` (Robin)
- [x] **S0.2** Main comparison pre-registered in the README (Robin)
- [x] **S0.3** GitHub repository created, first push (Robin)
- [ ] **S0.4** Amandine and Brian added as collaborators (Robin)
- [x] **S0.5** Repository documentation translated into English (Robin, 18/10)
- [ ] **S0.6** Everyone clones the repo, creates their `.venv`, `pip install -r requirements-dev.txt` (everyone)
- [ ] **S0.7** Check everyone's Colab GPU access + shared Drive folder for checkpoints (everyone)

## Stream A — Robin: Transformers, training, statistical comparison

| ID | Task | Deliverable | Date | Depends on |
|---|---|---|---|---|
| [x] A1 | Generic HF script: tokenisation (model's tokenizer), `Trainer`, `compute_metrics` (macro-F1), saving run + predictions in the common format | `src/models/transformer.py`, `scripts/train_transformer.py` | Thu 15/10 | B1 |
| [x] A2 | DistilBERT smoke test on 2,000 examples (loss goes down, not a single predicted class) | `results/runs/distilbert_smoke_*` | Thu 15/10 | A1 |
| [x] A3 | Set `max_length` (256 / 512) from the token lengths | decision in `PROGRESS.md` | Thu 15/10 | B3 |
| [x] A4 | DistilBERT on the full training set, **validation** evaluation | run + loss / macro-F1 curves | Fri 16/10 | A2 |
| [x] A5 | Screening `nlpaueb/legal-bert-base-uncased` and/or `microsoft/deberta-v3-base` (validation) | screening table | Sat 17/10 | A4 |
| [x] A6 | Choice of the final model **on validation**, final run, **single test evaluation** | `results/runs/<final>/`, test predictions | Sun 18/10 | A5 |
| [x] A7 | LoRA (r = 4, 8, 16; α = 2r; query/value) on the same encoder — **deviation: r = 8 only** (`legal-bert_lora-r8_20261020-0940`, 20/10); r = 4 / 16 not run unless GPU time allows | trade-off table (params, time, memory, adapter size, macro-F1): `results/lora_tradeoff.csv` | Mon 19/10 | A6 |
| [ ] A8 | Paired tests: BERT vs TF-IDF (**main**), full vs LoRA, BERT vs LSTM — script and full vs LoRA r=8 done (20/10); main and LSTM pairs wait for stream B's test predictions | `scripts/paired_tests.py`, `results/paired_tests.csv` → **tag m4** | Mon 19/10 | A6, A7, B4, B5, B6 |
| [x] A9 | Calibration (ECE, reliability diagram) + "human review" confidence threshold (coverage vs precision); per-class precision at the recommended threshold (test, head / middle / tail by training-frequency terciles) | figure + table; `results/calibration_per_class_<run_id>.csv` + `precision_per_class_at_threshold.png` | Tue 20/10 | A6 |
| [ ] A10 | README results section, PR reviews, tag `v1.0`, test on a fresh clone | final README | Thu 22/10 | all |

## Stream B — Amandine: Data, baselines, RNN

| ID | Task | Deliverable | Date | Depends on |
|---|---|---|---|---|
| B1 | Validate `src/data.py` on the real data (sizes 60k/10k/10k, 100 labels, names) — **priority, everyone depends on it** | `load_ledgar()` tested | Thu 15/10 | — |
| B2 | EDA: class distribution (log scale), split by year, examples per class | `notebooks/01_eda_baseline.ipynb` | Thu 15/10 | B1 |
| B3 | Lengths in words and WordPiece tokens (p50, p95, p99) → sent to A for `max_length` | figure + numbers in `PROGRESS.md` | Thu 15/10 | B1 |
| B4 | Train/test near-duplicates (exact normalised text + TF-IDF similarity > 0.95): share of the test set affected | number + list of de-duplicated `idx` | Fri 16/10 | B1 |
| B5 | **TF-IDF + LogReg** baseline (1-2 n-grams, C tuned on validation), val/test macro-F1, predictions in the common format → **tag m1** | run + predictions | Fri 16/10 | B1 |
| B6 | **LSTM / BiLSTM**: train vocabulary (`min_freq`), `<pad>`=0 `<unk>`=1, packing, GloVe, PyTorch loop, clipping, early stopping on validation | `src/models/lstm.py`, `notebooks/03_lstm_classifier.ipynb` | Sat 17/10 | B1 |
| B7 | Error analysis: per-class F1 vs support, top 10 confused pairs, 20 errors read and commented → **tag m3** | section of notebook 03 | Sun 18/10 | B5, B6 |
| B8 | Unified comparison table of all models + figures (macro-F1, per-class F1) | `results/summary_classification.csv` | Mon 19/10 | A6, A7 |
| B9 | Error analysis of the final model (stream A): which classes does BERT fix compared with TF-IDF? | figure + comment | Tue 20/10 | A6 |

## Stream C — Brian: Retrieval, audit, model card

| ID | Task | Deliverable | Date | Depends on |
|---|---|---|---|---|
| C1 | `all-MiniLM-L6-v2` index over the bank (train), normalised and cached embeddings, `search(query, k)` | `src/retrieval/dense.py` | Thu 15/10 | B1 |
| C2 | 10 queries (pasted clauses + questions) checked by hand → **tag m2** | `notebooks/02_retrieval_index.ipynb` | Fri 16/10 | C1 |
| C3 | Draft 28 lawyer questions; retain as illustrative future queries (no expert judgments, not evaluated) | `eval/retrieval_queries.jsonl` | Fri 16/10 | C1 |
| C4 | Compare MiniLM and BGE-small on a stratified 5% test-query sample, using same-label relevance and P@5 / MRR@10 / nDCG@10 | `results/retrieval_summary.csv` | Sat 17/10 | C1 |
| C5 | Evaluate BM25 and MiniLM+BM25 RRF on the same sampled test queries | `src/retrieval/hybrid.py`, `notebooks/05_retrieval_eval.ipynb` | Sun 18/10 | C1 |
| C6 | Classifier slice metrics: training-frequency groups, clause length, and explicitly labelled rule-detected party-role proxies | `src/audit/slices.py`, `notebooks/06_bias_audit.ipynb` | Sun 18/10 | A6 |
| C7 | One-feature counterfactual stress tests; bounded to at most 1,000 reproducibly sampled test clauses | `eval/counterfactuals/`, `src/audit/perturb.py` | Mon 19/10 | — |
| C8 | Focused audit on final classifier: per-slice F1 and counterfactual prediction/confidence changes; no retrieval or synthetic OOD audit claims | `results/audit/`, `notebooks/06_bias_audit.ipynb` | Tue 20/10 | A6, C6, C7 |
| C9 | Model card: intended / prohibited uses, measured results, sample sizes, and limitations | `docs/model_card.md` | Wed 21/10 | C8, A9 |

## Shared

- [ ] **T1** Slides (each stream provides its 2-3 slides) — Wed 21/10
- [ ] **T2** Timed 10-min rehearsal + likely CTO questions — Thu 22/10
- [ ] **T3** Code freeze, notebooks run on a fresh clone, tag `v1.0` — Thu 22/10 23:59
- [ ] **T4** (Bonus) Demo "clause → type + confidence + 5 closest clauses" (notebook or Streamlit) — if ahead
- [ ] **T5** Integrate the content of Session 7 (bias audit) into the C6-C9 plan as soon as it is available
