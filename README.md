# Clause Review Assistant — NLP core (LEDGAR)

Final project of the course **Deep Learning for Sequential Data and NLP** (MSA-DAT07-01, Albert School, G. Saint-Cirgue).

> We are the AI team of a European legal-tech start-up. We build, evaluate and audit the
> **NLP core of a contract-review assistant** that:
> 1. **classifies** every clause into one of the 100 clause types of the review checklist;
> 2. **retrieves** the most similar clauses from a clause bank, given a pasted clause or a lawyer's question
>    (*"show me termination clauses with a 30-day notice period"*);
> 3. **documents** where it can be trusted — overall, per clause type, per kind of contract and party — and **what it
>    must not be used for**.

## Presentation

> Link to the slides (10 min + 5 min Q&A): *to be added*.

## Team

| Member | Work stream | Course milestones |
|---|---|---|
| **Robin** | A — Transformers: BERT-class fine-tuning, LoRA, paired tests, calibration; error analysis and unified comparison (B7-B9) | m3, m4 |
| **Amandine** | B — Data, baseline & EDA: LEDGAR loading, TF-IDF | m1 |
| **Brian** | C — Retrieval & audit: sentence-transformer index, retrieval evaluation, bias audit, model card | m2, audit |

The initial plan (14/10) is in [`STRATEGY.md`](STRATEGY.md); the final status of every task is in
[`tasks/todo.md`](tasks/todo.md) and what was actually done, with the results, in [`PROGRESS.md`](PROGRESS.md).

## Data — LEDGAR (LexGLUE)

| | Train | Validation | Test |
|---|---:|---:|---:|
| Clauses | 60,000 | 10,000 | 10,000 |
| Contract years | 2016–2017 | 2018 | 2019 |

- **Source**: provisions from contracts filed with the SEC (**EDGAR** database), **LexGLUE** version
  (Chalkidis et al., 2022), Hugging Face: `coastalcph/lex_glue`, configuration `ledgar`.
- **Task**: **single-label** classification, **100 highly imbalanced classes**.
- **Chronological split**: the test set (2019) comes after the training set (2016-2017) → test performance already
  measures a form of **temporal drift**.
- The data is not versioned (`data/` is gitignored): it is downloaded by `src/data.py`.

## Pipeline and milestones

| Milestone | Content | Deliverable | Status |
|---|---|---|---|
| **m1** | LEDGAR loading, class distribution, exact-duplicate check, **TF-IDF + logistic regression** baseline (C = 1.0, not tuned), macro-F1 | `notebooks/01_eda_baseline.ipynb`, `results/runs/tfidf-logreg_baseline_20261023/` | ✅ |
| **m2** | **Sentence-transformer** retrieval index over the training clauses (embed, normalise, top-k cosine) + 10 searches checked by hand | `notebooks/02_retrieval_index.ipynb`, `src/retrieval/dense.py` | ✅ |
| **m3** | Error analysis of Legal-BERT vs TF-IDF: per-class F1 vs frequency, top confusions, truncation, 20 errors read (the LSTM was dropped on 25/10) | `notebooks/03_error_analysis.ipynb`, `src/analysis.py`, `results/error_analysis_*.csv` | ✅ (without LSTM) |
| **m4** | **BERT-class fine-tuning** (full) + **LoRA** r = 8 + trade-off table + paired tests + calibration | `notebooks/04_transformer_finetuning.ipynb`, `results/runs/`, `results/lora_tradeoff.csv`, `results/paired_tests.csv`, `results/calibration_*` | ✅ |
| audit | Two dense encoders, BM25 and hybrid retrieval evaluation on a fixed stratified 5% test sample; focused classifier slice and counterfactual audit; model card | `notebooks/05_retrieval_eval.ipynb`, `notebooks/06_bias_audit.ipynb`, `results/retrieval_summary.csv`, `results/audit/`, `docs/model_card.md` | ✅ |

The Git tags `m1` to `m4` and `v1.0` are not created yet (done at delivery).

To keep the retrieval comparison runnable, notebook 05 evaluates two dense encoders, BM25 and a MiniLM+BM25 hybrid on a reproducible, stratified 5% sample of test clauses. These are sampled estimates, not full-test scores; indexing still uses the full training bank. Lawyer-query examples are retained as drafts but are not evaluated because no domain-expert relevance judgments are available. Notebook 06 reports classifier slices and counterfactual results from a bounded sample. It does not claim a broad audit of contract types, parties, languages, or retrieval robustness.

**Scope note:** the planned LSTM comparison was dropped by team decision on 25/10. The classification comparison therefore covers the TF-IDF baseline, full Legal-BERT fine-tuning, and Legal-BERT LoRA. The train/test near-duplicate check (B4) was cancelled as out of scope: notebook 01 checks exact duplicates only, and no score on a de-duplicated test set is reported.

### Notebooks

| # | Notebook | Content | Where it runs |
|---|---|---|---|
| 01 | `01_eda_baseline.ipynb` | LEDGAR EDA, exact-duplicate check, TF-IDF + LogReg baseline | Local (CPU) |
| 02 | `02_retrieval_index.ipynb` | MiniLM index over the training bank, 10 searches checked by hand (m2) | Local |
| 03 | `03_error_analysis.ipynb` | Error analysis of Legal-BERT vs TF-IDF (m3) | Local, needs the test predictions |
| 04 | `04_transformer_finetuning.ipynb` | Colab fine-tuning: encoder screening, full fine-tuning, LoRA, single test evaluation | **Colab (GPU)** |
| 05 | `05_retrieval_eval.ipynb` | Retrieval evaluation: MiniLM, BGE-small, BM25, MiniLM + BM25 RRF | Local |
| 06 | `06_bias_audit.ipynb` | Classifier audit: per-slice performance, counterfactual stress test | Local, needs the test predictions and the final checkpoint |
| 07 | `07_demo.ipynb` | End-to-end demo (bonus T4): clause → top-3 types, automatic / send to a lawyer, 5 closest training clauses; one lawyer question | Local, needs the final checkpoint |

## Main comparison (pre-registered on 14 October 2026, before any experiment)

The course asks us to write the main comparison **before** running it. This block will not be changed after the
first results; any deviation from the protocol will be stated explicitly in [`PROGRESS.md`](PROGRESS.md).

- **Question**: does a fine-tuned BERT-class encoder significantly beat the TF-IDF + logistic regression baseline on
  LEDGAR?
- **Model A**: the best fine-tuned encoder (full fine-tuning), selected **on the validation set (2018) only** among
  `distilbert-base-uncased`, `bert-base-uncased`, `nlpaueb/legal-bert-base-uncased`, `microsoft/deberta-v3-base`
  (depending on the GPU budget).
- **Model B**: TF-IDF + logistic regression, hyperparameters (C, n-grams) tuned on the validation set.
- **Main metric**: **macro-F1** on the test set (2019), evaluated **once** per model. Secondary: accuracy (= micro-F1).
- **Test**: **paired bootstrap** on macro-F1 (1,000 resamples, 95 % CI, seed 42); **McNemar** on accuracy (exact
  if b + c < 25).
- **Hypothesis H1**: ΔF1 = F1(A) − F1(B) > 0, with a 95 % CI that excludes 0.
- **What we already know**: in LexGLUE the published gap is small (TF-IDF SVM 81.4 vs Legal-BERT 83.0 macro-F1).
  It is therefore **plausible that the gap is not significant** — this will be reported as such.
- **Secondary comparison**: full fine-tuning vs **LoRA** (r ∈ {4, 8, 16}) on the same encoder: trained parameters,
  time, GPU memory, macro-F1, paired test.

## Results

> This section is filled only with results actually obtained (see `results/`).

Unified comparison (task B8), generated by `python scripts/summary_classification.py` from the run files and the
paired tests; same cells as [`results/summary_classification.csv`](results/summary_classification.csv):

| Model | Trained parameters | % of full | Weights on disk (MB) | Train time (s) | Peak GPU memory (MB) | Val macro-F1 | Test macro-F1 | Test accuracy | ΔF1 (Legal-BERT − model) | 95 % CI of ΔF1 | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg (C = 1.0, not tuned) | 17,347,800 | 15.83 | not recorded | not recorded | not applicable (CPU) | 0.767 | 0.762 | 0.849 | +0.028 | [+0.018, +0.039] | 1.71e-16 |
| DistilBERT (full fine-tuning) | 67,030,372 | 61.18 | not recorded | 1,227.5 | 2,536.7 | 0.786 | not run | not run | not run | not run | not run |
| Legal-BERT (full fine-tuning) | 109,559,140 | 100.00 | 438.26 | 2,688.2 | 3,106.2 | 0.793 | 0.791 | 0.872 | reference | reference | reference |
| Legal-BERT (LoRA, r=8) | 371,812 | 0.34 | 1.49 | 2,021.5 | 1,889.0 | 0.755 | 0.769 | 0.853 | +0.021 | [+0.013, +0.036] | 1.74e-15 |

- **TF-IDF + LogReg**: trained parameters are the LogReg weights, 100 × (173,477 features + 1); its validation
  macro-F1 comes from the output of notebook 01 (the run's `metrics.json` has no validation block).
- **DistilBERT**: screening on validation only; the single test evaluation was reserved for the selected encoder.
- **Weights on disk**: full `model.safetensors` vs LoRA `adapter_model.safetensors` (the adapter still needs the
  pretrained Legal-BERT at serving time); measured as described in `PROGRESS.md` (A7).
- **Paired tests**: paired bootstrap on macro-F1 (1,000 resamples, seed 42) and McNemar on accuracy, from
  `results/paired_tests.csv`. Figures: `results/figures/summary_test_scores.png`, `summary_per_class_f1.png`,
  `summary_tradeoff.png`.

The classifier test set is the chronological 2019 split. On the same 10,000 test clauses, Legal-BERT exceeds TF-IDF by **+0.028 macro-F1** (95% paired-bootstrap CI **[0.018, 0.039]**, 1,000 resamples, seed 42; McNemar p = **1.71 × 10⁻¹⁶**), supporting a significant improvement under both pre-specified comparisons. The full fine-tuning vs. LoRA comparison gives Δ macro-F1 = +0.021 (95% paired-bootstrap CI [0.013, 0.036], 1,000 resamples; McNemar p = 1.74 × 10⁻¹⁵). Only LoRA r=8 was run.

**Deviation from protocol.** The pre-registered Model B was to have C and n-grams tuned on validation. The baseline actually compared (`tfidf-logreg_baseline_20261023`, notebook 01) uses a fixed C = 1.0 with 1-2 n-grams; a tuning grid was considered on 25/10 and not run (one fit takes about 187 s and 5.4 GB of RAM per C value). A reproduction of the same configuration gives a validation macro-F1 of 0.7666. Because the baseline was not tuned, the +0.0281 BERT vs TF-IDF gap may be slightly overstated; H1 is reported as supported with this caveat.

Retrieval relevance is defined as matching the query clause's LEDGAR label. These values are from the fixed stratified 5% test-query sample (499 queries), not a full-test estimate or legal relevance judgment.

| Retrieval method | P@5 (same label) | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| `all-MiniLM-L6-v2` | 0.7639 | 0.8512 | 0.7421 |
| `BAAI/bge-small-en-v1.5` | **0.7876** | **0.8752** | **0.7723** |
| BM25 | 0.7800 | 0.8683 | 0.7553 |
| MiniLM + BM25 (RRF, c=60) | 0.7844 | 0.8685 | 0.7665 |

## What the system must not do

Summary of the prohibited uses; each one is argued with the measured evidence in section 3 of the final
[model card](docs/model_card.md#3-prohibited-or-discouraged-uses):

- classify **non-English** clauses or clauses drafted far from US SEC practice (never tested);
- label clause types **outside the 100 LEDGAR classes** (there is no "other" class, and confidence is not a reliable
  alarm);
- give **legal advice** or judge the validity, enforceability or fairness of a clause;
- label **rare clause types** fully automatically (tail macro-F1 0.716 vs 0.932 for head classes);
- take **decisions without human review**, in particular from labels below the 0.892 confidence threshold;
- process **whole documents**: it was trained and evaluated clause by clause.

## Installation

```bash
git clone https://github.com/1RLevasseurAlbertSchool/Projet_Deep_Learning.git
cd Projet_Deep_Learning

python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements-dev.txt   # runtime dependencies + jupyterlab, ipykernel, pytest

python -m pytest -q          # unit tests (CPU, no model download)
python -m src.data           # checks LEDGAR loading (60k / 10k / 10k, 100 labels)
```

Heavy training (fine-tuning) runs on **Google Colab (GPU)**; everything else runs locally.
In Colab: clone the repo, `pip install -r requirements.txt` (runtime dependencies only: the dev tools would
upgrade the `ipykernel` / `jupyter-client` versions pinned by `google-colab`), then open the notebook.

## Reproducibility

- Global seed **42** (`src/utils.py::set_seed`), model and library versions pinned and reported.
- Every run writes its configuration, metrics and predictions to `results/` (format described in
  [`CLAUDE.md`](CLAUDE.md)).
- The classification test set is used **only once** per final model; retrieval uses a fixed stratified 5% sample
  solely for its final benchmark, not for model selection. Classifier decisions are made on the validation set.

## How to reproduce

The versioned files in `results/` (run configs and metrics, CSVs, figures) are enough to read every number in this
README and in the model card. Re-running a step needs the inputs below, which are **gitignored**:

| Gitignored input | Produced by | How to get it back |
|---|---|---|
| `data/` (LEDGAR) and `data/indexes/` (embedding cache) | `src/data.py`, `src/retrieval/dense.py` | Downloaded from Hugging Face / rebuilt automatically on first use |
| `results/predictions/legal-bert_final_20261019-1523_{validation,test}.parquet`, `legal-bert_lora-r8_20261020-0940_{validation,test}.parquet` | Notebook 04 on Colab | Copy them from `results/predictions/` of the Drive clone used by notebook 04 (the test set is evaluated once: `scripts/evaluate_checkpoint.py` refuses a second test evaluation of the same run unless `--force`) |
| `results/predictions/tfidf-logreg_baseline_20261023_test.parquet` | Notebook 01 | `python scripts/regenerate_tfidf_predictions.py` (refits the same configuration, saves only if the test metrics equal the run's `metrics.json`) |
| Checkpoints (`*.safetensors`, `models/`) | Notebook 04 on Colab | In the Drive checkpoints folder of notebook 04; notebook 06 expects the final checkpoint at `models/ledgar_checkpoints/legal-bert_full_20261019-0916/checkpoint-11250` |

Steps, in order:

1. **Local**: `pip install -r requirements-dev.txt`, `python -m pytest -q`, `python -m src.data`.
2. **Local (CPU)**: notebook 01 (TF-IDF baseline; one fit takes about 187 s and 5.4 GB of RAM), notebooks 02 and 05
   (retrieval).
3. **Colab (GPU)**: notebook 04 — `scripts/train_transformer.py` (DistilBERT and Legal-BERT screening on validation,
   LoRA r = 8 with `--eval-test`), then `scripts/evaluate_checkpoint.py --source-run legal-bert_full_20261019-0916`
   (single test evaluation). Writes `results/runs/<run_id>/`, the prediction files and the checkpoints.
4. **Local, from the prediction files**: `scripts/calibration_report.py` and `scripts/calibration_per_class.py`
   (`--run-id legal-bert_final_20261019-1523`), `scripts/lora_tradeoff_table.py`, `scripts/paired_tests.py`,
   `scripts/summary_classification.py`, then notebook 03 (error analysis) and notebook 06 (audit, also needs the
   checkpoint). The exact commands are in the docstring of each script and in [`PROGRESS.md`](PROGRESS.md).

## Demo

End-to-end demo of the product (bonus T4, no training): a clause goes in; out come the **3 most probable clause
types**, the decision **automatic / send to a lawyer** (confidence vs the 0.892 threshold chosen on validation) and the
**5 most similar training clauses**. A lawyer can also search the clause bank with a free-text question.
Code: `src/pipeline.py` (`ClauseReviewer`), `scripts/demo.py`, `notebooks/07_demo.ipynb`.

**1. Get the classifier checkpoint** (gitignored, 438 MB). From the Drive folder of notebook 04,
`MyDrive/ledgar_checkpoints/legal-bert_full_20261019-0916/checkpoint-11250/` (the best checkpoint of
`legal-bert_full_20261019-0916`, i.e. the model evaluated once on test as `legal-bert_final_20261019-1523`), copy these
4 files into `models/legal-bert_final/`:

```
models/legal-bert_final/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

If a file is missing, the demo stops with a message listing them.

**2. Run it** (MPS / CUDA if available, else CPU):

```bash
python scripts/demo.py "Either party may terminate this Agreement upon thirty (30) days' prior written notice."
```

```bash
python scripts/demo.py --question "termination clauses with a 30-day notice period"
```

Or open `notebooks/07_demo.ipynb` (3 test clauses, a hand-written European clause, a French clause, a lawyer's
question). The first run downloads LEDGAR and the retrieval encoder and builds the index over the 60,000 training
clauses (about 4.5 min on an Apple-silicon MPS GPU, encoder download included), cached in `data/indexes/`; later runs
load it in a few seconds.

**Caveats.** The retrieval encoder is BGE-small, ranked first on P@5 in `results/retrieval_summary.csv`; that
benchmark used a stratified 5 % sample of test queries, so this is a convenience choice for the demo, not a validated
selection. The demo clauses are illustrations, not an evaluation. Non-English clauses remain a prohibited use (see
*What the system must not do*).

## Project structure

```
├── README.md              # This file: overview, pre-registered comparison, results
├── CLAUDE.md              # Working rules of the project (humans + Claude Code)
├── PROGRESS.md            # Operational log: actual state, results, problems, next actions
├── STRATEGY.md            # Initial plan (14/10): A/B/C split, schedule, risks, presentation plan
├── requirements.txt       # Runtime dependencies (local and Colab)
├── requirements-dev.txt   # requirements.txt + jupyterlab, ipykernel, pytest (local only)
├── data/                  # (gitignored) LEDGAR downloaded by src/data.py, embedding cache in indexes/
├── notebooks/             # Numbered notebooks, read in order (01 → 07)
├── src/
│   ├── config.py          # Paths, seed, dataset name, shared constants
│   ├── data.py            # LEDGAR loading + label names + head/middle/tail groups (shared contract)
│   ├── metrics.py         # Macro-F1, per-class F1, paired bootstrap, McNemar
│   ├── utils.py           # Seed, saving runs and predictions, single-use test guard
│   ├── comparison.py      # Paired comparison of two runs on aligned prediction files (A8)
│   ├── calibration.py     # ECE, reliability diagram, human-review threshold, per-class view (A9)
│   ├── analysis.py        # Error analysis of Legal-BERT vs TF-IDF (B7, B9)
│   ├── pipeline.py        # End-to-end ClauseReviewer: type + review decision + similar clauses (T4)
│   ├── models/
│   │   └── transformer.py # HF fine-tuning (full and LoRA), single test evaluation from a checkpoint
│   ├── retrieval/
│   │   ├── dense.py       # Normalised sentence-embedding index, search(query, k)
│   │   ├── hybrid.py      # BM25 and reciprocal-rank fusion
│   │   └── evaluation.py  # P@k, MRR, nDCG with same-label relevance
│   └── audit/
│       ├── slices.py      # Per-slice metrics (frequency, length, party-role proxies)
│       ├── perturb.py     # Counterfactual perturbations
│       └── classifier.py  # Inference adapter for a saved HF checkpoint
├── scripts/
│   ├── token_lengths.py                 # Token-length distribution (choice of max_length, A3)
│   ├── train_transformer.py             # Fine-tuning, full or LoRA (A1-A5, A7)
│   ├── evaluate_checkpoint.py           # Single test evaluation of the run chosen on validation (A6)
│   ├── lora_tradeoff_table.py           # Full fine-tuning vs LoRA table (A7)
│   ├── paired_tests.py                  # Paired bootstrap + McNemar (A8)
│   ├── calibration_report.py            # Calibration and human-review threshold (A9)
│   ├── calibration_per_class.py         # Per-class precision at the threshold (A9)
│   ├── regenerate_tfidf_predictions.py  # TF-IDF test predictions, checked against the run metrics
│   ├── summary_classification.py        # Unified comparison table and figures (B8)
│   └── demo.py                          # Command-line demo of the clause reviewer (T4)
├── eval/                  # Draft lawyer questions, counterfactual notes, unlabelled OOD probes
├── tests/                 # Unit tests (pytest), e.g. metrics checked against the course exercises
├── results/               # Versioned metrics (JSON/CSV) and figures; predictions gitignored
├── docs/
│   └── model_card.md      # Intended / prohibited use, per-slice performance, limitations
└── tasks/
    ├── todo.md            # Task list per work stream, with final statuses
    └── lessons.md         # Improvement loop: mistakes and rules to avoid them
```

## References

- Chalkidis et al. (2022), *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*, ACL. [arXiv:2110.00976](https://arxiv.org/abs/2110.00976)
- Tuggener et al. (2020), *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts*, LREC.
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*.

## Licence

Academic project. LEDGAR / LexGLUE data is used under its authors' terms (see the dataset card on Hugging Face).
Course materials are not redistributed in this repository.
