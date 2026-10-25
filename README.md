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
| **Robin** | A — Transformers: BERT-class fine-tuning, LoRA, paired tests, calibration | m4 |
| **Amandine** | B — Data, baseline & EDA: LEDGAR loading, TF-IDF, error analysis | m1 |
| **Brian** | C — Retrieval & audit: sentence-transformer index, retrieval evaluation, bias audit, model card | m2, audit |

The detailed plan, task split and schedule are in [`STRATEGY.md`](STRATEGY.md) and [`tasks/todo.md`](tasks/todo.md).
The actual state of the project is in [`PROGRESS.md`](PROGRESS.md).

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

| Milestone | Content | Notebook / script | Status |
|---|---|---|---|
| **m1** | LEDGAR loading, class distribution, **TF-IDF + logistic regression** baseline, macro-F1 | `notebooks/01_eda_baseline.ipynb` | Results available |
| **m2** | **Sentence-transformer** retrieval index over the clauses (embed, normalise, top-k cosine) + queries checked by hand | `notebooks/02_retrieval_index.ipynb` | ⏳ |
| **m4** | **BERT-class fine-tuning** (full) + **LoRA** + trade-off table + paired tests | `notebooks/04_transformer_finetuning.ipynb` | Results available |
| audit | Two dense encoders, BM25 and hybrid retrieval evaluation on a fixed stratified 5% test sample; focused classifier slice and counterfactual audit; model card | `notebooks/05_retrieval_eval.ipynb`, `notebooks/06_bias_audit.ipynb` | Retrieval results available; classifier audit running |

To keep the retrieval comparison runnable, notebook 05 evaluates two dense encoders, BM25 and a MiniLM+BM25 hybrid on a reproducible, stratified 5% sample of test clauses. These are sampled estimates, not full-test scores; indexing still uses the full training bank. Lawyer-query examples are retained as drafts but are not evaluated because no domain-expert relevance judgments are available. Notebook 06 reports classifier slices and counterfactual results from a bounded sample. It does not claim a broad audit of contract types, parties, languages, or retrieval robustness.

**Scope note:** the planned LSTM comparison is excluded from this delivery. The classification comparison therefore covers the TF-IDF baseline, full Legal-BERT fine-tuning, and Legal-BERT LoRA.

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

| Model | Trained parameters | Val macro-F1 | Test macro-F1 | Test accuracy | Training time |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg (C = 1.0, not tuned) | — | 0.7666 | 0.7625 | 0.8486 | Not recorded |
| Legal-BERT (full fine-tuning) | 109,559,140 | 0.7930 | 0.7905 | 0.8717 | 2,688 s |
| Legal-BERT (LoRA, r=8) | 371,812 (0.34% of full) | 0.7552 | 0.7691 | 0.8527 | 2,021 s |

The classifier test set is the chronological 2019 split. On the same 10,000 test clauses, Legal-BERT exceeds TF-IDF by **+0.0281 macro-F1** (95% paired-bootstrap CI **[0.0180, 0.0390]**, 1,000 resamples, seed 42; McNemar p = **1.71 × 10⁻¹⁶**), supporting a significant improvement under both pre-specified comparisons. The full fine-tuning vs. LoRA comparison gives Δ macro-F1 = +0.0214 (95% paired-bootstrap CI [0.0126, 0.0365], 1,000 resamples; McNemar p = 1.74 × 10⁻¹⁵). Only LoRA r=8 was run.

**Deviation from protocol.** The pre-registered Model B was to have C and n-grams tuned on validation. The baseline actually compared (`tfidf-logreg_baseline_20261023`, notebook 01) uses a fixed C = 1.0 with 1-2 n-grams; a tuning grid was considered on 25/10 and not run (one fit takes about 187 s and 5.4 GB of RAM per C value). A reproduction of the same configuration gives a validation macro-F1 of 0.7666. Because the baseline was not tuned, the +0.0281 BERT vs TF-IDF gap may be slightly overstated; H1 is reported as supported with this caveat.

Retrieval relevance is defined as matching the query clause's LEDGAR label. These values are from the fixed stratified 5% test-query sample (499 queries), not a full-test estimate or legal relevance judgment.

| Retrieval method | P@5 (same label) | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| `all-MiniLM-L6-v2` | 0.7639 | 0.8512 | 0.7421 |
| `BAAI/bge-small-en-v1.5` | **0.7876** | **0.8752** | **0.7723** |
| BM25 | 0.7800 | 0.8683 | 0.7553 |
| MiniLM + BM25 (RRF, c=60) | 0.7844 | 0.8685 | 0.7665 |

## What the system must not do

Provisional summary — the argued, quantified version will be in [`docs/model_card.md`](docs/model_card.md):

- give legal advice or judge the validity / enforceability of a clause;
- classify **non-English** clauses (models trained on US legal English);
- treat a predicted class as reliable for clause types **outside the 100 LEDGAR classes**;
- replace a lawyer's review: it **prioritises** and **suggests**, it does not decide.

## Installation

```bash
git clone https://github.com/1RLevasseurAlbertSchool/Projet_Deep_Learning.git
cd Projet_Deep_Learning

python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements-dev.txt   # runtime dependencies + jupyterlab, ipykernel, pytest

python -m pytest -q          # tests of the shared metrics
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

## Project structure

```
├── README.md              # This file: overview, pre-registered comparison, results
├── CLAUDE.md              # Working rules of the project (humans + Claude Code)
├── PROGRESS.md            # Operational log: actual state, results, problems, next actions
├── STRATEGY.md            # Strategy, A/B/C split, schedule, risks, presentation plan
├── requirements.txt       # Runtime dependencies (local and Colab)
├── requirements-dev.txt   # requirements.txt + jupyterlab, ipykernel, pytest (local only)
├── data/                  # (gitignored) raw/ and processed/ — LEDGAR downloaded by src/data.py
├── notebooks/             # Numbered notebooks, read in order (00 → 06)
├── src/
│   ├── config.py          # Paths, seed, dataset name, shared constants
│   ├── data.py            # LEDGAR loading + label names + head/middle/tail groups (shared contract)
│   ├── metrics.py         # Macro-F1, per-class F1, paired bootstrap, McNemar
│   ├── utils.py           # Seed, saving runs
│   ├── calibration.py     # Stream A: ECE, reliability diagram, human-review threshold
│   ├── models/            # Streams A and B: TF-IDF, LSTM, HF fine-tuning, LoRA
│   ├── retrieval/         # Stream C: sentence-transformer index and retrieval evaluation
│   └── audit/             # Stream C: slices, counterfactual perturbations
├── scripts/               # Command-line entry points (training, evaluation)
├── eval/                  # Hand-built lawyer questions and counterfactuals
├── tests/                 # Unit tests (pytest), e.g. metrics checked against the course exercises
├── results/               # Versioned metrics (JSON/CSV) and figures; predictions gitignored
├── docs/
│   └── model_card.md      # Intended / prohibited use, per-slice performance, limitations
└── tasks/
    ├── todo.md            # Task list per work stream, with dates
    └── lessons.md         # Improvement loop: mistakes and rules to avoid them
```

## References

- Chalkidis et al. (2022), *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*, ACL. [arXiv:2110.00976](https://arxiv.org/abs/2110.00976)
- Tuggener et al. (2020), *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts*, LREC.
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*.

## Licence

Academic project. LEDGAR / LexGLUE data is used under its authors' terms (see the dataset card on Hugging Face).
Course materials are not redistributed in this repository.
