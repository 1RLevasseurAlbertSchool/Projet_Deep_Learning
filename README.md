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
| **Amandine** | B — Data, baselines & RNN: LEDGAR loading, EDA, TF-IDF, LSTM, error analysis | m1, m3 |
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
| **m1** | LEDGAR loading, class distribution, **TF-IDF + logistic regression** baseline, macro-F1 | `notebooks/01_eda_baseline.ipynb` | ⏳ |
| **m2** | **Sentence-transformer** retrieval index over the clauses (embed, normalise, top-k cosine) + queries checked by hand | `notebooks/02_retrieval_index.ipynb` | ⏳ |
| **m3** | **LSTM** classifier (GloVe, packing) + error analysis (per-class F1, most confused classes) | `notebooks/03_lstm_classifier.ipynb` | ⏳ |
| **m4** | **BERT-class fine-tuning** (full) + **LoRA** + trade-off table + **paired test** of the main comparison | `notebooks/04_transformer_finetuning.ipynb` | ⏳ |
| audit | Retrieval evaluation, **bias / robustness audit** per slice, model card | `notebooks/05_retrieval_eval.ipynb`, `notebooks/06_bias_audit.ipynb` | ⏳ |

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
| TF-IDF + LogReg | — | | | | |
| LSTM (GloVe) | | | | | |
| BERT-class (full FT) | | | | | |
| BERT-class + LoRA | | | | | |

| Retrieval | P@5 (same type) | MRR@10 | nDCG@10 | Manual queries judged |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | | | |

## What the system must not do

Provisional summary — the argued, quantified version will be in [`docs/model_card.md`](docs/model_card.md):

- give legal advice or judge the validity / enforceability of a clause;
- classify **non-English** clauses (models trained on US legal English);
- classify clause types **outside the 100 LEDGAR classes** without an uncertainty signal;
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
- The test set is used **only once** per final model; every decision is made on the validation set.

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
│   ├── retrieval/         # Stream C: sentence-transformer index, BM25, hybrid
│   └── audit/             # Stream C: slices, counterfactual perturbations
├── scripts/               # Command-line entry points (training, evaluation)
├── eval/                  # Hand-built sets: lawyer questions, counterfactuals, out-of-distribution
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
