# Project strategy

> **Initial plan of 14/10/2026, kept as written.** What was actually done (dropped LSTM, cancelled near-duplicate
> check, protocol deviations, results) is in [`PROGRESS.md`](PROGRESS.md) and [`tasks/todo.md`](tasks/todo.md).

> Reference document for the organisation: objectives, split into three work streams, schedule, dependencies,
> risks and presentation plan. The detailed task list is in [`tasks/todo.md`](tasks/todo.md).

## 1. What the jury will assess, and our answer

| Requirement of the brief | Our answer | Stream |
|---|---|---|
| Fine-tune a pretrained language model (BERT / CamemBERT) on clause classification | BERT-class encoder fine-tuned on LEDGAR (full + LoRA). LEDGAR is **in English**: CamemBERT (French) is not relevant here; we justify this in the presentation | A |
| Full pipeline: tokenisation, training loop, evaluation on a held-out set | The model's own tokenizer, the HF `Trainer` **and** a hand-written PyTorch loop for the LSTM; single evaluation on the 2019 test set | A, B |
| Sentence-transformer retrieval index | Normalised MiniLM index (cosine) and sampled held-out same-label evaluation; lawyer-query relevance not evaluated | C |
| Bias audit | Classifier performance by training frequency and clause length, with party-role proxies and a bounded counterfactual stress test | C |
| Evidence "overall, per clause type, across contracts and parties" | Overall macro-F1, per-class F1, available slice tables, and explicit limitations where reliable group metadata is absent | A, B, C |
| "What it must not be used for" | Model card with intended / prohibited uses, backed by the audit numbers | C (+ all) |
| Reproducible results | Seed, configs, versions, `results/runs/*`, repo that runs on a fresh clone | All |

**Common thread of the presentation**: a **ladder of models** (TF-IDF → LSTM → BERT → LoRA) where each step is
justified by a paired test, then an evaluated search engine, then an audit that honestly shows where the system
breaks. Expected highlight: in LexGLUE, TF-IDF is only ~1.6 macro-F1 points behind Legal-BERT → the question
"is deep learning worth its cost here?" is real, and we answer it with a statistical test.

## 2. Course milestones (Sessions 1 to 6)

| Tag | Required by the course | Owner | Target date |
|---|---|---|---|
| `m1` | Repo, LEDGAR loading, class distribution, **TF-IDF + LogReg**, macro-F1 | Amandine | Fri 16/10 |
| `m2` | **Sentence-embedding index** over LEDGAR (embed, normalise, top-k cosine) + queries checked by hand | Brian | Fri 16/10 |
| `m3` | **LSTM classifier** on LEDGAR + error analysis (per-class F1, most confused classes) | Amandine | Sun 18/10 |
| `m4` | **Fine-tuned BERT-class model** + results table; **LoRA runs** + trade-off table; main comparison written in the README **before** being run, then its **paired test** | Robin | Mon 19/10 |
| audit | Bias audit (Session 7 upcoming), retrieval evaluation, model card | Brian | Tue 20/10 |
| `v1.0` | Code freeze, final README | All | Thu 22/10 (evening) |

## 3. Split into three work streams

Each stream is **autonomous** thanks to the interface contracts in [`CLAUDE.md`](CLAUDE.md) (same `load_ledgar`,
same metrics, same prediction format). Dependencies between streams are limited and dated.

### Stream A — Robin: Transformers, training and statistical comparison

*Core of the "fine-tune a pretrained language model" requirement.*

1. Generic Hugging Face fine-tuning script (`src/models/transformer.py`, `scripts/train_transformer.py`),
   parameterised by the model name; smoke test on 2,000 examples.
2. **Choice of `max_length`** from the token lengths (stream B EDA) — 256 vs 512.
3. **Screening on the validation set**: DistilBERT (fast iterations), then `legal-bert-base-uncased` and/or
   `deberta-v3-base` depending on the GPU budget; selection **on the validation set only**.
4. **Final full fine-tuning run** → a single test evaluation → predictions in the common format.
5. **LoRA** on the same encoder, r ∈ {4, 8, 16}: trade-off table (trained parameters, time, memory, macro-F1).
6. **Paired tests**: paired bootstrap on macro-F1 + McNemar for (BERT vs TF-IDF) [main comparison],
   (full vs LoRA), (BERT vs LSTM).
7. **Calibration** (ECE, reliability diagram) and **confidence threshold** → "to be reviewed by a lawyer": coverage
   vs precision.
8. Integration: PR reviews, README results, tags.

### Stream B — Amandine: Data, baselines and recurrent network

*Shared foundations + the first and second rungs of the ladder.*

1. **`src/data.py` validated on day 1** (everyone depends on it): loading, label names, size checks.
2. **EDA**: class distribution (log scale), lengths in words and tokens (WordPiece), split by year,
   **train/test near-duplicates**, examples per class.
3. **TF-IDF + LogReg baseline** (n-grams, C tuned on validation) → `m1`; first predictions in the common format
   (lets stream C start the audit without waiting for BERT).
4. **LSTM / BiLSTM**: vocabulary on train, `<pad>`=0 / `<unk>`=1, packing, GloVe initialisation, hand-written
   PyTorch loop, clipping → `m3`.
5. **Error analysis**: per-class F1 vs support, top confused pairs, examples read and commented;
   repeated on stream A's final model.
6. **Unified comparison table** of all models (`results/summary_classification.csv`) + figures.

### Stream C — Brian: Retrieval, focused audit and model card

*The two other building blocks of the product: search and trust.*

1. **`m2` index**: `all-MiniLM-L6-v2` (course reference) over the bank = train; top-k cosine search;
   10 queries checked by hand.
2. **Focused retrieval evaluation**:
   - automatic: one fixed, reproducible **stratified 5% sample** of test clauses, relevant = same LEDGAR label → P@5, MRR@10, nDCG@10;
   - do not report lawyer-query metrics without domain-expert relevance judgments; retain drafted questions as future evaluation material;
   - do not interpret same-label relevance as a legal judgment or the sample as a full-test score.
3. **Focused classifier audit**:
   - per-slice: training-frequency group and clause length; rule-detected party roles are labelled as proxies, not contract metadata;
   - counterfactual stress test on at most **1,000 sampled test clauses**, changing one supported text feature at a time; inspect examples before interpretation;
   - do not claim year comparisons (the test split is 2019 only), representative European/French performance, or retrieval robustness.
4. **Model card** (`docs/model_card.md`): intended / prohibited uses, measured results, sample sizes and limitations.

### Shared work

- Presentation: slides + rehearsal (each person presents their stream, ~3 min).
- Cross-review: every PR is reviewed by another stream.
- Demo: a notebook or a small app (Streamlit) "paste a clause → type + confidence + 5 closest clauses" if time
  allows (bonus).

## 4. Schedule (project start: Wed 14/10 — delivery: Fri 23/10)

| Day | Stream A — Robin | Stream B — Amandine | Stream C — Brian |
|---|---|---|---|
| **Wed 14/10** | Repo initialised, docs, plan (done) | Clone + env | Clone + env |
| **Thu 15/10** | HF script + DistilBERT smoke test (2k ex.) | `src/data.py` validated, EDA | MiniLM index v0 + 10 queries |
| **Fri 16/10** | Full DistilBERT (val) | TF-IDF + LogReg → **tag m1** | Queries checked → **tag m2**; lawyer question set |
| **Sat 17/10** | Screening legal-bert / deberta (val) | LSTM + GloVe | Prepare fixed-sample MiniLM evaluation |
| **Sun 18/10** | Final full FT run (test once) | Error analysis → **tag m3** | Implement focused retrieval and classifier audit |
| **Mon 19/10** | LoRA r=4/8/16 + paired tests → **tag m4** | Unified comparison table | Build small counterfactual test set |
| **Tue 20/10** | Calibration + review threshold | Error analysis of the final model | Run bounded classifier audit |
| **Wed 21/10** | README results | Final figures | Model card |
| **Thu 22/10** | **Code freeze**, test on a fresh clone, tag `v1.0` | Slides | Slides |
| **Fri 23/10** | **Delivery** + rehearsal | | |

## 5. Critical dependencies

| Who is waiting | For what | From whom | At the latest |
|---|---|---|---|
| Everyone | `load_ledgar` + `get_label_names` validated | B | Thu 15/10 |
| A | Token lengths (choice of `max_length`) | B | Thu 15/10 |
| C | First predictions in the common format (TF-IDF) | B | Fri 16/10 |
| C, B | Final model predictions (test) | A | Sun 18/10 |
| A | TF-IDF and LSTM predictions (paired tests) | B | Sun 18/10 |

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Not enough Colab GPU budget for several base encoders | DistilBERT first; `max_length` 256 if the EDA allows it; fp16; a single final run; checkpoints on Drive |
| The BERT vs TF-IDF gap is not significant | That is a result: we present it as such (pre-registered comparison) |
| Leakage through train/test near-duplicates | Measured during EDA; metrics also reported on the de-duplicated test set |
| LEDGAR has no "contract type" metadata | Proxy through detection of party roles, documented as a limitation |
| Session 7 (bias audit) not covered yet | The framework is generic; we adjust it after the session |
| Git conflicts on notebooks | One notebook = one owner; shared logic in `src/` |
| Non-reproducible results | Seed, configs, versions; re-run `v1.0` on a fresh clone on 22/10 |

## 7. Presentation plan (10 min)

| Time | Content | Who |
|---|---|---|
| 1 min | Business problem, what the CTO has to decide | Robin |
| 1.5 min | LEDGAR data: 100 imbalanced classes, chronological split, pitfalls | Amandine |
| 3 min | Model ladder TF-IDF → LSTM → BERT → LoRA, paired tests, cost vs gain | Amandine, Robin |
| 2 min | Clause search: demo, metrics, failures (numeric constraints) | Brian |
| 2 min | Audit: where the system can / cannot be trusted, counterfactuals, human review threshold | Brian |
| 0.5 min | Deployment recommendation + prohibited uses | Robin |
