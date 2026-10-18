# Model card — Clause Review Assistant (NLP core)

> Skeleton created on 14/10/2026. The quantified sections are filled **only** from the results in `results/`
> (stream C, task C9). Sentences marked *(provisional)* are to be confirmed or corrected by the audit.

## 1. Description

| | |
|---|---|
| Components | (1) clause-type classifier, 100 LEDGAR classes; (2) similar-clause search engine |
| Classification model | *to be completed (HF name, revision, full FT or LoRA)* |
| Embedding model | *to be completed* |
| Training data | LEDGAR / LexGLUE: 60,000 clauses from contracts filed with the SEC (EDGAR), 2016-2017 |
| Language | English (legal, United States) |
| Team | Robin, Amandine, Brian — Albert School, October 2026 |

## 2. Intended use

- Pre-label the type of every clause during a due diligence, **to speed up** a lawyer's review.
- Suggest comparable clauses from the firm's clause bank.
- Flag low-confidence clauses for **priority human review**.

## 3. Prohibited or discouraged uses *(provisional)*

- Legal advice, assessment of the validity, enforceability or unfairness of a clause.
- **Non-English** clauses (French, German…) or clauses drafted with practices very different from SEC contracts.
- Automatic decisions without human review (e.g. marking a contract as "compliant").
- Clause types absent from the 100 classes: the model will still pick a known class.
- Whole documents: the system works clause by clause.

## 4. Overall performance

| Metric (2019 test set) | Value | 95 % CI |
|---|---:|---:|
| Macro-F1 | | |
| Accuracy (micro-F1) | | |
| Macro-F1, de-duplicated test set | | |

## 5. Per-slice performance

| Slice | Macro-F1 | Support | Comment |
|---|---:|---:|---|
| Frequent classes (head) | | | |
| Rare classes (tail) | | | |
| Short / long clauses | | | |
| Party type: Borrower / Lender | | | |
| Party type: Employer / Employee | | | |
| Party type: Landlord / Tenant | | | |

## 6. Robustness and bias

| Perturbation | Prediction flip rate | Confidence change |
|---|---:|---:|
| Swapping party roles | | |
| Pronouns he ↔ she | | |
| Jurisdiction (New York → Luxembourg…) | | |
| Currency (USD → EUR) | | |
| European-style clauses | | |
| French clauses | | |

## 7. Calibration and human review threshold

*Reliability diagram, ECE, coverage and precision above the selected threshold.*

## 8. Known limitations

- No "contract type" metadata in LEDGAR: the per-contract breakdown relies on a proxy (party roles detected by
  rules).
- Chronological split: future performance may drop as drafting practices evolve.

## 9. Reproducibility

Seed 42; configuration and versions of every run in `results/runs/<run_id>/config.json`.
