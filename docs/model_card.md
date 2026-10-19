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

Final classification model `legal-bert_final_20261019-1523` (Legal-BERT, full fine-tuning). Numbers from
[`results/calibration_legal-bert_final_20261019-1523.json`](../results/calibration_legal-bert_final_20261019-1523.json)
(`scripts/calibration_report.py`). Confidence = probability of the predicted class (top-label calibration); ECE over
15 equal-width bins.

| Split | Accuracy | Mean confidence | ECE |
|---|---:|---:|---:|
| Validation (2018) | 0.8729 | 0.9142 | 0.0422 |
| Test (2019) | 0.8717 | 0.9131 | 0.0425 |

![Reliability diagram](../results/figures/reliability_legal-bert_final_20261019-1523.png)

A clause is labelled automatically when its confidence is ≥ the threshold; otherwise it is sent to a lawyer. Each
threshold is chosen on the validation set (smallest threshold reaching the target precision there), then applied
unchanged to the test set. Precision = accuracy on the clauses labelled automatically (95 % Wilson interval).

| Target precision (validation) | Threshold | Test coverage | Test precision [95 % CI] | Test clauses sent to review | Test errors sent to review |
|---|---:|---:|---:|---:|---:|
| 0.90 | 0.548 | 94.7 % | 0.8988 [0.8926; 0.9047] | 534 | 25.3 % |
| **0.95 (recommended)** | **0.892** | **78.9 %** | **0.9489 [0.9438; 0.9536]** | **2,112** | **68.6 %** |
| 0.98 | 0.976 | 56.4 % | 0.9770 [0.9727; 0.9806] | 4,359 | 89.9 % |

![Coverage vs precision](../results/figures/coverage_legal-bert_final_20261019-1523.png)

**Interpretation.** The model is **over-confident**: on the 2019 test set it is on average 0.913 sure of its label but
right 0.872 of the time (ECE 0.0425; validation 0.0422), and accuracy is below confidence in every bin above 0.4 —
even in the top bin, which holds 72 % of the clauses, it says 0.983 and is right 0.960 of the time. With the
recommended threshold of 0.892 (chosen on validation for 95 % precision), a law firm reviewing 10,000 clauses would
let the model label 7,888 of them (78.9 %) on its own, 94.9 % of them correctly [95 % CI 94.4 %; 95.4 %], and send
2,112 to a lawyer; those 2,112 contain 68.6 % of the model's errors, while 403 wrong labels (5.1 % of the automatic
ones) would go through unreviewed.

**Limitations.** No temperature scaling (only the probability of the predicted class is stored, not the full
probability matrix). Top-label calibration says nothing about the 99 other probabilities or the top-5 suggestions.
One global threshold for the 100 classes: the precision per class at that threshold was not measured. The test
precisions sit 0.0011 to 0.0030 below their target (each target inside the 95 % interval), as expected from a
threshold chosen to just reach its target on validation.

## 8. Known limitations

- No "contract type" metadata in LEDGAR: the per-contract breakdown relies on a proxy (party roles detected by
  rules).
- Chronological split: future performance may drop as drafting practices evolve.

## 9. Reproducibility

Seed 42; configuration and versions of every run in `results/runs/<run_id>/config.json`.
