# Model card — Clause Review Assistant (NLP core)

> Skeleton created on 14/10/2026. The quantified sections are filled **only** from the results in `results/`
> (stream C, task C9). Sentences marked *(provisional)* are to be confirmed or corrected by the audit.
>
> **Audit status:** per-slice and bounded counterfactual results are pending execution against the final model checkpoint
> and test prediction files. Notebook 05 compares two dense encoders, BM25, and a MiniLM+BM25 hybrid on a fixed
> stratified 5% test sample. Draft lawyer queries are not evaluated because domain-expert relevance judgments are
> unavailable. Empty result cells below are intentionally not estimates.

## 1. Description

| | |
|---|---|
| Components | (1) clause-type classifier, 100 LEDGAR classes; (2) similar-clause search engine |
| Classification model | *to be completed (HF name, revision, full FT or LoRA)* |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (selected for the focused benchmark) |
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

The audit notebook computes slice macro-F1 over the true classes represented in each slice. Training-frequency
groups are defined by terciles of per-class train counts; clause-length groups use training word-count quartiles.
Party-role groups rely on keyword heuristics, can overlap, and are not verified contract-type or party metadata.
Automatic retrieval metrics use a reproducible sample of 5% of each test-label group (at least one query per
represented label), not the full test set. The full training bank is retained. Results remain pending until the
notebook is run.

## 5. Per-slice performance

| Slice | Macro-F1 | Support | Comment |
|---|---:|---:|---|
| Class frequency: head | | | |
| Class frequency: middle | | | |
| Class frequency: tail | | | |
| Clause length: short / medium / long | | | |
| Party roles: Borrower / Lender | | | |
| Party roles: Employer / Employee | | | |
| Party roles: Landlord / Tenant | | | |
| Party roles: Buyer / Seller | | | |

## 6. Robustness and bias

Counterfactuals are generated from a reproducible random sample of at most 1,000 test clauses that contain the
relevant text feature; one feature is changed per pair. These are heuristic stress tests, not legal redrafts, and
should be manually inspected. Their results are limited to this bounded sample and the specific transformations.
Synthetic European/French OOD probes and retrieval-stability tests are not part of the focused final audit. Results
are pending execution of [`notebooks/06_bias_audit.ipynb`](../notebooks/06_bias_audit.ipynb).

| Perturbation | Prediction flip rate | Confidence change |
|---|---:|---:|
| Swapping party roles | | |
| Pronouns he ↔ she | | |
| Jurisdiction (New York → Luxembourg…) | | |
| Currency (USD → EUR) | | |

## 7. Retrieval evaluation

Automatic relevance is defined as matching the query clause's LEDGAR label. Metrics are computed on the fixed,
stratified 5% sample of the test set and must not be presented as full-test estimates. Manual-query results are
not reported: the drafted questions lack domain-expert relevance judgments. Therefore, the automatic benchmark
does not establish that retrieved clauses satisfy detailed legal constraints such as notice periods or governing
jurisdiction.

| Method | P@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| MiniLM | | | |
| BGE-small | | | |
| BM25 | | | |
| MiniLM + BM25 (RRF) | | | |

## 8. Calibration and human review threshold

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

**Per class at the recommended threshold.** Numbers from
[`results/calibration_per_class_legal-bert_final_20261019-1523.csv`](../results/calibration_per_class_legal-bert_final_20261019-1523.csv)
(`scripts/calibration_per_class.py`, test set, threshold 0.892 unchanged). Precision of a class = share of the clauses
labelled automatically *as* that class that truly belong to it; it is unreliable below 20 such clauses. Frequency
group = tercile of the class's training-set count: tail 23–341, middle 345–569, head 578–3,167 clauses.

![Precision per class at the recommended threshold](../results/figures/precision_per_class_at_threshold.png)

| Group | Classes | Share of its test clauses labelled automatically | Precision of the automatic labels | Classes never labelled automatically | Classes below 0.90 (≥ 20 automatic labels) |
|---|---:|---:|---:|---:|---:|
| Tail | 34 | 63.2 % | 0.948 | 13 | 4 (2) |
| Middle | 33 | 61.9 % | 0.946 | 4 | 5 (5) |
| Head | 33 | 88.4 % | 0.950 | 0 | 2 (2) |

- **11 classes fall below 0.90 precision**: Effectiveness (0.750), Authority (0.808), Payments (0.827), Agreements
  (0.842), Representations (0.848), Miscellaneous (0.852), Binding Effects (0.867), Specific Performance (0.871),
  Vesting (0.885), Consents (0.887) and Employment (0.889). Effectiveness and Agreements rest on fewer than 20
  automatic labels (8 and 19) and are unreliable. Only Payments has its whole 95 % interval below 0.90
  ([0.752; 0.883], 127 automatic labels); the other ten intervals contain 0.90.
- **They are not mostly tail classes** (4 tail, 5 middle, 2 head), and the precision is the same in the three
  groups. Rare classes pay instead in **coverage and hidden errors**: 17 classes (13 tail, 4 middle) are never
  predicted at or above the threshold, yet 74 of their 586 test clauses still pass automatically under a wrong label;
  and 14.2 % of the tail clauses labelled automatically are wrong (115 of 811), vs 2.6 % for head classes (148 of
  5,611).
- **Recommendation**: the global threshold delivers ≈ 0.95 on average, not for every class. A lawyer should
  spot-check the automatic labels of the classes above, and of contracts rich in rare clause types. Per-class
  thresholds, if wanted, must be chosen on the validation set.

**Limitations.** No temperature scaling (only the probability of the predicted class is stored, not the full
probability matrix). Top-label calibration says nothing about the 99 other probabilities or the top-5 suggestions.
One global threshold for the 100 classes: the per-class precision rests on 0 to 545 automatic labels per class and was
measured on the test set only after the threshold was fixed. The test
precisions sit 0.0011 to 0.0030 below their target (each target inside the 95 % interval), as expected from a
threshold chosen to just reach its target on validation.

## 9. Known limitations

- No "contract type" metadata in LEDGAR: the per-contract breakdown relies on a proxy (party roles detected by
  rules).
- The shared LEDGAR loader does not expose row-level contract years. The test split is 2019 only, so no
  within-test year comparison is reported.
- Rule-based party detection can miss uncommon role terms, ambiguous drafting, or clauses that mention more than
  one role pair.
- Counterfactual replacements are lexical stress tests and may alter legal meaning or produce awkward wording;
  generated examples must be reviewed before interpreting changes.
- The OOD probe set is small, synthetic, and unlabelled; its confidence values are descriptive only and are not
  estimates of performance on European or French contracts.
- Chronological split: future performance may drop as drafting practices evolve.

## 10. Reproducibility

Seed 42; configuration and versions of every run in `results/runs/<run_id>/config.json`.
