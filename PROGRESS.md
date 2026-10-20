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
LoRA r=8 on Legal-BERT (A7, Colab, same recipe): 371,812 trainable parameters (0.34 % of full FT), adapter of
1.49 MB on disk vs 438.3 MB, −24.8 % training time, −39.2 % peak GPU memory, **test macro-F1 0.7691 vs 0.7905**; the
paired test (A8, secondary comparison) finds full FT significantly better: **ΔF1 = +0.0214 [+0.0126; +0.0365]**,
McNemar p = 1.7e-15. Only r = 8 was run (pre-registered: r ∈ {4, 8, 16}).
Calibration (A9): the final model is over-confident (test ECE 0.0425, mean confidence 0.9131 for accuracy 0.8717);
human-review threshold 0.892 chosen on validation → **78.9 % of the test clauses handled automatically at precision
0.9489**. Per class at that threshold: 11 classes below 0.90 precision (9 with ≥ 20 automatic labels), not mostly tail
(4 tail, 5 middle, 2 head); rare classes pay in coverage and hidden errors instead (tail clauses 63.2 % automatic vs
88.4 % head, and 14.2 % of the automatic ones wrong vs 2.6 %).
**Next main step: TF-IDF + LogReg predictions (B5) for the pre-registered main comparison (A8,
`scripts/paired_tests.py` ready); LSTM predictions (B6) for the third pair; MiniLM index (C) in parallel.**

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
- [x] Implemented reusable MiniLM dense retrieval with normalised cosine search and a validated embedding cache.
- [x] Created notebook 02 with 10 qualitative searches (5 lawyer questions + 5 validation clauses).
- [ ] Run notebook on the full LEDGAR data, build the index, and complete the 10 manual relevance checks.
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
- [x] LoRA path (`--lora-r`) + trade-off table script, checked locally (19/10).
- [x] Calibration (ECE, reliability diagram) + human-review threshold of the final model (A9, 19/10), with the
  per-class precision at the recommended threshold (head / middle / tail).
- [x] LoRA run + trade-off table (A7, 20/10) — r = 8 only, r = 4 / 16 not run (stated deviation).
- [ ] Paired tests (main comparison + secondary ones): script done, full FT vs LoRA r=8 done (20/10); BERT vs TF-IDF
  (main) and BERT vs LSTM wait for stream B's prediction files.
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

### Calibration and human-review threshold of the final model (A9, 19/10)

`python scripts/calibration_report.py --run-id legal-bert_final_20261019-1523 --target-precision 0.95` →
`results/calibration_legal-bert_final_20261019-1523.json`,
`results/figures/reliability_legal-bert_final_20261019-1523.png`,
`results/figures/coverage_legal-bert_final_20261019-1523.png`. Confidence = probability of the predicted class
(`proba_pred`), i.e. **top-label** calibration; ECE over 15 equal-width bins.

| Split | Clauses | Accuracy | Mean confidence | Confidence − accuracy | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|
| Validation (2018) | 10,000 | 0.8729 | 0.9142 | +0.0413 | 0.0422 |
| Test (2019) | 10,000 | 0.8717 | 0.9131 | +0.0414 | 0.0425 |

Each threshold is **chosen on validation** (smallest threshold of a 0.001 grid whose validation precision reaches the
target), then **applied unchanged to test**. Precision = accuracy on the clauses handled automatically (confidence ≥
threshold); 95 % Wilson interval on test.

| Target precision | Threshold (val.) | Val. coverage | Val. precision | Test coverage | Test precision [95 % CI] | Test clauses sent to review | Test errors sent to review |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.548 | 94.9 % | 0.9001 | 94.7 % | 0.8988 [0.8926; 0.9047] | 534 | 25.3 % |
| **0.95 (recommended)** | **0.892** | 79.0 % | 0.9502 | **78.9 %** | **0.9489 [0.9438; 0.9536]** | 2,112 | 68.6 % |
| 0.98 | 0.976 | 57.3 % | 0.9801 | 56.4 % | 0.9770 [0.9727; 0.9806] | 4,359 | 89.9 % |

- **Over-confidence by bin**: in each of the 9 bins above 0.4 (9,854 validation / 9,861 test clauses), mean
  confidence exceeds accuracy, by 0.021 to 0.130 (validation) and 0.023 to 0.151 (test). The top bin (0.933, 1]
  holds 7,291 / 7,224 clauses: mean confidence 0.983, accuracy 0.963 / 0.960. Below 0.4 the bins hold at most 61 / 75
  clauses (146 / 139 in total): their accuracy is noisy and falls on both sides of the diagonal.
- **One year of drift (2018 → 2019)**: calibration is unchanged (ECE 0.0422 → 0.0425) and the test coverage stays
  within 0.9 pt of validation at every threshold. The three test precisions are 0.0011 to 0.0030 below their target,
  each target inside the 95 % interval: expected, since each threshold is the smallest that just reaches its target
  on validation.
- **Interpretation.** The model is **over-confident**: on the 2019 test set it is on average 0.913 sure of its label
  but right 0.872 of the time (ECE 0.0425; validation 0.0422), and accuracy is below confidence in every bin above 0.4
  — even in the top bin, which holds 72 % of the clauses, it says 0.983 and is right 0.960 of the time. With the
  recommended threshold of 0.892 (chosen on validation for 95 % precision), a law firm reviewing 10,000 clauses would
  let the model label 7,888 of them (78.9 %) on its own, 94.9 % of them correctly [95 % CI 94.4 %; 95.4 %], and send
  2,112 to a lawyer; those 2,112 contain 68.6 % of the model's errors, while 403 wrong labels (5.1 % of the automatic
  ones) would go through unreviewed.

**Per class at the recommended threshold (test).**
`python scripts/calibration_per_class.py --run-id legal-bert_final_20261019-1523` →
`results/calibration_per_class_legal-bert_final_20261019-1523.csv` (one row per class),
`results/figures/precision_per_class_at_threshold.png`. The threshold 0.892 is read from the calibration JSON (chosen
on validation); nothing is selected on test. Per class: `support` (test clauses of the class), `share_auto` (share of
them labelled automatically), `precision_auto` = precision of the automatic labels of the class (share of the clauses
labelled automatically *as* that class that truly belong to it, as in sklearn; 95 % Wilson interval), `reliable` = at
least 20 such automatic labels. Frequency group = tercile of the class's training-set count
(`src.data.frequency_groups`): tail 23–341, middle 345–569, head 578–3,167 training clauses.

| Group | Classes | Test support | Share labelled automatically | Precision of the automatic labels (pooled) | Classes never labelled automatically | Unreliable (< 20 automatic labels) | Below 0.90 (of which reliable) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tail | 34 | 1,283 | 63.2 % | 0.9482 | 13 | 17 | 4 (2) |
| Middle | 33 | 2,367 | 61.9 % | 0.9465 | 4 | 7 | 5 (5) |
| Head | 33 | 6,350 | 88.4 % | 0.9496 | 0 | 0 | 2 (2) |

| Class below 0.90 | Group | Train clauses | Test support | Share labelled automatically | Automatic labels | Precision [95 % CI] | Reliable |
|---|---|---:|---:|---:|---:|---:|---|
| Effectiveness | tail | 229 | 31 | 25.8 % | 8 | 0.750 [0.409; 0.929] | no |
| Authority | middle | 549 | 62 | 38.7 % | 26 | 0.808 [0.621; 0.915] | yes |
| Payments | head | 856 | 126 | 88.9 % | 127 | 0.827 [0.752; 0.883] | yes |
| Agreements | tail | 252 | 48 | 47.9 % | 19 | 0.842 [0.624; 0.945] | no |
| Representations | middle | 569 | 94 | 35.1 % | 33 | 0.848 [0.691; 0.933] | yes |
| Miscellaneous | middle | 479 | 79 | 53.2 % | 27 | 0.852 [0.675; 0.941] | yes |
| Binding Effects | head | 588 | 126 | 44.4 % | 60 | 0.867 [0.758; 0.931] | yes |
| Specific Performance | tail | 326 | 43 | 65.1 % | 31 | 0.871 [0.711; 0.949] | yes |
| Vesting | middle | 451 | 87 | 89.7 % | 87 | 0.885 [0.801; 0.936] | yes |
| Consents | middle | 444 | 81 | 64.2 % | 53 | 0.887 [0.774; 0.947] | yes |
| Employment | tail | 308 | 47 | 53.2 % | 27 | 0.889 [0.719; 0.961] | yes |

- **11 of the 83 classes with at least one automatic label fall below 0.90 precision**, 9 of them with ≥ 20
  automatic labels (Authority, Payments, Representations, Miscellaneous, Binding Effects, Specific Performance,
  Vesting, Consents, Employment); Effectiveness (8) and Agreements (19) are unreliable. Authorizations sits exactly
  at 0.900; 45 classes reach 0.95.
- **They are not mostly tail classes**: 4 tail (2 of them unreliable), 5 middle, 2 head — 2 tail among the 9
  reliable ones. The pooled precision is the same in the three groups (0.948 / 0.946 / 0.950).
- **Only Payments is clearly below 0.90**: its whole 95 % interval lies below 0.90 ([0.752; 0.883]); 22 of its 127 automatic
  labels go to clauses of other classes. The 10 other intervals contain 0.90: with 8 to 87 automatic labels per
  class, "below 0.90" is an estimate, not an established fact.
- **Rarity costs coverage and hidden errors, not precision.** Tail clauses are labelled automatically less often
  (63.2 % vs 88.4 % for head). 17 classes (13 tail, 4 middle) are never predicted at or above the threshold, so their
  precision is undefined (not drawn); their clauses are not protected for all that: 74 of their 586 test clauses still
  pass automatically, all under a wrong label (Applicable Laws: 34 of 53). Counting errors by the **true** class
  (`n_auto − n_auto_correct`), 115 of the 811 tail clauses labelled automatically are wrong (14.2 %), vs 140 of 1,466
  (9.5 %) middle and 148 of 5,611 (2.6 %) head; 290 of the 403 automatic errors carry a head-class label. The model
  rarely predicts a rare class with confidence, but confidently gives rare-class clauses a wrong, mostly frequent,
  label — which a per-class precision cannot show.
- **Consequence**: the global threshold delivers ≈ 0.95 on average and in each frequency group, not in each class.
  Per-class (or per-group) thresholds would have to be chosen on validation, never on these test numbers, and would
  be noisy: 24 classes have fewer than 20 automatic labels on test.
- **Limitations**: (1) no temperature scaling — the common prediction format stores only `proba_pred` and `top5`, not
  the full probability matrix (or logits) that temperature scaling is fitted on; (2) top-label calibration only: it
  says nothing about the 99 other probabilities, nor about the reliability of the `top5` suggestions; (3) one global
  threshold for the 100 classes: the per-class precision above rests on 0 to 545 automatic labels per class, and it
  was measured on test only after the threshold was fixed; (4) the ECE depends on the binning, and 72-73 % of the
  clauses fall in the last of the 15 bins.

### Full fine-tuning vs LoRA (A7, 20/10) and paired test (A8, secondary comparison)

Run `legal-bert_lora-r8_20261020-0940` (notebook 04, section 10): `nlpaueb/legal-bert-base-uncased` (revision
`15b570cb`), LoRA r = 8, α = 16, dropout 0.1 on `query` / `value` of the 12 layers, `classifier` trained in full;
recipe of the full run (`max_length` 256, 3 epochs, batch 16, 10 % warmup, weight decay 0.01, seed 42, 11,250 steps),
only the learning rate differs (5e-4 vs 3e-5); Colab T4, fp16, transformers 5.20.1, peft 0.21.2; `--eval-test`, test
evaluated once. `python scripts/lora_tradeoff_table.py --full-run legal-bert_final_20261019-1523` →
`results/lora_tradeoff.csv` (no recipe warning):

| Run | Method | lr | Trainable params | % of full | Train time | Peak GPU memory | Weights on disk | Val macro-F1 | Val accuracy | Test macro-F1 | Test accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `legal-bert_final_20261019-1523` | full fine-tuning | 3e-5 | 109,559,140 | 100.00 | 2,688.2 s (44.8 min) | 3,106.2 MB | 438.3 MB | **0.7930** | **0.8729** | **0.7905** | **0.8717** |
| `legal-bert_lora-r8_20261020-0940` | LoRA r=8 | 5e-4 | 371,812 | 0.34 | 2,021.5 s (33.7 min) | 1,889.0 MB | 1.49 MB | 0.7552 | 0.8524 | 0.7691 | 0.8527 |
| LoRA − full | | | ÷ 295 | | −24.8 % | −1,217.2 MB (−39.2 %) | ÷ 293 | −0.0378 | −0.0205 | −0.0214 | −0.0190 |

- **Weights on disk**: `adapter_model.safetensors` = **1,494,128 bytes** (50 tensors: 24 `lora_A`, 24 `lora_B`, the
  classifier weight and bias; 371,812 fp32 parameters) vs `model.safetensors` = **438,260,096 bytes** for the full
  model (109,559,140 fp32 parameters): the adapter is **0.34 %** of the full file. The Colab checkpoints are on Drive,
  so both sizes were measured locally on files with the same tensors: the adapter of the local smoke run
  `legal-bert_smoke-lora-r8_20261019-1822` (same encoder, r, targets and head) and a 100-label Legal-BERT saved with
  `save_pretrained` (same architecture as the full checkpoint); a safetensors file is 4 bytes per fp32 parameter plus a
  small header, so the sizes carry over. Serving a LoRA model still needs the shared pretrained Legal-BERT; the
  adapter is what has to be stored per fine-tuned model. Optimizer states in the checkpoint folders are not counted.
- **LoRA was still improving at epoch 3**: validation macro-F1 0.6955 → 0.7353 → **0.7552** (+0.0199 from epoch 2 to
  3), accuracy 0.8165 → 0.8421 → 0.8524, loss 0.659 → 0.582 → 0.542; best epoch = last epoch (`checkpoint-11250`).
  Training loss 4.514 (step 225) → 0.489 (step 11,250), vs 0.329 for full FT. The full run was also still improving
  (+0.0209), and the validation gap did not shrink over the epochs (−0.0296, −0.0367, −0.0378): with 3 epochs we
  cannot tell whether LoRA would catch up with a longer schedule.
- **Test above validation for LoRA** (macro-F1 +0.0139; accuracy +0.0003), while full FT loses 0.0024: the macro-F1
  gap shrinks from −0.0378 (validation) to −0.0214 (test). Part of it is one rare class: 14 `Books` has 2 test clauses
  (none in validation); LoRA labels both right (F1 = 1) and full FT neither (F1 = 0), i.e. +0.01 of macro-F1 for LoRA
  from 2 clauses. Conversely, LoRA never predicts 4 `Applicable Laws` (53 test clauses; full FT: 1 right).
- **Per class (test)**: full FT is better on 80 classes, LoRA on 16, equal on 4; both predict 98 distinct classes. The
  two models agree on 92.9 % of the test clauses.

**Paired test** (pre-registered protocol: paired bootstrap on macro-F1, 1,000 resamples, seed 42, 95 % CI; McNemar
on accuracy). `python scripts/paired_tests.py --pair legal-bert_final_20261019-1523 legal-bert_lora-r8_20261020-0940`
→ `results/paired_tests.csv`. ΔF1 = F1(full) − F1(LoRA) on the 10,000 test clauses:

| A | B | n | Macro-F1 A | Macro-F1 B | ΔF1 [95 % CI] | Significant | Accuracy A | Accuracy B | b (A right, B wrong) | c (A wrong, B right) | McNemar p |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| full FT | LoRA r=8 | 10,000 | 0.7905 | 0.7691 | **+0.0214 [+0.0126; +0.0365]** | yes | 0.8717 | 0.8527 | 377 | 187 | 1.7e-15 (χ² = 63.3, with continuity correction) |

- **Result**: full fine-tuning is significantly better than LoRA r = 8 on both metrics: the 95 % interval of ΔF1
  excludes 0 (LoRA never matches full FT in the 1,000 resamples) and full FT is right on 377 clauses where LoRA is
  wrong vs 187 the other way. The trade-off is therefore real: −2.1 pt macro-F1 and −1.9 pt accuracy for 0.34 % of
  the trainable parameters, a 293 × smaller file, −25 % training time and −39 % GPU memory.
- The interval is asymmetric around +0.0214: macro-F1 averages 100 classes, some with 2 to 10 test clauses, whose
  F1 jumps when they are resampled.
- **Deviations from the pre-registered secondary comparison** (README: r ∈ {4, 8, 16}): only r = 8 was run; r = 4
  and r = 16 can be added later (same cell, `--lora-r 4` / `16`, each evaluated once). The two runs share every
  setting but the learning rate (5e-4 for LoRA, 3e-5 for full FT, each in the S06 range of its method); the full run
  used transformers 5.18.0 and the LoRA run 5.20.1 (Colab image update).

## 4. Problems met and solutions

### `peft` fails to import on Colab: incompatible `torchao` (20/10)

**Symptom:** on Colab, the LoRA path (notebook 04, section 10) failed with
`ImportError: Found an incompatible version of torchao (0.10.0, need > 0.16.0)` raised by `peft`.
**Cause:** Colab preinstalls `torchao` 0.10.0. `peft` (`import_utils.is_torchao_available`) raises an `ImportError`
when `torchao` is installed but older than 0.16.0, instead of treating it as absent; the project never uses `torchao`
(it is not in `requirements.txt`).
**Solution:** notebook 04, install cell: `!pip uninstall -y -q torchao` after `pip install -r requirements.txt`. No
change to `requirements.txt`.
**Status:** ✅ solved in the notebook.

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
   comparison A8 (Model A = `legal-bert_final_20261019-1523`). Then Robin runs
   `python scripts/paired_tests.py --pair legal-bert_final_20261019-1523 <tfidf_run_id>` (and the same with the LSTM
   run of B6), adding `--exclude-idx <file>` once the near-duplicate test idx of B4 exist. Needed in
   `results/predictions/`: `<tfidf_run_id>_test.parquet` and `<lstm_run_id>_test.parquet` (gitignored: copy them).
4. Robin: copy `results/predictions/legal-bert_final_20261019-1523_{validation,test}.parquet` and
   `legal-bert_lora-r8_20261020-0940_{validation,test}.parquet` from the Drive clone (gitignored, absent from local
   checkouts) to the machines running A8 and C8 (done on Robin's laptop).
5. Robin (optional, if GPU time allows): LoRA r = 4 and r = 16 (notebook 04, section 10, `--lora-r 4` / `16`, once
   each), then re-run the trade-off table and the full vs LoRA paired tests.
6. Team: confirm the `metrics.json` layout per split (see the 18/10 log entry) before B5 writes its first run.
7. Parallel work: B2-B5 (EDA, TF-IDF), C1-C3 (index, queries).
8. Brian: C6 / C8 — take the head / middle / tail slice from `src.data.frequency_groups` (already used by A9) so that
   the audit and the calibration report the same groups.

## 6. Rule for updating this file

After every significant step, state: what was done; the result obtained; the numbers or metrics; the problems met;
the solution adopted; the next action.

Never replace a real result with an assumption. A task stays `[ ]` until it has actually been run or confirmed.
Any deviation from the pre-registered comparison is logged here.

## 7. Chronological log

### 20 October 2026 — stream A: paired-test script, full FT vs LoRA (A8, secondary comparison, Robin)

- **Code**: `src/comparison.py` — `load_aligned_predictions` (reads `results/predictions/<run_id>_<split>.parquet`
  of both runs, refuses missing files and any difference in `idx` or `label`, length or order),
  `paired_comparison` (macro-F1 and accuracy of A and B with `classification_metrics(num_labels=100)`, then
  `paired_bootstrap_f1` and `mcnemar_test` from `src.metrics`, not re-implemented), `compare_runs` (row "all", plus
  a "deduplicated" row without the idx of an `--exclude-idx` file; idx absent from the split are refused),
  `update_results_csv` (replaces the earlier rows of the same pair, split and variant, keeps the others). CLI:
  `scripts/paired_tests.py --pair <A> <B> [--pair ...] [--split test] [--exclude-idx <file>]`; it stops before
  computing anything if a prediction file is missing.
- **Protocol**: as pre-registered (1,000 resamples, seed 42, 95 % percentile interval; McNemar with continuity
  correction, exact binomial if b + c < 25), test split, ΔF1 = F1(A) − F1(B).
- **Run**: full FT `legal-bert_final_20261019-1523` vs LoRA r=8 `legal-bert_lora-r8_20261020-0940` →
  **ΔF1 = +0.0214 [+0.0126; +0.0365], significant**; accuracy 0.8717 vs 0.8527, b = 377, c = 187, McNemar
  p = 1.7e-15. The macro-F1 and accuracy recomputed from the prediction files equal the `test` blocks of both
  `metrics.json`. Details in section 3.
- **Main comparison not run**: BERT vs TF-IDF + LogReg (H1) and BERT vs LSTM need stream B's test predictions; no
  TF-IDF or LSTM file is in `results/predictions/` (nor any run folder in `results/runs/`). H1 is not tested yet.
  A8 stays open.
- **Tests**: 3 new in `tests/test_comparison.py` (synthetic files): identical models → ΔF1 = 0, CI [0; 0], b = c = 0,
  McNemar p = 1; shifted `idx`, changed labels and a missing file are refused; deduplicated row size, CSV rows
  replaced and not duplicated, exclude idx outside the split refused. `python -m pytest -q` → 34 passed.

### 20 October 2026 — stream A: LoRA r=8 run and trade-off table (A7, Robin)

- **Colab fix**: `peft` raised `ImportError: Found an incompatible version of torchao (0.10.0, need > 0.16.0)`;
  Colab preinstalls `torchao`, which the project never uses. Notebook 04, install cell: `!pip uninstall -y -q
  torchao` (section 4; lesson in `tasks/lessons.md`).
- **Run** `legal-bert_lora-r8_20261020-0940` (notebook 04, section 10, `--eval-test`, test evaluated once): run folder
  committed, prediction files copied to `results/predictions/` (gitignored). Same recipe as the full run, only the
  learning rate differs (5e-4); `lora_tradeoff_table.py` raised no recipe warning.
- **Result** (`results/lora_tradeoff.csv`): 371,812 trainable parameters (0.34 % of full FT), adapter file 1.49 MB vs
  438.3 MB, 2,021.5 s (−24.8 %), peak GPU memory 1,889.0 MB (−39.2 %); validation macro-F1 0.7552 (full 0.7930),
  **test macro-F1 0.7691** (full 0.7905), test accuracy 0.8527 (full 0.8717). LoRA was **still improving at epoch 3**
  (validation macro-F1 +0.0199 from epoch 2 to 3, best epoch = last). Table and comments in section 3.
- **Deviation**: only r = 8 was run; the secondary comparison pre-registered in the README names r ∈ {4, 8, 16}.
  r = 4 / 16 would be added with the same cell if GPU time allows. A7 ticked with this deviation stated.

### 19 October 2026 — stream A: per-class view of the human-review threshold (A9, Robin)

- **Code**: `src.data.frequency_groups` (training count and head / middle / tail group of every class, terciles of the
  per-class training counts with `pd.qcut`: 34 / 33 / 33 classes on LEDGAR) — shared definition, to be reused by
  the audit slices (C6, C8); `src.calibration.per_class_operating_point` (per class at a fixed threshold: support,
  share labelled automatically, precision of the automatic labels per predicted class + Wilson interval, reliable
  flag) and `per_class_precision_figure`. CLI: `scripts/calibration_per_class.py --run-id <run>`.
- **Protocol**: the threshold is read from `results/calibration_<run_id>.json` (recommended target 0.95, chosen on
  validation) and applied to the test predictions; the script checks that the per-class counts add up to the global
  operating point (7,888 automatic clauses) and that the labels match the LEDGAR test split.
- **Result** (`results/calibration_per_class_legal-bert_final_20261019-1523.csv`,
  `results/figures/precision_per_class_at_threshold.png`): 11 classes below 0.90 precision, 9 of them with ≥ 20
  automatic labels; 4 tail, 5 middle, 2 head, so not mostly tail. Only Payments has its whole 95 % interval below
  0.90. Rare classes pay elsewhere: 17 classes (13 tail) are never predicted at or above the threshold, and 14.2 % of
  the tail clauses labelled automatically are wrong vs 2.6 % for head. Details in section 3.
- **Tests**: 2 new in `tests/test_calibration.py` (per-class counts by true and predicted class on a hand-checked
  example; per-class counts add up to the global operating point; figure written), 2 in `tests/test_data.py`
  (tercile groups, equal counts kept together, label check). `python -m pytest -q` → 31 passed.

### 19 October 2026 — stream A: calibration and human-review threshold (A9, Robin)

- **Code**: `src/calibration.py` — `calibration_bins` (15 equal-width bins (low, high]), `expected_calibration_error`,
  `coverage_precision_curve` (per threshold: coverage, precision = accuracy on the clauses handled automatically,
  share of the errors sent to review), `select_threshold` (smallest threshold of a 0.001 grid reaching the target
  precision, `None` if none does), `operating_point` (+ 95 % Wilson interval of the precision), `calibration_report`,
  `reliability_diagram`, `coverage_precision_figure`. CLI: `scripts/calibration_report.py --run-id <run>
  --target-precision 0.95` (the 0.90 and 0.98 targets are also reported, `--also-report`).
- **Protocol**: every threshold is chosen on validation only, then applied as is to test; the test predictions are
  only read, the model is not re-evaluated.
- **Run** on `legal-bert_final_20261019-1523` (its two prediction files copied from the Drive clone): recommended
  threshold **0.892** (target precision 0.95 on validation) → test coverage **78.9 %**, test precision **0.9489**
  [0.9438; 0.9536]; ECE 0.0422 (validation) / 0.0425 (test). Full tables in section 3. The JSON also stores the
  15 bins of each split (data of the reliability diagram).
- **Deviation from prompt P4**: no temperature scaling (`fit_temperature` not written). The prediction files of the
  common format hold only `proba_pred` and `top5`, not the full probability matrix that temperature scaling needs:
  calibration is measured on the probability of the predicted class only (top-label calibration). Adding it would
  require re-predicting validation and test from the checkpoint on Colab with the logits saved, and fitting the
  temperature on validation; it would not change the predicted class (so neither accuracy nor macro-F1), only the
  confidences and therefore the thresholds.
- **Interpretation**: the model is over-confident (mean confidence 0.913 vs accuracy 0.872 on test, accuracy below
  confidence in every bin above 0.4). At the recommended threshold, a firm would have 78.9 % of the clauses labelled
  automatically at 94.9 % precision and 21.1 % sent to a lawyer, who would see 68.6 % of the model's errors.
- **Tests**: 8 new in `tests/test_calibration.py` (synthetic data): ECE < 0.01 on 200,000 perfectly calibrated
  draws, exact ECE values, bin edges, threshold 0 → coverage 1, smallest threshold reaching the target (and `None`
  when unreachable), operating-point counts and interval, threshold chosen on validation and read on test, figures
  written. `python -m pytest -q` → 27 passed.

### 19 October 2026 — stream A: LoRA path and trade-off table script (A7 code, Robin)

- **Code**: `config["lora"]` implemented in `src/models/transformer.py` with `peft`
  (`LoraConfig(task_type=SEQ_CLS, r, lora_alpha=2r, lora_dropout=0.1)`). Target modules per architecture
  (`LORA_TARGET_MODULES`, checked with `print(model)`): `query`/`value` (BERT, Legal-BERT), `q_lin`/`v_lin`
  (DistilBERT), `query_proj`/`value_proj` (DeBERTa-v2/v3). The head is trained in full (`modules_to_save`): every
  module of the task model outside the pretrained encoder, i.e. `classifier` for BERT, `pre_classifier` +
  `classifier` for DistilBERT (peft's default would leave its randomly initialised `pre_classifier` frozen),
  `pooler` + `classifier` for DeBERTa. The resolved settings are recorded in `config.json` (`lora` block).
- **CLI**: `scripts/train_transformer.py --lora-r <r>`; the learning rate defaults to 5e-4 with `--lora-r`
  (S06: 1e-4 to 1e-3), 5e-5 otherwise, unless `--lr` is given. The run prints its trainable parameters and their
  share of full fine-tuning.
- **Parameter counts** (`count_parameters`): peft keeps a frozen copy of the original head next to the trained one;
  that copy is never used and not counted. For a LoRA run, `n_total_params` = encoder + LoRA matrices + head.
- **Trade-off table**: `scripts/lora_tradeoff_table.py --full-run <run_id>` reads the full fine-tuning run and every
  LoRA run of the same encoder on the full data, writes `results/lora_tradeoff.csv` and prints a markdown table
  (trainable parameters, % of full, training time, peak GPU memory, validation / test macro-F1 and accuracy). It
  warns when a LoRA run does not share the recipe of the full run (`max_length`, epochs, batch size, warmup,
  weight decay, seed). Checked on the current runs: one row (`legal-bert_final_20261019-1523`), smoke runs
  excluded; the CSV is written once the LoRA runs exist.
- **Notebook 04, section 10**: one cell, LoRA r=8 on `nlpaueb/legal-bert-base-uncased` with the recipe of
  `legal-bert_full_20261019-0916` (`max_length` 256, 3 epochs, batch 16, 10 % warmup, weight decay 0.01, seed 42);
  only the learning rate changes (5e-4). `--eval-test`: LoRA runs are final models of the secondary comparison,
  each evaluated once. r = 4 and r = 16 are added if GPU time allows; if they are not run, the secondary comparison
  pre-registered in the README (r ∈ {4, 8, 16}) is reported with r = 8 only, as a stated deviation.
- **Tests**: 5 new (tiny random BERT / DistilBERT, no download): r·(d_in + d_out) trainable parameters per targeted
  matrix plus the head, only LoRA and head parameters trainable, output at step 0 = base model (B = 0), settings
  not modified by peft, unknown architecture refused, trade-off table rows / filters / recipe warning.
  `python -m pytest -q` → 19 passed.
- **Local smoke run** `legal-bert_smoke-lora-r8_20261019-1822` (MPS, first 200 examples per split, 1 epoch =
  13 steps, batch 16, `max_length` 256, lr 5e-4), Legal-BERT revision `15b570cb` (same as Colab):
  - load report: only `classifier.weight` / `classifier.bias` are newly initialised (the pooler is pretrained),
    so the head = `classifier`;
  - **trainable parameters with r = 8: 371,812** = 147,456 (`lora_A`) + 147,456 (`lora_B`) + 76,900 (head)
    = 12 layers × 2 matrices × 8 × (768 + 768) + 768 × 100 + 100, i.e. **0.34 % of the 109,559,140** of full
    fine-tuning; `n_total_params` 109,854,052;
  - the best checkpoint holds only the 24 `lora_A`, 24 `lora_B` and the classifier; every `lora_B` moved away from
    its zero initialisation (norms 0.089 to 0.105);
  - 24 s of training; training loss 4.681 → 4.498 over 13 steps (noisy); validation macro-F1 0.0016, accuracy
    0.015, 9 distinct classes predicted: not a reportable score, the model has barely started.
- **Extra check** (scratch, not in `results/`): same set-up for 10 epochs (130 steps) on the same 200 clauses:
  training loss 4.719 → 3.408 (mean of the last 5 logged points 3.647), validation accuracy 0.015 → 0.075, loss
  4.528 → 4.398. The LoRA path learns; at this scale it learns slowly, so whether 3 epochs at lr 5e-4 are enough
  on the full data will be read on the Colab validation curve.
- **Problems met**: (1) for `SEQ_CLS`, peft appends its default head names to the `modules_to_save` list it
  receives, in place: the first smoke run recorded `["classifier", "classifier", "score"]` (the count was right,
  `score` does not exist in BERT). Fixed by passing copies; that run was deleted and re-run. (2) The local process
  did not exit after its results: transformers 5 starts a background thread that downloads the Hub's safetensors
  conversion of `.bin`-only models (Legal-BERT), and it stalled. Both lessons in `tasks/lessons.md`.
- A7 stays open until the Colab LoRA run(s) exist.

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
