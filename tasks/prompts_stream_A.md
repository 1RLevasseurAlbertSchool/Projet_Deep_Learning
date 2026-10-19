# Claude Code prompts — Stream A (Robin)

> How to use: open Claude Code at the root of the repository and paste one prompt at a time
> (or say: "Execute prompt P1 from `tasks/prompts_stream_A.md`").
> Run them in order. Review the diff before accepting, then push yourself.
> Claude Code reads `CLAUDE.md` automatically: the prompts rely on its rules and interface contracts.

---

## P0 — Session start check (no code)

```text
Read CLAUDE.md, PROGRESS.md, tasks/todo.md and tasks/lessons.md. Then:
1. Run `git status` and `git log --oneline -5` and tell me if anything is uncommitted or if main is behind origin.
2. Run `python -m pytest -q` and report the result.
3. Run `python -m src.data` and confirm the split sizes (60,000 / 10,000 / 10,000) and the 100 labels.
   If loading fails, diagnose the root cause (dataset name, datasets version, network) and propose a fix in
   src/data.py without changing its public interface (load_ledgar, get_label_names, to_pandas).
Do not change any other file. Summarise in 5 lines what is ready for stream A and what is missing.
```

---

## P1 — Tasks A1 + A2 + A3: generic fine-tuning script, token lengths, Colab notebook

```text
Context: stream A of the project (see CLAUDE.md and STRATEGY.md). Create a branch `feat/a-hf-finetuning`.
Goal: one reusable script to fine-tune any Hugging Face encoder on LEDGAR, respecting the interface contracts of
CLAUDE.md (runs in results/runs/, predictions in results/predictions/ via src.utils.save_run / save_predictions).

1. `src/models/transformer.py` with:
   - `token_length_stats(model_name, texts) -> dict`: p50, p90, p95, p99, max and share of texts above 128 / 256 /
     512 tokens, using the model's own AutoTokenizer (no truncation).
   - `train_and_evaluate(config: dict) -> dict` that:
     a. calls `set_seed`, loads data with `load_ledgar(subset=config["subset"])`;
     b. tokenises with `AutoTokenizer.from_pretrained(model_name)`, truncation at `max_length`, NO padding at
        tokenisation time (use `DataCollatorWithPadding` for dynamic padding per batch);
     c. loads `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=100, id2label, label2id)`
        with the label names from `get_label_names`;
     d. trains with the transformers 5 `Trainer` / `TrainingArguments` API (`eval_strategy`, `processing_class`,
        `warmup_steps` as a ratio, `report_to="none"`), AdamW, linear decay, weight decay 0.01,
        `eval_strategy` and `save_strategy` both "epoch", `load_best_model_at_end=True` with
        `metric_for_best_model="macro_f1"`, `save_total_limit=1`,
        fp16 when CUDA is available, MPS when available locally;
     e. `compute_metrics` uses `src.metrics.classification_metrics` (macro-F1 is the selection metric);
     f. records `train_time_s`, `n_trainable_params`, `n_total_params`, and peak GPU memory
        (`torch.cuda.max_memory_allocated`) when CUDA is used;
     g. predicts on the validation set ALWAYS, and on the test set ONLY if `config["eval_test"]` is True;
        softmax the logits and call `save_predictions(run_id, split, labels, preds, probas)`;
     h. calls `save_run(run_id, config, metrics)` with validation (and test if evaluated) metrics,
        `run_id = make_run_id(short_model_name, variant)`.
   - Keep the LoRA hook ready but not implemented: a `config["lora"]` key that is None by default.
2. `scripts/train_transformer.py`: argparse CLI over `train_and_evaluate` with arguments
   --model, --variant, --max-length (default 256), --epochs (3), --lr (5e-5), --batch-size (32),
   --warmup (0.1), --subset (None), --eval-test (flag, default off), --output-dir (checkpoints, default `models/`),
   --seed (42). Print the final metrics table. When --eval-test is set, print a loud warning that the test set must
   be evaluated only once per final model and log the run_id.
3. `scripts/token_lengths.py`: prints `token_length_stats` on the train split for a given --model.
4. `notebooks/04_transformer_finetuning.ipynb` for Google Colab, short markdown in English between cells:
   mount Google Drive; clone the private repo into /content/drive/MyDrive/Projet_Deep_Learning (or `git pull` if it
   exists) using a GitHub token read from Colab secrets (`google.colab.userdata.get("GITHUB_TOKEN")`), never printed;
   `pip install -r requirements.txt`; check the GPU (`nvidia-smi`); one cell per run calling
   `scripts/train_transformer.py` (smoke test, DistilBERT full, other encoders); checkpoints written to Drive.
   Keep outputs small.
5. Tests in `tests/test_transformer.py` that run on CPU in < 1 min: `token_length_stats` on 5 fake texts, and the
   prediction-to-parquet path on a tiny fake probability matrix (no model download in tests).
6. Verification, run locally:
   - `python -m pytest -q`;
   - `python scripts/token_lengths.py --model distilbert-base-uncased` → report the numbers;
   - smoke test: `python scripts/train_transformer.py --model distilbert-base-uncased --variant smoke
     --subset 2000 --epochs 1 --max-length 128` → check that the loss decreases and that the model does not predict
     a single class (report the number of distinct predicted classes on validation).
7. Update PROGRESS.md (log entry with the real numbers, the max_length decision with its justification from the
   token stats) and tick A1, A2, A3 in tasks/todo.md only if they actually ran. Commit on the branch, do not push.
Do not touch src/data.py, src/metrics.py or other streams' folders. Ask me before adding a dependency.
```

---

## Runs A4 → A6 (no new code: commands to launch in the Colab notebook)

```text
# A4 — DistilBERT on the full training set (validation only)
python scripts/train_transformer.py --model distilbert-base-uncased --variant full

# A5 — one bigger encoder if GPU time allows (validation only)
python scripts/train_transformer.py --model nlpaueb/legal-bert-base-uncased --variant full --lr 3e-5 --batch-size 16

# A6 — final run of the encoder chosen ON VALIDATION, with the single test evaluation
python scripts/train_transformer.py --model <chosen_model> --variant final --eval-test
```

After each run, ask Claude Code:

```text
Read results/runs/<run_id>/metrics.json and config.json. Add a dated entry to PROGRESS.md with the real numbers
(validation macro-F1, accuracy, training time, trainable parameters, GPU memory) and update the screening table.
If this was the A6 final run, also record which model was selected and why (validation macro-F1 only), and tick
A4/A5/A6 in tasks/todo.md accordingly. Commit only the metrics/config files and the docs (predictions are
gitignored). Do not push.
```

---

## P2 — Task A7: LoRA

```text
Context: stream A, task A7 (see tasks/todo.md). Create a branch `feat/a-lora`.
Goal: run LoRA on the same encoder as the final full fine-tuning run, with r in {4, 8, 16}, and produce a trade-off
table. Read results/runs/ to find the final model name.

1. Implement the `config["lora"]` hook in `src/models/transformer.py` with the `peft` library:
   `LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=2*r, lora_dropout=0.1, target_modules=...)`.
   Choose target_modules from the architecture: `q_lin`/`v_lin` (DistilBERT), `query`/`value` (BERT, Legal-BERT),
   `query_proj`/`value_proj` (DeBERTa-v3) — check with `print(model)` instead of assuming. The classification head
   must stay fully trainable. Log `n_trainable_params` and the ratio to the full model.
2. Add `--lora-r` to `scripts/train_transformer.py` (default None = full fine-tuning). Default LoRA learning rate:
   5e-4 when --lora-r is set and --lr is not given explicitly.
3. Add a unit test checking, on a tiny randomly initialised BERT config (no download), that LoRA r=8 on
   query/value gives exactly r*(d_in+d_out) trainable parameters per targeted matrix plus the head.
4. Add three cells to notebooks/04_transformer_finetuning.ipynb, one per r, all with --eval-test (LoRA variants are
   final models of the secondary comparison, evaluated once).
5. `scripts/lora_tradeoff_table.py`: reads the full-FT final run and the LoRA runs from results/runs/ and writes
   `results/lora_tradeoff.csv` + a markdown table (trainable params, % of full, train time, peak GPU memory,
   validation and test macro-F1, accuracy).
6. Run the tests locally, update PROGRESS.md and tasks/todo.md (A7) only with real numbers once the Colab runs are
   done. Commit on the branch, do not push.
```

---

## P3 — Task A8: paired tests (main comparison)

```text
Context: stream A, task A8. Create a branch `feat/a-paired-tests`.
The main comparison is pre-registered in README.md: read it first and follow it exactly (paired bootstrap on
macro-F1, 1,000 resamples, seed 42, 95 % CI; McNemar on accuracy, exact binomial if b + c < 25).

1. `scripts/paired_tests.py`:
   - input: one or more pairs `--pair <run_id_A> <run_id_B>`, split = test by default;
   - loads `results/predictions/<run_id>_test.parquet` for both runs, checks that `idx` and `label` are identical
     and in the same order (fail loudly otherwise);
   - calls `src.metrics.paired_bootstrap_f1` and `src.metrics.mcnemar_test` (do not re-implement them);
   - optional `--exclude-idx <file>`: a text/CSV file of test idx to drop (near-duplicates from stream B, task B4),
     to report the de-duplicated variant as an extra row;
   - writes/updates `results/paired_tests.csv` (one row per pair and variant: models, macro-F1 A, macro-F1 B,
     delta, CI low, CI high, significant, accuracy A, accuracy B, b, c, McNemar p-value, method) and prints a
     markdown table.
2. Unit test with synthetic predictions: identical models → delta 0 and p-value 1; misaligned idx → error.
3. Run it now on the pairs available (full FT vs each LoRA run). The pairs BERT vs TF-IDF (main) and BERT vs LSTM
   need stream B's prediction files: if they are missing in results/predictions/, say so and stop for those pairs.
4. Write the result of the main comparison in PROGRESS.md as a plain sentence ("H1 supported / not supported:
   ΔF1 = x [CI a; b]"), with no reinterpretation of the pre-registered protocol. Tick A8 only when the main pair is
   done. Commit on the branch, do not push.
```

---

## P4 — Task A9: calibration and human-review threshold

```text
Context: stream A, task A9. Create a branch `feat/a-calibration`.
Goal: show whether the final model's confidence can be trusted, and pick a threshold below which a clause is sent
to a lawyer.

1. `src/calibration.py` with:
   - `expected_calibration_error(confidences, correct, n_bins=15)`;
   - `reliability_diagram(confidences, correct, path)` (matplotlib, saved as PNG);
   - `coverage_precision_curve(confidences, correct, thresholds)`: for each threshold, share of clauses handled
     automatically (confidence ≥ threshold) and accuracy on those clauses.
   - optional `fit_temperature(val_logits_or_probas, val_labels)` (temperature scaling fitted on validation only).
2. `scripts/calibration_report.py --run-id <final_run_id> --target-precision 0.95`:
   - reads the validation and test prediction files (`proba_pred`, `pred`, `label`);
   - CHOOSES the threshold on VALIDATION (smallest threshold reaching the target precision), then REPORTS coverage
     and precision on TEST with that fixed threshold;
   - writes `results/figures/reliability_<run_id>.png`, `results/figures/coverage_<run_id>.png` and
     `results/calibration_<run_id>.json` (ECE val/test, threshold, coverage and precision on test).
3. Unit tests: perfectly calibrated synthetic data → ECE ≈ 0; threshold 0 → coverage 1.
4. Run it on the final model, add the numbers to PROGRESS.md and a short "Calibration and human review threshold"
   section to docs/model_card.md (numbers only from the JSON). Tick A9. Commit on the branch, do not push.
```

---

## P5 — Task A10: results in the README and delivery check

```text
Context: stream A, task A10 (delivery). Create a branch `feat/a-release`.
1. Read every results/runs/*/metrics.json, results/lora_tradeoff.csv, results/paired_tests.csv,
   results/calibration_*.json and the files of the other streams (results/summary_classification.csv,
   results/retrieval_summary.csv) if present.
2. Fill the "Results" section of README.md with these numbers ONLY. Leave a cell empty with "not run" when a number
   does not exist. Never round in a way that changes a conclusion; keep 3 decimals for F1.
3. Add under the pre-registered comparison a "Result of the main comparison" paragraph copied from PROGRESS.md.
   Do not edit the pre-registered block itself.
4. Delivery check in a fresh clone in a temporary folder: `pip install -r requirements-dev.txt`,
   `python -m pytest -q`, `python -m src.data`, and check that every notebook referenced in README exists.
   Report anything broken instead of hiding it.
5. Update the milestone statuses in README (⏳ → ✅ only if done), PROGRESS.md and tasks/todo.md.
   Commit on the branch, do not push. Remind me to create the `v1.0` tag after the merge.
```
