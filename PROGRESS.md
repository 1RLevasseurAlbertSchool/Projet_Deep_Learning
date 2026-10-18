# Progress — Clause Review Assistant (LEDGAR)

> This file is the **operational log** of the project. It is updated after every significant task.
> `CLAUDE.md` holds the durable rules and decisions; `STRATEGY.md` the plan; `PROGRESS.md` the actual state,
> the results, the problems met and the next actions.

## 1. Overall status

**Current status:** 🟡 Repository initialised and pushed to GitHub (structure, documentation, pre-registered main
comparison, shared modules `src/config.py`, `src/data.py`, `src/metrics.py`, `src/utils.py`), documentation
translated into English. No experiment logged in this file yet.
**Next main step: validate `load_ledgar()` on the real data (B1), then run in parallel EDA + TF-IDF (B),
HF script (A) and MiniLM index (C).**

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
- [ ] `all-MiniLM-L6-v2` index over the clause bank.
- [ ] 10 queries checked by hand.
- [ ] Tag `m2`.

### m3 — LSTM and error analysis (Amandine)
- [ ] LSTM / BiLSTM with packing and GloVe.
- [ ] Per-class F1, most confused classes.
- [ ] Tag `m3`.

### m4 — Transformers (Robin)
- [ ] DistilBERT smoke test.
- [ ] Encoder screening on the validation set.
- [ ] Final full fine-tuning run + single test evaluation.
- [ ] LoRA runs + trade-off table.
- [ ] Paired tests (main comparison + secondary ones).
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

*No experimental result logged yet.*

Published reference points (LexGLUE, LEDGAR test set — not to be confused with our results):

```text
Method          µ-F1   m-F1
TF-IDF SVM      87.0   81.4
BERT            87.6   81.8
RoBERTa         87.9   82.3
DeBERTa         88.2   83.1
Legal-BERT      88.2   83.0
```

## 4. Problems met and solutions

### `src/models/` excluded by `.gitignore` (14/10)

**Symptom:** `src/models/__init__.py` was missing from the first commit.
**Cause:** the unanchored pattern `models/` matched every folder named `models`.
**Solution:** root-level patterns anchored (`/models/`, `/checkpoints/`, `/outputs/`, `/out/`), commit amended
before the first push.
**Status:** ✅ solved.

## 5. Next immediate action

1. Robin: add Amandine and Brian as collaborators on GitHub (if not done yet).
2. Amandine (B1): run `python -m src.data` and confirm 60,000 / 10,000 / 10,000 and 100 labels.
3. Then parallel start: B2-B5 (EDA, TF-IDF), A1-A2 (HF script), C1-C3 (index, queries).

## 6. Rule for updating this file

After every significant step, state: what was done; the result obtained; the numbers or metrics; the problems met;
the solution adopted; the next action.

Never replace a real result with an assumption. A task stays `[ ]` until it has actually been run or confirmed.
Any deviation from the pre-registered comparison is logged here.

## 7. Chronological log

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
