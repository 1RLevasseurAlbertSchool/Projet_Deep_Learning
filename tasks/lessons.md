# Lessons learned

> Improvement loop: after any correction or mistake, add a line.
> Format: `[date] | what went wrong | rule to avoid it`.
> Read at the start of every working session.

| Date | What went wrong | Rule to avoid it |
|---|---|---|
| 2026-10-14 | `.gitignore` pattern `models/` also excluded `src/models/` | Anchor root-level ignore patterns with a leading `/` (`/models/`) and check `git ls-files` after changing `.gitignore` |
| 2026-10-18 | Documentation first written in French although one team member does not speak French | Write everything in English (code, docs, commits) from the start |
| 2026-10-18 | `python -m src.data` run in the anaconda base environment (`datasets` 2.12) ignored the `ledgar` config of the parquet-only Hub repo and downloaded `case_hold` | Always run project commands in the project `.venv` built from `requirements.txt`; check library versions before debugging the code |
| 2026-10-18 | Assumed `model.config._commit_hash` still existed; `transformers` 5.20 no longer sets it | Check attributes in the installed library source before relying on them (major versions remove internals) |
| 2026-10-19 | On Colab, `pip install -r requirements.txt` upgraded `ipykernel` / `jupyter-client` (listed for local notebooks) and broke the versions pinned by `google-colab` | Keep `requirements.txt` to runtime dependencies only; dev tools (jupyterlab, ipykernel, pytest) go in `requirements-dev.txt` (`-r requirements.txt`), installed locally, never on Colab |
| 2026-10-19 | A6 was planned as a retrain of the chosen encoder with `--eval-test`: one more GPU run, and the retrained weights (GPU non-determinism, fp16) are not the ones selected on validation | Evaluate the test set from the saved best checkpoint of the run chosen on validation (`scripts/evaluate_checkpoint.py`, single-use guard); never retrain a model just to evaluate it |
| 2026-10-19 | `CLAUDE.md` prescribed one feature branch + reviewed PR per task, but that is not what we did (`feat/a-eval-checkpoint` was merged into `main` locally) | Work directly on `main`: `git pull` before starting, commit small, push after review (`CLAUDE.md`, Git section); keep the documented workflow equal to the one actually followed |
| 2026-10-19 | The Colab Drive clone was updated with `checkout` + `pull --ff-only`, which aborts once run JSONs written there by Colab (untracked) are committed on `main` from a laptop | Treat the Drive clone as a read-only mirror: `checkout -f <BRANCH>` + `reset --hard origin/<BRANCH>` (notebook 04, section 2); never edit or commit in it |
