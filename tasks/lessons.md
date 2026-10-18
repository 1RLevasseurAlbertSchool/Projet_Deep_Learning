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
