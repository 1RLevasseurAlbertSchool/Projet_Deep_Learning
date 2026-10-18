# Lessons learned

> Improvement loop: after any correction or mistake, add a line.
> Format: `[date] | what went wrong | rule to avoid it`.
> Read at the start of every working session.

| Date | What went wrong | Rule to avoid it |
|---|---|---|
| 2026-10-14 | `.gitignore` pattern `models/` also excluded `src/models/` | Anchor root-level ignore patterns with a leading `/` (`/models/`) and check `git ls-files` after changing `.gitignore` |
| 2026-10-18 | Documentation first written in French although one team member does not speak French | Write everything in English (code, docs, commits) from the start |
