# Use `git grep` for rename sweeps — a single Grep-tool pass under-reported

**Symptom.** While renaming `L12b_svm_and_classic_methods` → `28_svm_and_classic_methods`, a
Grep-tool search for `classic_methods|svm_and_classic` returned 13 hits and I reported the
reference sweep as complete. A plain `grep -rn` running in the background then finished and
surfaced **three live references the first pass had missed**, all in `ml/04_trees/`:

```
ml/04_trees/WORK_PLAN.md:91   `ml/07_classic_methods/L12b_svm_and_classic_methods.tex`. Hoisting to...
ml/04_trees/WORK_PLAN.md:99   recompile [17]-[20] + `09_regression_metrics` + `L12b_svm_and_classic_methods`
ml/04_trees/L12b_classic_methods_survey_OUTLINE.md:40   `L12b_svm_and_classic_methods.tex`.
```

**Cause — not established.** The obvious explanations were checked and ruled out:

- `git check-ignore -v` on all three files exits 1 → **not gitignored**, so ripgrep's default
  ignore handling is not the reason.
- There is no `.ignore` or `.rgignore` anywhere in the repo.
- The first call used `head_limit: 60` and returned only 13 results, so it was **not truncated**.
- A second Grep-tool call, same pattern class, **did** find `WORK_PLAN.md:91`.

So the same tool, on the same unignored files, returned different result sets on two runs. The
mechanism is unknown; what matters is that a single pass is not trustworthy for this job.

**Consequences.** For a rename or any other "find every reference to X" sweep, verify with
`git grep`, which searches tracked files, is fast, and is authoritative for anything that will be
committed:

```bash
git grep -n "OLD_NAME" | grep -v "^docs/\|^PROGRESS.md\|^_work_sessions/"
```

Exclude the read-only archives (`PROGRESS.md`, `_work_sessions/`, `docs/`) — they are historical
records and *should* keep the old name. Everything else that still matches is a live reference and
has to be updated.

Do not run a repo-wide `grep -rn` from the repo root as the primary check: it took over two
minutes and had to be backgrounded, because it walks `docs/`, `.git/` and the OneDrive tree. That
is also the disk-thrash pattern `CLAUDE.md` warns about.

**Reporting rule that follows from this.** Do not say "every reference is updated" off one search.
Run the `git grep` confirmation first, then report — the failure mode here was not the missed hits
but claiming completeness before verifying it.
