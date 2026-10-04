# Two sessions working at once both took DECISIONS #60 and #61

**Symptom.** `DECISIONS.md` ended up with two `## #60` and two `## #61` headers:

```
12:## #61 - The language-model block comes before the autoencoders
48:## #60 - After CNN: autoencoders, then RNN, then attention, then a language-model chapter
77:## #61 - The practical after L16 is the liver biopsy, rebuilt for class; HW1c homework retired
105:## #60 - The practical after L16 reads traffic lights with classical vision only; CNNs wait
```

**Cause.** A course-planning session and a CNN-chapter session ran on the same repo on
2026-10-02/03. Each read the newest number (#59), took the next ones, and inserted above #59. The
Edit tool applies cleanly to the current file, so nothing warned that the numbers were already
taken: the second writer's entries simply landed below the first's.

**Fix applied.** The planning session's two entries were renumbered #62 and #63, with a line in each
saying so, because they were cited in only one file (`ml/LLM_CHAPTER_PLAN.md`), while the CNN
session's #61 was already cited by `ml/12_cnn/traffic_light/README.md`. Renumber whichever side has
fewer citations, and grep for `#NN\b` across `*.md, *.tex, *.qmd` before deciding.

**Consequences.**

- Re-read the top of `DECISIONS.md` (`grep -n "^## #" DECISIONS.md | head -3`) **immediately**
  before writing an entry, not at the start of the task.
- After writing, check for duplicates: `grep -oE "^## #[0-9]+" DECISIONS.md | sort | uniq -d`
  must print nothing.
- When another session is known to be active (uncommitted changes in files you did not touch,
  see `git status`), expect this.
