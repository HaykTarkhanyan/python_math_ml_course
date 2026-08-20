# `git grep` cannot see untracked work, and two whole chapters were untracked

**Symptom.** Renumbering `ml/09_clustering/` and `ml/10_dimensionality_reduction/` to make room for
the new colour-spaces deck, I ran the reference sweep the way
`_learnings/2026-08-12-1615_use-git-grep-for-rename-sweeps.md` prescribes - `git grep` as the
authoritative check. It returned **4 files**. Ripgrep over `ml/` on the same patterns returned
**13**.

```
$ git grep -l "33_image_compression\|34_dimensionality_reduction\|35_umap\|36_eigenfaces\|32_clustering"
DECISIONS.md
ml/00_plan.md
ml/MISSING_TOPICS.md
ml/ch10_diffusion/DIFFUSION_CHAPTER_PLAN.md
```

Nine live references missing, including `09_clustering.qmd`, `10_dimensionality_reduction.qmd`, both
`REVIEW.md` files, `32_clustering.tex`, `35_dimensionality_reduction.tex`, `36_umap.tex` and
`py_src/umap_demos.py`.

**Cause - established, not guessed.**

```
$ git status --porcelain ml/09_clustering ml/10_dimensionality_reduction
?? ml/09_clustering/
?? ml/10_dimensionality_reduction/

$ git ls-files ml/09_clustering/ ml/10_dimensionality_reduction/
(empty)
```

Both folders are **entirely untracked**. `git grep` searches the index and tracked working-tree
files by design, so every file in those two directories is invisible to it. This is not the
mysterious flakiness the 2026-08-12 learning documented - it is `git grep` behaving exactly as
specified, against a repo state that violated the earlier learning's unstated assumption.

**The unstated assumption is the real lesson.** The 2026-08-12 learning says:

> verify with `git grep`, which searches tracked files, is fast, and is authoritative for anything
> that will be committed

It even names the limitation ("searches tracked files") and then draws the opposite conclusion, on
the strength of a repo where the relevant work happened to be committed. In this repo a whole
chapter can sit untracked for days - the Aug 16 clustering/dimred renumber was never committed, and
neither was the Aug 19-20 work on top of it - so "anything that will be committed" is precisely the
set `git grep` cannot see.

**Consequences.**

- Before trusting `git grep` for a sweep, check that the target files are tracked:
  `git status --porcelain <dirs>`. A `??` on a directory means `git grep` is blind to all of it.
- The safe sweep is **both**: `git grep` for tracked files, plus a Grep-tool / ripgrep pass **scoped
  to the affected directories** (not the repo root - that walks `docs/` and the OneDrive tree, which
  is the disk-thrash pattern `CLAUDE.md` warns about). Reconcile the two lists.
- Do not silently "fix" the earlier learning by editing it. Both are true under their own
  conditions, and the interesting content is the condition, not the verdict.
- Corollary worth remembering: an untracked rename is *cheap* - no history to rewrite, `git mv`
  does not apply, plain `mv` is correct. The renumber itself was the easy part; finding the
  references was the risk.

**Sweep result, for the record.** 13 files updated with a single `sed` pass carrying all six rules
at once. Ordering was safe because the patterns are full distinct names
(`33_image_compression` → `34_image_compression`, `34_dimensionality_reduction` →
`35_dimensionality_reduction`, ...) so no output of one rule is the input of another - the naive
ascending or descending pass would have double-shifted. Verified clean afterwards with a repeat
ripgrep that returned nothing outside the outline file documenting the mapping.
