# A matplotlib redraw dirties every PDF it writes, even when nothing changed

**Symptom.** After adding one figure to `ml/14_llms/py_src/decoding_figs.py` and redrawing, and
after changing one icon in `llm_roadmap.py`, `git status` listed 42 figure PDFs as modified.

**Cause.** matplotlib writes the save time into each PDF's `/CreationDate`. A byte comparison
with that field stripped (throwaway script, 2026-10-06) split the 42 files:

```
DATE ONLY: 27      dec_*.pdf (12), roadmap_*_pass.pdf (15)
REAL CHANGE: 15    roadmap_*_life.pdf (the icon that actually changed)
```

**Consequences.**
- Do not stage figures just because `git status` shows them. Check which ones really changed
  (strip `/CreationDate (D:...)` and compare with `git show HEAD:<path>`), stage those, and
  `git checkout --` the date-only ones.
- New figure scripts save with `fig.savefig(out, metadata={"CreationDate": None})`, so an
  unchanged figure redraws to the same bytes (`head_figs.py` and `pretraining_figs.py`, from
  2026-10-06). Older scripts still stamp the date. Switching one over dirties its figures once,
  so do it when its figures change anyway.
