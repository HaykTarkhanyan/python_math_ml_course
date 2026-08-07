# non_essential/

Maintained, reusable tooling that is **not** needed to build or run the course. Nothing here is
imported by live code. Everything here is expected to still work.

(Load-bearing scripts stay at the repo root: `clean_latex.py`, `render_only_changed.py`.)

| Script | What it does | When you would run it |
|---|---|---|
| `detect_clipped_slides.py` | Compares the prose in each Beamer frame against the prose actually present in the rendered PDF, and reports what did not make it onto the slide. | On any deck before committing it, as part of the deck polish loop. Added 2026-08-08. |

## Why `detect_clipped_slides.py` exists

Beamer discards overflowing content **silently** - no Overfull vbox warning, correct page count,
clean log. `LEARNINGS.md` records this and concludes that only visual inspection catches it,
which is true but means eyeballing every page of every deck; a reviewer who spot-checks will
miss things.

It earned its place on 2026-08-07: `ml/ch12_vlm/L34_vlm_drawing.tex` had silently lost the tail
of the final list item on its **recap frame** - the last slide of the chapter - and the deck had
already passed a self-review that happened to check other pages. `pdflatex` reported zero
problems.

Usage:

```bash
./ma/Scripts/python.exe non_essential/detect_clipped_slides.py ml/ch12_vlm
```

Exit code 1 if anything is flagged, 0 if clean.

**It produces false positives by design.** N-grams that span a figure, a table or a column
boundary get flagged, as does LaTeX residue and any accented character (the tokenizer is
ASCII-only, so "Muller" becomes "ller"). Treat a flag as a place to look, not a verdict - open
the rendered page and decide.

It also only knows about **text**. A clipped figure, a box drawn past the slide edge, or an
overlapping label are invisible to it, so visual checking is still required. What this changes is
that the checking becomes targeted rather than exhaustive.
