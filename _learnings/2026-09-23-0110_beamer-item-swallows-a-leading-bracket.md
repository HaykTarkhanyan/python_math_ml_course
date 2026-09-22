# `\item [42]'s ...` turns the lecture number into the bullet

**Symptom.** A new L21 frame had the bullet

```latex
\item [42]'s GPUs want thousands of \emph{independent} multiply-adds at once.
```

It compiled with 0 errors and both overflow detectors passed, but the rendered slide showed no
bullet and the text started "42 's GPUs want ..." - the bracket group was eaten.

**Cause.** `\item` takes an optional argument in square brackets: `\item[label]` replaces the
bullet with `label`. A space between `\item` and `[` does not stop LaTeX from reading it as that
argument. The course's lecture-number convention (`[42]`, `[41\_1]`, `[06]`) puts brackets at the
start of a sentence often enough for this to recur.

**Rule.** Never start an item's text with a bracketed lecture number. Rephrase ("The GPUs from
[42] ...") or protect it (`\item {[42]}'s ...`). Check after adding bracket callbacks:

```bash
grep -nE '\\item\s*\[' DECK.tex      # every hit must be an intentional custom label
```

Only a rendered page shows this; `pdflatex`, `detect_clipped_slides.py` and
`detect_footer_collisions.py` all pass it (same family as the frame-subtitle trap in
`_work_sessions/2026-09-14-1250_*.toml`: a brace group right after `\begin{frame}{Title}` becomes
the subtitle).
