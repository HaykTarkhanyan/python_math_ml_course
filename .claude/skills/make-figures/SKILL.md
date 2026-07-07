---
name: make-figures
description: >
  Use when a slide deck needs generated figures (matplotlib PDFs), when
  creating or editing a figure script under ml/**/py_src/, or when replacing
  a hand-drawn TikZ sketch or third-party image with a real generated figure.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
user-invocable: true
---

# Deck figure generation (py_src -> fig)

Every essential figure is Python-generated (matplotlib), not TikZ - that covers
anything data-driven and any diagram carrying a frame's core message. TikZ is
only for small throwaway visuals (quick boxes-and-arrows the frame could live
without). Scripts live in a sibling `py_src/`, outputs go to a sibling `fig/`
as PDF (vector), and they run with the project venv - never a fresh environment:

```bash
./ma/Scripts/python.exe ml/CHAPTER/py_src/SCRIPT.py
```

Missing package? `uv pip install --python ./ma/Scripts/python.exe PKG`.

## Required script skeleton

Copy the pattern from an existing script in the same chapter (canonical example:
`ml/03_classification/py_src/imbalanced_figs.py`):

- Module docstring: which deck it serves, every output PDF with a one-line
  purpose, and the run command.
- `matplotlib.use("Agg")` before importing pyplot.
- `SEED = 509`; seed every RNG from it.
- `ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"` - no
  enforced figure palette; these Armenian-flag colors are the global fallback
  for charts with 3+ colors (see `ml/SLIDE_STYLE.md` Figures section).
- Paths from `Path(__file__)`: figures to `CH_DIR / "fig"`, logs to
  `REPO_ROOT / "logs"` (mkdir before the FileHandler).
- `logging` to console + `logs/<script>.log`; no `print`. Log every saved path.
- One function per figure.

## Chart rules (global, non-negotiable)

- No pie or doughnut charts, ever. Bar charts instead.
- Value labels on bars: `ax.bar_label(...)`.
- Fonts must stay readable at slide scale - copy figsize/fontsize from an
  existing script in the chapter rather than inventing new values.
- Keep compute light: small sample counts, no big sweeps or long simulations
  without asking first (this laptop freezes under sustained load).

## Verify - all four, in order

1. Script runs end-to-end under the `ma` venv, exit code 0, no ERROR in the log.
2. Expected PDFs exist in `fig/` with non-trivial file sizes.
3. Open or render each new figure and look: labels not overlapping, nothing
   clipped, text legible.
4. Add the `\includegraphics` lines and recompile the deck (`compile-deck`
   skill) - a figure that is generated but never embedded is not done.
