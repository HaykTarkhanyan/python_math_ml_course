---
name: slide-style
description: >
  Use when creating, building, drafting, editing, or reviewing a Beamer / LaTeX
  slide deck (.tex) in this repo, especially under ml_new/. Loads the instructor's
  slide-design conventions - deck structure, color palette, figures, tone,
  math/code depth, and the new-deck workflow - so every deck comes out consistent.
  The full guide is ml_new/SLIDE_STYLE.md (single source of truth).
allowed-tools:
  - Read
  - Bash
  - Edit
  - Write
  - Glob
  - Grep
user-invocable: true
---

# Slide style (ml_new Beamer decks)

**First action: read `ml_new/SLIDE_STYLE.md` and follow it.** That file is the single
source of truth; this skill is just the trigger. The essentials:

## New-deck workflow (do in this order)

1. **Interview the instructor on content** for this deck — scope, include vs cut,
   examples, depth. Don't assume.
2. **Draft an outline** (sections + frame list, an outline `.md`) and get approval.
3. **Then build** the `.tex`, compile, clean, open the PDF for review.

For edits to an existing deck: just make the change, recompile, verify.

## Must-knows

- `\documentclass[aspectratio=169]{beamer}` + `\input{../preamble}`. End the `.tex` with a `% Provenance:` block.
- Compile `pdflatex -interaction=nonstopmode -halt-on-error FILE.tex` **twice**, then `./ma/Scripts/python.exe clean_latex.py`. Beamer clips overflow silently - eyeball rendered pages or run `/beamer-overflow-check`.
- **Skeleton:** cold-open hook -> Outline -> `\section`s each with a `[plain]` transition slide (popblue title + one motivation line) -> recap + paramgreen "Next:" box. No fixed length; one idea per frame.
- **Palette / callouts:** popblue=theory, armred=data/warnings, paramgreen=params/takeaways, armorange=watch-outs. Boxes: armblue!8=key, armred!8=trap, paramgreen!8=takeaway/Next, armorange!12=watch-out.
- **Content:** English body; mostly straight tone with occasional light touch; predict-first frames where a result is counter-intuitive; a by-hand worked-numbers frame where mechanics are computable; **full step-by-step** derivations; **minimal code** (one canonical sklearn snippet max).
- **Figures:** real matplotlib for data, TikZ for schematics; scripts in sibling `py_src/`, output to sibling `fig/`, run with the `ma` venv; no enforced figure palette; attribution line on embedded third-party figures.
- **Prose:** no em-dashes, no curly quotes.

If anything here is ambiguous for the deck at hand, ask before building.
