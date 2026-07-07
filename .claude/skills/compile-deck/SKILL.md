---
name: compile-deck
description: >
  Use when compiling or verifying any Beamer/LaTeX deck (.tex) in this repo -
  after editing a deck, before committing one, or when a PDF looks wrong
  (blank Outline frame, stale page counts like 8/7, clipped or missing
  content). Runs the full compile -> verify -> clean -> review loop.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Grep
  - Glob
user-invocable: true
---

# Compile and verify a Beamer deck

A deck is not done when pdflatex exits 0. Beamer clips overflow silently, and a
single pass leaves a stale TOC and page counters. Follow every step, in order.

## The loop

1. **Pre-compile grep** - catches a recurring Write-tool typo (`>` instead of `}`):

   ```bash
   grep -nE "end\{(center|frame)>" FILE.tex
   ```

   Success = no output (grep will exit 1 in that case - that is the GOOD
   outcome, not a failed command). Any printed line is a typo to fix.

2. **Compile twice, from inside the deck's own directory** - decks load the
   shared preamble with `\input{../preamble}`, which pdflatex resolves relative
   to the CWD, so compiling from the repo root fails with "file not found":

   ```bash
   cd path/to/deck_dir
   pdflatex -interaction=nonstopmode -halt-on-error FILE.tex
   pdflatex -interaction=nonstopmode -halt-on-error FILE.tex
   ```

3. **Check the log, not the exit code:**

   ```bash
   grep -cE "^!" FILE.log
   ```

   The printed count must be `0`. (grep itself exits nonzero when the count
   is 0 - that is the success case; judge by the printed number.)

   `Package pgfkeys Error: The key '/tikz/NAME' requires a value` means a TikZ
   style used a reserved name (`step`, `grid`, `line`, ...). Rename it with a
   purpose prefix (`lrtstep`, `cibox`).

4. **Visual overflow check** - required for new frames or edited TikZ: run the
   `beamer-overflow-check` skill. A clean log does NOT prove the layout fits.

5. **Clean build junk** (always before a commit):

   ```bash
   ./ma/Scripts/python.exe clean_latex.py PATH_TO_DIR_OR_TEX
   ```

6. **Open for instructor review** - PowerShell, one file at a time, ~0.5 s apart
   (rapid `cmd /c start` loops drop windows):

   ```powershell
   Start-Process "msedge.exe" -ArgumentList '"C:\full\path\deck.pdf"'
   ```

## Red flags - the deck is NOT done if you are thinking:

| Thought | Reality |
|---|---|
| "pdflatex exited 0, so the deck is done" | With `-halt-on-error` that does mean no `!` errors, but the log grep is cheap insurance (and catches runs where the flag was dropped). Compiled is still not the same as laid out correctly. |
| "Page count looks right" | Beamer clips silently. Render pages and look. |
| "Nothing structural changed, one pass is fine" | Counters and TOC go stale. Always two passes. |
| "I'll clean aux files later" | Later means they get committed. Clean now. |
