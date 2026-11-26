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

1b. **Deck not taught yet? Check its `_notes` copy BEFORE compiling.** The
   instructor annotates `DECK_notes.pdf` live in class, so every compile of a
   deck that has not been delivered also refreshes that copy (instructor,
   2026-10-23). Check now, while the old clean PDF is still on disk:

   ```bash
   { [ ! -e DECK_notes.pdf ] || cmp -s DECK.pdf DECK_notes.pdf; } && echo "notes copy: safe to refresh"
   ```

   After step 3 passes, and only if that line printed, run
   `cp DECK.pdf DECK_notes.pdf`. Nothing printed means the copy differs from the
   clean PDF: it may hold annotations from class, which cannot be regenerated, so
   leave it alone and tell the instructor. A delivered deck without a
   `_notes.pdf` gets none. Commit the `_notes.pdf` only after the lecture.

   **Also check the CLEAN PDF for ink before compiling** - if the instructor
   annotated the clean file by mistake, the compile would erase it:

   ```bash
   ./ma/.venv/Scripts/python.exe -c "from pypdf import PdfReader; import sys; r=PdfReader(sys.argv[1]); n=sum(1 for p in r.pages for a in (p.get('/Annots') or []) if str(a.get_object().get('/Subtype'))=='/Ink'); print('ink strokes in clean PDF:', n)" DECK.pdf
   ```

   Anything above 0: stop, back the file up, move it to `DECK_notes.pdf` (if
   that one is still a plain copy), then compile. (2026-10-28: 653 strokes found
   in the clean LLM-1 PDF.)

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
   ./ma/.venv/Scripts/python.exe clean_latex.py PATH_TO_DIR_OR_TEX
   ```

6. **Open for instructor review** - PowerShell, one file at a time, ~0.5 s apart
   (rapid `cmd /c start` loops drop windows). **Open `DECK_notes.pdf` when it
   exists**, not the clean PDF: the instructor annotates whatever is open, and
   the clean PDF is overwritten by the next compile.

   ```powershell
   Start-Process "msedge.exe" -ArgumentList '"C:\full\path\deck_notes.pdf"'
   ```

## Red flags - the deck is NOT done if you are thinking:

| Thought | Reality |
|---|---|
| "pdflatex exited 0, so the deck is done" | With `-halt-on-error` that does mean no `!` errors, but the log grep is cheap insurance (and catches runs where the flag was dropped). Compiled is still not the same as laid out correctly. |
| "Page count looks right" | Beamer clips silently. Render pages and look. |
| "Nothing structural changed, one pass is fine" | Counters and TOC go stale. Always two passes. |
| "I'll clean aux files later" | Later means they get committed. Clean now. |
| "The `_notes.pdf` is just out of date, overwrite it" | If it differs from the clean PDF it may hold the instructor's annotations, which exist nowhere else. Ask. |
