# LEARNINGS.md

Non-obvious lessons, gotchas, and incidents from working in this repo.
Append new entries with a date. Strike through obsolete ones, do not delete.

## LaTeX / Beamer

- **2026-04:** Recurring Write-tool typo: `\end{center>` / `\end{frame>` (`>` instead of `}`). After writing any `.tex`, run `grep -nE "end\{(center|frame)>" FILE.tex` - it must return nothing - before compiling.
- **2026-04:** `step` is a reserved TikZ key (grid spacing). Defining a style named `step/.style` produces `Package pgfkeys Error: The key '/tikz/step' requires a value`. Other risky generic names: `arrow`, `line`, `grid`, `node`. Prefix styles by purpose instead: `lrtstep`, `cibox`.
- Beamer **silently clips** overflowing content - no overfull-vbox warnings. A clean log and correct page count prove nothing about layout; only visual inspection of rendered pages catches clipping (`beamer-overflow-check` skill).
- pdflatex must run **twice**: after deleting the `.toc`, a single pass leaves the Outline frame empty and page counters stale (footer shows things like `8/7`).
- "Compiled clean" means `grep -cE "^!" FILE.log` prints `0` (the count - grep's exit code will be nonzero in that success case), not merely that pdflatex exited.

## Quarto

- A blank line is required before lists, blockquotes, code fences, and tables. Without it the content renders as plain text, silently.
- `_quarto.yml` paths must match the on-disk case exactly. GitHub Actions CI runs Linux (case-sensitive); Windows hides the mismatch locally, so it only breaks on push.
- Solutions render: `quarto render FILE.qmd --profile=solution --output FILE_sol.html`.

## Windows / tooling

- **Armenian (any non-ASCII) text gets corrupted by the Edit tool.** Make such edits with a small Python script instead: `pathlib.Path(...).read_text(encoding="utf-8")` -> replace -> `write_text(encoding="utf-8")`.
- Opening several PDFs for review: a rapid `cmd /c start msedge` loop drops windows. Use PowerShell `Start-Process "msedge.exe" -ArgumentList '"C:\full\path.pdf"'`, one file at a time, ~0.5 s apart.
- Paths produced inside Git Bash (e.g. `pdftoppm` output under `/tmp`) may need `cygpath -w` before the Read tool can open them.
- Glob/ripgrep over the whole repo can time out (OneDrive + `ma/` venv with thousands of files). Search specific subfolders, or use `ls`/`git ls-files` for existence checks.

## Python / notebooks

- **2026-07-08:** `ax.bar_label(ax.containers[0], ...)` mislabels when the bars were drawn with `yerr=`. `ax.bar(..., yerr=...)` also appends an `ErrorbarContainer` to `ax.containers`, so `containers[0]` can be the errorbar, raising `AttributeError: 'ErrorbarContainer' object has no attribute 'patches'`. Capture the bars: `bars = ax.bar(...)` then `ax.bar_label(bars, ...)`.
- **2026-07-08:** The global `jupyter` / `nbconvert` launcher is broken on this machine - the system `jupyter_contrib_nbextensions` raises `ModuleNotFoundError: No module named 'notebook.services'` when nbconvert enumerates exporters, so `python -m jupyter nbconvert --execute` fails. To execute a notebook with the `ma` stack (imblearn etc.), bypass the launcher and drive `nbclient` directly with the already-registered `ma` kernel: `NotebookClient(nb, timeout=600, kernel_name="ma", resources={"metadata": {"path": "ml/03_classification"}}).execute()`. Set the `path` resource so relative loads (`data/...`) resolve. Bonus: running the real notebook this way caught a `bar_label` bug the standalone verification script missed - execute the notebook, not just the code.

## Machine / usage limits (incidents)

- **2026-05-21:** Two parallel review agents each running large Monte Carlo simulations froze the 16 GB laptop. Heavy compute: ask first, run sequentially, cap sample counts, prefer deterministic checks.
- **2026-07-06:** A medium-effort code review fanned out 11 subagents and drained the session usage limit. Default to inline single-agent work; any multi-agent plan needs explicit user approval with an agent-count estimate up front.

## Course logistics

- YouTube lecture numbering does NOT match slide deck numbering (e.g. "Lecture 46" is deck `08_stat`). The deck -> homework-qmd mapping table lives in the `update-youtube` skill; the lecture-number offset is not tabulated anywhere - verify per video, don't guess.
- Course events go on the "Metric" Google Calendar, 17:30-19:00 Europe/Berlin.
- Delta method: slides cover it but it was deferred in class (to causal inference); do not assume students have seen it when writing homework.
- Match solution depth to homework position in the course - never reference concepts (eigenvalues, Jordan form, ...) that later modules introduce.
