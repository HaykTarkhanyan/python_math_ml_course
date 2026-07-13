# LEARNINGS.md

Non-obvious lessons, gotchas, and incidents from working in this repo.
Append new entries with a date. Strike through obsolete ones, do not delete.

## LaTeX / Beamer

- **2026-04:** Recurring Write-tool typo: `\end{center>` / `\end{frame>` (`>` instead of `}`). After writing any `.tex`, run `grep -nE "end\{(center|frame)>" FILE.tex` - it must return nothing - before compiling.
- **2026-04:** `step` is a reserved TikZ key (grid spacing). Defining a style named `step/.style` produces `Package pgfkeys Error: The key '/tikz/step' requires a value`. Other risky generic names: `arrow`, `line`, `grid`, `node`. Prefix styles by purpose instead: `lrtstep`, `cibox`.
- Beamer **silently clips** overflowing content - no overfull-vbox warnings. A clean log and correct page count prove nothing about layout; only visual inspection of rendered pages catches clipping (`beamer-overflow-check` skill).
- pdflatex must run **twice**: after deleting the `.toc`, a single pass leaves the Outline frame empty and page counters stale (footer shows things like `8/7`).
- "Compiled clean" means `grep -cE "^!" FILE.log` prints `0` (the count - grep's exit code will be nonzero in that success case), not merely that pdflatex exited.
- **2026-07-10:** A `pdflatex` run killed by a **timeout mid-compile** leaves a truncated `.aux`/`.toc`; the *next* run then dies with `! File ended while scanning use of \@writefile.` pointing (misleadingly) at `\begin{document}`. It is not a source error - delete the deck's aux files (`.aux .toc .nav .snm .out .vrb`) and recompile. Heavy decks (e.g. `20_advanced_boosting`, ~2.7 MB / 40+ pp) can exceed a 2-min tool timeout for two passes - compile the slow deck alone with a longer timeout, and never run a multi-deck loop that might get cut off mid-pass.

## Quarto

- A blank line is required before lists, blockquotes, code fences, and tables. Without it the content renders as plain text, silently.
- `_quarto.yml` paths must match the on-disk case exactly. GitHub Actions CI runs Linux (case-sensitive); Windows hides the mismatch locally, so it only breaks on push.
- Solutions render: `quarto render FILE.qmd --profile=solution --output FILE_sol.html`.

## Windows / tooling

- Opening several PDFs for review: a rapid `cmd /c start msedge` loop drops windows. Use PowerShell `Start-Process "msedge.exe" -ArgumentList '"C:\full\path.pdf"'`, one file at a time, ~0.5 s apart.
- Paths produced inside Git Bash (e.g. `pdftoppm` output under `/tmp`) may need `cygpath -w` before the Read tool can open them.
- Glob/ripgrep over the whole repo can time out (OneDrive + `ma/` venv with thousands of files). Search specific subfolders, or use `ls`/`git ls-files` for existence checks.
- **2026-07-13:** Armenian text in Python logging **vanishes silently** on Windows: the cp1252 console cannot encode it and the default-encoding `FileHandler` drops it too - no exception, the lines just never appear. Pass `encoding="utf-8"` to `logging.FileHandler` and keep console output ASCII (or guard it) when logging Armenian.

## Python / notebooks

- **2026-07-08:** `ax.bar_label(ax.containers[0], ...)` mislabels when the bars were drawn with `yerr=`. `ax.bar(..., yerr=...)` also appends an `ErrorbarContainer` to `ax.containers`, so `containers[0]` can be the errorbar, raising `AttributeError: 'ErrorbarContainer' object has no attribute 'patches'`. Capture the bars: `bars = ax.bar(...)` then `ax.bar_label(bars, ...)`.
- **2026-07-13:** Fused `nn.RNN`/`nn.LSTM` hide the per-step recurrence from autograd's exposed tensors: `retain_grad()` on the stacked output yields all-zero gradients except at the loss's direct touch point (the last step). To measure per-timestep gradient flow, manually unroll with `nn.RNNCell`/`nn.LSTMCell` and call `retain_grad()` on each individual `h_t`. (Found building L20's gradient_flow.py.)
- **2026-07-13:** matplotlib **mathtext silently drops** non-mathtext text (e.g. Armenian) appended in the same `ax.text()` string as `$...$` - no warning, just missing glyphs in the output. Keep Armenian/unicode in plain-text annotations, never mixed into mathtext. Also promote font fallback to a hard error when rendering Armenian: `warnings.filterwarnings("error", message=".*Glyph.*missing.*")` - otherwise tofu reaches the slide silently. Verify by rendering pages to PNG and looking; the PDF text-extraction layer is separately unreliable for Armenian and proves nothing.
- **2026-07-08:** The global `jupyter` / `nbconvert` launcher is broken on this machine - the system `jupyter_contrib_nbextensions` raises `ModuleNotFoundError: No module named 'notebook.services'` when nbconvert enumerates exporters, so `python -m jupyter nbconvert --execute` fails. To execute a notebook with the `ma` stack (imblearn etc.), bypass the launcher and drive `nbclient` directly with the already-registered `ma` kernel: `NotebookClient(nb, timeout=600, kernel_name="ma", resources={"metadata": {"path": "ml/03_classification"}}).execute()`. Set the `path` resource so relative loads (`data/...`) resolve. Bonus: running the real notebook this way caught a `bar_label` bug the standalone verification script missed - execute the notebook, not just the code.

## Git

- **2026-07-10:** `git mv` **stages the rename immediately**. A later bare `git commit` (no pathspec) then sweeps those already-staged renames into whatever commit you are making - e.g. ch4 deck renames landed inside a commit labelled "ch3". Stage explicitly per commit (`git add <paths>`) and check `git diff --cached --name-status` before committing. Recovery (nothing pushed): `git reset --soft HEAD~1` then `git restore --staged <dir>` to unstage the wrong group, then re-commit each group cleanly.
- **2026-07-10:** Re-running a figure script regenerates *all* its PDFs; matplotlib output is not byte-deterministic, so untouched figures show as modified (metadata only). Commit only the figures whose generating code actually changed; `git checkout -- <unchanged-figs>` to drop the spurious diffs.

## Machine / usage limits (incidents)

- **2026-05-21:** Two parallel review agents each running large Monte Carlo simulations froze the 16 GB laptop. Heavy compute: ask first, run sequentially, cap sample counts, prefer deterministic checks.
- **2026-07-06:** A medium-effort code review fanned out 11 subagents and drained the session usage limit. Default to inline single-agent work; any multi-agent plan needs explicit user approval with an agent-count estimate up front.

## Course logistics

- YouTube lecture numbering does NOT match slide deck numbering (e.g. "Lecture 46" is deck `08_stat`). The deck -> homework-qmd mapping table lives in the `update-youtube` skill; the lecture-number offset is not tabulated anywhere - verify per video, don't guess.
- Course events go on the "Metric" Google Calendar, 17:30-19:00 Europe/Berlin.
- Delta method: slides cover it but it was deferred in class (to causal inference); do not assume students have seen it when writing homework.
- Match solution depth to homework position in the course - never reference concepts (eigenvalues, Jordan form, ...) that later modules introduce.
