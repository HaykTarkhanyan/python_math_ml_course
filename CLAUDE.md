# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Deployment

- **GitHub Actions** auto-renders Quarto site on every push to `main` (~5 min). No local rendering needed.
- Workflow: `.github/workflows/publish.yml` — renders `.qmd`/`.ipynb` files, deploys `docs/` to GitHub Pages
- Legacy `Makefile` and `render_only_changed.py` still exist but are superseded by Actions
- `make push render=false msg="message"` — quick commit+push without rendering
- Site: https://hayktarkhanyan.github.io/python_math_ml_course/

## LaTeX / Beamer Slides

- **pdflatex** via TeX Live 2025; compile with `pdflatex -interaction=nonstopmode FILE.tex` — run it **twice** so `\tableofcontents` (the Outline frame) and cross-references resolve. A single pass after deleting the `.toc` leaves the Outline blank.
- Two slide collections: `math/Lectures/stat/` (16 decks, 541 frames) and `math/Lectures/optim/` (6 decks)
- Also: `misc/claude_code/slides/` (5 decks on Claude Code topics)
- Shared conventions: Beamer `default` theme, `dove` color scheme, 16:9 aspect ratio
- Color palette: `popblue` (theory), `sampred` (data), `paramgreen` (parameters), `warnred` (warnings), `orange1`, `violet1`
- All diagrams are TikZ/pgfplots code — no external images
- Use `/beamer-overflow-check <pdf_path>` skill after compiling to visually detect clipped content

## Quarto Configuration

- Project type: **book** (`_quarto.yml`)
- `execute: enabled: false` — notebooks are pre-rendered, not executed during build
- `_quarto.yml` paths must be **exact case** — Linux CI is case-sensitive
- Custom callouts defined: Links, Python, Libraries, Math, ML, Misc
- Themes: cosmo (light) / darkly (dark) with toggle
- **Markdown formatting gotcha:** Quarto requires a blank line before lists, blockquotes, and other block elements. Without it the content renders as plain text instead of formatted markup.

```markdown
<!-- BAD — Quarto won't render the list -->
Some text:
- item a
- item b

<!-- GOOD — blank line before the list -->
Some text:

- item a
- item b
```

This also applies to ordered lists, blockquotes (`>`), code fences, and tables. Always leave a blank line before any block-level element.

## Homework .qmd Files

- Located in `math/00_*.qmd` through `math/25_*.qmd`
- Difficulty: `{data-difficulty="1"}` → 🧀, `"2"` → 🧀🧀, `"3"` → 🧀🧀🧀
- Bonus: `.bonus-problem` class → 🎁
- Solutions use Quarto profiles: `{.content-visible when-profile="solution"}`

## Python Environment

- **Use the project-local venv `ma/`** (at the repo root) for ALL Python in this repo — it already has the data-science stack (numpy, scikit-learn, matplotlib, pandas). Do NOT spin up ephemeral `uv run --with ...` envs; they re-download packages on a cold cache and waste time.
- Run scripts by calling the venv interpreter directly (it's a plain venv, no activation needed):
  - Git Bash: `./ma/Scripts/python.exe path/to/script.py`
  - PowerShell: `.\ma\Scripts\python.exe path\to\script.py`
- conda is installed (`base` + several envs under `~/.conda/envs/`) but is NOT on the Git Bash/PowerShell PATH. `ma` is a venv, not a conda env.
- To add a package, install into `ma` with uv (per the global "use uv" rule): `uv pip install --python ./ma/Scripts/python.exe <pkg>`
- Figure-generation scripts (e.g. `ml_new/**/py_src/*.py`) write PDFs to a sibling `fig/` and logs to `logs/`; run them with the `ma` interpreter.

## Testing

- pytest configured for `python_libs/` directory
- Run: `pytest python_libs/` or individual test files
- Tests are teaching examples (calculator, elections, movies), not CI-enforced

## Architecture

- **python/** — 18 Jupyter notebooks: Python fundamentals → OOP → capstone
- **python_libs/** — 18 Jupyter notebooks: data science stack (NumPy, Pandas, etc.)
- **math/** — 26 `.qmd` homework modules + `Lectures/` (Beamer slide decks)
- **ml/** — skeleton only (6 empty chapter dirs), not started
- **misc/** — supplementary materials: Google Colab guide, dl4nlp lectures, Claude Code slides
- Course is bilingual: Armenian (Հայերեն) + English throughout all materials
