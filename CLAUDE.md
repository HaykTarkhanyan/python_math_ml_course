# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Deployment

- **GitHub Actions** auto-renders Quarto site on every push to `main` (~5 min). No local rendering needed.
- Workflow: `.github/workflows/publish.yml` — renders `.qmd`/`.ipynb` files, deploys `docs/` to GitHub Pages
- Legacy `Makefile` and `render_only_changed.py` still exist but are superseded by Actions
- `make push render=false msg="message"` — quick commit+push without rendering
- Site: https://hayktarkhanyan.github.io/python_math_ml_course/

## LaTeX / Beamer Slides

- **pdflatex** via TeX Live 2025; compile with `pdflatex -interaction=nonstopmode FILE.tex`
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

## Homework .qmd Files

- Located in `math/00_*.qmd` through `math/25_*.qmd`
- Difficulty: `{data-difficulty="1"}` → 🧀, `"2"` → 🧀🧀, `"3"` → 🧀🧀🧀
- Bonus: `.bonus-problem` class → 🎁
- Solutions use Quarto profiles: `{.content-visible when-profile="solution"}`

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
