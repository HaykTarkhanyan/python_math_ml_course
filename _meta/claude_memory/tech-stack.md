---
name: Technology Stack & Build System
description: Tools, build process, deployment, and compilation details for the course repo
type: project
---

## Build & Deployment
- **Quarto** renders .qmd and .ipynb → HTML book (output in docs/)
- **GitHub Pages** serves from docs/ directory
- **No CI/CD** — all builds are local, manual deployment
- **Makefile** has `make push` target: renders changed files → git add → commit with timestamp → push
- **render_only_changed.py** — incremental rendering (skips unchanged files)

## LaTeX / Presentations
- **pdflatex** via TeX Live 2025
- Beamer with `default` theme, `dove` color scheme, 16:9 aspect ratio
- Heavy TikZ/pgfplots — all diagrams are code-generated (no external images)
- pgfplotsset{compat=1.18}
- Key packages: amsmath, amssymb, pgfplots, tikz, colortbl, fontenc

## Python Environment
- Python 3.10+ (main course)
- Conda for package management
- uv used in ma/ subproject
- pytest configured in .vscode/settings.json

## Quarto Config (_quarto.yml)
- Project type: book
- Themes: cosmo (light) / darkly (dark) with toggle
- Custom callouts: Links, Python, Libraries, Math, ML, Misc
- Search enabled, TOC on right, code tools (Copy/Collapse)

## Repository
- Single branch (main), clean history
- GitHub remote with one completed PR (#1 — stat audit)
- .gitignore covers Python + LaTeX artifacts, allows math/assets/knn.csv
