---
name: Technology Stack & Build System
description: Tools, build process, deployment, and compilation details for the course repo
type: project
---

## Build & Deployment
- **GitHub Actions** auto-renders Quarto site on every push to `main` (~5 min)
  - Workflow: `.github/workflows/publish.yml`
  - Uses `quarto-dev/quarto-actions` + `actions/deploy-pages`
  - No local Quarto rendering needed anymore
- **GitHub Pages** deploys via Actions artifacts (not from docs/ branch)
- `docs/` is in `.gitignore` — no longer committed to git
- Old `Makefile` / `render_only_changed.py` still exist but are legacy (superseded by Actions)
- `_quarto.yml` paths must be exact case — Linux CI is case-sensitive

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
- `execute: enabled: false` — notebooks are pre-rendered, not executed during build

## Repository
- Public repo: https://github.com/HaykTarkhanyan/python_math_ml_course
- Single branch (main), clean history
- .gitignore covers Python + LaTeX artifacts + docs/
