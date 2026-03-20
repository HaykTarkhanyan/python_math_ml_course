# Quarto
- File paths in `_quarto.yml` must match exact case on disk — Linux CI (GitHub Actions) is case-sensitive even if Windows isn't.
- Be careful not to have super long outputs (which may go unnoticed when displayed on one line). Also you can add `--verbose` to quarto render.
- Quarto sols: `quarto render 05_probability.qmd --profile=solution --output 05_probability_sol.html`

## Armenian
Just add this to the YAML header of the file:
```yaml
format:
  pdf:
    include-in-header:
      - text: |
          \usepackage{armtex}
```


# Useful commands

## Deployment

**GitHub Actions** automatically renders the Quarto site and deploys to GitHub Pages on every push to `main` (~5 min). See `.github/workflows/publish.yml`.

No local rendering needed — just `git push` and the site updates.

### Legacy: Make Commands (no longer needed for deployment)

The Makefile `make push` target used to render locally and commit `docs/`. This is now superseded by GitHub Actions, but the Makefile still works for local testing:

- `make push render=false msg="your message"`: Just commit and push (recommended now)
- `make push`: Full local render + push (legacy)

# Tex in VS Code
- Install the LaTeX Workshop extension.
- install the TeX Live distribution.
- `tlmgr install latexmk`
- if fails, `tlmgr update --self`
- then repeat the previous command.
- to check if everything is fine, run `latexmk -v`