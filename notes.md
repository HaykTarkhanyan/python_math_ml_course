# Quarto
- Files must be lowercase, I have no idea why. GitHub may not approach names in a case-sensitive way.
- Be careful not to have super long outputs (which may go unnoticed when displayed on one line). Also you can add `--verbose` to quarto render.

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
`make` or `make push`: Runs Quarto render, stages all changes, commits with the default message rendering YYYY-MM-DD HH:MM, and pushes.

`make push render=false`: Skips the Quarto render step, then stages, commits (same default message), and pushes.

`make push msg="cheese"`: Renders (unless `render=false`), stages, commits with your custom message, and pushes.

You can combine flags, e.g.
`make push render=false msg="cheese"` — skips rendering and commits with the specified message.

# Tex in VS Code
- Install the LaTeX Workshop extension.
- install the TeX Live distribution.
- `tlmgr install latexmk`
- if fails, `tlmgr update --self`
- then repeat the previous command.
- to check if everything is fine, run `latexmk -v`