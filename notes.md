# Quarto
- Files must be lowercase, I have no idea why. GitHub may not approach names in a case-sensitive way.
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

## Make Commands

The Makefile has been updated to support a new workflow with separate commits for manual changes and rendered output.

### Basic Usage
- `make` or `make push`: 
  1. Pulls latest changes
  2. Commits current changes with default message "rendering YYYY-MM-DD HH:MM"
  3. Renders only changed notebooks (using Python script)
  4. Commits rendered files with message "render YYYY-MM-DD HH:MM"
  5. Pushes all commits

### Flags

#### Rendering Control
- `make push render=false`: Skips the rendering step entirely
- `make push all=true`: Renders ALL files instead of just changed ones (uses `quarto render`)
- `make push all=false`: Renders only changed files (uses `render_only_changed.py`) - **default**

#### Custom Messages
- `make push msg="your message"`: Uses custom commit message for your manual changes

### Examples
```bash
# Default: commit changes, render only changed files, push
make push

# Commit with custom message, render only changed files
make push msg="Added new exercise to notebook 15"

# Full render of all files
make push all=true msg="Complete rebuild after major changes"

# Skip rendering entirely
make push render=false msg="Updated documentation only"

# Combinations
make push all=true render=true msg="Major update - full rebuild needed"
```

# Tex in VS Code
- Install the LaTeX Workshop extension.
- install the TeX Live distribution.
- `tlmgr install latexmk`
- if fails, `tlmgr update --self`
- then repeat the previous command.
- to check if everything is fine, run `latexmk -v`