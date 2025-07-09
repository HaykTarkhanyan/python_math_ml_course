# Quarto
- Files must be lowercase, I have no idea why. GitHub may not approach names in a case-sensitive way.
- Be careful not to have super long outputs (which may go unnoticed when their displayed on one line). Also you can add `--verbose` to quarto render.

# Useful commands
`make` or `make push`: Runs Quarto render, stages all changes, commits with the default message rendering YYYY-MM-DD HH:MM, and pushes.

`make push render=false`: Skips the Quarto render step, then stages, commits (same default message), and pushes.

`make push msg="cheese"`: Renders (unless `render=false`), stages, commits with your custom message, and pushes.

You can combine flags, e.g.
`make push render=false msg="cheese"` — skips rendering and commits with the specified message.
