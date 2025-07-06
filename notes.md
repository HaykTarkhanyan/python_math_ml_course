# Quarto
- Files must be lowercase, I have no idea why. GitHub may not approach names in a case-sensitive way.
- Be careful not to have super long outputs (which may go unnoticed when their displayed on one line). Also you can add `--verbose` to quarto render.

# Useful commands
`make render`: quarto render-s and pushes to GitHub
`make push`: just git add, commit and push
`make render/push msg="message"`: renders and pushes with a custom message (defaults to "rendering")