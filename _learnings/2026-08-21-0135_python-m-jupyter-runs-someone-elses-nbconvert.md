# `python -m jupyter nbconvert` can silently run a different Python's nbconvert

**Symptom.** `./ma/Scripts/python.exe -m jupyter nbconvert --execute ...` on the glitch-token
notebook crashed with `ModuleNotFoundError: No module named 'notebook.services'` - inside
`C:\Users\hayk_\AppData\Local\Programs\Python\Python310\lib\site-packages\`. The venv's own
nbconvert never ran. Worse, the failure initially looked like success because the command was
piped through `| tail -20`, and a pipeline's exit code is the LAST command's - `tail` returned 0.

**Cause.** `python -m jupyter <subcommand>` does not import the subcommand; it spawns a
`jupyter-nbconvert.EXE` found on PATH. On this machine PATH hits the system Python 3.10 first,
which has a stale `jupyter_contrib_nbextensions` that crashes on import. So the `ma` venv's
interpreter faithfully launched a different installation's broken entry point.

**Fix / rule.** Two rules, both cheap:

- Run nbconvert as a module of the interpreter you mean: `python -m nbconvert --execute ...`
  (works, stays inside the venv). Same for other jupyter subcommands: `-m jupytext`, not
  `-m jupyter jupytext`.
- Never judge a command through a pipe. `cmd | tail` reports `tail`'s exit code; drop the pipe
  or `set -o pipefail` before trusting green.
