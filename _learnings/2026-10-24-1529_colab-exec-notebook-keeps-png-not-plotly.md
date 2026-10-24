# `colab exec -f notebook.ipynb` returns every output, but Plotly figures come back as HTML that GitHub cannot show

**Symptom.** We wanted the RNN project's solution notebook to have all its outputs without running
anything on the laptop. The `colab` CLI (WSL, `colab-gpu` skill) can execute a whole notebook:
`colab exec -s NAME --timeout 3600 -f nb.ipynb` writes `nb_output.ipynb` next to the input.

**What comes back** (tested 2026-10-24 on a 4-cell notebook, then the 41-cell project):

```
0 None [('stream', '3.14.6\n')]                          %pip install + print
1 None [('display_data', ['image/png', 'text/plain'])]   matplotlib: PNG embedded
2 None [('display_data', ['text/html'])]                 plotly: HTML that needs plotly.js
3 None [('stream', 'done\n')]
```

- `%pip install` magics work. Printed text and matplotlib PNGs are kept.
- Plotly figures are kept only as `text/html` that loads plotly.js. GitHub's notebook viewer runs
  no JavaScript, so they show as nothing there.
- `execution_count` is `None` for every cell.
- Kernel state persists across `colab exec` calls in one session. Run `colab restart-kernel -s NAME`
  before the real run, so a smoke-test run's variables cannot hide a bug.

**Consequences.**
- For notebooks that students read on GitHub, draw figures with matplotlib. This is a deliberate
  exception to the global "Plotly by default" rule.
- Build with a `--merge executed.ipynb` mode (see `ml/13_rnns/py_src/build_seq2seq_project_nb.py`).
  It rebuilds the notebook from the builder and copies each code cell's outputs from the executed
  copy, failing loudly if any code cell's source differs. Prose can then be fixed after the run
  without paying for another GPU run.
- Smoke-test first: a copy with every training step count cut to 30 caught nothing here, but it
  costs about a minute against about 11 for the real run.
