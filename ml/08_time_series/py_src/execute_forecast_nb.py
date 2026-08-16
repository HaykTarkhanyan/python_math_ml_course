"""Execute 31_electricity_forecast_solution.ipynb in place, with the `ma` kernel.

Do NOT use `python -m jupyter nbconvert` for this. On this machine the `jupyter`
entry point resolves to the SYSTEM Python 3.10, whose jupyter_contrib_nbextensions
is broken (`No module named 'notebook.services'`), so nbconvert dies at import.
Worse, when piped the shell reports exit 0 and the notebook is left unexecuted with
every cell's execution_count = None. nbclient in-process avoids all of that.

    ./ma/Scripts/python.exe ml/08_time_series/py_src/execute_forecast_nb.py
"""

import logging
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
NB = CH_DIR / "31_electricity_forecast_solution.ipynb"


def setup_logging() -> logging.Logger:
    logs = REPO_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log = logging.getLogger("execute_forecast_nb")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(logs / "execute_forecast_nb.log")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def main() -> None:
    log = setup_logging()
    nb = nbformat.read(NB, as_version=4)
    log.info(f"executing {NB.name}: {len(nb.cells)} cells")

    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(CH_DIR)}},
        allow_errors=False,          # fail loud: a broken cell must stop the build
    )
    client.execute()
    nbformat.write(nb, NB)

    code = [c for c in nb.cells if c.cell_type == "code"]
    unrun = [c for c in code if c.execution_count is None]
    figures = sum(1 for c in code for o in c.get("outputs", [])
                  if "image/png" in o.get("data", {}))
    empty = [c for c in code if not "".join(c.source).strip()]
    log.info(f"executed {len(code)} code cells, {figures} figures, "
             f"{len(empty)} empty, {len(unrun)} never ran")
    if unrun:
        raise RuntimeError(f"{len(unrun)} code cells produced no execution_count")
    log.info("notebook executed cleanly")


if __name__ == "__main__":
    main()
