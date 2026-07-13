"""Real figure for the L16 CNN Foundations deck (Section 2, edges are derivatives).

Generates into ml/ch6_cnn/fig/:
  edge_1d.pdf  -- a 1D intensity profile with an edge (top) and its discrete derivative
                  f(x+1)-f(x) (bottom), which spikes exactly at the edge. The point: an
                  edge is a large derivative, and the derivative is a difference of
                  neighbours.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/edge_1d.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 509
ARM_BLUE, ARM_RED = "#0033A0", "#D90012"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_edge_1d")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "edge_1d.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    x = np.arange(60)
    # clean piecewise profile: flat 40, a short ramp over [28,32], flat 205 (no blur, so
    # no boundary artifacts in the difference).
    f = np.clip(40 + (205 - 40) * (x - 28) / 4.0, 40, 205)
    d = np.zeros_like(f); d[1:] = f[1:] - f[:-1]       # forward difference f(x+1)-f(x)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 4.2), sharex=True)
    a1.plot(x, f, color=ARM_BLUE, lw=2.2)
    a1.set_ylabel("intensity f(x)")
    a1.set_title("An edge is a fast change in intensity")
    a1.grid(alpha=0.18)
    a2.plot(x, d, color=ARM_RED, lw=2.2)
    a2.axhline(0, color="gray", lw=0.7)
    a2.set_ylabel("f(x+1) - f(x)")
    a2.set_xlabel("x (pixel)")
    a2.set_title("Its derivative (a difference of neighbours) spikes at the edge")
    a2.grid(alpha=0.18)
    fig.tight_layout()
    out = FIG_DIR / "edge_1d.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
