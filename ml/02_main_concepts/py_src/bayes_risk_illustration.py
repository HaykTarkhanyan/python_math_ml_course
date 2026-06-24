"""Generate the Bayes-risk / noise-floor figure for the L03 Regularization deck.

Produces one PDF into ``ml/upcoming_lectures/fig/`` using the SAME DGP as
the cold-open (noisy cubic, sigma=20) for visual continuity:

  l03_bayes_risk.pdf -- the true cubic drawn through the data with a +-sigma
                        band, annotated with the irreducible error sigma^2.

Point of the frame: even an oracle that knows the true f cannot beat MSE = sigma^2.
Regularization closes the gap ABOVE this floor, never below it.

Run with the project venv (see repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/upcoming_lectures/py_src/bayes_risk_illustration.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 509
SIGMA = 20

ARM_BLUE = "#0033A0"
ARM_ORANGE = "#F2A800"

HERE = Path(__file__).resolve()
UPCOMING_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = UPCOMING_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l03_bayes_risk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "l03_bayes_risk.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def true_cubic(x):
    return 1 + 2 * x + 3 * x ** 2 + 4 * x ** 3


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)

    np.random.seed(SEED)
    N = 100
    X = np.linspace(-3, 3, N)
    y = true_cubic(X) + np.random.normal(0, SIGMA, size=N)
    xx = np.linspace(-3, 3, 400)
    yt = true_cubic(xx)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(X, y, s=20, color=ARM_BLUE, alpha=0.6, label="observed data")
    ax.plot(xx, yt, "--", color="black", lw=2.5, label="true $f$ (Bayes-optimal)")
    ax.annotate(r"even the true $f$ leaves scatter $\sigma^2 = 400$",
                xy=(1.2, true_cubic(1.2)),
                xytext=(-2.9, 130), fontsize=11, color="black",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("The noise floor: no model beats the irreducible error")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "l03_bayes_risk.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name} (sigma={SIGMA}, sigma^2={SIGMA**2}, N={N})")
    logger.info("done.")


if __name__ == "__main__":
    main()
