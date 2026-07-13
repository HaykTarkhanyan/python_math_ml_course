"""Real figure for the L16 CNN Foundations deck (Section 3, parameter explosion).

Generates into ml/ch6_cnn/fig/:
  param_explosion.pdf  -- one 100x100x3 image, one layer that keeps the 100x100 output:
                          a dense layer needs 300,000,000 weights; a single 5x5 conv
                          filter needs 75. Log-scale labeled bars.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/param_explosion.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag palette, value labels on bars.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
ARM_BLUE, ARM_RED = "#0033A0", "#D90012"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_param_explosion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "param_explosion.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)

    # 100x100x3 input; one layer that produces a 100x100 output map.
    dense = (100 * 100 * 3) * (100 * 100)     # 30,000 inputs x 10,000 outputs = 3e8
    conv = 5 * 5 * 3                          # one 5x5 filter across depth 3 = 75
    log.info(f"dense={dense:,}  conv={conv:,}  ratio={dense // conv:,}x")

    labels = ["Dense layer\n(same output size)", "One 5x5 conv filter"]
    vals = [dense, conv]
    colors = [ARM_RED, ARM_BLUE]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e9)
    ax.set_ylabel("parameters (log scale)")
    ax.bar_label(bars, labels=[f"{v:,}" for v in vals], padding=5, fontsize=12)
    ax.set_title("100x100x3 image, one layer: dense vs convolution")
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.5, 0.90, f"{dense // conv:,}x fewer", transform=ax.transAxes,
            ha="center", fontsize=11, color=ARM_BLUE, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "param_explosion.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
