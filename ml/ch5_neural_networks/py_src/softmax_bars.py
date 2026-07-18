"""Real figure for the dl_multilayer_nets deck (multi-class / softmax section).

Generates into ml/ch5_neural_networks/fig/:
  softmax_bars.pdf -- two bar charts showing softmax turning raw scores (logits) into a
                      probability distribution. Uses the SAME numbers as LMU's embedded
                      forward-pass example: logits f_in = (0.57, 2.95, -0.92) -> softmax
                      f = (0.083, 0.897, 0.018). The winning class is highlighted.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch5_neural_networks/py_src/softmax_bars.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
ARMBLUE, ARMRED = "#0033A0", "#D90012"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("softmax_bars")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "softmax_bars.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def fig_softmax(log):
    classes = ["class 1", "class 2", "class 3"]
    logits = np.array([0.57, 2.95, -0.92])          # LMU's f_in
    exp = np.exp(logits)
    probs = exp / exp.sum()                          # -> (0.083, 0.897, 0.018)
    log.info(f"logits={logits}, probs={np.round(probs,3)}, sum={probs.sum():.3f}")
    win = int(np.argmax(probs))
    colors = [ARMRED if i == win else ARMBLUE for i in range(3)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    bL = axL.bar(classes, logits, color=colors, edgecolor="black", linewidth=0.5)
    axL.bar_label(bL, labels=[f"{v:.2f}" for v in logits], padding=3,
                  fontsize=12, fontweight="bold")
    axL.set_title("Raw scores (logits $f_{in}$)", fontsize=13)
    axL.axhline(0, color="#888888", lw=0.8)
    axL.set_ylim(-1.6, 3.6)

    bR = axR.bar(classes, probs, color=colors, edgecolor="black", linewidth=0.5)
    axR.bar_label(bR, labels=[f"{v:.2f}" for v in probs], padding=3,
                  fontsize=12, fontweight="bold")
    axR.set_title("Softmax probabilities $f$  (sum $= 1$)", fontsize=13)
    axR.set_ylim(0, 1.05)

    for ax in (axL, axR):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11)

    fig.suptitle("Softmax turns arbitrary scores into a probability distribution",
                 fontsize=14)
    # arrow between the two panels
    fig.text(0.5, 0.5, r"$\longrightarrow$", ha="center", va="center", fontsize=26)
    fig.text(0.5, 0.585, "softmax", ha="center", va="center", fontsize=11, color="#444444")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "softmax_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_softmax(log)
    log.info("done")


if __name__ == "__main__":
    main()
