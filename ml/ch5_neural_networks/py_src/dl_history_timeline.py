"""Real figure for the L14 Neural Networks deck (Section 1: Why neural networks?).

Generates into ml/ch5_neural_networks/fig/:
  dl_history.pdf -- a 1943-2020s ribbon of deep learning's milestones, through two AI
                    winters, in the house timeline style (matches ch7 rnn_timeline.py):
                      1943 McCulloch-Pitts artificial neuron
                      1958 Perceptron (Rosenblatt)
                      1969 AI winter: XOR limit (Minsky & Papert)
                      1986 Backprop popularized (Rumelhart, Hinton, Williams)
                      1998 LeNet-5: CNNs read digits (LeCun)
                      2012 AlexNet wins ImageNet (the deep-learning breakthrough)
                      2017 Transformers ("Attention Is All You Need", Vaswani et al.)
                      2020s Foundation models and LLMs (GPT, Claude)
                    Dates are textbook-standard DL history.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch5_neural_networks/py_src/dl_history_timeline.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
ARMBLUE, ARMRED, ARMORANGE, GRAY = "#0033A0", "#D90012", "#F2A800", "#888888"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l14_dl_history")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "dl_history_timeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# (year_x, year_label, text, era, above) -- era in {"early", "winter", "deep", "modern"};
# `above` is hand-assigned so tightly-spaced beats land on opposite sides of the axis.
EVENTS = [
    (1943, "1943", "McCulloch-Pitts\nartificial neuron", "early", True),
    (1958, "1958", "Perceptron\n(Rosenblatt)", "early", False),
    (1969, "1969", "AI winter: XOR limit\n(Minsky & Papert)", "winter", True),
    (1986, "1986", "Backprop popularized\n(Rumelhart, Hinton, Williams)", "early", False),
    (1998, "1998", "LeNet-5: CNNs\nread digits (LeCun)", "deep", True),
    (2012, "2012", "AlexNet wins\nImageNet", "deep", False),
    (2017, "2017", '"Attention Is All\nYou Need" (transformers)', "modern", True),
    (2022, "2020s", "Foundation models\n& LLMs (GPT, Claude)", "modern", False),
]
ERA_COLOR = {"early": ARMBLUE, "winter": GRAY, "deep": ARMORANGE, "modern": ARMRED}
YEAR_MIN, YEAR_MAX = 1940, 2027


def fig_timeline(log):
    fig, ax = plt.subplots(figsize=(12.8, 3.7))
    ax.plot([YEAR_MIN, YEAR_MAX], [0, 0], color="#CCCCCC", lw=2.0, zorder=0)

    for year, ylab, label, era, above in EVENTS:
        color = ERA_COLOR[era]
        big = era == "deep" and year == 2012        # AlexNet = the breakthrough, emphasize it
        ax.plot([year], [0], marker="o", markersize=15 if big else 12, color=color,
                zorder=3, markeredgecolor="black", markeredgewidth=0.6)
        y_text = 0.62 if above else -0.62
        va = "bottom" if above else "top"
        ax.plot([year, year], [0, y_text * 0.82], color=color, lw=1.2, zorder=1)
        ax.text(year, y_text, ylab, ha="center", va=va, fontsize=12,
                fontweight="bold", color=color)
        y_label = y_text + (0.16 if above else -0.16)
        ax.text(year, y_label, label, ha="center", va=va, fontsize=8.6, color="black")
        log.info(f"{year} [{era}]: {label.splitlines()[0]}")

    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_title("One idea, 80 years in the making: a brief history of deep learning",
                 fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "dl_history.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_timeline(log)
    log.info("done")


if __name__ == "__main__":
    main()
