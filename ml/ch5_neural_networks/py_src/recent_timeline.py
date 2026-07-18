"""Real figure for the dl_intro_history deck (our modern-applications addendum).

Generates into ml/ch5_neural_networks/fig/:
  recent_timeline.pdf -- a 2012-2026 ribbon of deep learning's RECENT explosion, in the
                         house timeline style (matches dl_history_timeline.py). Events are
                         EQUALLY SPACED (not true-to-year) so the dense 2024-2026 cluster
                         does not collide. Three eras in the Armenian-flag palette:
                           deep-vision (2012-2016, orange)
                           transformers & scale (2017-2020, blue)
                           foundation / generative / reasoning (2022-2026, red)

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch5_neural_networks/py_src/recent_timeline.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
ARMBLUE, ARMRED, ARMORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("recent_timeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "recent_timeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# (year_label, text, era, above); events are placed at equal x = index.
EVENTS = [
    ("2012", "AlexNet wins\nImageNet", "vision", True),
    ("2014", "GANs\n(Goodfellow)", "vision", False),
    ("2016", "AlphaGo beats\nworld champion", "vision", True),
    ("2017", '"Attention Is All\nYou Need"', "scale", False),
    ("2018", "BERT & GPT:\npretraining era", "scale", True),
    ("2020", "GPT-3 (175B)\n& AlphaFold2", "scale", False),
    ("2022", "ChatGPT &\nStable Diffusion", "found", True),
    ("2023", "GPT-4:\nmultimodal", "found", False),
    ("2024", "Two AI Nobels\n(Physics + Chem.)", "found", True),
    ("2025", "IMO gold; reasoning\nmodels; DeepSeek-R1", "found", False),
    ("2026", "Agents & reasoning\ngo mainstream", "found", True),
]
ERA_COLOR = {"vision": ARMORANGE, "scale": ARMBLUE, "found": ARMRED}


def fig_timeline(log):
    n = len(EVENTS)
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.plot([-0.6, n - 0.4], [0, 0], color="#CCCCCC", lw=2.0, zorder=0)

    for i, (ylab, label, era, above) in enumerate(EVENTS):
        color = ERA_COLOR[era]
        big = ylab in ("2012", "2017", "2022")   # era-opening milestones
        ax.plot([i], [0], marker="o", markersize=15 if big else 12, color=color,
                zorder=3, markeredgecolor="black", markeredgewidth=0.6)
        y_text = 0.62 if above else -0.62
        va = "bottom" if above else "top"
        ax.plot([i, i], [0, y_text * 0.82], color=color, lw=1.2, zorder=1)
        ax.text(i, y_text, ylab, ha="center", va=va, fontsize=13,
                fontweight="bold", color=color)
        y_label = y_text + (0.17 if above else -0.17)
        ax.text(i, y_label, label, ha="center", va=va, fontsize=9.0, color="black")
        log.info(f"{ylab} [{era}]: {label.splitlines()[0]}")

    # era legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ARMORANGE,
                   markersize=11, label="deep vision (2012-16)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ARMBLUE,
                   markersize=11, label="transformers & scale (2017-20)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ARMRED,
                   markersize=11, label="foundation / generative / reasoning (2022-26)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, fontsize=9.5, handletextpad=0.2, columnspacing=1.2)

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_title("Deep learning, 2012-2026: the recent explosion", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "recent_timeline.pdf"
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
