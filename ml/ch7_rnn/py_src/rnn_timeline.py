"""Real figure for the L21 "Road to Attention" deck (Section 5: epilogue).

Generates into ml/ch7_rnn/fig/:
  rnn_timeline.pdf -- a 1997-2026 ribbon: LSTM -> seq2seq/GRU -> "Attention Is All You
                      Need" (RNNs retired from NLP) -> the 2023+ comeback (Mamba,
                      xLSTM). All facts web-verified at build (see L21_DECISIONS.md):
                        1997 LSTM -- Hochreiter & Schmidhuber
                        2014 seq2seq -- Sutskever, Vinyals & Le; GRU -- Cho et al.
                        2017 "Attention Is All You Need" -- Vaswani et al.
                        2023 Mamba -- Gu & Dao
                        2024 xLSTM -- Beck et al. (Hochreiter, senior author)

No `ml/ch6_cnn/py_src/timeline_ribbon.py` exists in this repo to copy the exact visual
language from (checked at build time) -- this is a fresh, from-scratch ribbon in the
same house style (horizontal axis, boxed year/event labels, Armenian-palette colors for
the three eras: pre-2017 RNN era, the 2017 transformer pivot, the 2023+ comeback).

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch7_rnn/py_src/rnn_timeline.py
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
    logger = logging.getLogger("l21_rnn_timeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "rnn_timeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# (year, label, era, above) -- era in {"rnn", "pivot", "comeback"}; web-verified at
# build. `above` is assigned by hand (not alternated by index) so tightly-spaced pairs
# (2014/2017, 2023/2024) land on opposite sides of the axis instead of colliding.
EVENTS = [
    (1997, "LSTM\n(Hochreiter & Schmidhuber)", "rnn", True),
    (2014, "seq2seq (Sutskever et al.)\nGRU (Cho et al.)", "rnn", False),
    (2017, '"Attention Is All\nYou Need" (Vaswani et al.)', "pivot", True),
    (2023, "Mamba (Gu & Dao)\nstate-space models return", "comeback", False),
    (2024, "xLSTM (Beck et al.)\nHochreiter, senior author", "comeback", True),
]
ERA_COLOR = {"rnn": ARMBLUE, "pivot": ARMRED, "comeback": ARMORANGE}
YEAR_MIN, YEAR_MAX = 1996, 2026.5


def fig_timeline(log):
    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    ax.plot([YEAR_MIN, YEAR_MAX], [0, 0], color="#CCCCCC", lw=2.0, zorder=0)

    for year, label, era, above in EVENTS:
        color = ERA_COLOR[era]
        ax.plot([year], [0], marker="o", markersize=13, color=color, zorder=3,
                markeredgecolor="black", markeredgewidth=0.6)
        y_text = 0.62 if above else -0.62
        va = "bottom" if above else "top"
        ax.plot([year, year], [0, y_text * 0.82], color=color, lw=1.2, zorder=1)
        ax.text(year, y_text, f"{year}", ha="center", va=va, fontsize=12,
                fontweight="bold", color=color)
        y_label = y_text + (0.16 if above else -0.16)
        ax.text(year, y_label, label, ha="center", va=va, fontsize=8.7, color="black")
        log.info(f"{year} [{era}]: {label.splitlines()[0]}")

    ax.text(2026, 0.95, "2026: RNNs still earn their keep in streaming/edge\n"
            "and small-data settings", ha="center", va="bottom", fontsize=9,
            color=GRAY, style="italic")

    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([])
    ax.set_xlabel("")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_title("Recurrence, retirement, and a comeback: 1997-2026", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "rnn_timeline.pdf"
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
