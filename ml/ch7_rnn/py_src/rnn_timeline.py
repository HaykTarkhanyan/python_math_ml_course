"""Real figure for the L21 "Road to Attention" deck (Section 5: epilogue).

Generates into ml/ch7_rnn/fig/:
  rnn_timeline.pdf -- a 1990-2026 ribbon: Elman RNN -> vanishing-gradient diagnosis ->
                      LSTM -> seq2seq/GRU -> "Attention Is All You
                      Need" (RNNs retired from NLP) -> the 2023+ comeback (Mamba,
                      xLSTM). All facts web-verified at build (see L21_DECISIONS.md;
                      1990/1991 events added + re-verified 2026-09-06):
                        1990 "Finding Structure in Time" -- Elman (Cognitive Science)
                        1991 vanishing gradients diagnosed -- Hochreiter's diploma
                             thesis, TU Munich (in German)
                        1997 LSTM -- Hochreiter & Schmidhuber
                        2014 seq2seq -- Sutskever, Vinyals & Le; GRU -- Cho et al.
                        2017 "Attention Is All You Need" -- Vaswani et al.
                        2023 Mamba -- Gu & Dao
                        2024 xLSTM -- Beck et al. (Hochreiter, senior author)

No `ml/12_cnn/py_src/timeline_ribbon.py` exists in this repo to copy the exact visual
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


# (year, label, era, side, level) -- era in {"rnn", "pivot", "comeback"}; facts web-verified
# at build (2026-07-13, 2026-09-06, 2026-09-22). Redrawn 2026-10-03 at the slot size (L21:
# 0.86 x linewidth, ~4.7 in) with short labels: at that width one year is ~0.11 in, so the
# old two-line captions could not be read. side +1 = above the axis, -1 = below; level 2
# sits further out, for pairs a year or two apart. 2025: Qwen3-Next (Sept 2025), the first
# of the hybrid LLMs whose layers are mostly a fixed-size recurrent state (Gated DeltaNet)
# - the claim the epilogue teaser makes.
EVENTS = [
    (1990, "Elman RNN", "rnn", +1, 1, "center"),
    (1991, "vanishing gradients\ndiagnosed", "rnn", -1, 1, "left"),
    (1997, "LSTM", "rnn", +1, 1, "center"),
    (2014, "seq2seq, GRU", "rnn", -1, 1, "center"),
    (2017, "Attention Is All\nYou Need", "pivot", +1, 1, "center"),
    (2023, "Mamba", "comeback", -1, 1, "center"),
    (2024, "xLSTM", "comeback", +1, 1, "center"),
    (2025, "recurrent layers\ninside LLMs", "comeback", -1, 2, "right"),
]
# label anchor offset in years for edge labels, so they grow inward instead of off the page
HA_SHIFT = {"center": 0.0, "left": -1.6, "right": 1.6}
ERA_COLOR = {"rnn": ARMBLUE, "pivot": ARMRED, "comeback": ARMORANGE}
YEAR_MIN, YEAR_MAX = 1987.5, 2027.5


def fig_timeline(log):
    fig, ax = plt.subplots(figsize=(4.7, 1.75))
    ax.set_position([0.01, 0.02, 0.98, 0.96])
    ax.plot([YEAR_MIN, YEAR_MAX], [0, 0], color="#CCCCCC", lw=1.5, zorder=0)

    for year, label, era, side, level, ha in EVENTS:
        color = ERA_COLOR[era]
        ax.plot([year], [0], marker="o", markersize=6, color=color, zorder=3,
                markeredgecolor="black", markeredgewidth=0.4)
        reach = 0.32 if level == 1 else 0.95
        va = "bottom" if side > 0 else "top"
        ax.plot([year, year], [0, side * reach], color=color, lw=0.8, zorder=1)
        ax.text(year, side * (reach + 0.04), f"{year}", ha="center", va=va, fontsize=7.5,
                fontweight="bold", color=color)
        ax.text(year + HA_SHIFT[ha], side * (reach + 0.30), label, ha=ha, va=va, fontsize=7,
                color="black", linespacing=1.05)
        log.info(f"{year} [{era}]: {label.splitlines()[0]}")

    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_ylim(-2.05, 1.45)
    ax.axis("off")
    out = FIG_DIR / "rnn_timeline.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_timeline(log)
    log.info("done")


if __name__ == "__main__":
    main()
