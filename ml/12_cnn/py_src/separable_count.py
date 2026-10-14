"""Separable-convolution multiplication-count bars for the L19 deck (Section 2).

The locked worked example (LMU cnn2 / Bai 2019 numbers): input 12x12x3, 5x5 filters,
256 output channels, valid conv -> output 8x8.

  standard conv:      256 filters x (5*5*3) mults x (8*8) positions = 1,228,800
  depthwise:          3 filters x (5*5) x (8*8)                     =     4,800
  pointwise (1x1):    256 filters x 3 x (8*8)                       =    49,152
  separable total:                                                  =    53,952

All four numbers are recomputed in code and asserted against the spec before
plotting - the frame text quotes them exactly. ~23x fewer multiplications.

Generates into ml/12_cnn/fig/:
  separable_count.pdf

Run with the project venv:
    ./ma/Scripts/python.exe ml/12_cnn/py_src/separable_count.py

2026-10-01: redrawn at its slide size (figsize x 0.42, fonts mapped to 7-10.5 pt)
so the labels are readable on the slide; they rendered at 2-5 pt before.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8})

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l19_separable_count")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "separable_count.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)

    k, c_in, c_out = 5, 3, 256
    out_hw = 12 - k + 1          # valid conv on a 12x12 input
    positions = out_hw * out_hw  # 8*8 = 64

    standard = c_out * (k * k * c_in) * positions
    depthwise = c_in * (k * k) * positions
    pointwise = c_out * c_in * positions
    separable = depthwise + pointwise
    ratio = standard / separable
    log.info(f"standard={standard:,} depthwise={depthwise:,} "
             f"pointwise={pointwise:,} separable={separable:,} ratio={ratio:.1f}x")
    if (standard, depthwise, pointwise, separable) != (1228800, 4800, 49152, 53952):
        raise AssertionError("multiplication counts drifted from the locked spec")

    fig, ax = plt.subplots(figsize=(3.19, 1.85))
    labels = ["standard\n5x5 conv", "depthwise +\npointwise"]
    bars = ax.bar(labels, [standard, separable], width=0.55,
                  color=[ARM_BLUE, ARM_ORANGE])
    ax.bar_label(bars, labels=[f"{standard:,}", f"{separable:,}"], padding=2,
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    ax.set_ylim(0, standard * 1.16)
    ax.set_ylabel("multiplications", fontsize=8)
    ax.set_yticks([])                      # the bars carry their own values
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    # the depthwise / pointwise split is spelled out in the slide's equations, not here
    ax.annotate(f"~{ratio:.0f}x fewer",
                xy=(1.0, standard * 0.15), xytext=(1.0, standard * 0.62),
                ha="center", fontsize=9, fontweight="bold", color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.04))

    out = FIG_DIR / "separable_count.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
