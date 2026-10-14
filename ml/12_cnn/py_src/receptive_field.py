"""Figures for the L17 CNN Architectures deck (Section 3, receptive field -> VGG).

Generates into ml/12_cnn/fig/:
  receptive_field.pdf  -- how the receptive field of one output neuron grows when 3x3
                          convolutions are stacked: 3x3 after one layer, 5x5 after two, 7x7
                          after three - the same reach as a single 7x7 filter.
  vgg_cost.pdf         -- (split out 2026-10-01) the price of that reach: three stacked 3x3 layers
                          need 3 * 9 C^2 = 27 C^2 weights, one 7x7 layer needs 49 C^2 (VGG paper,
                          Simonyan & Zisserman 2015, Sec. 2.3: "81% more").

2026-10-01: the two panels used to be one figure on the receptive-field frame, drawn 2.3x too wide
(4.1 pt text). The deck now teaches the receptive field FIRST and VGG's cost argument SECOND
(instructor: "VGG - I didn't get it"), so each panel is its own figure, drawn at its slide size.

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/receptive_field.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l17_receptive_field")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "receptive_field.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def receptive_field(log):
    fig, ax = plt.subplots(figsize=(3.1, 2.45))     # its size on the slide
    ax.set_position([0, 0, 1, 1])
    n = 7
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((j, i), 1, 1, fc="#f2f2f2", ec="#dddddd", lw=0.6))
    c = n // 2
    ax.add_patch(Rectangle((c, c), 1, 1, fc="#999999", ec="#666", lw=0.6))
    specs = [(3, ARM_RED, "1 layer: 3x3"), (5, ARM_BLUE, "2 layers: 5x5"),
             (7, ARM_ORANGE, "3 layers: 7x7")]
    for size, color, _ in specs:
        h = size // 2
        ax.add_patch(Rectangle((c - h, c - h), size, size, fc="none", ec=color, lw=2.0))
    for k, (_, color, label) in enumerate(specs):
        y = n - 0.6 - k * 0.95
        ax.plot([7.5, 8.2], [y, y], color=color, lw=2.0)
        ax.text(8.35, y, label, color=color, fontsize=8.5, va="center")
    ax.text(8.35, n - 0.6 - 3 * 0.95, "grey = the\nneuron's own\nposition", color="#555",
            fontsize=7.5, va="top")
    ax.set_xlim(-0.2, 12.6); ax.set_ylim(-0.2, n + 0.2)
    ax.set_aspect("equal"); ax.axis("off")
    out = FIG_DIR / "receptive_field.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    log.info(f"saved {out}")


def vgg_cost(log):
    coeffs = [27, 49]                  # 3 * (3*3) C^2 vs 7*7 C^2
    if (coeffs[1] - coeffs[0]) / coeffs[0] < 0.8:
        raise ValueError("the VGG paper's '81% more' no longer holds")
    fig, ax = plt.subplots(figsize=(2.3, 2.0))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.2)
    bars = ax.bar(["three 3x3", "one 7x7"], coeffs, width=0.6, color=[ARM_BLUE, ARM_RED])
    ax.bar_label(bars, labels=[f"{c} C²" for c in coeffs], padding=2, fontsize=9,
                 fontweight="bold")
    ax.set_ylim(0, 60)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_title("weights, same 7x7 view", fontsize=9)
    out = FIG_DIR / "vgg_cost.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out} ({coeffs[1] / coeffs[0] - 1:.0%} more for one 7x7)")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    receptive_field(log)
    vgg_cost(log)


if __name__ == "__main__":
    main()
