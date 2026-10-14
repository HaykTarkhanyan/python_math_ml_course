"""Standard vs depthwise-separable convolution, drawn as slabs - for the [51] Vision Tasks deck.

Frame "Separable convolutions: the idea", same numbers as the count frame (separable_count.py):
input 12x12x3, 5x5 filters, 256 output channels, output 8x8.

  top    standard: each of 256 filters is 5x5x3 - space and channels in one go -> 8x8x256.
  bottom depthwise: 3 filters of 5x5x1, one per channel (space only) -> 8x8x3;
         pointwise: 256 filters of 1x1x3 (channels only) -> 8x8x256.

Weight counts printed on the figure are asserted: standard 5*5*3*256 = 19,200;
separable 5*5*3 + 3*256 = 75 + 768 = 843.

Generates into ml/12_cnn/fig/:
  separable_idea.pdf

Run with the project venv (~2 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/separable_idea.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
CH_COLORS = ["#f0a3a3", "#a6dca6", "#a8bff0"]     # R, G, B channels
OUT_GRAY = "#c9c9c9"
FS = 7.2

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

K, C_IN, C_OUT = 5, 3, 256


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l52_separable_idea")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "separable_idea.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def slab(ax, x, y, size, colors, d=0.045):
    """Squares stacked back to front, offset up-right: a pseudo-3D tensor. (x, y) = front centre."""
    n = len(colors)
    for k in reversed(range(n)):
        ax.add_patch(Rectangle((x - size / 2 + k * d, y - size / 2 + k * d), size, size,
                               fc=colors[k], ec="black", lw=0.5))
    return x + size / 2 + (n - 1) * d                 # right edge


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7,
                                 color="black", lw=0.8))


def label(ax, x, y, text, **kw):
    ax.text(x, y, text, ha="center", va="top", fontsize=FS, linespacing=1.1, **kw)


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    std_w, dw_w, pw_w = K * K * C_IN * C_OUT, K * K * C_IN, C_IN * C_OUT
    log.info(f"weights: standard {std_w:,}; depthwise {dw_w} + pointwise {pw_w} = {dw_w + pw_w}")
    if (std_w, dw_w + pw_w) != (19_200, 843):
        raise AssertionError("weight counts drifted from the figure text")

    W, H = 5.45, 2.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    many = [OUT_GRAY] * 9

    # row 1: standard
    y = 1.74
    ax.text(0.02, H - 0.04, "standard", fontsize=8, fontweight="bold", va="top")
    r = slab(ax, 0.45, y, 0.5, CH_COLORS)
    label(ax, 0.5, y - 0.33, "input\n12x12x3")
    arrow(ax, r + 0.06, r + 0.3, y)
    r = slab(ax, r + 0.5, y, 0.22, CH_COLORS)
    label(ax, r - 0.12, y - 0.33, "one 5x5x3\nfilter")
    arrow(ax, r + 0.08, r + 0.32, y)
    r2 = slab(ax, r + 0.55, y, 0.36, [OUT_GRAY])
    label(ax, r2 - 0.18, y - 0.33, "1 map\n8x8")
    ax.text(r2 + 0.1, y, "x 256", fontsize=FS, va="center")
    r3 = slab(ax, r2 + 0.75, y, 0.36, many, d=0.03)
    label(ax, r3 - 0.25, y - 0.33, "8x8x256")
    ax.text(r3 + 0.15, y + 0.12, "every filter does\nspace + channels\nat once",
            fontsize=FS, va="center", linespacing=1.1)
    ax.text(r3 + 0.15, y - 0.3, f"{std_w:,} weights", fontsize=FS, va="center",
            fontweight="bold")

    # row 2: depthwise, then pointwise
    y = 0.64
    ax.text(0.02, 1.13, "separable", fontsize=8, fontweight="bold", va="top")
    r = slab(ax, 0.45, y, 0.5, CH_COLORS)
    label(ax, 0.5, y - 0.33, "input\n12x12x3")
    arrow(ax, r + 0.06, r + 0.25, y)
    xs = r + 0.4
    for k, c in enumerate(CH_COLORS):                  # 3 separate 5x5x1 filters
        ax.add_patch(Rectangle((xs + k * 0.25 - 0.1, y - 0.1), 0.2, 0.2, fc=c, ec="black",
                               lw=0.5))
    label(ax, xs + 0.25, y - 0.18, "1. depthwise:\n3 filters 5x5x1,\none per channel")
    arrow(ax, xs + 0.62, xs + 0.84, y)
    r = slab(ax, xs + 1.06, y, 0.36, CH_COLORS)
    label(ax, r - 0.22, y - 0.33, "8x8x3")
    arrow(ax, r + 0.06, r + 0.22, y)
    xp = r + 0.36
    for k, c in enumerate(CH_COLORS):                  # one 1x1x3 filter
        ax.add_patch(Rectangle((xp - 0.04 + k * 0.03, y - 0.04 + k * 0.03), 0.08, 0.08,
                               fc=c, ec="black", lw=0.5))
    label(ax, xp + 0.03, y - 0.18, "2. pointwise:\n1x1x3, x 256")
    arrow(ax, xp + 0.2, xp + 0.4, y)
    r3 = slab(ax, xp + 0.66, y, 0.36, many, d=0.03)
    label(ax, r3 - 0.25, y - 0.33, "8x8x256")
    ax.text(r3 + 0.15, y + 0.12, "space first,\nchannels second",
            fontsize=FS, va="center", linespacing=1.1)
    ax.text(r3 + 0.15, y - 0.2, f"{dw_w} + {pw_w}\n= {dw_w + pw_w} weights", fontsize=FS,
            va="center", fontweight="bold", linespacing=1.1)

    out = FIG_DIR / "separable_idea.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out}; done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
