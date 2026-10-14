"""Figures for the L16 CNN Foundations deck (Section 4, channels).

Generates into ml/12_cnn/fig/:
  multichannel_conv.pdf  -- one filter on a colour patch: each 2x2 kernel slice meets its own
                            channel (R, G, B), the three dot products and the bias add up to ONE
                            number. Numbers are asserted below before drawing.
  conv_layer_shapes.pdf  -- a whole layer: a 3x32x32 input, 16 filters of 3x3x3, a 16x32x32
                            output (same padding); filter 1 -> map 1 highlighted. 448 params.

Both are drawn at their size on the slide (5.5 in wide), so font sizes are slide sizes (8-10 pt).

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/multichannel_conv.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"
GREEN, GRAY = "#008C46", "#555555"
CH_TINT = {"R": "#f6c9cc", "G": "#cde8d6", "B": "#cdd6ee"}
CH_EDGE = {"R": ARM_RED, "G": GREEN, "B": ARM_BLUE}

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

# patch values and kernel slices (R and G as on the old slide, B added)
PATCH = {"R": np.array([[1, 0], [2, 1]]), "G": np.array([[0, 3], [1, 1]]),
         "B": np.array([[2, 1], [0, 1]])}
KERN = {"R": np.array([[1, 0], [0, 1]]), "G": np.array([[2, 1], [0, 0]]),
        "B": np.array([[0, 1], [1, 0]])}
BIAS = 1


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_multichannel_conv")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "multichannel_conv.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def grid(ax, x, y, vals, fc, ec, s=0.42, fs=9, lw=1.0):
    """Draw a 2x2 grid of numbers with its top-left corner at (x, y)."""
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.add_patch(Rectangle((x + j * s, y - (i + 1) * s), s, s, fc=fc, ec=ec, lw=lw))
            ax.text(x + (j + 0.5) * s, y - (i + 0.5) * s, f"{vals[i, j]:g}",
                    ha="center", va="center", fontsize=fs)


def multichannel(log):
    sums = {c: int((PATCH[c] * KERN[c]).sum()) for c in "RGB"}
    total = sum(sums.values()) + BIAS
    if (sums["R"], sums["G"], sums["B"], total) != (2, 3, 1, 7):
        raise ValueError(f"worked numbers changed: {sums}, total {total}")
    log.info(f"per-channel sums {sums}, bias {BIAS}, output {total}")

    fig, ax = plt.subplots(figsize=(5.5, 1.85))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 11.0); ax.set_ylim(0, 3.7); ax.set_aspect("equal"); ax.axis("off")

    ax.text(1.05, 3.55, "patch", ha="center", va="top", fontsize=8.5)
    ax.text(3.15, 3.55, "filter slice", ha="center", va="top", fontsize=8.5)
    ax.text(5.55, 3.55, "dot product", ha="center", va="top", fontsize=8.5)
    row_y = [3.1, 2.05, 1.0]
    for c, y in zip("RGB", row_y):
        ax.text(0.15, y - 0.42, c, ha="center", va="center", fontsize=10, fontweight="bold",
                color=CH_EDGE[c])
        grid(ax, 0.63, y, PATCH[c], CH_TINT[c], CH_EDGE[c])
        ax.text(2.0, y - 0.42, r"$\odot$", ha="center", va="center", fontsize=12)
        grid(ax, 2.73, y, KERN[c], "#fdf3d6", "#caa53a")
        ax.text(3.95, y - 0.42, "=", ha="center", va="center", fontsize=11)
        p, k = PATCH[c].ravel(), KERN[c].ravel()
        expr = " + ".join(f"{a}·{b}" for a, b in zip(p, k))
        ax.text(4.25, y - 0.42, f"{expr} = {sums[c]}", ha="left", va="center", fontsize=8.5,
                color=CH_EDGE[c])

    # bracket collecting the three rows
    bx = 7.75
    ax.plot([bx, bx + 0.15, bx + 0.15, bx], [3.0, 3.0, 0.2, 0.2], color=GRAY, lw=1.2)
    ax.annotate("", xy=(8.55, 1.6), xytext=(bx + 0.15, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2))
    ax.text(9.65, 2.55, f"{sums['R']} + {sums['G']} + {sums['B']}", ha="center", fontsize=9)
    ax.text(9.65, 2.15, f"+ bias {BIAS}", ha="center", fontsize=9)
    ax.add_patch(Rectangle((9.25, 0.95), 0.8, 0.8, fc="#e3f1e8", ec=GREEN, lw=2.0))
    ax.text(9.65, 1.35, f"{total}", ha="center", va="center", fontsize=13, fontweight="bold",
            color=GREEN)
    ax.text(9.65, 0.7, "one number", ha="center", va="top", fontsize=8.5, color=GREEN)
    ax.text(9.65, 0.35, "in one map", ha="center", va="top", fontsize=8.5, color=GREEN)

    out = FIG_DIR / "multichannel_conv.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")


def stack(ax, x, y, w, n_draw, step, colors, ec="#333", lw=0.8):
    """Draw n_draw squares of side w, each shifted by (step, -step); return the front square's
    lower-left corner."""
    for k in range(n_draw):
        ax.add_patch(Rectangle((x + k * step, y - k * step), w, w,
                               fc=colors[k % len(colors)], ec=ec, lw=lw))
    return x + (n_draw - 1) * step, y - (n_draw - 1) * step


def layer_shapes(log):
    c_in, c_out, k = 3, 16, 3
    params = k * k * c_in * c_out + c_out
    if params != 448:
        raise ValueError(f"expected 448 parameters, got {params}")
    log.info(f"layer {c_in}x32x32 -> {c_out} filters {c_in}x{k}x{k} -> {c_out}x32x32, {params} params")

    fig, ax = plt.subplots(figsize=(5.5, 1.95))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 11.0); ax.set_ylim(0, 3.9); ax.set_aspect("equal"); ax.axis("off")

    # input: 3 channel slices
    fx, fy = stack(ax, 0.3, 1.25, 1.9, 3, 0.22, [CH_TINT["R"], CH_TINT["G"], CH_TINT["B"]])
    ax.text(1.5, 0.3, "input\n3 × 32 × 32", ha="center", va="center", fontsize=8.5)

    # filter bank: 4 small 3-slice cubes drawn, "..." for the rest
    xs = [3.55, 4.55, 5.55]
    for n, x in enumerate(xs):
        hot = (n == 0)
        stack(ax, x, 2.55, 0.62, 3, 0.09,
              ["#fbe3c4" if hot else "#fdf3d6"], ec=ARM_ORANGE if hot else "#caa53a",
              lw=1.6 if hot else 0.8)
    ax.text(6.55, 2.75, "...", ha="center", va="center", fontsize=12)
    ax.text(5.1, 3.55, "16 filters, each 3 × 3 × 3", ha="center", fontsize=8.5)
    ax.text(5.1, 1.75, "(27 weights + 1 bias each)", ha="center", fontsize=8, color=GRAY)

    # output: 16 maps (draw 8 for readability)
    ox, oy = stack(ax, 7.6, 2.05, 1.75, 8, 0.12, ["#e3f1e8"], ec="#6a9a7c", lw=0.7)
    ax.add_patch(Rectangle((7.6, 2.05), 1.75, 1.75, fc="#fbe3c4", ec=ARM_ORANGE, lw=1.8))
    ax.text(9.0, 0.3, "output\n16 × 32 × 32", ha="center", va="center", fontsize=8.5)

    # arrows: input -> filter 1 -> map 1
    ax.add_patch(FancyArrowPatch((2.75, 2.6), (3.45, 2.6), arrowstyle="-|>",
                                 mutation_scale=9, color=GRAY, lw=1.0))
    ax.add_patch(FancyArrowPatch((4.25, 2.95), (7.5, 3.35), arrowstyle="-|>",
                                 mutation_scale=9, color=ARM_ORANGE, lw=1.3,
                                 connectionstyle="arc3,rad=-0.15"))
    ax.text(5.1, 1.15, f"params = 3·3·3·16 + 16 = {params}", ha="center", fontsize=9,
            fontweight="bold")

    out = FIG_DIR / "conv_layer_shapes.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")


def main():
    t0 = time.perf_counter()
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)
    multichannel(log)
    layer_shapes(log)
    log.info(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
