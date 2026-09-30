"""Real figure for the L16 CNN Foundations deck (Section: kernel variations).

Generates into ml/ch6_cnn/fig/:
  transposed_conv_demo.pdf  -- transposed convolution as the "opposite" of a normal
                               conv: instead of shrinking the map, it GROWS it. Each
                               input cell stamps a copy of the 3x3 kernel onto the
                               output; where stamps overlap, the contributions add.
                               A 2x2 input, 3x3 kernel, stride 2 -> a 5x5 output
                               (upsampling). Used in U-Net decoders and GANs (L19).

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/transposed_conv_demo.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SEED = 509
# one distinct color per input cell
COLORS = {(0, 0): "#D90012", (0, 1): "#0033A0", (1, 0): "#F2A800", (1, 1): "#008C46"}
K = 3        # kernel size
STRIDE = 2
IN = 2       # input is IN x IN
OUT = (IN - 1) * STRIDE + K   # = 5

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_transposed_conv_demo")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "transposed_conv_demo.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def draw_grid_edges(ax, x0, y0, n, ec="#888", lw=1.2):
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + j, y0 - 1 - i), 1, 1, ec=ec, fc="none", lw=lw))


def input_panel(ax):
    ax.set_title("input  2x2", fontsize=13, pad=8)
    for (i, j), col in COLORS.items():
        ax.add_patch(Rectangle((j, IN - 1 - i), 1, 1, ec="#222", fc=col, lw=1.5))
    ax.set_xlim(-0.4, IN + 0.4); ax.set_ylim(-0.4, IN + 0.4)
    ax.set_aspect("equal"); ax.axis("off")


def output_panel(ax):
    ax.set_title("output  5x5   (each cell stamps a 3x3 kernel; overlaps add)",
                 fontsize=13, pad=8)
    overlap = np.zeros((OUT, OUT), dtype=int)
    # each input cell (i,j) stamps its KxK kernel at output rows/cols starting i*stride
    for (i, j), col in COLORS.items():
        r0, c0 = i * STRIDE, j * STRIDE
        for a in range(K):
            for b in range(K):
                r, c = r0 + a, c0 + b
                ax.add_patch(Rectangle((c, OUT - 1 - r), 1, 1, ec="none", fc=col, alpha=0.42))
                overlap[r, c] += 1
    # crisp grid lines over the translucent stamps
    draw_grid_edges(ax, 0, OUT, OUT, ec="#666", lw=1.0)
    # mark the cells where two stamps overlap and add
    ys, xs = np.where(overlap >= 2)
    for r, c in zip(ys, xs):
        ax.plot(c + 0.5, OUT - 1 - r + 0.5, "+", color="#222", ms=9, mew=2.0)
    ax.set_xlim(-0.4, OUT + 0.4); ax.set_ylim(-0.4, OUT + 0.4)
    ax.set_aspect("equal"); ax.axis("off")
    return overlap


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                 gridspec_kw={"width_ratios": [1, 2]})
    input_panel(a1)
    overlap = output_panel(a2)
    # arrow between panels
    a1.annotate("", xy=(IN + 1.35, IN / 2), xytext=(IN + 0.2, IN / 2),
                xycoords="data", textcoords="data", annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=2.4))
    a1.text(IN + 0.78, IN / 2 + 0.28, "stride 2", ha="center", fontsize=10, color="#444")

    if overlap.shape != (OUT, OUT):
        raise ValueError("output size mismatch")
    log.info(f"transposed conv {IN}x{IN} -> {OUT}x{OUT}; overlap cells (+): {(overlap >= 2).sum()}")

    fig.suptitle("Transposed convolution: the opposite of a shrinking conv - it upsamples "
                 "(2x2 -> 5x5)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "transposed_conv_demo.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
