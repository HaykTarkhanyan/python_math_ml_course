"""Real figure for the L16 CNN Foundations deck (Section: kernel variations).

Generates into ml/12_cnn/fig/:
  transposed_conv_demo.pdf  -- transposed convolution as the "opposite" of a normal
                               conv: instead of shrinking the map, it GROWS it. Each
                               input cell stamps a copy of the 3x3 kernel onto the
                               output; where stamps overlap, the contributions add.
                               A 2x2 input, 3x3 kernel, stride 2 -> a 5x5 output
                               (upsampling). Used in U-Net decoders and GANs (L19).
  transposed_steps.pdf      -- (added 2026-10-01, instructor: "not clear how it's done") 1D by
                               hand: x = (1, 2), kernel (1, 1, 1), stride 2. Left: each input
                               value stamps value x kernel, the stamps land 2 apart, overlaps
                               add -> (1, 1, 3, 2, 2). Right: the same numbers from an ordinary
                               stride-1 convolution after spreading the input with zeros and
                               padding it. Both computed and asserted equal before drawing.

2026-10-01: both figures are drawn at their size on the slide (3.05 and 5.5 in wide), so font sizes
are slide sizes (7.5-9.5 pt; the old figure rendered at 3.4 pt). The suptitle moved to the slide.

Run with the project venv:
    ./ma/Scripts/python.exe ml/12_cnn/py_src/transposed_conv_demo.py
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
    fh = logging.FileHandler(LOGS_DIR / "transposed_conv_demo.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def draw_grid_edges(ax, x0, y0, n, ec="#888", lw=1.2):
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + j, y0 - 1 - i), 1, 1, ec=ec, fc="none", lw=lw))


def stamps_2d(ax, x0, top):
    """Each input cell (i, j) stamps a KxK copy of the kernel at output rows/cols starting
    i*STRIDE, j*STRIDE; returns how many stamps land on each output cell."""
    overlap = np.zeros((OUT, OUT), dtype=int)
    for (i, j), col in COLORS.items():
        r0, c0 = i * STRIDE, j * STRIDE
        for a in range(K):
            for b in range(K):
                r, c = r0 + a, c0 + b
                ax.add_patch(Rectangle((x0 + c, top - 1 - r), 1, 1, ec="none", fc=col,
                                       alpha=0.42))
                overlap[r, c] += 1
    draw_grid_edges(ax, x0, top, OUT, ec="#666", lw=0.6)    # crisp lines over the stamps
    ys, xs = np.where(overlap >= 2)
    for r, c in zip(ys, xs):
        ax.plot(x0 + c + 0.5, top - 1 - r + 0.5, "+", color="#222", ms=6, mew=1.4)
    return overlap


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    # one axes, drawn at its slide size (3.05 x 1.95 in)
    fig, ax = plt.subplots(figsize=(3.05, 1.95))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-0.3, 11.2); ax.set_ylim(-0.45, 7.0); ax.set_aspect("equal"); ax.axis("off")
    in_x0, in_top = 0.4, 3.5
    for (i, j), col in COLORS.items():
        ax.add_patch(Rectangle((in_x0 + j, in_top - 1 - i), 1, 1, ec="#222", fc=col, lw=1.0))
    ax.text(in_x0 + 1, in_top + 0.3, "input 2x2", ha="center", va="bottom", fontsize=9)
    ax.annotate("", xy=(4.95, 2.5), xytext=(2.85, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.3))
    ax.text(3.9, 2.8, "stride 2", ha="center", va="bottom", fontsize=8.5, color="#444")
    overlap = stamps_2d(ax, 5.6, 5.75)
    ax.text(5.6 + OUT / 2, 6.0, "output 5x5", ha="center", va="bottom", fontsize=9)
    ax.text(5.6 + OUT / 2, 0.6, "+ = stamps overlap, add", ha="center", va="top",
            fontsize=8, color="#444")
    if overlap.shape != (OUT, OUT) or (overlap >= 2).sum() != 9:
        raise ValueError(f"expected 9 overlap cells, got {(overlap >= 2).sum()}")
    log.info(f"transposed conv {IN}x{IN} -> {OUT}x{OUT}; overlap cells (+): {(overlap >= 2).sum()}")
    out = FIG_DIR / "transposed_conv_demo.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")
    steps_1d(log)


X_T = np.array([1, 2])
K_T = np.array([1, 1, 1])
S_T = 2


def stamp_and_add(x, k, s):
    out = np.zeros(s * (len(x) - 1) + len(k), dtype=int)
    for i, v in enumerate(x):
        out[i * s:i * s + len(k)] += v * k
    return out


def zero_insert_conv(x, k, s):
    """Equivalent view: put s-1 zeros between inputs, pad k-1 zeros each side, then an
    ordinary (stride 1) convolution with the flipped kernel."""
    spread = np.zeros(s * (len(x) - 1) + 1, dtype=int)
    spread[::s] = x
    padded = np.pad(spread, len(k) - 1)
    kf = k[::-1]
    return padded, np.array([int((padded[i:i + len(k)] * kf).sum())
                             for i in range(len(padded) - len(k) + 1)])


def row(ax, x0, y, vals, colors=None, edge=None, fs=9, bold=None):
    for j, v in enumerate(vals):
        fc = colors[j] if colors else "white"
        ax.add_patch(Rectangle((x0 + j, y), 1, 1, fc=fc, ec=edge[j] if edge else "#999",
                               lw=1.6 if edge and edge[j] != "#999" else 0.6))
        if v is not None:
            ax.text(x0 + j + 0.5, y + 0.5, str(v), ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold and bold[j] else "normal")


def steps_1d(log):
    """1D, by hand: x = (1, 2), kernel (1, 1, 1), stride 2 -> (1, 1, 3, 2, 2)."""
    y = stamp_and_add(X_T, K_T, S_T)
    padded, y2 = zero_insert_conv(X_T, K_T, S_T)
    if y.tolist() != [1, 1, 3, 2, 2] or y2.tolist() != y.tolist():
        raise ValueError(f"stamp {y.tolist()} vs zero-insert conv {y2.tolist()}")
    log.info(f"1D: x={X_T.tolist()}, k={K_T.tolist()}, s={S_T} -> {y.tolist()} "
             f"(zero-insert view {padded.tolist()} -> {y2.tolist()})")
    RED_T, BLUE_T, PURPLE = "#f6c9cc", "#cdd6ee", "#e2c6ec"

    fig, ax = plt.subplots(figsize=(5.5, 1.95))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-0.2, 18.3); ax.set_ylim(-0.3, 6.25); ax.set_aspect("equal"); ax.axis("off")

    # left: stamp and add
    lx = 4.4
    ax.text(lx + 2.5, 5.85, "stamp and add", ha="center", va="center", fontsize=9.5,
            fontweight="bold")
    labels = ["input", "1 x kernel, at 0", "2 x kernel, at 2", "sum"]
    ys = [4.3, 3.0, 1.7, 0.2]
    for lab, yy in zip(labels, ys):
        ax.text(lx - 0.2, yy + 0.5, lab, ha="right", va="center", fontsize=8.5)
    row(ax, lx, ys[0], [1, 2], colors=[RED_T, BLUE_T])
    row(ax, lx, ys[1], [1, 1, 1, None, None], colors=[RED_T] * 3 + ["white"] * 2)
    row(ax, lx, ys[2], [None, None, 2, 2, 2], colors=["white"] * 2 + [BLUE_T] * 3)
    ax.plot([lx, lx + 5], [1.45, 1.45], color="#444", lw=0.8)
    row(ax, lx, ys[3], y.tolist(), colors=[RED_T, RED_T, PURPLE, BLUE_T, BLUE_T],
        bold=[False, False, True, False, False])

    # right: the same numbers from an ordinary convolution
    rx = 10.9
    ax.text(rx + 3.5, 5.85, "same, as a normal conv", ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    cols = ["white"] * len(padded)
    cols[2], cols[4] = RED_T, BLUE_T
    row(ax, rx, 3.6, padded.tolist(), colors=cols)
    # the window that produces the overlap value 3
    ax.add_patch(Rectangle((rx + 2, 3.6), 3, 1, fc="none", ec="#E8590C", lw=2.0))
    ax.text(rx + 3.5, 4.75, "zeros between and around the input", ha="center", va="center",
            fontsize=8, color="#444")
    ax.text(rx + 3.5, 2.95, "slide kernel (1, 1, 1), stride 1", ha="center", va="center",
            fontsize=8, color="#E8590C")
    row(ax, rx + 1, 1.0, y2.tolist(), colors=[RED_T, RED_T, PURPLE, BLUE_T, BLUE_T],
        bold=[False, False, True, False, False])
    ax.text(rx + 3.5, 0.6, "the same 5 numbers", ha="center", va="top", fontsize=8,
            color="#444")
    out = FIG_DIR / "transposed_steps.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
