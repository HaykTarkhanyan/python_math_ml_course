"""Real figure for the L16 CNN Foundations deck (Section: kernel variations).

Generates into ml/12_cnn/fig/:
  dilated_conv_demo.pdf  -- one 3x3 kernel shown at dilation d = 1, 2, 3 over a 9x9
                            input. The 9 tapped cells (blue) spread apart as d grows,
                            so the receptive field (orange box) widens to (2d+1)x(2d+1)
                            -- 3x3, 5x5, 7x7 -- while the kernel keeps its SAME 9 weights.
                            Dilated (atrous) convolution: wider view, no extra params,
                            no lost resolution. Revisited for segmentation in L19.
  dilated_1d.pdf         -- (added 2026-10-01, instructor: "explain dilation better") the
                            mechanics in 1D, by hand: kernel (1, 2, 1) on x = (3,1,4,1,5,9,2,6,5)
                            at d = 1 (taps x1,x2,x3 -> y1 = 9, 7 outputs) and d = 2 (taps
                            x1,x3,x5 -> y1 = 16, 5 outputs). Every output is computed and the
                            first one asserted before drawing.

2026-10-01: both figures are drawn at their size on the slide (5.2-5.5 in wide), so font sizes are
slide sizes (7.5-9.5 pt; the old figure rendered at 5.2 pt). The suptitle moved to the slide.

Run with the project venv:
    ./ma/Scripts/python.exe ml/12_cnn/py_src/dilated_conv_demo.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SEED = 509
BLUE, ORANGE, GREEN = "#0033A0", "#F2A800", "#008C46"
N = 9              # input grid is NxN
CENTER = N // 2    # kernel is centered here (4,4)

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_dilated_conv_demo")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "dilated_conv_demo.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def panel(ax, d):
    # base grid: row i drawn top-down so it reads like a matrix
    for i in range(N):
        for j in range(N):
            ax.add_patch(Rectangle((j, N - 1 - i), 1, 1, ec="#bbb", fc="white", lw=0.5))
    # the 9 tapped cells for a 3x3 kernel at dilation d
    taps = [(CENTER + a * d, CENTER + b * d) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    for (i, j) in taps:
        fc = GREEN if (i, j) == (CENTER, CENTER) else BLUE
        ax.add_patch(Rectangle((j, N - 1 - i), 1, 1, ec="#222", fc=fc, lw=0.8, alpha=0.9))
    # receptive-field box: (2d+1) x (2d+1) centered on the middle cell
    lo = CENTER - d
    k_eff = 2 * d + 1
    ax.add_patch(Rectangle((lo, N - 1 - (CENTER + d)), k_eff, k_eff,
                           ec=ORANGE, fc="none", lw=2.0))
    ax.set_title(f"d = {d}: sees {k_eff}x{k_eff}", fontsize=9, pad=3)
    ax.set_xlim(-0.3, N + 0.3); ax.set_ylim(-0.3, N + 0.3)
    ax.set_aspect("equal"); ax.axis("off")


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(5.2, 1.9))
    for ax, d in zip(axes, (1, 2, 3)):
        panel(ax, d)
        if 2 * d + 1 > N:
            raise ValueError(f"dilation d={d} overflows the {N}x{N} grid")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.08)
    out = FIG_DIR / "dilated_conv_demo.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")
    dilated_1d(log)


X1D = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])
W1D = np.array([1, 2, 1])


def dilated_out(x, w, d):
    """y_i = sum_k w_k x_{i + k d}, valid positions only."""
    span = (len(w) - 1) * d + 1
    return np.array([sum(w[k] * x[i + k * d] for k in range(len(w)))
                     for i in range(len(x) - span + 1)])


def dilated_1d(log):
    fig, ax = plt.subplots(figsize=(5.5, 1.7))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-0.3, 20.3); ax.set_ylim(-0.25, 6.1); ax.set_aspect("equal"); ax.axis("off")
    for x0, d in ((0.0, 1), (10.6, 2)):
        y = dilated_out(X1D, W1D, d)
        taps = [k * d for k in range(len(W1D))]
        expect = {1: 9, 2: 16}[d]
        if y[0] != expect:
            raise ValueError(f"d={d}: y1 = {y[0]}, expected {expect}")
        log.info(f"d={d}: taps x{[t + 1 for t in taps]}, outputs {y.tolist()}")
        # input row
        for j, v in enumerate(X1D):
            hot = j in taps
            ax.add_patch(Rectangle((x0 + j, 3.3), 1, 1, fc="#cdd6ee" if hot else "white",
                                   ec=BLUE if hot else "#999", lw=1.4 if hot else 0.6))
            ax.text(x0 + j + 0.5, 3.8, str(v), ha="center", va="center", fontsize=9)
        ax.text(x0 + 4.5, 4.45, f"kernel (1, 2, 1), taps {d} apart", ha="center", va="bottom",
                fontsize=8, color=BLUE)
        # output row: y_i sits under its window's first tap
        for i, v in enumerate(y):
            first = (i == 0)
            ax.add_patch(Rectangle((x0 + i, 0.9), 1, 1, fc="#e3f1e8" if first else "white",
                                   ec=GREEN if first else "#999", lw=1.6 if first else 0.6))
            ax.text(x0 + i + 0.5, 1.4, str(v), ha="center", va="center", fontsize=9,
                    fontweight="bold" if first else "normal")
        for t in taps:
            ax.plot([x0 + t + 0.5, x0 + 0.5], [3.3, 1.9], color=BLUE, lw=0.9)
        ax.text(x0 + 4.5, 5.25, f"d = {d}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold")
        terms = " + ".join(f"{W1D[k]}·{X1D[t]}" for k, t in enumerate(taps))
        ax.text(x0 + 4.5, 0.65, f"$y_1$ = {terms} = {y[0]}   ({len(y)} outputs)", ha="center",
                va="top", fontsize=8, color=GREEN)
    out = FIG_DIR / "dilated_1d.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
