"""Figure for the L16 CNN Foundations deck (Section 3, regularization by architecture).

Generates into ml/12_cnn/fig/:
  conv_as_matrix.pdf  -- the same map y = W x (8 inputs -> 6 outputs) twice: a dense layer, where
                         all 48 entries of W are free, and a 1D convolution with kernel
                         (w1, w2, w3), where W is banded: 30 entries forced to 0, the other 18
                         tied to just 3 values. Counts asserted before drawing.

Drawn at its size on the slide (5.5 x 2.1 in), so font sizes are slide sizes (8-10 pt).

Run with the project venv (~2 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/conv_as_matrix.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"
TAP_COLORS = [ARM_BLUE, ARM_ORANGE, "#008C46"]          # w1, w2, w3
N_IN, K = 8, 3
N_OUT = N_IN - K + 1                                     # valid convolution: 6 outputs

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_conv_as_matrix")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "conv_as_matrix.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def conv_tap_index() -> np.ndarray:
    """tap[i, j] = which kernel weight (0, 1, 2) multiplies x_j in output y_i, or -1 for a
    forced zero. Row i of a valid 1D convolution uses x_i, x_{i+1}, x_{i+2}."""
    tap = -np.ones((N_OUT, N_IN), dtype=int)
    for i in range(N_OUT):
        for k in range(K):
            tap[i, i + k] = k
    return tap


def draw_matrix(ax, x0, title, colors, labels, sub):
    s = 0.36
    for i in range(N_OUT):
        for j in range(N_IN):
            ax.add_patch(Rectangle((x0 + j * s, -(i + 1) * s), s, s, fc=colors[i][j],
                                   ec="#999", lw=0.5))
            if labels[i][j]:
                ax.text(x0 + (j + 0.5) * s, -(i + 0.5) * s, labels[i][j], ha="center",
                        va="center", fontsize=7.5,
                        color="white" if labels[i][j].startswith("w") else "#bbb")
    for j in range(N_IN):
        ax.text(x0 + (j + 0.5) * s, 0.12, f"$x_{j + 1}$", ha="center", va="bottom", fontsize=8)
    for i in range(N_OUT):
        ax.text(x0 - 0.08, -(i + 0.5) * s, f"$y_{i + 1}$", ha="right", va="center", fontsize=8)
    ax.text(x0 + N_IN * s / 2, 0.55, title, ha="center", va="bottom", fontsize=9.5,
            fontweight="bold")
    ax.text(x0 + N_IN * s / 2, -N_OUT * s - 0.12, sub, ha="center", va="top", fontsize=9)


def main():
    t0 = time.perf_counter()
    log = setup_logging()
    rng = np.random.default_rng(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    tap = conv_tap_index()
    n_zero, n_tied, n_free = int((tap < 0).sum()), int((tap >= 0).sum()), K
    if (N_OUT * N_IN, n_zero, n_tied) != (48, 30, 18):
        raise ValueError(f"counts changed: {N_OUT * N_IN} total, {n_zero} zero, {n_tied} tied")
    log.info(f"dense: {N_OUT * N_IN} free weights | conv: {n_zero} forced to 0, "
             f"{n_tied} entries tied to {n_free} weights")

    # dense: every entry its own random shade (48 different weights)
    cmap = plt.get_cmap("RdBu")
    dense_cols = [[cmap(0.2 + 0.6 * rng.random()) for _ in range(N_IN)] for _ in range(N_OUT)]
    dense_lab = [["" for _ in range(N_IN)] for _ in range(N_OUT)]
    conv_cols = [[TAP_COLORS[tap[i, j]] if tap[i, j] >= 0 else "white" for j in range(N_IN)]
                 for i in range(N_OUT)]
    conv_lab = [[f"w{tap[i, j] + 1}" if tap[i, j] >= 0 else "0" for j in range(N_IN)]
                for i in range(N_OUT)]

    fig, ax = plt.subplots(figsize=(5.5, 2.1))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-0.55, 6.85); ax.set_ylim(-2.55, 0.28 + 0.55 + 0.27)
    ax.set_aspect("equal"); ax.axis("off")
    draw_matrix(ax, 0.0, "dense layer", dense_cols, dense_lab,
                f"{N_OUT * N_IN} free weights")
    draw_matrix(ax, 3.85, f"1D convolution, kernel ($w_1, w_2, w_3$)", conv_cols, conv_lab,
                f"{n_zero} forced to 0, the rest tied: {n_free} free weights")

    out = FIG_DIR / "conv_as_matrix.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")
    log.info(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
