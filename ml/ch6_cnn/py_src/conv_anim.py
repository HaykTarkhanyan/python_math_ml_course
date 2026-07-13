"""Real figures for the L16 CNN Foundations deck (Section 2, convolution animation).

Generates into ml/ch6_cnn/fig/:
  conv_anim_0.pdf .. conv_anim_8.pdf  -- nine flip-book frames: a 3x3 Gaussian-blur kernel
                     (weights 1,2,4) slides over a 5x5 grayscale patch; each output cell is
                     the weighted sum of the window under it, so the patch gets smoothed.
                     Input and output cells are shaded by value (light = high). Shown on one
                     beamer frame via \\only<n>, advanced by clicking.

All nine frames share identical geometry (fixed axis limits, no tight bbox) so the picture
does not jump between clicks.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/conv_anim.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SEED = 509
BLUE, GREEN = "#0033A0", "#008C46"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

# a small grayscale patch: a bright blob on a darker background (values 0-9)
INPUT = np.array([[1, 2, 3, 2, 1],
                  [2, 5, 7, 5, 2],
                  [3, 7, 9, 7, 3],
                  [2, 5, 7, 5, 2],
                  [1, 2, 3, 2, 1]], float)
KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float)   # Gaussian blur (unnormalised)


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_conv_anim")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "conv_anim.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def output_full():
    """Weighted average: dot product of window and kernel, divided by the kernel's weight
    sum (16). Output stays on the input's 0-9 scale, so it reads as a smoothed copy."""
    out = np.zeros((3, 3))
    wsum = KERNEL.sum()
    for r in range(3):
        for c in range(3):
            out[r, c] = (INPUT[r:r + 3, c:c + 3] * KERNEL).sum() / wsum
    return np.round(out, 1)


def shade(v, vmin, vmax):
    t = float(np.clip((v - vmin) / (vmax - vmin + 1e-9), 0, 1))
    return (t, t, t), ("white" if t < 0.5 else "black")


def cell(ax, x, y, s, val=None, fc="white", ec="#555", lw=1.0, fs=12, tc="black"):
    ax.add_patch(Rectangle((x, y), s, s, ec=ec, fc=fc, lw=lw))
    if val is not None:
        ax.text(x + s / 2, y + s / 2, f"{val:g}", ha="center", va="center",
                fontsize=fs, color=tc)


def draw_frame(step, OUT, log):
    r, c = divmod(step, 3)
    imin, imax = INPUT.min(), INPUT.max()

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.set_position([0, 0, 1, 1])

    ax.text(8.0, 8.55, "Convolution: slide, multiply, sum", ha="center", fontsize=14)
    ax.text(8.0, 8.05, f"step {step + 1} of 9", ha="center", fontsize=11, color="#555")

    # input 5x5 (top-left cell at x=0.5, y=7.5), shaded by value
    for i in range(5):
        for j in range(5):
            fc, tc = shade(INPUT[i, j], imin, imax)
            in_win = (r <= i < r + 3) and (c <= j < c + 3)
            cell(ax, 0.5 + j, 7.5 - (i + 1), 1, INPUT[i, j], fc=fc, tc=tc,
                 ec=BLUE if in_win else "#888", lw=1.8 if in_win else 0.8)
    ax.add_patch(Rectangle((0.5 + c, 7.5 - (r + 3)), 3, 3, ec=BLUE, fc="none", lw=3))
    ax.text(3.0, 7.75, "input  (5x5, shaded by value)", ha="center", fontsize=11)

    # kernel 3x3 (Gaussian blur)
    kx, ky, ks = 6.6, 4.3, 0.58
    for i in range(3):
        for j in range(3):
            cell(ax, kx + j * ks, ky + (2 - i) * ks, ks, KERNEL[i, j],
                 fc="#fdf3d6", ec="#caa53a", fs=10)
    ax.text(kx + 1.5 * ks, ky + 3 * ks + 0.15, "kernel: Gaussian blur / 16", ha="center",
            fontsize=10)

    # computation
    val = OUT[r, c]
    ax.text(6.6, 3.2, f"output({r},{c}) =", fontsize=12, color=GREEN)
    ax.text(6.6, 2.65, "weighted average of", fontsize=10, color="#444")
    ax.text(6.6, 2.25, "the 9 window values", fontsize=10, color="#444")
    ax.text(6.6, 1.4, f"= {val:g}", fontsize=15, color=GREEN, fontweight="bold")

    # output 3x3 (top-left at x=10.9, y=6.6), shaded by value, filled through this step
    ox, oy = 10.9, 6.6
    for i in range(3):
        for j in range(3):
            filled = i * 3 + j <= step
            is_cur = (i, j) == (r, c)
            if filled:
                fc, tc = shade(OUT[i, j], imin, imax)
                cell(ax, ox + j, oy - (i + 1), 1, OUT[i, j], fc=fc, tc=tc,
                     ec=GREEN if is_cur else "#888", lw=2.6 if is_cur else 0.8)
            else:
                cell(ax, ox + j, oy - (i + 1), 1, None, fc="#f4f4f4", ec="#ccc")
    ax.text(ox + 1.5, 6.85, "output  (smoothed)", ha="center", fontsize=11)

    ax.set_xlim(0, 15.5); ax.set_ylim(0.4, 9.0)
    ax.set_aspect("equal"); ax.axis("off")
    out_path = FIG_DIR / f"conv_anim_{step}.pdf"
    fig.savefig(out_path); plt.close(fig)          # no tight bbox -> identical geometry
    log.info(f"saved {out_path}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    OUT = output_full()
    log.info(f"output grid:\n{OUT}")
    for step in range(9):
        draw_frame(step, OUT, log)


if __name__ == "__main__":
    main()
