"""Real figures for the L16 CNN Foundations deck (Section 2, convolution animation).

Generates into ml/12_cnn/fig/:
  conv_anim_0.pdf .. conv_anim_8.pdf  -- nine flip-book frames: a 3x3 Gaussian-blur kernel
                     (weights 1,2,4) slides over a 5x5 grayscale patch; each output cell is
                     the weighted sum of the window under it, so the patch gets smoothed.
                     Input and output cells are shaded by value (light = high). Shown on one
                     beamer frame via \\only<n>, advanced by clicking.

All nine frames share identical geometry (fixed axis limits, no tight bbox) so the picture
does not jump between clicks.

2026-10-01: redrawn at its slide size (4.87 x 2.45 in, embedded at width=4.87in, so font sizes
are the sizes on the slide: 8-12 pt; they were 4.5 pt before). The sliding window is outlined in
orange - the old dark-blue outline vanished against the dark cells (instructor request). The
in-figure title was dropped; the frame title already says it.

Run with the project venv:
    ./ma/Scripts/python.exe ml/12_cnn/py_src/conv_anim.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SEED = 509
WINDOW, GREEN = "#E8590C", "#008C46"   # window outline: orange reads on black AND white cells

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
    fh = logging.FileHandler(LOGS_DIR / "conv_anim.log", encoding="utf-8"); fh.setFormatter(fmt)
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


def cell(ax, x, y, s, val=None, fc="white", ec="#555", lw=1.0, fs=10, tc="black"):
    ax.add_patch(Rectangle((x, y), s, s, ec=ec, fc=fc, lw=lw))
    if val is not None:
        ax.text(x + s / 2, y + s / 2, f"{val:g}", ha="center", va="center",
                fontsize=fs, color=tc)


def draw_frame(step, OUT, log):
    r, c = divmod(step, 3)
    imin, imax = INPUT.min(), INPUT.max()

    fig, ax = plt.subplots(figsize=(4.87, 2.45))   # = its size on the slide
    ax.set_position([0, 0, 1, 1])

    ax.text(7.75, 7.6, f"step {step + 1} of 9", ha="center", fontsize=8.5, color="#555")

    # input 5x5 (top-left cell at x=0.5, y=7.5), shaded by value
    for i in range(5):
        for j in range(5):
            fc, tc = shade(INPUT[i, j], imin, imax)
            in_win = (r <= i < r + 3) and (c <= j < c + 3)
            cell(ax, 0.5 + j, 7.5 - (i + 1), 1, INPUT[i, j], fc=fc, tc=tc,
                 ec=WINDOW if in_win else "#888", lw=1.4 if in_win else 0.6)
    ax.add_patch(Rectangle((0.5 + c, 7.5 - (r + 3)), 3, 3, ec=WINDOW, fc="none", lw=3.2))
    ax.text(3.0, 7.65, "input (5x5, shaded by value)", ha="center", fontsize=8.5)

    # kernel 3x3 (Gaussian blur)
    kx, ky, ks = 6.9, 4.3, 0.62
    for i in range(3):
        for j in range(3):
            cell(ax, kx + j * ks, ky + (2 - i) * ks, ks, KERNEL[i, j],
                 fc="#fdf3d6", ec="#caa53a", fs=8.5, lw=0.6)
    ax.text(kx + 1.5 * ks, ky + 3 * ks + 0.2, "Gaussian kernel, / 16", ha="center",
            fontsize=8)

    # computation
    val = OUT[r, c]
    ax.text(6.6, 3.25, f"output({r},{c}) =", fontsize=9.5, color=GREEN)
    ax.text(6.6, 2.6, "weighted average of", fontsize=8, color="#444")
    ax.text(6.6, 2.1, "the 9 window values", fontsize=8, color="#444")
    ax.text(6.6, 1.2, f"= {val:g}", fontsize=12, color=GREEN, fontweight="bold")

    # output 3x3 (top-left at x=10.9, y=6.6), shaded by value, filled through this step
    ox, oy = 10.9, 6.6
    for i in range(3):
        for j in range(3):
            filled = i * 3 + j <= step
            is_cur = (i, j) == (r, c)
            if filled:
                fc, tc = shade(OUT[i, j], imin, imax)
                cell(ax, ox + j, oy - (i + 1), 1, OUT[i, j], fc=fc, tc=tc,
                     ec=GREEN if is_cur else "#888", lw=2.6 if is_cur else 0.6)
            else:
                cell(ax, ox + j, oy - (i + 1), 1, None, fc="#f4f4f4", ec="#ccc")
    ax.text(ox + 1.5, 6.75, "output (smoothed)", ha="center", fontsize=8.5)

    ax.set_xlim(0.3, 14.4); ax.set_ylim(0.9, 8.0)
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
