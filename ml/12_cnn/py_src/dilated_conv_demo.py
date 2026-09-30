"""Real figure for the L16 CNN Foundations deck (Section: kernel variations).

Generates into ml/ch6_cnn/fig/:
  dilated_conv_demo.pdf  -- one 3x3 kernel shown at dilation d = 1, 2, 3 over a 9x9
                            input. The 9 tapped cells (blue) spread apart as d grows,
                            so the receptive field (orange box) widens to (2d+1)x(2d+1)
                            -- 3x3, 5x5, 7x7 -- while the kernel keeps its SAME 9 weights.
                            Dilated (atrous) convolution: wider view, no extra params,
                            no lost resolution. Revisited for segmentation in L19.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/dilated_conv_demo.py
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
    fh = logging.FileHandler(LOGS_DIR / "dilated_conv_demo.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def panel(ax, d):
    # base grid: row i drawn top-down so it reads like a matrix
    for i in range(N):
        for j in range(N):
            ax.add_patch(Rectangle((j, N - 1 - i), 1, 1, ec="#bbb", fc="white", lw=1.0))
    # the 9 tapped cells for a 3x3 kernel at dilation d
    taps = [(CENTER + a * d, CENTER + b * d) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    for (i, j) in taps:
        fc = GREEN if (i, j) == (CENTER, CENTER) else BLUE
        ax.add_patch(Rectangle((j, N - 1 - i), 1, 1, ec="#222", fc=fc, lw=1.4, alpha=0.9))
    # receptive-field box: (2d+1) x (2d+1) centered on the middle cell
    lo = CENTER - d
    k_eff = 2 * d + 1
    ax.add_patch(Rectangle((lo, N - 1 - (CENTER + d)), k_eff, k_eff,
                           ec=ORANGE, fc="none", lw=3.0))
    ax.set_title(f"dilation d = {d}\n9 weights, sees {k_eff}x{k_eff}",
                 fontsize=13, pad=8)
    ax.set_xlim(-0.3, N + 0.3); ax.set_ylim(-0.3, N + 0.3)
    ax.set_aspect("equal"); ax.axis("off")


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4))
    for ax, d in zip(axes, (1, 2, 3)):
        panel(ax, d)
        if 2 * d + 1 > N:
            raise ValueError(f"dilation d={d} overflows the {N}x{N} grid")
    fig.suptitle("Dilated convolution: gaps in the kernel widen the receptive field "
                 "for free (green = center tap)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "dilated_conv_demo.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
