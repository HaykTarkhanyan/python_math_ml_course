"""Segmentation's dilemma, for the L19 deck (the frame before transposed and dilated conv).

Generates into ml/12_cnn/fig/:
  seg_dilemma.pdf -- the same fruit mask twice on the belt photo: kept on a 7-row grid (the
                     resolution a classification CNN ends with), then at full resolution (what
                     a per-pixel answer needs).

The mask is task_zoo.py's hand-drawn fruit ellipses (no model ran), imported so both figures
always agree. The coarse version: average the mask over ~183 px blocks (7 rows x 10 columns),
threshold at 0.5, blow it back up with nearest neighbour. The overlay is baked into the RGB
pixels (array alpha on imshow is ignored in PDF output, see task_zoo.py).

Run (~5 s on CPU, guess):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/seg_dilemma.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task_zoo import FRUIT_ELLIPSES, PHOTO

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
ROWS, COLS = 7, 10            # a 7x7-style final grid, stretched to the photo's 3:2 shape


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("seg_dilemma")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "seg_dilemma.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def fruit_mask(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.zeros((h, w), bool)
    for cx, cy, rx, ry in FRUIT_ELLIPSES:
        m |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    return m


def coarse(mask):
    """Average over ROWS x COLS blocks, threshold at 0.5, nearest-neighbour back to full size."""
    h, w = mask.shape
    rb, cb = np.linspace(0, h, ROWS + 1).astype(int), np.linspace(0, w, COLS + 1).astype(int)
    out = np.zeros_like(mask)
    for i in range(ROWS):
        for j in range(COLS):
            block = mask[rb[i]:rb[i + 1], cb[j]:cb[j + 1]]
            out[rb[i]:rb[i + 1], cb[j]:cb[j + 1]] = block.mean() >= 0.5
    return out, rb, cb


def overlay(img, mask):
    """Fruit pixels tinted orange, background darkened blue - baked into RGB."""
    base = img.astype(float) / 255
    fruit = np.array([0.95, 0.66, 0.0])
    back = np.array([0.0, 0.2, 0.63])
    out = np.where(mask[..., None], 0.45 * base + 0.55 * fruit, 0.55 * base + 0.45 * back)
    return np.clip(out, 0, 1)


def main():
    log = setup_logging()
    t0 = time.perf_counter()
    img = plt.imread(PHOTO)
    h, w = img.shape[:2]
    if (w, h) != (1920, 1280):
        raise ValueError(f"task_zoo's ellipses are placed for 1920x1280, got {w}x{h}")
    fine = fruit_mask(h, w)
    blocky, rb, cb = coarse(fine)
    agree = (fine == blocky).mean()
    log.info(f"mask {fine.mean():.1%} fruit; the {ROWS}x{COLS} version agrees on {agree:.1%} of pixels")

    fig, axes = plt.subplots(1, 2, figsize=(2.62, 1.12))     # its size on the slide (0.48 column)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.8, bottom=0.01, wspace=0.05)
    axes[0].imshow(overlay(img, blocky))
    for y in rb[1:-1]:
        axes[0].axhline(y, color="white", lw=0.3, alpha=0.6)
    for x in cb[1:-1]:
        axes[0].axvline(x, color="white", lw=0.3, alpha=0.6)
    axes[0].set_title(f"{ROWS}-row grid\n(a classifier's view)",
                      fontsize=7, pad=2, linespacing=1.0)
    axes[1].imshow(overlay(img, fine))
    axes[1].set_title("full resolution\n(what we need)", fontsize=7, pad=2,
                      linespacing=1.0)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    out = FIG_DIR / "seg_dilemma.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info(f"saved {out}; done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
