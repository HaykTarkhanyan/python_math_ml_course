"""Segment Anything's design, drawn on the belt photo - for the [51] Vision Tasks deck.

Frame "Segment Anything: one model, any object". The point of the drawing is the split that makes
SAM interactive (Kirillov et al. 2023, Fig. 4 and Sec. 3): a HEAVY image encoder runs once per
photo; a LIGHT prompt encoder + mask decoder then answers each prompt (a click, a box, or - in
SAM 3 - a typed concept) "in a web browser, on CPU, in 50ms". The output masks are
ILLUSTRATIONS (task_zoo.py's hand-placed fruit ellipses) - no SAM model runs here.

Generates into ml/12_cnn/fig/:
  sam_diagram.pdf

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/sam_diagram.py
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from task_zoo import FRUIT_ELLIPSES, PHOTO  # noqa: E402

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#008c46"
FS = 8.0

CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
CLICKED = 1                   # FRUIT_ELLIPSES[1]: the centre split-open fruit


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l52_sam_diagram")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "sam_diagram.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def box(ax, x, y, w, h, text, fc, ec, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS,
            linespacing=1.1, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color="black"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color,
                                 lw=0.9, shrinkA=1, shrinkB=1))


def photo_panel(fig, rect, img, masks, click=None):
    a = fig.add_axes(rect)
    a.imshow(img)
    for i in masks:
        cx, cy, rx, ry = FRUIT_ELLIPSES[i]
        a.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fc=ARM_BLUE, alpha=0.55, ec="white",
                            lw=0.8))
    if click is not None:
        a.plot([click[0]], [click[1]], marker="*", color="#39ff14", ms=9, mec="black", mew=0.6)
    a.set_xticks([]); a.set_yticks([])
    return a


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    img = plt.imread(PHOTO)
    h, w = img.shape[:2]
    W, H = 5.45, 2.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    pw = 0.95                                   # photo panel width in inches
    ph = pw * h / w
    cx, cy = FRUIT_ELLIPSES[CLICKED][:2]
    photo_panel(fig, [0.02 / W, (1.3 - ph / 2) / H, pw / W, ph / H], img, [], click=(cx, cy))
    ax.text(0.02 + pw / 2, 1.3 + ph / 2 + 0.05, "photo", ha="center", va="bottom", fontsize=FS)

    box(ax, 1.12, 1.0, 1.08, 0.6, "image encoder\n(big ViT)\nonce per photo", "#dce6f5",
        ARM_BLUE)
    arrow(ax, (0.02 + pw, 1.3), (1.12, 1.3))
    box(ax, 1.12, 0.2, 1.08, 0.5, "prompt\nencoder", "#f3f3f3", "#777777")
    ax.text(0.5, 0.65, "prompts:", ha="center", va="center", fontsize=FS, fontweight="bold")
    ax.text(0.5, 0.36, "a click (star)\na box\na word (SAM 3)", ha="center", va="center",
            fontsize=FS, linespacing=1.2)
    arrow(ax, (1.0, 0.45), (1.15, 0.45))

    box(ax, 2.38, 0.57, 1.0, 0.66, "mask decoder\n(light)\n~50 ms per\nprompt", "#dff2e6",
        GREEN)
    arrow(ax, (2.2, 1.3), (2.38, 1.05))
    arrow(ax, (2.2, 0.45), (2.38, 0.75))

    # outputs: the click gives one mask; the typed concept gives every fruit
    pw2 = 0.82
    ph2 = pw2 * h / w
    photo_panel(fig, [3.62 / W, 1.08 / H, pw2 / W, ph2 / H], img, [CLICKED], click=(cx, cy))
    ax.text(3.62 + pw2 + 0.06, 1.08 + ph2 / 2, "click ->\none mask", ha="left", va="center",
            fontsize=FS, linespacing=1.1)
    photo_panel(fig, [3.62 / W, 0.12 / H, pw2 / W, ph2 / H], img,
                list(range(len(FRUIT_ELLIPSES))))
    ax.text(3.62 + pw2 + 0.06, 0.12 + ph2 / 2, "\"pomegranate\"\n-> every one", ha="left",
            va="center", fontsize=FS, linespacing=1.1)
    arrow(ax, (3.4, 0.95), (3.62, 1.08 + ph2 / 2))
    arrow(ax, (3.4, 0.85), (3.62, 0.12 + ph2 / 2))      # "masks drawn" note is on the slide

    out = FIG_DIR / "sam_diagram.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info(f"saved {out}; done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
