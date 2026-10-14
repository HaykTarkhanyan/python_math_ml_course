"""Diagrams for the L18 Transfer Learning deck ("whose shoulders you stand on", 2026-10-01,
instructor: l18_todo.md item 1 - explain the 2026 backbones better).

Generates into ml/12_cnn/fig/:
  ssl_crops.pdf    -- self-supervised learning (the DINO family's idea): two random crops of the same
                      photo go through the network; training makes their feature vectors match, so
                      the features must describe WHAT is in the photo, not the pixels. No labels.
  clip_matrix.pdf  -- CLIP's training objective on a batch of 4 image-caption pairs: the 4 x 4
                      similarity table, matching pairs (diagonal) pushed up, the 12 mismatches
                      pushed down. A picture of the objective, not a measurement - no numbers shown.

Photos: the chapter's pomegranate (fig/src_pomegranate.jpg) and three public-domain skimage test
images (astronaut, chelsea the cat, coffee). Drawn at slide size (5.5 in wide), fonts 7.5-9 pt.

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/backbone_diagrams.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE, GREEN = "#0033A0", "#D90012", "#F2A800", "#008C46"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
PHOTO = FIG_DIR / "src_pomegranate.jpg"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l18_backbone_diagrams")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "backbone_diagrams.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def square(img: Image.Image, size: int) -> np.ndarray:
    w, h = img.size
    s = min(w, h)
    return np.asarray(img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
                      .resize((size, size)))


def arrow(ax, p, q, color="#555"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=color, lw=1.0))


def ssl_crops(log):
    if not PHOTO.exists():
        raise FileNotFoundError(f"chapter photo missing: {PHOTO}")
    img = Image.open(PHOTO).convert("RGB")
    full = square(img, 400)
    big = (40, 30, 300)                      # (x, y, size) in the 400-px square
    small = (230, 220, 120)
    crops = [full[y:y + s, x:x + s] for x, y, s in (big, small)]

    fig, ax = plt.subplots(figsize=(5.5, 1.85))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.7); ax.axis("off")
    ax.imshow(full, extent=(0.1, 3.0, 0.3, 3.2))
    for (x, y, s), col in zip((big, small), (ARM_BLUE, ARM_ORANGE)):
        fx = 0.1 + 2.9 * x / 400; fy = 3.2 - 2.9 * (y + s) / 400; fs = 2.9 * s / 400
        ax.add_patch(Rectangle((fx, fy), fs, fs, fill=False, ec=col, lw=1.8))
    ax.text(1.55, 3.3, "one photo, no label", ha="center", va="bottom", fontsize=8)
    ys = [2.0, 0.35]
    for crop, y, col, name in zip(crops, ys, (ARM_BLUE, ARM_ORANGE), ("big crop", "small crop")):
        ax.imshow(crop, extent=(3.6, 4.7, y, y + 1.1))
        ax.add_patch(Rectangle((3.6, y), 1.1, 1.1, fill=False, ec=col, lw=1.8))
        ax.text(4.15, y - 0.05, name, ha="center", va="top", fontsize=7.5, color=col)
        arrow(ax, (4.8, y + 0.55), (5.5, y + 0.55))
        ax.add_patch(FancyBboxPatch((5.55, y + 0.15), 1.5, 0.8, boxstyle="round,pad=0.03",
                                    fc="#eef1f8", ec=ARM_BLUE, lw=1.0))
        ax.text(6.3, y + 0.55, "network", ha="center", va="center", fontsize=8.5)
        arrow(ax, (7.1, y + 0.55), (7.6, y + 0.55))
        rng = np.random.default_rng(SEED)              # the SAME vector twice: they "match"
        vec = rng.uniform(0.2, 1.0, 8)
        for i, v in enumerate(vec):
            ax.add_patch(Rectangle((7.7 + i * 0.16, y + 0.15), 0.12, 0.8 * v, fc=col, ec="none"))
    ax.text(6.3, 1.72, "same weights", ha="center", va="center", fontsize=7.5, color="#555")
    ax.plot([9.15, 9.3, 9.3, 9.15], [2.55, 2.55, 0.9, 0.9], color=GREEN, lw=1.4)
    ax.text(9.45, 1.72, "make the\ntwo vectors\nmatch", ha="left", va="center", fontsize=8.5,
            color=GREEN, fontweight="bold", linespacing=1.0)
    out = FIG_DIR / "ssl_crops.pdf"
    fig.savefig(out, dpi=200); plt.close(fig)
    log.info(f"saved {out}")


def clip_matrix(log):
    from skimage import data
    if not PHOTO.exists():
        raise FileNotFoundError(f"chapter photo missing: {PHOTO}")
    imgs = [square(Image.open(PHOTO).convert("RGB"), 120),
            square(Image.fromarray(data.astronaut()), 120),
            square(Image.fromarray(data.chelsea()), 120),
            square(Image.fromarray(data.coffee()), 120)]
    caps = ["a pomegranate", "an astronaut", "a cat", "a cup of coffee"]
    n = len(caps)

    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 13.75); ax.set_ylim(0, 6.0); ax.axis("off")
    cell, x0, top = 0.95, 2.0, 3.95
    for i, im in enumerate(imgs):
        y = top - (i + 1) * cell
        ax.imshow(im, extent=(0.95, 0.95 + cell * 0.9, y + 0.05, y + 0.05 + cell * 0.9))
        for j in range(n):
            hit = (i == j)
            ax.add_patch(Rectangle((x0 + j * cell, y), cell, cell, fc=GREEN if hit else "#e6e6e6",
                                   ec="white", lw=1.5))
            ax.text(x0 + (j + 0.5) * cell, y + cell / 2, "high" if hit else "low",
                    ha="center", va="center", fontsize=7.5, color="white" if hit else "#888")
    for j, cap in enumerate(caps):
        ax.text(x0 + (j + 0.3) * cell, top + 0.1, cap, ha="left", va="bottom", fontsize=7.5,
                rotation=35, rotation_mode="anchor")
    ax.text(0.55, top - n * cell / 2, "images", ha="center", va="center", fontsize=8,
            rotation=90)
    tx = x0 + n * cell + 0.5
    ax.text(tx, 4.0, "One batch: 4 images, 4 captions.", fontsize=8.5, va="center")
    ax.text(tx, 3.3, "Two encoders, one for images, one for text.", fontsize=8.5, va="center")
    ax.text(tx, 2.6, "Training pushes each image towards\nits own caption (diagonal) and away\n"
            "from the other 3 captions.", fontsize=8.5, va="center", linespacing=1.05)
    ax.text(tx, 1.35, "Afterwards: classify a photo by\nwriting the class names as captions.",
            fontsize=8.5, va="center", color=ARM_BLUE, linespacing=1.05)
    out = FIG_DIR / "clip_matrix.pdf"
    fig.savefig(out, dpi=200); plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    ssl_crops(log)
    clip_matrix(log)


if __name__ == "__main__":
    main()
