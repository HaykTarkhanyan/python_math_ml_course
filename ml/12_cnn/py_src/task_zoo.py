"""Task-zoo figure for the L19 Vision Tasks deck (Section 1).

One photo, four questions: the same multi-pomegranate market photo shown four ways -
classification (one label), classification + localization (one box), object detection
(a box per fruit), semantic segmentation (a class per pixel). Overlay coordinates are
hand-placed for fig/borrowed/pomegranate_market.jpg (1920x1280); no model inference.

Per LEARNINGS: array-alpha on imshow is silently ignored in PDF output, so the
segmentation overlay is baked into the RGB pixels directly.

Generates into ml/ch6_cnn/fig/:
  task_zoo.pdf

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/task_zoo.py
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
PHOTO = FIG_DIR / "borrowed" / "pomegranate_market.jpg"

# hand-placed fruit boxes (x0, y0, x1, y1) in full-res pixel coords (1920x1280)
FRUIT_BOXES = [
    (30, 660, 600, 1250),     # 1 front-left whole fruit
    (640, 560, 1310, 1230),   # 2 center split-open fruit  <- localization target
    (1320, 660, 1920, 1270),  # 3 front-right whole fruit
    (0, 200, 520, 690),       # 4 mid-left fruit
    (390, 30, 940, 660),      # 5 center-back fruit with the long stalk
    (1080, 40, 1600, 620),    # 6 top-right large fruit
    (920, 0, 1200, 230),      # 7 top-center partial fruit
    (1630, 0, 1920, 520),     # 8 top-right corner split half
]
LOC_TARGET = 1  # index into FRUIT_BOXES for the localization panel
DET_SCORES = [0.96, 0.93, 0.95, 0.88, 0.91, 0.90, 0.62, 0.71]

# ellipses for the semantic mask: (cx, cy, rx, ry) full-res coords
FRUIT_ELLIPSES = [
    (310, 960, 290, 300),
    (975, 900, 330, 330),
    (1630, 970, 310, 300),
    (240, 450, 260, 250),
    (660, 350, 270, 310),
    (1340, 330, 260, 290),
    (1060, 100, 150, 130),
    (1790, 250, 160, 270),
]

PANELS = [
    ("1. Classification", 'one label for the whole image'),
    ("2. + Localization", 'the label and one box'),
    ("3. Object detection", 'a box for every object'),
    ("4. Semantic segmentation", 'a class for every pixel'),
]


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l19_task_zoo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "task_zoo.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def build_semantic_overlay(img: np.ndarray) -> np.ndarray:
    """Bake a semantic fruit-vs-background tint directly into the RGB pixels."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    fruit = np.zeros((h, w), dtype=bool)
    for cx, cy, rx, ry in FRUIT_ELLIPSES:
        fruit |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    out = img.astype(float)
    orange = np.array([242, 168, 0], dtype=float)   # fruit class tint
    blue = np.array([20, 40, 110], dtype=float)     # background class tint
    a_fruit, a_bg = 0.45, 0.62
    out[fruit] = (1 - a_fruit) * out[fruit] + a_fruit * orange
    out[~fruit] = (1 - a_bg) * out[~fruit] + a_bg * blue
    return out.astype(np.uint8)


def chip(ax, x, y, text, fc, fontsize=11):
    """A small filled label chip in axes-pixel (data) coordinates."""
    ax.text(x, y, text, fontsize=fontsize, fontweight="bold", color="white",
            va="top", ha="left",
            bbox=dict(boxstyle="square,pad=0.35", fc=fc, ec="none"))


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)
    if not PHOTO.is_file():
        raise FileNotFoundError(
            f"missing belt photo {PHOTO} - download it into fig/borrowed/ first")

    img = plt.imread(PHOTO)
    h, w = img.shape[:2]
    log.info(f"loaded {PHOTO.name} {w}x{h}")
    if (w, h) != (1920, 1280):
        raise ValueError(f"overlay coords are hand-placed for 1920x1280, got {w}x{h}")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.03, hspace=0.16)

    for k, (ax, (title, sub)) in enumerate(zip(axes.flat, PANELS)):
        ax.imshow(build_semantic_overlay(img) if k == 3 else img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#888888")
        ax.set_title(f"{title}\n{sub}", fontsize=12.5, pad=6)

        if k == 0:
            chip(ax, 55, 90, '"pomegranates"', ARM_BLUE, fontsize=13)
        elif k == 1:
            x0, y0, x1, y1 = FRUIT_BOXES[LOC_TARGET]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, ec=ARM_BLUE, lw=3.0))
            chip(ax, x0 + 12, y0 + 16, "pomegranate", ARM_BLUE, fontsize=11)
        elif k == 2:
            for (x0, y0, x1, y1), s in zip(FRUIT_BOXES, DET_SCORES):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, ec=ARM_ORANGE, lw=2.4))
                ax.text(x0 + 10, y0 + 14, f"{s:.2f}", fontsize=8.5,
                        fontweight="bold", color="black", va="top",
                        bbox=dict(boxstyle="square,pad=0.22", fc=ARM_ORANGE,
                                  ec="none"))
        else:
            chip(ax, 55, 90, "fruit", ARM_ORANGE, fontsize=11)
            chip(ax, 1450, 1240 - 60, "background", "#1e2a6e", fontsize=11)

    out = FIG_DIR / "task_zoo.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out} ({len(FRUIT_BOXES)} detection boxes, "
             f"{len(FRUIT_ELLIPSES)} mask ellipses)")


if __name__ == "__main__":
    main()
