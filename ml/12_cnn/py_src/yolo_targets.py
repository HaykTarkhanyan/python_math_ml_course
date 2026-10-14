"""YOLO v1 training targets on the belt photo - for the [51] Vision Tasks deck.

Frame "Training the grid: who learns what": a 7x7 grid over the market photo, the hand-placed
true fruit boxes from task_zoo.py, each box's centre, and the cell that contains that centre
shaded green - YOLO v1's rule: "If the center of an object falls into a grid cell, that grid
cell is responsible for detecting that object" (Redmon et al. 2016, Sec. 2).

Counts quoted on the slide, asserted here: 8 fruit -> 8 distinct responsible cells (no two
centres share a cell); 49 cells x 2 predictors = 98 box predictors; 8 responsible predictors
(one per fruit) and 98 - 8 = 90 predictors trained towards confidence 0.

YOLO squashes every photo to 448x448 before the grid; a 7x7 grid on the squashed photo is
the same 7x7 grid on the original, so it is drawn on the original aspect ratio here.

Generates into ml/12_cnn/fig/:
  yolo_targets.pdf

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/yolo_targets.py
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from task_zoo import FRUIT_BOXES, PHOTO  # noqa: E402  (same boxes as the task-zoo figure)

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
TRAIN_GREEN = "#008c46"

CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

S, B = 7, 2          # YOLO v1: 7x7 grid, 2 box predictors per cell


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l52_yolo_targets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "yolo_targets.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def responsible_cells(w: int, h: int, log: logging.Logger) -> list:
    cells = []
    for (x0, y0, x1, y1) in FRUIT_BOXES:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        col, row = int(cx // (w / S)), int(cy // (h / S))
        cells.append((row, col, cx, cy))
        log.info(f"box ({x0},{y0})-({x1},{y1}) centre ({cx:.0f},{cy:.0f}) -> cell row {row}, col {col}")
    n_fruit = len(FRUIT_BOXES)
    distinct = {(r, c) for r, c, _, _ in cells}
    n_pred = S * S * B
    log.info(f"{n_fruit} fruit, {len(distinct)} responsible cells, {n_pred} predictors, "
             f"{n_pred - n_fruit} pushed to confidence 0")
    if (n_fruit, len(distinct), n_pred, n_pred - n_fruit) != (8, 8, 98, 90):
        raise AssertionError("target counts drifted from the numbers on the slide")
    return cells


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    if not PHOTO.is_file():
        raise FileNotFoundError(f"missing belt photo {PHOTO}")
    img = plt.imread(PHOTO)
    h, w = img.shape[:2]
    if (w, h) != (1920, 1280):
        raise ValueError(f"boxes are hand-placed for 1920x1280, got {w}x{h}")
    cells = responsible_cells(w, h, log)

    fig, ax = plt.subplots(figsize=(2.85, 1.9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img, alpha=0.75)      # scalar alpha is honoured in PDF (array alpha is not)
    cw, ch = w / S, h / S
    for (row, col, _, _) in cells:
        ax.add_patch(Rectangle((col * cw, row * ch), cw, ch, fc=TRAIN_GREEN,
                               ec="none", alpha=0.45))
    for k in range(1, S):
        ax.axvline(k * cw, color="white", lw=0.7)
        ax.axhline(k * ch, color="white", lw=0.7)
    for (row, col, _, _) in cells:    # outline the responsible cells over the grid
        ax.add_patch(Rectangle((col * cw, row * ch), cw, ch, fill=False,
                               ec="#00ff7f", lw=1.3, zorder=3))   # lines sit at zorder 2
    for (x0, y0, x1, y1) in FRUIT_BOXES:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec=ARM_ORANGE, lw=1.0, zorder=4))
    for (_, _, cx, cy) in cells:
        ax.plot(cx, cy, "o", ms=3.6, mfc=ARM_RED, mec="white", mew=0.6, zorder=5)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#888888")

    out = FIG_DIR / "yolo_targets.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    log.info(f"saved {out}; done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
