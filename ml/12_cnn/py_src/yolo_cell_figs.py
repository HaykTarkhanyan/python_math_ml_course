"""Two YOLO v1 figures for the [51] Vision Tasks deck.

  yolo_address.pdf -- frame "A cell is an address, not a window": left, what one 7x7 cell's own
                      patch of the belt photo shows (a slab of red skin and seeds - no way to
                      tell the fruit's size from it); right, the whole photo with that cell and
                      the fruit's true box, which spans several cells. YOLO v1 ends in two fully
                      connected layers, so every cell's numbers are computed from every pixel
                      (Redmon et al. 2016, Sec. 2: "uses features from the entire image to
                      predict each bounding box").
  yolo_decode.pdf  -- frame "Decoding one YOLO cell, by hand": the 448x448 input with its 7x7
                      grid (64 px cells), cell (row 3, col 2), the decoded centre (160, 208)
                      and the 90 x 134 px box - bigger than the 64 px cell, because w and h are
                      fractions of the IMAGE. The decode numbers are asserted.

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/yolo_cell_figs.py
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
GREEN = "#00c060"

CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

S = 7
FRUIT = 1                    # FRUIT_BOXES[1]: the centre split-open fruit


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l52_yolo_cell_figs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "yolo_cell_figs.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def address(log: logging.Logger) -> None:
    img = plt.imread(PHOTO)
    h, w = img.shape[:2]
    cw, ch = w / S, h / S
    x0, y0, x1, y1 = FRUIT_BOXES[FRUIT]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    col, row = int(cx // cw), int(cy // ch)
    cells_w, cells_h = (x1 - x0) / cw, (y1 - y0) / ch
    log.info(f"fruit centre ({cx:.0f},{cy:.0f}) -> cell row {row}, col {col}; cell "
             f"{cw:.0f}x{ch:.0f} px; box {x1 - x0}x{y1 - y0} px = {cells_w:.1f} x {cells_h:.1f} cells")
    if cells_w < 2 or cells_h < 2:
        raise AssertionError("the point of the figure is a box several cells wide")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 1.72),
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.08)
    px0, py0 = int(col * cw), int(row * ch)
    a1.imshow(img[py0:int(py0 + ch), px0:int(px0 + cw)])
    a1.set_title("the cell's own patch, alone", fontsize=7.5, pad=2)
    for sp in a1.spines.values():
        sp.set_edgecolor(GREEN); sp.set_linewidth(2.5)

    a2.imshow(img, alpha=0.85)
    for k in range(1, S):
        a2.axvline(k * cw, color="white", lw=0.5)
        a2.axhline(k * ch, color="white", lw=0.5)
    a2.add_patch(Rectangle((px0, py0), cw, ch, fc=GREEN, alpha=0.5, ec=GREEN, lw=1.5))
    a2.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ARM_ORANGE, lw=2.0))
    a2.plot([cx], [cy], "o", color=ARM_RED, ms=4)
    a2.set_xlim(0, w); a2.set_ylim(h, 0)
    a2.set_title("its prediction: computed from the whole photo", fontsize=7.5, pad=2)
    for ax in (a1, a2):
        ax.set_xticks([]); ax.set_yticks([])
    out = FIG_DIR / "yolo_address.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info(f"saved {out}")


def decode(log: logging.Logger) -> None:
    n, cell = 448, 448 // S
    row, col = 3, 2
    x, y, bw, bh = 0.5, 0.25, 0.2, 0.3
    cx, cy = (col + x) * cell, (row + y) * cell
    pw, ph = bw * n, bh * n
    log.info(f"cell {cell} px; centre ({cx:.0f}, {cy:.0f}); box {pw:.1f} x {ph:.1f} px")
    if (cell, round(cx), round(cy), round(pw), round(ph)) != (64, 160, 208, 90, 134):
        raise AssertionError("decode drifted from the numbers on the slide")

    fig, ax = plt.subplots(figsize=(2.35, 2.35))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.10)
    ax.add_patch(Rectangle((0, 0), n, n, fc="#f4f4f4", ec="black", lw=0.8))
    for k in range(1, S):
        ax.axvline(k * cell, color="#bbbbbb", lw=0.6)
        ax.axhline(k * cell, color="#bbbbbb", lw=0.6)
    ax.add_patch(Rectangle((col * cell, row * cell), cell, cell, fc=GREEN, alpha=0.45,
                           ec=GREEN, lw=1.5))
    ax.add_patch(Rectangle((cx - pw / 2, cy - ph / 2), pw, ph, fill=False, ec=ARM_ORANGE,
                           lw=2.0))
    ax.plot([cx], [cy], "o", color=ARM_RED, ms=4, zorder=5)
    ax.annotate("centre (160, 208)", (cx, cy), (cx + 60, cy - 120), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, shrinkB=2))
    ax.text(cx, cy + ph / 2 + 8, "w = 0.2 x 448 = 90 px", ha="center", va="top", fontsize=7)
    ax.text(cx + pw / 2 + 8, cy, "h = 0.3 x 448\n= 134 px", ha="left", va="center",
            fontsize=7, linespacing=1.1)
    ax.text(col * cell + 3, (row + 1) * cell - 3, "cell", ha="left", va="bottom",
            fontsize=6.5, color="#00602f")
    ticks = list(range(0, n + 1, 2 * cell))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(labelsize=6.5, length=2)
    ax.set_xlim(0, n); ax.set_ylim(n, 0)
    out = FIG_DIR / "yolo_decode.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    address(log)
    decode(log)
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
