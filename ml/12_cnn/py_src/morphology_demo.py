"""Real figure for the L16 CNN Foundations deck (Section: kernel variations).

Generates into ml/12_cnn/fig/:
  morphology_demo.pdf  -- a synthetic binary shape shown three ways: original,
                          after DILATION (3x3 max, grows and fills small holes) and
                          after EROSION (3x3 min, shrinks and breaks thin parts).
                          Dilation and erosion are opposites (morphological duals):
                          the erosion of the shape is the dilation of the background.
                          NOTE: these are max/min neighborhood operations, NOT the
                          weighted-sum convolution the rest of the deck uses.
  morphology_by_hand.pdf -- (added 2026-10-01, instructor: "only results with some titles")
                          HOW it is computed: a 7x7 mask with one 3x3 window on the square's
                          corner pixel; the window's max is 1 (dilation keeps it on) and its
                          min is 0 (erosion switches it off); then both full results.

2026-10-01: both figures are drawn at their size on the slide (5.4-5.5 in wide), so font sizes
are slide sizes (8-9.5 pt; the old figure rendered at 4.4-5.2 pt). The suptitle moved to the slide.

Run with the project venv:
    ./ma/Scripts/python.exe ml/12_cnn/py_src/morphology_demo.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

SEED = 509
BLUE, ORANGE, RED = "#0033A0", "#E8590C", "#D90012"
LIGHT_BLUE = "#8fa6db"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_morphology_demo")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "morphology_demo.log", encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def make_shape(n: int = 44) -> np.ndarray:
    """A binary test image with a filled square, a square with a hole, a thin line,
    and isolated dots -- features that respond visibly and differently to dilation
    and erosion."""
    img = np.zeros((n, n), dtype=bool)
    # filled square (top-left): grows / shrinks uniformly
    img[5:13, 5:13] = True
    # square with a small hole (top-right): dilation fills the hole, erosion enlarges it
    img[4:16, 27:39] = True
    img[9:12, 32:35] = False
    # thin horizontal line, 1 px tall (middle): erosion breaks it, dilation thickens it
    img[24, 5:33] = True
    # isolated single-pixel dots (bottom): erosion deletes them, dilation grows them
    for c in (6, 12, 18, 24):
        img[34, c] = True
    # a small plus (bottom-right): thin arms vanish under erosion
    img[31:40, 34] = True
    img[35, 30:39] = True
    return img


def dilate(b: np.ndarray) -> np.ndarray:
    """3x3 binary dilation = max over the 3x3 neighborhood (pad with background)."""
    h, w = b.shape
    p = np.pad(b, 1, constant_values=False)
    out = np.zeros_like(b)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out |= p[1 + di:1 + di + h, 1 + dj:1 + dj + w]
    return out


def erode_zero_pad(b: np.ndarray) -> np.ndarray:
    """3x3 binary erosion with background padding: a pixel survives only if its whole
    window, borders included, is foreground (the by-hand mask never touches the border)."""
    h, w = b.shape
    p = np.pad(b, 1, constant_values=False)
    out = np.ones_like(b)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out &= p[1 + di:1 + di + h, 1 + dj:1 + dj + w]
    return out


def erode(b: np.ndarray) -> np.ndarray:
    """3x3 binary erosion = min over the 3x3 neighborhood (pad with foreground so the
    image border is not treated as an edge of the shape)."""
    h, w = b.shape
    p = np.pad(b, 1, constant_values=True)
    out = np.ones_like(b)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out &= p[1 + di:1 + di + h, 1 + dj:1 + dj + w]
    return out


def panel(ax, img, title, color):
    cmap = ListedColormap(["white", BLUE])
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9, color=color, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#bbb")


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    base = make_shape()
    dil = dilate(base)
    ero = erode(base)
    log.info(f"foreground px  original={base.sum()}  dilation={dil.sum()}  erosion={ero.sum()}")
    if not (ero.sum() < base.sum() < dil.sum()):
        raise ValueError("expected erosion < original < dilation in foreground pixel count")

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0))
    panel(axes[0], base, "original", "#333")
    panel(axes[1], dil, "dilation (3x3 max)", BLUE)
    panel(axes[2], ero, "erosion (3x3 min)", RED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.06)
    out = FIG_DIR / "morphology_demo.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")
    by_hand(log)


def cells(ax, img, x0, title, color, hl=None, new_on=None):
    """Draw a binary mask as a cell grid; hl = set of (i, j) to outline (switched-off pixels),
    new_on = set of (i, j) to fill light blue (switched-on pixels)."""
    n = img.shape[0]
    for i in range(n):
        for j in range(n):
            fc = BLUE if img[i, j] else "white"
            if new_on and (i, j) in new_on:
                fc = LIGHT_BLUE
            ax.add_patch(Rectangle((x0 + j, n - 1 - i), 1, 1, fc=fc, ec="#aaa", lw=0.5))
            if hl and (i, j) in hl:
                ax.add_patch(Rectangle((x0 + j + 0.12, n - 1 - i + 0.12), 0.76, 0.76,
                                       fc="none", ec=color, lw=1.4, ls=(0, (2, 1))))
    ax.text(x0 + n / 2, n + 0.35, title, ha="center", va="bottom", fontsize=8.5, color=color)


def by_hand(log):
    """A 7x7 mask: a 3x3 square and one stray dot. The window sits on the square's corner."""
    m = np.zeros((7, 7), dtype=bool)
    m[2:5, 1:4] = True
    m[1, 5] = True
    dil, ero = dilate(m), erode_zero_pad(m)
    win_rc = (2, 1)                                   # the square's top-left corner pixel
    r, c = win_rc
    window = np.pad(m, 1)[r:r + 3, c:c + 3]           # 3x3 window centred on (r, c)
    if not (window.max() == 1 and window.min() == 0 and m[r, c]):
        raise ValueError("the chosen window must mix 0s and 1s on a foreground pixel")
    if not (ero.sum() == 1 and ero[3, 2] and dil.sum() == 32):   # 25 + 9 - 2 overlap
        raise ValueError(f"unexpected results: erosion {ero.sum()} px, dilation {dil.sum()} px")
    log.info(f"by hand: window max {window.max():d} min {window.min():d}; "
             f"dilation {dil.sum()} px, erosion {ero.sum()} px (the square's centre)")

    fig, ax = plt.subplots(figsize=(5.5, 1.75))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-0.9, 31.0); ax.set_ylim(-0.6, 8.6); ax.set_aspect("equal"); ax.axis("off")

    cells(ax, m, 0, "input mask (1 = blue)", "#333")
    ax.add_patch(Rectangle((c - 1, 7 - 1 - (r + 1)), 3, 3, fc="none", ec=ORANGE, lw=2.2))

    # zoomed window
    zx = 8.6
    for i in range(3):
        for j in range(3):
            v = int(window[i, j])
            ax.add_patch(Rectangle((zx + j * 1.3, 5.2 - i * 1.3), 1.3, 1.3,
                                   fc=BLUE if v else "white", ec=ORANGE, lw=1.0))
            ax.text(zx + j * 1.3 + 0.65, 5.2 - i * 1.3 + 0.65, str(v), ha="center", va="center",
                    fontsize=8.5, color="white" if v else "#333")
    ax.text(zx + 1.95, 7.35, "this window", ha="center", va="bottom", fontsize=8.5, color=ORANGE)
    ax.text(zx + 1.95, 2.25, "max = 1", ha="center", va="top", fontsize=8.5, color=BLUE)
    ax.text(zx + 1.95, 1.0, "min = 0", ha="center", va="top", fontsize=8.5, color=RED)

    new_on = {(i, j) for i in range(7) for j in range(7) if dil[i, j] and not m[i, j]}
    gone = {(i, j) for i in range(7) for j in range(7) if m[i, j] and not ero[i, j]}
    cells(ax, dil, 15.4, "dilation: any 1 -> 1", BLUE, new_on=new_on)
    cells(ax, np.where(ero, True, False), 23.6, "erosion: all 1 -> 1", RED, hl=gone)
    ax.text(15.4 + 3.5, -0.45, "light = switched on", ha="center", va="top", fontsize=7.5,
            color=BLUE)
    ax.text(23.6 + 3.5, -0.45, "dashed = switched off", ha="center", va="top", fontsize=7.5,
            color=RED)

    out = FIG_DIR / "morphology_by_hand.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
