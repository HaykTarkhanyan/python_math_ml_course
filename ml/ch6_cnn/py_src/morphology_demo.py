"""Real figure for the L16 CNN Foundations deck (Section: kernel variations).

Generates into ml/ch6_cnn/fig/:
  morphology_demo.pdf  -- a synthetic binary shape shown three ways: original,
                          after DILATION (3x3 max, grows and fills small holes) and
                          after EROSION (3x3 min, shrinks and breaks thin parts).
                          Dilation and erosion are opposites (morphological duals):
                          the erosion of the shape is the dilation of the background.
                          NOTE: these are max/min neighborhood operations, NOT the
                          weighted-sum convolution the rest of the deck uses.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/morphology_demo.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

SEED = 509
BLUE = "#0033A0"

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
    fh = logging.FileHandler(LOGS_DIR / "morphology_demo.log"); fh.setFormatter(fmt)
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
    ax.set_title(title, fontsize=13, color=color, pad=8)
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

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))
    panel(axes[0], base, "Original", "#333")
    panel(axes[1], dil, "Dilation (3x3 max):  grows, fills holes", BLUE)
    panel(axes[2], ero, "Erosion (3x3 min):  shrinks, breaks thin parts", "#D90012")
    fig.suptitle(
        "Morphology: neighborhood max vs min - opposites (duals), not weighted-sum convolution",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "morphology_demo.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
