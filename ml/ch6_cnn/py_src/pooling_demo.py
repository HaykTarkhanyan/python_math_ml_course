"""Real figure for the L16 CNN Foundations deck (Section 4, pooling).

Generates into ml/ch6_cnn/fig/:
  pooling_demo.pdf  -- the four 2x2 windows a 4x4 map splits into, laid out in their spatial
                       positions. In each box the max cell is outlined; below each box its
                       max and average. That is all pooling does: one number per box.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/pooling_demo.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SEED = 509

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

GRID = np.array([[1, 3, 2, 4],
                 [5, 6, 1, 2],
                 [7, 8, 0, 1],
                 [3, 2, 4, 0]], float)
QC = {(0, 0): "#f7d6d9", (0, 1): "#d6e0f5", (1, 0): "#fbeccb", (1, 1): "#d7f0e0"}


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_pooling_demo")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "pooling_demo.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def cell(ax, x, y, val, fc, ec="#555", lw=1.0, fs=16, tc="black"):
    ax.add_patch(Rectangle((x, y), 1, 1, ec=ec, fc=fc, lw=lw))
    ax.text(x + 0.5, y + 0.5, f"{val:g}", ha="center", va="center", fontsize=fs, color=tc)


def draw_box(ax, x0, y0, block, color):
    """A 2x2 window with top-left at (x0,y0); outline its max cell; label max and avg below."""
    mr, mc = np.unravel_index(block.argmax(), block.shape)
    for i in range(2):
        for j in range(2):
            is_max = (i, j) == (mr, mc)
            cell(ax, x0 + j, y0 - (i + 1), block[i, j], fc=color,
                 ec="#111" if is_max else "#999", lw=3.0 if is_max else 1.0)
    ax.text(x0 + 1.0, y0 - 2.42, f"max = {block.max():g}", ha="center", fontsize=13,
            color="#111", fontweight="bold")
    ax.text(x0 + 1.0, y0 - 2.92, f"avg = {block.mean():g}", ha="center", fontsize=13,
            color="#444")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.4, 3.0))
    # lay the four windows out in a single row (landscape, fits the slide)
    for idx, q in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        block = GRID[q[0] * 2:q[0] * 2 + 2, q[1] * 2:q[1] * 2 + 2]
        draw_box(ax, idx * 2.9, 2.4, block, QC[q])
        log.info(f"box {q}: max={block.max():g} avg={block.mean():g}")
    ax.set_xlim(-0.4, 11.4); ax.set_ylim(-0.9, 2.9)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "pooling_demo.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
