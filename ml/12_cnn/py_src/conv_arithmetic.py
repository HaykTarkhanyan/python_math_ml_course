"""Real figure for the L16 CNN Foundations deck (Section 4, padding).

Generates into ml/ch6_cnn/fig/:
  conv_arithmetic.pdf  -- two schematic grids: "valid" (no padding) shrinks a 5x5 input
                          to 3x3 with a 3x3 kernel; "same" (zero-pad 1) keeps it 5x5.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/conv_arithmetic.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SEED = 509
ARM_BLUE, ARM_RED, ARM_GREEN = "#0033A0", "#D90012", "#008C46"
PAD_FC = "#e9edf6"        # light blue-grey for the zero-padding ring

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

CELL = 1.0


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_conv_arithmetic")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "conv_arithmetic.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def draw_grid(ax, x0, y0, rows, cols, face="white", edge="#444", lw=1.1, ring=0):
    """Grid whose top-left cell sits at (x0, y0); rows grow downward.

    If ring > 0, the outer `ring` cells are drawn as padding (PAD_FC), the inner as `face`.
    """
    for i in range(rows):
        for j in range(cols):
            is_pad = ring > 0 and (i < ring or j < ring
                                   or i >= rows - ring or j >= cols - ring)
            ax.add_patch(Rectangle((x0 + j * CELL, y0 - (i + 1) * CELL), CELL, CELL,
                                    ec=edge, fc=PAD_FC if is_pad else face, lw=lw))


def window(ax, x0, y0, color=ARM_BLUE):
    """Outline a 3x3 kernel window with top-left cell at (x0, y0)."""
    ax.add_patch(Rectangle((x0, y0 - 3 * CELL), 3 * CELL, 3 * CELL,
                           ec=color, fc="none", lw=2.6))


def panel(ax, title, ring, out_n, note):
    """One panel: 5x5 input (optionally with a padding ring) -> out_n x out_n output."""
    grid_n = 5 + 2 * ring
    top = grid_n * CELL
    draw_grid(ax, 0, top, grid_n, grid_n, ring=ring)
    window(ax, 0, top)                                    # kernel at the top-left
    # arrow
    ax.annotate("", xy=(grid_n + 2.4, top / 2), xytext=(grid_n + 0.5, top / 2),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=2))
    # output grid, vertically centred against the input
    out_top = top / 2 + out_n * CELL / 2
    draw_grid(ax, grid_n + 2.9, out_top, out_n, out_n, face="#eafaf1", edge=ARM_GREEN)
    ax.text(grid_n + 2.9 + out_n * CELL / 2, out_top - out_n * CELL - 0.55,
            f"{out_n}x{out_n}", ha="center", va="top", fontsize=12, color=ARM_GREEN)
    ax.text(grid_n / 2, top - grid_n * CELL - 0.55, "5x5 input" + note,
            ha="center", va="top", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-0.5, grid_n + 2.9 + out_n + 0.6)
    ax.set_ylim(-1.6, top + 0.6)
    ax.set_aspect("equal"); ax.axis("off")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    panel(axA, "Valid: no padding  (k=3, p=0, s=1)", ring=0, out_n=3,
          note="  ->  shrinks")
    panel(axB, "Same: zero-pad 1  (k=3, p=1, s=1)", ring=1, out_n=5,
          note=" + zero ring  ->  preserved")
    fig.tight_layout()
    out = FIG_DIR / "conv_arithmetic.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
