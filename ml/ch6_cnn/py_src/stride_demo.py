"""Real figure for the L16 CNN Foundations deck (Section 4, stride).

Generates into ml/ch6_cnn/fig/:
  stride_demo.pdf  -- same 5x5 input, 3x3 kernel. Stride 1 stops at 9 positions -> 3x3
                      output; stride 2 jumps by two and stops at 4 -> 2x2 output. Stride
                      controls how many stops, hence the output size.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/stride_demo.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SEED = 509
BLUE, GREEN = "#0033A0", "#008C46"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_stride_demo")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "stride_demo.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def grid(ax, x0, y0, n, m, face="white", ec="#777"):
    for i in range(n):
        for j in range(m):
            ax.add_patch(Rectangle((x0 + j, y0 - (i + 1)), 1, 1, ec=ec, fc=face, lw=1.0))


def panel(ax, stride, title):
    top = 5
    grid(ax, 0, top, 5, 5)
    # window top-left positions for a 3x3 kernel over 5x5 with this stride
    positions = [(r, c) for r in range(0, 3, stride) for c in range(0, 3, stride)]
    out_n = len(range(0, 3, stride))
    # mark each stop with a small dot at the window's top-left, and outline the first window
    for (r, c) in positions:
        ax.plot(c + 1.5, top - (r + 1.5), "o", color=BLUE, ms=6)
    ax.add_patch(Rectangle((0, top - 3), 3, 3, ec=BLUE, fc="none", lw=2.6))
    ax.annotate("", xy=(7.3, top - 2.5), xytext=(5.4, top - 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=2))
    grid(ax, 7.8, top - 1 + out_n / 2, out_n, out_n, face="#eafaf1", ec=GREEN)
    ax.text(7.8 + out_n / 2, top - 1 + out_n / 2 - out_n - 0.5, f"{out_n}x{out_n} output",
            ha="center", va="top", fontsize=12, color=GREEN)
    ax.text(2.5, top - 5 - 0.5, f"{len(positions)} window stops",
            ha="center", va="top", fontsize=11, color=BLUE)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-0.4, 7.8 + out_n + 0.6); ax.set_ylim(-1.6, top + 0.5)
    ax.set_aspect("equal"); ax.axis("off")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    panel(a1, 1, "Stride 1: step by one  (5x5, k=3 -> 3x3)")
    panel(a2, 2, "Stride 2: jump by two  (5x5, k=3 -> 2x2)")
    fig.tight_layout()
    out = FIG_DIR / "stride_demo.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
