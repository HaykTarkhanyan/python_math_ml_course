"""Real figure for the L17 CNN Architectures deck (wider-zoo section, grouped conv).

Generates into ml/ch6_cnn/fig/:
  grouped_conv_demo.pdf  -- standard vs grouped convolution as channel connectivity.
                            Standard: every output channel connects to every input
                            channel (dense, C_in*C_out links). Grouped (g=2): channels
                            split into 2 groups, each output sees only its group -> half
                            the connections and weights. AlexNet's two-GPU split is g=2;
                            depthwise conv is g = C (one group per channel).

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/grouped_conv_demo.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

SEED = 509
BLUE, ORANGE, GREEN, GREY = "#0033A0", "#F2A800", "#008C46", "#9aa0a6"
C = 4   # input and output channels

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l17_grouped_conv_demo")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "grouped_conv_demo.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def y_of(ch, gap):
    """vertical position of channel `ch` (0=top), with an extra `gap` after the first
    half so the two groups read as separate blocks."""
    y = (C - 1 - ch)
    if ch >= C // 2:
        y -= gap
    return y


def node(ax, x, y, color):
    ax.add_patch(Rectangle((x - 0.28, y - 0.28), 0.56, 0.56, fc=color, ec="#222", lw=1.2))


def panel(ax, grouped, gap, title):
    xin, xout = 0.0, 3.0
    incol = [BLUE, BLUE, ORANGE, ORANGE] if grouped else [GREY] * C
    outcol = [BLUE, BLUE, ORANGE, ORANGE] if grouped else [GREY] * C
    links = 0
    for i in range(C):
        for o in range(C):
            same_group = (i // (C // 2)) == (o // (C // 2))
            if grouped and not same_group:
                continue
            col = incol[i] if grouped else GREY
            ax.plot([xin + 0.28, xout - 0.28], [y_of(i, gap), y_of(o, gap)],
                    color=col, lw=1.4, alpha=0.75, zorder=1)
            links += 1
    for i in range(C):
        node(ax, xin, y_of(i, gap), incol[i])
        node(ax, xout, y_of(i, gap), outcol[i])
    ax.text(xin, C + 0.35 - (gap if False else 0), "input\nchannels", ha="center",
            va="bottom", fontsize=10, color="#444")
    ax.text(xout, C + 0.35, "output\nchannels", ha="center", va="bottom",
            fontsize=10, color="#444")
    ax.text(1.5, -1.15 - gap, f"{links} connections", ha="center", va="top",
            fontsize=11, color="#222")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(-1.0, 4.0); ax.set_ylim(-1.7 - gap, C + 1.1)
    ax.set_aspect("equal"); ax.axis("off")
    return links


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    n_std = panel(a1, False, 0.0, "Standard: every output sees every channel")
    n_grp = panel(a2, True, 0.9, "Grouped (g = 2): each output sees only its group")
    if not (n_std == C * C and n_grp == C * C // 2):
        raise ValueError(f"connection counts off: standard={n_std}, grouped={n_grp}")
    log.info(f"connections  standard={n_std}  grouped(g=2)={n_grp}  (ratio {n_std / n_grp:.0f}x)")
    fig.suptitle("Grouped convolution: fewer cross-channel connections -> g x fewer weights",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "grouped_conv_demo.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
