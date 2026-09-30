"""Chapter-arc strip for the L19 Vision Tasks deck (chapter-close recap frame).

Four beats, one band - the whole ch6 story in the visual language of L17's timeline
ribbons: L16 built the layer -> L17 evolved architectures -> L18 taught reuse and
trust -> L19 structured the output. L19 is highlighted in red ("you are here").
Each beat carries a small hand-drawn pictogram (matplotlib patches, no images).

Generates into ml/ch6_cnn/fig/:
  chapter_arc.pdf

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/chapter_arc.py
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch, Rectangle

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GRAY = "#9a9a9a"
DARK = "#444444"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

BEATS = [
    ("L16", "the layer", "one neuron, reused\nat every pixel"),
    ("L17", "the architectures", "the ImageNet decade:\nLeNet to ResNet"),
    ("L18", "reuse + trust", "transfer a trunk,\naudit with Grad-CAM"),
    ("L19", "structured outputs", "boxes, masks -\nthe task is the output"),
]


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l19_chapter_arc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "chapter_arc.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def picto_layer(ax, cx, cy, color):
    """L16: a 3x3 kernel grid -> one output pixel."""
    s = 0.26
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((cx - 0.75 + j * s, cy - 0.42 + i * s), s, s,
                                   fc="#dde4f0", ec=color, lw=1.1))
    ax.add_patch(FancyArrow(cx + 0.18, cy, 0.42, 0, width=0.015,
                            head_width=0.10, head_length=0.10, fc=color,
                            ec=color))
    ax.add_patch(Rectangle((cx + 0.70, cy - s / 2), s, s, fc=color, ec=color))


def picto_architectures(ax, cx, cy, color):
    """L17: an ascending stack of blocks (deeper networks)."""
    heights = [0.30, 0.52, 0.74, 0.96]
    for k, h in enumerate(heights):
        ax.add_patch(Rectangle((cx - 0.86 + k * 0.47, cy - 0.48), 0.34, h,
                               fc="#dde4f0", ec=color, lw=1.2))


def picto_transfer(ax, cx, cy, color):
    """L18: frozen trunk blocks + one training head block."""
    for k in range(3):
        ax.add_patch(Rectangle((cx - 0.95 + k * 0.47, cy - 0.24), 0.36, 0.55,
                               fc="#dcdcdc", ec="#9a9a9a", lw=1.2))
    ax.add_patch(Rectangle((cx + 0.46, cy - 0.24), 0.36, 0.55,
                           fc="#bfe3cf", ec="#008C46", lw=1.4))


def picto_outputs(ax, cx, cy, color):
    """L19: a detection box + a segmentation blob."""
    ax.add_patch(Rectangle((cx - 0.95, cy - 0.40), 0.85, 0.85, fill=False,
                           ec=color, lw=2.0))
    ax.add_patch(Circle((cx - 0.525, cy + 0.025), 0.30, fc="#f6c8cb",
                        ec="none"))
    theta = np.linspace(0, 2 * np.pi, 60)
    r = 0.42 * (1 + 0.18 * np.sin(3 * theta))
    ax.fill(cx + 0.55 + r * np.cos(theta), cy + 0.02 + r * np.sin(theta),
            fc=color, alpha=0.75, ec="none")


PICTOS = [picto_layer, picto_architectures, picto_transfer, picto_outputs]


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.6, 3.1))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")

    panel_w, gap = 2.86, 0.36
    x = 0.12
    for k, ((lnum, title, sub), picto) in enumerate(zip(BEATS, PICTOS)):
        hot = (k == 3)
        accent = ARM_RED if hot else ARM_BLUE
        ax.add_patch(FancyBboxPatch(
            (x, 0.22), panel_w, 2.66,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            fc="#fdf4f4" if hot else "#fafbfd",
            ec=accent, lw=2.2 if hot else 1.4))
        ax.text(x + 0.16, 2.62, lnum, fontsize=15, fontweight="bold",
                color="white", va="center", ha="left",
                bbox=dict(boxstyle="square,pad=0.25", fc=accent, ec="none"))
        ax.text(x + 0.78, 2.62, title, fontsize=13.5, fontweight="bold",
                color=DARK, va="center", ha="left")
        picto(ax, x + panel_w / 2, 1.62, accent)
        ax.text(x + panel_w / 2, 0.66, sub, fontsize=10.5, color=DARK,
                ha="center", va="center", linespacing=1.25)
        if k < 3:
            ax.add_patch(FancyArrow(x + panel_w + 0.05, 1.55, gap - 0.16, 0,
                                    width=0.02, head_width=0.16,
                                    head_length=0.12, fc=GRAY, ec=GRAY))
        x += panel_w + gap

    out = FIG_DIR / "chapter_arc.pdf"
    fig.savefig(out)  # fixed geometry - no tight bbox
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
