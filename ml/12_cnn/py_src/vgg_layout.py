"""Figure for the L17 CNN Architectures deck (Section 3, VGG-16 layer by layer).

Generates into ml/12_cnn/fig/:
  vgg_layout.pdf  -- VGG-16 as five blocks of 3x3 convolutions (2, 2, 3, 3, 3 layers; 64, 128,
                     256, 512, 512 channels), each followed by a 2x2 max-pool that halves the grid
                     (224 -> 7), then three dense layers. Bar height = grid size; the weight count
                     of every block is computed here and the total checked against torchvision's
                     vgg16() (138,357,544 - built without weights, nothing is downloaded).
                     Punchline in the figure: the three dense layers hold 124M of the 138M weights.

Drawn at its slide size (5.5 x 1.9 in), so font sizes are slide sizes (7.5-9 pt).

Run with the project venv (~10 s, mostly importing torch):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/vgg_layout.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
BLOCK_COLORS = ["#c6d3ef", "#9fb3e3", "#7891d4", "#4f6fc2", "#2f52ad"]

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

BLOCKS = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]   # (conv layers, channels)
DENSE = [(512 * 7 * 7, 4096), (4096, 4096), (4096, 1000)]


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l17_vgg_layout")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "vgg_layout.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def counts():
    """Weights per conv block and per dense layer (3x3 kernels, biases included)."""
    conv, c_in = [], 3
    for n_layers, c_out in BLOCKS:
        w = 0
        for _ in range(n_layers):
            w += 3 * 3 * c_in * c_out + c_out
            c_in = c_out
        conv.append(w)
    dense = [i * o + o for i, o in DENSE]
    return conv, dense


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    conv, dense = counts()
    total = sum(conv) + sum(dense)

    import torch
    from torchvision.models import vgg16
    ref = sum(p.numel() for p in vgg16(weights=None).parameters())
    if total != ref:
        raise ValueError(f"hand count {total:,} != torchvision vgg16 {ref:,}")
    log.info(f"VGG-16: conv {sum(conv):,} + dense {sum(dense):,} = {total:,} (torchvision agrees)")

    fig, ax = plt.subplots(figsize=(5.5, 1.9))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-0.6, 21.4); ax.set_ylim(-1.75, 5.85); ax.axis("off")

    x, size = 0.4, 224
    for b, ((n_layers, ch), w, col) in enumerate(zip(BLOCKS, conv, BLOCK_COLORS)):
        h = 4.0 * size / 224 + 0.25                     # bar height ~ grid size
        for _ in range(n_layers):
            ax.add_patch(Rectangle((x, 0), 0.55, h, fc=col, ec="#333", lw=0.5))
            x += 0.7
        ax.add_patch(Rectangle((x, 0), 0.3, h * 0.5, fc=ARM_RED, ec="none"))   # max-pool
        mid = x - 0.7 * n_layers / 2
        ax.text(mid, -0.15, f"{n_layers} x conv\n{ch} ch\n{size}x{size}", ha="center",
                va="top", fontsize=7.5, linespacing=1.0)
        x += 0.95
        size //= 2
    # dense head
    x += 0.3
    x_dense = x
    for k, ((_, o), w) in enumerate(zip(DENSE, dense)):
        ax.add_patch(Rectangle((x, 0), 0.55, 2.6 if o == 4096 else 1.4, fc=ARM_ORANGE,
                               ec="#333", lw=0.5))
        x += 0.85
    ax.text((x_dense + x - 0.3) / 2, -0.15, "dense\n4096, 4096,\n1000", ha="center", va="top",
            fontsize=7.5, linespacing=1.0)
    ax.text(0.4, 5.75, "13 conv layers, every one 3x3", ha="left", va="top", fontsize=8.5,
            color=ARM_BLUE, fontweight="bold")
    ax.text(0.4, 5.0, "red = 2x2 max-pool: the grid halves, then the channels double",
            ha="left", va="top", fontsize=7.5, color=ARM_RED)
    ax.text((x_dense + x - 0.3) / 2, 5.75,
            f"3 dense layers:\n{sum(dense) / 1e6:.0f}M of the\n{total / 1e6:.0f}M weights",
            ha="center", va="top", fontsize=8.5, color="#b07d00", fontweight="bold",
            linespacing=1.0)
    out = FIG_DIR / "vgg_layout.pdf"
    fig.savefig(out); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
