"""Neural style transfer, idea level - two figures for the [51] Vision Tasks deck.

  style_loop.pdf -- frame "Neural style transfer: train the picture, not the network": a photo,
                    a painting and the picture being made all go through ONE frozen pretrained
                    CNN (VGG-19 in Gatys et al. 2015); the content loss compares deep-layer
                    maps with the photo's, the style loss compares feature correlations with the
                    painting's, and the gradient updates the PIXELS. Images = the paper's own
                    Fig. 2 panels (fig/borrowed/gatys_content.png, gatys_starry.png); the
                    painting thumbnail is cut from the inset of gatys_starry.png.
  gram_toy.pdf   -- frame "What 'style' means to a CNN": two 4x4 feature maps ("swirl", "yellow")
                    and their Gram matrix by hand; then the same 16 positions shuffled (one
                    permutation for both maps) - the maps look different, the Gram matrix is
                    identical. Style keeps WHAT fires together, not WHERE. Values asserted.

Gatys facts used on the slides (arXiv 1508.06576, Methods): VGG-19's 16 conv and 5 pooling
layers, no fully connected layers; content from conv4_2, style from conv1_1 ... conv5_1;
"gradient descent on a white noise image"; style = "correlations between the different filter
responses", given by the Gram matrix.

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/style_transfer_figs.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#008c46"
FS = 7.2

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
BORROWED = FIG_DIR / "borrowed"

# swirl / yellow detector maps (1 = fires), hand-chosen so the Gram matrix is easy by hand
F_SWIRL = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]])
F_YELLOW = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
EXPECTED_G = np.array([[7, 5], [5, 5]])


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l52_style_transfer_figs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "style_transfer_figs.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def arrow(ax, p, q, color="black", lw=0.9, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 shrinkA=1, shrinkB=1, linestyle=ls))


def thumb(fig, W, H, x, y, w, img, title):
    h = w * img.shape[0] / img.shape[1]
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.imshow(img); a.set_xticks([]); a.set_yticks([])
    a.set_title(title, fontsize=FS, pad=2)
    return h


def style_loop(log) -> None:
    content = plt.imread(BORROWED / "gatys_content.png")[..., :3]
    starry = plt.imread(BORROWED / "gatys_starry.png")[..., :3]
    if starry.shape[:2] != (726, 969):
        raise AssertionError(f"gatys_starry.png is {starry.shape[:2]}; crop boxes assume 726x969")
    painting = starry[470:712, 26:336]          # the inset: Van Gogh's Starry Night
    result = starry[38:660, 98:930]             # the stylized photo, inset and "C" trimmed off
    content = content[30:, 60:] if content.shape[1] > 800 else content

    W, H = 5.45, 2.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    tw = 0.95
    thumb(fig, W, H, 0.05, 1.12, tw, content, "photo (content)")
    thumb(fig, W, H, 0.05, 0.12, tw, painting, "painting (style)")
    xr, yr, wr = 4.35, 0.75, 1.05
    hr = thumb(fig, W, H, xr, yr, wr, result, "the picture being made")

    # one frozen network; all three images go through it
    ax.add_patch(FancyBboxPatch((1.45, 0.6), 1.0, 0.9, boxstyle="round,pad=0.02",
                                fc="#e8e8e8", ec="#555555", lw=1.0))
    ax.text(1.95, 1.05, "pretrained CNN\n(VGG-19)\nFROZEN", ha="center", va="center",
            fontsize=FS, fontweight="bold", linespacing=1.15)
    arrow(ax, (1.0, 1.45), (1.45, 1.3))
    arrow(ax, (1.0, 0.45), (1.45, 0.8))
    # the picture goes through the same network (path along the bottom)
    ax.plot([xr + wr / 2, xr + wr / 2, 1.95], [yr - 0.02, 0.36, 0.36], color="#555555",
            lw=0.9)
    arrow(ax, (1.95, 0.36), (1.95, 0.6), color="#555555")
    ax.text(3.35, 0.27, "the picture goes through the same CNN", ha="center", va="center",
            fontsize=6.5, color="#555555")

    # two losses on its maps, then the gradient to the pixels
    ax.text(3.15, 1.62, "content loss:\ndeep maps\nvs the photo's", ha="center", va="center",
            fontsize=FS, color=ARM_BLUE, linespacing=1.1)
    ax.text(3.15, 0.85, "style loss:\nfeature correlations\nvs the painting's", ha="center",
            va="center", fontsize=FS, color=ARM_RED, linespacing=1.1)
    arrow(ax, (2.45, 1.3), (2.68, 1.55), color=ARM_BLUE)
    arrow(ax, (2.45, 0.85), (2.6, 0.85), color=ARM_RED)
    arrow(ax, (3.72, 1.55), (xr - 0.02, yr + hr * 0.75), color=GREEN, lw=1.4)
    arrow(ax, (3.85, 0.95), (xr - 0.02, yr + hr * 0.35), color=GREEN, lw=1.4)
    ax.text(xr + wr / 2, 1.98, "gradient updates\nthe PIXELS", ha="center", va="top",
            fontsize=FS, color=GREEN, fontweight="bold", linespacing=1.05)
    out = FIG_DIR / "style_loop.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info(f"saved {out}")


def gram(f1, f2):
    F = np.stack([f1.ravel(), f2.ravel()])     # 2 maps x 16 positions
    return F @ F.T


def draw_map(ax, x, y, m, cell, color, title):
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((x + j * cell, y - (i + 1) * cell), cell, cell,
                                   fc=color if m[i, j] else "white", ec="#888888", lw=0.5))
    ax.text(x + 2 * cell, y + 0.05, title, ha="center", va="bottom", fontsize=FS)


def draw_gram(ax, x, y, g, cell):
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((x + j * cell, y - (i + 1) * cell), cell, cell,
                                   fc="#f6f0d8", ec="black", lw=0.6))
            ax.text(x + (j + 0.5) * cell, y - (i + 0.5) * cell, str(g[i, j]), ha="center",
                    va="center", fontsize=8.5, fontweight="bold")
    ax.text(x - 0.05, y - 0.5 * cell, "swirl", ha="right", va="center", fontsize=6.5)
    ax.text(x - 0.05, y - 1.5 * cell, "yellow", ha="right", va="center", fontsize=6.5)


def gram_toy(log) -> None:
    rng = np.random.default_rng(SEED)
    g = gram(F_SWIRL, F_YELLOW)
    perm = rng.permutation(16)
    s1, s2 = F_SWIRL.ravel()[perm].reshape(4, 4), F_YELLOW.ravel()[perm].reshape(4, 4)
    g_shuf = gram(s1, s2)
    log.info(f"Gram {g.tolist()}; after shuffling positions {g_shuf.tolist()}")
    if not (np.array_equal(g, EXPECTED_G) and np.array_equal(g_shuf, EXPECTED_G)):
        raise AssertionError("Gram matrices drifted from the slide")
    if np.array_equal(s1, F_SWIRL):
        raise AssertionError("the shuffle left the maps unchanged - pick another seed")

    W, H = 5.0, 1.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    cell, gc = 0.13, 0.22
    for row, (m1, m2, label) in enumerate([(F_SWIRL, F_YELLOW, "two maps of\nthe painting"),
                                           (s1, s2, "same pixels, shuffled")]):
        y = 1.55 - row * 0.82
        ax.text(0.02, y - 2 * cell, label, ha="left", va="center", fontsize=FS,
                fontweight="bold")
        draw_map(ax, 1.25, y, m1, cell, "#4a6fd6", "\"swirl\" map")
        draw_map(ax, 2.05, y, m2, cell, ARM_ORANGE, "\"yellow\" map")
        arrow(ax, (2.68, y - 2 * cell), (3.15, y - 2 * cell))
        ax.text(2.92, y - 2 * cell + 0.05, "Gram", ha="center", va="bottom", fontsize=6.5)
        draw_gram(ax, 3.55, y - 2 * cell + gc, g if row == 0 else g_shuf, gc)
    ax.text(4.12, 1.55 - 2 * cell, "7 swirl cells,\n5 yellow, 5 both", ha="left",
            va="center", fontsize=6.5, linespacing=1.1)
    ax.text(4.12, 0.73 - 2 * cell, "identical:\nthe layout\nis gone", ha="left", va="center",
            fontsize=6.5, linespacing=1.1, color=ARM_RED, fontweight="bold")
    out = FIG_DIR / "gram_toy.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    style_loop(log)
    gram_toy(log)
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
