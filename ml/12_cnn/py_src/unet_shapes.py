"""A small U-Net, drawn with its tensor shapes - for the [51] Vision Tasks deck.

Frame "U-Net: the segmentation workhorse": the shrink-then-grow-back idea with real
shapes for a 224x224 photo, three pooling steps deep (the original U-Net goes four deep,
572x572 input, unpadded convs - Ronneberger et al. 2015; this is the padded, shallower
version students would build). Each block = two 3x3 convs (padding 1), so only pooling
and transposed convs change the height and width.

  encoder  224x224x64 -> 112x112x128 -> 56x56x256 -> bottleneck 28x28x512
  decoder  56x56x256 -> 112x112x128 -> 224x224x64 -> 1x1 conv -> 224x224xK

The shapes are computed by pushing a zero tensor through the matching torch layers and
asserted, so the labels cannot drift from what PyTorch would produce.

Generates into ml/12_cnn/fig/:
  unet_shapes.pdf

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/unet_shapes.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import torch
from torch import nn

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
UP_GREEN = "#008c46"
SKIP_GRAY = "#8a8a8a"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

CHANNELS = [64, 128, 256]      # encoder levels; bottleneck has 512
FS = 7.5


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l52_unet_shapes")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "unet_shapes.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU())


def traced_shapes(log: logging.Logger) -> dict:
    """Push a zero photo through a 3-level U-Net and record (H, W, C) at every stage."""
    torch.manual_seed(SEED)
    k = 2                                   # classes, e.g. fruit / background
    x = torch.zeros(1, 3, 224, 224)
    shapes, skips, c_prev = {}, [], 3
    with torch.no_grad():
        for lvl, c in enumerate(CHANNELS):
            x = block(c_prev, c)(x); c_prev = c
            shapes[f"enc{lvl}"] = tuple(x.shape[2:]) + (x.shape[1],)
            skips.append(x)
            x = nn.MaxPool2d(2)(x)
        x = block(c_prev, 512)(x); c_prev = 512
        shapes["bottleneck"] = tuple(x.shape[2:]) + (x.shape[1],)
        for lvl in reversed(range(len(CHANNELS))):
            c = CHANNELS[lvl]
            x = nn.ConvTranspose2d(c_prev, c, 2, stride=2)(x)
            x = torch.cat([skips[lvl], x], dim=1)          # the skip: concatenate
            x = block(2 * c, c)(x); c_prev = c
            shapes[f"dec{lvl}"] = tuple(x.shape[2:]) + (x.shape[1],)
        x = nn.Conv2d(c_prev, k, 1)(x)
        shapes["out"] = tuple(x.shape[2:]) + (x.shape[1],)
    for name, s in shapes.items():
        log.info(f"{name:10s} {s[0]}x{s[1]}x{s[2]}")
    expected = {"enc0": (224, 224, 64), "enc1": (112, 112, 128), "enc2": (56, 56, 256),
                "bottleneck": (28, 28, 512), "dec2": (56, 56, 256),
                "dec1": (112, 112, 128), "dec0": (224, 224, 64), "out": (224, 224, k)}
    if shapes != expected:
        raise AssertionError(f"U-Net shapes drifted: {shapes}")
    return shapes


def draw(shapes: dict, log: logging.Logger) -> None:
    W, H = 5.45, 2.12
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ys = {0: 1.62, 1: 1.14, 2: 0.73, 3: 0.37}               # level -> centre height
    hs = {0: 0.56, 1: 0.40, 2: 0.29, 3: 0.20}               # block height ~ spatial size
    ws = {64: 0.09, 128: 0.13, 256: 0.17, 512: 0.22}       # block width ~ channels
    xs_enc = {0: 1.05, 1: 1.42, 2: 1.79}
    xs_dec = {0: 4.05, 1: 3.68, 2: 3.31}
    x_bot = 2.55

    def rect(xc, lvl, c, color):
        w, h = ws[c], hs[lvl]
        ax.add_patch(Rectangle((xc - w / 2, ys[lvl] - h / 2), w, h, fc=color,
                               ec="black", lw=0.6))
        return xc - w / 2, xc + w / 2, ys[lvl] - h / 2, ys[lvl] + h / 2

    def arrow(p, q, color, lw=1.0, style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7,
                                     color=color, lw=lw, shrinkA=1, shrinkB=1))

    def lab(s):
        h, w, c = s
        return f"{h}x{w}x{c}"

    enc_box, dec_box = {}, {}
    for lvl in range(3):
        c = CHANNELS[lvl]
        enc_box[lvl] = rect(xs_enc[lvl], lvl, c, "#cfdcf2")
        dec_box[lvl] = rect(xs_dec[lvl], lvl, c, "#cfeedd")
    bot = rect(x_bot, 3, 512, "#f6dfb0")

    # labels: encoder to the left, decoder to the right, top level above
    ax.text(xs_enc[0], enc_box[0][3] + 0.05, lab(shapes["enc0"]), ha="center",
            va="bottom", fontsize=FS)
    ax.text(xs_dec[0], dec_box[0][3] + 0.05, lab(shapes["dec0"]), ha="center",
            va="bottom", fontsize=FS)
    for lvl in (1, 2):
        ax.text(enc_box[lvl][0] - 0.05, ys[lvl], lab(shapes[f"enc{lvl}"]), ha="right",
                va="center", fontsize=FS)
        ax.text(dec_box[lvl][1] + 0.05, ys[lvl], lab(shapes[f"dec{lvl}"]), ha="left",
                va="center", fontsize=FS)
    ax.text(x_bot, bot[2] - 0.04, lab(shapes["bottleneck"]) + "  bottleneck",
            ha="center", va="top", fontsize=FS)

    # input and output
    ax.text(0.02, ys[0], "photo\n224x224x3", ha="left", va="center", fontsize=FS,
            linespacing=1.1)
    arrow((0.55, ys[0]), (enc_box[0][0], ys[0]), "black")
    out_x0 = 4.5
    ax.add_patch(Rectangle((out_x0, ys[0] - hs[0] / 2), 0.035, hs[0], fc=ARM_ORANGE,
                           ec="black", lw=0.6))
    arrow((dec_box[0][1], ys[0]), (out_x0, ys[0]), "black")
    ax.text(out_x0 + 0.08, ys[0], f"{shapes['out'][0]}x{shapes['out'][1]}xK\n"
            "a class score\nper pixel", ha="left", va="center", fontsize=FS,
            linespacing=1.1)
    ax.text((dec_box[0][1] + out_x0) / 2, ys[0] + 0.05, "1x1", ha="center",
            va="bottom", fontsize=6.5)

    # pooling (down), transposed conv (up), skips (across)
    for lvl in range(2):
        arrow((xs_enc[lvl], enc_box[lvl][2]), (enc_box[lvl + 1][0] + 0.02, ys[lvl + 1]),
              ARM_RED, lw=1.2)
        arrow((dec_box[lvl + 1][1] - 0.02, ys[lvl + 1]), (xs_dec[lvl], dec_box[lvl][2]),
              UP_GREEN, lw=1.2)
    arrow((xs_enc[2], enc_box[2][2]), (bot[0], ys[3]), ARM_RED, lw=1.2)
    arrow((bot[1], ys[3]), (xs_dec[2], dec_box[2][2]), UP_GREEN, lw=1.2)
    for lvl in range(3):
        arrow((enc_box[lvl][1], ys[lvl]), (dec_box[lvl][0], ys[lvl]), SKIP_GRAY, lw=0.9)
    ax.text((enc_box[0][1] + dec_box[0][0]) / 2, ys[0] + 0.04,
            "skip: copy the sharp map across, concatenate", ha="center", va="bottom",
            fontsize=6.8, color="#555555")

    # legend
    lx, ly = 3.3, 0.34           # bottom-right corner, below the decoder's lowest block
    for i, (color, text) in enumerate([
            (ARM_RED, "max-pool: halve height and width"),
            (UP_GREEN, "transposed conv: double them"),
            (SKIP_GRAY, "skip connection")]):
        y = ly - i * 0.13
        arrow((lx, y), (lx + 0.22, y), color, lw=1.2)
        ax.text(lx + 0.27, y, text, ha="left", va="center", fontsize=6.8)

    out = FIG_DIR / "unet_shapes.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    t0 = time.time()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    draw(traced_shapes(log), log)
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
