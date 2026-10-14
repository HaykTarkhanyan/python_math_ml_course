"""The U-Net road result, re-laid out as input -> output -> truth -> check - for the [51] deck.

Frame "U-Net in action: roads from the sky". The source figure (LMU I2DL, CC BY 4.0,
fig/borrowed/unet_road_result.png) puts the four panels in a 2x2 grid with the INPUT in the
bottom-right corner and tiny labels ("true", "predicted", "coloured satellite", "original
satellite"), so students could not tell what goes in and what comes out. This script cuts the
four panels out (pixel boxes measured on the 592x627 PNG: panel borders at x 10/294/298/582,
y 23/306/319/606, labels trimmed off) and lays them out left to right in reading order.

Generates into ml/12_cnn/fig/:
  road_panels.pdf

Run with the project venv (~2 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/road_panels.py
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
SRC = FIG_DIR / "borrowed" / "unet_road_result.png"

# (x0, y0, x1, y1) in source pixels; the top ~10 px of each panel hold its label
PANELS = {"input": (300, 331, 582, 606), "output": (300, 35, 582, 306),
          "truth": (12, 35, 294, 306), "check": (12, 331, 294, 606)}
TITLES = {"input": "IN: satellite photo\n(RGB, 3 numbers/pixel)",
          "output": "OUT: P(road) per pixel\n(1 number; black = road)",
          "truth": "truth: roads drawn\nby a person",
          "check": "check: yellow hit,\nblue missed, red false"}


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l52_road_panels")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "road_panels.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    t0 = time.time()
    log = setup_logging()
    if not SRC.is_file():
        raise FileNotFoundError(f"missing {SRC}")
    img = plt.imread(SRC)
    if img.shape[:2] != (627, 592):
        raise AssertionError(f"source PNG is {img.shape[:2]}, the crop boxes assume (627, 592)")
    img = img[..., :3]

    fig, axes = plt.subplots(1, 4, figsize=(5.45, 1.62))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.79, bottom=0.01, wspace=0.16)
    for ax, key in zip(axes, ("input", "output", "truth", "check")):
        x0, y0, x1, y1 = PANELS[key]
        crop = img[y0:y1, x0:x1]
        if crop.mean() > 0.995 or crop.mean() < 0.005:
            raise AssertionError(f"panel {key} came out blank - crop box is off")
        ax.imshow(crop)
        ax.set_title(TITLES[key], fontsize=6.8, pad=2, linespacing=1.05,
                     fontweight="bold" if key in ("input", "output") else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        log.info(f"{key}: crop {x1 - x0}x{y1 - y0} px, mean {crop.mean():.3f}")
    # "U-Net" arrow between IN and OUT, "compare" bracket between OUT/truth and check
    for (a, b, text) in ((0, 1, "U-Net"), (2, 3, "compare")):
        p0 = axes[a].get_position(); p1 = axes[b].get_position()
        y = (p0.y0 + p0.y1) / 2
        fig.patches.append(FancyArrowPatch((p0.x1 + 0.002, y), (p1.x0 - 0.002, y),
                                           transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=8, color=ARM_RED, lw=1.2))
        fig.text((p0.x1 + p1.x0) / 2, y + 0.05, text, ha="center", va="bottom", fontsize=6.5,
                 color=ARM_RED, rotation=90)
    out = FIG_DIR / "road_panels.pdf"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    log.info(f"saved {out}; done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
