"""Real figure for the L16 CNN Foundations deck (Section 1, image as a function).

Generates into ml/ch6_cnn/fig/:
  image_as_function.pdf  -- a synthetic grayscale image with three horizontal scan lines
                            and their intensity profiles f(x). Flat regions are constant;
                            edges are jumps; a gradient is a ramp. Sets up "edges = large
                            derivative".

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/image_as_function.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_image_as_function")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "image_as_function.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def make_image(n=120):
    img = np.full((n, n), 40.0)                       # dark background
    img[18:48, 12:52] = 205                           # bright square
    img[78:104, 58:104] = 125                         # mid-gray square
    img[52:66, :] = np.linspace(30, 235, n)[None, :]  # horizontal gradient strip
    yy, xx = np.mgrid[0:n, 0:n]
    img[(xx - 92) ** 2 + (yy - 30) ** 2 < 13 ** 2] = 250   # bright circle
    return img


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    img = make_image()
    lines = [30, 59, 90]
    colors = [ARM_RED, ARM_BLUE, ARM_ORANGE]

    fig = plt.figure(figsize=(9.4, 3.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.35], hspace=0.35, wspace=0.15)
    axim = fig.add_subplot(gs[:, 0])
    axim.imshow(img, cmap="gray", vmin=0, vmax=255)
    for y, c in zip(lines, colors):
        axim.axhline(y, color=c, lw=1.8)
    axim.set_title("image", fontsize=12); axim.axis("off")

    for k, (y, c) in enumerate(zip(lines, colors)):
        ax = fig.add_subplot(gs[k, 1])
        ax.plot(img[y, :], color=c, lw=1.8)
        ax.set_ylim(0, 255); ax.set_ylabel(f"y={y}", fontsize=9)
        ax.grid(alpha=0.15)
        if k < 2:
            ax.set_xticks([])
    ax.set_xlabel("x (pixel)")
    fig.suptitle("An image is a function: intensity f(x) along each scan line", fontsize=12)
    out = FIG_DIR / "image_as_function.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
