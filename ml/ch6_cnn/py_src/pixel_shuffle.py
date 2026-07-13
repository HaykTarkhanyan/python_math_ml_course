"""Real figures for the L16 CNN Foundations deck (Section 1).

Generates into ml/ch6_cnn/fig/:
  image_numbers.pdf  -- grayscale pomegranate + a zoomed 8x8 patch with raw 0-255 values,
                        for the "an image is a grid of numbers" frame.
  pixel_shuffle.pdf  -- the grayscale photo vs the same pixels under a fixed permutation,
                        for the "an MLP cannot see structure" frame.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/pixel_shuffle.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings.
The pomegranate photo is required; the script fails loudly if it is missing.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

SEED = 509

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
SRC_PHOTO = FIG_DIR / "src_pomegranate.jpg"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_pixel_shuffle")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "pixel_shuffle.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def load_gray(max_w=320):
    if not SRC_PHOTO.exists():
        raise FileNotFoundError(
            f"Required photo missing: {SRC_PHOTO}. Drop the pomegranate image there "
            "(CNN chapter plan open question 5)."
        )
    img = Image.open(SRC_PHOTO).convert("L")
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def fig_image_numbers(gray, log):
    """Full grayscale photo with a boxed patch, next to that patch as raw numbers."""
    H, W = gray.shape
    p = 8
    r0, c0 = int(H * 0.60), int(W * 0.50)          # texture-rich arils region
    patch = gray[r0:r0 + p, c0:c0 + p]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.9))
    axL.imshow(gray, cmap="gray", vmin=0, vmax=255)
    axL.add_patch(Rectangle((c0, r0), p, p, ec="#D90012", fc="none", lw=1.8))
    axL.set_title("A grayscale image", fontsize=12)
    axL.axis("off")
    axR.imshow(patch, cmap="gray", vmin=0, vmax=255)
    for i in range(p):
        for j in range(p):
            v = int(patch[i, j])
            axR.text(j, i, str(v), ha="center", va="center", fontsize=7.5,
                     color="white" if v < 128 else "black")
    axR.set_title(f"{p}x{p} patch: just numbers, 0-255", fontsize=12)
    axR.set_xticks([]); axR.set_yticks([])
    fig.tight_layout()
    out = FIG_DIR / "image_numbers.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


def fig_pixel_shuffle(gray, log):
    """Original vs a fixed permutation of all pixels: identical to a flatten-then-MLP."""
    rng = np.random.default_rng(SEED)
    flat = gray.ravel()
    shuffled = flat[rng.permutation(flat.size)].reshape(gray.shape)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.9))
    axL.imshow(gray, cmap="gray", vmin=0, vmax=255)
    axL.set_title("Original", fontsize=12); axL.axis("off")
    axR.imshow(shuffled, cmap="gray", vmin=0, vmax=255)
    axR.set_title("Same pixels, one fixed shuffle", fontsize=12); axR.axis("off")
    fig.suptitle("A flatten-then-MLP sees these two as equally learnable", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "pixel_shuffle.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    gray = load_gray()
    log.info(f"loaded {SRC_PHOTO.name} -> grayscale {gray.shape}")
    fig_image_numbers(gray, log)
    fig_pixel_shuffle(gray, log)
    log.info("done")


if __name__ == "__main__":
    main()
