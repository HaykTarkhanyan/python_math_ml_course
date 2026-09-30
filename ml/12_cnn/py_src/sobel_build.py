"""Real figure for the L16 CNN Foundations deck (Section 2, building Sobel).

Generates into ml/ch6_cnn/fig/:
  sobel_build.pdf  -- the astronaut in grayscale: original; the bare adjacent-difference
                      [-1,0,1] (edges, but noisy); Sobel-X (the same difference, smoothed
                      by a [1,2,1] column); and the full gradient magnitude.

Uses skimage's public-domain 'astronaut' test image. Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/sobel_build.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data

SEED = 509

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_sobel_build")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "sobel_build.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def conv2d(img, kernel):
    """Vectorised same-size cross-correlation (sum of shifted, weighted copies)."""
    k = kernel.shape[0]
    pad = k // 2
    p = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img)
    for a in range(k):
        for b in range(k):
            out += kernel[a, b] * p[a:a + img.shape[0], b:b + img.shape[1]]
    return out


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    g = np.asarray(Image.fromarray(data.astronaut()).convert("L").resize((256, 256)), float)

    diff = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], float)   # adjacent difference row
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)   # Sobel X
    sy = sx.T                                                     # Sobel Y
    d = np.abs(conv2d(g, diff))
    gx, gy = conv2d(g, sx), conv2d(g, sy)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    log.info("computed difference, Sobel X/Y, gradient magnitude")

    panels = [(g, "original"),
              (d, "adjacent diff  [-1 0 1]"),
              (np.abs(gx), "Sobel X (smoothed)"),
              (mag, "gradient magnitude")]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im, cmap="gray")
        ax.set_title(t, fontsize=11); ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "sobel_build.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
