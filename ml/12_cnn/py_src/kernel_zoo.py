"""Real figure for the L16 CNN Foundations deck (Section 2, the kernel zoo).

Generates into ml/ch6_cnn/fig/:
  kernel_zoo.pdf  -- the grayscale astronaut test image + 7 classic 3x3 kernels applied by
                     hand (identity, box blur, Gaussian blur, sharpen, Sobel X, Sobel Y,
                     emboss).

The 2D convolution is a naive NumPy double loop on purpose: it is the exact loop HW1
Part A asks students to write, so the deck figure and the homework share one implementation.

Uses skimage's public-domain 'astronaut' test image (strong edges for the filters).
Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/kernel_zoo.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings.
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

# 3x3 kernels (values follow the deck build notes / LMU cnn1-conv2d).
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "box blur": np.ones((3, 3)) / 9.0,
    "Gaussian blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
}


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_kernel_zoo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "kernel_zoo.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def load_gray(max_w=300):
    img = Image.fromarray(data.astronaut()).convert("L")
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    return np.asarray(img, dtype=float)


def conv2d(img, kernel):
    """Naive same-size 2D cross-correlation, zero-padded (the HW1 Part A loop)."""
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(img, pad, mode="constant")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i + k, j:j + k] * kernel)
    return out


def display_norm(name, out):
    """Map a filtered result into a displayable range, per kernel type."""
    if name in ("Sobel X", "Sobel Y"):
        return np.abs(out)                       # edges: show magnitude
    if name == "emboss":
        return np.clip(out + 128, 0, 255)        # emboss centres on mid-gray
    return np.clip(out, 0, 255)


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    gray = load_gray()
    log.info(f"loaded grayscale {gray.shape}")

    fig, axes = plt.subplots(2, 4, figsize=(9.4, 5.2))
    axes = axes.ravel()
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original", fontsize=12); axes[0].axis("off")
    for ax, (name, ker) in zip(axes[1:], KERNELS.items()):
        result = display_norm(name, conv2d(gray, ker))
        ax.imshow(result, cmap="gray")
        ax.set_title(name, fontsize=12); ax.axis("off")
        log.info(f"applied {name}")
    fig.tight_layout()
    out = FIG_DIR / "kernel_zoo.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
