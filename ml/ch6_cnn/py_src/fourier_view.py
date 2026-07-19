"""Real figure for the L16 CNN Foundations deck (Section 2, the frequency aside).

Generates into ml/ch6_cnn/fig/:
  fourier_view.pdf  -- the grayscale astronaut, its 2D Fourier magnitude spectrum, and two
                       frequency-domain filters reconstructed back to image space:
                         low-pass  (keep the central disk of frequencies)  -> blur
                         high-pass (delete the central disk)               -> edges
                       This is the convolution theorem made visual: "multiply in frequency"
                       is just masking the spectrum. A box blur is a low-pass filter; a Sobel
                       edge detector is a high-pass filter - the same kernels from the zoo,
                       seen from the frequency side.

Uses skimage's public-domain 'astronaut' test image (same image as kernel_zoo.py).
Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/fourier_view.py

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

SIZE = 256          # square crop keeps the spectrum symmetric and clean
R_LOW = 28          # low-pass: keep frequencies within this radius (px) of the centre
R_HIGH = 18          # high-pass: delete frequencies within this radius of the centre


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_fourier_view")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "fourier_view.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def load_gray(size=SIZE):
    img = Image.fromarray(data.astronaut()).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=float)


def radial_mask(shape, radius, inside=True):
    """Boolean mask: True inside (or outside) a centred disk of the given radius."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(yy - h / 2.0, xx - w / 2.0)
    return dist <= radius if inside else dist > radius


def filter_in_frequency(img, mask):
    """Multiply the (shifted) spectrum by mask, invert back to image space."""
    fsh = np.fft.fftshift(np.fft.fft2(img))
    out = np.fft.ifft2(np.fft.ifftshift(fsh * mask))
    return np.real(out)


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    gray = load_gray()
    log.info(f"loaded grayscale {gray.shape}")

    fsh = np.fft.fftshift(np.fft.fft2(gray))
    spectrum = np.log1p(np.abs(fsh))          # log so the huge DC term does not wash it out

    low = filter_in_frequency(gray, radial_mask(gray.shape, R_LOW, inside=True))
    high = filter_in_frequency(gray, radial_mask(gray.shape, R_HIGH, inside=False))
    log.info(f"low-pass radius {R_LOW}px, high-pass radius {R_HIGH}px")

    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.2))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("image  $f$", fontsize=12)
    axes[1].imshow(spectrum, cmap="magma")
    axes[1].set_title(r"its spectrum  $\mathcal{F}\{f\}$", fontsize=12)
    axes[2].imshow(np.clip(low, 0, 255), cmap="gray")
    axes[2].set_title("keep low freq  $\\to$  blur", fontsize=12)
    axes[3].imshow(np.abs(high), cmap="gray")
    axes[3].set_title("keep high freq  $\\to$  edges", fontsize=12)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "fourier_view.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
