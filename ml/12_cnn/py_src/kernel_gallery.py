"""Kernel gallery for the L16 CNN Foundations deck (Section 2, after "The kernel zoo").

Each figure shows four kernels as colored grids of their actual values (red > 0, white = 0,
blue < 0) above what each one does to the same photo.

Generates into ml/12_cnn/fig/:
  kernel_gallery_blur.pdf   -- identity, box blur 5x5, Gaussian 5x5, motion blur 5x5
  kernel_gallery_edges.pdf  -- Sobel X, Laplacian, sharpen, emboss
  kernel_gallery_gabor.pdf  -- four Gabor kernels (0, 45, 90, 135 degrees) and their responses

Convolution is scipy.ndimage.correlate (no kernel flip, exactly what nn.Conv2d computes).
Image: skimage's public-domain 'astronaut' (grayscale) and 'brick' test images.

Run (~30 s on CPU, measured 2026-10-02; a cold first import of skimage can add a minute):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/kernel_gallery.py

Drawn at slide size (5.4 in wide = the full text width), fonts 7-9 pt.
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import correlate
from skimage import color, data

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"
DIVERGING = LinearSegmentedColormap.from_list("arm_div", [ARM_BLUE, "white", ARM_RED])

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

FIGSIZE = (5.4, 2.3)          # its size on the slide


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("kernel_gallery")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "kernel_gallery.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def face_crop() -> np.ndarray:
    """Grayscale astronaut, 0-255 floats, cropped tight on the face so a 5x5 blur is visible."""
    gray = color.rgb2gray(data.astronaut()) * 255.0
    return gray[55:165, 185:295]


def gabor(theta_deg: float, size: int = 21, sigma: float = 4.0, lam: float = 8.0,
          gamma: float = 0.5) -> np.ndarray:
    """g(x, y) = exp(-(x'^2 + gamma^2 y'^2) / (2 sigma^2)) * cos(2 pi x' / lam), zero mean."""
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    t = np.deg2rad(theta_deg)
    xr = x * np.cos(t) + y * np.sin(t)
    yr = -x * np.sin(t) + y * np.cos(t)
    g = np.exp(-(xr ** 2 + (gamma * yr) ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xr / lam)
    return g - g.mean()            # zero mean: flat regions give 0, like an edge kernel


def draw_kernel(ax, k: np.ndarray, title: str, numbers: np.ndarray | None):
    """Kernel as a colored grid; `numbers` (same shape) are printed in the cells if given."""
    m = np.abs(k).max()
    ax.imshow(k, cmap=DIVERGING, vmin=-m, vmax=m)
    n = k.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    if numbers is not None:
        ax.grid(which="minor", color="#999999", lw=0.4)
        fs = 8 if n <= 3 else 6.5
        for (i, j), v in np.ndenumerate(numbers):
            dark = abs(k[i, j]) > 0.6 * m
            ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=fs,
                    color="white" if dark else "black")
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=8, pad=3)


def gallery(rows, img, out: Path, log, show):
    """rows: list of (title, kernel, numbers-or-None, note); show(response) -> (array, kwargs)."""
    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE,
                             gridspec_kw=dict(hspace=0.12, wspace=0.18))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.11)
    for c, (title, k, numbers, note) in enumerate(rows):
        draw_kernel(axes[0, c], k, title, numbers)
        resp = correlate(img, k, mode="reflect")
        arr, kw = show(c, resp)
        axes[1, c].imshow(arr, cmap="gray", **kw)
        axes[1, c].axis("off")
        axes[1, c].text(0.5, -0.05, note, transform=axes[1, c].transAxes, ha="center",
                        va="top", fontsize=7, color="#333333")
        log.info(f"{out.stem}: {title} sum={k.sum():.3f} resp range [{resp.min():.1f}, {resp.max():.1f}]")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)   # no dead space
    plt.close(fig)
    log.info(f"saved {out}")


def blur_gallery(log):
    img = face_crop()
    ident = np.zeros((3, 3)); ident[1, 1] = 1
    box = np.ones((5, 5))
    b = np.array([1, 4, 6, 4, 1], float)
    gauss = np.outer(b, b)                                   # binomial: sums to 256
    motion = np.eye(5)
    assert gauss.sum() == 256
    rows = [
        ("identity", ident, ident, "a copy"),
        ("box blur  (/25)", box / 25, box, "plain average"),
        ("Gaussian  (/256)", gauss / 256, gauss, "centre weighted: softer"),
        ("motion blur  (/5)", motion / 5, motion, "averaged along a line"),
    ]
    gallery(rows, img, FIG_DIR / "kernel_gallery_blur.pdf", log,
            lambda c, r: (r, dict(vmin=0, vmax=255)))


def edge_gallery(log):
    img = face_crop()
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float)
    emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float)
    assert np.allclose(sharpen, np.eye(3)[1][:, None] * np.eye(3)[1][None, :] - laplace)
    rows = [
        ("Sobel X", sobel_x, sobel_x, "vertical edges, signed"),
        ("Laplacian", laplace, laplace, "edges, all directions"),
        ("sharpen", sharpen, sharpen, "identity - Laplacian"),
        ("emboss", emboss, emboss, "image + relief"),
    ]

    def show(c, r):
        if c in (0, 1):                     # signed responses: mid-gray = 0
            m = np.percentile(np.abs(r), 98)
            return r, dict(vmin=-m, vmax=m)
        if c == 2:
            return np.clip(r, 0, 255), dict(vmin=0, vmax=255)
        return np.clip(r, 0, 255), dict(vmin=0, vmax=255)      # emboss sums to 1: no offset

    gallery(rows, img, FIG_DIR / "kernel_gallery_edges.pdf", log, show)


def gabor_gallery(log):
    """Like gallery(), plus the input image in a first column (this frame uses a new photo)."""
    img = data.brick()[96:352, 96:352].astype(float)    # already grayscale, 0-255
    angles = [(0, "| lines light up"), (45, "/ lines (few here)"),
              (90, "- lines light up"), (135, "\\ lines (few here)")]
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 5, width_ratios=[1.25, 1, 1, 1, 1], hspace=0.12, wspace=0.18,
                          left=0.01, right=0.99, top=0.91, bottom=0.11)
    ax_in = fig.add_subplot(gs[:, 0])
    ax_in.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax_in.set_title("input: a brick wall", fontsize=8, pad=3)
    ax_in.axis("off")
    out = FIG_DIR / "kernel_gallery_gabor.pdf"
    for c, (a, note) in enumerate(angles, start=1):
        k = gabor(a)
        draw_kernel(fig.add_subplot(gs[0, c]), k, f"Gabor, {a}\u00b0", None)
        resp = np.abs(correlate(img, k, mode="reflect"))
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(resp, cmap="gray", vmin=0, vmax=np.percentile(resp, 99.5))
        ax.axis("off")
        ax.text(0.5, -0.05, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color="#333333")
        log.info(f"gabor {a} deg: mean |response| {resp.mean():.1f}")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)   # no dead space
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    t0 = time.perf_counter()
    FIG_DIR.mkdir(exist_ok=True)
    blur_gallery(log)
    edge_gallery(log)
    gabor_gallery(log)
    log.info(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
