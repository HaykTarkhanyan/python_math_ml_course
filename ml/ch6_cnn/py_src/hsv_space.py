"""Real figure for the L16 CNN Foundations deck (Section 1, HSV colour space).

Generates into ml/ch6_cnn/fig/:
  hsv_space.pdf  -- top row: the astronaut portrait decomposed into H, S, V.
                    bottom row: what the axes buy you - two ways to "darken" in RGB
                    (one shifts hue, one does not) vs moving V, plus a hue rotation.

The bottom row backs the slide's claim with measured numbers, printed to the log:
  - subtracting a constant in RGB shifts hue by a measurable amount
  - scaling V in HSV is exactly a uniform RGB multiply (asserted, not asserted-by-hope)

Uses skimage's public-domain 'astronaut' test image (NASA), same source image as
rgb_channels.pdf / kernel_zoo.pdf so the section stays visually consistent. Run with the
project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/hsv_space.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from skimage import data

SEED = 509

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

DARKEN_SUBTRACT = 110 / 255.0   # the naive "make it darker" in RGB
DARKEN_SCALE = 0.55             # the correct one, and what moving V does
HUE_ROTATION = 60.0             # degrees
SAT_FLOOR = 0.20                # below this, hue is numerically meaningless


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_hsv_space")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "hsv_space.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def circular_hue_shift(h_ref: np.ndarray, h_new: np.ndarray) -> np.ndarray:
    """Absolute hue difference in degrees, respecting the 0/360 wrap."""
    d = np.abs(h_new - h_ref) * 360.0
    return np.minimum(d, 360.0 - d)


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)

    rgb = np.asarray(Image.fromarray(data.astronaut()).resize((256, 256))) / 255.0
    hsv = rgb_to_hsv(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    log.info(f"image {rgb.shape}, H in [{H.min():.3f},{H.max():.3f}], "
             f"S mean {S.mean():.3f}, V mean {V.mean():.3f}")

    # --- the two RGB "darken" candidates -------------------------------------------------
    sub = np.clip(rgb - DARKEN_SUBTRACT, 0.0, 1.0)
    mul = np.clip(rgb * DARKEN_SCALE, 0.0, 1.0)

    # --- the HSV move: change V only -----------------------------------------------------
    hsv_dark = hsv.copy()
    hsv_dark[..., 2] = V * DARKEN_SCALE
    v_scaled = hsv_to_rgb(hsv_dark)

    # Scaling V is exactly a uniform RGB multiply. Assert it rather than claim it.
    max_gap = float(np.abs(v_scaled - mul).max())
    log.info(f"max |HSV V-scale - RGB multiply| = {max_gap:.2e}")
    assert max_gap < 1e-6, (
        f"V-scaling and uniform RGB multiply should coincide, max gap {max_gap:.2e}")

    # How badly does the naive subtract move the hue? Only ask where hue is meaningful.
    mask = S > SAT_FLOOR
    shift_sub = circular_hue_shift(H, rgb_to_hsv(sub)[..., 0])[mask]
    shift_mul = circular_hue_shift(H, rgb_to_hsv(mul)[..., 0])[mask]
    crushed = float((sub.max(axis=2) == 0).mean() * 100)
    log.info(f"hue shift over {mask.mean()*100:.1f}% of pixels with S>{SAT_FLOOR}: "
             f"subtract mean {shift_sub.mean():.1f} deg / max {shift_sub.max():.1f} deg; "
             f"multiply mean {shift_mul.mean():.2e} deg")
    log.info(f"pixels crushed to pure black by the subtract: {crushed:.1f}%")

    # --- the move RGB has no clean expression for: rotate every hue --------------------
    hsv_rot = hsv.copy()
    hsv_rot[..., 0] = (H + HUE_ROTATION / 360.0) % 1.0
    rotated = hsv_to_rgb(hsv_rot)

    # --- figure ---------------------------------------------------------------------------
    # Wide-and-short: this lands on a 16:9 frame that also carries text and a callout box,
    # so vertical space is the binding constraint. No suptitle - the slide body says it.
    fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.5))

    top = [
        (rgb, "RGB image", None),
        (H, "H  hue (which colour)", "hsv"),
        (S, "S  saturation (how pure)", "gray"),
        (V, "V  value (how bright)", "gray"),
    ]
    for ax, (im, title, cmap) in zip(axes[0], top):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1) if cmap else ax.imshow(im)
        ax.set_title(title, fontsize=11); ax.axis("off")

    bottom = [
        (rgb, "original"),
        (sub, f"RGB: subtract {int(DARKEN_SUBTRACT*255)}\n"
              f"hue moves {shift_sub.mean():.0f} deg, {crushed:.0f}% crushed to black"),
        (v_scaled, f"HSV: V x {DARKEN_SCALE}\nhue and saturation cannot move"),
        (rotated, f"HSV: H + {int(HUE_ROTATION)} deg\nsame brightness, new colour"),
    ]
    for ax, (im, title) in zip(axes[1], bottom):
        ax.imshow(im); ax.set_title(title, fontsize=9.5); ax.axis("off")

    fig.tight_layout(h_pad=2.8)
    out = FIG_DIR / "hsv_space.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
