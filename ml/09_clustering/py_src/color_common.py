"""Shared helpers for the 33_color_spaces figure scripts.

Every figure in the colour-spaces deck uses the same source image (the Saryan landscape
that the image-compression practical clusters) and the same logging setup, so both live
here instead of being copy-pasted into seven scripts.

Not a general-purpose module - it exists for `py_src/color_*.py`, `rgb_channels.py`,
`hsv_space.py` and `eye_cones.py` in this folder.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 509

# Armenian flag palette, the repo-wide fallback for 3+ colour charts.
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
ARM_GREEN = "#008C46"

HERE = Path(__file__).resolve()
PY_SRC_DIR = HERE.parent
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
IMG_PATH = CH_DIR / "img" / "saryan_mountains.jpg"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging(name: str) -> logging.Logger:
    """Console + file logging, per the repo convention. Log lands in logs/<name>.log."""
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def load_saryan(size: int = 320, as_float: bool = False) -> np.ndarray:
    """The Saryan landscape, square-resized. uint8 0-255, or float 0-1 if as_float.

    Same painting the image-compression practical quantizes, so the deck and the
    notebook are visibly about the same pixels.
    """
    if not IMG_PATH.exists():
        raise FileNotFoundError(
            f"source image missing: {IMG_PATH}. The colour-spaces figures need it; "
            "it ships with the clustering chapter under img/."
        )
    arr = np.asarray(Image.open(IMG_PATH).convert("RGB").resize((size, size)))
    return arr.astype(np.float64) / 255.0 if as_float else arr


def srgb_to_linear(s: np.ndarray) -> np.ndarray:
    """sRGB (0-1, gamma-encoded) -> linear light. IEC 61966-2-1 / ICC spec constants."""
    s = np.asarray(s, dtype=np.float64)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(u: np.ndarray) -> np.ndarray:
    """Linear light (0-1) -> sRGB gamma-encoded. Inverse of srgb_to_linear."""
    u = np.asarray(u, dtype=np.float64)
    return np.where(u <= 0.0031308, u * 12.92, 1.055 * u ** (1 / 2.4) - 0.055)


def ensure_fig_dir() -> Path:
    FIG_DIR.mkdir(exist_ok=True)
    return FIG_DIR
