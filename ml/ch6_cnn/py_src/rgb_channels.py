"""Real figure for the L16 CNN Foundations deck (Section 1, a pixel is three numbers).

Generates into ml/ch6_cnn/fig/:
  rgb_channels.pdf  -- the astronaut portrait + its R, G, B channels (each tinted in its
                       own color). One pixel = (R, G, B), each 0-255.

Uses skimage's public-domain 'astronaut' test image (NASA). Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/rgb_channels.py
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
    logger = logging.getLogger("l16_rgb_channels")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "rgb_channels.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    arr = np.asarray(Image.fromarray(data.astronaut()).resize((256, 256)))

    R = arr.copy(); R[:, :, [1, 2]] = 0
    G = arr.copy(); G[:, :, [0, 2]] = 0
    B = arr.copy(); B[:, :, [0, 1]] = 0

    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9))
    for ax, im, t in zip(axes, [arr, R, G, B],
                         ["RGB image", "R channel", "G channel", "B channel"]):
        ax.imshow(im); ax.set_title(t, fontsize=12); ax.axis("off")
    fig.suptitle("Every pixel is three numbers: (R, G, B), each 0-255", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "rgb_channels.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
