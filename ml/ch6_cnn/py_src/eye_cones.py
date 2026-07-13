"""Real figure for the L16 CNN Foundations deck (Section 1, how the eye sees color).

Generates into ml/ch6_cnn/fig/:
  eye_cones.pdf  -- approximate S/M/L cone sensitivity curves vs wavelength. Three cone
                    types -> three numbers per point -> RGB.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/eye_cones.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 509
ARM_BLUE, ARM_RED, ARM_GREEN = "#0033A0", "#D90012", "#008C46"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_eye_cones")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "eye_cones.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def gaussian(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    wl = np.linspace(400, 700, 400)
    S = gaussian(wl, 445, 22)     # short  -> blue
    M = gaussian(wl, 540, 34)     # medium -> green
    L = gaussian(wl, 568, 38)     # long   -> red

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for curve, color, lab in [(S, ARM_BLUE, "S cones (blue)"),
                              (M, ARM_GREEN, "M cones (green)"),
                              (L, ARM_RED, "L cones (red)")]:
        ax.plot(wl, curve, color=color, lw=2.2, label=lab)
        ax.fill_between(wl, curve, alpha=0.08, color=color)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("relative sensitivity")
    ax.set_title("Three cone types -> three numbers per point (R, G, B)")
    ax.set_xlim(400, 700); ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = FIG_DIR / "eye_cones.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
