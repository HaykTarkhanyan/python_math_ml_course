"""Real figure for the L16 CNN Foundations deck (Section 2, the Taylor -> Fourier bridge).

Generates into ml/ch6_cnn/fig/:
  taylor_vs_fourier.pdf  -- side by side, the same idea in two bases:
      left  (Taylor):  cos(x) rebuilt from a sum of POWERS  (1, x^2, x^4, ...), better near 0.
      right (Fourier): a square wave rebuilt from a sum of WAVES (sin x, sin 3x, sin 5x, ...),
                       better everywhere as harmonics are added (the classic Gibbs picture).
  Students already know Taylor; this frames Fourier as the same move with a different set of
  building blocks - blocks labelled by frequency, which is what we later filter.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/taylor_vs_fourier.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag palette for 3+ series (red/blue/orange).
"""

import logging
from math import factorial
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
    logger = logging.getLogger("l16_taylor_vs_fourier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "taylor_vs_fourier.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def taylor_cos(x, degree):
    """Partial Taylor series of cos(x) about 0 up to the given even degree."""
    out = np.zeros_like(x)
    for n in range(0, degree + 1, 2):
        out += (-1) ** (n // 2) * x ** n / factorial(n)
    return out


def fourier_square(x, n_harmonics):
    """Partial Fourier series of a unit square wave: (4/pi) sum_{odd k} sin(kx)/k."""
    out = np.zeros_like(x)
    for k in range(1, 2 * n_harmonics, 2):
        out += np.sin(k * x) / k
    return (4.0 / np.pi) * out


def fig_taylor(ax, log):
    x = np.linspace(-4.2, 4.2, 600)
    ax.plot(x, np.cos(x), color="black", lw=2.6, label=r"$\cos x$ (target)")
    for deg, col in [(2, ARM_RED), (4, ARM_ORANGE), (8, ARM_BLUE)]:
        ax.plot(x, taylor_cos(x, deg), color=col, lw=1.8, label=f"degree {deg}")
    ax.set_ylim(-2.2, 1.8)
    ax.set_title("Taylor: sum of POWERS  $1, x^2, x^4, \\ldots$", fontsize=12)
    ax.axhline(0, color="0.7", lw=0.6); ax.axvline(0, color="0.7", lw=0.6)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    ax.text(0.5, -0.14, "great near a point, drifts off far away",
            transform=ax.transAxes, ha="center", fontsize=9, color="0.35")
    log.info("drew Taylor panel")


def fig_fourier(ax, log):
    x = np.linspace(-np.pi, np.pi, 1200)
    square = np.sign(np.sin(x))
    ax.plot(x, square, color="black", lw=2.6, label="square wave (target)")
    for n, col in [(1, ARM_RED), (3, ARM_ORANGE), (12, ARM_BLUE)]:
        lab = "1 wave" if n == 1 else f"{n} waves"
        ax.plot(x, fourier_square(x, n), color=col, lw=1.6, label=lab)
    ax.set_ylim(-1.6, 1.6)
    ax.set_title("Fourier: sum of WAVES  $\\sin x, \\sin 3x, \\ldots$", fontsize=12)
    ax.axhline(0, color="0.7", lw=0.6); ax.axvline(0, color="0.7", lw=0.6)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    ax.text(0.5, -0.14, "more waves = better fit, everywhere",
            transform=ax.transAxes, ha="center", fontsize=9, color="0.35")
    log.info("drew Fourier panel")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    fig_taylor(axes[0], log)
    fig_fourier(axes[1], log)
    fig.suptitle("Same idea, different building blocks: break a function into simple pieces",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "taylor_vs_fourier.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
