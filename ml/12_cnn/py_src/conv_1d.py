"""Real figures for the L16 CNN Foundations deck (Section 2, the 1D case).

Generates into ml/12_cnn/fig/:
  moving_average.pdf  -- a noisy 1D signal and its 5-point moving average (convolution with
                         a box kernel = smoothing).
  dice_conv.pdf       -- P(die A) * P(die B) = P(sum): two uniform dice distributions and
                         their convolution, the triangular sum distribution. Convolution is
                         not just an image trick - it is how you add two random variables.

Run with the project venv:
    ./ma/Scripts/python.exe ml/12_cnn/py_src/conv_1d.py

2026-10-01: redrawn at its slide size (figsize x 0.34, fonts mapped to 7-10.5 pt)
so the labels are readable on the slide; they rendered at 2-5 pt before.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8})
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
    logger = logging.getLogger("l16_conv_1d")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "conv_1d.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def fig_moving_average(log):
    rng = np.random.default_rng(SEED)
    x = np.arange(80)
    clean = 3 + 2 * np.sin(x / 7.0)
    noisy = clean + rng.normal(0, 0.7, x.size)
    k = np.ones(5) / 5.0
    smooth = np.convolve(noisy, k, mode="same")

    fig, ax = plt.subplots(figsize=(3.55, 1.75))   # moving average: its own 3.55 in slot
    ax.plot(x, noisy, color="#b8c4d9", lw=0.7, label="noisy signal")
    ax.plot(x, smooth, color=ARM_BLUE, lw=1.4, label="5-point moving average")
    ax.set_xlabel("position", fontsize=7.5, labelpad=1); ax.set_ylabel("value", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_title("box kernel (1/5 each) = moving average", fontsize=8.5, pad=2)
    ax.legend(loc="upper right", fontsize=7.45); ax.grid(alpha=0.2)
    fig.tight_layout()
    out = FIG_DIR / "moving_average.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


def fig_dice(log):
    die = np.ones(6) / 6.0                       # uniform on 1..6
    summ = np.convolve(die, die)                 # distribution of the sum, support 2..12
    faces = np.arange(1, 7)
    sums = np.arange(2, 13)

    fig, axes = plt.subplots(1, 3, figsize=(3.6, 1.6))
    for ax, (xs, ys, t, col) in zip(
        axes,
        [(faces, die, "P(die A)", ARM_RED),
         (faces, die, "P(die B)", ARM_ORANGE),
         (sums, summ, "P(A + B)", ARM_BLUE)],
    ):
        b = ax.bar(xs, ys, color=col, width=0.7)
        ax.set_title(t, fontsize=8, pad=2)
        ax.set_ylim(0, summ.max() * 1.25)
        ax.set_xticks(xs[::2] if len(xs) > 6 else xs)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    peak = int(np.argmax(summ))
    axes[2].text(sums[peak], summ[peak], "6/36", ha="center", va="bottom", fontsize=7)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    # (suptitle removed 2026-10-01: the slide says it)
    fig.tight_layout()
    out = FIG_DIR / "dice_conv.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig_moving_average(log)
    fig_dice(log)


if __name__ == "__main__":
    main()
