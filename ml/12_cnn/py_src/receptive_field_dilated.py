"""Dilated-convolution figures for the L19 Vision Tasks deck (Section 3).

L17's receptive_field.pdf (stacked 3x3) has no dilated panel, and its file must not
be regenerated (the compiled L17 deck embeds it; matplotlib output is not
byte-deterministic). This script therefore emits NEW files only:

  receptive_field_dilated.pdf   -- stacked standard 3x3 (RF 3->5->7, linear) vs a
                                   stacked dilated 1,2,4 net (RF 3->7->15,
                                   exponential), same 9 weights per layer
  dilation_anim_0..2.pdf        -- ANIM: one 3x3 kernel's taps at dilation 1, 2, 4
                                   (span 3 -> 5 -> 9, still 9 weights), fixed
                                   geometry across frames

Receptive-field arithmetic (verified in code): stacking a 3x3 conv with dilation d
adds 2d to the receptive field. Standard stack d=1,1,1: 3,5,7. WaveNet-style stack
d=1,2,4: 3,7,15.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/receptive_field_dilated.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l19_receptive_field_dilated")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "receptive_field_dilated.log",
                             encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def rf_after(dilations) -> list:
    """Receptive field after each layer of stacked 3x3 convs with given dilations."""
    rf, out = 1, []
    for d in dilations:
        rf += 2 * d
        out.append(rf)
    return out


def draw_grid(ax, n, rf_specs, title):
    """An n x n pixel grid with nested receptive-field squares (size, color, label)."""
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((j, i), 1, 1, fc="#f2f2f2", ec="#dddddd",
                                   lw=0.7))
    c = n // 2
    ax.add_patch(Rectangle((c, c), 1, 1, fc="#cccccc", ec="#999999", lw=0.7))
    for size, color, _ in rf_specs:
        half = size // 2
        ax.add_patch(Rectangle((c - half, c - half), size, size, fill=False,
                               ec=color, lw=2.8))
    ax.set_xlim(-0.3, n + 0.3)
    ax.set_ylim(-3.1, n + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12.5)
    for idx, (size, color, label) in enumerate(rf_specs):
        y = -0.75 - idx * 0.78
        ax.plot([0.2, 1.0], [y, y], color=color, lw=2.8)
        ax.text(1.25, y, label, color=color, fontsize=10.5, va="center")


def make_static(log):
    dil_std, dil_wav = [1, 1, 1], [1, 2, 4]
    rf_std, rf_wav = rf_after(dil_std), rf_after(dil_wav)
    log.info(f"standard stack RF: {rf_std}; dilated stack RF: {rf_wav}")
    if rf_std != [3, 5, 7] or rf_wav != [3, 7, 15]:
        raise AssertionError("receptive-field arithmetic drifted from the spec")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 5.0))
    n = 15
    colors = [ARM_RED, ARM_BLUE, ARM_ORANGE]
    draw_grid(axL, n,
              [(s, c, f"layer {k + 1} (d=1): {s}x{s}")
               for k, (s, c) in enumerate(zip(rf_std, colors))],
              "3 standard 3x3 layers\nreceptive field grows linearly")
    draw_grid(axR, n,
              [(s, c, f"layer {k + 1} (d={d}): {s}x{s}")
               for k, (s, c, d) in enumerate(zip(rf_wav, colors, dil_wav))],
              "3 dilated 3x3 layers (d = 1, 2, 4)\nreceptive field grows exponentially")
    fig.text(0.5, 0.015, "same cost either way: 9 weights per layer",
             ha="center", fontsize=12, fontweight="bold", color=ARM_BLUE)
    out = FIG_DIR / "receptive_field_dilated.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


def make_anim(log):
    n = 11
    c = n // 2
    for idx, d in enumerate([1, 2, 4]):
        fig = plt.figure(figsize=(5.6, 5.6))
        ax = fig.add_axes([0.03, 0.10, 0.94, 0.80])  # fixed rect - no jitter
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((j, i), 1, 1, fc="#f2f2f2",
                                       ec="#dddddd", lw=0.7))
        span = 2 * d + 1
        half = span // 2
        ax.add_patch(Rectangle((c - half, c - half), span, span, fill=False,
                               ec=ARM_ORANGE, lw=2.6))
        for di in (-d, 0, d):
            for dj in (-d, 0, d):
                ax.add_patch(Rectangle((c + dj, c + di), 1, 1, fc=ARM_BLUE,
                                       ec="#1e3a7a", lw=0.8))
        ax.set_xlim(-0.3, n + 0.3)
        ax.set_ylim(-0.3, n + 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"dilation d = {d}: the same 9 weights span "
                     f"{span}x{span}", fontsize=13)
        out = FIG_DIR / f"dilation_anim_{idx}.pdf"
        fig.savefig(out)  # deliberately NOT tight: fixed geometry across frames
        plt.close(fig)
        log.info(f"saved {out.name} (d={d}, span {span})")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    make_static(log)
    make_anim(log)


if __name__ == "__main__":
    main()
