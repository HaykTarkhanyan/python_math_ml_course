"""Real figure for the L17 CNN Architectures deck (cold open / Section 1).

Generates into ml/ch6_cnn/fig/:
  imagenet_error.pdf            -- ILSVRC top-5 error by year, 2010-2015: labeled bars
                                   (winner + error %), a human-baseline line at ~5.1%.
  imagenet_anim_0.pdf .. _6.pdf -- flip-book: empty axes (frame 0) then one bar revealed
                                   per click (frames 1-6). The last frame equals the full
                                   static chart. Shown on the cold-open frame via \\only<n>.

Data source: ILSVRC winning entries (Russakovsky et al., "ImageNet Large Scale Visual
Recognition Challenge", IJCV 2015) + the winning papers; human baseline ~5.1% from
Russakovsky/Karpathy. Numbers web-verified 2026-07-14 (match the canonical CS231n chart).

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/imagenet_error.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag palette, value labels on bars.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

# (year, winning system, top-5 error %, era)  era: "classical" (pre-deep) or "deep"
DATA = [
    (2010, "NEC-UIUC", 28.2, "classical"),
    (2011, "XRCE", 25.8, "classical"),
    (2012, "AlexNet", 16.4, "deep"),
    (2013, "ZFNet", 11.7, "deep"),
    (2014, "GoogLeNet", 6.7, "deep"),
    (2015, "ResNet", 3.57, "deep"),
]
HUMAN = 5.1


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l17_imagenet_error")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "imagenet_error.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def draw(k, log, out_path):
    """Draw the chart with the first k bars revealed (k in 0..6)."""
    xs = list(range(len(DATA)))

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    # fixed geometry so anim frames do not jump between clicks
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.15)

    # human baseline line + label (present in every frame)
    ax.axhline(HUMAN, color=ARM_ORANGE, lw=2.4, ls="--", zorder=1)
    ax.text(len(DATA) - 0.5, HUMAN + 0.7, f"human  ~{HUMAN}%", color="#b07d00",
            ha="right", va="bottom", fontsize=11, fontweight="bold")

    for i, (year, name, err, era) in enumerate(DATA):
        if i >= k:
            continue
        color = ARM_RED if era == "classical" else ARM_BLUE
        ax.bar(i, err, width=0.66, color=color, zorder=3)
        ax.text(i, err + 0.6, f"{err:g}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#222")

    # winner names live in the tick labels - and only appear once their bar is
    # revealed, so the flip-book does not spoil the story
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{d[0]}\n{d[1]}" if i < k else f"{d[0]}\n " for i, d in enumerate(DATA)],
        fontsize=11)
    ax.set_xlim(-0.6, len(DATA) - 0.4)
    ax.set_ylim(0, 32)
    ax.set_ylabel("top-5 error (%)", fontsize=12)
    ax.set_title("ILSVRC image classification: top-5 error by year", fontsize=14)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    # legend-ish note distinguishing the two eras (only once both eras exist)
    if k >= 3:
        ax.text(0.98, 0.965, "red = pre-deep-learning winners", color=ARM_RED,
                fontsize=10, ha="right", va="top", transform=ax.transAxes)
        ax.text(0.98, 0.905, "blue = deep CNN winners", color=ARM_BLUE,
                fontsize=10, ha="right", va="top", transform=ax.transAxes)
    ax.text(0.99, -0.125, "Data: ILSVRC winning entries (Russakovsky et al. 2015).",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#888")

    fig.savefig(out_path)   # no tight bbox: identical geometry across frames
    plt.close(fig)
    log.info(f"saved {out_path} (k={k})")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    # anim frames 0..6 (0 = empty axes + human line, 1..6 reveal bars cumulatively)
    for k in range(len(DATA) + 1):
        draw(k, log, FIG_DIR / f"imagenet_anim_{k}.pdf")
    # standalone full static chart (same geometry as the last anim frame)
    draw(len(DATA), log, FIG_DIR / "imagenet_error.pdf")


if __name__ == "__main__":
    main()
