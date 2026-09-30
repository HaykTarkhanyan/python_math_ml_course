"""Timeline ribbons for the L17 CNN Architectures deck (section-transition slides).

One thin horizontal 1989-2026 timeline strip per section transition, with the current
story position highlighted in red (everything else gray). All six variants share
identical geometry so the ribbon does not jump between sections.

Generates into ml/ch6_cnn/fig/:
  ribbon_1.pdf  -- Section 1 (the benchmark):        highlight 2009 ImageNet
  ribbon_2.pdf  -- Section 2 (the classics):          highlight 1989 / 2012 / 2014
  ribbon_3.pdf  -- Section 3 (degradation + ResNet):  highlight 2015 ResNet
  ribbon_4.pdf  -- Section 4 (the wider zoo):         highlight 2014 VGG/GoogLeNet
  ribbon_5.pdf  -- Section 5 (practice):              highlight 2026 today
  ribbon_6.pdf  -- Section 6 (epilogue):              highlight 2017..2025 (the frontier)

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/timeline_ribbon.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 509
ARM_RED = "#D90012"
GRAY = "#9a9a9a"
DARK = "#444444"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

# (year, label, side: +1 above / -1 below, lane: 1 near / 2 far)
# lanes hand-assigned so no two same-side-same-lane neighbors collide, even when
# highlighted (larger font)
EVENTS = [
    (1989, "LeNet", 1, 1),
    (2009, "ImageNet", 1, 1),
    (2012, "AlexNet", -1, 1),
    (2014, "VGG /\nGoogLeNet", 1, 1),
    (2015, "ResNet", -1, 1),
    (2017, "Transformer", -1, 2),
    (2020, "ViT", -1, 1),
    (2021, "CLIP", 1, 1),
    (2022, "Stable\nDiffusion", -1, 2),
    (2023, "SAM", 1, 1),
    (2025, "world\nmodels", -1, 1),
    (2026, "today", 1, 1),
]

# variant -> set of highlighted years
VARIANTS = {
    1: {2009},
    2: {1989, 2012, 2014},
    3: {2015},
    4: {2014},
    5: {2026},
    6: {2017, 2020, 2021, 2022, 2023, 2025},
}


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l17_timeline_ribbon")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "timeline_ribbon.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def draw(variant: int, highlight: set, log):
    fig, ax = plt.subplots(figsize=(12.0, 1.85))
    fig.subplots_adjust(left=0.015, right=0.985, top=1.0, bottom=0.0)

    y0 = 0.0
    ax.plot([1988, 2027], [y0, y0], color="#cccccc", lw=2.0, zorder=1,
            solid_capstyle="round")

    for year, label, side, lane in EVENTS:
        hot = year in highlight
        color = ARM_RED if hot else GRAY
        ax.plot([year], [y0], marker="o", ms=9 if hot else 5.5,
                color=color, zorder=3)
        ty = side * (0.55 if lane == 1 else 2.0)
        ax.annotate(f"{label}\n{year}" if side > 0 else f"{year}\n{label}",
                    xy=(year, ty), ha="center",
                    va="bottom" if side > 0 else "top",
                    fontsize=14 if hot else 11,
                    fontweight="bold" if hot else "normal",
                    color=ARM_RED if hot else DARK,
                    linespacing=0.95, annotation_clip=False)
        ax.plot([year, year], [y0, ty * (0.72 if lane == 1 else 0.9)],
                color=color, lw=1.0, zorder=2)

    ax.set_xlim(1987.2, 2027.8)
    ax.set_ylim(-4.4, 2.5)
    ax.axis("off")

    out = FIG_DIR / f"ribbon_{variant}.pdf"
    fig.savefig(out)   # fixed geometry - no tight bbox
    plt.close(fig)
    log.info(f"saved {out} (highlight={sorted(highlight)})")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    for variant, highlight in VARIANTS.items():
        draw(variant, highlight, log)


if __name__ == "__main__":
    main()
