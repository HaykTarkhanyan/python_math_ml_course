"""Timeline ribbons for the L17 CNN Architectures deck (section-transition slides).

One thin horizontal 1989-2026 timeline strip per section transition, with the current
story position highlighted in red (everything else gray). All six variants share
identical geometry so the ribbon does not jump between sections.

Generates into ml/12_cnn/fig/ (2026-10-01: sections reordered chronologically, instructor):
  ribbon_1.pdf  -- Section 1 (the benchmark):                 highlight 2009 ImageNet
  ribbon_2.pdf  -- Section 2 (the breakout: LeNet..ZFNet):    highlight 1989 / 2012 / 2013
  ribbon_3.pdf  -- Section 3 (2014: VGG and GoogLeNet):       highlight 2014
  ribbon_4.pdf  -- Section 4 (2015: ResNet):                  highlight 2015
  ribbon_5.pdf  -- Section 5 (after ImageNet, in practice):   highlight 2026 today
  ribbon_6.pdf  -- Section 6 (epilogue):                      highlight 2017..2025 (the frontier)

2026-10-01: drawn at its slide width (5.4 x 1.2 in), so labels are 8-9.5 pt on the slide (they
were 5 pt); three label lanes per side so neighbours do not collide at that size.

Run with the project venv (~3 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/timeline_ribbon.py
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

# (year, label, side: +1 above / -1 below, lane: 1 near .. 3 far)
# lanes hand-assigned so no two same-side-same-lane neighbors collide, even when
# highlighted (larger font)
EVENTS = [
    (1989, "LeNet", 1, 1),
    (2009, "ImageNet", 1, 1),
    (2012, "AlexNet", -1, 1),
    (2013, "ZFNet", 1, 1),
    (2014, "VGG, GoogLeNet", -1, 2),
    (2015, "ResNet", 1, 2),
    (2017, "Transformer", -1, 1),
    (2020, "ViT", 1, 1),
    (2021, "CLIP", -1, 2),
    (2022, "Stable Diffusion", 1, 2),
    (2023, "SAM", -1, 1),
    (2025, "world models", 1, 1),
    (2026, "today", -1, 2),
]
LANE_Y = {1: 0.4, 2: 1.9}                # label distance from the axis per lane (data units)
BREAK = 2008                              # years before this sit at x = 2006, after a // break


def xpos(year: int) -> float:
    """Broken axis: 1989-2008 would eat half the strip, so LeNet sits just left of a break."""
    return year if year >= BREAK else 2006.0

# variant -> set of highlighted years
VARIANTS = {
    1: {2009},
    2: {1989, 2012, 2013},
    3: {2014},
    4: {2015},
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
    fig, ax = plt.subplots(figsize=(5.4, 1.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.0)

    y0 = 0.0
    ax.plot([2005.2, 2007.2], [y0, y0], color="#cccccc", lw=1.5, zorder=1,
            solid_capstyle="round")
    ax.plot([2007.8, 2027.0], [y0, y0], color="#cccccc", lw=1.5, zorder=1,
            solid_capstyle="round")
    ax.text(2007.5, y0, "//", ha="center", va="center", fontsize=8, color=GRAY)

    for year, label, side, lane in EVENTS:
        hot = year in highlight
        color = ARM_RED if hot else GRAY
        x = xpos(year)
        ax.plot([x], [y0], marker="o", ms=5.5 if hot else 3.5, color=color, zorder=3)
        ty = side * LANE_Y[lane]
        text = f"{label}\n{year}" if side > 0 else f"{year}\n{label}"
        ax.annotate(text, xy=(x, ty), ha="center", va="bottom" if side > 0 else "top",
                    fontsize=8.5 if hot else 7.5, linespacing=0.95,
                    fontweight="bold" if hot else "normal",
                    color=ARM_RED if hot else DARK, annotation_clip=False)
        ax.plot([x, x], [y0, ty], color=color, lw=0.7, zorder=2)

    ax.set_xlim(2004.4, 2027.8)
    ax.set_ylim(-3.6, 3.6)
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
