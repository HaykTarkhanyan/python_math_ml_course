"""IoU worked example + NMS panels for the L19 Vision Tasks deck (Section 2).

Left panel: the locked IoU worked example - A = (0,0)-(4,4), B = (2,0)-(6,4):
intersection 8, union 16 + 16 - 8 = 24, IoU = 8/24 = 1/3. The numbers are verified
in code and annotated on the boxes; the frame text uses exactly these values.

Right panel: many overlapping candidate detections around three objects vs the NMS
survivors (greedy NMS, IoU threshold 0.45). All boxes hard-coded, no model.

nms_anim.py imports CANDIDATES / greedy_nms / draw_nms_panel from this module so the
animation's final frame is pixel-identical in content to the right panel here.

Generates into ml/ch6_cnn/fig/:
  iou_nms.pdf   (combined two-panel figure)
  iou_only.pdf  (left panel alone - for the IoU frame)
  nms_only.pdf  (right panel alone)

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/iou_nms.py
"""

import logging
from fractions import Fraction
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

SEED = 509
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#008C46"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

# the locked worked example (do not change - frame text quotes these)
BOX_A = (0.0, 0.0, 4.0, 4.0)
BOX_B = (2.0, 0.0, 6.0, 4.0)

# NMS demo: three schematic objects + nine candidate boxes (x0, y0, w, h, score)
OBJECTS = [(2.25, 4.65, 1.05), (6.45, 4.95, 0.95), (4.70, 1.95, 1.00)]
CANDIDATES = [
    (1.15, 3.50, 2.30, 2.30, 0.92),
    (1.45, 3.75, 2.20, 2.10, 0.78),
    (0.90, 3.30, 2.50, 2.40, 0.61),
    (1.30, 3.20, 2.10, 2.50, 0.55),
    (5.45, 3.95, 2.00, 2.00, 0.88),
    (5.70, 4.15, 1.90, 1.75, 0.72),
    (5.20, 3.70, 2.30, 2.20, 0.50),
    (3.60, 0.85, 2.20, 2.20, 0.95),
    (3.90, 1.05, 2.05, 1.90, 0.66),
]
NMS_THRESHOLD = 0.45

# score-chip corner per candidate ("tl", "tr", "bl", "br"), hand-picked so chips
# in the same cluster never overlap
LABEL_ANCHORS = ["tl", "tr", "bl", "br", "tl", "tr", "bl", "tl", "br"]


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l19_iou_nms")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "iou_nms.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def iou_corners(a, b) -> float:
    """IoU of two boxes given as (x0, y0, x1, y1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def iou_xywh(a, b) -> float:
    """IoU of two boxes given as (x0, y0, w, h, ...)."""
    return iou_corners((a[0], a[1], a[0] + a[2], a[1] + a[3]),
                       (b[0], b[1], b[0] + b[2], b[1] + b[3]))


def greedy_nms(candidates, threshold):
    """Greedy NMS. Returns a list of steps: (kept_index, [dropped_indices])."""
    remaining = sorted(range(len(candidates)),
                       key=lambda i: candidates[i][4], reverse=True)
    steps = []
    while remaining:
        keep = remaining.pop(0)
        dropped = [i for i in remaining
                   if iou_xywh(candidates[keep], candidates[i]) > threshold]
        remaining = [i for i in remaining if i not in dropped]
        steps.append((keep, dropped))
    return steps


def draw_iou_panel(ax):
    """The locked worked example, annotated on the boxes."""
    ax.set_xlim(-0.7, 7.3)
    ax.set_ylim(-2.6, 5.3)
    ax.set_aspect("equal")
    ax.axis("off")
    for g in range(0, 7):
        ax.plot([g, g], [0, 4], color="#e3e3e3", lw=0.7, zorder=0)
    for g in range(0, 5):
        ax.plot([0, 6], [g, g], color="#e3e3e3", lw=0.7, zorder=0)

    # intersection strip (2,0)-(4,4)
    ax.add_patch(Rectangle((2, 0), 2, 4, fc="#b78ccf", ec="none",
                           alpha=0.55, zorder=1))
    ax.add_patch(Rectangle((0, 0), 4, 4, fill=False, ec=ARM_BLUE, lw=3.2,
                           zorder=3))
    ax.add_patch(Rectangle((2, 0), 4, 4, fill=False, ec=ARM_ORANGE, lw=3.2,
                           zorder=3))

    ax.text(0.15, 4.12, "A = (0,0)-(4,4)", color=ARM_BLUE, fontsize=12,
            fontweight="bold", va="bottom")
    ax.text(3.6, -0.62, "B = (2,0)-(6,4)", color="#b57900", fontsize=12,
            fontweight="bold", va="top")
    ax.text(3.0, 2.0, "8", color="#5a2d7a", fontsize=17, fontweight="bold",
            ha="center", va="center")
    ax.text(1.0, 2.0, "area 16", color=ARM_BLUE, fontsize=10.5, ha="center")
    ax.text(5.0, 2.0, "area 16", color="#b57900", fontsize=10.5, ha="center")

    ax.text(3.0, -1.55,
            "intersection = 8      union = 16 + 16 - 8 = 24\n"
            "IoU = 8 / 24 = 1/3",
            fontsize=13, ha="center", va="center", linespacing=1.5)
    ax.set_title("IoU: how much do two boxes agree?", fontsize=13)


def draw_nms_panel(ax, keep=None, drop=None, current=None, current_drop=None,
                   title=None, show_scores=True):
    """One state of the NMS demo.

    keep / drop: index sets already decided (survivors green, dropped faded).
    current: the box being kept this click (bold green).
    current_drop: boxes being dropped this click (red, dashed).
    Undecided boxes are drawn in blue with their scores.
    """
    keep = set(keep or [])
    drop = set(drop or [])
    current_drop = set(current_drop or [])

    ax.set_xlim(-0.2, 9.2)
    ax.set_ylim(-0.9, 7.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 9, 6.6, fc="#f7f7f7", ec="#bbbbbb",
                           lw=1.0, zorder=0))
    for cx, cy, r in OBJECTS:
        ax.add_patch(Circle((cx, cy), r, fc="#d9d9d9", ec="#a8a8a8", lw=1.2,
                            zorder=1))

    for i, (x0, y0, w, h, s) in enumerate(CANDIDATES):
        if i == current:
            ec, lw, ls, alpha, z = GREEN, 3.4, "-", 1.0, 5
        elif i in keep:
            ec, lw, ls, alpha, z = GREEN, 3.0, "-", 1.0, 4
        elif i in current_drop:
            ec, lw, ls, alpha, z = ARM_RED, 2.2, "--", 0.95, 4
        elif i in drop:
            ec, lw, ls, alpha, z = "#c9c9c9", 1.4, "--", 0.8, 2
        else:
            ec, lw, ls, alpha, z = ARM_BLUE, 1.9, "-", 0.95, 3
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=ec, lw=lw,
                               ls=ls, alpha=alpha, zorder=z))
        if show_scores and not (i in drop and i not in current_drop):
            color = (GREEN if (i == current or i in keep)
                     else ARM_RED if i in current_drop else ARM_BLUE)
            anchor = LABEL_ANCHORS[i]
            tx = x0 + 0.10 if anchor[1] == "l" else x0 + w - 0.10
            ty = y0 + h - 0.12 if anchor[0] == "t" else y0 + 0.12
            ax.text(tx, ty, f"{s:.2f}", fontsize=9.5,
                    fontweight="bold", color="white", zorder=6,
                    va="top" if anchor[0] == "t" else "bottom",
                    ha="left" if anchor[1] == "l" else "right",
                    bbox=dict(boxstyle="square,pad=0.16", fc=color, ec="none",
                              alpha=alpha))
    if title is not None:
        ax.set_title(title, fontsize=13)


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    # ---- verify the locked worked example in code ----
    ix = min(BOX_A[2], BOX_B[2]) - max(BOX_A[0], BOX_B[0])
    iy = min(BOX_A[3], BOX_B[3]) - max(BOX_A[1], BOX_B[1])
    inter = ix * iy
    area_a = (BOX_A[2] - BOX_A[0]) * (BOX_A[3] - BOX_A[1])
    area_b = (BOX_B[2] - BOX_B[0]) * (BOX_B[3] - BOX_B[1])
    union = area_a + area_b - inter
    iou = Fraction(int(inter), int(union))
    log.info(f"worked example: inter={inter}, union={union}, IoU={iou}")
    if not (inter == 8 and union == 24 and iou == Fraction(1, 3)):
        raise AssertionError("worked-example numbers drifted from the locked spec")

    # ---- verify the NMS demo collapses to exactly three survivors ----
    steps = greedy_nms(CANDIDATES, NMS_THRESHOLD)
    survivors = [k for k, _ in steps]
    log.info(f"NMS steps: {steps}")
    if len(survivors) != 3:
        raise AssertionError(f"expected 3 survivors, got {survivors}")
    dropped_all = {i for _, d in steps for i in d}
    if dropped_all | set(survivors) != set(range(len(CANDIDATES))):
        raise AssertionError("NMS steps do not cover all candidates")

    final_keep, final_drop = set(survivors), dropped_all

    # ---- combined two-panel figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    draw_iou_panel(axL)
    draw_nms_panel(axR, keep=final_keep, drop=final_drop,
                   title=f"NMS: 9 candidates $\\to$ 3 survivors "
                         f"(IoU threshold {NMS_THRESHOLD})")
    fig.savefig(FIG_DIR / "iou_nms.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {FIG_DIR / 'iou_nms.pdf'}")

    # ---- single panels ----
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    draw_iou_panel(ax)
    fig.savefig(FIG_DIR / "iou_only.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {FIG_DIR / 'iou_only.pdf'}")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    draw_nms_panel(ax, keep=final_keep, drop=final_drop,
                   title=f"9 candidates $\\to$ 3 survivors "
                         f"(IoU threshold {NMS_THRESHOLD})")
    fig.savefig(FIG_DIR / "nms_only.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {FIG_DIR / 'nms_only.pdf'}")


if __name__ == "__main__":
    main()
