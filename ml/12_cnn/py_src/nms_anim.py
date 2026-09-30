"""NMS flip-book animation for the L19 Vision Tasks deck (Section 2, ANIM).

Frames of greedy non-max suppression pruning overlapping candidate detections click
by click: keep the highest-scoring box (green), drop its neighbors above the IoU
threshold (red dashed), repeat. The final frame is the same state as the right panel
of iou_nms.pdf - boxes, NMS logic and drawing are imported from iou_nms.py so the two
figures cannot drift apart.

Geometry is fixed across frames (same figsize, same axes rect, no tight bbox) so
nothing jitters between clicks.

Generates into ml/ch6_cnn/fig/:
  nms_anim_0.pdf .. nms_anim_N.pdf   (N reported in the log)

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/nms_anim.py
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iou_nms import CANDIDATES, NMS_THRESHOLD, greedy_nms, draw_nms_panel

SEED = 509

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l19_nms_anim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "nms_anim.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def save_frame(idx: int, log, **panel_kwargs):
    fig = plt.figure(figsize=(7.6, 5.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])  # fixed rect - no jitter
    draw_nms_panel(ax, **panel_kwargs)
    out = FIG_DIR / f"nms_anim_{idx}.pdf"
    fig.savefig(out)  # deliberately NOT bbox_inches="tight": fixed geometry
    plt.close(fig)
    log.info(f"saved {out.name}")


def main():
    log = setup_logging()
    np.random.seed(SEED)
    FIG_DIR.mkdir(exist_ok=True)

    steps = greedy_nms(CANDIDATES, NMS_THRESHOLD)
    log.info(f"NMS steps: {steps}")
    if len(steps) != 3:
        raise AssertionError(f"expected 3 NMS steps for the demo, got {len(steps)}")

    n_cand = len(CANDIDATES)
    frame = 0

    # frame 0: the raw flood of candidates
    save_frame(frame, log,
               title=f"the net proposes {n_cand} boxes for 3 objects")
    frame += 1

    # one frame per NMS iteration: current keep (bold green) + its drops (red)
    kept, dropped = set(), set()
    for keep_idx, drop_idxs in steps:
        score = CANDIDATES[keep_idx][4]
        save_frame(frame, log,
                   keep=kept, drop=dropped,
                   current=keep_idx, current_drop=drop_idxs,
                   title=f"keep the best remaining ({score:.2f}), "
                         f"drop neighbors with IoU > {NMS_THRESHOLD}")
        kept.add(keep_idx)
        dropped.update(drop_idxs)
        frame += 1

    # final frame: identical state to the iou_nms.pdf right panel
    save_frame(frame, log,
               keep=kept, drop=dropped,
               title=f"{n_cand} candidates $\\to$ {len(kept)} survivors: "
                     "one box per object")
    log.info(f"emitted {frame + 1} frames (nms_anim_0..{frame})")


if __name__ == "__main__":
    main()
