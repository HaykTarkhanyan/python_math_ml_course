"""A real detector's raw output on the belt photo, for the L19 deck ("why so many boxes?").

Generates into ml/12_cnn/fig/:
  detector_candidates.pdf -- three panels on the pomegranate photo: every candidate box
                             scoring above 5%, the ones above the confidence threshold, and what
                             is left after non-max suppression (NMS).
and caches the raw model output in ml/12_cnn/results/detector_candidates.npz (+ .json summary),
so the figure can be redrawn with --plot-only without running the model.

Model: torchvision SSDlite320-MobileNetV3 (one-stage, COCO-trained, 13.4 MB weights, downloaded
to the torch hub cache on first run). COCO has no pomegranate class, so it calls them "apple" /
"orange" - left as is, real output is messy. NMS here is class-agnostic: one box per object,
whatever class it was given.

Run (~1 min on CPU incl. model load; --plot-only ~5 s):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/detector_candidates.py [--plot-only]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS = CH_DIR / "results" / "detector_candidates.npz"
SUMMARY = CH_DIR / "results" / "detector_candidates.json"
PHOTO = FIG_DIR / "borrowed" / "pomegranate_market.jpg"

RAW_MIN = 0.05        # panel 1: everything the model gives at least a 5% score
CONF = 0.4            # panel 2: the confidence threshold
NMS_IOU = 0.45        # panel 3: same threshold as the NMS frame


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("detector_candidates")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "detector_candidates.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def run_model(log):
    import torch
    from PIL import Image
    from torchvision.models.detection import (SSDLite320_MobileNet_V3_Large_Weights,
                                              ssdlite320_mobilenet_v3_large)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    cats = weights.meta["categories"]
    x = weights.transforms()(Image.open(PHOTO).convert("RGB"))
    with torch.no_grad():
        images, _ = model.transform([x])                 # the 320x320 input the model sees
        feats = list(model.backbone(images.tensors).values())
        n_anchors = model.anchor_generator(images, feats)[0].shape[0]
        # raw output: every (candidate, class) pair, no score cut, no NMS
        model.score_thresh, model.nms_thresh = 0.0, 1.0
        model.topk_candidates, model.detections_per_img = 10 ** 6, 10 ** 7
        out = model([x])[0]
    keep = out["scores"] > RAW_MIN
    boxes, scores = out["boxes"][keep].numpy(), out["scores"][keep].numpy()
    names = np.array([cats[i] for i in out["labels"][keep].tolist()])
    log.info(f"{n_anchors} candidate boxes x {len(cats)} class scores; "
             f"{keep.sum().item()} (box, class) pairs score > {RAW_MIN}")
    RESULTS.parent.mkdir(exist_ok=True)
    np.savez(RESULTS, boxes=boxes, scores=scores, names=names, n_anchors=n_anchors,
             n_classes=len(cats))
    log.info(f"saved {RESULTS}")


def nms(boxes, scores, iou_thr):
    """Class-agnostic greedy NMS (the NMS frame's algorithm), returns kept indices."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        x0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        y0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        y1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area(boxes[[i]]) + area(boxes[order[1:]]) - inter)
        order = order[1:][iou <= iou_thr]
    return np.array(keep)


def place_labels(boxes, scores, names, kept, shape):
    """Top-left of each kept box, highest score first; a label that would overlap one already
    placed slides down inside its own box. Sizes are in image pixels for a 5.5 pt label on a
    ~1.77 in wide panel of the 1920 px photo (~45 px per character, ~120 px per line)."""
    char_w, line_h = 45, 120
    placed, pos = [], {}
    for i in sorted(kept, key=lambda k: -scores[k]):
        x0, y0, x1, y1 = boxes[i]
        w = len(f"{names[i]} {scores[i]:.2f}") * char_w
        lx, ly = x0 + 15, y0 + 15
        while any(lx < px + pw and px < lx + w and ly < py + line_h and py < ly + line_h
                  for px, py, pw in placed) and ly + 2 * line_h < y1:
            ly += line_h
        placed.append((lx, ly, w))
        pos[i] = (lx, ly)
    return pos


def plot(log):
    if not RESULTS.exists():
        raise FileNotFoundError(f"{RESULTS} missing - run without --plot-only first")
    r = np.load(RESULTS)
    boxes, scores, names = r["boxes"], r["scores"], r["names"]
    conf = scores > CONF
    kept = np.flatnonzero(conf)[nms(boxes[conf], scores[conf], NMS_IOU)]
    summary = {"n_candidates": int(r["n_anchors"]), "n_classes": int(r["n_classes"]),
               f"pairs_above_{RAW_MIN}": int(len(scores)), f"above_{CONF}": int(conf.sum()),
               "after_nms": int(len(kept)),
               "kept": [{"class": str(names[i]), "score": round(float(scores[i]), 2),
                         "box": [int(v) for v in boxes[i]]} for i in kept]}
    SUMMARY.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info(f"summary: {json.dumps({k: v for k, v in summary.items() if k != 'kept'})}")
    for k in summary["kept"]:
        log.info(f"  kept {k}")

    img = plt.imread(PHOTO)
    label_pos = place_labels(boxes, scores, names, kept, img.shape)
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.4))      # its size on the slide
    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.01, wspace=0.04)
    panels = [(np.arange(len(scores)), f"1. raw: {len(scores)} boxes score $>$ {RAW_MIN:.0%}"),
              (np.flatnonzero(conf), f"2. confidence $>$ {CONF}: {conf.sum()} boxes"),
              (kept, f"3. after NMS: {len(kept)} boxes")]
    for ax, (idx, title) in zip(axes, panels):
        ax.imshow(img)
        for i in idx[np.argsort(scores[idx])]:                # strongest drawn on top
            x0, y0, x1, y1 = boxes[i]
            last = ax is axes[2]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec=ARM_ORANGE if last else "white",
                                   lw=1.0 if last else 0.35,
                                   alpha=1.0 if last else float(np.clip(0.25 + scores[i], 0, 1))))
            if last:
                lx, ly = label_pos[i]
                ax.text(lx, ly, f"{names[i]} {scores[i]:.2f}", fontsize=5.5, va="top",
                        color="black",
                        bbox=dict(boxstyle="square,pad=0.1", fc=ARM_ORANGE, ec="none"))
        ax.set_title(title, fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    out = FIG_DIR / "detector_candidates.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="redraw from the saved npz")
    args = ap.parse_args()
    log = setup_logging()
    t0 = time.perf_counter()
    FIG_DIR.mkdir(exist_ok=True)
    if not args.plot_only:
        run_model(log)
    plot(log)
    log.info(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
