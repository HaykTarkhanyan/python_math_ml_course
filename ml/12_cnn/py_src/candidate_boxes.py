"""Where a detector's candidate boxes come from - three figures for the [51] Vision Tasks deck.

Frames "Idea 1: crop and classify", "Idea 2: one pass, a small head at every cell" and
"Box sizes are not fixed: anchors + offsets", all on the belt photo:

  sliding_windows.pdf   -- idea 1, measured: crop the photo at 3 window sizes (stride = half a
                           window), run an ImageNet ResNet-18 on EVERY crop, one forward pass
                           each; three crops shown with their real P(pomegranate).
  ssd_grid_anchors.pdf  -- idea 2, real model: SSDlite320-MobileNetV3 (COCO) squashes the photo
                           to 320x320 and predicts on 6 grids (20x20 ... 1x1) with 6 anchor shapes
                           per cell = 3,234 candidates. Shown: the 5x5 grid, the cell whose anchor
                           gave the top detection, and that cell's 6 anchors.
  ssd_anchor_decode.pdf -- the same top detection: its anchor (a preset shape) and the final box
                           after the head's predicted shift and stretch.

Every number on the slides is in results/candidate_boxes.json (the artifact of record); the
figures can be redrawn from it with --plot-only. The SSDlite top box matches the 0.93 "apple"
of detector_candidates.json (same model, same photo).

Run (~40 s on CPU incl. model loads; --plot-only ~5 s; weights from the torch hub cache):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/candidate_boxes.py [--plot-only]
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
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#00c060"
ANCHOR = "#29b6f6"        # anchors: dashed light blue in both anchor figures

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS = CH_DIR / "results" / "candidate_boxes.json"
PHOTO = FIG_DIR / "borrowed" / "pomegranate_market.jpg"

WINDOWS = (320, 480, 640)        # sliding-window sizes in photo pixels, stride = size / 2
SHOW_CROPS = [(640, 640, 640), (160, 800, 320), (1600, 960, 320)]   # (x0, y0, size) to label


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("candidate_boxes")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "candidate_boxes.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def window_grid(w: int, h: int) -> list:
    crops = []
    for s in WINDOWS:
        step = s // 2
        for y0 in range(0, h - s + 1, step):
            for x0 in range(0, w - s + 1, step):
                crops.append((x0, y0, s))
    return crops


def sliding_windows(img, log) -> dict:
    """Idea 1, for real: one ResNet-18 forward pass per crop."""
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()
    tf = weights.transforms()
    pom = weights.meta["categories"].index("pomegranate")
    crops = window_grid(*img.size)
    for c in SHOW_CROPS:
        if c not in crops:
            raise AssertionError(f"shown crop {c} is not on the sliding-window grid")
    t0 = time.perf_counter()
    probs = []
    with torch.no_grad():
        for i, (x0, y0, s) in enumerate(crops):
            x = tf(img.crop((x0, y0, x0 + s, y0 + s))).unsqueeze(0)   # one pass per crop
            probs.append(float(model(x).softmax(-1)[0, pom]))
            if (i + 1) % 40 == 0:
                el = time.perf_counter() - t0
                log.info(f"{i + 1}/{len(crops)} crops, {el:.0f}s elapsed, "
                         f"~{el / (i + 1) * (len(crops) - i - 1):.0f}s left")
    secs = time.perf_counter() - t0
    shown = {str(c): round(probs[crops.index(c)], 3) for c in SHOW_CROPS}
    log.info(f"{len(crops)} crops, {len(crops)} forward passes, {secs:.1f} s; shown {shown}")
    return {"n_crops": len(crops), "seconds": round(secs, 1), "shown": shown,
            "per_size": {s: sum(c[2] == s for c in crops) for s in WINDOWS}}


def ssd_top_detection(img, log) -> dict:
    """Idea 2, for real: SSDlite's grids, anchors and the decode of its best candidate."""
    import torch
    from torchvision.models.detection import (SSDLite320_MobileNet_V3_Large_Weights,
                                              ssdlite320_mobilenet_v3_large)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    cats = weights.meta["categories"]
    with torch.no_grad():
        images, _ = model.transform([weights.transforms()(img)])
        feats = list(model.backbone(images.tensors).values())
        anchors = model.anchor_generator(images, feats)[0]
        head = model.head(feats)
        reg, probs = head["bbox_regression"][0], head["cls_logits"][0].softmax(-1)
        boxes = model.box_coder.decode_single(reg, anchors)
    grids = [int(f.shape[-1]) for f in feats]
    per_cell = model.anchor_generator.num_anchors_per_location()
    n = sum(g * g * a for g, a in zip(grids, per_cell))
    if n != anchors.shape[0] or n != 3234:
        raise AssertionError(f"candidate count {n} / {anchors.shape[0]} != 3,234 on the slide")
    best_p, best_c = probs[:, 1:].max(-1)                  # class 0 is "background"
    i = int(best_p.argmax())
    start = 0
    for lvl, (g, a) in enumerate(zip(grids, per_cell)):
        if i < start + g * g * a:
            cell, k = divmod(i - start, a)
            row, col = divmod(cell, g)
            cell_anchors = anchors[start + cell * a: start + (cell + 1) * a].tolist()
            break
        start += g * g * a
    ax0, ay0, ax1, ay1 = anchors[i].tolist()
    bx0, by0, bx1, by1 = boxes[i].tolist()
    out = {"input": int(images.tensors.shape[-1]), "grids": grids, "anchors_per_cell": per_cell,
           "n_candidates": n, "n_classes": len(cats), "per_anchor_numbers": 4 + len(cats),
           "top": {"class": cats[int(best_c[i]) + 1], "score": round(float(best_p[i]), 3),
                   "grid": grids[lvl], "row": row, "col": col, "anchor_index": k,
                   "anchor": [round(v, 1) for v in (ax0, ay0, ax1, ay1)],
                   "box": [round(v, 1) for v in (bx0, by0, bx1, by1)],
                   "cell_anchors": [[round(v, 1) for v in b] for b in cell_anchors],
                   "shift_px": [round((bx0 + bx1 - ax0 - ax1) / 2, 1),
                                round((by0 + by1 - ay0 - ay1) / 2, 1)],
                   "stretch": [round((bx1 - bx0) / (ax1 - ax0), 2),
                               round((by1 - by0) / (ay1 - ay0), 2)]}}
    log.info(f"grids {grids} x {per_cell} anchors = {n} candidates; top: {out['top']}")
    return out


def to_photo(b, sx, sy):
    x0, y0, x1, y1 = b
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def rect(ax, b, **kw):
    x0, y0, x1, y1 = b
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, **kw))


def plot(r: dict, log) -> None:
    img = plt.imread(PHOTO)
    h, w = img.shape[:2]

    # 1. sliding windows
    fig, ax = plt.subplots(figsize=(2.75, 1.95))
    fig.subplots_adjust(left=0, right=1, top=0.89, bottom=0)
    ax.imshow(img)
    for (x0, y0, s) in window_grid(w, h):
        if s == WINDOWS[0]:
            rect(ax, (x0, y0, x0 + s, y0 + s), ec="white", lw=0.25, alpha=0.5)
    for c, color in zip(SHOW_CROPS, (ARM_ORANGE, GREEN, "#ff4fd8")):
        x0, y0, s = c
        ins = 12                                   # keep edges on the photo's border visible
        rect(ax, (x0 + ins, y0 + ins, x0 + s - ins, y0 + s - ins), ec=color, lw=1.6)
        p = r["windows"]["shown"][str(c)]
        right = x0 + s > 0.8 * w
        ax.text(x0 + s - 30 if right else x0 + 30, y0 + 35, f"p = {p:.2f}", fontsize=6.5,
                va="top", ha="right" if right else "left",
                bbox=dict(boxstyle="square,pad=0.15", fc=color, ec="none"))
    ax.set_title(f"{r['windows']['n_crops']} crops, 3 sizes: {r['windows']['n_crops']} "
                 f"forward passes", fontsize=7.5, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(FIG_DIR / "sliding_windows.pdf", dpi=200)
    plt.close(fig)

    # 2. the 5x5 grid, one cell, its 6 anchors
    t = r["ssd"]["top"]
    g, n_in = t["grid"], r["ssd"]["input"]
    sx, sy = w / n_in, h / n_in
    fig, ax = plt.subplots(figsize=(2.75, 1.95))
    fig.subplots_adjust(left=0, right=1, top=0.89, bottom=0)
    ax.imshow(img, alpha=0.8)
    for k in range(1, g):
        ax.axvline(k * w / g, color="white", lw=0.8)
        ax.axhline(k * h / g, color="white", lw=0.8)
    cw, ch = w / g, h / g
    ax.add_patch(Rectangle((t["col"] * cw, t["row"] * ch), cw, ch, fc=GREEN, alpha=0.45,
                           ec="none"))
    rect(ax, (t["col"] * cw, t["row"] * ch, (t["col"] + 1) * cw, (t["row"] + 1) * ch),
         ec=GREEN, lw=1.6)
    for b in t["cell_anchors"]:
        rect(ax, to_photo(b, sx, sy), ec=ANCHOR, lw=1.0, ls=(0, (3, 2)))
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title(f"{g}x{g} grid: one cell (green), its {len(t['cell_anchors'])} anchors",
                 fontsize=7.5, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(FIG_DIR / "ssd_grid_anchors.pdf", dpi=200)
    plt.close(fig)

    # 3. anchor -> final box
    anc, box = to_photo(t["anchor"], sx, sy), to_photo(t["box"], sx, sy)
    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img)
    rect(ax, (t["col"] * cw, t["row"] * ch, (t["col"] + 1) * cw, (t["row"] + 1) * ch),
         ec=GREEN, lw=1.6)
    rect(ax, anc, ec=ANCHOR, lw=1.4, ls=(0, (4, 2)))
    rect(ax, box, ec=ARM_ORANGE, lw=2.0)
    ax.plot([(anc[0] + anc[2]) / 2], [(anc[1] + anc[3]) / 2], "o", color=ANCHOR, ms=3)
    ax.plot([(box[0] + box[2]) / 2], [(box[1] + box[3]) / 2], "o", color=ARM_ORANGE, ms=3)
    pad = 60
    ax.set_xlim(min(anc[0], box[0]) - pad, max(anc[2], box[2]) + pad)
    ax.set_ylim(min(anc[3], box[3]) + pad, max(min(anc[1], box[1]) - pad, 0))
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, ec=GREEN, lw=1.6),
                       Rectangle((0, 0), 1, 1, fill=False, ec=ANCHOR, lw=1.4, ls=(0, (4, 2))),
                       Rectangle((0, 0), 1, 1, fill=False, ec=ARM_ORANGE, lw=2.0)],
              labels=["the cell", "its anchor (preset)", f"final box: {t['class']} {t['score']:.2f}"],
              loc="lower left", fontsize=6.5, framealpha=0.85, borderpad=0.3, handlelength=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(FIG_DIR / "ssd_anchor_decode.pdf", dpi=200)
    plt.close(fig)
    log.info("saved sliding_windows.pdf, ssd_grid_anchors.pdf, ssd_anchor_decode.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="redraw from the saved json")
    args = ap.parse_args()
    log = setup_logging()
    t0 = time.perf_counter()
    FIG_DIR.mkdir(exist_ok=True)
    if args.plot_only:
        if not RESULTS.exists():
            raise FileNotFoundError(f"{RESULTS} missing - run without --plot-only first")
        r = json.loads(RESULTS.read_text(encoding="utf-8"))
    else:
        from PIL import Image
        img = Image.open(PHOTO).convert("RGB")
        r = {"windows": sliding_windows(img, log), "ssd": ssd_top_detection(img, log)}
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(r, indent=1), encoding="utf-8")
        log.info(f"saved {RESULTS}")
    plot(r, log)
    log.info(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
