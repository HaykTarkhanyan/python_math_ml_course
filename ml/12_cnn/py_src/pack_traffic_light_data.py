"""Copy the traffic-light data the L16 practical needs out of ml/claude_projects/traffic_light_state.

Writes ml/12_cnn/extra_traffic_light/data/ (3.1 MB; git-ignored, see the practical's README):
  s2tld/<crop_id>.png        341 dashcam crops in scope (red / yellow / green / off), verified
  cams/<cam>/<frame>_<head>.png  153 hand-labeled crops from 5 public junction cams
  context/<name>.jpg         a few full frames (downscaled) to show where the crops come from
  labels.json                every crop's label + lighting / camera / hard tag, and the
                             context frames' head boxes (in downscaled pixels)

The source project evaluates crops cut from the full frames at run time (the PNGs in its crops/
folders are padded, 192-px-tall copies made for visual review - not what was scored). So the
crops are cut here from the frames with the exact boxes and saved as lossless PNG, then read
back and compared, so the notebook sees the pixels the project's numbers were measured on.
Anything missing or different raises.

Also writes data/cnn_reference.json: the project's tiny-CNN accuracies (S2TLD by lighting,
cams by camera) for the practical's closing teaser - the CNN itself is the next practical.

Run once (~30 s); --reference-only rewrites just cnn_reference.json:
    ./ma/Scripts/python.exe ml/12_cnn/py_src/pack_traffic_light_data.py [--reference-only]
"""

import json
import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
SRC = REPO_ROOT / "ml" / "claude_projects" / "traffic_light_state" / "data"
DST = CH / "extra_traffic_light" / "data"
LOGS_DIR = REPO_ROOT / "logs"
CLASSES = ["red", "yellow", "green", "off"]
CONTEXT_W = 960                 # downscaled width of the context frames

log = logging.getLogger("pack_traffic_light_data")


def save_slice(frame, box, dst):
    """Cut the exact box, save lossless, read back and compare."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"empty crop for {dst.name} box {box}")
    cv2.imwrite(str(dst), crop)
    back = cv2.imread(str(dst))
    if back is None or not np.array_equal(back, crop):
        raise ValueError(f"{dst} did not round-trip losslessly")


def save_context(frame, boxes, name, ctx):
    scale = CONTEXT_W / frame.shape[1]
    small = cv2.resize(frame, (CONTEXT_W, round(frame.shape[0] * scale)),
                       interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DST / "context" / f"{name}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    ctx[name] = [[round(v * scale) for v in b] for b in boxes]


def pack_s2tld(labels, ctx):
    ann = json.loads((SRC / "eval_smoke" / "annotations.json").read_text(encoding="utf-8"))
    out, shown = [], set()
    for fr in ann["frames"]:
        frame = cv2.imread(str(SRC / "eval_smoke" / "frames" / f"{fr['frame_id']}.jpg"))
        if frame is None:
            raise FileNotFoundError(f"missing frame {fr['frame_id']}")
        boxes = []
        for o in fr["objects"]:
            if not o.get("verified") or o.get("excluded") or o["state"] not in CLASSES:
                continue
            save_slice(frame, o["box"], DST / "s2tld" / f"{o['crop_id']}.png")
            out.append({"crop_id": o["crop_id"], "frame_id": fr["frame_id"],
                        "sequence": fr["source"], "lighting": fr["lighting"],
                        "state": o["state"], "hard": bool(o.get("hard")),
                        "note": o.get("note", "")})
            boxes.append(o["box"])
        # one context frame per lighting condition, the first with at least two heads
        if fr["lighting"] not in shown and len(boxes) >= 2:
            save_context(frame, boxes, f"s2tld_{fr['lighting']}", ctx)
            shown.add(fr["lighting"])
    labels["s2tld"] = out
    log.info(f"s2tld: {len(out)} crops | {dict(Counter(r['state'] for r in out))} | "
             f"{dict(Counter(r['lighting'] for r in out))}")


def pack_cams(labels, ctx):
    states = json.loads((SRC / "raw" / "cam_states.json").read_text(encoding="utf-8"))
    calib = json.loads((SRC / "raw" / "cam_calib.json").read_text(encoding="utf-8"))
    out = []
    for cam, lab in states.items():
        heads = {h["id"]: h["box"] for h in calib[cam]}
        (DST / "cams" / cam).mkdir(parents=True, exist_ok=True)
        for key, state in lab.items():
            if state == "unknown":
                continue
            frame_name, head = key.rsplit("_", 1)
            frame = cv2.imread(str(SRC / "raw" / "cam_frames" / cam / f"{frame_name}.jpg"))
            if frame is None:
                raise FileNotFoundError(f"missing cam frame {cam}/{frame_name}")
            save_slice(frame, heads[head], DST / "cams" / cam / f"{key}.png")
            out.append({"cam": cam, "key": key, "state": state})
        first = sorted(lab)[0].rsplit("_", 1)[0]
        frame = cv2.imread(str(SRC / "raw" / "cam_frames" / cam / f"{first}.jpg"))
        save_context(frame, [h["box"] for h in calib[cam]], f"cam_{cam}", ctx)
    labels["cams"] = out
    log.info(f"cams: {len(out)} crops | {dict(Counter(r['state'] for r in out))} | "
             f"{dict(Counter(r['cam'] for r in out))}")


def pack_cnn_reference():
    """The tiny CNN's numbers from the project's artifact of record, for the teaser."""
    res = json.loads((SRC.parent / "out" / "eval_results.json").read_text(encoding="utf-8"))
    s2, cam = res["approaches"]["tiny_cnn"], res["cam_slice"]["approaches"]["tiny_cnn"]
    ref = {"source": "ml/claude_projects/traffic_light_state/out/eval_results.json "
                     "(train_tiny_cnn.py: ~25k parameters, 32x32 input, trained on S2TLD "
                     "frames outside this eval set - same cameras)",
           "s2tld": {"accuracy": s2["accuracy"], "n": res["n"],
                     "by_lighting": {k: v["acc"] for k, v in s2["by_lighting"].items()}},
           "cams": {"accuracy": cam["accuracy"], "n": res["cam_slice"]["n"],
                    "by_cam": {k: v["acc"] for k, v in cam["by_cam"].items()}}}
    (DST / "cnn_reference.json").write_text(json.dumps(ref, indent=1), encoding="utf-8")
    log.info(f"cnn reference: s2tld {ref['s2tld']['accuracy']:.3f}, cams {ref['cams']['accuracy']:.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-only", action="store_true")
    args = ap.parse_args()
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "pack_traffic_light_data.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    t0 = time.perf_counter()
    if args.reference_only:
        pack_cnn_reference()
        return
    if DST.exists():
        raise FileExistsError(f"{DST} already exists - move it aside to repack")
    for sub in ("s2tld", "cams", "context"):
        (DST / sub).mkdir(parents=True)
    labels, ctx = {"source": "ml/claude_projects/traffic_light_state (S2TLD smoke eval v1 + "
                             "traffic-cam slice), packed 2026-10-02"}, {}
    pack_s2tld(labels, ctx)
    pack_cams(labels, ctx)
    labels["context_boxes"] = ctx
    (DST / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    pack_cnn_reference()
    size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1e6
    log.info(f"wrote {DST} ({size:.1f} MB) in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
