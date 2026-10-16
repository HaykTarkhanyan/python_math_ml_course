"""Record your own thumb-direction dataset from the webcam.

Five classes: up, down, left, right, nothing.
Keys:  W = up    S = down    A = left    D = right    N = nothing
       B = move the box to the other side    Q or Esc = quit

Each key press records one BURST: a 2 s countdown, then 50 frames over about 5 s. During a
burst keep your thumb in the box and move a little - nearer, farther, slightly tilted.
Record 2-3 bursts per class and change something between them (where you sit, the light).
"nothing" = no command: an empty box, a fist without the thumb, an open palm, a hand passing.

Directions are YOUR directions. The preview is mirrored like a selfie camera and the saved crops
are mirrored too, so a "left" image shows the thumb pointing to the left of the picture. The
notebook and play_snake.py use the same mirror (gesture_common.py), so all three agree.

Saved: gesture_data/<user>/<class>/<user>_<burst>_<nn>.jpg, 128x128 JPEG, about 5 KB each.
The burst id (a timestamp) is kept in the name because frames from one burst are near-copies:
the notebook splits by burst, never at random.

Run (about 3 minutes for two bursts of every class):
    python record_gestures.py --user anna
Needs opencv-python and numpy (tested with opencv-python 4.11.0.86, numpy 1.26.4).
"""
import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import cv2

from gesture_common import CLASSES, box_coords, crop_box, imwrite_u, mirror

KEYS = {ord("w"): "up", ord("s"): "down", ord("a"): "left", ord("d"): "right", ord("n"): "nothing"}
BURST_FRAMES = 50
BURST_SECONDS = 5.0
COUNTDOWN_SECONDS = 2.0
WINDOW = "record gestures - W/A/S/D/N record, B move box, Q quit"
HERE = Path(__file__).resolve().parent

log = logging.getLogger("record_gestures")


def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler("logs/record_gestures.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def draw_hud(view, box, counts, status, recording):
    x0, y0, s = box
    color = (0, 0, 255) if recording else (0, 200, 0)
    cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), color, 3)
    cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
    panel = view[30:45 + 24 * len(CLASSES), :190]
    panel //= 3  # darken in place, so the counts read on a white wall too
    cv2.putText(view, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for i, c in enumerate(CLASSES):
        cv2.putText(view, f"{c}: {counts[c]}", (8, 55 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="your name in Latin letters, e.g. anna")
    ap.add_argument("--camera", type=int, default=0, help="camera index (try 1 if 0 is wrong)")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9-]+", args.user):
        raise SystemExit(f"--user must be Latin letters, digits or '-', got {args.user!r}")

    setup_logging()
    user_dir = HERE / "gesture_data" / args.user
    for c in CLASSES:
        (user_dir / c).mkdir(parents=True, exist_ok=True)
    counts = {c: len(list((user_dir / c).glob("*.jpg"))) for c in CLASSES}
    log.info(f"saving to {user_dir}; already there: {counts}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {args.camera}. Close other apps that use the "
                           "webcam (Zoom, Teams, a browser tab) or try --camera 1")
    side, burst = "right", None
    cv2.namedWindow(WINDOW)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("the camera stopped returning frames")
            frame = mirror(frame)
            box = box_coords(frame.shape, side)
            status = "W up  S down  A left  D right  N nothing  |  B move box  Q quit"

            if burst is not None:
                t = time.monotonic() - burst["start"]
                if t < COUNTDOWN_SECONDS:
                    status = f"get ready: {burst['label'].upper()} in {COUNTDOWN_SECONDS - t:.1f} s"
                else:
                    due = min(BURST_FRAMES, int((t - COUNTDOWN_SECONDS) * BURST_FRAMES / BURST_SECONDS) + 1)
                    if burst["saved"] < due:
                        name = f"{args.user}_{burst['id']}_{burst['saved']:02d}.jpg"
                        imwrite_u(user_dir / burst["label"] / name, crop_box(frame, box))
                        burst["saved"] += 1
                    status = f"RECORDING {burst['label'].upper()}  {burst['saved']}/{BURST_FRAMES}"
                    if burst["saved"] == BURST_FRAMES:
                        counts[burst["label"]] += BURST_FRAMES
                        log.info(f"burst {burst['id']}: {BURST_FRAMES} x {burst['label']}; totals {counts}")
                        burst = None

            view = frame.copy()
            draw_hud(view, box, counts, status, recording=burst is not None)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += 32  # caps lock on: treat W like w
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if burst is None and key == ord("b"):
                side = "left" if side == "right" else "right"
            elif burst is None and key in KEYS:
                burst = {"label": KEYS[key], "start": time.monotonic(), "saved": 0,
                         "id": datetime.now().strftime("%Y%m%d-%H%M%S")}
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if burst is not None:
        log.warning(f"quit mid-burst: {burst['saved']} '{burst['label']}' frames of burst "
                    f"{burst['id']} were saved; delete them if the burst was bad")
    log.info(f"done; totals {counts}")


if __name__ == "__main__":
    main()
