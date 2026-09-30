"""Shared by record_gestures.py, the training notebook and play_snake.py.

Everything that must be IDENTICAL at recording, training and play time lives here: the class
list, where the box sits on the frame, how a crop is cut and resized, and how a crop becomes the
tensor the model eats. If the three disagreed (say, one of them forgot the mirror), "left" would
silently turn into "right".

Model contract (what the notebook exports and play_snake.py loads):
    input : float32 tensor, shape (N, 3, CROP_SIZE, CROP_SIZE), RGB, values in [0, 1]
    output: logits, shape (N, len(CLASSES)), in the order of CLASSES
Anything else the model needs (resizing, normalization) goes INSIDE the exported model.
"""
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("up", "down", "left", "right", "nothing")
CROP_SIZE = 128   # saved crops are CROP_SIZE x CROP_SIZE pixels
BOX_FRAC = 0.6    # box side as a fraction of the frame height
BOX_MARGIN = 0.05  # gap between the box and the frame edge, as a fraction of the frame width


def mirror(frame):
    """Selfie view: your left is the screen's left. Recording and playing both use it."""
    return cv2.flip(frame, 1)


def box_coords(frame_shape, side="right"):
    """(x0, y0, side_px) of the square box on a frame of shape (h, w, ...)."""
    h, w = frame_shape[:2]
    s = int(BOX_FRAC * h)
    y0 = (h - s) // 2
    margin = int(BOX_MARGIN * w)
    x0 = w - s - margin if side == "right" else margin
    if x0 < 0 or s > w:
        raise ValueError(f"frame {w}x{h} is too narrow for a {s} px box")
    return x0, y0, s


def crop_box(frame, box):
    """Cut the box out of a (mirrored) BGR frame and resize it to CROP_SIZE."""
    x0, y0, s = box
    return cv2.resize(frame[y0:y0 + s, x0:x0 + s], (CROP_SIZE, CROP_SIZE),
                      interpolation=cv2.INTER_AREA)


def imwrite_u(path, img, quality=90):
    """cv2.imwrite fails SILENTLY on non-ANSI Windows paths (e.g. an Armenian user name).
    Encode in memory and write through numpy instead, and raise if anything goes wrong."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"could not encode image for {path}")
    buf.tofile(str(path))


def imread_u(path):
    """cv2.imread twin of imwrite_u: returns a BGR uint8 array or raises."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"unreadable image: {path}")
    return img


def bgr_to_tensor(imgs):
    """BGR uint8 image(s), (H, W, 3) or (N, H, W, 3) -> float tensor (N, 3, H, W), RGB, [0, 1]."""
    import torch  # imported here so that recording works without torch installed
    arr = np.asarray(imgs)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) BGR images, got shape {arr.shape}")
    rgb = np.ascontiguousarray(arr[..., ::-1])
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0


def parse_name(path):
    """gesture_data/<user>/<class>/<user>_<burst>_<nn>.jpg -> (user, burst, class)."""
    path = Path(path)
    user, burst, _ = path.stem.rsplit("_", 2)
    return user, burst, path.parent.name
