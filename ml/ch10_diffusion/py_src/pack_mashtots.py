"""Pack the five ՊԱՆԻՐ letter folders into one small .npz.

The raw Kaggle download is 70k PNGs behind a competition login; this is the ~2.6 MB file
that actually ships with the course, so students need no Kaggle account.

Preprocessing, in order (each step was measured, see DECISIONS.md):
  1. crop to the ink bounding box, pad square with a 12% margin - the glyph fills only
     ~34-40 px of the 64 px frame, and resizing the raw frame dissolves the thin strokes
  2. resize to 24x24
  3. per-image contrast normalisation - antialiased strokes peak around 150-190/255

Labels are 0..4 in ՊԱՆԻՐ order, so y=[0,1,2,3,4] spells the word.

Reads:  data/mashtots_raw/  (run extract_mashtots.py first)
Writes: data/mashtots_panir_24.npz
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# This script logs Armenian letters; the Windows console defaults to cp1252 and mangles them.
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8")

CHAPTER = Path(__file__).resolve().parent.parent
RAW = CHAPTER / "data" / "mashtots_raw" / "Train" / "Train"
# Output resolution. 24 is the chapter default (DECISIONS.md #6); pass another on the command
# line to test whether the thin strokes survive better at a larger size:
#   python pack_mashtots.py 32
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 24
OUT = CHAPTER / "data" / f"mashtots_panir_{SIZE}.npz"

LOGS = CHAPTER.parent.parent / "logs"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "pack_mashtots.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# (competition folder id, letter) in ՊԱՆԻՐ order -> label 0..4
LETTERS = [(26, "Պ"), (0, "Ա"), (21, "Ն"), (10, "Ի"), (32, "Ր")]
THRESH = 40
MIN_INK = 0.01      # one image in the set is essentially blank; drop it loudly


def crop_to_ink(a, pad_frac=0.12):
    ys, xs = np.where(a > THRESH)
    top, bot, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = bot - top + 1, right - left + 1
    side = int(max(h, w) * (1 + 2 * pad_frac))
    out = np.zeros((side, side), dtype=a.dtype)
    y0, x0 = side // 2 - h // 2, side // 2 - w // 2
    out[y0:y0 + h, x0:x0 + w] = a[top:bot + 1, left:right + 1]
    return out


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"missing extracted data: {RAW} - run extract_mashtots.py first")

    xs, ys, dropped = [], [], []
    for label, (cid, arm) in enumerate(LETTERS):
        files = sorted((RAW / str(cid)).glob("*.png"))
        if not files:
            raise RuntimeError(f"no PNGs for class {cid} in {RAW}")
        kept = 0
        for f in files:
            a = np.asarray(Image.open(f).convert("L"))
            ink = float((a > THRESH).mean())
            if ink < MIN_INK:
                log.warning("dropping near-blank image (ink=%.4f): %s", ink, f.relative_to(RAW))
                dropped.append(str(f.relative_to(RAW)))
                continue
            c = crop_to_ink(a)
            c = np.asarray(Image.fromarray(c).resize((SIZE, SIZE), Image.LANCZOS)).astype(np.float32)
            peak = c.max()
            if peak <= 0:
                raise RuntimeError(f"all-zero image after resize: {f}")
            xs.append(np.clip(c * (255.0 / peak), 0, 255).astype(np.uint8))
            ys.append(label)
            kept += 1
        log.info("label %d  %s  folder %-2d  kept %4d / %4d", label, arm, cid, kept, len(files))

    X = np.stack(xs)
    y = np.array(ys, dtype=np.int64)
    letters = np.array([arm for _, arm in LETTERS])
    folder_ids = np.array([cid for cid, _ in LETTERS], dtype=np.int64)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y, letters=letters, folder_ids=folder_ids)

    log.info("X %s %s   y %s   classes %s", X.shape, X.dtype, y.shape, np.bincount(y).tolist())
    log.info("dropped %d image(s): %s", len(dropped), dropped or "none")
    log.info("wrote %s (%.2f MB)", OUT, OUT.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
