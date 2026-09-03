"""Embed YOUR photos with CLIP -- GIVEN code for ch10 Project 3 (own-photos finale).

Point it at a folder of photos and it writes `<folder>_clip.npz` next to it, in the
same layout as the chapter's `imagenette_clip.npz`, so your notebook code and
`photo_map.make_photo_map` work on it unchanged:

    python embed_my_photos.py path/to/my_photos

Output npz keys:
    embeddings     (n, 512) float16 -- L2-normalized CLIP image vectors
    filenames      (n,)             -- so you can find a photo again
    thumb_blob / thumb_offsets      -- small JPEG thumbnails for the hover map

Needs (one-time):  pip install torch transformers pillow
For iPhone HEIC/HEIF photos add:  pip install pillow-heif
CPU is fine: roughly a minute per couple hundred photos, plus a one-time model
download (~600 MB). Any collection works -- trip photos, screenshots, a folder of
paintings. No faces required.

One gotcha: only the top level of the folder is scanned (subfolders are ignored).
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"   # same model as the chapter dataset
BATCH = 16
THUMB_PX = 48
THUMB_QUALITY = 70
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

# iPhone HEIC support is optional: if pillow-heif is installed, PIL learns to open
# .heic/.heif and everything below just works; if not, we fail loudly on such files.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False


def _thumb_jpeg(img: Image.Image) -> bytes:
    """Tiny square thumbnail for the hover map ONLY - this is display, not model input.

    The embedding never sees this: CLIP gets the full image, and CLIPProcessor does the
    model's own preprocessing (resize + center-crop to 224x224) internally. We make a
    uniform 48px square here so a few hundred thumbnails pack into a small HTML file."""
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=THUMB_QUALITY)
    return buf.getvalue()


def main(folder: Path) -> Path:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images ({'/'.join(sorted(EXTS))}) found in {folder} "
                         "(note: subfolders are not scanned)")
    n_heic = sum(p.suffix.lower() in (".heic", ".heif") for p in files)
    if n_heic and not HEIC_OK:
        raise SystemExit(f"{n_heic} HEIC/HEIF photo(s) in {folder} but pillow-heif is not "
                         "installed - run: pip install pillow-heif")
    logging.info("embedding %d photos from %s with %s", len(files), folder, MODEL_ID)

    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()

    vecs, thumbs, names = [], [], []
    for start in range(0, len(files), BATCH):
        batch = files[start:start + BATCH]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        # full images in; the processor handles CLIP's own resize/crop to 224x224
        inputs = processor(images=imgs, return_tensors="pt")
        with torch.no_grad():
            # transformers >= 5 returns a model-output object; .pooler_output is the
            # 512-d CLIP image vector (same call the chapter dataset was built with).
            feats = model.get_image_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.numpy().astype(np.float16))
        thumbs.extend(_thumb_jpeg(im) for im in imgs)
        names.extend(p.name for p in batch)
        logging.info("  %d / %d", min(start + BATCH, len(files)), len(files))

    offsets = np.zeros(len(thumbs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(t) for t in thumbs])
    blob = np.frombuffer(b"".join(thumbs), dtype=np.uint8)

    out = folder.parent / f"{folder.name}_clip.npz"
    np.savez_compressed(out, embeddings=np.vstack(vecs), filenames=np.array(names),
                        thumb_blob=blob, thumb_offsets=offsets,
                        thumb_px=np.int16(THUMB_PX), model_id=MODEL_ID)
    logging.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) != 2:
        raise SystemExit("usage: python embed_my_photos.py path/to/photo_folder")
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    main(folder)
