"""Encode a sample of Imagenette photos with CLIP and cache the vectors.

Instructor-side, run once. Students never run this - they load the committed
`data/imagenette_clip.npz`, which holds the embeddings, the labels, small JPEG
thumbnails and a set of text-prompt embeddings, and needs nothing but numpy.

The point of caching: the practical is about clustering, not about installing torch.

Run:  ./ma/Scripts/python.exe ml/09_clustering/py_src/embed_images_clip.py
"""

from __future__ import annotations

import io
import logging
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# --- configuration -------------------------------------------------------------

CHAPTER = Path(__file__).resolve().parents[1]
OUT_NPZ = CHAPTER / "data" / "imagenette_clip.npz"

CACHE = Path.home() / ".cache" / "imagenette"
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
TARBALL = CACHE / "imagenette2-320.tgz"

MODEL_ID = "openai/clip-vit-base-patch32"
SEED = 509
PER_CLASS = 200          # 10 classes -> 2000 images
BATCH = 32
THUMB_PX = 48
THUMB_QUALITY = 70
THREADS = max(1, (torch.get_num_threads() or 8) - 2)   # leave the machine usable

# Imagenette's ten WordNet ids, in the order fast.ai ships them.
CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer spaniel",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

# Deliberately larger than the ten true classes, so naming a cluster from text is a real
# retrieval over a vocabulary rather than a ten-way lookup with the answer already in it.
DISTRACTORS = [
    "cat", "horse", "bicycle", "laptop", "coffee cup", "mountain", "beach", "pizza",
    "guitar", "airplane", "train", "boat", "flower", "bridge", "skyscraper", "tractor",
    "violin", "drum kit", "telephone", "camera", "clock", "chair", "book", "shoe",
    "umbrella", "helicopter", "motorcycle", "snake", "bird", "tree",
]
PROMPT = "a photo of a {}"

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(Path("logs") / "embed_images_clip.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)


# --- data ----------------------------------------------------------------------


def download() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if TARBALL.exists():
        log.info("tarball already cached: %s (%.0f MB)", TARBALL, TARBALL.stat().st_size / 1e6)
        return
    log.info("downloading %s", URL)
    urllib.request.urlretrieve(URL, TARBALL)
    log.info("saved %s (%.0f MB)", TARBALL, TARBALL.stat().st_size / 1e6)


def wnid_of(name: str) -> str | None:
    """imagenette2-320/train/n03000684/xxx.JPEG -> n03000684"""
    parts = name.split("/")
    return parts[2] if len(parts) > 3 and parts[2] in CLASSES else None


def choose_members() -> dict[str, list[str]]:
    """First pass over the archive: list every training image, then sample per class."""
    rng = np.random.default_rng(SEED)
    by_class: dict[str, list[str]] = {w: [] for w in CLASSES}
    with tarfile.open(TARBALL, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or "/train/" not in member.name:
                continue
            wnid = wnid_of(member.name)
            if wnid:
                by_class[wnid].append(member.name)

    chosen = {}
    for wnid, names in by_class.items():
        if len(names) < PER_CLASS:
            raise RuntimeError(f"{wnid} has only {len(names)} images, need {PER_CLASS}")
        idx = rng.choice(len(names), PER_CLASS, replace=False)
        chosen[wnid] = sorted(names[i] for i in idx)
        log.info("  %-12s %-26s %5d available -> %d sampled",
                 wnid, CLASSES[wnid], len(names), PER_CLASS)
    return chosen


def thumbnail_bytes(img: Image.Image) -> bytes:
    """Square centre crop, small, JPEG-encoded - what the hover tooltip will show."""
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    img = img.resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=THUMB_QUALITY)
    return buf.getvalue()


# --- main ----------------------------------------------------------------------


def main() -> None:
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    log.info("torch threads: %d", THREADS)

    download()
    log.info("first pass: listing training images")
    chosen = choose_members()
    wanted = {name: wnid for wnid, names in chosen.items() for name in names}
    log.info("sampled %d images across %d classes", len(wanted), len(CLASSES))

    log.info("loading %s", MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    embeddings, labels, thumbs, sources = [], [], [], []
    batch_imgs, batch_meta = [], []
    wnid_index = {w: i for i, w in enumerate(CLASSES)}

    def flush() -> None:
        if not batch_imgs:
            return
        with torch.no_grad():
            inputs = processor(images=batch_imgs, return_tensors="pt")
            feats = model.get_image_features(**inputs).pooler_output
        embeddings.append(feats.numpy().astype(np.float32))
        for name, wnid in batch_meta:
            labels.append(wnid_index[wnid])
            sources.append(name)
        batch_imgs.clear()
        batch_meta.clear()

    log.info("second pass: reading and embedding (%d images, batch %d)", len(wanted), BATCH)
    done = 0
    with tarfile.open(TARBALL, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.name not in wanted:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise RuntimeError(f"could not read {member.name} from the archive")
            img = Image.open(io.BytesIO(handle.read())).convert("RGB")
            thumbs.append(thumbnail_bytes(img))
            batch_imgs.append(img)
            batch_meta.append((member.name, wanted[member.name]))
            if len(batch_imgs) == BATCH:
                flush()
                done += BATCH
                if done % 320 == 0:
                    log.info("  %d / %d embedded", done, len(wanted))
    flush()

    X = np.concatenate(embeddings, axis=0)
    y = np.array(labels, dtype=np.int16)
    if X.shape[0] != len(wanted):
        raise RuntimeError(f"embedded {X.shape[0]} images but sampled {len(wanted)}")
    log.info("image embeddings: %s", X.shape)

    log.info("embedding the text vocabulary")
    vocabulary = list(CLASSES.values()) + DISTRACTORS
    with torch.no_grad():
        text_inputs = processor(text=[PROMPT.format(v) for v in vocabulary],
                                return_tensors="pt", padding=True)
        T = model.get_text_features(**text_inputs).pooler_output.numpy().astype(np.float32)
    log.info("text embeddings: %s over %d candidate words", T.shape, len(vocabulary))

    thumb_blob = np.frombuffer(b"".join(thumbs), dtype=np.uint8)
    thumb_offsets = np.cumsum([0] + [len(t) for t in thumbs]).astype(np.int64)

    np.savez_compressed(
        OUT_NPZ,
        embeddings=X.astype(np.float16),
        labels=y,
        class_names=np.array(list(CLASSES.values())),
        wnids=np.array(list(CLASSES)),
        text_embeddings=T.astype(np.float16),
        vocabulary=np.array(vocabulary),
        prompt_template=PROMPT,
        thumb_blob=thumb_blob,
        thumb_offsets=thumb_offsets,
        thumb_px=np.int16(THUMB_PX),
        sources=np.array(sources),
        model_id=MODEL_ID,
        seed=np.int32(SEED),
    )
    log.info("wrote %s (%.1f MB)", OUT_NPZ, OUT_NPZ.stat().st_size / 1e6)
    log.info("  embeddings %s fp16 | thumbs %.1f MB | %d text vectors",
             X.shape, thumb_blob.nbytes / 1e6, T.shape[0])


if __name__ == "__main__":
    main()
