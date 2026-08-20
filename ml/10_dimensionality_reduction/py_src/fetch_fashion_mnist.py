#!/usr/bin/env python
"""Cache a stratified Fashion-MNIST subsample for the dimensionality-reduction chapter.

Instructor-side, run once. The deck's figure script (``dimred_demos.py``) and the
homework load the committed ``data/fashion_mnist.npz`` instead of downloading
anything, so the chapter is reproducible offline and needs nothing but numpy.

Reads the raw IDX files torchvision downloads. If they are missing, fetch them with:

    ./ma/Scripts/python.exe -c "from torchvision.datasets import FashionMNIST; \
        FashionMNIST(root=r'C:/Users/hayk_/.cache/torchvision', download=True)"

Run:  ./ma/Scripts/python.exe ml/10_dimensionality_reduction/py_src/fetch_fashion_mnist.py
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
OUT_NPZ = CHAPTER / "data" / "fashion_mnist.npz"
RAW = Path.home() / ".cache" / "torchvision" / "FashionMNIST" / "raw"

SEED = 509
PER_CLASS = 1200          # 10 classes -> 12,000 images

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def setup_logging() -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(logs / "fetch_fashion_mnist.log", encoding="utf-8")],
    )


def _read_idx(name: str) -> np.ndarray:
    """Read one IDX file, preferring the plain copy and falling back to the .gz."""
    plain, gz = RAW / name, RAW / f"{name}.gz"
    if plain.exists():
        buf = plain.read_bytes()
    elif gz.exists():
        buf = gzip.decompress(gz.read_bytes())
    else:
        raise FileNotFoundError(
            f"Neither {plain} nor {gz} exists. Download Fashion-MNIST first "
            f"(see this file's docstring)."
        )

    magic, n = int.from_bytes(buf[:4], "big"), int.from_bytes(buf[4:8], "big")
    if magic == 2051:                                   # images: 28x28 uint8
        rows = int.from_bytes(buf[8:12], "big")
        cols = int.from_bytes(buf[12:16], "big")
        return np.frombuffer(buf, np.uint8, offset=16).reshape(n, rows, cols)
    if magic == 2049:                                   # labels
        return np.frombuffer(buf, np.uint8, offset=8)
    raise ValueError(f"{name}: unexpected IDX magic number {magic}")


def main() -> None:
    setup_logging()
    OUT_NPZ.parent.mkdir(exist_ok=True)

    images = _read_idx("train-images-idx3-ubyte")
    labels = _read_idx("train-labels-idx1-ubyte")
    logging.info("read %s images, %s labels from %s", images.shape, labels.shape, RAW)

    rng = np.random.RandomState(SEED)
    keep = np.concatenate([
        rng.choice(np.flatnonzero(labels == c), PER_CLASS, replace=False)
        for c in range(len(CLASS_NAMES))
    ])
    rng.shuffle(keep)                                   # so a head-slice stays balanced

    np.savez_compressed(
        OUT_NPZ,
        images=images[keep],                            # (N, 28, 28) uint8
        labels=labels[keep].astype(np.int8),
        class_names=np.array(CLASS_NAMES),
        seed=np.int32(SEED),
    )
    mb = OUT_NPZ.stat().st_size / 1e6
    logging.info("wrote %s  (%d images, %.1f MB)", OUT_NPZ, len(keep), mb)


if __name__ == "__main__":
    main()
