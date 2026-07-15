"""Real figure for the L17 CNN Architectures deck (Section 2, "what AlexNet learned").

Loads a torchvision resnet18 with pretrained ImageNet weights and plots its first
convolutional layer (conv1: 64 filters, 7x7, RGB) as an 8x8 grid. Each filter is
min-max normalized on its own so the Gabor-like edge / blob / color detectors are
visible. NO training - weights are loaded and plotted only.

Generates into ml/ch6_cnn/fig/:
  pretrained_filters.pdf

First run downloads the resnet18 weights once (~45 MB, sanctioned).

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/pretrained_filters.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

SEED = 509

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l17_pretrained_filters")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "pretrained_filters.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    log.info("loading resnet18 pretrained weights (one-time download if not cached)")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    w = model.conv1.weight.data.clone().numpy()   # (64, 3, 7, 7)
    log.info(f"conv1 weight shape: {w.shape}")
    if w.shape != (64, 3, 7, 7):
        raise ValueError(f"unexpected conv1 shape {w.shape}; expected (64,3,7,7)")

    n = w.shape[0]  # 64
    fig, axes = plt.subplots(8, 8, figsize=(6.4, 6.4))
    for idx in range(n):
        ax = axes[idx // 8, idx % 8]
        f = w[idx].transpose(1, 2, 0)             # (7,7,3)
        fmin, fmax = f.min(), f.max()
        f = (f - fmin) / (fmax - fmin + 1e-9)     # per-filter min-max normalize
        ax.imshow(f, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("ResNet-18 first-layer filters, learned on ImageNet (64 x 7x7 RGB)",
                 fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                        wspace=0.12, hspace=0.12)
    out = FIG_DIR / "pretrained_filters.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
