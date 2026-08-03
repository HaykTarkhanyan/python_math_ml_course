"""Shared setup for the GAN chapter figures.

Self-contained rather than importing from ch10's py_src - a cross-chapter import would
couple two chapters that should be independently runnable.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits

SEED = 509
LATENT = 32

# Freeze-safety (HARD, per AE_CHAPTER_PLAN.md): 16 GB laptop, integrated graphics,
# documented lock-up history. Never remove; never run these scripts in parallel.
torch.set_num_threads(4)


def setup_logging(name):
    log_dir = Path(__file__).resolve().parents[3] / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / f"{name}.log", mode="w", encoding="utf-8")],
        force=True,
    )
    return logging.getLogger(name)


def fig_dir():
    d = Path(__file__).resolve().parent.parent / "fig"
    d.mkdir(exist_ok=True)
    return d


def save(fig, name, log):
    path = fig_dir() / name
    fig.savefig(path, bbox_inches="tight")
    log.info(f"wrote {path}")
    return path


def load_data():
    """8x8 digits scaled to [-1, 1], matching the diffusion chapter exactly so the two
    families can be compared on identical data."""
    d = load_digits()
    x = d.images.astype(np.float32) / 16.0 * 2.0 - 1.0
    return torch.tensor(x).reshape(len(x), -1), d.target


class Generator(nn.Module):
    def __init__(self, latent=LATENT, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 64), nn.Tanh(),   # tanh -> [-1, 1], the data range
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, hidden), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden, 1),               # logits; BCEWithLogits applies the sigmoid
        )

    def forward(self, x):
        return self.net(x)
