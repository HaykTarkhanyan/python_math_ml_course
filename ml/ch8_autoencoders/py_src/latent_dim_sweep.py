"""Reconstructions vs latent dimension - the 'how big should the code be?' figure for L22.

Trains a tiny MLP autoencoder at latent dims {2, 8, 32} on cached MNIST and shows how the
reconstruction sharpens as the code grows. Light: thread-capped, ~30s, no download. Seed 509.
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)
torch.manual_seed(509); np.random.seed(509)

CH = Path(__file__).resolve().parent.parent
FIG = CH / "fig"; LOGS = CH / "logs"
FIG.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOGS / "latent_dim_sweep.log", encoding="utf-8"),
                              logging.StreamHandler()])
log = logging.getLogger("sweep")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AE(nn.Module):
    """One hidden layer each side, so the LATENT is the true bottleneck (not a hidden layer)."""
    def __init__(self, latent):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(),
                                 nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def main():
    from torchvision import datasets, transforms
    tr = datasets.MNIST(str(CH / "practical" / "data"), train=True, download=False,
                        transform=transforms.ToTensor())
    te = datasets.MNIST(str(CH / "practical" / "data"), train=False, download=False,
                        transform=transforms.ToTensor())
    X = tr.data.float().div(255).view(-1, 784)[:20000]
    Xte = te.data.float().div(255).view(-1, 784)
    yte = te.targets

    dims = [2, 8, 32]
    recons = {}
    for d in dims:
        m = AE(d)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        for ep in range(14):
            for (xb,) in DataLoader(TensorDataset(X), batch_size=256, shuffle=True):
                loss = F.binary_cross_entropy(m(xb), xb)
                opt.zero_grad(); loss.backward(); opt.step()
        log.info(f"latent {d}: final bce {loss.item():.4f}")
        with torch.no_grad():
            idx = torch.tensor([int((yte == c).nonzero()[0, 0]) for c in range(8)])
            recons[d] = m(Xte[idx]).numpy()
    with torch.no_grad():
        originals = Xte[torch.tensor([int((yte == c).nonzero()[0, 0]) for c in range(8)])].numpy()

    rows = [("original", originals)] + [(f"latent {d}", recons[d]) for d in dims]
    fig, axs = plt.subplots(len(rows), 8, figsize=(9, 4.5))
    for r, (lab, imgs) in enumerate(rows):
        for c in range(8):
            axs[r, c].imshow(imgs[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axs[r, c].axis("off")
        axs[r, 0].axis("on"); axs[r, 0].set_xticks([]); axs[r, 0].set_yticks([])
        axs[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=11)
    fig.suptitle("Bigger code $\\to$ sharper reconstruction", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "latent_dim_recon.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("saved latent_dim_recon.pdf")


if __name__ == "__main__":
    main()
