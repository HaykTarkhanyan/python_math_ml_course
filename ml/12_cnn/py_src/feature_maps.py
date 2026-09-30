"""Real figure for the L16 CNN Foundations deck (Section 5, what the layers learn).

Generates into ml/ch6_cnn/fig/:
  feature_maps.pdf  -- the 8 first-layer 3x3 kernels a small CNN learned on Fashion-MNIST
                       (nobody designed them), each above the feature map it produces on a
                       sample image. The "kernels are learned" payoff.

Trains the chapter's small CNN (9,098 params) on a 15k Fashion-MNIST subsample for 15
epochs - a few minutes on CPU. (Same config as cnn_vs_mlp.py, so the learned kernels shown
here are the same net that wins the payoff figure.)

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/feature_maps.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SEED = 509
torch.manual_seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_feature_maps")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "feature_maps.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


class SmallCNN(nn.Module):
    """Chapter CNN: conv(1->8,3,pad1)-relu-pool2 - conv(8->16,3,pad1)-relu-pool2 - fc(784->10).

    9,098 params. This exact class also lives in cnn_vs_mlp.py - keep them identical.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.fc(x.flatten(1))


def load_fmnist(n_train=8000, n_test=2000):
    from tensorflow.keras.datasets import fashion_mnist
    (xtr, ytr), (xte, yte) = fashion_mnist.load_data()
    rng = np.random.default_rng(SEED)
    tri = rng.choice(len(xtr), n_train, replace=False)
    tei = rng.choice(len(xte), n_test, replace=False)
    Xtr = torch.tensor(xtr[tri][:, None].astype("float32") / 255.0)
    ytr_t = torch.tensor(ytr[tri].astype("int64"))
    Xte = torch.tensor(xte[tei][:, None].astype("float32") / 255.0)
    yte_t = torch.tensor(yte[tei].astype("int64"))
    return Xtr, ytr_t, Xte, yte_t


def train(model, Xtr, ytr, epochs=5, lr=1e-3, batch=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    return model


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = load_fmnist(n_train=15000)
    log.info(f"data train {tuple(Xtr.shape)} test {tuple(Xte.shape)}")

    model = SmallCNN()
    nparams = sum(p.numel() for p in model.parameters())
    log.info(f"SmallCNN params = {nparams} (expect 9098)")
    train(model, Xtr, ytr, epochs=15)
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    log.info(f"test acc {acc:.3f}")

    # a sample image (an ankle-boot / sneaker reads well), conv1 kernels + activations
    sample = Xte[0:1]
    with torch.no_grad():
        act = torch.relu(model.conv1(sample))[0]        # (8, 28, 28)
    kernels = model.conv1.weight.detach()[:, 0]         # (8, 3, 3)

    fig, axes = plt.subplots(2, 9, figsize=(11.5, 3.2))
    axes[0, 0].axis("off")
    axes[0, 0].text(0.5, 0.5, "kernel", ha="center", va="center", fontsize=10)
    axes[1, 0].imshow(sample[0, 0], cmap="gray")
    axes[1, 0].set_title("input", fontsize=9); axes[1, 0].axis("off")
    for k in range(8):
        axes[0, k + 1].imshow(kernels[k], cmap="gray"); axes[0, k + 1].axis("off")
        axes[1, k + 1].imshow(act[k], cmap="gray"); axes[1, k + 1].axis("off")
    fig.suptitle("First-layer kernels a CNN learned on Fashion-MNIST (top) "
                 "and their feature maps on one image (bottom)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "feature_maps.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}  (test acc {acc:.3f}, params {nparams})")


if __name__ == "__main__":
    main()
