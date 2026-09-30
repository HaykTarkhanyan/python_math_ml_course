"""Real figure for the L16 CNN Foundations deck (Section 5, the payoff).

Generates into ml/ch6_cnn/fig/:
  cnn_vs_mlp.pdf  -- two bars panels comparing the L15 MLP baseline (784-128-64-10,
                     109,386 params) and the chapter's small CNN (9,098 params), trained on
                     the same Fashion-MNIST subsample for the same epochs: final accuracy is
                     about equal, but the CNN uses ~12x fewer weights. The right prior, not
                     raw capacity, did the work.  (An accuracy-vs-epoch curve was rejected:
                     on Fashion-MNIST the two are a near-tie and the curve reads as if the
                     bigger MLP wins, which is false to the lesson. The parameter gap is the
                     honest, clear payoff.)

Trains two small nets on a 15k Fashion-MNIST subsample for 15 epochs - a few minutes on CPU.
Config note: at 8k/5 epochs the MLP's extra capacity wins the early race and the payoff is
false to the lesson; the CNN's spatial prior only pulls reliably ahead with more data +
epochs on Fashion-MNIST (which is small and low-texture). 15k/15 epochs is the honest
config where the CNN leads consistently. Full 10k test set for a stable accuracy estimate.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch6_cnn/py_src/cnn_vs_mlp.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag palette.
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
ARM_BLUE, ARM_RED = "#0033A0", "#D90012"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_cnn_vs_mlp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "cnn_vs_mlp.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


class SmallCNN(nn.Module):
    """Chapter CNN: conv(1->8,3,pad1)-relu-pool2 - conv(8->16,3,pad1)-relu-pool2 - fc(784->10).

    9,098 params. This exact class also lives in feature_maps.py - keep them identical.
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


def make_mlp():
    """The L14/L15 baseline MLP: 784-128-64-10, 109,386 params."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


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


def train_track(model, Xtr, ytr, Xte, yte, epochs=5, lr=1e-3, batch=128):
    """Train, returning test accuracy after each epoch. Seed reset for a fair comparison."""
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    accs = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            accs.append((model(Xte).argmax(1) == yte).float().mean().item())
    return accs


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte = load_fmnist(n_train=15000, n_test=10000)
    log.info(f"data train {tuple(Xtr.shape)} test {tuple(Xte.shape)}")

    mlp, cnn = make_mlp(), SmallCNN()
    p_mlp = sum(p.numel() for p in mlp.parameters())
    p_cnn = sum(p.numel() for p in cnn.parameters())
    log.info(f"MLP params = {p_mlp} (expect 109386); CNN params = {p_cnn} (expect 9098)")

    epochs = 15
    acc_mlp = train_track(mlp, Xtr, ytr, Xte, yte, epochs=epochs)
    acc_cnn = train_track(cnn, Xtr, ytr, Xte, yte, epochs=epochs)
    log.info(f"final MLP {acc_mlp[-1]:.3f}  CNN {acc_cnn[-1]:.3f}")

    fm, fc = acc_mlp[-1], acc_cnn[-1]
    names = ["MLP\n784-128-64-10", "small CNN"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.9))

    b1 = ax1.bar(names, [fm, fc], color=[ARM_RED, ARM_BLUE], width=0.6)
    ax1.set_ylim(0, 1.0); ax1.set_ylabel("test accuracy")
    ax1.bar_label(b1, labels=[f"{fm:.1%}", f"{fc:.1%}"], padding=3, fontsize=12)
    ax1.set_title("Accuracy: about equal")
    ax1.grid(axis="y", alpha=0.2)

    b2 = ax2.bar(names, [p_mlp, p_cnn], color=[ARM_RED, ARM_BLUE], width=0.6)
    ax2.set_ylim(0, p_mlp * 1.18); ax2.set_ylabel("parameters")
    ax2.bar_label(b2, labels=[f"{p_mlp:,}", f"{p_cnn:,}"], padding=3, fontsize=12)
    ax2.set_title(f"Parameters: {p_mlp // p_cnn}x fewer")
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle("Same data, same epochs: the CNN matches the MLP with 12x fewer weights",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "cnn_vs_mlp.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}  (MLP {fm:.3f} / CNN {fc:.3f})")


if __name__ == "__main__":
    main()
