"""Figures for Deck 2 (L15 Training neural networks). Fashion-MNIST, PyTorch, CPU.

Generates:
  dropout_test_error.pdf      test error vs epoch for a few dropout rates
  lr_curves.pdf               train loss vs epoch for too-high / good / too-low lr
  training_curves_4cases.pdf  good / overfit / underfit / val<train (heavy dropout)
  init_variance.pdf           per-layer activation std, bad init vs He (forward-pass only)
  batchnorm_curves.pdf        train loss with vs without BatchNorm

Run: ./ma/Scripts/python.exe ml/ch5_neural_networks/py_src/nn_training_figs.py

Light by design: subsampled train set, small nets, ~25-40 epochs, torch defaults to 4 threads.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SEED = 509
ARM_RED = "#D90012"
ARM_BLUE = "#0033A0"
ARM_ORANGE = "#F2A800"

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/nn_training_figs.log")],
)
log = logging.getLogger(__name__)
FIG = Path(__file__).resolve().parents[1] / "fig"
FIG.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data(n_train=7000, n_test=2000):
    from tensorflow.keras.datasets import fashion_mnist

    (xtr, ytr), (xte, yte) = fashion_mnist.load_data()
    rng = np.random.default_rng(SEED)
    itr = rng.choice(len(xtr), n_train, replace=False)
    ite = rng.choice(len(xte), n_test, replace=False)
    Xtr = torch.tensor(xtr[itr].reshape(n_train, -1) / 255.0, dtype=torch.float32)
    Ytr = torch.tensor(ytr[itr].astype("int64"))
    Xte = torch.tensor(xte[ite].reshape(n_test, -1) / 255.0, dtype=torch.float32)
    Yte = torch.tensor(yte[ite].astype("int64"))
    log.info(f"data: train {tuple(Xtr.shape)}, test {tuple(Xte.shape)}")
    return Xtr, Ytr, Xte, Yte


def mlp(sizes, p_drop=0.0, batchnorm=False):
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if batchnorm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU())
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


def train(model, data, epochs=25, lr=1e-3, opt="adam", batch=128):
    Xtr, Ytr, Xte, Yte = data
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=lr)
        if opt == "adam"
        else torch.optim.SGD(model.parameters(), lr=lr)
    )
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    hist = {"train_loss": [], "val_loss": [], "test_err": []}
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            optimizer.zero_grad()
            loss = lossf(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            optimizer.step()
            tot += loss.item() * len(idx)
        hist["train_loss"].append(tot / n)
        model.eval()
        with torch.no_grad():
            oute = model(Xte)
            hist["val_loss"].append(lossf(oute, Yte).item())
            hist["test_err"].append((oute.argmax(1) != Yte).float().mean().item())
    return hist


def fig_dropout(data):
    # Strong overfit regime (1.5k train, big net) + smoothing so the trend is legible.
    Xtr, Ytr, Xte, Yte = data
    small = (Xtr[:1500], Ytr[:1500], Xte, Yte)
    xs = np.arange(1, 41)

    def smooth(a):
        return np.convolve(np.array(a), np.ones(5) / 5, mode="valid")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for p, c in [(0.0, ARM_RED), (0.5, ARM_BLUE)]:
        torch.manual_seed(SEED)
        h = train(mlp([784, 1024, 512, 10], p_drop=p), small, epochs=40, lr=1e-3)
        ax.plot(xs[2:-2], smooth(h["test_err"]), color=c, lw=2.4, label=f"dropout $p={p}$")
        log.info(f"dropout p={p}: min {min(h['test_err']):.3f}, final {h['test_err'][-1]:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test error (smoothed)")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_title("Dropout curbs overfitting (Fashion-MNIST, 1.5k train)")
    fig.tight_layout()
    fig.savefig(FIG / "dropout_test_error.pdf")
    plt.close(fig)


def fig_lr(data):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for lr, c, lab in [
        (1.0, ARM_RED, "lr = 1.0"),
        (0.1, ARM_ORANGE, "lr = 0.1"),
        (0.002, ARM_BLUE, "lr = 0.002"),
    ]:
        torch.manual_seed(SEED)
        h = train(mlp([784, 256, 128, 10]), data, epochs=25, lr=lr, opt="sgd")
        tl = np.clip(np.nan_to_num(np.array(h["train_loss"]), nan=3.0, posinf=3.0), 0, 3.0)
        ax.plot(range(1, 26), tl, color=c, lw=2, label=lab)
        log.info(f"lr={lr}: final train loss {tl[-1]:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (clipped at 3)")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_title("Learning rate: too high diverges, too low crawls")
    fig.tight_layout()
    fig.savefig(FIG / "lr_curves.pdf")
    plt.close(fig)


def fig_curves(data):
    configs = [
        ("Good fit", mlp([784, 128, 64, 10], p_drop=0.2), dict(epochs=30, lr=1e-3)),
        ("Overfitting", mlp([784, 1024, 512, 10]), dict(epochs=40, lr=1e-3)),
        ("Underfitting", mlp([784, 8, 10]), dict(epochs=30, lr=1e-3)),
        ("val < train (heavy dropout)", mlp([784, 1024, 512, 10], p_drop=0.8), dict(epochs=25, lr=1e-3)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    for ax, (title, m, kw) in zip(axes.ravel(), configs):
        torch.manual_seed(SEED)
        h = train(m, data, **kw)
        e = range(1, kw["epochs"] + 1)
        ax.plot(e, h["train_loss"], color=ARM_BLUE, lw=2, label="train")
        ax.plot(e, h["val_loss"], color=ARM_RED, lw=2, label="val")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("loss", fontsize=8)
        log.info(f"curves[{title}]: train {h['train_loss'][-1]:.3f}, val {h['val_loss'][-1]:.3f}")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves_4cases.pdf")
    plt.close(fig)


def fig_init():
    torch.manual_seed(SEED)
    depth, width = 12, 256
    x0 = torch.randn(512, width)

    def run(kind):
        a = x0.clone()
        stds = [a.std().item()]
        for _ in range(depth):
            W = torch.empty(width, width)
            if kind == "bad":
                nn.init.normal_(W, 0.0, 1.0)
            else:
                nn.init.kaiming_normal_(W, nonlinearity="relu")
            a = torch.relu(a @ W.T)
            stds.append(a.std().item())
        return stds

    bad, he = run("bad"), run("he")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(range(depth + 1), bad, "o-", color=ARM_RED, lw=2, label=r"bad init  $\mathcal{N}(0,1)$")
    ax.plot(range(depth + 1), he, "o-", color=ARM_BLUE, lw=2, label="He init")
    ax.set_yscale("log")
    ax.set_xlabel("layer depth")
    ax.set_ylabel("activation std (log scale)")
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    ax.set_title("Good init keeps the activation scale stable with depth")
    fig.tight_layout()
    fig.savefig(FIG / "init_variance.pdf")
    plt.close(fig)
    log.info(f"init: bad final std {bad[-1]:.2e}, He final std {he[-1]:.2e}")


def fig_batchnorm(data):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for bn, c, lab in [(False, ARM_RED, "no BatchNorm"), (True, ARM_BLUE, "BatchNorm")]:
        torch.manual_seed(SEED)
        h = train(mlp([784, 256, 256, 256, 10], batchnorm=bn), data, epochs=25, lr=0.1, opt="sgd")
        ax.plot(range(1, 26), h["train_loss"], color=c, lw=2, label=lab)
        log.info(f"bn={bn}: final train loss {h['train_loss'][-1]:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_title("BatchNorm speeds up and stabilizes training")
    fig.tight_layout()
    fig.savefig(FIG / "batchnorm_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    data = load_data()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only == "dropout":
        fig_dropout(data)
    elif only == "lr":
        fig_lr(data)
    else:
        fig_init()
        fig_lr(data)
        fig_dropout(data)
        fig_batchnorm(data)
        fig_curves(data)
    log.info("all Deck 2 figures written to %s", FIG)
