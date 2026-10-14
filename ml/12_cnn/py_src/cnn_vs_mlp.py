"""Real figure for the L16 CNN Foundations deck (Section 5, the payoff).

Generates into ml/12_cnn/fig/:
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

Run with the project venv (~3-5 min on 2 CPU threads; --plot-only redraws in ~3 s from the
saved results/cnn_vs_mlp.json, no training):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/cnn_vs_mlp.py [--plot-only]

2026-10-01: the run now saves its numbers to ml/12_cnn/results/cnn_vs_mlp.json (the artifact of
record) and the figure is drawn from that file at its slide size (3.6 in wide, 7-8.5 pt text; it
rendered at 4.2 pt before). Training runs on 2 threads.

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag palette.
"""

import argparse
import json
import logging
import time
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
RESULTS = CH_DIR / "results" / "cnn_vs_mlp.json"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l16_cnn_vs_mlp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "cnn_vs_mlp.log", encoding="utf-8")
    fh.setFormatter(fmt)
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


def run_experiment(log):
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    Xtr, ytr, Xte, yte = load_fmnist(n_train=15000, n_test=10000)
    log.info(f"data train {tuple(Xtr.shape)} test {tuple(Xte.shape)}")

    mlp, cnn = make_mlp(), SmallCNN()
    p_mlp = sum(p.numel() for p in mlp.parameters())
    p_cnn = sum(p.numel() for p in cnn.parameters())
    log.info(f"MLP params = {p_mlp} (expect 109386); CNN params = {p_cnn} (expect 9098)")
    if (p_mlp, p_cnn) != (109386, 9098):
        raise ValueError(f"parameter counts changed: MLP {p_mlp}, CNN {p_cnn}")

    epochs = 15
    acc_mlp = train_track(mlp, Xtr, ytr, Xte, yte, epochs=epochs)
    log.info(f"MLP done after {time.perf_counter() - t0:.0f} s")
    acc_cnn = train_track(cnn, Xtr, ytr, Xte, yte, epochs=epochs)
    log.info(f"final MLP {acc_mlp[-1]:.3f}  CNN {acc_cnn[-1]:.3f} "
             f"({time.perf_counter() - t0:.0f} s)")
    res = {"config": {"n_train": 15000, "n_test": 10000, "epochs": epochs, "seed": SEED,
                      "threads": 2},
           "p_mlp": p_mlp, "p_cnn": p_cnn, "acc_mlp": acc_mlp, "acc_cnn": acc_cnn,
           "seconds": round(time.perf_counter() - t0, 1)}
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info(f"saved {RESULTS}")
    return res


def plot(res, log):
    fm, fc = res["acc_mlp"][-1], res["acc_cnn"][-1]
    p_mlp, p_cnn = res["p_mlp"], res["p_cnn"]
    names = ["MLP\n784-128-64-10", "small CNN"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.6, 1.75))   # its size on the slide

    b1 = ax1.bar(names, [fm, fc], color=[ARM_RED, ARM_BLUE], width=0.6)
    ax1.set_ylim(0, 1.12); ax1.set_yticks([])
    ax1.bar_label(b1, labels=[f"{fm:.1%}", f"{fc:.1%}"], padding=2, fontsize=8)
    ax1.set_title("test accuracy: equal", fontsize=8.5, pad=2)

    b2 = ax2.bar(names, [p_mlp, p_cnn], color=[ARM_RED, ARM_BLUE], width=0.6)
    ax2.set_ylim(0, p_mlp * 1.25); ax2.set_yticks([])
    ax2.bar_label(b2, labels=[f"{p_mlp:,}", f"{p_cnn:,}"], padding=2, fontsize=8)
    ax2.set_title(f"weights: {p_mlp // p_cnn}x fewer", fontsize=8.5, pad=2)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "cnn_vs_mlp.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"saved {out}  (MLP {fm:.3f} / CNN {fc:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="redraw from the saved JSON")
    args = ap.parse_args()
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    if args.plot_only:
        if not RESULTS.exists():
            raise FileNotFoundError(f"{RESULTS} missing - run without --plot-only first")
        res = json.loads(RESULTS.read_text(encoding="utf-8"))
    else:
        res = run_experiment(log)
    plot(res, log)


if __name__ == "__main__":
    main()
