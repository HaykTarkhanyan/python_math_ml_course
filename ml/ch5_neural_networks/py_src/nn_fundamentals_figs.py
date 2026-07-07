"""Figures for Deck 1 (L14 Neural network fundamentals).

Generates:
  - activations.pdf       sigmoid / tanh / ReLU on one axis
  - two_moons_logreg.pdf  two-moons + logistic-regression (linear) boundary  -> fails
  - two_moons_mlp.pdf     same data + MLP (bent) boundary                     -> works

Run with the repo `ma` venv:
  ./ma/Scripts/python.exe ml/ch5_neural_networks/py_src/nn_fundamentals_figs.py
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

SEED = 509
ARM_RED = "#D90012"
ARM_BLUE = "#0033A0"
ARM_ORANGE = "#F2A800"

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/nn_fundamentals_figs.log"),
    ],
)
log = logging.getLogger(__name__)

FIG = Path(__file__).resolve().parents[1] / "fig"
FIG.mkdir(exist_ok=True)


def fig_activations():
    x = np.linspace(-5, 5, 400)
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.plot(x, 1 / (1 + np.exp(-x)), color=ARM_BLUE, lw=2.6, label=r"sigmoid $\sigma(x)$")
    ax.plot(x, np.tanh(x), color=ARM_RED, lw=2.6, label=r"$\tanh(x)$")
    ax.plot(x, np.maximum(0, x), color=ARM_ORANGE, lw=2.6, label=r"ReLU$(x)=\max(0,x)$")
    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.3, 3.0)
    ax.set_xlabel("x")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "activations.pdf")
    plt.close(fig)
    log.info("wrote activations.pdf")


def _plot_boundary(ax, clf, X, y, title):
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.22, cmap=ListedColormap([ARM_BLUE, ARM_RED]))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=ARM_BLUE, s=18, edgecolor="white", lw=0.4)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=ARM_RED, s=18, edgecolor="white", lw=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def two_moons_figs():
    X, y = make_moons(n_samples=300, noise=0.25, random_state=SEED)

    logreg = LogisticRegression().fit(X, y)
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        max_iter=3000,
        random_state=SEED,
    ).fit(X, y)

    for clf, name, fname in [
        (logreg, "Logistic regression: one straight cut", "two_moons_logreg.pdf"),
        (mlp, "MLP: a boundary that bends", "two_moons_mlp.pdf"),
    ]:
        acc = clf.score(X, y)
        fig, ax = plt.subplots(figsize=(5, 3.7))
        _plot_boundary(ax, clf, X, y, f"{name}\ntrain accuracy = {acc:.2f}")
        fig.tight_layout()
        fig.savefig(FIG / fname)
        plt.close(fig)
        log.info(f"wrote {fname} (train acc = {acc:.3f})")


def fig_activation_grads():
    """Derivatives: sigmoid/tanh saturate (grad -> 0), ReLU stays 1. Motivates ReLU."""
    x = np.linspace(-5, 5, 400)
    sig = 1 / (1 + np.exp(-x))
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.plot(x, sig * (1 - sig), color=ARM_BLUE, lw=2.5, label=r"sigmoid$'$ (max $0.25$)")
    ax.plot(x, 1 - np.tanh(x) ** 2, color=ARM_RED, lw=2.5, label=r"tanh$'$ (max $1$)")
    ax.plot(x, (x > 0).astype(float), color=ARM_ORANGE, lw=2.5, label=r"ReLU$'$ ($0$ or $1$)")
    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel("z")
    ax.set_ylabel("gradient")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="upper center", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "activation_grads.pdf")
    plt.close(fig)
    log.info("wrote activation_grads.pdf")


def fig_folding_tent():
    """Tent-map composition: each 'fold' doubles the number of linear pieces."""
    x = np.linspace(0, 1, 4001)

    def tent(v):
        return 1 - np.abs(2 * v - 1)

    outs = [tent(x)]
    for _ in range(2):
        outs.append(tent(outs[-1]))
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), sharey=True)
    for ax, y, k in zip(axes, outs, [1, 2, 3]):
        ax.plot(x, y, color=ARM_BLUE, lw=2.0)
        ax.set_title(f"{k} fold(s) $\\to$ {2**k} linear pieces", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "folding_tent.pdf")
    plt.close(fig)
    log.info("wrote folding_tent.pdf")


def fig_folding_space():
    """A deep ReLU net carves the input plane into many linear regions (folded space)."""
    rng = np.random.default_rng(SEED)
    n = 420
    xs = np.linspace(-3, 3, n)
    XX, YY = np.meshgrid(xs, xs)
    a = np.stack([XX.ravel(), YY.ravel()], axis=1)
    dims = [2, 8, 8, 8]
    patterns = []
    for din, dout in zip(dims[:-1], dims[1:]):
        W = rng.normal(0, 1.4, size=(din, dout))
        b = rng.normal(0, 1.0, size=dout)
        z = a @ W + b
        patterns.append((z > 0).astype(np.int8))
        a = np.maximum(0, z)
    allpat = np.concatenate(patterns, axis=1)
    _, inv = np.unique(allpat, axis=0, return_inverse=True)
    idmap = inv.reshape(n, n)
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.imshow(idmap % 20, cmap="tab20", origin="lower", extent=[-3, 3, -3, 3],
              interpolation="nearest")
    ax.set_title("Input plane folded into linear regions\n(depth-3 ReLU net)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "folding_space.pdf")
    plt.close(fig)
    log.info(f"wrote folding_space.pdf ({idmap.max() + 1} regions)")


if __name__ == "__main__":
    fig_activations()
    fig_activation_grads()
    two_moons_figs()
    fig_folding_tent()
    fig_folding_space()
    log.info("all Deck 1 figures written to %s", FIG)
