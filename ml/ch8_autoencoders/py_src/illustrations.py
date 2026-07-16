"""Conceptual illustrations for ch8 (no training, no downloads).

- manifold_pca_vs_ae.pdf : S-curve -> PCA folds it (flat) vs Isomap unrolls it (nonlinear).
  Real, cheap (sklearn). Carries L22's flat-sheet / curved-sheet intuition.
- vae_tug_of_war.pdf : 3-panel schematic of the ELBO's two forces on the latent blobs.
  Carries L23's reconstruction-vs-KL tug-of-war intuition.
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

CH = Path(__file__).resolve().parent.parent
FIG = CH / "fig"
LOGS = CH / "logs"
FIG.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOGS / "illustrations.log", encoding="utf-8"),
                              logging.StreamHandler()])
log = logging.getLogger("illus")
rng = np.random.default_rng(509)


def manifold_figure():
    from sklearn.datasets import make_swiss_roll
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    X, t = make_swiss_roll(1500, noise=0.05, random_state=509)
    pca = PCA(2).fit_transform(X)
    iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    axs[0].scatter(X[:, 0], X[:, 2], c=t, cmap="viridis", s=6)
    axs[0].set_title("data: a rolled-up sheet")
    axs[1].scatter(pca[:, 0], pca[:, 1], c=t, cmap="viridis", s=6)
    axs[1].set_title("PCA (linear): flat projection\nthe roll overlaps itself")
    axs[2].scatter(iso[:, 0], iso[:, 1], c=t, cmap="viridis", s=6)
    axs[2].set_title("nonlinear: unrolled\n(Isomap here; an AE does the same)")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "manifold_pca_vs_ae.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("saved manifold_pca_vs_ae.pdf")


def tug_of_war_figure():
    cols = plt.cm.tab10(np.linspace(0, 1, 6))

    def blob(ax, xy, w, c):
        ax.add_patch(Ellipse(xy, w, w, alpha=0.55, color=c))

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
    # (1) reconstruction alone: blobs fly apart, gaps between them
    pos1 = [(-2.5, 2.0), (2.3, 1.8), (-2.0, -2.2), (2.6, -1.4), (0.2, 2.9), (-2.9, 0.1)]
    for p, c in zip(pos1, cols):
        blob(axs[0], p, 0.55, c)
    axs[0].set_title("reconstruction alone:\nblobs fly apart $\\rightarrow$ holes")
    # (2) KL alone: all collapse onto the prior, indistinguishable
    for c in cols:
        blob(axs[1], (rng.normal(0, 0.12), rng.normal(0, 0.12)), 0.55, c)
    axs[1].add_patch(Circle((0, 0), 1.7, fill=False, ls="--", color="k"))
    axs[1].set_title("KL alone:\nall pile on $\\mathcal{N}(0,I)$ $\\rightarrow$ no info")
    # (3) balance: blobs spread but tile the prior
    pos3 = [(-0.85, 0.75), (0.85, 0.65), (-0.7, -0.85), (0.95, -0.7), (0.05, 1.15), (0.0, -1.15)]
    for p, c in zip(pos3, cols):
        blob(axs[2], p, 0.75, c)
    axs[2].add_patch(Circle((0, 0), 1.8, fill=False, ls="--", color="k"))
    axs[2].set_title("balance (the VAE):\nblobs tile $\\mathcal{N}(0,I)$")
    for ax in axs:
        ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "vae_tug_of_war.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("saved vae_tug_of_war.pdf")


def mnist_examples():
    from torchvision import datasets, transforms
    ds = datasets.MNIST(str(CH / "practical" / "data"), train=True, download=False,
                        transform=transforms.ToTensor())
    X = ds.data.numpy(); y = ds.targets.numpy()
    nrow, ncol = 4, 10
    fig, axs = plt.subplots(nrow, ncol, figsize=(10, 4.2))
    for c in range(10):
        idxs = np.where(y == c)[0][:nrow]
        for r in range(nrow):
            ax = axs[r, c]
            ax.imshow(X[idxs[r]], cmap="gray"); ax.axis("off")
            if r == 0:
                ax.set_title(str(c), fontsize=14)
    fig.suptitle("MNIST: 70,000 handwritten digits, 28$\\times$28 grayscale (one column per class)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "mnist_examples.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("saved mnist_examples.pdf")


def sparse_activations():
    r2 = np.random.default_rng(1)
    n = 16
    dense = np.abs(r2.normal(0.6, 0.3, n)).clip(0.05, 1.2)
    sparse = np.zeros(n)
    active = r2.choice(n, 3, replace=False)
    sparse[active] = r2.uniform(0.7, 1.15, 3)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.1), sharey=True)
    axs[0].bar(range(n), dense, color="#0033A0")
    axs[0].set_title("dense code: most neurons fire")
    axs[1].bar(range(n), sparse, color="#D90012")
    axs[1].set_title("sparse code: a few fire, the rest $\\approx 0$")
    for ax in axs:
        ax.set_xlabel("code neuron"); ax.set_xticks(range(0, n, 3))
    axs[0].set_ylabel("activation")
    fig.tight_layout()
    fig.savefig(FIG / "sparse_activations.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("saved sparse_activations.pdf")


def two_failures():
    r = np.random.default_rng(3)
    centers = [(-1.7, 1.3), (1.5, 1.5), (1.7, -1.4), (-1.5, -1.3), (-0.1, 0.0)]
    cols = plt.cm.tab10(np.linspace(0, 1, len(centers)))
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for c, col in zip(centers, cols):
        pts = r.normal(c, 0.33, size=(45, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=col, alpha=0.55)
    # (1) random draw lands in an empty gap
    ax.scatter([0.6], [1.95], marker="*", s=300, color="#D90012", edgecolor="k", zorder=5)
    ax.annotate("(1) random draw\nlands in a gap", xy=(0.6, 1.95), xytext=(-3.0, 2.1),
                fontsize=9.5, color="#D90012", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#D90012"))
    # (2) nudge a real code -> drifts into a gap
    ax.plot([-0.1, 0.95], [0.0, 0.85], "k--", lw=1.2)
    ax.scatter([0.95], [0.85], marker="X", s=110, color="k", zorder=5)
    ax.annotate("(2) nudge a real code\n-> off into nonsense", xy=(0.95, 0.85), xytext=(1.7, 0.35),
                fontsize=9.5, color="k",
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_title("Plain AE latent: gaps everywhere, so you cannot sample it")
    ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-3.3, 3.3); ax.set_ylim(-2.6, 2.9)
    fig.tight_layout()
    fig.savefig(FIG / "ae_two_failures.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("saved ae_two_failures.pdf")


if __name__ == "__main__":
    manifold_figure()
    tug_of_war_figure()
    mnist_examples()
    sparse_activations()
    two_failures()
    log.info("DONE")
