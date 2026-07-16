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


if __name__ == "__main__":
    manifold_figure()
    tug_of_war_figure()
    log.info("DONE")
