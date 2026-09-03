"""t-SNE and UMAP on the 1000 Genomes genotypes -- the deck's recipe on real DNA.

Pipeline is exactly what lecture 35 teaches: Patterson-scale, PCA to 25 components
(denoise + speed), then t-SNE and UMAP on the PC scores. Two figures:

  out/gg_pca_tsne_umap.png    -- PCA | t-SNE | UMAP side by side, super-pop colors
                                 (the genotype twin of dr_compare_fashion.pdf)
  out/gg_umap_populations.png -- UMAP colored by all 26 populations: how much finer
                                 than continents does the neighbor graph see?

Logged measurements: silhouette of the 5 super-populations and of the 26
populations in each 2-D embedding, so "UMAP separates more" is a number, not a vibe.

Run:  ./ma/Scripts/python.exe ml/10_dimensionality_reduction/py_src/genes_geography_nonlinear.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap

SEED = 509
CHAPTER = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
NPZ = CHAPTER / "data" / "genomes_1000g_chr22.npz"
OUT = CHAPTER / "out"
N_PCS = 25

SUPER_COLORS = {"AFR": "tab:red", "EUR": "tab:blue", "EAS": "tab:green",
                "SAS": "tab:purple", "AMR": "tab:orange"}
SUPER_NAMES = {"AFR": "African", "AMR": "American", "EAS": "East Asian",
               "EUR": "European", "SAS": "South Asian"}


def patterson_pcs(X: np.ndarray, n: int) -> np.ndarray:
    X = X.astype(np.float32)
    p = X.mean(axis=0) / 2.0
    keep = (p > 0) & (p < 1)
    if not keep.all():
        logging.info("dropping %d monomorphic SNPs", int((~keep).sum()))
        X, p = X[:, keep], p[keep]
    Xc = (X - 2.0 * p) / np.sqrt(p * (1.0 - p))
    return PCA(n_components=n, random_state=SEED).fit_transform(Xc)


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    pop, sup = d["pop"], d["super_pop"]
    Zp = patterson_pcs(d["genotypes"], N_PCS)
    logging.info("PC scores: %s", Zp.shape)

    logging.info("t-SNE (perplexity 30, pca init) ...")
    Zt = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=SEED).fit_transform(Zp)
    logging.info("UMAP (n_neighbors 15, min_dist 0.1) ...")
    Zu = umap.UMAP(n_neighbors=15, min_dist=0.1,
                   random_state=SEED).fit_transform(Zp)

    embeddings = [("PCA (PC1/PC2)", Zp[:, :2]), ("t-SNE", Zt), ("UMAP", Zu)]
    for name, Z in embeddings:
        logging.info("%-12s silhouette: 5 super-pops = %.3f | 26 populations = %.3f",
                     name, silhouette_score(Z, sup), silhouette_score(Z, pop))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (name, Z) in zip(axes, embeddings):
        for s in sorted(SUPER_NAMES):
            m = sup == s
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.65, color=SUPER_COLORS[s],
                       linewidths=0, label=f"{s} — {SUPER_NAMES[s]}")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.7")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Same 2,504 genomes, three maps — the deck's PCA-then-nonlinear recipe on DNA",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "gg_pca_tsne_umap.png", dpi=150); plt.close(fig)
    logging.info("wrote %s", OUT / "gg_pca_tsne_umap.png")

    # UMAP again, now colored by all 26 populations, legend grouped by super-pop.
    fig, ax = plt.subplots(figsize=(9.8, 7.2))
    pops_sorted = sorted(set(pop), key=lambda q: (sup[pop == q][0], q))
    cmap = plt.cm.tab20.colors + plt.cm.tab20b.colors
    for i, q in enumerate(pops_sorted):
        m = pop == q
        ax.scatter(Zu[m, 0], Zu[m, 1], s=8, alpha=0.75, color=cmap[i % len(cmap)],
                   linewidths=0, label=f"{q} ({sup[m][0]})")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("0.7")
    ax.set_title("UMAP of the genotype PCs, colored by all 26 populations", fontsize=12)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "gg_umap_populations.png", dpi=150); plt.close(fig)
    logging.info("wrote %s", OUT / "gg_umap_populations.png")


if __name__ == "__main__":
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / "genes_geography_nonlinear.log", encoding="utf-8")],
    )
    main()
