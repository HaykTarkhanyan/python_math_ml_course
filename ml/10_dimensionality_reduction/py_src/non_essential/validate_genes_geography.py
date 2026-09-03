"""Validate the "genes mirror geography" practical BEFORE it is written or taught.

Same rule as validate_lfw_project.py: the practical asks the room to watch a
specific story appear (continents separate on PC1/PC2 of raw genotypes; the
European populations line up on a north-south gradient), so the story is run and
measured here first. If a number below does not hold, the practical does not get
built around it.

Reads data/genomes_1000g_chr22.npz (fetch_1000g_genotypes.py), writes:
  out/gg_world_pca.png   -- 2,504 people on PC1/PC2, colored by super-population
  out/gg_europe_pca.png  -- EUR-only refit, Novembre-style (text codes + medians)
and logs the measurements (EVR, silhouette, k-means ARI, per-pop medians).

Run:  ./ma/Scripts/python.exe ml/10_dimensionality_reduction/py_src/non_essential/validate_genes_geography.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

SEED = 509
CHAPTER = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[4]
NPZ = CHAPTER / "data" / "genomes_1000g_chr22.npz"
OUT = CHAPTER / "out"

SUPER_COLORS = {"AFR": "tab:red", "EUR": "tab:blue", "EAS": "tab:green",
                "SAS": "tab:purple", "AMR": "tab:orange"}


def pca_scores(X: np.ndarray, n: int = 10, patterson: bool = True):
    """Center; optionally scale each SNP by sqrt(p(1-p)) (Patterson et al. 2006).

    A SNP that is monomorphic WITHIN the analyzed subset (p = 0 or 1) has zero
    variance there and an undefined Patterson weight; dropping such columns is the
    standard step, done explicitly and logged (323 of 16,202 for the EUR subset)."""
    X = X.astype(np.float32)
    p = X.mean(axis=0) / 2.0
    if patterson:
        keep = (p > 0) & (p < 1)
        if not keep.all():
            logging.info("  dropping %d SNPs monomorphic in this subset", int((~keep).sum()))
            X, p = X[:, keep], p[keep]
    Xc = X - 2.0 * p
    if patterson:
        Xc /= np.sqrt(p * (1.0 - p))
    pca = PCA(n_components=n, random_state=SEED)
    return pca.fit_transform(Xc), pca.explained_variance_ratio_


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    X, pop, sup = d["genotypes"], d["pop"], d["super_pop"]
    logging.info("matrix %s, %d populations, %d super-populations",
                 X.shape, len(set(pop)), len(set(sup)))

    for patterson in (False, True):
        Z, evr = pca_scores(X, patterson=patterson)
        sil = silhouette_score(Z[:, :2], sup)
        ari = adjusted_rand_score(sup, KMeans(5, n_init=10, random_state=SEED).fit_predict(Z[:, :4]))
        logging.info("patterson=%-5s EVR1-4 = %s | silhouette(super_pop, PC1-2) = %.3f | "
                     "kmeans5-vs-superpop ARI (PC1-4) = %.3f",
                     patterson, np.round(evr[:4], 4).tolist(), sil, ari)

    Z, evr = pca_scores(X, patterson=True)

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for s in sorted(set(sup)):
        m = sup == s
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.65, label=f"{s} ({m.sum()})",
                   color=SUPER_COLORS[s], linewidths=0)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("1000 Genomes, chr22 only: PCA of raw genotype counts", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "gg_world_pca.png", dpi=150); plt.close(fig)
    logging.info("wrote %s", OUT / "gg_world_pca.png")

    # Europe only, refit -- the Novembre-style panel (text codes + median dots).
    eur = sup == "EUR"
    Ze, evr_e = pca_scores(X[eur], patterson=True)
    pe = pop[eur]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    cmap = dict(zip(sorted(set(pe)), plt.cm.tab10.colors))
    for i in range(len(Ze)):
        ax.text(Ze[i, 0], Ze[i, 1], pe[i], color=cmap[pe[i]], fontsize=5,
                ha="center", va="center", alpha=0.7)
    for p in sorted(set(pe)):
        m = pe == p
        med = np.median(Ze[m, :2], axis=0)
        ax.scatter(*med, s=180, color=cmap[p], edgecolors="black", zorder=5)
        ax.annotate(p, med, textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold", color=cmap[p])
        logging.info("EUR median PC1/PC2  %s: %s", p, np.round(med, 2).tolist())
    ax.set_xlim(Ze[:, 0].min() * 1.15, Ze[:, 0].max() * 1.15)
    ax.set_ylim(Ze[:, 1].min() * 1.15, Ze[:, 1].max() * 1.15)
    ax.set_xlabel(f"PC1 ({evr_e[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr_e[1]*100:.1f}%)")
    ax.set_title("Europeans only, PCA refit: FIN / CEU / GBR / IBS / TSI", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "gg_europe_pca.png", dpi=150); plt.close(fig)
    logging.info("wrote %s", OUT / "gg_europe_pca.png")


if __name__ == "__main__":
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / "validate_genes_geography.log", encoding="utf-8")],
    )
    main()
