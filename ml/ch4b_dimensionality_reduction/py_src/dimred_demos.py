#!/usr/bin/env python
"""Figures for the dimensionality-reduction deck (PCA / t-SNE / UMAP).

Outputs PDFs to the sibling ``fig/``. Run with the ma venv:
    ./ma/Scripts/python.exe ml/ch4b_dimensionality_reduction/py_src/dimred_demos.py
(Requires umap-learn in the ma venv.)
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from sklearn.datasets import load_digits, load_iris, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import umap

SEED = 509
FIG = Path(__file__).resolve().parent.parent / "fig"
ROOT = Path(__file__).resolve().parents[3]
TAB = plt.cm.tab10.colors
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"


def setup_logging():
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(logs / "dimred_demos.log", encoding="utf-8")],
    )


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    logging.info("wrote %s", name)


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.7")


def _digits():
    d = load_digits()
    return d.data, d.target


# ----------------------------------------------------------------------
def fig_digit_samples():
    d = load_digits()
    fig, axes = plt.subplots(2, 5, figsize=(6.4, 2.8))
    for ax, i in zip(axes.ravel(), range(10)):
        idx = np.where(d.target == i)[0][0]
        ax.imshow(d.images[idx], cmap="gray_r")
        ax.set_title(str(i), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "dr_digit_samples.pdf")


def fig_pca_anim(n_sweep=4):
    """Teaching sweep: rotate a direction, projecting the points onto it (red); variance peaks at the PC."""
    rng = np.random.RandomState(SEED)
    th0 = np.deg2rad(30)
    R = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])
    X = (rng.randn(200, 2) * np.array([2.3, 0.6])) @ R.T
    ctr = X.mean(0); Xc = X - ctr
    vals, vecs = np.linalg.eigh(np.cov(Xc.T))
    pc = vecs[:, vals.argmax()]
    pc_ang = np.arctan2(pc[1], pc[0])
    angs = list(np.linspace(pc_ang - np.deg2rad(75), pc_ang, n_sweep))
    L = 5.0

    def frame(u, title, fname):
        proj = Xc @ u
        feet = ctr + np.outer(proj, u)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.scatter(X[:, 0], X[:, 1], s=10, c="0.78", edgecolors="white", linewidths=0.2)
        ax.plot([ctr[0] - L * u[0], ctr[0] + L * u[0]],
                [ctr[1] - L * u[1], ctr[1] + L * u[1]], color=ARM_BLUE, lw=2)
        for p, f in zip(X, feet):                      # show each point's projection (the spread = variance)
            ax.plot([p[0], f[0]], [p[1], f[1]], color=ARM_RED, lw=0.4, alpha=0.3)
        ax.scatter(feet[:, 0], feet[:, 1], s=10, color=ARM_RED, zorder=5)
        ax.set_title(title, fontsize=10)
        _clean(ax); ax.set_aspect("equal")
        save(fig, fname)

    for i, a in enumerate(angs):
        u = np.array([np.cos(a), np.sin(a)])
        frame(u, f"candidate axis --- projected variance = {(Xc @ u).var():.2f}",
              f"dr_pca_anim_{i + 1}.pdf")
    frame(pc, f"max variance ({(Xc @ pc).var():.2f}) --- keep this axis",
          f"dr_pca_anim_{n_sweep + 1}.pdf")


def fig_curse_distances():
    rng = np.random.RandomState(SEED)
    dims = [2, 5, 10, 20, 50, 100, 200, 500]
    contrast = []
    for d in dims:
        dd = pdist(rng.rand(300, d))
        contrast.append((dd.max() - dd.min()) / dd.mean())  # relative contrast
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(dims, contrast, "o-", color=ARM_BLUE, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("dimension (log)"); ax.set_ylabel("(max $-$ min) / mean distance")
    ax.set_title("distances concentrate as dimension grows", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "dr_curse_distances.pdf")


def fig_byhand_pca():
    rng = np.random.RandomState(SEED)
    th = np.deg2rad(30)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    X = (rng.randn(12, 2) * np.array([1.7, 0.85])) @ R.T + np.array([3.0, 3.0])
    mu = X.mean(0); cov = np.cov((X - mu).T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    vecs = vecs * np.sign(vecs[0])  # orient both eigenvectors to point right (+x)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(X[:, 0], X[:, 1], s=55, color=TAB[0], edgecolors="white", zorder=3)
    ax.scatter(*mu, color="black", marker="X", s=90, zorder=5)
    ax.margins(0.3)
    for j, col in enumerate((ARM_RED, ARM_ORANGE)):
        v = vecs[:, j] * np.sqrt(vals[j]) * 1.4
        ax.annotate("", xy=mu + v, xytext=mu, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=3, mutation_scale=22))
        ax.text(*(mu + v * 1.22), f"PC{j+1}", color=col, fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=6)
    txt = (f"cov = [[{cov[0,0]:.2f}, {cov[0,1]:.2f}],\n"
           f"            [{cov[1,0]:.2f}, {cov[1,1]:.2f}]]\n"
           f"$\\lambda$ = {vals[0]:.2f},  {vals[1]:.2f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    _clean(ax); ax.set_aspect("equal")
    ax.set_title("covariance $\\to$ eigenvectors $=$ principal axes", fontsize=10)
    save(fig, "dr_byhand_pca.pdf")


def fig_scree():
    X = load_digits().data
    p = PCA().fit(X)
    ev = p.explained_variance_ratio_; cum = np.cumsum(ev)
    k95 = int(np.argmax(cum >= 0.95) + 1)
    n = 30
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(range(1, n + 1), ev[:n], color=ARM_BLUE, alpha=0.7)
    ax.set_xlabel("component"); ax.set_ylabel("explained variance ratio", color=ARM_BLUE)
    ax2 = ax.twinx()
    ax2.plot(range(1, n + 1), cum[:n], color=ARM_RED, lw=2, marker=".")
    ax2.axhline(0.95, ls="--", color="0.5", lw=1)
    ax2.scatter([k95], [cum[k95 - 1]], s=130, facecolors="none", edgecolors=ARM_RED, lw=2, zorder=5)
    ax2.annotate(f"95% at k={k95}", (k95, 0.95), textcoords="offset points",
                 xytext=(10, -16), fontsize=9, color=ARM_RED)
    ax2.set_ylabel("cumulative", color=ARM_RED); ax2.set_ylim(0, 1.03)
    ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    ax.set_title("scree + cumulative explained variance (digits)", fontsize=10)
    save(fig, "dr_scree.pdf")


def fig_biplot():
    d = load_iris()
    X = StandardScaler().fit_transform(d.data)
    p = PCA(n_components=2).fit(X); Z = p.transform(X)
    load = p.components_.T * np.sqrt(p.explained_variance_)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    flag = [ARM_RED, ARM_BLUE, ARM_ORANGE]
    for c in range(3):
        m = d.target == c
        ax.scatter(Z[m, 0], Z[m, 1], s=18, color=flag[c], alpha=0.7, label=d.target_names[c])
    sc = 2.4
    for i, name in enumerate(d.feature_names):
        ax.annotate("", xy=(load[i, 0] * sc, load[i, 1] * sc), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4))
        ax.text(load[i, 0] * sc * 1.13, load[i, 1] * sc * 1.13,
                name.replace(" (cm)", ""), fontsize=8, color="0.3", ha="center")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("iris biplot: samples + feature loadings", fontsize=10)
    save(fig, "dr_biplot.pdf")


def fig_reconstruction():
    X = load_digits().data
    rng = np.random.RandomState(SEED)
    idx = 15
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.6))
    axes[0, 0].imshow(X[idx].reshape(8, 8), cmap="gray_r"); axes[0, 0].set_title("original", fontsize=9)
    for col, k in zip([1, 2, 3], [5, 20, 50]):
        p = PCA(n_components=k).fit(X)
        rec = p.inverse_transform(p.transform(X[idx:idx + 1]))[0]
        axes[0, col].imshow(rec.reshape(8, 8), cmap="gray_r"); axes[0, col].set_title(f"k={k}", fontsize=9)
    p20 = PCA(n_components=20).fit(X)
    clean = X[idx]; noisy = np.clip(clean + rng.normal(0, 4, clean.shape), 0, 16)
    den = p20.inverse_transform(p20.transform(noisy.reshape(1, -1)))[0]
    axes[1, 0].imshow(clean.reshape(8, 8), cmap="gray_r"); axes[1, 0].set_title("clean", fontsize=9)
    axes[1, 1].imshow(noisy.reshape(8, 8), cmap="gray_r"); axes[1, 1].set_title("+ noise", fontsize=9)
    axes[1, 2].imshow(den.reshape(8, 8), cmap="gray_r"); axes[1, 2].set_title("denoised (k=20)", fontsize=9)
    axes[1, 3].axis("off")
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("top: compression (fewer PCs) ---  bottom: denoising", fontsize=10, y=1.0)
    save(fig, "dr_reconstruction.pdf")


def fig_swiss_roll():
    X, t = make_swiss_roll(n_samples=1200, noise=0.1, random_state=SEED)
    pca = PCA(n_components=2).fit_transform(X)
    um = umap.UMAP(n_neighbors=12, min_dist=0.3, random_state=SEED).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].scatter(pca[:, 0], pca[:, 1], c=t, cmap="Spectral", s=8)
    axes[0].set_title("PCA (linear): roll stays folded", fontsize=10)
    axes[1].scatter(um[:, 0], um[:, 1], c=t, cmap="Spectral", s=8)
    axes[1].set_title("UMAP: unrolls it", fontsize=10)
    for a in axes:
        _clean(a)
    save(fig, "dr_swiss_roll.pdf")


def _embed_plot(Z, y, title, name):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=6, alpha=0.8)
    cb = fig.colorbar(sc, ax=ax, ticks=range(10), fraction=0.046, pad=0.02)
    cb.set_label("digit")
    _clean(ax); ax.set_title(title, fontsize=11)
    save(fig, name)


def fig_tsne_digits():
    X, y = _digits()
    Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=SEED).fit_transform(X)
    _embed_plot(Z, y, "t-SNE of digits", "dr_tsne_digits.pdf")


def fig_umap_digits():
    X, y = _digits()
    Z = umap.UMAP(n_components=2, random_state=SEED).fit_transform(X)
    _embed_plot(Z, y, "UMAP of digits", "dr_umap_digits.pdf")


def fig_tsne_perplexity():
    X, y = _digits()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, perp in zip(axes, [5, 30, 100]):
        Z = TSNE(n_components=2, init="pca", perplexity=perp, random_state=SEED).fit_transform(X)
        ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5, alpha=0.75)
        _clean(ax); ax.set_title(f"perplexity = {perp}", fontsize=10)
    fig.suptitle("same digits, three perplexities --- the picture changes", fontsize=11, y=1.03)
    save(fig, "dr_tsne_perplexity.pdf")


def fig_compare_digits():
    X, y = _digits()
    Xs, ys = shuffle(X, y, random_state=SEED)
    Xs, ys = Xs[:700], ys[:700]
    embeds = [
        ("PCA", PCA(n_components=2).fit_transform(Xs)),
        ("t-SNE", TSNE(n_components=2, init="pca", perplexity=30, random_state=SEED).fit_transform(Xs)),
        ("UMAP", umap.UMAP(n_components=2, random_state=SEED).fit_transform(Xs)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (ttl, Z) in zip(axes, embeds):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=ys, cmap="tab10", s=8, alpha=0.75)
        _clean(ax); ax.set_title(ttl, fontsize=11)
    cb = fig.colorbar(sc, ax=axes, ticks=range(10), fraction=0.022, pad=0.01)
    cb.set_label("digit")
    save(fig, "dr_compare_digits.pdf")


def main():
    setup_logging()
    FIG.mkdir(exist_ok=True)
    logging.info("generating dim-reduction figures -> %s", FIG)
    fig_digit_samples()
    fig_pca_anim()
    fig_curse_distances()
    fig_byhand_pca()
    fig_scree()
    fig_biplot()
    fig_reconstruction()
    fig_swiss_roll()
    fig_tsne_digits()
    fig_umap_digits()
    fig_tsne_perplexity()
    fig_compare_digits()
    logging.info("done.")


if __name__ == "__main__":
    main()
