#!/usr/bin/env python
"""Figures for the dimensionality-reduction deck (PCA / t-SNE / UMAP).

Running dataset: **Fashion-MNIST**, 28x28 grayscale, 784 features, 10 garment classes,
loaded from the committed ``data/fashion_mnist.npz`` (built by ``fetch_fashion_mnist.py``).
It replaced sklearn's 8x8 digits, which were too coarse to read on a projector -- at 8x8 a
reconstruction at k=5 and one at k=50 look equally like grey mush, so the compression
lesson never landed as a picture.

One figure uses the CLIP embeddings committed for the clustering chapter
(``ml/09_clustering/data/imagenette_clip.npz``) to show DR on a real 512-d embedding space.

Outputs PDFs to the sibling ``fig/``. Run with the ma venv:
    ./ma/Scripts/python.exe ml/10_dimensionality_reduction/py_src/dimred_demos.py
(Requires umap-learn in the ma venv. Takes a few minutes -- t-SNE dominates.)
"""
import io
import logging
import os
from pathlib import Path

# Keep the machine usable: cap BLAS/OpenMP threads before numpy/sklearn import them.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from sklearn.datasets import load_iris, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

SEED = 509
CHAPTER = Path(__file__).resolve().parent.parent
FIG = CHAPTER / "fig"
ROOT = Path(__file__).resolve().parents[3]
FASHION_NPZ = CHAPTER / "data" / "fashion_mnist.npz"
CLIP_NPZ = ROOT / "ml" / "09_clustering" / "data" / "imagenette_clip.npz"

TAB = plt.cm.tab10.colors
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

# t-SNE is O(n^2)-ish and runs five times below; 5000 points keeps the whole script
# to a few minutes while the picture is visually identical to the full 12k.
EMBED_N = 5000
# Standard practice on 784-d pixels: PCA down to 50 first, then t-SNE/UMAP. Kills most
# of the pixel noise and is much faster, with no visible change to the layout.
PRE_PCA = 50


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


# ----------------------------------------------------------------------
# data


def _fashion():
    """(X flat float64 in [0,255], y, images, class_names)."""
    d = np.load(FASHION_NPZ, allow_pickle=True)
    imgs = d["images"]
    return imgs.reshape(len(imgs), -1).astype(np.float64), d["labels"], imgs, d["class_names"]


def _fashion_embed_input():
    """The PCA-50 representation the nonlinear methods actually consume."""
    X, y, _, names = _fashion()
    X, y = X[:EMBED_N], y[:EMBED_N]          # npz is pre-shuffled, so a head-slice is balanced
    Z = PCA(n_components=PRE_PCA, random_state=SEED).fit_transform(X)
    return Z, y, names


# ----------------------------------------------------------------------
def fig_fashion_samples():
    _, y, imgs, names = _fashion()
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.6))
    for ax, c in zip(axes.ravel(), range(10)):
        ax.imshow(imgs[np.flatnonzero(y == c)[0]], cmap="gray_r")
        ax.set_title(str(names[c]), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "dr_fashion_samples.pdf")


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
    # Stop the sweep short of the PC so the final frame is a genuine reveal, not a relabel
    # of the axis the previous frame already found (REVIEW.md section 3).
    angs = list(np.linspace(pc_ang - np.deg2rad(75), pc_ang - np.deg2rad(12), n_sweep))
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
    """Beyer et al. relative contrast: (Dmax - Dmin)/Dmin measured FROM A QUERY POINT.

    The old version used (max - min)/mean over all pairwise distances, which shows the same
    concentration but is not the quantity the slide's question asks about (REVIEW.md sec. 3).
    """
    rng = np.random.RandomState(SEED)
    dims = [2, 5, 10, 20, 50, 100, 200, 500]
    contrast = []
    for d in dims:
        data = rng.rand(500, d)
        queries = rng.rand(50, d)
        per_query = []
        for q in queries:
            dd = np.linalg.norm(data - q, axis=1)
            per_query.append((dd.max() - dd.min()) / dd.min())
        contrast.append(np.mean(per_query))
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(dims, contrast, "o-", color=ARM_BLUE, lw=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dimension (log)")
    ax.set_ylabel("relative contrast\n$(D_{max}-D_{min})/D_{min}$")
    ax.set_title("your farthest point stops being much farther\nthan your nearest", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    logging.info("relative contrast: %s", dict(zip(dims, np.round(contrast, 2))))
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
    X, _, _, _ = _fashion()
    p = PCA().fit(X)
    ev = p.explained_variance_ratio_; cum = np.cumsum(ev)
    k95 = int(np.argmax(cum >= 0.95) + 1)
    n = 40
    # Two panels, not a twinx: the scree bars live in the first 40 components while the 95%
    # mark sits at k=184, and forcing both onto one x-axis squashed the bars to invisibility.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax.bar(range(1, n + 1), ev[:n], color=ARM_BLUE, alpha=0.85)
    ax.set_xlabel("component"); ax.set_ylabel("explained variance ratio")
    ax.set_title(f"scree: the first {n} components", fontsize=10)
    ax.annotate(f"PC1 = {ev[0]*100:.0f}%", (1, ev[0]), textcoords="offset points",
                xytext=(12, -4), fontsize=9, color=ARM_BLUE)
    ax.spines[["top", "right"]].set_visible(False)

    ax2.plot(range(1, len(cum) + 1), cum, color=ARM_RED, lw=2)
    ax2.axhline(0.95, ls="--", color="0.5", lw=1)
    ax2.axvline(k95, ls="--", color="0.5", lw=1)
    ax2.scatter([k95], [cum[k95 - 1]], s=130, facecolors="none", edgecolors=ARM_RED, lw=2, zorder=5)
    ax2.annotate(f"95% at k={k95}\nof {len(ev)}", (k95, 0.95), textcoords="offset points",
                 xytext=(16, -34), fontsize=9, color=ARM_RED)
    ax2.set_xlabel("component"); ax2.set_ylabel("cumulative explained variance")
    ax2.set_ylim(0, 1.03); ax2.set_xlim(0, len(ev))
    ax2.set_title("cumulative: the tail is long and flat", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    logging.info("scree: PC1=%.3f  PC1+2=%.3f  95%% at k=%d of %d",
                 ev[0], cum[1], k95, len(ev))
    save(fig, "dr_scree.pdf")


def fig_scaling_trap():
    """Show-don't-tell for the standardize-or-not trap (REVIEW.md item 6).

    Two features on wildly different scales: income in dollars, age in years. Unscaled,
    PC1 is essentially the income axis; standardized, it is a real combination.
    """
    rng = np.random.RandomState(SEED)
    n = 300
    age = rng.normal(40, 12, n)
    income = 1200 * age + rng.normal(0, 12000, n) + 20000   # correlated with age
    X = np.column_stack([income, age])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, (Xi, ttl, xlab, ylab) in zip(axes, [
        (X, "raw units: PC1 $\\approx$ the income axis", "income ($)", "age (years)"),
        (StandardScaler().fit_transform(X), "standardized: PC1 is a real combination",
         "income (z)", "age (z)"),
    ]):
        p = PCA(n_components=2).fit(Xi)
        mu = Xi.mean(0)
        ax.scatter(Xi[:, 0], Xi[:, 1], s=12, color="0.6", edgecolors="white", linewidths=0.2)
        for j, col in enumerate((ARM_RED, ARM_ORANGE)):
            v = p.components_[j] * np.sqrt(p.explained_variance_[j]) * 2.2
            ax.annotate("", xy=mu + v, xytext=mu,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6, mutation_scale=18))
        w = p.components_[0]
        ang = np.rad2deg(np.arctan2(abs(w[1]), abs(w[0])))
        ax.set_title(f"{ttl}\n" + r"PC1 $=$ (%.2f, %.2f), %.0f$^\circ$ off the income axis"
                     % (w[0], w[1], ang), fontsize=9)
        ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        logging.info("scaling trap %-14s PC1 = %s", xlab, np.round(w, 3))
    fig.tight_layout()
    save(fig, "dr_scaling_trap.pdf")


def fig_biplot():
    d = load_iris()
    X = StandardScaler().fit_transform(d.data)
    p = PCA(n_components=2).fit(X); Z = p.transform(X)
    load = p.components_.T * np.sqrt(p.explained_variance_)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    flag = [ARM_RED, ARM_BLUE, ARM_ORANGE]
    for c in range(3):
        m = d.target == c
        ax.scatter(Z[m, 0], Z[m, 1], s=18, color=flag[c], alpha=0.7, label=d.target_names[c])
    sc = 2.4
    # Per-arrow label radius + vertical nudge: the three right-hand loadings used to print
    # on top of each other and were unreadable on a projector (REVIEW.md item 3).
    short = ["sepal L", "sepal W", "petal L", "petal W"]
    radius = [1.30, 1.13, 1.13, 1.34]
    dy = [0.30, 0.0, -0.30, 0.12]
    for i, name in enumerate(short):
        ax.annotate("", xy=(load[i, 0] * sc, load[i, 1] * sc), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4))
        ax.text(load[i, 0] * sc * radius[i], load[i, 1] * sc * radius[i] + dy[i],
                name, fontsize=9, color="0.15", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("iris biplot: samples + feature loadings", fontsize=10)
    save(fig, "dr_biplot.pdf")


def fig_reconstruction():
    X, y, imgs, names = _fashion()
    rng = np.random.RandomState(SEED)
    # An ankle boot: a strong, unambiguous silhouette. A printed T-shirt was tried first and
    # read as noise at every k, which hid the very thing the figure is meant to show.
    idx = int(np.flatnonzero(y == 9)[1])
    ks = [5, 20, 50]
    cum = np.cumsum(PCA().fit(X).explained_variance_ratio_)

    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.6))
    axes[0, 0].imshow(imgs[idx], cmap="gray_r"); axes[0, 0].set_title("original (784 numbers)", fontsize=9)
    for col, k in zip([1, 2, 3], ks):
        p = PCA(n_components=k).fit(X)
        rec = p.inverse_transform(p.transform(X[idx:idx + 1]))[0]
        axes[0, col].imshow(rec.reshape(28, 28), cmap="gray_r")
        axes[0, col].set_title(f"k={k}  ({cum[k-1]*100:.0f}% var)", fontsize=9)
    p20 = PCA(n_components=20).fit(X)
    clean = X[idx]
    noisy = np.clip(clean + rng.normal(0, 60, clean.shape), 0, 255)
    den = p20.inverse_transform(p20.transform(noisy.reshape(1, -1)))[0]
    axes[1, 0].imshow(clean.reshape(28, 28), cmap="gray_r"); axes[1, 0].set_title("clean", fontsize=9)
    axes[1, 1].imshow(noisy.reshape(28, 28), cmap="gray_r"); axes[1, 1].set_title("+ noise", fontsize=9)
    axes[1, 2].imshow(den.reshape(28, 28), cmap="gray_r"); axes[1, 2].set_title("denoised (k=20)", fontsize=9)
    axes[1, 3].axis("off")
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("top: compression (fewer PCs) ---  bottom: denoising", fontsize=10, y=1.0)
    logging.info("reconstruction ks=%s cumvar=%s", ks, np.round(cum[[k - 1 for k in ks]], 3))
    save(fig, "dr_reconstruction.pdf")


def fig_swiss_roll():
    """Three panels so 'unfold' is legible: the 3-D roll, PCA's flat shadow, UMAP's strip.

    The old two-panel version left students unable to see what PCA was failing to do,
    because its output still looks like an ordered rainbow (REVIEW.md item 7).
    """
    # n_neighbors=50 matters here: at the default 15 the roll comes out torn into
    # fragments with the colour order broken, which would contradict the caption.
    # Denser sampling (2500 pts, less noise) also helps the graph span the sheet.
    X, t = make_swiss_roll(n_samples=2500, noise=0.05, random_state=SEED)
    pca = PCA(n_components=2).fit_transform(X)
    um = umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=SEED).fit_transform(X)

    fig = plt.figure(figsize=(11.0, 3.6))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="Spectral", s=6)
    ax0.set_title("the data: a 2-D sheet\nrolled up in 3-D", fontsize=10)
    ax0.set_xticklabels([]); ax0.set_yticklabels([]); ax0.set_zticklabels([])
    ax0.view_init(elev=12, azim=-72)

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.scatter(pca[:, 0], pca[:, 1], c=t, cmap="Spectral", s=8)
    ax1.set_title("PCA (linear): a shadow ---\nthe layers stay overlapped", fontsize=10)
    _clean(ax1)

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.scatter(um[:, 0], um[:, 1], c=t, cmap="Spectral", s=8)
    ax2.set_title("UMAP: one sweep of colour,\nend to end, nothing overlapping", fontsize=10)
    _clean(ax2)
    fig.tight_layout()
    save(fig, "dr_swiss_roll.pdf")


def _embed_plot(Z, y, names, title, name):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for c in range(10):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.75, color=TAB[c], label=str(names[c]))
    ax.legend(fontsize=6.5, markerscale=2.2, loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=False)
    _clean(ax); ax.set_title(title, fontsize=11)
    save(fig, name)


def fig_tsne_fashion():
    Z, y, names = _fashion_embed_input()
    E = TSNE(n_components=2, init="pca", perplexity=30, random_state=SEED).fit_transform(Z)
    _embed_plot(E, y, names, f"t-SNE of Fashion-MNIST ({EMBED_N} items)", "dr_tsne_fashion.pdf")


def fig_umap_fashion():
    Z, y, names = _fashion_embed_input()
    E = umap.UMAP(n_components=2, random_state=SEED).fit_transform(Z)
    _embed_plot(E, y, names, f"UMAP of Fashion-MNIST ({EMBED_N} items)", "dr_umap_fashion.pdf")


def fig_tsne_perplexity():
    Z, y, _ = _fashion_embed_input()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, perp in zip(axes, [5, 30, 100]):
        E = TSNE(n_components=2, init="pca", perplexity=perp, random_state=SEED).fit_transform(Z)
        ax.scatter(E[:, 0], E[:, 1], c=y, cmap="tab10", s=4, alpha=0.75)
        _clean(ax); ax.set_title(f"perplexity = {perp}", fontsize=10)
    fig.suptitle("same data, three perplexities --- the picture changes", fontsize=11, y=1.03)
    save(fig, "dr_tsne_perplexity.pdf")


def fig_compare_fashion():
    X, y, _, _ = _fashion()
    X, y = X[:EMBED_N], y[:EMBED_N]
    Z = PCA(n_components=PRE_PCA, random_state=SEED).fit_transform(X)
    embeds = [
        ("PCA", PCA(n_components=2, random_state=SEED).fit_transform(X)),
        ("t-SNE", TSNE(n_components=2, init="pca", perplexity=30, random_state=SEED).fit_transform(Z)),
        ("UMAP", umap.UMAP(n_components=2, random_state=SEED).fit_transform(Z)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (ttl, E) in zip(axes, embeds):
        sc = ax.scatter(E[:, 0], E[:, 1], c=y, cmap="tab10", s=5, alpha=0.75)
        _clean(ax); ax.set_title(ttl, fontsize=11)
    cb = fig.colorbar(sc, ax=axes, ticks=range(10), fraction=0.022, pad=0.01)
    cb.set_label("garment class")
    save(fig, "dr_compare_fashion.pdf")


def fig_clip_atlas():
    """DR on a real embedding space: UMAP of CLIP vectors, drawn with the actual photos.

    Reuses the file the clustering chapter already commits, so this costs no new data
    and no torch -- the embeddings and the JPEG thumbnails are both inside the npz.
    """
    d = np.load(CLIP_NPZ, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)      # CLIP space is cosine
    labels, names = d["labels"], d["class_names"]
    blob, offs = d["thumb_blob"], d["thumb_offsets"]

    # min_dist=0.8 on purpose. CLIP separates these ten classes so cleanly that the default
    # (0.1) collapses each class into a dot -- 281 thumbnails then stack into ten unreadable
    # clumps. Loosening the packing spreads each class into a visible patch of photos. This
    # only changes how tightly points may sit, not which points are neighbours.
    E = umap.UMAP(n_components=2, metric="cosine", n_neighbors=30, min_dist=0.8,
                  random_state=SEED).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.scatter(E[:, 0], E[:, 1], c=labels, cmap="tab10", s=18, alpha=0.45, linewidths=0)

    # Pick thumbnails on a GRID, not at random. UMAP clusters are tight, so a random sample
    # stacks 200 thumbnails onto a dozen spots and all but the topmost are hidden. One image
    # per occupied grid cell spreads them out and covers the whole map.
    # Percentile bounds, not min/max: a handful of far-flung points stretch the raw range so
    # far that everything else falls into a few cells (37 of 468 on the first attempt).
    # Sized against the loosened embedding above: 34x24 fills 252 of 816 cells, which is about
    # as many thumbnails as fit at this figure size without piling up.
    gx, gy = 34, 24
    x0, x1 = np.percentile(E[:, 0], [1, 99])
    y0, y1 = np.percentile(E[:, 1], [1, 99])
    cell_x = np.clip(((E[:, 0] - x0) / (x1 - x0) * gx).astype(int), 0, gx - 1)
    cell_y = np.clip(((E[:, 1] - y0) / (y1 - y0) * gy).astype(int), 0, gy - 1)
    chosen = {}
    for i, (cx, cy) in enumerate(zip(cell_x, cell_y)):
        chosen.setdefault((cx, cy), i)      # first point in each cell wins; npz is shuffled
    for i in chosen.values():
        img = Image.open(io.BytesIO(blob[offs[i]:offs[i + 1]].tobytes()))
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(img), zoom=0.40),
                                     E[i], frameon=False, pad=0))
    logging.info("clip atlas: %d thumbnails over a %dx%d grid", len(chosen), gx, gy)
    # No legend: the thumbnails cover the coloured scatter anyway, and the whole point of the
    # frame is that you can read the groups off the photos without a key.
    _clean(ax)
    ax.set_title("UMAP of 2000 photos through CLIP (512-d), drawn with the photos",
                 fontsize=10)
    save(fig, "dr_clip_atlas.pdf")


def main():
    setup_logging()
    FIG.mkdir(exist_ok=True)
    logging.info("generating dim-reduction figures -> %s", FIG)
    fig_fashion_samples()
    fig_pca_anim()
    fig_curse_distances()
    fig_byhand_pca()
    fig_scree()
    fig_scaling_trap()
    fig_biplot()
    fig_reconstruction()
    fig_swiss_roll()
    fig_tsne_fashion()
    fig_umap_fashion()
    fig_tsne_perplexity()
    fig_compare_fashion()
    fig_clip_atlas()
    logging.info("done.")


if __name__ == "__main__":
    main()
