#!/usr/bin/env python
"""Figures for the L13 Clustering deck (toy 2D blobs, matplotlib).

Outputs PDFs to the sibling ``fig/``. Run with the ma venv:
    ./ma/Scripts/python.exe ml/ch4_clustering/py_src/cluster_demos.py
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from scipy.cluster.hierarchy import dendrogram, linkage

SEED = 509
FIG = Path(__file__).resolve().parent.parent / "fig"
SRC = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
TAB = plt.cm.tab10.colors                       # cluster scatters (need >3 colors -> tab10)
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"  # flag colors for <=3-color plots

# 4-blob toy set: groups are discernible but overlap at the edges (not trivially separated).
BLOBS_CENTERS = np.array([[-3.2, -3.0], [3.4, -2.3], [-2.3, 3.6], [3.6, 3.2]])


def blobs4(n=320, std=1.25):
    return make_blobs(n_samples=n, centers=BLOBS_CENTERS, cluster_std=std, random_state=SEED)


def setup_logging():
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(logs / "cluster_demos.log", encoding="utf-8")],
    )


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    logging.info("wrote %s", name)


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.7")


def _scatter(ax, X, lab=None, s=14):
    if lab is None:
        ax.scatter(X[:, 0], X[:, 1], s=s, c="0.45", edgecolors="white", linewidths=0.3)
    else:
        for j in sorted(set(lab)):
            m = lab == j
            if j == -1:  # DBSCAN noise
                ax.scatter(X[m, 0], X[m, 1], s=s, c="0.2", marker="x", linewidths=0.8)
            else:
                ax.scatter(X[m, 0], X[m, 1], s=s, color=TAB[j % 10],
                           edgecolors="white", linewidths=0.3)
    _clean(ax)


# ----------------------------------------------------------------------
def fig_hook_unlabeled():
    X, _ = blobs4(std=1.25)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    _scatter(ax, X)
    ax.set_title("Unlabeled data: how many groups? which?", fontsize=11)
    save(fig, "clu_hook_unlabeled.pdf")


def _lloyd_history(X, k, init, n_iter=12):
    c = init.copy().astype(float)
    hist = []
    for _ in range(n_iter):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        hist.append((c.copy(), lab.copy()))
        newc = c.copy()
        for j in range(k):
            if (lab == j).any():
                newc[j] = X[lab == j].mean(0)
        if np.allclose(newc, c):
            break
        c = newc
    return hist


def fig_kmeans_anim(n_frames=6):
    """Step-by-step k-means: one PDF per state, shown as a beamer overlay."""
    X, _ = blobs4(std=1.25)
    rng = np.random.RandomState(SEED + 1)  # offset so the start isn't already well placed
    init = X[rng.choice(len(X), 4, replace=False)]
    hist = _lloyd_history(X, 4, init)
    for i in range(n_frames):
        c, lab = hist[min(i, len(hist) - 1)]
        fig, ax = plt.subplots(figsize=(5, 3.7))
        _scatter(ax, X, lab, s=12)
        ax.scatter(c[:, 0], c[:, 1], c="black", marker="X", s=110,
                   edgecolors="white", linewidths=1.3, zorder=5)
        step = "initial centroids" if i == 0 else (
            "converged" if i >= len(hist) - 1 else f"iteration {i}")
        ax.set_title(f"k-means --- {step}", fontsize=11)
        save(fig, f"clu_km_anim_{i + 1}.pdf")


def fig_worked():
    P = np.array([[1, 1], [2, 1], [4, 3], [5, 4], [2, 2]], dtype=float)
    names = ["P1", "P2", "P3", "P4", "P5"]
    mu = np.array([[1.5, 1.0], [4.0, 4.0]])  # init centroids, NOT on any data point
    lab = (((P[:, None] - mu[None]) ** 2).sum(2)).argmin(1)
    newmu = np.array([P[lab == j].mean(0) for j in range(2)])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.7))
    for i, (p, nm) in enumerate(zip(P, names)):
        a1.scatter(*p, color=TAB[lab[i]], s=90, edgecolors="white", zorder=3)
        a1.annotate(nm, p, textcoords="offset points", xytext=(6, 5), fontsize=9)
    a1.scatter(mu[:, 0], mu[:, 1], c="black", marker="X", s=130, edgecolors="white", zorder=4)
    a1.set_title("1. assign to nearest centroid", fontsize=10)
    for i, p in enumerate(P):
        a2.scatter(*p, color=TAB[lab[i]], s=90, edgecolors="white", zorder=3)
    a2.scatter(mu[:, 0], mu[:, 1], c="0.6", marker="x", s=90, zorder=2)
    a2.scatter(newmu[:, 0], newmu[:, 1], c="black", marker="X", s=130, edgecolors="white", zorder=4)
    for j in range(2):
        a2.annotate("", xy=newmu[j], xytext=mu[j],
                    arrowprops=dict(arrowstyle="->", color="0.5", lw=1.3))
    a2.set_title("2. move centroid to the cluster mean", fontsize=10)
    for a in (a1, a2):
        _clean(a); a.set_aspect("equal")
    save(fig, "clu_worked.pdf")


def fig_kmeans_init():
    X, _ = make_blobs(n_samples=320, centers=4, cluster_std=1.3, random_state=SEED)
    good = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=SEED).fit(X)
    worst = None  # worst of 60 single random inits -> a representative bad local optimum
    for s in range(60):
        km = KMeans(n_clusters=4, init="random", n_init=1, random_state=s).fit(X)
        if worst is None or km.inertia_ > worst.inertia_:
            worst = km
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, km, ttl in [(axes[0], good, "k-means++"), (axes[1], worst, "one bad random init")]:
        _scatter(ax, X, km.labels_, s=11)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                   c="black", marker="X", s=90, edgecolors="white", linewidths=1.2)
        ax.set_title(f"{ttl}\ninertia = {km.inertia_:.0f}", fontsize=10)
    save(fig, "clu_kmeans_init.pdf")


def fig_elbow():
    X, _ = blobs4(std=1.1)
    ks = range(1, 10)
    inertia = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(list(ks), inertia, "o-", color=ARM_BLUE, lw=2)
    ax.scatter([4], [inertia[3]], s=160, facecolors="none", edgecolors=ARM_RED, lw=2.2, zorder=5)
    ax.annotate("elbow at k=4", (4, inertia[3]), textcoords="offset points",
                xytext=(28, 18), fontsize=10, color=ARM_RED)
    ax.set_xlabel("number of clusters k"); ax.set_ylabel("inertia (WCSS)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "clu_elbow.pdf")


def fig_kmeans_fail():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    # A: two parallel elongated clusters -> k-means cuts across them
    rng = np.random.RandomState(SEED)
    th = np.deg2rad(35)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    base1 = np.column_stack([rng.randn(220) * 3.0, rng.randn(220) * 0.32]) @ R.T
    base2 = np.column_stack([rng.randn(220) * 3.0, rng.randn(220) * 0.32]) @ R.T
    perp = 1.9 * np.array([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)])
    Xa = np.vstack([base1, base2 + perp])
    la = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(Xa)
    _scatter(axes[0], Xa, la, s=11)
    axes[0].set_aspect("equal")
    axes[0].set_title("elongated clusters", fontsize=10)
    # B: unequal sizes / variances
    Xv, _ = make_blobs(n_samples=[450, 90, 90],
                       centers=[[-7, 0], [1, 1], [5, -2]],
                       cluster_std=[2.7, 0.5, 0.5], random_state=SEED)
    lv = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(Xv)
    _scatter(axes[1], Xv, lv, s=11)
    axes[1].set_title("unequal sizes / variances", fontsize=10)
    fig.suptitle("k-means forces round, balanced clusters", fontsize=11, y=1.04)
    save(fig, "clu_kmeans_fail.pdf")


def _circles_with_noise():
    X, _ = make_circles(n_samples=400, factor=0.45, noise=0.05, random_state=SEED)
    rng = np.random.RandomState(SEED)
    return np.vstack([X, rng.uniform(-1.3, 1.3, size=(25, 2))])  # scattered outliers


def fig_dbscan_circles():
    X = _circles_with_noise()
    km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(X)
    db = DBSCAN(eps=0.15, min_samples=5).fit_predict(X)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9))
    _scatter(axes[0], X, km, s=12); axes[0].set_aspect("equal")
    axes[0].set_title("k-means (k=2): splits left/right", fontsize=10)
    _scatter(axes[1], X, db, s=12); axes[1].set_aspect("equal")
    axes[1].set_title("DBSCAN: finds the rings (x = noise)", fontsize=10)
    save(fig, "clu_dbscan_circles.pdf")


def fig_dbscan_anim():
    """DBSCAN walkthrough (overlay): count -> core -> grow -> complete -> border -> noise."""
    rng = np.random.RandomState(SEED)
    b1 = rng.randn(28, 2) * 0.5 + [0, 0]
    b2 = rng.randn(28, 2) * 0.5 + [3.2, 0.3]
    noise = rng.uniform([-1.7, -1.9], [4.9, 2.3], size=(7, 2))
    X = np.vstack([b1, b2, noise])
    eps, ms = 0.65, 4
    db = DBSCAN(eps=eps, min_samples=ms).fit(X)
    lab = db.labels_
    core = np.zeros(len(X), bool); core[db.core_sample_indices_] = True
    c0 = lab == 0
    cand = np.where(c0 & core)[0]
    seed = cand[X[cand, 0].argmin()]                 # leftmost core in cluster 0
    dist = np.linalg.norm(X - X[seed], axis=1)

    def base(ax):
        ax.set_aspect("equal"); _clean(ax)
        ax.set_xlim(X[:, 0].min() - 0.6, X[:, 0].max() + 0.6)
        ax.set_ylim(X[:, 1].min() - 0.6, X[:, 1].max() + 0.6)

    def circ(ax, i, color, ls="--"):
        ax.add_patch(Circle(X[i], eps, fill=False, ls=ls, color=color, lw=1.5))

    def gray(ax, m):
        ax.scatter(X[m, 0], X[m, 1], s=24, c="0.72", edgecolors="white", linewidths=0.3)

    def col(ax, m, c):
        ax.scatter(X[m, 0], X[m, 1], s=30, color=c, edgecolors="white", linewidths=0.3)

    # 1: pick a point, count neighbors within eps
    fig, ax = plt.subplots(figsize=(5, 3.7)); base(ax)
    ax.scatter(X[:, 0], X[:, 1], s=24, c="0.6", edgecolors="white", linewidths=0.3)
    circ(ax, seed, ARM_BLUE); ax.scatter(*X[seed], s=60, color=ARM_BLUE, zorder=5)
    ax.set_title(r"1. pick a point: count neighbors within $\epsilon$", fontsize=10)
    save(fig, "clu_db_anim_1.pdf")

    # 2: >= min_samples -> core; its neighbors join
    near = dist <= eps
    fig, ax = plt.subplots(figsize=(5, 3.7)); base(ax)
    gray(ax, ~near); col(ax, near, TAB[0]); circ(ax, seed, ARM_BLUE)
    ax.set_title(r"2. $\geq$ min\_samples $\to$ core; neighbors join", fontsize=10)
    save(fig, "clu_db_anim_2.pdf")

    # 3: expand from a frontier core (density-reachable)
    g3 = c0 & (dist <= 1.8 * eps)
    fig, ax = plt.subplots(figsize=(5, 3.7)); base(ax)
    gray(ax, ~g3); col(ax, g3, TAB[0])
    front = np.where(g3 & core)[0]
    circ(ax, front[dist[front].argmax()], TAB[0], ls=":")
    ax.set_title("3. expand: each core pulls in its neighbors", fontsize=10)
    save(fig, "clu_db_anim_3.pdf")

    # 4: cluster 0 fully connected
    fig, ax = plt.subplots(figsize=(5, 3.7)); base(ax)
    gray(ax, ~c0); col(ax, c0, TAB[0])
    ax.set_title("4. cluster 0 complete (density-connected)", fontsize=10)
    save(fig, "clu_db_anim_4.pdf")

    # 5: other clusters grow; a border point attaches
    fig, ax = plt.subplots(figsize=(5, 3.7)); base(ax)
    gray(ax, lab == -1)
    for j in (0, 1):
        col(ax, lab == j, TAB[j])
    bord = np.where((lab >= 0) & (~core))[0]
    if len(bord):
        bp = bord[0]
        ax.scatter(*X[bp], s=160, facecolors="none", edgecolors="black", lw=1.6, zorder=6)
        ax.annotate("border\n(in reach, not core)", X[bp], textcoords="offset points",
                    xytext=(8, 10), fontsize=8)
    ax.set_title("5. other clusters grow; borders attach", fontsize=10)
    save(fig, "clu_db_anim_5.pdf")

    # 6: leftover sparse points = noise
    fig, ax = plt.subplots(figsize=(5, 3.7)); base(ax)
    _scatter(ax, X, lab, s=28)
    npt = np.where(lab == -1)[0]
    if len(npt):
        ax.scatter(*X[npt[0]], s=160, facecolors="none", edgecolors=ARM_RED, lw=1.6, zorder=6)
        ax.annotate("noise\n(too few neighbors)", X[npt[0]], textcoords="offset points",
                    xytext=(8, 10), fontsize=8, color=ARM_RED)
    ax.set_title("6. leftover sparse points = noise (x)", fontsize=10)
    save(fig, "clu_db_anim_6.pdf")


def fig_hdbscan():
    """Varying-density data: one global eps (DBSCAN) fails; HDBSCAN adapts."""
    rng = np.random.RandomState(SEED)
    dense = rng.randn(80, 2) * 0.22 + [0.0, 0.0]
    medium = rng.randn(60, 2) * 0.55 + [3.2, 0.4]
    sparse = rng.randn(45, 2) * 1.15 + [1.4, 4.2]
    noise = rng.uniform([-2.0, -2.0], [5.6, 6.8], size=(12, 2))
    X = np.vstack([dense, medium, sparse, noise])
    db = DBSCAN(eps=0.4, min_samples=6).fit_predict(X)
    hb = HDBSCAN(min_cluster_size=15).fit_predict(X)

    def nclust(lbl):
        return len(set(lbl)) - (1 if -1 in lbl else 0)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    _scatter(axes[0], X, db, s=16); axes[0].set_aspect("equal")
    axes[0].set_title(f"DBSCAN, one eps=0.4 -> {nclust(db)} clusters\n(sparse group lost to noise)", fontsize=9)
    _scatter(axes[1], X, hb, s=16); axes[1].set_aspect("equal")
    axes[1].set_title(f"HDBSCAN -> {nclust(hb)} clusters\n(adapts to each density)", fontsize=9)
    save(fig, "clu_hdbscan.pdf")


def fig_kdistance():
    X = _circles_with_noise()
    k = 5
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    kth = np.sort(d[:, -1])
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(np.arange(len(kth)), kth, color=ARM_BLUE, lw=2)
    knee = 0.15
    ax.axhline(knee, color=ARM_RED, ls="--", lw=1.4)
    ax.annotate(r"knee $\to$ $\epsilon \approx 0.15$", (len(kth) * 0.05, knee),
                textcoords="offset points", xytext=(6, 8), fontsize=10, color=ARM_RED)
    ax.set_xlabel("points, sorted"); ax.set_ylabel(f"distance to {k}-th neighbor")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "clu_kdistance.pdf")


def fig_dendrogram():
    X, _ = make_blobs(n_samples=28, centers=3, cluster_std=0.8, random_state=SEED)
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    dendrogram(Z, ax=ax, color_threshold=0.5 * Z[:, 2].max(), no_labels=True)
    ax.set_ylabel("merge distance (Ward)")
    ax.set_xlabel("data points (each leaf = one sample)")
    ax.set_title("Dendrogram: cut the tree to choose #clusters", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "clu_dendrogram.pdf")


def fig_linkage():
    X, _ = make_moons(n_samples=300, noise=0.06, random_state=SEED)
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, link in zip(axes, ["single", "complete", "average", "ward"]):
        lab = AgglomerativeClustering(n_clusters=2, linkage=link).fit_predict(X)
        _scatter(ax, X, lab, s=9)
        ax.set_title(link, fontsize=10)
    fig.suptitle("Linkage changes the clusters (same data, k=2)", fontsize=11, y=1.05)
    save(fig, "clu_linkage.pdf")


def _draw_ellipse(ax, mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for n in (1, 2):
        w, h = 2 * n * np.sqrt(vals)
        ax.add_patch(Ellipse(mean, w, h, angle=angle, fill=False,
                             edgecolor=color, lw=1.6, alpha=0.8))


def fig_gmm_vs_kmeans():
    X, _ = make_blobs(n_samples=500, centers=3, cluster_std=1.1, random_state=SEED)
    X = X @ np.array([[0.7, -0.55], [-0.3, 0.9]])
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X)
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=SEED).fit(X)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    Z = km.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[0].contourf(xx, yy, Z, alpha=0.12, colors=[TAB[i] for i in range(3)], levels=[-.5, .5, 1.5, 2.5])
    _scatter(axes[0], X, km.labels_, s=9)
    axes[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                    c="black", marker="X", s=80, edgecolors="white", linewidths=1.1)
    axes[0].set_title("k-means: round, hard (Voronoi)", fontsize=10)
    gl = gmm.predict(X)
    _scatter(axes[1], X, gl, s=9)
    for k in range(3):
        _draw_ellipse(axes[1], gmm.means_[k], gmm.covariances_[k], TAB[k])
    axes[1].set_title("GMM: elliptical, soft", fontsize=10)
    save(fig, "clu_gmm_vs_kmeans.pdf")


def fig_silhouette():
    X, _ = blobs4(std=1.3)
    lab = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit_predict(X)
    sil = silhouette_samples(X, lab)
    avg = silhouette_score(X, lab)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.6, 3.7))
    y = 10
    for j in range(4):
        vals = np.sort(sil[lab == j])
        axl.fill_betweenx(np.arange(y, y + len(vals)), 0, vals, color=TAB[j], alpha=0.8)
        y += len(vals) + 10
    axl.axvline(avg, color="red", ls="--", lw=1.3, label=f"mean = {avg:.2f}")
    axl.set_xlabel("silhouette value")
    axl.set_ylabel("points (grouped by cluster)")
    axl.set_yticks([]); axl.legend(fontsize=9, loc="lower right")
    axl.set_title("silhouette per point, by cluster", fontsize=10)
    axl.spines[["top", "right", "left"]].set_visible(False)
    _scatter(axr, X, lab, s=11); axr.set_title("the clustering", fontsize=10)
    save(fig, "clu_silhouette.pdf")


def fig_image_quantization():
    img = plt.imread(SRC / "saryan_mountains.jpg").astype(float) / 255.0
    w, h, d = img.shape
    X = img.reshape(-1, d)
    sample = shuffle(X, random_state=SEED)[:10000]
    k = 16
    km = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(sample)
    quant = km.cluster_centers_[km.predict(X)].reshape(w, h, d)
    n_orig = len(np.unique((X * 255).astype(np.uint8), axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].imshow(img); axes[0].set_title(f"original ({n_orig:,} colors)", fontsize=10)
    axes[1].imshow(quant); axes[1].set_title(f"k-means, {k} colors", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "clu_image_quantization.pdf")


def fig_ari_points():
    """The 11 points behind the ARI worked example: cluster (dashed) x class (color)."""
    rng = np.random.RandomState(SEED)
    c1 = rng.randn(6, 2) * 0.42 + [0.0, 0.0]
    c2 = rng.randn(5, 2) * 0.42 + [3.0, 0.0]
    P = np.vstack([c1, c2])
    cls = np.array(["A"] * 5 + ["B"] * 1 + ["A"] * 2 + ["B"] * 3)  # C1: 5A,1B ; C2: 2A,3B
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for center, lab in [((0, 0), "$C_1$ (cluster)"), ((3, 0), "$C_2$ (cluster)")]:
        ax.add_patch(Ellipse(center, 2.6, 2.6, fill=False, ls="--", ec="0.5", lw=1.5))
        ax.text(center[0], center[1] + 1.55, lab, ha="center", fontsize=11, color="0.4")
    for cl, color in [("A", ARM_BLUE), ("B", ARM_RED)]:
        m = cls == cl
        ax.scatter(P[m, 0], P[m, 1], color=color, s=95, edgecolors="white",
                   zorder=3, label=f"class {cl}")
    ax.legend(fontsize=10, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16))
    _clean(ax); ax.set_aspect("equal")
    ax.set_xlim(-1.8, 4.8); ax.set_ylim(-1.8, 2.4)
    save(fig, "clu_ari_points.pdf")


def fig_agglo_anim():
    """Agglomerative merging: scatter with cluster MIDPOINTS + a dendrogram that builds up.

    Uses centroid linkage, so the merge distance d = distance between the two cluster midpoints.
    """
    X = np.array([[0.2, 0.2], [0.7, 0.5], [0.4, 1.0], [3.0, 0.3], [3.5, 0.7],
                  [1.8, 2.7], [2.4, 3.0]], dtype=float)
    n = len(X)
    Z = linkage(X, method="centroid")
    comp = {i: [i] for i in range(n)}
    for t in range(len(Z)):
        comp[n + t] = comp[int(Z[t, 0])] + comp[int(Z[t, 1])]
    active = set(range(n)); steps = [(sorted(active), None, None)]
    for t in range(len(Z)):
        a, b, d = int(Z[t, 0]), int(Z[t, 1]), Z[t, 2]
        merged = (comp[a], comp[b])
        active.discard(a); active.discard(b); active.add(n + t)
        steps.append((sorted(active), merged, d))

    # manual dendrogram layout so we can reveal it one merge at a time
    order = dendrogram(Z, no_plot=True)["leaves"]
    nx = {leaf: i for i, leaf in enumerate(order)}
    ht = {i: 0.0 for i in range(n)}
    links = []
    for t in range(len(Z)):
        a, b, d = int(Z[t, 0]), int(Z[t, 1]), Z[t, 2]
        links.append((nx[a], ht[a], nx[b], ht[b], d))
        nx[n + t] = (nx[a] + nx[b]) / 2.0; ht[n + t] = d
    hmax = float(Z[:, 2].max())

    def base(ax):
        ax.set_aspect("equal"); _clean(ax)
        ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.8)

    for s, (act, merged, d) in enumerate(steps):
        fig, (aL, aR) = plt.subplots(1, 2, figsize=(8.8, 3.7),
                                     gridspec_kw={"width_ratios": [1.15, 1.0]})
        base(aL)
        ci = 0
        for cid in act:
            pts = X[comp[cid]]
            if len(comp[cid]) == 1:
                aL.scatter(pts[:, 0], pts[:, 1], s=55, color="0.6", edgecolors="white", zorder=3)
            else:
                color = TAB[ci % 10]; ci += 1
                aL.scatter(pts[:, 0], pts[:, 1], s=55, color=color, edgecolors="white", zorder=3)
                aL.scatter(*pts.mean(0), marker="x", s=80, color=color, lw=2.4, zorder=4)  # midpoint
        if merged is not None:
            pa, pb = X[merged[0]].mean(0), X[merged[1]].mean(0)
            aL.scatter([pa[0], pb[0]], [pa[1], pb[1]], marker="x", s=85, color="black", lw=2.4, zorder=5)
            aL.plot([pa[0], pb[0]], [pa[1], pb[1]], color="black", lw=1.6, ls="--", zorder=2)
            aL.set_title(f"merge {s}: nearest midpoints, d = {d:.2f}", fontsize=9)
        else:
            aL.set_title("start: every point is its own cluster", fontsize=9)
        # right: dendrogram revealed up to s merges (latest in red)
        for i in range(s):
            x1, h1, x2, h2, dd = links[i]
            c = ARM_RED if i == s - 1 else "0.35"
            lw = 2.3 if i == s - 1 else 1.4
            aR.plot([x1, x1], [h1, dd], color=c, lw=lw)
            aR.plot([x1, x2], [dd, dd], color=c, lw=lw)
            aR.plot([x2, x2], [h2, dd], color=c, lw=lw)
        aR.set_xlim(-0.6, n - 0.4); aR.set_ylim(0, hmax * 1.12)
        aR.set_xticks([]); aR.set_ylabel("merge distance", fontsize=8)
        aR.set_xlabel("data points (leaves)", fontsize=8)
        aR.tick_params(labelsize=7)
        aR.set_title("dendrogram", fontsize=9)
        aR.spines[["top", "right"]].set_visible(False)
        save(fig, f"clu_agglo_anim_{s + 1}.pdf")


def main():
    setup_logging()
    FIG.mkdir(exist_ok=True)
    logging.info("generating clustering figures -> %s", FIG)
    fig_hook_unlabeled()
    fig_kmeans_anim()
    fig_worked()
    fig_kmeans_init()
    fig_elbow()
    fig_kmeans_fail()
    fig_dbscan_circles()
    fig_dbscan_anim()
    fig_hdbscan()
    fig_kdistance()
    fig_dendrogram()
    fig_agglo_anim()
    fig_linkage()
    fig_gmm_vs_kmeans()
    fig_silhouette()
    fig_ari_points()
    fig_image_quantization()
    logging.info("done.")


if __name__ == "__main__":
    main()
