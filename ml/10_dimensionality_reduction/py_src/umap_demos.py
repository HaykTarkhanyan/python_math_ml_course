#!/usr/bin/env python
"""Figures for 36_umap -- UMAP in depth.

Two families:

* **Toy figures** trace UMAP's high-dimensional graph construction on 6 labelled points.
  Every number on the slides comes from here, so the deck cannot drift from the math.
  The toy runs at ``k=3``, not ``k=2``: UMAP's binary search targets ``log2(k)``, and at
  ``k=2`` the nearest neighbour alone already contributes exactly ``log2(2) = 1``, which
  drives ``sigma -> 0``. At ``k=3`` the target 1.585 gives a genuine ``sigma_A ~ 0.106``.
* **Real-embedding figures** run UMAP on Fashion-MNIST at the settings the slides discuss,
  so the hyperparameter frames show what the knobs actually do rather than a sketch of it.

Outputs PDFs to the sibling ``fig/``. Run with the ma venv:
    ./ma/Scripts/python.exe ml/10_dimensionality_reduction/py_src/umap_demos.py
(Fits UMAP about a dozen times; allow several minutes.)
"""
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
import umap
from umap.umap_ import find_ab_params

SEED = 509
CHAPTER = Path(__file__).resolve().parent.parent
FIG = CHAPTER / "fig"
ROOT = Path(__file__).resolve().parents[3]
FASHION_NPZ = CHAPTER / "data" / "fashion_mnist.npz"

TAB = plt.cm.tab10.colors
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

EMBED_N = 5000
PRE_PCA = 50

# The running toy: two clusters of three, well separated.
#
# The apex of each triangle is deliberately OFF-CENTRE (0.45 not 0.50, 5.45 not 5.50).
# With a centred apex, C sits at exactly the same distance from A and from B, so two
# neighbours tie at rho and the sum sum_j exp(-(d_j - rho)/sigma) is pinned at >= 2 no
# matter how small sigma gets -- while the target is log2(3) = 1.585. The binary search
# then has no solution at all. Nudging the apex breaks the tie.
TOY = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.45, 0.8),
       "D": (5.0, 0.0), "E": (6.0, 0.0), "F": (5.45, 0.7)}
TOY_NAMES = list(TOY)
TOY_XY = np.array([TOY[n] for n in TOY_NAMES])
TOY_CLUSTER = np.array([0, 0, 0, 1, 1, 1])
K = 3


def setup_logging():
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(logs / "umap_demos.log", encoding="utf-8")],
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
# UMAP's high-dimensional graph, computed the way the paper defines it


def toy_distances():
    d = np.linalg.norm(TOY_XY[:, None, :] - TOY_XY[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)                       # a point is not its own neighbour
    return d


def smooth_knn(dists_i, k, tol=1e-9, max_iter=100):
    """One point's rho and sigma. rho = distance to the nearest neighbour; sigma solves
    sum_j exp(-max(d_j - rho, 0)/sigma) = log2(k) over the k nearest neighbours."""
    nn = np.sort(dists_i)[:k]
    rho = nn[0]
    target = np.log2(k)
    shifted = np.maximum(nn - rho, 0.0)

    lo, hi = 0.0, np.inf
    sigma = 1.0
    for _ in range(max_iter):
        total = np.exp(-shifted / sigma).sum()
        if abs(total - target) < tol:
            break
        if total > target:                            # too much mass -> shrink sigma
            hi = sigma
            sigma = (lo + hi) / 2.0
        else:
            lo = sigma
            sigma = sigma * 2.0 if hi == np.inf else (lo + hi) / 2.0
    else:
        raise RuntimeError(f"sigma binary search did not converge (last total={total})")
    return rho, sigma


def toy_graph():
    """(rho, sigma, directed p_{j|i}, symmetrised p_ij) for the toy."""
    d = toy_distances()
    n = len(TOY_NAMES)
    rho = np.zeros(n); sigma = np.zeros(n)
    for i in range(n):
        rho[i], sigma[i] = smooth_knn(d[i], K)

    p_dir = np.zeros((n, n))
    for i in range(n):
        nbrs = np.argsort(d[i])[:K]
        for j in nbrs:
            p_dir[i, j] = np.exp(-max(d[i, j] - rho[i], 0.0) / sigma[i])
    p_sym = p_dir + p_dir.T - p_dir * p_dir.T          # fuzzy union
    return rho, sigma, p_dir, p_sym


# ----------------------------------------------------------------------
def fig_toy_points():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, name in enumerate(TOY_NAMES):
        col = ARM_BLUE if TOY_CLUSTER[i] == 0 else ARM_RED
        ax.scatter(*TOY_XY[i], s=190, color=col, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(TOY_XY[i, 0], TOY_XY[i, 1], name, color="white", fontsize=10,
                fontweight="bold", ha="center", va="center", zorder=4)
    d = toy_distances()
    ax.annotate("", xy=TOY_XY[1], xytext=TOY_XY[0],
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1))
    ax.text(0.5, -0.22, f"{d[0,1]:.2f}", color="0.35", fontsize=8, ha="center")
    ax.annotate("", xy=TOY_XY[3], xytext=TOY_XY[1],
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1, linestyle=":"))
    ax.text(3.0, 0.12, f"{d[1,3]:.2f}", color="0.35", fontsize=8, ha="center")
    ax.set_xlim(-0.8, 6.8); ax.set_ylim(-0.6, 1.3)
    ax.set_title(f"the running toy: 6 points, two clusters, k={K}", fontsize=10)
    _clean(ax); ax.set_aspect("equal")
    save(fig, "umap_toy_points.pdf")


def fig_rho_sigma():
    """The local decay curve, with the real binary-search sigma for point A."""
    d = toy_distances()
    rho, sigma, p_dir, _ = toy_graph()
    iA = TOY_NAMES.index("A")
    rA, sA = rho[iA], sigma[iA]
    logging.info("toy rho = %s", dict(zip(TOY_NAMES, np.round(rho, 4))))
    logging.info("toy sigma = %s", dict(zip(TOY_NAMES, np.round(sigma, 4))))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = np.linspace(0, 0.30, 400)
    ax.plot(xs, np.exp(-xs / sA), color=ARM_BLUE, lw=2.4)
    for name, col in (("C", ARM_ORANGE), ("B", ARM_RED)):
        j = TOY_NAMES.index(name)
        x = max(d[iA, j] - rA, 0.0)
        y = np.exp(-x / sA)
        ax.scatter([x], [y], s=90, color=col, zorder=5, edgecolors="white", linewidths=1.2)
        ax.annotate(f"$p_{{{name}|A}}$ = {y:.3f}\n$d$ = {d[iA, j]:.3f}",
                    (x, y), textcoords="offset points", xytext=(16, -4),
                    fontsize=9, color=col)
    ax.axhline(1.0, ls=":", color="0.6", lw=1)
    ax.set_xlabel(r"$\max(d(A,j) - \rho_A,\ 0)$")
    ax.set_ylabel(r"$p_{j|A}$")
    ax.set_title(rf"point A: $\rho_A$ = {rA:.4f}, binary search gives $\sigma_A$ = {sA:.4f}"
                 "\n" rf"(target $\log_2 k$ = {np.log2(K):.3f})", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "umap_rho_sigma.pdf")


def fig_toy_graph():
    """One large graph with the symmetrised weights printed on the edges.

    An earlier version showed the directed and symmetrised graphs side by side, but the two
    render almost identically at slide size (the directed pair overlaps into one line), so the
    panel cost half the width and taught nothing. The numbers are the payload.
    """
    _, _, _, p_sym = toy_graph()
    n = len(TOY_NAMES)
    fig, ax = plt.subplots(figsize=(9.2, 2.9))

    for i in range(n):
        for j in range(i + 1, n):
            w = p_sym[i, j]
            if w <= 1e-3:
                continue
            ax.plot(*zip(TOY_XY[i], TOY_XY[j]), color="0.4", lw=0.6 + 4.5 * w,
                    alpha=0.7, zorder=1, solid_capstyle="round")
            mid = (TOY_XY[i] + TOY_XY[j]) / 2
            ax.text(mid[0], mid[1] + 0.10, f"{w:.3f}", fontsize=8.5, color="0.15",
                    ha="center", va="bottom", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    # The one cross-cluster pair that IS on a neighbour list, to show what it is worth.
    ax.plot(*zip(TOY_XY[0], TOY_XY[3]), color=ARM_RED, lw=0.7, ls=":", alpha=0.8, zorder=1)
    ax.text(2.95, -0.30, f"A-D = {p_sym[0, 3]:.4f}", fontsize=8.5, color=ARM_RED,
            ha="center", va="center")

    for i, name in enumerate(TOY_NAMES):
        col = ARM_BLUE if TOY_CLUSTER[i] == 0 else ARM_RED
        ax.scatter(*TOY_XY[i], s=300, color=col, zorder=3, edgecolors="white", linewidths=1.8)
        ax.text(TOY_XY[i, 0], TOY_XY[i, 1], name, color="white", fontsize=11,
                fontweight="bold", ha="center", va="center", zorder=4)

    ax.set_xlim(-0.7, 6.7); ax.set_ylim(-0.55, 1.35)
    ax.set_title(r"the symmetrised graph $p_{ij}$ -- edge width and label $=$ weight",
                 fontsize=11)
    _clean(ax); ax.set_aspect("equal")
    logging.info("toy p_sym A-B %.4f  A-C %.4f  A-D %.4f",
                 p_sym[0, 1], p_sym[0, 2], p_sym[0, 3])
    save(fig, "umap_toy_graph.pdf")


def fig_lowdim_kernel():
    """q(d) for two min_dist values, using UMAP's own fitted a, b."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    xs = np.linspace(0, 3, 400)
    for md, col in ((0.0, ARM_BLUE), (0.8, ARM_ORANGE)):
        a, b = find_ab_params(spread=1.0, min_dist=md)
        ax.plot(xs, 1.0 / (1.0 + a * xs ** (2 * b)), lw=2.4, color=col,
                label=f"min_dist = {md}   (a = {a:.2f}, b = {b:.2f})")
        logging.info("find_ab_params(min_dist=%.2f) -> a=%.4f b=%.4f", md, a, b)
    ax.set_xlabel(r"low-dimensional distance $\|y_i - y_j\|$")
    ax.set_ylabel(r"$q_{ij}$")
    ax.set_title("min_dist only reshapes this curve -- how close points\n"
                 "are allowed to sit before the similarity stops rising", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "umap_lowdim_kernel.pdf")


def fig_attract_repel():
    """The two halves of the cross-entropy, and where each one bites."""
    q = np.linspace(1e-3, 1 - 1e-3, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(q, -np.log(q), color=ARM_RED, lw=2.4,
            label=r"attraction: $-p_{ij}\log q_{ij}$  (neighbours, $p\approx 1$)")
    ax.plot(q, -np.log(1 - q), color=ARM_BLUE, lw=2.4,
            label=r"repulsion: $-(1-p_{ij})\log(1-q_{ij})$  (non-neighbours, $p\approx 0$)")
    ax.set_ylim(0, 5)
    ax.set_xlabel(r"$q_{ij}$  (low-dimensional similarity)")
    ax.set_ylabel("loss contribution")
    ax.set_title("neighbours pay for being far apart, non-neighbours pay for being close\n"
                 "t-SNE's KL has only the first term", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "umap_attract_repel.pdf")


# ----------------------------------------------------------------------
# real embeddings


def _fashion_pca():
    d = np.load(FASHION_NPZ, allow_pickle=True)
    imgs = d["images"][:EMBED_N]
    X = imgs.reshape(len(imgs), -1).astype(np.float64)
    return PCA(n_components=PRE_PCA, random_state=SEED).fit_transform(X), d["labels"][:EMBED_N]


def _panels(runs, title, name, suptitle_y=1.02):
    """runs: list of (subtitle, embedding, y)."""
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 3.5))
    for ax, (sub, E, y) in zip(np.atleast_1d(axes), runs):
        ax.scatter(E[:, 0], E[:, 1], c=y, cmap="tab10", s=4, alpha=0.75, linewidths=0)
        _clean(ax); ax.set_title(sub, fontsize=10)
    fig.suptitle(title, fontsize=11, y=suptitle_y)
    save(fig, name)


def fig_n_neighbors():
    Z, y = _fashion_pca()
    runs = []
    for k in (5, 15, 50):
        E = umap.UMAP(n_neighbors=k, min_dist=0.1, random_state=SEED).fit_transform(Z)
        runs.append((f"n_neighbors = {k}", E, y))
        logging.info("n_neighbors=%d done", k)
    _panels(runs, "small k sees local detail, large k sees the global shape",
            "umap_n_neighbors.pdf")


def fig_min_dist():
    Z, y = _fashion_pca()
    runs = []
    for md in (0.0, 0.1, 0.8):
        E = umap.UMAP(n_neighbors=15, min_dist=md, random_state=SEED).fit_transform(Z)
        runs.append((f"min_dist = {md}", E, y))
        logging.info("min_dist=%.1f done", md)
    _panels(runs, "min_dist changes packing, not which points are neighbours",
            "umap_min_dist.pdf")


def fig_seeds():
    Z, y = _fashion_pca()
    runs = []
    for s in (509, 42, 2026):
        E = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=s).fit_transform(Z)
        runs.append((f"random_state = {s}", E, y))
        logging.info("seed=%d done", s)
    _panels(runs, "same data, same settings, three seeds:\n"
                  "the clusters survive, their positions and orientation do not",
            "umap_seeds.pdf", suptitle_y=1.06)


def fig_epochs():
    Z, y = _fashion_pca()
    runs = []
    for ep in (10, 30, 100, 500):
        E = umap.UMAP(n_neighbors=15, min_dist=0.1, n_epochs=ep,
                      random_state=SEED).fit_transform(Z)
        runs.append((f"{ep} epochs", E, y))
        logging.info("n_epochs=%d done", ep)
    _panels(runs, "the springs settling: attraction and repulsion reaching equilibrium",
            "umap_epochs.pdf")


def main():
    setup_logging()
    FIG.mkdir(exist_ok=True)
    logging.info("generating UMAP figures -> %s", FIG)
    fig_toy_points()
    fig_rho_sigma()
    fig_toy_graph()
    fig_lowdim_kernel()
    fig_attract_repel()
    fig_n_neighbors()
    fig_min_dist()
    fig_seeds()
    fig_epochs()
    logging.info("done.")


if __name__ == "__main__":
    main()
