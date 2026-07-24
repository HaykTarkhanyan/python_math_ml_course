"""Generate figures for the classic-methods deck (SVM-centered + KNN/NB/LDA-QDA/GP).

Real matplotlib on tiny 2D/1D toys, house palette (Armenian-flag colors), seed 509.
Outputs PDFs to ../fig/ ; logs to ./logs/. Run with the ma venv:
    ./ma/Scripts/python.exe ml/04b_classic_methods/py_src/make_classic_figures.py
"""
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

SEED = 509
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "fig"
FIG.mkdir(exist_ok=True)
(HERE / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(HERE / "logs" / "make_classic_figures.log"),
    ],
)
log = logging.getLogger(__name__)

# House palette
RED = "#C81E28"      # armred  (class -1 / negative)
BLUE = "#1E46A0"     # armblue (class +1 / positive)
ORANGE = "#E6A01E"   # armorange (boundary / highlight)
GREEN = "#008C46"    # paramgreen
POP = "#3465A4"      # popblue
GREY = "#999999"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CMAP = matplotlib.colors.ListedColormap([RED, BLUE])
CMAP_LIGHT = matplotlib.colors.ListedColormap(["#F3CDD0", "#CBD6EC"])


def _mesh(X, h=0.02, pad=0.5):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def _scatter(ax, X, y):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=RED, s=22, edgecolor="white",
               linewidth=0.5, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=BLUE, s=22, edgecolor="white",
               linewidth=0.5, zorder=3)


def _cov_ellipse(ax, mean, cov, color, nstd=2.0):
    """Draw the nstd-sigma Gaussian contour ellipse for (mean, cov)."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nstd * np.sqrt(vals)
    e = Ellipse(xy=mean, width=w, height=h, angle=angle, fill=False,
                edgecolor=color, lw=2.0, ls="--", zorder=4)
    ax.add_patch(e)


def save(fig, name):
    out = FIG / name
    fig.savefig(out)
    plt.close(fig)
    log.info(f"wrote {out}")


# ---------------------------------------------------------------- KNN boundary
def knn_boundary():
    X, y = make_moons(n_samples=200, noise=0.30, random_state=SEED)
    xx, yy = _mesh(X)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, k in zip(axes, [1, 50]):
        clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.9)
        ax.contour(xx, yy, Z, levels=[0.5], colors=ORANGE, linewidths=2)
        _scatter(ax, X, y)
        ax.set_title(f"k = {k}" + ("  (jagged, overfit)" if k == 1
                                   else "  (smooth, more bias)"))
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "cm_knn_boundary.pdf")


# ---------------------------------------------------------------- LDA vs QDA
def lda_qda():
    rng = np.random.RandomState(SEED)
    n = 200
    # class 0: tight, class 1: wide + rotated -> QDA should curve
    X0 = rng.multivariate_normal([-1.5, 0], [[0.5, 0], [0, 0.5]], n)
    X1 = rng.multivariate_normal([1.5, 0.5], [[2.2, 1.4], [1.4, 1.6]], n)
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    xx, yy = _mesh(X, pad=1.0)

    # class means + covariances; pooled (shared) covariance is the LDA assumption
    m0, m1 = X0.mean(0), X1.mean(0)
    C0, C1 = np.cov(X0.T), np.cov(X1.T)
    C_pool = 0.5 * (C0 + C1)

    specs = [
        ("LDA: shared covariance -> linear", LinearDiscriminantAnalysis(),
         [(m0, C_pool), (m1, C_pool)]),   # both classes: same ellipse shape
        ("QDA: per-class covariance -> curved", QuadraticDiscriminantAnalysis(),
         [(m0, C0), (m1, C1)]),           # each class: its own ellipse
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, (name, clf, ells) in zip(axes, specs):
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.9)
        ax.contour(xx, yy, Z, levels=[0.5], colors=ORANGE, linewidths=2)
        _scatter(ax, X, y)
        for (mean, cov), col in zip(ells, [RED, BLUE]):
            _cov_ellipse(ax, mean, cov, col)
            ax.scatter(*mean, c=col, s=110, marker="X", edgecolor="white",
                       linewidth=1.3, zorder=5)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "cm_lda_qda.pdf")


# ---------------------------------------------------------- SVM hard margin
def svm_margin():
    X, y = make_blobs(n_samples=60, centers=[[-1.6, -1.2], [1.6, 1.2]],
                      cluster_std=0.85, random_state=SEED)
    clf = SVC(kernel="linear", C=1000).fit(X, y)
    xx, yy = _mesh(X, h=0.01, pad=0.8)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[GREY, ORANGE, GREY],
               linestyles=["--", "-", "--"], linewidths=[1.4, 2.2, 1.4])
    ax.contourf(xx, yy, (Z > 0).astype(int), cmap=CMAP_LIGHT, alpha=0.55)
    _scatter(ax, X, y)
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=170, facecolors="none",
               edgecolors="black", linewidths=1.6, zorder=4,
               label="support vectors")
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "cm_svm_margin.pdf")


# ---------------------------------------------------------- SVM soft margin C
def svm_soft_margin():
    X, y = make_blobs(n_samples=120, centers=[[-1.1, -0.8], [1.1, 0.8]],
                      cluster_std=1.5, random_state=SEED)
    xx, yy = _mesh(X, h=0.01, pad=0.8)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, C in zip(axes, [0.05, 100]):
        clf = SVC(kernel="linear", C=C).fit(X, y)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[GREY, ORANGE, GREY],
                   linestyles=["--", "-", "--"], linewidths=[1.2, 2, 1.2])
        _scatter(ax, X, y)
        nsv = len(clf.support_vectors_)
        ax.set_title(f"C = {C}  ({'wide margin' if C < 1 else 'narrow margin'}, "
                     f"{nsv} SVs)")
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "cm_svm_soft_margin.pdf")


# ---------------------------------------------------------------- hinge loss
def hinge_loss():
    m = np.linspace(-2.2, 3.0, 400)
    hinge = np.maximum(0, 1 - m)
    sq_hinge = np.maximum(0, 1 - m) ** 2
    logloss = np.log2(1 + np.exp(-m))
    zero_one = (m < 0).astype(float)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(m, zero_one, color=GREY, lw=2, label="0-1 loss (what we want)")
    ax.plot(m, hinge, color=ORANGE, lw=2.4, label="hinge  max(0, 1$-$yf)  [SVM]")
    ax.plot(m, logloss, color=BLUE, lw=2.2, label="log-loss  [logistic reg.]")
    ax.plot(m, sq_hinge, color=RED, lw=1.8, ls="--", label="squared hinge  [LS-SVM]")
    ax.axvline(1, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"margin  $y \cdot f(x)$")
    ax.set_ylabel("loss")
    ax.set_ylim(-0.15, 3.2)
    ax.legend(fontsize=9.5, loc="upper right")
    save(fig, "cm_hinge_loss.pdf")


# ------------------------------------------------------------ circles + lift
def circles_lift():
    X, y = make_circles(n_samples=200, factor=0.35, noise=0.09, random_state=SEED)
    z = X[:, 0] ** 2 + X[:, 1] ** 2
    fig = plt.figure(figsize=(8.6, 3.9))
    ax1 = fig.add_subplot(1, 2, 1)
    _scatter(ax1, X, y)
    ax1.set_title(r"2D: no line separates")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], z[y == 0], c=RED, s=18, depthshade=False)
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], z[y == 1], c=BLUE, s=18, depthshade=False)
    # separating plane at z = threshold
    thr = 0.5 * (z[y == 0].min() + z[y == 1].max())
    gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 8), np.linspace(-1.2, 1.2, 8))
    ax2.plot_surface(gx, gy, np.full_like(gx, thr), alpha=0.25, color=ORANGE)
    ax2.set_title(r"lift $\phi=(x_1,x_2,x_1^2{+}x_2^2)$: a plane separates")
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    save(fig, "cm_circles_lift.pdf")


# ------------------------------------------------------- SVM linear vs RBF
def svm_kernels():
    X, y = make_circles(n_samples=250, factor=0.35, noise=0.11, random_state=SEED)
    xx, yy = _mesh(X, h=0.01, pad=0.4)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, (name, clf) in zip(axes, [
        ("linear kernel: fails", SVC(kernel="linear", C=1)),
        ("RBF kernel: bends around the data", SVC(kernel="rbf", C=5, gamma=1.5)),
    ]):
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.9)
        ax.contour(xx, yy, Z, levels=[0.5], colors=ORANGE, linewidths=2)
        _scatter(ax, X, y)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    save(fig, "cm_svm_kernels.pdf")


# ----------------------------------------------------------------- GP band
def gp_band():
    # 1D rent-vs-area toy (continuity with L09/L11 synthetic Yerevan rent)
    rng = np.random.RandomState(SEED)
    Xtr = np.array([28, 40, 52, 66, 80, 104]).reshape(-1, 1).astype(float)
    ytr = 2.9 * Xtr.ravel() + 120 + rng.normal(0, 18, Xtr.shape[0])
    kernel = ConstantKernel(1.0) * RBF(length_scale=18.0) + WhiteKernel(1.0)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=3, random_state=SEED)
    gp.fit(Xtr, ytr)
    Xte = np.linspace(15, 120, 300).reshape(-1, 1)
    mu, sd = gp.predict(Xte, return_std=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.fill_between(Xte.ravel(), mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.18,
                    label=r"$\pm 2\sigma$ uncertainty")
    ax.plot(Xte.ravel(), mu, color=BLUE, lw=2.2, label="GP mean")
    ax.scatter(Xtr.ravel(), ytr, color=RED, s=45, zorder=4, label="data")
    ax.set_xlabel(r"area (m$^2$)")
    ax.set_ylabel("rent (kAMD)")
    ax.set_title("GP: band pinches at data, balloons away")
    ax.legend(fontsize=10, loc="upper left")
    save(fig, "cm_gp_band.pdf")


# --------------------------------------------------------------- synthesis
def synthesis():
    X, y = make_moons(n_samples=220, noise=0.26, random_state=SEED)
    xx, yy = _mesh(X, h=0.02)
    grid = np.c_[xx.ravel(), yy.ravel()]
    panels = [
        ("KNN (k=15): local", KNeighborsClassifier(15), False),
        ("QDA: Gaussian blobs", QuadraticDiscriminantAnalysis(), False),
        ("SVM (RBF): kernel curve", SVC(kernel="rbf", C=3, gamma=1.2), False),
        ("SVM (RBF) proba: soft", SVC(kernel="rbf", C=3, gamma=1.2,
                                       probability=True, random_state=SEED), True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    for ax, (name, clf, proba) in zip(axes.ravel(), panels):
        clf.fit(X, y)
        if proba:
            Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
            cf = ax.contourf(xx, yy, Z, levels=12, cmap="coolwarm", alpha=0.85)
            ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
        else:
            Z = clf.predict(grid).reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.9)
            ax.contour(xx, yy, Z, levels=[0.5], colors=ORANGE, linewidths=2)
        _scatter(ax, X, y)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "cm_synthesis.pdf")


if __name__ == "__main__":
    log.info("generating classic-methods figures ...")
    knn_boundary()
    lda_qda()
    svm_margin()
    svm_soft_margin()
    hinge_loss()
    circles_lift()
    svm_kernels()
    gp_band()
    synthesis()
    log.info("done.")
