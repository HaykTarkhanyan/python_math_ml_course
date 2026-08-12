"""Generate figures for the classic-methods deck (SVM-centered + KNN/NB/LDA-QDA/GP).

Real matplotlib on tiny 2D/1D toys, house palette (Armenian-flag colors), seed 509.
Outputs PDFs to ../fig/ ; logs to ./logs/. Run with the ma venv:
    ./ma/Scripts/python.exe ml/07_classic_methods/py_src/make_classic_figures.py
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


# ------------------------------------------------ why "scale first" is not advice
def scaling_matters():
    """One useless column with big numbers destroys KNN. The same data, scaled, recovers it.

    Feature 1 carries all the signal; feature 2 is pure noise on a 100x larger range.
    Accuracies are measured on a held-out split, not asserted.
    """
    rng = np.random.RandomState(SEED)
    n = 400
    y = rng.randint(0, 2, n)
    f1 = rng.normal(0, 1, n) + np.where(y == 1, 1.6, -1.6)   # informative
    f2 = rng.normal(0, 1, n) * 100.0                          # pure noise, huge range
    X = np.c_[f1, f2]
    tr, te = np.arange(n) % 4 != 0, np.arange(n) % 4 == 0     # 75/25, deterministic

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    accs = {}
    for ax, scaled in zip(axes, [False, True]):
        Xw = X.copy()
        if scaled:
            Xw = (Xw - Xw[tr].mean(0)) / Xw[tr].std(0)        # statistics from train only
        clf = KNeighborsClassifier(n_neighbors=15).fit(Xw[tr], y[tr])
        acc = clf.score(Xw[te], y[te])
        accs["scaled" if scaled else "raw"] = acc
        step = (Xw[:, 0].max() - Xw[:, 0].min() + Xw[:, 1].max() - Xw[:, 1].min()) / 300.0
        xx, yy = _mesh(Xw, h=step, pad=0.4)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.9)
        ax.contour(xx, yy, Z, levels=[0.5], colors=ORANGE, linewidths=2)
        _scatter(ax, Xw, y)
        head = "standardized" if scaled else "raw units"
        ax.set_title(f"{head}  --  test accuracy {acc:.0%}", fontsize=12)
        ax.set_xlabel("feature 1  (the signal)")
        ax.set_ylabel("feature 2  (pure noise)")
        ax.set_xticks([]); ax.set_yticks([])
    log.info(f"scaling_matters: raw={accs['raw']:.3f} scaled={accs['scaled']:.3f}")
    save(fig, "cm_scaling_matters.pdf")


# ---------------------------------------------------------- the RBF width knob
def svm_gamma():
    """gamma sets how wide each support vector's bump is: underfit -> good -> islands."""
    X, y = make_moons(n_samples=260, noise=0.26, random_state=SEED)
    X = (X - X.mean(0)) / X.std(0)
    tr, te = np.arange(len(X)) % 4 != 0, np.arange(len(X)) % 4 == 0
    xx, yy = _mesh(X, pad=0.6)

    gammas = [0.05, 1.0, 60.0]
    labels = ["too small: almost linear", "about right", "too large: islands"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for ax, g, lab in zip(axes, gammas, labels):
        clf = SVC(kernel="rbf", C=10, gamma=g).fit(X[tr], y[tr])
        tr_acc, te_acc = clf.score(X[tr], y[tr]), clf.score(X[te], y[te])
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.9)
        ax.contour(xx, yy, Z, levels=[0.5], colors=ORANGE, linewidths=2)
        _scatter(ax, X, y)
        ax.set_title(rf"$\gamma$ = {g}  --  {lab}", fontsize=11)
        ax.set_xlabel(f"train {tr_acc:.0%}    test {te_acc:.0%}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        log.info(f"svm_gamma: gamma={g} train={tr_acc:.3f} test={te_acc:.3f}")
    save(fig, "cm_svm_gamma.pdf")


# -------------------------------------------- LDA: the generative story in 1-D
def lda_story():
    """Three panels: raw data -> one scaled Gaussian per class -> the taller one wins.

    Everything (means, pooled sd, priors, boundary) is estimated from the sample --
    nothing is hand-placed, so the crossing point is a real computed quantity.
    """
    rng = np.random.RandomState(SEED)
    n0, n1 = 60, 40                       # unequal on purpose: the prior will show
    x0 = rng.normal(-1.4, 1.0, n0)
    x1 = rng.normal(1.6, 1.0, n1)
    mu0, mu1 = x0.mean(), x1.mean()
    # pooled sd = LDA's "shared Sigma" assumption, in one dimension
    sd = np.sqrt(((n0 - 1) * x0.var(ddof=1) + (n1 - 1) * x1.var(ddof=1))
                 / (n0 + n1 - 2))
    pi0, pi1 = n0 / (n0 + n1), n1 / (n0 + n1)

    grid = np.linspace(-5.5, 5.5, 700)

    def scaled(pi, mu):
        """pi_k * p(x | y=k) -- the quantity LDA actually compares."""
        return pi * np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    f0, f1 = scaled(pi0, mu0), scaled(pi1, mu1)
    # equal variances => exactly one crossing, available in closed form
    boundary = 0.5 * (mu0 + mu1) + (sd ** 2 / (mu1 - mu0)) * np.log(pi0 / pi1)
    top = max(f0.max(), f1.max())

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    # (1) the raw data
    ax = axes[0]
    ax.scatter(x0, rng.uniform(-0.10, 0.10, n0), c=RED, s=20, alpha=0.85,
               edgecolor="white", linewidth=0.4)
    ax.scatter(x1, rng.uniform(-0.10, 0.10, n1), c=BLUE, s=20, alpha=0.85,
               edgecolor="white", linewidth=0.4)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_title("1. One feature, two classes")
    ax.text(0.5, -0.42, "they overlap -- no clean cut", ha="center", fontsize=10,
            color=GREY, transform=ax.transData)

    # (2) a Gaussian per class, each scaled by its prior, evaluated at a query point
    ax = axes[1]
    ax.plot(grid, f0, color=RED, lw=2.2)
    ax.plot(grid, f1, color=BLUE, lw=2.2)
    xq = 0.0
    h0 = float(pi0 * np.exp(-0.5 * ((xq - mu0) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)))
    h1 = float(pi1 * np.exp(-0.5 * ((xq - mu1) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)))
    ax.vlines(xq, 0, max(h0, h1), color=GREY, ls=":", lw=1.6)
    ax.scatter([xq, xq], [h0, h1], c=[RED, BLUE], s=55, zorder=5,
               edgecolor="white", linewidth=1.0)
    ax.annotate(f"{h0:.3f}", (xq, h0), textcoords="offset points", xytext=(8, 2),
                color=RED, fontsize=10, fontweight="bold")
    ax.annotate(f"{h1:.3f}", (xq, h1), textcoords="offset points", xytext=(8, -4),
                color=BLUE, fontsize=10, fontweight="bold")
    ax.set_title(r"2. Score each class: $\pi_k \, p(x \mid y{=}k)$")
    ax.set_ylim(0, top * 1.25)
    ax.set_yticks([])

    # (3) whoever is taller wins -> the boundary is where they cross
    ax = axes[2]
    ax.plot(grid, f0, color=RED, lw=2.2)
    ax.plot(grid, f1, color=BLUE, lw=2.2)
    ax.fill_between(grid, 0, top * 1.25, where=f0 >= f1, color="#F3CDD0", alpha=0.75)
    ax.fill_between(grid, 0, top * 1.25, where=f1 > f0, color="#CBD6EC", alpha=0.75)
    ax.axvline(boundary, color=ORANGE, lw=2.2)
    ax.annotate(f"boundary\nx = {boundary:.2f}", (boundary, top * 1.10),
                textcoords="offset points", xytext=(8, 0), color=ORANGE,
                fontsize=10, fontweight="bold")
    ax.set_title("3. Predict the taller curve")
    ax.set_ylim(0, top * 1.25)
    ax.set_yticks([])

    for ax in axes:
        ax.set_xlim(-5.5, 5.5)
        ax.set_xlabel("x")
    save(fig, "cm_lda_story.pdf")


# --------------------------------------- generative = you can sample new data
def generative_sampling():
    """Fit a Gaussian per class, then draw brand-new points from the fitted model.

    Means and covariances are estimated from the real sample; the right panel is
    drawn from those estimates, so not one of its points appeared in training.
    """
    rng = np.random.RandomState(SEED)
    n = 240
    X0 = rng.multivariate_normal([-1.7, 0.2], [[0.62, 0.28], [0.28, 0.50]], n)
    X1 = rng.multivariate_normal([1.5, 0.6], [[1.60, -0.85], [-0.85, 1.05]], n)

    # this IS the model: one mean + one covariance per class, estimated from data
    m0, C0 = X0.mean(0), np.cov(X0.T)
    m1, C1 = X1.mean(0), np.cov(X1.T)
    S0 = rng.multivariate_normal(m0, C0, n)     # never-seen points
    S1 = rng.multivariate_normal(m1, C1, n)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    panels = [("real training data", X0, X1), ("sampled from the fitted model", S0, S1)]
    for ax, (title, A, B) in zip(axes, panels):
        ax.scatter(A[:, 0], A[:, 1], c=RED, s=18, alpha=0.75,
                   edgecolor="white", linewidth=0.4)
        ax.scatter(B[:, 0], B[:, 1], c=BLUE, s=18, alpha=0.75,
                   edgecolor="white", linewidth=0.4)
        for mean, cov, col in [(m0, C0, RED), (m1, C1, BLUE)]:
            _cov_ellipse(ax, mean, cov, col)
        ax.set_title(title)
        ax.set_xlim(-4.6, 5.2); ax.set_ylim(-3.2, 3.6)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].text(0.5, -0.08, "not one of these points was in the training set",
                 ha="center", fontsize=10, color=GREY, transform=axes[1].transAxes)
    save(fig, "cm_generative_sampling.pdf")


# ------------------------------------------- what Sigma^{-1} actually measures
def mahalanobis():
    """Same Euclidean distance, different Mahalanobis distance -- the whole idea."""
    rng = np.random.RandomState(SEED)
    cov = np.array([[4.0, 0.0], [0.0, 1.0]])      # four times wider than tall
    pts = rng.multivariate_normal([0.0, 0.0], cov, 260)
    A, B = np.array([2.0, 0.0]), np.array([0.0, 2.0])
    cinv = np.linalg.inv(cov)
    d2A, d2B = float(A @ cinv @ A), float(B @ cinv @ B)
    # these are the exact numbers quoted on the slide; fail loudly if they drift
    assert np.isclose(d2A, 1.0) and np.isclose(d2B, 4.0), (d2A, d2B)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for ax, kind in zip(axes, ["euclid", "maha"]):
        ax.scatter(pts[:, 0], pts[:, 1], c=GREY, s=12, alpha=0.35, zorder=1)
        if kind == "euclid":
            circ = plt.Circle((0, 0), 2.0, fill=False, edgecolor=GREY, ls="--",
                              lw=2.0, zorder=4)
            ax.add_patch(circ)
            ax.set_title("Euclidean: both are 2 away")
            note = "one ruler for every direction"
        else:
            for c, col in [(1.0, ORANGE), (2.0, POP)]:
                _cov_ellipse(ax, [0, 0], cov, col, nstd=c)
            ax.set_title(r"Mahalanobis: the class's own ruler")
            note = "wide axis = cheap, narrow axis = expensive"
        # A sits at the right edge, so its label goes below the point, not beside it
        offsets = {"A": (10, 8), "B": (10, 8)} if kind == "euclid" \
            else {"A": (-16, -30), "B": (8, 10)}
        for pt, lab, col in [(A, "A", RED), (B, "B", BLUE)]:
            ax.scatter(*pt, c=col, s=110, marker="X", edgecolor="white",
                       linewidth=1.3, zorder=6)
            d2 = d2A if lab == "A" else d2B
            txt = lab if kind == "euclid" else f"{lab}:  $d^2={d2:.0f}$"
            ax.annotate(txt, pt, textcoords="offset points", xytext=offsets[lab],
                        color=col, fontsize=11, fontweight="bold", zorder=7)
        ax.scatter(0, 0, c="black", s=40, marker="+", zorder=6)
        ax.set_xlim(-6.6, 6.6); ax.set_ylim(-3.4, 3.4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.09, note, ha="center", fontsize=10, color=GREY,
                transform=ax.transAxes)
    save(fig, "cm_mahalanobis.pdf")


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
    scaling_matters()
    svm_gamma()
    lda_story()
    generative_sampling()
    mahalanobis()
    lda_qda()
    svm_margin()
    svm_soft_margin()
    hinge_loss()
    circles_lift()
    svm_kernels()
    gp_band()
    synthesis()
    log.info("done.")
