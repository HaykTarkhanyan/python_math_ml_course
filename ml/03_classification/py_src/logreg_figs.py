"""Real figures for the L11 Classification + Logistic Regression deck.

Replaces hand-drawn TikZ with real fitted-model plots (house style: real
figures preferred where they add credibility). Writes PDFs into
ml/03_classification/fig/:

  logreg_boundary_2d.pdf     -- 2D probability surface + 0.5 decision boundary on
                                real, overlapping cheese data (aging temp x moisture).
  logreg_loss_convexity.pdf  -- squared-loss-on-sigmoid (non-convex) vs log-loss
                                (convex) as a single weight w varies.

Run with the project venv (see repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/03_classification/py_src/logreg_figs.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

SEED = 509
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]              # ml/03_classification
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"

CMAP_BR = LinearSegmentedColormap.from_list("blue_white_red", [ARM_BLUE, "white", ARM_RED])


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l11_logreg")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "logreg_figs.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def fig_boundary_2d(logger):
    """2D probability heatmap + 0.5 boundary on real, overlapping data.

    Two overlapping Gaussian clusters placed on the diagonal (good = lower-left,
    bad = upper-right) so BOTH features carry weight and the boundary is a clear
    diagonal -- and bad sits upper-right, matching the deck's text.
    """
    rng = np.random.RandomState(SEED)
    n = 160
    cov = np.array([[1.0, 0.35], [0.35, 1.0]])
    X0 = rng.multivariate_normal([-1.15, -1.15], cov, n)   # good, lower-left
    X1 = rng.multivariate_normal([1.15, 1.15], cov, n)     # bad, upper-right
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    clf = LogisticRegression().fit(X, y)

    pad = 0.6
    x0 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300)
    x1 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300)
    xx, yy = np.meshgrid(x0, x1)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    cf = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap=CMAP_BR,
                     vmin=0, vmax=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors=[ARM_ORANGE], linewidths=2.8)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=22, c=ARM_BLUE, marker="o",
               edgecolors="black", linewidths=0.5, label="good")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=26, c=ARM_RED, marker="^",
               edgecolors="black", linewidths=0.5, label="bad")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\hat p$(bad)")
    cb.set_ticks([0, 0.5, 1])
    ax.set_xlabel("aging temperature")
    ax.set_ylabel("moisture")
    ax.set_title("Probability surface + 0.5 boundary")
    ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor="white",
              framealpha=0.9)
    acc = clf.score(X, y)
    logger.info(f"boundary_2d: n={len(y)}, train acc={acc:.3f}, "
                f"coef={clf.coef_.ravel().round(2)}, intercept={clf.intercept_[0]:.2f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logreg_boundary_2d.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote logreg_boundary_2d.pdf")


def fig_loss_convexity(logger):
    """Squared loss on sigmoid is non-convex; log-loss is convex (1 weight w)."""
    # Clean classes at x ~ +-1, plus stubborn outliers (hot batches that are GOOD):
    x = np.array([-1.2, -1.0, -1.0, -0.8, 0.8, 1.0, 1.0, 1.2, 4.0, 5.0])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    ws = np.linspace(-2.0, 4.5, 500)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    mse, ll = [], []
    for w in ws:
        p = np.clip(sig(w * x), 1e-9, 1 - 1e-9)
        mse.append(np.mean((y - p) ** 2))
        ll.append(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    mse, ll = np.array(mse), np.array(ll)

    span = mse.max() - mse.min()
    loc_min = [i for i in range(1, len(ws) - 1)
               if mse[i] < mse[i - 1] and mse[i] <= mse[i + 1]]
    i_glob = int(np.argmin(mse))
    # the barrier = the local max between the shallow basin and the global one
    i_lo = loc_min[0] if loc_min else None
    i_bump = None
    if i_lo is not None and i_lo != i_glob:
        lo, hi = sorted((i_lo, i_glob))
        i_bump = lo + int(np.argmax(mse[lo:hi + 1]))
    logger.info(f"loss-convexity: MSE interior local minima at w="
                f"{[round(float(ws[i]), 2) for i in loc_min]}; global min w={ws[i_glob]:.2f}; "
                f"barrier w={None if i_bump is None else round(float(ws[i_bump]),2)}; "
                f"log-loss min at w={ws[int(np.argmin(ll))]:.2f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    a1.plot(ws, mse, color=ARM_RED, lw=2.6)
    a1.set_title("squared loss on sigmoid: non-convex")
    a2.plot(ws, ll, color=ARM_BLUE, lw=2.6)
    a2.set_title("log-loss: one convex bowl")
    if i_lo is not None:
        a1.scatter([ws[i_lo]], [mse[i_lo]], color=ARM_ORANGE, zorder=5, s=38)
        a1.annotate("shallow local min\n(GD can stall)", xy=(ws[i_lo], mse[i_lo]),
                    xytext=(ws[i_lo] - 0.1, mse[i_lo] + 0.22 * span),
                    fontsize=8, color=ARM_ORANGE, ha="center",
                    arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    if i_bump is not None:
        a1.annotate("barrier", xy=(ws[i_bump], mse[i_bump]),
                    xytext=(ws[i_bump] + 0.5, mse[i_bump] + 0.12 * span),
                    fontsize=8, color="0.35",
                    arrowprops=dict(arrowstyle="->", color="0.35"))
    a1.scatter([ws[i_glob]], [mse[i_glob]], color=ARM_RED, zorder=5, s=38)
    a1.annotate("global min", xy=(ws[i_glob], mse[i_glob]),
                xytext=(ws[i_glob] - 2.0, mse[i_glob] + 0.30 * span),
                fontsize=8, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    j = int(np.argmin(ll))
    a2.scatter([ws[j]], [ll[j]], color=ARM_BLUE, zorder=5, s=38)
    a2.annotate("one minimum\n(GD always finds it)", xy=(ws[j], ll[j]),
                xytext=(ws[j] + 0.3, ll[j] + 0.32 * (ll.max() - ll.min())),
                fontsize=8, color=ARM_BLUE,
                arrowprops=dict(arrowstyle="->", color=ARM_BLUE))
    for a in (a1, a2):
        a.set_xlabel("weight $w$")
        a.spines[["top", "right"]].set_visible(False)
    a1.set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logreg_loss_convexity.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote logreg_loss_convexity.pdf")


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig_boundary_2d(logger)
    fig_loss_convexity(logger)
    logger.info("done.")


if __name__ == "__main__":
    main()
