"""Extra figures for the L03 Regularization deck (2026-06-20 revision pass).

Produces 3 PDFs into ``ml_new/upcoming_lectures/fig/``:

  1. l03_coef_norms.pdf    -- two fits on the SAME data: an unrestricted model
                              (huge ||theta||, wiggly) vs a regularized one
                              (small ||theta||, smooth). Big coefs = complex.
  2. l03_priors.pdf        -- Normal vs Laplace prior densities. Penalty = prior:
                              ridge <-> Normal, lasso <-> Laplace (sharp peak -> sparsity).
  3. l03_early_stopping.pdf-- linear regression trained by gradient descent;
                              train + validation MSE vs iteration, with the
                              early-stopping point (val minimum) marked.

Run with the project venv (see repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml_new/upcoming_lectures/py_src/extra_figures.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 509
DEGREE = 20
N_POINTS = 50
TEST_SIZE = 0.4

ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"

HERE = Path(__file__).resolve()
UPCOMING_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = UPCOMING_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l03_extra")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "l03_extra.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def true_cubic(x):
    return 1 + 2 * x + 3 * x ** 2 + 4 * x ** 3


def make_data():
    np.random.seed(SEED)
    X = np.linspace(-3, 3, N_POINTS).reshape(-1, 1)
    y = true_cubic(X.ravel()) + np.random.normal(0, 20, size=N_POINTS)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


def fit_pipe(alpha):
    return make_pipeline(
        PolynomialFeatures(DEGREE, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def coef_norm(pipe):
    return float(np.linalg.norm(pipe.named_steps["ridge"].coef_))


def fig_coef_norms(logger):
    Xtr, Xte, ytr, yte = make_data()
    big = fit_pipe(1e-6).fit(Xtr, ytr)      # unrestricted -> chases noise
    small = fit_pipe(10.0).fit(Xtr, ytr)    # regularized  -> smooth
    n_big, n_small = coef_norm(big), coef_norm(small)
    xx = np.linspace(-3, 3, 400)
    ylo, yhi = ytr.min() - 50, ytr.max() + 90

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(Xtr, ytr, s=22, color="0.55", alpha=0.75, label="training data")
    ax.plot(xx, true_cubic(xx), "--", color="black", lw=2, label="true cubic")
    ax.plot(xx, big.predict(xx.reshape(-1, 1)), color=ARM_RED, lw=2.5,
            label=fr"unrestricted:  $\|\theta\|_2 = {n_big:,.0f}$")
    ax.plot(xx, small.predict(xx.reshape(-1, 1)), color=ARM_BLUE, lw=2.5,
            label=fr"regularized:  $\|\theta\|_2 = {n_small:,.0f}$")
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Same data, same model class --- different coefficient size")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "l03_coef_norms.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    cb, cs = big.named_steps["ridge"], small.named_steps["ridge"]
    np.set_printoptions(suppress=True, linewidth=240)
    logger.info(f"{out.name}: ||theta|| unrestricted={n_big:.1f}, regularized={n_small:.1f}")
    logger.info(f"  unrestricted: b0={cb.intercept_:.2f}, coef(full)={np.round(cb.coef_, 1)}")
    logger.info(f"  regularized:  b0={cs.intercept_:.2f}, coef(full)={np.round(cs.coef_, 2)}")


def fig_priors(logger):
    x = np.linspace(-4, 4, 800)
    sigma = 1.0
    b = sigma / np.sqrt(2)                   # equal variance (Var_Laplace = 2 b^2)
    normal = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    laplace = np.exp(-np.abs(x) / b) / (2 * b)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.axvline(0, color="0.85", lw=1, zorder=0)
    ax.plot(x, normal, color=ARM_BLUE, lw=2.5, label=r"Normal prior (ridge, $L_2$)")
    ax.plot(x, laplace, color=ARM_RED, lw=2.5, label=r"Laplace prior (lasso, $L_1$)")
    ax.annotate("sharp peak at 0\n$\\Rightarrow$ pushes weights to exactly 0",
                xy=(0, laplace.max()), xytext=(0.7, 0.62),
                fontsize=10, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    ax.set_xlabel(r"weight value $\theta_j$")
    ax.set_ylabel("prior density")
    ax.set_title(r"A penalty is a prior: ridge $\leftrightarrow$ Normal, lasso $\leftrightarrow$ Laplace")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "l03_priors.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"{out.name}: sigma={sigma}, b={b:.3f} (equal variance)")


def fig_early_stopping(logger):
    Xtr, Xva, ytr, yva = make_data()         # use the held-out split as validation
    poly = PolynomialFeatures(DEGREE, include_bias=False)
    sc = StandardScaler()
    Xtr_ = sc.fit_transform(poly.fit_transform(Xtr))
    Xva_ = sc.transform(poly.transform(Xva))
    ymean = ytr.mean()
    ytr_c = ytr - ymean
    n, p = Xtr_.shape

    w = np.zeros(p)
    lr, iters = 0.02, 15000
    its, tr, va = [], [], []
    for t in range(iters):
        grad = (2.0 / n) * Xtr_.T @ (Xtr_ @ w - ytr_c)
        w -= lr * grad
        if t % 25 == 0 or t == iters - 1:
            tr.append(np.mean((Xtr_ @ w + ymean - ytr) ** 2))
            va.append(np.mean((Xva_ @ w + ymean - yva) ** 2))
            its.append(t + 1)
    its, tr, va = np.array(its), np.array(tr), np.array(va)
    best = int(np.argmin(va))
    best_it = its[best]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(its, tr, color=ARM_BLUE, lw=2, label="train MSE")
    ax.plot(its, va, color=ARM_RED, lw=2, label="validation MSE")
    ax.axvline(best_it, color=ARM_ORANGE, ls="--", lw=1.8)
    ax.annotate(f"stop here\n(iter {best_it})", xy=(best_it, va[best]),
                xytext=(best_it * 3, va[best] + 0.18 * (va.max() - va.min())),
                color=ARM_ORANGE, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xscale("log")
    ax.set_xlabel("gradient-descent iteration (log scale)")
    ax.set_ylabel("MSE")
    ax.set_title("Early stopping: the validation curve has a minimum")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "l03_early_stopping.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"{out.name}: early stop at iter {best_it}, val MSE={va[best]:.1f}; "
                f"final val MSE={va[-1]:.1f}, final train MSE={tr[-1]:.1f}")


def fig_ridge_name(logger):
    """Why 'ridge': collinear features make the RSS a long valley; +lambda*I rounds it."""
    np.random.seed(SEED)
    n = 60
    x1 = np.random.normal(0, 1, n)
    x2 = x1 + np.random.normal(0, 0.08, n)        # corr ~ 0.99 (near-collinear)
    X = np.column_stack([x1, x2])
    y = X @ np.array([1.5, 1.5]) + np.random.normal(0, 0.5, n)

    A = X.T @ X
    b = X.T @ y
    c = float(y @ y)
    evals = np.linalg.eigvalsh(A)
    lam = float(0.30 * evals.max())               # rounds the narrow direction
    A_ridge = A + lam * np.eye(2)
    ols = np.linalg.solve(A, b)
    rdg = np.linalg.solve(A_ridge, b)
    logger.info(f"ridge-name: eigvals(A)={evals}, cond={evals.max()/evals.min():.0f}, "
                f"lambda={lam:.2f}, OLS={ols}, ridge={rdg}")

    g = np.linspace(-1.5, 4.5, 320)
    T1, T2 = np.meshgrid(g, g)

    def rss(M, bb):
        return (M[0, 0] * T1 ** 2 + 2 * M[0, 1] * T1 * T2 + M[1, 1] * T2 ** 2
                - 2 * (bb[0] * T1 + bb[1] * T2) + c)

    Z_ols = rss(A, b)
    Z_rdg = rss(A_ridge, b)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax, Z, pt, title, note, ptcol in [
        (axes[0], Z_ols, ols, "Collinear features: a least-squares ``ridge''",
         "long flat valley:\n$\\theta$ barely determined", ARM_RED),
        (axes[1], Z_rdg, rdg, r"Add $\lambda$ to the diagonal of $X^\top X$",
         "rounded bowl:\nunique, stable min", ARM_ORANGE),
    ]:
        levels = Z.min() + np.geomspace(1.0, 600.0, 11)
        ax.contour(T1, T2, Z, levels=levels, colors=ARM_BLUE, linewidths=0.8)
        ax.plot(*pt, "o", color=ptcol, ms=9)
        ax.plot(0, 0, "+", color="0.4", ms=10)
        ax.annotate(note, xy=(2.2, 2.2), xytext=(-1.2, 3.6), fontsize=9, color=ptcol)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_aspect("equal")
    fig.tight_layout()
    out = FIG_DIR / "l03_ridge_name.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_l1_l2_geometry(logger):
    """L1-diamond vs L2-ball with elongated, tilted RSS ellipses (ESL-style).

    The lasso solution lands on a corner (theta_1 = 0, sparse); ridge does not.
    """
    bhat = np.array([0.4, 2.3])                # OLS estimate (placed so lasso hits the corner)
    rad = 1.1
    Q = np.array([[1.0, 0.2], [0.2, 0.55]])    # tilted, elongated loss curvature

    def qf(P):                                  # quadratic loss at array of points P (...,2)
        d = P - bhat
        return np.einsum("...i,ij,...j->...", d, Q, d)

    g = np.linspace(-1.9, 2.9, 440)
    G1, G2 = np.meshgrid(g, g)
    loss = qf(np.stack([G1, G2], axis=-1))

    ang = np.linspace(0, 2 * np.pi, 4000)
    circ = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    l2 = circ[np.argmin(qf(circ))]

    corners = np.array([[rad, 0], [0, rad], [-rad, 0], [0, -rad], [rad, 0]])
    seg = np.linspace(0, 1, 1000)[:, None]
    dia = np.vstack([corners[k] + seg * (corners[k + 1] - corners[k]) for k in range(4)])
    l1 = dia[np.argmin(qf(dia))]
    logger.info(f"l1_l2_geometry: L2 sol={np.round(l2,3)}, L1 sol={np.round(l1,3)}")

    levels = np.unique(np.concatenate([np.geomspace(0.18, 9.0, 6),
                                       [float(qf(l1[None])[0]), float(qf(l2[None])[0])]]))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6))
    for ax, kind in zip(axes, ["L2", "L1"]):
        ax.contour(G1, G2, loss, levels=levels, colors=ARM_BLUE, linewidths=0.8, alpha=0.65)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.plot(*bhat, "x", color="black", ms=10, mew=2)
        ax.annotate(r"$\hat\theta_{\mathrm{OLS}}$", bhat, textcoords="offset points",
                    xytext=(7, 2), fontsize=11)
        if kind == "L2":
            ax.add_patch(Circle((0, 0), rad, fill=True, color=ARM_BLUE, alpha=0.10,
                                ec=ARM_BLUE, lw=2))
            ax.plot(*l2, "o", color=ARM_ORANGE, ms=11)
            ax.annotate(r"both $\theta_j \neq 0$", l2, textcoords="offset points",
                        xytext=(10, -6), color=ARM_ORANGE, fontsize=10)
            ax.set_title(r"Ridge ($L_2$): round ball $\Rightarrow$ no zeros")
        else:
            ax.add_patch(Polygon([[rad, 0], [0, rad], [-rad, 0], [0, -rad]], closed=True,
                                 fill=True, color=ARM_RED, alpha=0.10, ec=ARM_RED, lw=2))
            ax.plot(*l1, "o", color=ARM_ORANGE, ms=11)
            ax.annotate(r"$\theta_1 = 0$ (sparse!)", l1, textcoords="offset points",
                        xytext=(12, -2), color=ARM_ORANGE, fontsize=10)
            ax.set_title(r"Lasso ($L_1$): corner $\Rightarrow$ $\theta_1 = 0$")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_aspect("equal")
        ax.set_xlim(-1.9, 2.7)
        ax.set_ylim(-1.5, 2.95)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "l03_l1_l2_geometry.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig_coef_norms(logger)
    fig_priors(logger)
    fig_early_stopping(logger)
    fig_ridge_name(logger)
    fig_l1_l2_geometry(logger)
    logger.info("done.")


if __name__ == "__main__":
    main()
