"""Generate the cold-open + U-curve figures for the L03 Regularization deck.

Produces 4 PDFs into ``ml/upcoming_lectures/fig/`` from a degree-20
polynomial Ridge fit on a noisy cubic (DGP adapted from the old Ch.1 notebook
``02_Regression_Main_Concepts.ipynb`` cell 218):

  1. l03_open_1_overfit.pdf   -- tiny lambda: the degree-20 poly chases noise.
  2. l03_open_2_sweet.pdf     -- sweet-spot lambda: the fit hugs the true cubic.
  3. l03_open_3_underfit.pdf  -- huge lambda: over-regularized, flat-ish.
  4. l03_mse_vs_lambda.pdf    -- train + test MSE vs lambda (the U-curve).

Features are standardized before Ridge (PolynomialFeatures -> StandardScaler
-> Ridge). Without scaling the x^20 column (up to 3^20 ~ 3.5e9) dwarfs x^1, and
the L2 penalty is applied unfairly across columns -- which is exactly the lesson
of the deck's "scale before regularizing" frame, so we practice what we preach.

Run with the project venv (see repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/upcoming_lectures/py_src/cold_open_ridge_demo.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 509
DEGREE = 20
# Few training points relative to the 20 poly features -> the unpenalised fit is
# genuinely in the high-variance regime, so the cold-open overfit is dramatic
# (not just a sliver above the sigma^2=400 noise floor).
N_POINTS = 50
TEST_SIZE = 0.4

# Armenian flag palette (per CLAUDE.md) -- blue=train, red=test, orange=fit.
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"

HERE = Path(__file__).resolve()
UPCOMING_DIR = HERE.parents[1]         # ml/upcoming_lectures
REPO_ROOT = HERE.parents[3]            # repo root
FIG_DIR = UPCOMING_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l03_cold_open")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "l03_cold_open.log")
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


def plot_fit(alpha, title, fname, data, xx, ylim, logger):
    Xtr, Xte, ytr, yte = data
    pipe = fit_pipe(alpha).fit(Xtr, ytr)
    tr_mse = mean_squared_error(ytr, pipe.predict(Xtr))
    te_mse = mean_squared_error(yte, pipe.predict(Xte))
    yy_fit = pipe.predict(xx.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(Xtr, ytr, s=18, color=ARM_BLUE, alpha=0.6, label="train")
    ax.scatter(Xte, yte, s=20, color=ARM_RED, alpha=0.75, label="test")
    ax.plot(xx, true_cubic(xx), "--", color="black", lw=2, label="true cubic")
    ax.plot(xx, yy_fit, "-", color=ARM_ORANGE, lw=3, label=f"degree-{DEGREE} fit")
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.text(0.03, 0.97, f"train MSE = {tr_mse:,.0f}\ntest MSE  = {te_mse:,.0f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"{fname}: alpha={alpha:g}  train MSE={tr_mse:.1f}  test MSE={te_mse:.1f}")
    return tr_mse, te_mse


def fig_mse_vs_lambda(data, logger):
    Xtr, Xte, ytr, yte = data
    alphas = np.logspace(-4, 5, 40)
    tr, te = [], []
    for a in alphas:
        pipe = fit_pipe(a).fit(Xtr, ytr)
        tr.append(mean_squared_error(ytr, pipe.predict(Xtr)))
        te.append(mean_squared_error(yte, pipe.predict(Xte)))
    best_i = int(np.argmin(te))
    best_alpha = alphas[best_i]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(alphas, tr, "-o", color=ARM_BLUE, lw=2, ms=3, label="train MSE")
    ax.plot(alphas, te, "-o", color=ARM_RED, lw=2, ms=3, label="test MSE")
    ax.axvline(best_alpha, color=ARM_ORANGE, ls="--", lw=1.8,
               label=f"best lambda = {best_alpha:.2g}")
    ax.set_xscale("log")
    ax.set_xlabel("regularization strength  $\\lambda$  (log scale)")
    ax.set_ylabel("MSE")
    ax.set_title("Train and test MSE vs regularization strength")
    ax.legend(loc="upper center", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "l03_mse_vs_lambda.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"l03_mse_vs_lambda.pdf: best alpha={best_alpha:.4g} "
                f"(test MSE={te[best_i]:.1f}, train MSE={tr[best_i]:.1f})")
    logger.info(f"  endpoints: alpha={alphas[0]:.4g} test MSE={te[0]:.1f}; "
                f"alpha={alphas[-1]:.4g} test MSE={te[-1]:.1f}")
    return best_alpha


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    data = make_data()
    Xtr, Xte, ytr, yte = data
    y_all = np.concatenate([ytr, yte])
    ylim = (y_all.min() - 40, y_all.max() + 40)
    xx = np.linspace(-3, 3, 400)

    best_alpha = fig_mse_vs_lambda(data, logger)

    plot_fit(1e-4, r"$\lambda \approx 0$  (no penalty)",
             "l03_open_1_overfit.pdf", data, xx, ylim, logger)
    plot_fit(best_alpha, fr"$\lambda = {best_alpha:.2g}$  (sweet spot)",
             "l03_open_2_sweet.pdf", data, xx, ylim, logger)
    plot_fit(1e4, r"$\lambda = 10{,}000$  (over-regularized)",
             "l03_open_3_underfit.pdf", data, xx, ylim, logger)

    logger.info("=== SUMMARY for the slides ===")
    logger.info(f"sweet-spot lambda = {best_alpha:.3g} (use this number on the slides)")
    logger.info("done.")


if __name__ == "__main__":
    main()
