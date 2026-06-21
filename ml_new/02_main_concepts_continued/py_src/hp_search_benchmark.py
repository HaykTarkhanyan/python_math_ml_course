"""Visualize WHICH hyperparameters each search method tries, for the L01e deck.

Tunes 2 hyperparameters of an XGBoost regressor on the built-in sklearn
``diabetes`` toy dataset (442 rows -- no download, reproducible):

  * n_estimators  -- the iteration count (number of boosting rounds)
  * reg_lambda    -- L2 regularization strength (lambda)

learning_rate and max_depth are FIXED so the search is genuinely 2D and easy
to plot. Three methods get the SAME budget (= the grid's size) and the SAME
5-fold split (seed 509); the only thing that differs is WHICH points each one
samples in the (iter-count, lambda) plane:

  * Grid    -> a regular lattice
  * Random  -> a uniform scatter
  * Optuna  -> adaptive; coloured by trial order to show it concentrating

Produces one figure into ``ml_new/upcoming_lectures/fig/``:

  l01e_hp_search_patterns.pdf -- 3 panels (grid / random / Optuna).

Single-threaded everywhere (n_jobs=1) so the laptop stays cool and Optuna's
sequential nature is reflected honestly. Run with the project venv:
    ./ma/Scripts/python.exe ml_new/upcoming_lectures/py_src/hp_search_benchmark.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours.
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.stats import loguniform, randint
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from xgboost import XGBRegressor

SEED = 509
N_SPLITS = 5

# Fixed (not searched) -- keeps the search a clean 2D problem.
FIXED = dict(learning_rate=0.1, max_depth=3, objective="reg:squarederror",
             n_jobs=1, random_state=SEED, verbosity=0)

# Grid over the two searched HPs: 8 iter-counts x 7 lambdas = 56 configs.
N_ESTIMATORS_GRID = [50, 130, 210, 290, 370, 450, 530, 600]
REG_LAMBDA_GRID = list(np.logspace(-2, 2, 7))          # 1e-2 .. 1e2
BUDGET = len(N_ESTIMATORS_GRID) * len(REG_LAMBDA_GRID)  # 56; random/Optuna match it

# Search ranges for random / Optuna (cover the grid's span).
N_EST_LO, N_EST_HI = 50, 600
LAM_LO, LAM_HI = 1e-2, 1e2

# Armenian flag palette (per CLAUDE.md): grid=blue, random=red, Optuna=orange.
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"
METHOD_COLOR = {"Grid": ARM_BLUE, "Random": ARM_RED, "Optuna (TPE)": ARM_ORANGE}

HERE = Path(__file__).resolve()
UPCOMING_DIR = HERE.parents[1]         # ml_new/upcoming_lectures
REPO_ROOT = HERE.parents[3]            # repo root
FIG_DIR = UPCOMING_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l01e_hp_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "l01e_hp_benchmark.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def rmse(model, X, y) -> float:
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))


def run_grid(cv, Xtr, ytr, Xte, yte, logger):
    grid = {"n_estimators": N_ESTIMATORS_GRID, "reg_lambda": REG_LAMBDA_GRID}
    gs = GridSearchCV(XGBRegressor(**FIXED), grid, cv=cv,
                      scoring="neg_mean_squared_error", n_jobs=1)
    t0 = time.perf_counter()
    gs.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    pts = [(p["n_estimators"], p["reg_lambda"]) for p in gs.cv_results_["params"]]
    test = rmse(gs.best_estimator_, Xte, yte)
    logger.info(f"Grid:   test RMSE={test:.2f}  time={elapsed:.2f}s  "
                f"pts={len(pts)}  best={gs.best_params_}")
    return {"method": "Grid", "pts": pts, "best": gs.best_params_,
            "test_rmse": test, "time": elapsed}


def run_random(cv, Xtr, ytr, Xte, yte, logger):
    dist = {"n_estimators": randint(N_EST_LO, N_EST_HI + 1),
            "reg_lambda": loguniform(LAM_LO, LAM_HI)}
    rs = RandomizedSearchCV(XGBRegressor(**FIXED), dist, n_iter=BUDGET, cv=cv,
                            scoring="neg_mean_squared_error",
                            random_state=SEED, n_jobs=1)
    t0 = time.perf_counter()
    rs.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0
    pts = [(p["n_estimators"], p["reg_lambda"]) for p in rs.cv_results_["params"]]
    test = rmse(rs.best_estimator_, Xte, yte)
    logger.info(f"Random: test RMSE={test:.2f}  time={elapsed:.2f}s  "
                f"pts={len(pts)}  best={rs.best_params_}")
    return {"method": "Random", "pts": pts, "best": rs.best_params_,
            "test_rmse": test, "time": elapsed}


def run_optuna(cv, Xtr, ytr, Xte, yte, logger):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", N_EST_LO, N_EST_HI),
            "reg_lambda": trial.suggest_float("reg_lambda", LAM_LO, LAM_HI, log=True),
        }
        scores = cross_val_score(XGBRegressor(**FIXED, **params), Xtr, ytr, cv=cv,
                                 scoring="neg_mean_squared_error", n_jobs=1)
        return scores.mean()                      # Optuna maximizes

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=BUDGET, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    # ordered by trial number -> the SEQUENCE Optuna explored
    pts = [(t.params["n_estimators"], t.params["reg_lambda"]) for t in study.trials]
    order = [t.number for t in study.trials]
    best = XGBRegressor(**FIXED, **study.best_params).fit(Xtr, ytr)
    test = rmse(best, Xte, yte)
    logger.info(f"Optuna: test RMSE={test:.2f}  time={elapsed:.2f}s  "
                f"pts={len(pts)}  best={study.best_params}")
    return {"method": "Optuna (TPE)", "pts": pts, "order": order,
            "best": study.best_params, "test_rmse": test, "time": elapsed}


def _scatter_axes(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_yscale("log")
    ax.set_xlim(0, 650)
    ax.set_ylim(LAM_LO * 0.5, LAM_HI * 2)
    ax.set_xlabel("n_estimators  (iter count)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_patterns(results, logger):
    grid, rand, opt = results
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=True)

    # Grid -- regular lattice
    gx, gy = zip(*grid["pts"])
    axes[0].scatter(gx, gy, s=26, color=ARM_BLUE, alpha=0.85)
    _scatter_axes(axes[0], f"Grid - {len(grid['pts'])} pts (lattice)")
    axes[0].set_ylabel("reg_lambda  ($\\lambda$, log)", fontsize=9)
    axes[0].scatter([grid["best"]["n_estimators"]], [grid["best"]["reg_lambda"]],
                    marker="*", s=260, color="white", edgecolor="black", linewidth=1.2, zorder=5)

    # Random -- uniform scatter
    rx, ry = zip(*rand["pts"])
    axes[1].scatter(rx, ry, s=26, color=ARM_RED, alpha=0.85)
    _scatter_axes(axes[1], f"Random - {len(rand['pts'])} pts")
    axes[1].scatter([rand["best"]["n_estimators"]], [rand["best"]["reg_lambda"]],
                    marker="*", s=260, color="white", edgecolor="black", linewidth=1.2, zorder=5)

    # Optuna -- coloured by trial order (the sequence)
    ox, oy = zip(*opt["pts"])
    sc = axes[2].scatter(ox, oy, s=30, c=opt["order"], cmap="viridis", alpha=0.9)
    _scatter_axes(axes[2], f"Optuna - {len(opt['pts'])} trials")
    axes[2].scatter([opt["best"]["n_estimators"]], [opt["best"]["reg_lambda"]],
                    marker="*", s=260, color="white", edgecolor="black", linewidth=1.2, zorder=5)
    cb = fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label("trial order (early -> late)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = FIG_DIR / "l01e_hp_search_patterns.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out}")


def plot_bars(results, logger):
    """Score + cost summary -- one bar per method (companion to the patterns)."""
    methods = [r["method"] for r in results]
    colors = [METHOD_COLOR[m] for m in methods]
    rmses = [r["test_rmse"] for r in results]
    times = [r["time"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.7))

    b1 = ax1.bar(methods, rmses, color=colors, width=0.6)
    ax1.set_ylabel("held-out test RMSE")
    ax1.set_title("Predictive performance (lower is better)", fontsize=10)
    ax1.bar_label(b1, fmt="%.1f", padding=2, fontsize=9)
    ax1.set_ylim(0, max(rmses) * 1.18)

    b2 = ax2.bar(methods, times, color=colors, width=0.6)
    ax2.set_ylabel("search wall-clock (s)")
    ax2.set_title("Cost (single-threaded)", fontsize=10)
    ax2.bar_label(b2, fmt="%.1fs", padding=2, fontsize=9)
    ax2.set_ylim(0, max(times) * 1.18)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    out = FIG_DIR / "l01e_hp_benchmark_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out}")


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)

    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    logger.info(f"diabetes: train={Xtr.shape}  test={Xte.shape}  "
                f"2 HPs (n_estimators, reg_lambda), equal budget={BUDGET}, folds={N_SPLITS}")

    # Warm up XGBoost's one-time init (OpenMP thread pool, first-call overhead)
    # so it isn't charged to whichever search runs first -- makes the
    # wall-clock comparison fair (grid otherwise eats the cold start).
    XGBRegressor(**FIXED, n_estimators=50).fit(Xtr, ytr)

    results = [
        run_grid(cv, Xtr, ytr, Xte, yte, logger),
        run_random(cv, Xtr, ytr, Xte, yte, logger),
        run_optuna(cv, Xtr, ytr, Xte, yte, logger),
    ]
    plot_patterns(results, logger)
    plot_bars(results, logger)

    logger.info("=== SUMMARY for the slides ===")
    for r in results:
        logger.info(f"  {r['method']:>13}: test RMSE={r['test_rmse']:.2f}  "
                    f"time={r['time']:.2f}s  best={r['best']}")
    logger.info("done.")


if __name__ == "__main__":
    main()
