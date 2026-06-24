"""Figures for the Advanced Boosting (L12) deck.

Real figures from one synthetic make_classification dataset (seed 509), fit once:
  - 12_xgb_histogram_split.pdf : percentile bin edges as candidate splits
  - 12_xgb_earlystop.pdf       : train/val logloss vs rounds, early-stop point
  - 12_xgb_importance.pdf      : gain-based feature importance bar chart

Run with the ma venv:  ./ma/Scripts/python.exe ml_new/ch3_trees/py_src/make_xgb_figures.py
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/make_xgb_figures.log")],
)
log = logging.getLogger(__name__)

SEED = 509
np.random.seed(SEED)
FIG = Path(__file__).resolve().parents[1] / "fig"
FIG.mkdir(exist_ok=True)
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

# one synthetic dataset, reused everywhere
X, y = make_classification(
    n_samples=4000, n_features=12, n_informative=6, n_redundant=2,
    n_clusters_per_class=2, class_sep=1.0, random_state=SEED,
)
FEAT = [f"f{i}" for i in range(X.shape[1])]
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
log.info("data: X_tr=%s X_val=%s", X_tr.shape, X_val.shape)


def fig_histogram_split():
    x = X_tr[:, 0]
    edges = np.quantile(x, np.linspace(0, 1, 11))   # 10 percentile bins -> 11 edges
    chosen = edges[6]                               # pretend this edge maximises the gain
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=45, color="0.82", edgecolor="white")
    for e in edges:
        ax.axvline(e, color=ARM_BLUE, lw=1.0, alpha=0.7)
    ax.axvline(chosen, color=ARM_RED, lw=2.6)
    ax.set(title="Approximate split finding: candidates = percentile bin edges",
           xlabel="feature value", ylabel="count")
    ax.legend(handles=[
        Line2D([0], [0], color=ARM_BLUE, lw=1.0, label="percentile edges (candidate splits)"),
        Line2D([0], [0], color=ARM_RED, lw=2.6, label="chosen split (max gain)"),
    ], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "12_xgb_histogram_split.pdf")
    plt.close(fig)
    log.info("wrote 12_xgb_histogram_split.pdf")


def fig_earlystop():
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = dict(objective="binary:logistic", eval_metric="logloss",
                  eta=0.3, max_depth=5, seed=SEED)
    res = {}
    booster = xgb.train(params, dtr, num_boost_round=400,
                        evals=[(dtr, "train"), (dval, "val")],
                        early_stopping_rounds=20, evals_result=res, verbose_eval=False)
    tr, va = res["train"]["logloss"], res["val"]["logloss"]
    best = booster.best_iteration
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tr, color=ARM_BLUE, lw=2, label="train logloss")
    ax.plot(va, color=ARM_RED, lw=2, label="validation logloss")
    ax.axvline(best, color=ARM_ORANGE, ls="--", lw=2, label=f"early stop @ round {best}")
    ax.set(title="Early stopping: validation logloss stops improving",
           xlabel="boosting round", ylabel="logloss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "12_xgb_earlystop.pdf")
    plt.close(fig)
    log.info("wrote 12_xgb_earlystop.pdf (best_iteration=%d, val rounds=%d)", best, len(va))


def fig_importance():
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.3,
                            importance_type="gain", random_state=SEED, eval_metric="logloss")
    clf.fit(X_tr, y_tr)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar([FEAT[i] for i in order], imp[order], color=ARM_BLUE)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set(title="Gain-based feature importance (XGBoost)",
           xlabel="feature", ylabel="importance (gain, normalized)")
    ax.margins(y=0.15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "12_xgb_importance.pdf")
    plt.close(fig)
    log.info("wrote 12_xgb_importance.pdf")


if __name__ == "__main__":
    fig_histogram_split()
    fig_earlystop()
    fig_importance()
    log.info("done")
