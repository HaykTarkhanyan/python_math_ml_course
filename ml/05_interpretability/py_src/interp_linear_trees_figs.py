"""Figures + worked numbers for Interpretability Deck 1 (linear models & trees).

Downloads the UCI Bike Sharing day.csv (cached), then:
  - imp_methods_disagree.pdf : standardized linear |coef| vs RF impurity importance (rankings differ).
  - tree_bike.pdf            : a shallow regression tree (readable splits) via plot_tree.
  - logs WORKED NUMBERS for the deck:
      * variance reduction of the fitted tree's ROOT split (real bike numbers),
      * information gain on a tiny classification toy (exact).

Run: ./ma/Scripts/python.exe ml/05_interpretability/py_src/interp_linear_trees_figs.py
Conventions: logging to console + logs/, seed 509, f-strings, Armenian-flag colours.
"""
import io
import logging
import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
DATA_DIR = CH_DIR / "data"
LOGS_DIR = REPO_ROOT / "logs"
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed"]


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("deck1"); log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "interp_deck1.log")):
        h.setFormatter(fmt); log.addHandler(h)
    return log


def get_bike(log):
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "bike-day.csv"
    if not out.exists():
        log.info(f"downloading {URL}")
        z = zipfile.ZipFile(io.BytesIO(urlopen(URL, timeout=60).read()))
        log.info(f"zip members: {z.namelist()}")
        out.write_bytes(z.read("day.csv"))
        log.info(f"wrote {out} ({out.stat().st_size} bytes)")
    df = pd.read_csv(out)
    log.info(f"bike day.csv: shape={df.shape}")
    return df


def fig_methods_disagree(df, log):
    X, y = df[FEATURES].values, df["cnt"].values
    Xs = StandardScaler().fit_transform(X)
    lin = LinearRegression().fit(Xs, y)
    lin_imp = np.abs(lin.coef_); lin_imp = lin_imp / lin_imp.sum()
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X, y)
    rf_imp = rf.feature_importances_

    lin_s = pd.Series(lin_imp, index=FEATURES).sort_values()
    rf_s = pd.Series(rf_imp, index=FEATURES).sort_values()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.6))
    a1.barh(range(len(lin_s)), lin_s.values, color=ARM_BLUE)
    a1.set_yticks(range(len(lin_s)), lin_s.index, fontsize=8)
    a1.set_title("Linear model: |standardized coef|", fontsize=10)
    a2.barh(range(len(rf_s)), rf_s.values, color=ARM_RED)
    a2.set_yticks(range(len(rf_s)), rf_s.index, fontsize=8)
    a2.set_title("Random forest: impurity importance", fontsize=10)
    for a in (a1, a2):
        a.set_xlabel("relative importance", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same data, different rankings", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "imp_methods_disagree.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"linear top3: {list(lin_s.tail(3).index[::-1])}")
    log.info(f"rf top3:     {list(rf_s.tail(3).index[::-1])}")
    log.info("wrote imp_methods_disagree.pdf")


def fig_tree(df, log):
    from sklearn.tree import export_text
    X, y = df[FEATURES].values, df["cnt"].values
    # depth 2: 4 leaves, big font -> actually readable on a projector (was depth 3, too cramped).
    tree = DecisionTreeRegressor(max_depth=2, random_state=SEED).fit(X, y)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    plot_tree(tree, feature_names=FEATURES, filled=True, rounded=True,
              impurity=False, proportion=True, fontsize=12, precision=2, ax=ax)
    ax.set_title("A shallow regression tree on bike rentals (read a path as a rule)", fontsize=13)
    fig.savefig(FIG_DIR / "tree_bike.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("TREE STRUCTURE (depth 2):\n" + export_text(tree, feature_names=FEATURES, decimals=0))

    # --- worked numbers: variance reduction of the ROOT split ---
    f_idx = tree.tree_.feature[0]
    thr = tree.tree_.threshold[0]
    feat = FEATURES[f_idx]
    parent_var = np.var(y)
    left = y[X[:, f_idx] <= thr]; right = y[X[:, f_idx] > thr]
    wl, wr = len(left) / len(y), len(right) / len(y)
    child_var = wl * np.var(left) + wr * np.var(right)
    log.info(f"ROOT SPLIT worked numbers: feature='{feat}' <= {thr:.3f}")
    log.info(f"  parent var(cnt) = {parent_var:,.0f}; n={len(y)}")
    log.info(f"  left  n={len(left)} ({wl:.2f}) var={np.var(left):,.0f}; "
             f"right n={len(right)} ({wr:.2f}) var={np.var(right):,.0f}")
    log.info(f"  weighted child var = {child_var:,.0f}")
    log.info(f"  variance reduction = {parent_var - child_var:,.0f}")
    log.info("wrote tree_bike.pdf")


def worked_info_gain(log):
    """Tiny classification toy: 10 samples, exact information gain (entropy)."""
    def H(c1, c0):
        n = c1 + c0
        p = np.array([c1, c0]) / n
        return -sum(pi * np.log2(pi) for pi in p if pi > 0)
    parent = H(5, 5)                       # 5 yes / 5 no
    # split -> left 4yes/1no, right 1yes/4no
    nl, nr = 5, 5
    left, right = H(4, 1), H(1, 4)
    child = (nl / 10) * left + (nr / 10) * right
    log.info(f"INFO-GAIN toy: parent H={parent:.3f} (5/5); "
             f"left H={left:.3f} (4/1), right H={right:.3f} (1/4)")
    log.info(f"  weighted child H = {child:.3f}; information gain = {parent - child:.3f}")


def lasso_selection(df, log):
    """How many features Lasso keeps on SCALED bike (the deck's selection claim)."""
    from sklearn.linear_model import LassoCV, Lasso
    X, y = df[FEATURES].values, df["cnt"].values
    Xs = StandardScaler().fit_transform(X)
    cv = LassoCV(cv=5, random_state=SEED, max_iter=10000).fit(Xs, y)
    n_keep = int(np.sum(cv.coef_ != 0))
    kept = [f for f, c in zip(FEATURES, cv.coef_) if c != 0]
    log.info(f"LASSO (scaled, LassoCV alpha={cv.alpha_:.1f}): keeps {n_keep}/{len(FEATURES)} "
             f"-> {kept}")
    for mult in (3, 10, 30):
        a = cv.alpha_ * mult
        k = int(np.sum(Lasso(alpha=a, max_iter=10000).fit(Xs, y).coef_ != 0))
        log.info(f"  stronger penalty alpha={a:.0f} (x{mult}): keeps {k}/{len(FEATURES)}")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    df = get_bike(log)
    fig_methods_disagree(df, log)
    fig_tree(df, log)
    worked_info_gain(log)
    lasso_selection(df, log)
    log.info("done.")


if __name__ == "__main__":
    main()
