"""Generated figures for Interpretability Deck 2 (PFI & feature effects), on bike.

  pfi_bar.pdf            : permutation feature importance on the bike test set (with error bars).
  pdp_bike.pdf           : PDP for temp & hum (the effect shape).
  ice_interaction_toy.pdf: a synthetic case where the PDP is ~flat but ICE fans out
                           (ICE reveals an interaction the PDP averages away).

(LMU CC-BY figures reused in the deck: pfi_demo2 animation, pfi_test_vs_train, pfi_extrapolation,
 ale_vs_pdp.)

Run: ./ma/Scripts/python.exe ml/05_interpretability/py_src/interp_pfi_effects_figs.py
Conventions: logging to console + logs/, seed 509, f-strings, Armenian colours.
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
DATA = CH_DIR / "data" / "bike-day.csv"
LOGS_DIR = REPO_ROOT / "logs"
FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed"]


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("deck2"); log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "interp_deck2.log")):
        h.setFormatter(fmt); log.addHandler(h)
    return log


def fig_pfi_and_pdp(log):
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["cnt"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(Xtr, ytr)

    # --- PFI on test ---
    r = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=SEED,
                               scoring="neg_mean_absolute_error")
    imp = pd.Series(r.importances_mean, index=FEATURES)  # = MAE increase (neg-MAE units)
    err = pd.Series(r.importances_std, index=FEATURES)
    order = imp.sort_values().index
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.barh(range(len(order)), imp[order].values, xerr=err[order].values,
            color=ARM_RED, error_kw=dict(ecolor="0.4", lw=1))
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlabel("increase in test MAE when permuted")
    ax.set_title("Permutation feature importance (bike, test set)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "pfi_bar.pdf", bbox_inches="tight"); plt.close(fig)
    log.info(f"PFI top3 (test): {list(imp.sort_values().tail(3).index[::-1])}")
    log.info("wrote pfi_bar.pdf")

    # --- PDP for temp & hum ---
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.8))
    PartialDependenceDisplay.from_estimator(
        rf, Xte, ["temp", "hum"], kind="average", ax=ax,
        line_kw=dict(color=ARM_BLUE, lw=2.4))
    for a in ax:
        a.set_ylabel("partial dependence (cnt)", fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Partial dependence: how the prediction moves with a feature", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "pdp_bike.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("wrote pdp_bike.pdf")


def fig_interaction_toy(log):
    """PDP ~flat but ICE fans out: y = group * x, so opposing per-instance slopes cancel."""
    rng = np.random.RandomState(SEED)
    n = 600
    x = rng.uniform(-3, 3, n)
    group = rng.choice([-1.0, 1.0], n)
    y = group * x + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"x": x, "group": group})
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X, y)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    PartialDependenceDisplay.from_estimator(
        rf, X, ["x"], kind="both", ax=ax, ice_lines_kw=dict(color=ARM_BLUE, alpha=0.12),
        pd_line_kw=dict(color=ARM_ORANGE, lw=3, label="PDP (average)"))
    ax.set_ylabel("prediction")
    ax.set_title("PDP is flat, but ICE fans out\n(ICE reveals the interaction PDP hides)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(FIG_DIR / "ice_interaction_toy.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote ice_interaction_toy.pdf")


def fig_correlation_trap(log):
    """Bike temp vs atemp: permuting temp breaks the joint -> impossible rows (extrapolation)."""
    df = pd.read_csv(DATA)
    t, at = df["temp"].values, df["atemp"].values
    rng = np.random.RandomState(SEED)
    t_perm = rng.permutation(t)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    a1.scatter(t, at, s=10, c=ARM_BLUE, alpha=0.5)
    a1.set_title("real data: temp ≈ atemp (correlated)", fontsize=10)
    a2.scatter(t_perm, at, s=10, c=ARM_RED, alpha=0.5)
    a2.set_title("after permuting temp: impossible rows", fontsize=10)
    a2.annotate("warm temp,\ncold atemp\n(never happens)", xy=(0.8, 0.15), xytext=(0.45, 0.05),
                fontsize=8, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    for a in (a1, a2):
        a.set_xlabel("temp"); a.spines[["top", "right"]].set_visible(False)
    a1.set_ylabel("atemp (feels-like)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "trap_temp_atemp.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"trap: corr(temp, atemp) = {np.corrcoef(t, at)[0,1]:.2f}")
    log.info("wrote trap_temp_atemp.pdf")


def fig_ale_vs_pdp(log):
    """PDP vs a manual 1D ALE for temp on the bike RF (ALE stays inside the real data)."""
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["cnt"].values
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X, y)
    t = X["temp"].values
    K = 20
    edges = np.unique(np.quantile(t, np.linspace(0, 1, K + 1)))
    ale = [0.0]; counts = []
    for k in range(1, len(edges)):
        m = (t > edges[k - 1]) & (t <= edges[k])
        if k == 1:
            m |= (t == edges[0])
        if m.sum() == 0:
            ale.append(ale[-1]); counts.append(0); continue
        lo = X[m].copy(); lo["temp"] = edges[k - 1]
        hi = X[m].copy(); hi["temp"] = edges[k]
        ale.append(ale[-1] + np.mean(rf.predict(hi) - rf.predict(lo)))
        counts.append(int(m.sum()))
    ale = np.array(ale)
    centers = (ale[:-1] + ale[1:]) / 2
    ale -= np.average(centers, weights=counts)          # centre ALE

    grid = np.linspace(t.min(), t.max(), 60)
    pdp = np.array([rf.predict(X.assign(temp=g)).mean() for g in grid])
    pdp -= pdp.mean()                                    # centre PDP to compare

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(grid, pdp, color=ARM_BLUE, lw=2.4, label="PDP (averages over all rows)")
    ax.plot(edges, ale, color=ARM_ORANGE, lw=2.4, label="ALE (local, stays realistic)")
    ax.set_xlabel("temp"); ax.set_ylabel("centred effect on cnt")
    ax.set_title("PDP vs ALE for temp (bike)")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "ale_vs_pdp_bike.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"ALE vs PDP: max abs gap = {np.max(np.abs(np.interp(edges, grid, pdp) - ale)):.0f} rentals")
    log.info("wrote ale_vs_pdp_bike.pdf")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig_pfi_and_pdp(log)
    fig_interaction_toy(log)
    fig_correlation_trap(log)
    fig_ale_vs_pdp(log)
    log.info("done.")


if __name__ == "__main__":
    main()
