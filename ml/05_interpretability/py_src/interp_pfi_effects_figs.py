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

    ale_on_grid = np.interp(grid, edges, ale)            # ALE on the PDP grid, for shading
    gap = pdp - ale_on_grid
    j = int(np.argmax(np.abs(gap)))                      # where the two disagree most (the extreme)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    # shade the disagreement so it survives slide scale
    ax.fill_between(grid, pdp, ale_on_grid, color=ARM_RED, alpha=0.15, lw=0,
                    label="disagreement (extrapolation)")
    ax.plot(grid, pdp, color=ARM_BLUE, lw=2.6, label="PDP (averages over all rows)")
    ax.plot(edges, ale, color=ARM_ORANGE, lw=2.6, label="ALE (local, stays realistic)")
    # annotate the biggest gap, at the extreme
    ax.annotate(f"PDP overshoots here\ngap $\\approx$ {abs(gap[j]):.0f} rentals",
                xy=(grid[j], (pdp[j] + ale_on_grid[j]) / 2), xytext=(0.30, 0.30),
                textcoords="axes fraction", fontsize=8, color=ARM_RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.2))
    ax.set_xlabel("temp"); ax.set_ylabel("centred effect on cnt")
    ax.set_title("PDP vs ALE for temp (bike)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "ale_vs_pdp_bike.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"ALE vs PDP: max abs gap = {abs(gap[j]):.0f} rentals at temp={grid[j]:.2f}")
    log.info("wrote ale_vs_pdp_bike.pdf")


def fig_interactions(log):
    """Friedman's pairwise H-statistic on the bike RF + a 2-D PDP of the top-interacting pair.

    H_{jk} in [0,1]: 0 = the two features act additively (joint effect = sum of the two main
    effects); ->1 = the joint effect is essentially all interaction. Computed on subsampled
    partial-dependence grids (light: one vectorised predict per PD function).
    """
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["cnt"].values
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X, y)

    rng = np.random.RandomState(SEED)
    E = X.iloc[rng.choice(len(X), size=min(150, len(X)), replace=False)].reset_index(drop=True)
    B = X.iloc[rng.choice(len(X), size=min(80, len(X)), replace=False)].reset_index(drop=True)
    cols = list(X.columns)
    B_arr, E_arr = B.to_numpy(), E.to_numpy()
    nb, ne = len(B), len(E)

    def pd_centered(feat_cols):
        """Centred partial dependence over feat_cols, evaluated at the E points."""
        ci = [cols.index(c) for c in feat_cols]
        M = np.tile(B_arr, (ne, 1))                       # background replicated per eval point
        M[:, ci] = np.repeat(E_arr[:, ci], nb, axis=0)    # overwrite the fixed feature(s)
        preds = rf.predict(pd.DataFrame(M, columns=cols)).reshape(ne, nb).mean(axis=1)
        return preds - preds.mean()                       # centre to zero mean

    # atemp excluded: corr(temp, atemp) = 0.99 -> H would be a correlation artefact, not real interaction
    feats = ["yr", "temp", "hum", "season", "mnth", "weathersit", "windspeed", "weekday"]
    pdj = {f: pd_centered([f]) for f in feats}
    H = {}
    for i, a in enumerate(feats):
        for c in feats[i + 1:]:
            pdac = pd_centered([a, c])
            num = np.sum((pdac - pdj[a] - pdj[c]) ** 2)
            den = np.sum(pdac ** 2) + 1e-12
            H[(a, c)] = float(np.clip(np.sqrt(num / den), 0.0, 1.0))
    top = sorted(H, key=H.get, reverse=True)[:8]
    log.info("H top pairs: " + ", ".join(f"{a}x{c}={H[(a, c)]:.2f}" for (a, c) in top))

    # --- H bar (top pairs) ---
    labels = [f"{a} × {c}" for (a, c) in top][::-1]
    vals = [H[p] for p in top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.barh(range(len(top)), vals, color=ARM_ORANGE)
    ax.set_yticks(range(len(top)), labels, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.20)                      # right pad so end labels don't clip
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlabel("interaction strength  H   (0 = additive,  1 = all interaction)")
    ax.set_title("Friedman's H-statistic: strongest feature pairs (bike)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "h_statistic_bar.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("wrote h_statistic_bar.pdf")

    # --- 2-D PDP on a SYNTHETIC strongly-interacting pair (illustrative). y = x1 * x2, so x1's
    #     slope reverses with the sign of x2 -> the surface is a clear saddle. The bike model's own
    #     interactions (H bar above) are mild, so this clean toy carries the "bands fan out =
    #     interaction" teaching point; the real ranking is measured on the H bar. ---
    rng2 = np.random.RandomState(SEED)
    nt = 800
    Xt = pd.DataFrame({"x1": rng2.uniform(-2, 2, nt), "x2": rng2.uniform(-2, 2, nt)})
    yt = Xt["x1"].values * Xt["x2"].values + rng2.normal(0, 0.2, nt)
    rft = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(Xt, yt)
    g = np.linspace(-2, 2, 25)
    bg = Xt.sample(150, random_state=SEED).to_numpy()
    Z = np.empty((len(g), len(g)))
    for ii, v1 in enumerate(g):
        for jj, v2 in enumerate(g):
            m = bg.copy(); m[:, 0] = v1; m[:, 1] = v2
            Z[jj, ii] = rft.predict(pd.DataFrame(m, columns=["x1", "x2"])).mean()
    fig, ax = plt.subplots(figsize=(5.9, 4.5))
    cf = ax.contourf(g, g, Z, levels=14, cmap="RdBu_r")
    fig.colorbar(cf, ax=ax, label="partial dependence")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("2-D PDP, strongly interacting pair (illustrative)\n"
                 r"$x_1$'s effect reverses with $x_2$: bands fan out", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "pdp_2d_interaction.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("wrote pdp_2d_interaction.pdf (synthetic saddle y=x1*x2)")


def main():
    log = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig_pfi_and_pdp(log)
    fig_interaction_toy(log)
    fig_correlation_trap(log)
    fig_ale_vs_pdp(log)
    fig_interactions(log)
    log.info("done.")


if __name__ == "__main__":
    main()
