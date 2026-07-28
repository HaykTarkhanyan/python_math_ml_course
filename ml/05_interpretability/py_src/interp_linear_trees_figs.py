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


def fig_bike_overview(df, log):
    """Introduce the running dataset: daily rentals over the two years.

    Shows at a glance the two effects the whole chapter keeps rediscovering - the year-on-year
    growth (why `yr` tops every importance ranking) and the seasonal temperature swing.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["dteday"])
    m11 = d.loc[d["yr"] == 0, "cnt"].mean()
    m12 = d.loc[d["yr"] == 1, "cnt"].mean()

    # Means go in the legend labels, not as free-floating annotations: any text placed in the
    # upper-left collides with the legend, and anywhere else collides with the data.
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    for yr, colour, label, mean in [(0, ARM_BLUE, "2011", m11), (1, ARM_RED, "2012", m12)]:
        s = d[d["yr"] == yr]
        ax.plot(s["date"], s["cnt"], lw=0.9, color=colour, alpha=0.85,
                label=f"{label}  (mean {mean:,.0f})")
        ax.axhline(mean, color=colour, ls="--", lw=1.5)
    ax.set_ylabel("daily rentals (cnt)", fontsize=10)
    ax.set_xlabel("date", fontsize=10)
    ax.set_title("Bike sharing, Washington DC: 731 days, 2011-2012", fontsize=11.5)
    ax.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bike_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"BIKE OVERVIEW: 2011 mean {m11:,.0f}, 2012 mean {m12:,.0f} "
             f"(+{100 * (m12 / m11 - 1):.0f}%)")
    log.info(f"  cnt range {d['cnt'].min()}-{d['cnt'].max()}, temp range "
             f"{d['temp'].min():.3f}-{d['temp'].max():.3f}")
    # UCI normalisation: temp = (T+8)/47 -> report the real degrees behind the deck's numbers
    for v in (0.43, d["temp"].mean(), d["temp"].min(), d["temp"].max()):
        log.info(f"  temp {v:.3f} -> {v * 47 - 8:.1f} C")
    log.info("wrote bike_overview.pdf")


def fig_temp_shape(df, log):
    """Why the forest ranks temp above the linear model does: the effect is not a straight line.

    Binned mean rentals vs temp, with the fitted single slope on top. The line has to average
    the steep climb and the flat/falling top end, so it under-states how much temp matters.
    """
    t, y = df["temp"].values, df["cnt"].values
    bins = np.linspace(t.min(), t.max(), 11)
    idx = np.digitize(t, bins) - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.array([y[idx == k].mean() for k in range(len(centers))])

    slope, intercept = np.polyfit(t, y, 1)
    quad = np.polyfit(t, y, 2)
    grid = np.linspace(t.min(), t.max(), 200)
    r2_lin = 1 - ((y - np.polyval([slope, intercept], t)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2_quad = 1 - ((y - np.polyval(quad, t)) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.scatter(t, y, s=8, alpha=0.18, color="0.5", edgecolors="none", label="days")
    ax.plot(centers, means, "o-", color=ARM_RED, lw=2.2, ms=7,
            label="actual mean rentals per temp bin")
    ax.plot(grid, np.polyval([slope, intercept], grid), ls="--", lw=2.2, color=ARM_BLUE,
            label=f"what the linear model fits ($R^2$={r2_lin:.2f})")
    ax.annotate("climbs steeply", xy=(0.3, 3000), xytext=(0.22, 5600), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.annotate("then flattens and dips", xy=(0.78, 5400), xytext=(0.45, 1800), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_xlabel("temp (normalised)", fontsize=10)
    ax.set_ylabel("daily rentals", fontsize=10)
    ax.set_title("One slope cannot say this", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temp_effect_shape.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"TEMP SHAPE: linear R^2={r2_lin:.3f} vs quadratic R^2={r2_quad:.3f} "
             f"(quad x^2 coef = {quad[0]:,.0f}, negative = concave)")
    log.info(f"  binned means: {[f'{m:,.0f}' for m in means]}")
    log.info("wrote temp_effect_shape.pdf")


def fig_corr_credit(df, log):
    """Correlated features split the credit: temp vs atemp on bike.

    Left  : the two features are almost the same column (Pearson r).
    Right : bootstrap scatter of their two standardized coefficients. The credit sloshes
            between them across resamples while their SUM barely moves - that is the
            "split credit" claim, shown rather than asserted.
    """
    X, y = df[FEATURES].values, df["cnt"].values
    Xs = StandardScaler().fit_transform(X)
    i_t, i_a = FEATURES.index("temp"), FEATURES.index("atemp")
    r = float(np.corrcoef(df["temp"], df["atemp"])[0, 1])

    rng = np.random.default_rng(SEED)
    n_boot = 400
    coefs = np.empty((n_boot, 2))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        c = LinearRegression().fit(Xs[idx], y[idx]).coef_
        coefs[b] = (c[i_t], c[i_a])
    sd_t, sd_a = coefs[:, 0].std(), coefs[:, 1].std()
    sd_sum = (coefs[:, 0] + coefs[:, 1]).std()
    full = LinearRegression().fit(Xs, y).coef_

    # reference: drop atemp -> temp alone carries the whole effect
    keep = [f for f in FEATURES if f != "atemp"]
    Xs_drop = StandardScaler().fit_transform(df[keep].values)
    coef_drop = LinearRegression().fit(Xs_drop, y).coef_[keep.index("temp")]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.3))

    a1.scatter(df["temp"], df["atemp"], s=9, alpha=0.45, color=ARM_BLUE, edgecolors="none")
    a1.plot([0, 1], [0, 1], ls="--", lw=1, color="0.45")
    a1.set_xlabel("temp", fontsize=9.5)
    a1.set_ylabel("atemp (feels-like)", fontsize=9.5)
    a1.set_title(f"Two features, one signal:  r = {r:.2f}", fontsize=10.5)

    a2.scatter(coefs[:, 0], coefs[:, 1], s=11, alpha=0.5, color=ARM_ORANGE, edgecolors="none")
    a2.scatter([full[i_t]], [full[i_a]], s=70, marker="X", color=ARM_RED,
               zorder=5, label="fit on the full data")
    lo = min(coefs.min(), 0) - 200
    hi = coefs.max() + 200
    s_mean = coefs.sum(axis=1).mean()
    a2.plot([lo, hi], [s_mean - lo, s_mean - hi], ls="--", lw=1.2, color=ARM_BLUE,
            label=f"constant sum ($\\approx${s_mean:,.0f})")
    a2.set_xlim(lo, hi); a2.set_ylim(lo, hi)
    a2.set_xlabel("coefficient on temp", fontsize=9.5)
    a2.set_ylabel("coefficient on atemp", fontsize=9.5)
    a2.set_title("400 bootstrap refits: the credit moves, the total does not", fontsize=10.5)
    a2.legend(fontsize=8, loc="upper right", framealpha=0.9)
    a2.text(0.03, 0.03,
            f"sd(temp) = {sd_t:,.0f}\nsd(atemp) = {sd_a:,.0f}\nsd(temp+atemp) = {sd_sum:,.0f}",
            transform=a2.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "corr_credit_split.pdf", bbox_inches="tight")
    plt.close(fig)

    log.info(f"CORR CREDIT: r(temp, atemp) = {r:.4f}")
    log.info(f"  full-data standardized coefs: temp={full[i_t]:+,.0f}  atemp={full[i_a]:+,.0f}  "
             f"sum={full[i_t] + full[i_a]:+,.0f}")
    log.info(f"  bootstrap sd: temp={sd_t:,.0f}  atemp={sd_a:,.0f}  SUM={sd_sum:,.0f} "
             f"(sum is {sd_t / sd_sum:.1f}x more stable than temp alone)")
    log.info(f"  sign flips across resamples: temp {np.mean(coefs[:, 0] < 0):.0%}, "
             f"atemp {np.mean(coefs[:, 1] < 0):.0%}")
    log.info(f"  drop atemp -> temp coef = {coef_drop:+,.0f} (vs {full[i_t]:+,.0f} with atemp in)")
    log.info("wrote corr_credit_split.pdf")


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
    fig_corr_credit(df, log)
    fig_temp_shape(df, log)
    fig_tree(df, log)
    worked_info_gain(log)
    lasso_selection(df, log)
    log.info("done.")


if __name__ == "__main__":
    main()
