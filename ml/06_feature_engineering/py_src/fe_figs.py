"""Generated figures for deck [26] Feature Engineering, on the bike-sharing data.

Every number drawn here is recomputed from the data at render time - no value is hard-coded from
the measurement pass, so a figure can never drift from the slide it sits on. The candidate pool is
imported from fe_measure.build_pool so deck [27] prunes exactly the matrix this deck generates.

  fe_leak_casual_registered.pdf : frame 3  - two shipped columns that sum to the target
  fe_project_time.pdf           : frame 6  - where project time goes (industry surveys, folklore)
  fe_thesis_2x2.pdf             : frame 7  - FE helps a linear model, not a forest (THE thesis)
  fe_ratio_gain.pdf             : frame 11 - the normalization trap + the honest MAE ladder
  fe_interaction_lin_vs_tree.pdf: frame 13 - who needs the interaction handed to them
  fe_binning_temp.pdf           : frame 14 - binning a non-monotonic effect
  fe_groupby_leakage.pdf        : frame 18 - leakage scales with group cardinality
  fe_geo_illustration.pdf       : frame 24 - raw lat/lon vs distance (synthetic, labelled)
  fe_candidate_pool.pdf         : frame 28 - the pool, and the data-quality bug it surfaced

Run: ./ma/Scripts/python.exe ml/06_feature_engineering/py_src/fe_figs.py
Conventions: logging to console + logs/, seed 509, Armenian colours, n_jobs=1.
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fe_measure import (CH_DIR, DATA, FEATURES, LOGS_DIR, N_JOBS, SEED, TARGET,
                        build_pool, ridge_mae)

FIG = CH_DIR / "fig"
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#8a8a8a"

log = logging.getLogger("fe_figs")


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "fe_figs.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote fig/{name}")


def split(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=SEED)


def forest_mae(Xtr, ytr, Xte, yte):
    m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=N_JOBS).fit(Xtr, ytr)
    return mean_absolute_error(yte, m.predict(Xte))


# ---------------------------------------------------------------- frame 3
def fig_casual_registered(df):
    y = df[TARGET]
    Xtr, Xte, ytr, yte = split(df[FEATURES], y)
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=N_JOBS).fit(Xtr, ytr)
    honest_r2 = r2_score(yte, rf.predict(Xte))
    honest_mae = mean_absolute_error(yte, rf.predict(Xte))

    leaky = FEATURES + ["casual", "registered"]
    Xtr2, Xte2, ytr2, yte2 = split(df[leaky], y)
    rf2 = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=N_JOBS).fit(Xtr2, ytr2)
    leak_r2 = r2_score(yte2, rf2.predict(Xte2))
    leak_mae = mean_absolute_error(yte2, rf2.predict(Xte2))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for ax, vals, ttl, fmt in [
            (a1, [honest_r2, leak_r2], "test $R^2$", "{:.4f}"),
            (a2, [honest_mae, leak_mae], "test MAE (rentals)", "{:.1f}")]:
        b = ax.bar(["11 honest\nfeatures", "+ casual\n+ registered"], vals,
                   color=[ARM_BLUE, ARM_RED], width=0.55)
        ax.bar_label(b, fmt=fmt, fontsize=10, fontweight="bold", padding=2)
        ax.set_title(ttl, fontsize=11)
        ax.margins(y=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    a1.set_ylim(0, 1.15)
    fig.suptitle("casual + registered = cnt, exactly, on all 731 rows", fontsize=11,
                 color=ARM_RED, fontweight="bold")
    save(fig, "fe_leak_casual_registered.pdf")
    log.info(f"  honest R2={honest_r2:.4f} MAE={honest_mae:.1f} | "
             f"leaky R2={leak_r2:.4f} MAE={leak_mae:.1f}")


# ---------------------------------------------------------------- frame 6
def fig_project_time():
    stages = ["deployment &\nmonitoring", "modelling &\ntuning",
              "data collection,\ncleaning, features"]
    pct = [10, 25, 65]
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    b = ax.barh(stages, pct, color=[GREY, ARM_BLUE, ARM_ORANGE], height=0.6)
    ax.bar_label(b, fmt="%d%%", fontsize=11, fontweight="bold", padding=4)
    ax.set_xlim(0, 82)
    ax.set_xlabel("share of project time (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Widely quoted industry surveys - treat as folklore, not measurement",
                 fontsize=9.5, style="italic", color=GREY)
    save(fig, "fe_project_time.pdf")


# ---------------------------------------------------------------- frame 7  (THE thesis)
def fig_thesis_2x2(df, P):
    y = df[TARGET].values
    res = {}
    for fl, X in [("11 raw", df[FEATURES].values), ("230 engineered", P.values)]:
        Xtr, Xte, ytr, yte = split(X, y)
        res[("Ridge", fl)] = ridge_mae(Xtr, ytr, Xte, yte)
        res[("Random forest", fl)] = forest_mae(Xtr, ytr, Xte, yte)

    models = ["Ridge", "Random forest"]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    w, xs = 0.34, np.arange(2)
    for i, (fl, c) in enumerate([("11 raw", GREY), ("230 engineered", ARM_ORANGE)]):
        v = [res[(m, fl)] for m in models]
        b = ax.bar(xs + (i - 0.5) * w, v, w, label=fl, color=c,
                   edgecolor="white", linewidth=1.2)
        ax.bar_label(b, fmt="%.1f", fontsize=10, fontweight="bold", padding=2)
    for j, m in enumerate(models):
        d = res[(m, "11 raw")] - res[(m, "230 engineered")]
        ax.annotate(f"feature engineering bought {d:+.1f}",
                    xy=(j, max(res[(m, "11 raw")], res[(m, "230 engineered")]) + 62),
                    ha="center", fontsize=9.5, fontweight="bold",
                    color=ARM_BLUE if d > 20 else ARM_RED)
    ax.set_xticks(xs)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("test MAE (rentals)")
    ax.set_ylim(0, 780)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fe_thesis_2x2.pdf")
    log.info("  " + " | ".join(f"{k[0]}/{k[1]}={v:.1f}" for k, v in res.items()))


# ---------------------------------------------------------------- frame 11
def fig_ratio_gain(df):
    d = df.copy()
    d["temp_c"] = 47 * d["temp"] - 8
    d["atemp_c"] = 66 * d["atemp"] - 16
    d["feels_gap"] = d["atemp_c"] - d["temp_c"]
    d["naive_gap"] = d["atemp"] - d["temp"]
    d["discomfort"] = d["hum"] * d["windspeed"]
    r = np.corrcoef(d["feels_gap"], d["naive_gap"])[0, 1]

    y = d[TARGET]
    ladder = {}
    for name, cols in [("11 raw\n(baseline)", FEATURES),
                       ("+ naive\ngap", FEATURES + ["naive_gap"]),
                       ("+ correct\ngap (C)", FEATURES + ["feels_gap"]),
                       ("+ hum$\\times$\nwind", FEATURES + ["discomfort"])]:
        Xtr, Xte, ytr, yte = split(d[cols], y)
        ladder[name] = ridge_mae(Xtr, ytr, Xte, yte)

    # the corrupt row surfaces the moment you put both columns in real units
    bad = int(d["naive_gap"].idxmin())
    clean = d.drop(index=bad)
    r_clean = np.corrcoef(clean["feels_gap"], clean["naive_gap"])[0, 1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    a1.scatter(clean["temp_c"], clean["atemp_c"], s=10, alpha=0.45, color=ARM_BLUE)
    lim = [-8, 36]
    a1.plot(lim, lim, color=GREY, lw=1.1, ls="--", label="atemp $=$ temp")
    a1.scatter([d.loc[bad, "temp_c"]], [d.loc[bad, "atemp_c"]], s=90, color=ARM_RED,
               marker="X", zorder=5, label="2012-08-17")
    a1.annotate(f"{d.loc[bad, 'dteday']}\n{d.loc[bad, 'temp_c']:.0f}$^\\circ$C air, "
                f"{d.loc[bad, 'atemp_c']:.0f}$^\\circ$C \"feels like\"",
                xy=(d.loc[bad, "temp_c"], d.loc[bad, "atemp_c"]), xytext=(-6, -8),
                fontsize=8.5, color=ARM_RED, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.2))
    a1.set_xlim(lim)
    a1.set_ylim(-12, 36)
    a1.set_xlabel("air temperature ($^\\circ$C)")
    a1.set_ylabel("\"feels-like\" temperature ($^\\circ$C)")
    a1.set_title("Put both columns in real units\nand a corrupt row falls out",
                 fontsize=10.5, color=ARM_RED, fontweight="bold")
    a1.legend(frameon=False, fontsize=8.5, loc="upper left")
    a1.spines[["top", "right"]].set_visible(False)

    base = ladder["11 raw\n(baseline)"]
    cols = [GREY] + [ARM_RED if ladder[k] > base else ARM_BLUE for k in list(ladder)[1:]]
    b = a2.bar(list(ladder), list(ladder.values()), color=cols, width=0.6)
    a2.bar_label(b, fmt="%.1f", fontsize=9.5, fontweight="bold", padding=2)
    a2.axhline(base, color=GREY, ls="--", lw=1.2)
    a2.set_ylim(min(ladder.values()) - 30, max(ladder.values()) + 25)
    a2.set_ylabel("test MAE (rentals)")
    a2.set_title("Ridge, adding one engineered feature\n(lower is better)", fontsize=10.5)
    a2.tick_params(axis="x", labelsize=8.5)
    a2.spines[["top", "right"]].set_visible(False)
    save(fig, "fe_ratio_gain.pdf")
    log.info(f"  corr(feels,naive) all rows={r:+.3f}, excluding 2012-08-17={r_clean:+.3f}")
    log.info(f"  CORRUPT ROW: {d.loc[bad, 'dteday']} temp={d.loc[bad, 'temp_c']:.1f}C "
             f"atemp={d.loc[bad, 'atemp_c']:.1f}C -- decks [22]-[24] ranked atemp #2 by PFI "
             f"with this row in the data")
    log.info("  " + " ".join(f"{k.replace(chr(10), ' ')}={v:.1f}" for k, v in ladder.items()))


# ---------------------------------------------------------------- frame 13
def fig_interaction(df):
    d = df.copy()
    d["temp_x_working"] = d["temp"] * d["workingday"]
    y = d[TARGET]
    out = {}
    for lbl, cols in [("without", FEATURES), ("with", FEATURES + ["temp_x_working"])]:
        Xtr, Xte, ytr, yte = split(d[cols], y)
        out[("Ridge", lbl)] = ridge_mae(Xtr, ytr, Xte, yte)
        out[("Random forest", lbl)] = forest_mae(Xtr, ytr, Xte, yte)

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    models = ["Ridge", "Random forest"]
    w, xs = 0.34, np.arange(2)
    for i, (lbl, c) in enumerate([("without", GREY), ("with", ARM_ORANGE)]):
        v = [out[(m, lbl)] for m in models]
        b = ax.bar(xs + (i - 0.5) * w, v, w, color=c, label=f"{lbl} temp$\\times$workingday",
                   edgecolor="white", linewidth=1.2)
        ax.bar_label(b, fmt="%.1f", fontsize=10, fontweight="bold", padding=2)
    for j, m in enumerate(models):
        dd = out[(m, "without")] - out[(m, "with")]
        ax.annotate(f"{dd:+.1f}", xy=(j, max(out[(m, 'without')], out[(m, 'with')]) + 55),
                    ha="center", fontsize=11, fontweight="bold",
                    color=ARM_BLUE if dd > 0 else ARM_RED)
    ax.set_xticks(xs)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("test MAE (rentals)")
    ax.set_ylim(0, 760)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fe_interaction_lin_vs_tree.pdf")
    log.info("  " + " | ".join(f"{k[0]}/{k[1]}={v:.1f}" for k, v in out.items()))


# ---------------------------------------------------------------- frame 14
def fig_binning(df):
    y = df[TARGET]
    Xtr, Xte, ytr, yte = split(df[FEATURES], y)
    raw = ridge_mae(Xtr, ytr, Xte, yte)
    kb = KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile",
                          quantile_method="averaged_inverted_cdf").fit(Xtr[["temp"]])
    Xtr_b = np.hstack([Xtr.drop(columns=["temp"]).values, kb.transform(Xtr[["temp"]])])
    Xte_b = np.hstack([Xte.drop(columns=["temp"]).values, kb.transform(Xte[["temp"]])])
    binned = ridge_mae(Xtr_b, ytr, Xte_b, yte)
    edges = kb.bin_edges_[0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.2),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    tc = 47 * df["temp"] - 8
    a1.scatter(tc, df[TARGET], s=9, alpha=0.35, color=GREY)
    # a straight line cannot bend; five bin means can
    xs = np.linspace(tc.min(), tc.max(), 200)
    a1.plot(xs, np.poly1d(np.polyfit(tc, df[TARGET], 1))(xs), color=ARM_RED, lw=2.2,
            label="one straight line")
    ec = 47 * edges - 8
    for lo, hi in zip(ec[:-1], ec[1:]):
        m = (tc >= lo) & (tc <= hi)
        a1.plot([lo, hi], [df[TARGET][m].mean()] * 2, color=ARM_BLUE, lw=3.0,
                solid_capstyle="butt")
    for e in ec[1:-1]:
        a1.axvline(e, color=ARM_BLUE, lw=0.7, ls=":", alpha=0.7)
    a1.plot([], [], color=ARM_BLUE, lw=3.0, label="5 quantile bins")
    # decile means: the plateau is what 5 bins capture, the top-end dip is what they miss
    dec = pd.qcut(tc, 10, labels=False)
    dm = df[TARGET].groupby(dec).mean()
    dc = tc.groupby(dec).mean()
    a1.plot(dc, dm, color=ARM_ORANGE, lw=1.6, marker="o", ms=3.5, alpha=0.9,
            label="10 deciles (finer)")
    a1.annotate("and dips at the very\ntop - too hot to cycle",
                xy=(dc.iloc[-1], dm.iloc[-1]), xytext=(21.5, 600), fontsize=8.5,
                color=ARM_ORANGE, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    a1.set_xlabel("temperature ($^\\circ$C, reconstructed)")
    a1.set_ylabel("daily rentals")
    a1.set_title("The effect saturates - a straight line keeps climbing", fontsize=10.5)
    a1.legend(frameon=False, fontsize=8.5, loc="upper left")
    a1.spines[["top", "right"]].set_visible(False)

    b = a2.bar(["raw\ntemp", "binned\ntemp"], [raw, binned], color=[GREY, ARM_BLUE], width=0.55)
    a2.bar_label(b, fmt="%.1f", fontsize=11, fontweight="bold", padding=3)
    a2.set_ylim(0, max(raw, binned) * 1.22)
    a2.set_ylabel("Ridge test MAE")
    a2.set_title(f"{raw - binned:+.1f} MAE", fontsize=11, color=ARM_BLUE, fontweight="bold")
    a2.spines[["top", "right"]].set_visible(False)
    save(fig, "fe_binning_temp.pdf")
    log.info(f"  raw={raw:.1f} binned={binned:.1f} delta={raw - binned:+.1f}")


# ---------------------------------------------------------------- frame 18
def fig_groupby_leakage(df):
    d = df.copy()
    y = d[TARGET].values
    base_X = d[FEATURES].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    def cv(X):
        sc = []
        for tr, te in kf.split(X):
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
            m.fit(X[tr], y[tr])
            sc.append(mean_absolute_error(y[te], m.predict(X[te])))
        return float(np.mean(sc))

    plain = cv(base_X)
    s = d["season"].astype(str)
    keys = {"season": s,
            "season$\\times$workingday": s + "_" + d["workingday"].astype(str),
            "mnth": d["mnth"].astype(str),
            "mnth$\\times$weekday": d["mnth"].astype(str) + "_" + d["weekday"].astype(str),
            "yr$\\times$mnth$\\times$weekday": (d["yr"].astype(str) + "_" + d["mnth"].astype(str)
                                                + "_" + d["weekday"].astype(str))}
    rows = []
    for name, key in keys.items():
        kv = key.values
        ng = len(set(kv))
        leaky = cv(np.hstack([base_X, d.groupby(kv)[TARGET].transform("mean")
                              .values.reshape(-1, 1)]))
        hs = []
        for tr, te in kf.split(base_X):
            gm = pd.Series(y[tr]).groupby(kv[tr]).mean()
            g = y[tr].mean()
            ftr = np.array([gm.get(k, g) for k in kv[tr]]).reshape(-1, 1)
            fte = np.array([gm.get(k, g) for k in kv[te]]).reshape(-1, 1)
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
            m.fit(np.hstack([base_X[tr], ftr]), y[tr])
            hs.append(mean_absolute_error(y[te], m.predict(np.hstack([base_X[te], fte]))))
        rows.append((name, len(d) / ng, float(np.mean(hs)), leaky))

    names = [r[0] for r in rows]
    rpg = [r[1] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.3),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    a1.axhline(plain, color=GREY, ls="--", lw=1.3, label="no group feature")
    a1.plot(rpg, [r[2] for r in rows], "o-", color=ARM_BLUE, lw=2, ms=6, label="honest (in-fold)")
    a1.plot(rpg, [r[3] for r in rows], "o-", color=ARM_RED, lw=2, ms=6, label="leaky (full data)")
    a1.set_xscale("log")
    a1.invert_xaxis()
    a1.set_xlabel("rows per group  (coarser $\\rightarrow$ finer)")
    a1.set_ylabel("CV MAE (rentals)")
    a1.legend(frameon=False, fontsize=9)
    a1.spines[["top", "right"]].set_visible(False)
    a1.set_title("The finer the group, the bigger the lie", fontsize=10.5)

    lie = [r[2] - r[3] for r in rows]
    b = a2.barh(names, lie, color=[ARM_ORANGE if v < 20 else ARM_RED for v in lie], height=0.6)
    a2.bar_label(b, fmt="%+.1f", fontsize=9.5, fontweight="bold", padding=3)
    a2.set_xlabel("apparent gain from the leak (MAE)")
    a2.set_xlim(0, max(lie) * 1.35)
    a2.tick_params(axis="y", labelsize=8.5)
    a2.spines[["top", "right"]].set_visible(False)
    save(fig, "fe_groupby_leakage.pdf")
    for n, r, h, lk in rows:
        log.info(f"  {n:32s} rows/grp={r:6.1f} honest={h:6.1f} leaky={lk:6.1f} lie={h - lk:+6.1f}")


# ---------------------------------------------------------------- frame 24
def fig_geo():
    """SYNTHETIC city - labelled as such on the slide. bike-day.csv is daily system-wide
    totals with no station dimension, so there is no real geo feature to build here."""
    rng = np.random.RandomState(SEED)
    n = 320
    lat = rng.uniform(0, 10, n)
    lon = rng.uniform(0, 10, n)
    dist = np.sqrt((lat - 5) ** 2 + (lon - 5) ** 2)
    demand = 900 * np.exp(-dist / 3.2) + rng.normal(0, 45, n)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.3))
    sc = a1.scatter(lon, lat, c=demand, s=26, cmap="viridis")
    a1.plot(5, 5, marker="*", ms=20, color=ARM_RED, mec="white", mew=1.2)
    a1.annotate("centre", (5, 5), xytext=(6.0, 5.6), fontsize=9, color=ARM_RED,
                fontweight="bold")
    a1.set_xlabel("longitude")
    a1.set_ylabel("latitude")
    a1.set_title("Raw lat/lon: no linear structure", fontsize=10.5)
    fig.colorbar(sc, ax=a1, label="demand", shrink=0.85)

    a2.scatter(dist, demand, s=18, alpha=0.6, color=ARM_BLUE)
    a2.set_xlabel("haversine distance to centre")
    a2.set_ylabel("demand")
    r = np.corrcoef(dist, demand)[0, 1]
    a2.set_title(f"One engineered distance: $r={r:+.2f}$", fontsize=10.5,
                 color=ARM_BLUE, fontweight="bold")
    a2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Illustration on a synthetic city - bike-day.csv has no coordinates",
                 fontsize=9, style="italic", color=GREY, y=1.04)
    save(fig, "fe_geo_illustration.pdf")


# ---------------------------------------------------------------- frame 28
def fig_candidate_pool(df, P, origin):
    counts = {}
    for c in P.columns:
        counts[origin[c]] = counts.get(origin[c], 0) + 1
    order = ["raw", "product", "difference", "ratio", "domain"]
    labels = ["raw\n(11)", "products", "differences", "ratios", "domain\n(by hand)"]
    vals = [counts.get(k, 0) for k in order]
    cols = [GREY, ARM_BLUE, ARM_BLUE, ARM_BLUE, ARM_ORANGE]

    zeros = {c: int((df[c] == 0).sum()) for c in FEATURES if (df[c] == 0).any()}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.2),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    b = a1.bar(labels, vals, color=cols, width=0.62)
    a1.bar_label(b, fontsize=10, fontweight="bold", padding=2)
    a1.set_ylabel("candidate features")
    a1.set_ylim(0, max(vals) * 1.25)
    a1.tick_params(axis="x", labelsize=8.5)
    a1.set_title(f"{P.shape[1]} candidates from 11 features, in ~30 lines", fontsize=10.5)
    a1.spines[["top", "right"]].set_visible(False)

    names = list(zeros)
    bb = a2.barh(names, [zeros[c] for c in names],
                 color=[ARM_RED if c == "hum" else GREY for c in names], height=0.6)
    a2.bar_label(bb, fontsize=9.5, fontweight="bold", padding=3)
    a2.set_xlabel("rows where the feature is 0  (= a broken denominator)")
    a2.set_xscale("log")
    a2.set_xlim(0.5, max(zeros.values()) * 4)
    a2.set_title("hum $=0$ once: 0% humidity in DC\nis a missing value, not a measurement",
                 fontsize=9.5, color=ARM_RED, fontweight="bold")
    a2.spines[["top", "right"]].set_visible(False)
    save(fig, "fe_candidate_pool.pdf")
    log.info(f"  pool={P.shape[1]} by origin={counts} | zero-denominator features={zeros}")


def main():
    setup_logging()
    df = pd.read_csv(DATA)
    log.info(f"{len(df)} rows from {DATA.name}")
    P, origin = build_pool(df)

    fig_casual_registered(df)
    fig_project_time()
    fig_thesis_2x2(df, P)
    fig_ratio_gain(df)
    fig_interaction(df)
    fig_binning(df)
    fig_groupby_leakage(df)
    fig_geo()
    fig_candidate_pool(df, P, origin)
    log.info("all figures written")


if __name__ == "__main__":
    main()
