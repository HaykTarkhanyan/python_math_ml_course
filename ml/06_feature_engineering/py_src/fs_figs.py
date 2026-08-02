"""Generated figures for deck [27] Feature Selection, on the bike-sharing data.

HARD DEPENDENCY: this script LOADS the candidate pool written by fe_measure.build_pool
(data/bike_candidates.csv). It does not regenerate it. Decks 26 and 27 must prune exactly the
matrix deck 26 generated, or the chaining between them is decorative.
Run fe_measure.py first if the file is missing (it is gitignored - regenerable in ~10s).

  fs_both_regimes.pdf      : frames 7-8  - selection buys nothing at p=11, +68.2 at p=230
  fs_add_features_ucurve.pdf: frame 6    - train falls forever, test is U-shaped
  fs_filter_scores.pdf     : frame 17    - where F-test and mutual information disagree (mnth)
  fs_xor_trap.pdf          : frame 18    - the independence trap
  fs_multiple_testing.pdf  : frame 20    - how many pure-noise features pass at alpha=0.05
  fs_rfecv_curve.pdf       : frame 23    - where the cutoff comes from, and what survives
  fs_boruta_shadow.pdf     : frame 26    - real features vs the max shadow
  fs_stability.pdf         : frame 28    - selection frequency across 100 bootstraps
  fs_selection_bias.pdf    : frame 31    - selecting outside the CV loop inflates the score

Run: ./ma/Scripts/python.exe ml/06_feature_engineering/py_src/fs_figs.py
Conventions: logging to console + logs/, seed 509, Armenian colours, n_jobs=1 (single core).
"""
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, f_regression, mutual_info_regression
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fe_measure import CH_DIR, DATA, FEATURES, LOGS_DIR, N_JOBS, OUT_DIR, SEED, TARGET

FIG = CH_DIR / "fig"
POOL_CSV = OUT_DIR / "bike_candidates.csv"
ORIGIN_CSV = OUT_DIR / "bike_candidates_origin.csv"
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#8a8a8a"

log = logging.getLogger("fs_figs")


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(),
              logging.FileHandler(LOGS_DIR / "fs_figs.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def load_pool():
    if not POOL_CSV.exists():
        raise FileNotFoundError(
            f"{POOL_CSV} is missing. Deck 27 prunes the pool deck 26 generates - run "
            f"./ma/Scripts/python.exe ml/06_feature_engineering/py_src/fe_measure.py first.")
    P = pd.read_csv(POOL_CSV)
    y = P.pop(TARGET).values
    origin = pd.read_csv(ORIGIN_CSV, index_col=0)["origin"].to_dict()
    log.info(f"loaded pool: {P.shape[1]} candidates x {P.shape[0]} rows")
    return P, y, origin


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote fig/{name}")


def split(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=SEED)


def ridge_mae(Xtr, ytr, Xte, yte):
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED)).fit(Xtr, ytr)
    return mean_absolute_error(yte, m.predict(Xte))


def rfecv_fit(Xtr, ytr, step):
    sel = RFECV(Ridge(alpha=1.0, random_state=SEED), step=step, cv=5,
                scoring="neg_mean_absolute_error", min_features_to_select=2, n_jobs=N_JOBS)
    sel.fit(StandardScaler().fit_transform(Xtr), ytr)
    return sel


# ---------------------------------------------------------------- frames 7-8
def fig_both_regimes(raw_df, P, y):
    rows = []
    for label, X in [("11 raw features", raw_df[FEATURES].values),
                     (f"{P.shape[1]} candidates", P.values)]:
        Xtr, Xte, ytr, yte = split(X, y)
        keep = ridge_mae(Xtr, ytr, Xte, yte)
        sel = rfecv_fit(Xtr, ytr, 1 if X.shape[1] < 30 else 5)
        chosen = ridge_mae(Xtr[:, sel.support_], ytr, Xte[:, sel.support_], yte)
        rows.append((label, X.shape[1], int(sel.n_features_), keep, chosen))

    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    xs, w = np.arange(2), 0.34
    for i, (lbl, c, idx) in enumerate([("keep everything", GREY, 3),
                                       ("after selection", ARM_BLUE, 4)]):
        v = [r[idx] for r in rows]
        b = ax.bar(xs + (i - 0.5) * w, v, w, label=lbl, color=c, edgecolor="white", linewidth=1.2)
        ax.bar_label(b, fmt="%.1f", fontsize=10, fontweight="bold", padding=2)
    for j, r in enumerate(rows):
        d = r[3] - r[4]
        ax.annotate(f"selection bought {d:+.1f}" + ("  (nothing)" if abs(d) < 0.05 else ""),
                    xy=(j, max(r[3], r[4]) + 95), ha="center", fontsize=9.5, fontweight="bold",
                    color=ARM_RED if abs(d) < 0.05 else ARM_BLUE)
    ax.axhline(455.1, color=ARM_ORANGE, ls="--", lw=1.8,
               label="a forest on the raw 11, no engineering (455.1)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r[0]}\n(kept {r[2]} of {r[1]})" for r in rows], fontsize=10)
    ax.set_ylabel("test MAE (rentals)")
    ax.set_ylim(0, 860)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.19))
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fs_both_regimes.pdf")
    for r in rows:
        log.info(f"  {r[0]:22s} p={r[1]:4d} k={r[2]:3d} keep-all={r[3]:7.1f} sel={r[4]:7.1f}")


# ---------------------------------------------------------------- frame 6
def fig_ucurve(P, y):
    Xtr, Xte, ytr, yte = split(P.values, y)
    f, _ = f_regression(Xtr, ytr)                     # rank on TRAIN only
    order = np.argsort(-np.nan_to_num(f))
    ks = list(range(1, 61)) + list(range(65, P.shape[1] + 1, 5))
    tr_e, te_e = [], []
    for k in ks:
        cols = order[:k]
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
        m.fit(Xtr[:, cols], ytr)
        tr_e.append(mean_absolute_error(ytr, m.predict(Xtr[:, cols])))
        te_e.append(mean_absolute_error(yte, m.predict(Xte[:, cols])))
    best = int(np.argmin(te_e))

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.plot(ks, tr_e, color=ARM_BLUE, lw=2, label="training error")
    ax.plot(ks, te_e, color=ARM_RED, lw=2, label="test error")
    ax.plot(ks[best], te_e[best], marker="v", ms=11, color=ARM_ORANGE, zorder=5)
    ax.annotate(f"best: {ks[best]} features\n{te_e[best]:.1f} MAE",
                xy=(ks[best], te_e[best]), xytext=(ks[best] + 22, te_e[best] + 55),
                fontsize=9, color=ARM_ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE, lw=1.2))
    ax.set_xlabel("features kept (added best-first by univariate score)")
    ax.set_ylabel("MAE (rentals)")
    ax.legend(frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fs_add_features_ucurve.pdf")
    log.info(f"  U-curve: best k={ks[best]} at {te_e[best]:.1f}; "
             f"train {tr_e[0]:.0f}->{tr_e[-1]:.0f}, test {te_e[0]:.0f}->{te_e[-1]:.0f}")


# ---------------------------------------------------------------- frame 17
def fig_filter_scores(raw_df, y):
    X = raw_df[FEATURES]
    f, _ = f_regression(X, y)
    mi = mutual_info_regression(X, y, random_state=SEED)
    rf_ = pd.Series(f, index=FEATURES).rank(ascending=False)
    rmi = pd.Series(mi, index=FEATURES).rank(ascending=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for feat in FEATURES:
        gap = abs(rf_[feat] - rmi[feat])
        c = ARM_RED if gap >= 3 else GREY
        ax.plot([0, 1], [rf_[feat], rmi[feat]], "-o", color=c, lw=2.4 if gap >= 3 else 1.0,
                ms=5 if gap >= 3 else 3.5, alpha=1.0 if gap >= 3 else 0.55, zorder=3 if gap >= 3 else 1)
        ax.annotate(feat, xy=(-0.03, rf_[feat]), ha="right", va="center",
                    fontsize=8.5, color=c, fontweight="bold" if gap >= 3 else "normal")
        ax.annotate(feat, xy=(1.03, rmi[feat]), ha="left", va="center",
                    fontsize=8.5, color=c, fontweight="bold" if gap >= 3 else "normal")
    ax.set_xlim(-0.42, 1.42)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["F-test\n(linear)", "mutual information\n(any dependence)"], fontsize=10)
    ax.set_ylabel("rank (1 = most informative)")
    ax.set_yticks(range(1, 12))
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_title("Only one feature moves: mnth", fontsize=11, color=ARM_RED, fontweight="bold")
    save(fig, "fs_filter_scores.pdf")
    for feat in FEATURES:
        log.info(f"  {feat:12s} F-rank {rf_[feat]:.0f}  MI-rank {rmi[feat]:.0f}")


# ---------------------------------------------------------------- frame 18
def fig_xor_trap():
    rng = np.random.RandomState(SEED)
    n = 200
    x1 = rng.randint(0, 2, n)
    x2 = rng.randint(0, 2, n)
    yv = (x1 ^ x2).astype(float) + rng.normal(0, 0.12, n)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.4, 2.9))
    for ax, x, nm in [(a1, x1, "$x_1$"), (a2, x2, "$x_2$")]:
        means = [yv[x == v].mean() for v in (0, 1)]
        b = ax.bar(["0", "1"], means, color=GREY, width=0.5)
        ax.bar_label(b, fmt="%.2f", fontsize=10, fontweight="bold", padding=2)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel(nm)
        ax.set_ylabel("mean $y$" if ax is a1 else "")
        ax.set_title(f"{nm} alone: corr $=$ {np.corrcoef(x, yv)[0, 1]:+.2f}",
                     fontsize=10, color=ARM_RED)
        ax.spines[["top", "right"]].set_visible(False)

    for (v1, v2), c in zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                           [ARM_BLUE, ARM_RED, ARM_RED, ARM_BLUE]):
        m = (x1 == v1) & (x2 == v2)
        a3.scatter(x1[m] + rng.normal(0, 0.06, m.sum()),
                   x2[m] + rng.normal(0, 0.06, m.sum()), s=14, color=c, alpha=0.7)
    a3.set_xticks([0, 1])
    a3.set_yticks([0, 1])
    a3.set_xlabel("$x_1$")
    a3.set_ylabel("$x_2$")
    a3.set_title("together: perfectly separable", fontsize=10, color=ARM_BLUE,
                 fontweight="bold")
    a3.spines[["top", "right"]].set_visible(False)
    save(fig, "fs_xor_trap.pdf")


# ---------------------------------------------------------------- frame 20
def fig_multiple_testing(y, n_noise=10_000):
    rng = np.random.RandomState(SEED)
    noise = rng.normal(size=(len(y), n_noise))
    _, p = f_regression(noise, y)
    n05, n01 = int((p < 0.05).sum()), int((p < 0.01).sum())
    bonf = int((p < 0.05 / n_noise).sum())
    order = np.sort(p)
    bh = int((order <= 0.05 * np.arange(1, n_noise + 1) / n_noise).sum())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.0),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    a1.hist(p, bins=40, color=GREY, edgecolor="white")
    a1.axvspan(0, 0.05, color=ARM_RED, alpha=0.28)
    a1.axhline(n_noise / 40, color=ARM_BLUE, ls="--", lw=1.4, label="uniform expectation")
    a1.set_xlabel("$p$-value against the target")
    a1.set_ylabel("noise features")
    a1.set_title(f"{n_noise:,} columns of pure random numbers", fontsize=10.5)
    a1.annotate(f"{n05} land here", xy=(0.05, n_noise / 40 * 1.6), xytext=(0.18, n_noise / 40 * 2.1),
                fontsize=9.5, color=ARM_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ARM_RED, lw=1.3))
    a1.legend(frameon=False, fontsize=8.5, loc="lower right")
    a1.spines[["top", "right"]].set_visible(False)

    b = a2.bar(["$\\alpha=0.05$", "$\\alpha=0.01$", "Bonferroni", "Benjamini-\nHochberg"],
               [n05, n01, bonf, bh],
               color=[ARM_RED, ARM_RED, ARM_BLUE, ARM_BLUE], width=0.6)
    a2.bar_label(b, fontsize=10, fontweight="bold", padding=2)
    a2.set_ylabel("false discoveries")
    a2.set_ylim(0, n05 * 1.28)
    a2.tick_params(axis="x", labelsize=8.5)
    a2.set_title("Correction works. Nobody applies it.", fontsize=10.5)
    a2.spines[["top", "right"]].set_visible(False)
    save(fig, "fs_multiple_testing.pdf")
    log.info(f"  noise: 0.05->{n05}, 0.01->{n01}, Bonferroni->{bonf}, BH->{bh}, "
             f"best p={p.min():.2e}")


# ---------------------------------------------------------------- frame 23
def fig_rfecv(P, y, origin):
    Xtr, Xte, ytr, yte = split(P.values, y)
    sel = rfecv_fit(Xtr, ytr, 5)
    k = int(sel.n_features_)
    kept = P.columns[sel.support_].tolist()
    counts = {}
    for c in kept:
        counts[origin[c]] = counts.get(origin[c], 0) + 1

    grid = sel.cv_results_["mean_test_score"]
    n_feat = np.arange(len(grid)) * 5 + 2
    n_feat = n_feat[:len(grid)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.2),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    a1.plot(n_feat, -grid, color=ARM_BLUE, lw=2)
    a1.plot(k, -grid[int(np.argmax(grid))], marker="v", ms=12, color=ARM_ORANGE, zorder=5)
    a1.annotate(f"cross-validation picks $k={k}$", xy=(k, -grid[int(np.argmax(grid))]),
                xytext=(0.42, 0.55), textcoords="axes fraction", fontsize=9.5,
                color=ARM_ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE, lw=1.3))
    a1.set_xlabel("features kept")
    a1.set_ylabel("cross-validated MAE")
    a1.set_title("The cutoff comes from a curve, not a bar chart", fontsize=10.5)
    a1.spines[["top", "right"]].set_visible(False)

    order = ["raw", "product", "difference", "ratio", "domain"]
    offered = {o: sum(1 for c in P.columns if origin[c] == o) for o in order}
    surv = [counts.get(o, 0) for o in order]
    rate = [100 * counts.get(o, 0) / offered[o] for o in order]
    cols = [GREY, ARM_BLUE, ARM_BLUE, ARM_BLUE, ARM_ORANGE]
    b = a2.bar(order, rate, color=cols, width=0.62)
    a2.bar_label(b, labels=[f"{s}/{offered[o]}" for s, o in zip(surv, order)],
                 fontsize=9, fontweight="bold", padding=2)
    a2.set_ylabel("% of that kind that survived")
    a2.set_ylim(0, max(rate) * 1.3)
    a2.tick_params(axis="x", labelsize=8.5, rotation=20)
    a2.set_title("Hand-designed features survive at the\nhighest rate - but there are only 3",
                 fontsize=9.5)
    a2.spines[["top", "right"]].set_visible(False)
    save(fig, "fs_rfecv_curve.pdf")
    log.info(f"  RFE-CV k={k}, survivors by origin={counts}")
    log.info(f"  domain survivors: {[c for c in kept if origin[c] == 'domain']}")
    log.info(f"  survival rate by origin: " +
             ", ".join(f"{o}={counts.get(o, 0)}/{offered[o]}" for o in order))


# ---------------------------------------------------------------- frame 26
def fig_boruta(raw_df, y):
    from boruta import BorutaPy
    X = raw_df[FEATURES].values
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=N_JOBS, max_depth=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BorutaPy(rf, n_estimators="auto", max_iter=100, random_state=SEED, verbose=0)
        b.fit(X, y)

    # one shadow run, to draw the threshold the algorithm compares against
    rng = np.random.RandomState(SEED)
    Xs = np.hstack([X, np.apply_along_axis(rng.permutation, 0, X)])
    rf2 = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=N_JOBS,
                                max_depth=7).fit(Xs, y)
    imp = rf2.feature_importances_[:len(FEATURES)]
    shadow_max = rf2.feature_importances_[len(FEATURES):].max()

    o = np.argsort(imp)
    names = [FEATURES[i] for i in o]
    cols = [ARM_BLUE if b.support_[i] else ARM_RED for i in o]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.barh(names, imp[o], color=cols, height=0.66)
    ax.bar_label(bars, fmt="%.3f", fontsize=8.5, padding=3)
    ax.axvline(shadow_max, color="black", ls="--", lw=1.6)
    ax.annotate("max shadow importance\n(the bar to beat)", xy=(shadow_max, 3.0),
                xytext=(0.12, 2.6), fontsize=8.5, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlabel("random-forest importance")
    ax.set_xlim(0, max(imp) * 1.32)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Blue = confirmed,  red = rejected as no better than noise",
                 fontsize=10.5)
    save(fig, "fs_boruta_shadow.pdf")
    conf = [f for f, s in zip(FEATURES, b.support_) if s]
    log.info(f"  confirmed: {conf}")
    log.info(f"  rejected : {[f for f, s in zip(FEATURES, b.support_) if not s]}")
    log.info(f"  shadow_max={shadow_max:.4f}")


# ---------------------------------------------------------------- frame 28
def fig_stability(raw_df, y, n_boot=100):
    X = StandardScaler().fit_transform(raw_df[FEATURES].values)
    alpha = LassoCV(cv=5, random_state=SEED, n_jobs=N_JOBS, max_iter=5000).fit(X, y).alpha_
    counts = np.zeros(len(FEATURES))
    union = 0
    it, ia = FEATURES.index("temp"), FEATURES.index("atemp")
    for i in range(n_boot):
        Xb, yb = resample(X, y, random_state=SEED + i)
        s = Lasso(alpha=alpha, random_state=SEED, max_iter=5000).fit(Xb, yb).coef_ != 0
        counts += s
        union += bool(s[it] or s[ia])
    freq = counts / n_boot
    o = np.argsort(freq)
    names = [FEATURES[i] for i in o]
    cols = [ARM_ORANGE if FEATURES[i] in ("temp", "atemp")
            else (ARM_RED if freq[i] < 0.5 else GREY) for i in o]

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    b = ax.barh(names, freq[o] * 100, color=cols, height=0.66)
    ax.bar_label(b, fmt="%.0f%%", fontsize=9, fontweight="bold", padding=3)
    ax.axvline(50, color="black", ls="--", lw=1.5)
    ax.annotate("keep-threshold", xy=(50, -0.45), fontsize=8.5, ha="center")
    ax.barh(["temp OR atemp"], [union / n_boot * 100], color=ARM_BLUE, height=0.5)
    ax.set_xlabel("selected in % of 100 bootstrap Lasso fits")
    ax.set_xlim(0, 118)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"temp {freq[it]:.0%} + atemp {freq[ia]:.0%}, but one of them is always in "
                 f"({union / n_boot:.0%})", fontsize=10, color=ARM_BLUE, fontweight="bold")
    save(fig, "fs_stability.pdf")
    for f_, v in sorted(zip(FEATURES, freq), key=lambda t: -t[1]):
        log.info(f"  {f_:12s} {v:5.0%}")
    log.info(f"  union(temp or atemp) = {union / n_boot:.0%}")


# ---------------------------------------------------------------- frame 31
def fig_selection_bias(P, y, k=20, n_noise=500):
    """Two datasets, same mistake. Real pool AND pure noise: select on everything, then CV."""
    rng = np.random.RandomState(SEED)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {}
    for label, X in [("real candidates", P.values),
                     (f"{n_noise} pure-noise columns", rng.normal(size=(len(y), n_noise)))]:
        # WRONG: rank on all the data, then cross-validate the winners
        f_all, _ = f_regression(X, y)
        top = np.argsort(-np.nan_to_num(f_all))[:k]
        wrong = []
        for tr, te in kf.split(X):
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
            m.fit(X[tr][:, top], y[tr])
            wrong.append(mean_absolute_error(y[te], m.predict(X[te][:, top])))
        # RIGHT: rank inside each fold, on the training part only
        right = []
        for tr, te in kf.split(X):
            f_tr, _ = f_regression(X[tr], y[tr])
            t = np.argsort(-np.nan_to_num(f_tr))[:k]
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
            m.fit(X[tr][:, t], y[tr])
            right.append(mean_absolute_error(y[te], m.predict(X[te][:, t])))
        out[label] = (float(np.mean(wrong)), float(np.mean(right)))

    labels = list(out)
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    xs, w = np.arange(2), 0.34
    for i, (nm, c, idx) in enumerate([("selected on ALL data, then CV", ARM_RED, 0),
                                      ("selected INSIDE each fold", ARM_BLUE, 1)]):
        v = [out[l][idx] for l in labels]
        b = ax.bar(xs + (i - 0.5) * w, v, w, label=nm, color=c, edgecolor="white", linewidth=1.2)
        ax.bar_label(b, fmt="%.1f", fontsize=10, fontweight="bold", padding=2)
    top = max(max(v) for v in out.values())
    ax.set_ylim(0, top * 1.34)
    for j, l in enumerate(labels):
        gap = out[l][1] - out[l][0]
        ax.annotate(f"the lie: {gap:+.1f} MAE", xy=(j, max(out[l]) + top * 0.10), ha="center",
                    fontsize=10, fontweight="bold", color=ARM_RED)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(f"CV MAE of the top-{k} features")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fs_selection_bias.pdf")
    for l in labels:
        log.info(f"  {l:28s} wrong={out[l][0]:7.1f}  right={out[l][1]:7.1f}  "
                 f"lie={out[l][1] - out[l][0]:+7.1f}")


def main():
    setup_logging()
    raw_df = pd.read_csv(DATA)
    P, y, origin = load_pool()

    fig_both_regimes(raw_df, P, y)
    fig_ucurve(P, y)
    fig_filter_scores(raw_df, y)
    fig_xor_trap()
    fig_multiple_testing(y)
    fig_rfecv(P, y, origin)
    fig_boruta(raw_df, y)
    fig_stability(raw_df, y)
    fig_selection_bias(P, y)
    log.info("all figures written")


if __name__ == "__main__":
    main()
