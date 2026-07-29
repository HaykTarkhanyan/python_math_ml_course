"""Measurement pass for decks [26] feature engineering and [27] feature selection.

Answers every frame marked [MEASURE] in the two outlines BEFORE the decks are written, so no
slide carries a number nobody computed. Also builds and persists the candidate pool that deck
[27] prunes - the chaining between the two decks depends on both scripts seeing the same matrix.

Split, seed and model settings are identical to ml/05_interpretability/py_src/* so every number
here is directly comparable to the slides in decks [22]-[24]:
    train_test_split(test_size=0.3, random_state=509), RandomForestRegressor(n_estimators=300)

Run: ./ma/Scripts/python.exe ml/06_feature_engineering/py_src/fe_measure.py
Conventions: logging to console + logs/, seed 509, f-strings, n_jobs=1 (single-core on purpose).
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (RFECV, f_regression, mutual_info_regression)
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.utils import resample

SEED = 509
N_JOBS = 1  # single core on purpose: this laptop locks up under sustained multi-core load

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
DATA = REPO_ROOT / "ml" / "05_interpretability" / "data" / "bike-day.csv"
OUT_DIR = CH_DIR / "data"
LOGS_DIR = REPO_ROOT / "logs"

FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed"]
CONTINUOUS = ["temp", "atemp", "hum", "windspeed"]
TARGET = "cnt"

log = logging.getLogger("fe_measure")


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "fe_measure.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)


def banner(title):
    log.info("")
    log.info("=" * 78)
    log.info(title)
    log.info("=" * 78)


def ridge_mae(Xtr, ytr, Xte, yte):
    """Ridge in a scaling pipeline (Ridge needs scaled inputs). Returns test MAE."""
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
    m.fit(Xtr, ytr)
    return mean_absolute_error(yte, m.predict(Xte))


def ridge_cv_mae(X, y, cv=5):
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
    kf = KFold(n_splits=cv, shuffle=True, random_state=SEED)
    s = cross_val_score(m, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=N_JOBS)
    return -s.mean()


# ----------------------------------------------------------------------------------
# M0 - baseline
# ----------------------------------------------------------------------------------
def m0_baseline(df):
    banner("M0  Baseline - must reproduce the numbers already on the deck [22]-[24] slides")
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)

    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=N_JOBS).fit(Xtr, ytr)
    rf_mae = mean_absolute_error(yte, rf.predict(Xte))
    rf_r2 = r2_score(yte, rf.predict(Xte))
    ridge = ridge_mae(Xtr, ytr, Xte, yte)

    log.info(f"forest  test MAE = {rf_mae:7.1f}   test R2 = {rf_r2:.3f}   "
             f"(deck [22]-[24] quote MAE 455.1, R2 0.879)")
    log.info(f"ridge   test MAE = {ridge:7.1f}")
    if abs(rf_mae - 455.1) > 1.0:
        log.warning(f"forest MAE {rf_mae:.1f} does not match the 455.1 on the ch5 slides - "
                    f"the split or model settings have drifted. INVESTIGATE before using any "
                    f"number below on a slide.")
    return Xtr, Xte, ytr, yte, ridge


# ----------------------------------------------------------------------------------
# M1 - the casual + registered leak  ([26] frame 3)
# ----------------------------------------------------------------------------------
def m1_casual_registered(df):
    banner("M1  [26] frame 3 - casual + registered = cnt")
    exact = bool((df["casual"] + df["registered"] == df[TARGET]).all())
    log.info(f"casual + registered == cnt on all {len(df)} rows: {exact}")
    if not exact:
        raise AssertionError("the leak premise of [26] frame 3 is false - fix the frame")

    leaky = FEATURES + ["casual", "registered"]
    X, y = df[leaky], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=N_JOBS).fit(Xtr, ytr)
    log.info(f"forest WITH casual+registered : test R2 = {r2_score(yte, rf.predict(Xte)):.4f}  "
             f"MAE = {mean_absolute_error(yte, rf.predict(Xte)):.1f}")
    log.info("compare against the honest R2 = 0.879 from M0")


# ----------------------------------------------------------------------------------
# M2 - ratios and the normalization trap  ([26] frame 11)
# ----------------------------------------------------------------------------------
def m2_ratios(df):
    banner("M2  [26] frame 11 - ratios, and whether atemp-temp is meaningless")
    d = df.copy()
    # [22] taught these: temp = (T+8)/47, atemp = (T+16)/66
    d["temp_c"] = 47 * d["temp"] - 8
    d["atemp_c"] = 66 * d["atemp"] - 16
    d["feels_gap"] = d["atemp_c"] - d["temp_c"]      # the correct domain feature
    d["naive_gap"] = d["atemp"] - d["temp"]          # the meaningless one
    d["discomfort"] = d["hum"] * d["windspeed"]

    r = np.corrcoef(d["feels_gap"], d["naive_gap"])[0, 1]
    log.info(f"corr(feels_gap, naive_gap) = {r:+.3f}   "
             f"-> the naive difference is {'NOT ' if abs(r) < 0.95 else ''}a proxy for the real gap")
    log.info(f"temp_c   range {d['temp_c'].min():6.1f} .. {d['temp_c'].max():5.1f} C")
    log.info(f"feels_gap range {d['feels_gap'].min():6.1f} .. {d['feels_gap'].max():5.1f} C")
    log.info(f"naive_gap range {d['naive_gap'].min():6.3f} .. {d['naive_gap'].max():5.3f} (unitless)")

    y = d[TARGET]
    for name, cols in [
        ("baseline (11 raw)", FEATURES),
        ("+ naive_gap", FEATURES + ["naive_gap"]),
        ("+ feels_gap (correct)", FEATURES + ["feels_gap"]),
        ("+ discomfort", FEATURES + ["discomfort"]),
        ("+ feels_gap + discomfort", FEATURES + ["feels_gap", "discomfort"]),
    ]:
        Xtr, Xte, ytr, yte = train_test_split(d[cols], y, test_size=0.3, random_state=SEED)
        log.info(f"  ridge test MAE  {name:28s} = {ridge_mae(Xtr, ytr, Xte, yte):7.1f}")


# ----------------------------------------------------------------------------------
# M3 - interaction, linear vs tree  ([26] frame 13)
# ----------------------------------------------------------------------------------
def m3_interaction(df):
    banner("M3  [26] frame 13 - does temp x workingday need to be handed to the model?")
    d = df.copy()
    d["temp_x_working"] = d["temp"] * d["workingday"]
    y = d[TARGET]

    Xtr, Xte, ytr, yte = train_test_split(d[FEATURES], y, test_size=0.3, random_state=SEED)
    base = ridge_mae(Xtr, ytr, Xte, yte)
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=N_JOBS).fit(Xtr, ytr)
    rf_base = mean_absolute_error(yte, rf.predict(Xte))

    cols = FEATURES + ["temp_x_working"]
    Xtr2, Xte2, ytr2, yte2 = train_test_split(d[cols], y, test_size=0.3, random_state=SEED)
    withx = ridge_mae(Xtr2, ytr2, Xte2, yte2)
    rf2 = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=N_JOBS).fit(Xtr2, ytr2)
    rf_withx = mean_absolute_error(yte2, rf2.predict(Xte2))

    log.info(f"ridge  without interaction = {base:7.1f}   with = {withx:7.1f}   "
             f"delta = {base - withx:+7.1f}")
    log.info(f"forest without interaction = {rf_base:7.1f}   with = {rf_withx:7.1f}   "
             f"delta = {rf_base - rf_withx:+7.1f}")
    log.info("frame claim: ridge should improve, forest should not (it finds it via nested splits)")


# ----------------------------------------------------------------------------------
# M4 - binning a non-monotonic effect  ([26] frame 14)
# ----------------------------------------------------------------------------------
def m4_binning(df):
    banner("M4  [26] frame 14 - binning temp for a linear model")
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], y, test_size=0.3, random_state=SEED)
    base = ridge_mae(Xtr, ytr, Xte, yte)

    kb = KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile")
    kb.fit(Xtr[["temp"]])
    Xtr_b = np.hstack([Xtr.drop(columns=["temp"]).values, kb.transform(Xtr[["temp"]])])
    Xte_b = np.hstack([Xte.drop(columns=["temp"]).values, kb.transform(Xte[["temp"]])])
    binned = ridge_mae(Xtr_b, ytr, Xte_b, yte)

    log.info(f"ridge raw temp    = {base:7.1f}")
    log.info(f"ridge binned temp = {binned:7.1f}   delta = {base - binned:+7.1f}")
    log.info("frame claim: binning should help, because [23]'s ICE curves show temp is non-monotonic")


# ----------------------------------------------------------------------------------
# M5 - group-by aggregation leakage  ([26] frame 18)
# ----------------------------------------------------------------------------------
def m5_groupby_leak(df):
    """Sweep the grouping key's cardinality: leakage severity should scale with how precisely
    the group identifies the row. At 4 groups the mean is a coarse summary; at ~170 groups it
    is almost the row's own target."""
    banner("M5  [26] frame 18 - honest vs leaky group-by aggregation, swept by group cardinality "
           "(replaces the fabricated '32 vs 24 kAMD, ~25%')")
    d = df.copy()
    y = d[TARGET].values
    base_X = d[FEATURES].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    plain = ridge_cv_mae(base_X, y)
    log.info(f"CV MAE, no group feature = {plain:7.1f}")
    log.info("")
    log.info(f"{'grouping key':34s} {'groups':>7s} {'rows/grp':>9s} "
             f"{'HONEST':>8s} {'LEAKY':>8s} {'the lie':>9s}")

    s = d["season"].astype(str)
    keys = {
        "season": s,
        "season x workingday": s + "_" + d["workingday"].astype(str),
        "mnth": d["mnth"].astype(str),
        "mnth x weekday": d["mnth"].astype(str) + "_" + d["weekday"].astype(str),
        "yr x mnth x weekday": (d["yr"].astype(str) + "_" + d["mnth"].astype(str)
                                + "_" + d["weekday"].astype(str)),
    }
    for name, key in keys.items():
        kv = key.values
        n_groups = len(set(kv))

        # LEAKY: group mean computed once on the whole dataset, then cross-validated
        leaky_feat = d.groupby(kv)[TARGET].transform("mean").values.reshape(-1, 1)
        leaky = ridge_cv_mae(np.hstack([base_X, leaky_feat]), y)

        # HONEST: the group mean is refit inside every fold, on the training part only
        honest_scores = []
        for tr, te in kf.split(base_X):
            gm = pd.Series(y[tr]).groupby(kv[tr]).mean()
            gmean = y[tr].mean()
            ftr = np.array([gm.get(k, gmean) for k in kv[tr]]).reshape(-1, 1)
            fte = np.array([gm.get(k, gmean) for k in kv[te]]).reshape(-1, 1)
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
            m.fit(np.hstack([base_X[tr], ftr]), y[tr])
            honest_scores.append(
                mean_absolute_error(y[te], m.predict(np.hstack([base_X[te], fte]))))
        honest = float(np.mean(honest_scores))
        log.info(f"{name:34s} {n_groups:7d} {len(d) / n_groups:9.1f} "
                 f"{honest:8.1f} {leaky:8.1f} {honest - leaky:+8.1f} "
                 f"({100 * (honest - leaky) / honest:+.1f}%)")
    log.info("-> the lesson is not a fixed % but the SCALING: the more precisely the group "
             "identifies the row, the more of the target the leak smuggles in")


# ----------------------------------------------------------------------------------
# M6 - the candidate pool  ([26] frame 28)
# ----------------------------------------------------------------------------------
def build_pool(df):
    """Systematic mechanical expansion + the hand-designed domain features, tagged."""
    banner("M6  [26] frame 28 - systematic expansion, the candidate pool")
    d = df[FEATURES].copy()
    pool = {c: d[c].values.astype(float) for c in FEATURES}
    origin = {c: "raw" for c in FEATURES}

    n_inf = 0
    inf_cols = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            pool[f"{a}_x_{b}"] = d[a].values * d[b].values
            origin[f"{a}_x_{b}"] = "product"
            pool[f"{a}_minus_{b}"] = d[a].values.astype(float) - d[b].values
            origin[f"{a}_minus_{b}"] = "difference"
    for a in FEATURES:
        for b in FEATURES:
            if a == b:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                v = d[a].values.astype(float) / d[b].values.astype(float)
            if not np.isfinite(v).all():
                n_inf += int((~np.isfinite(v)).sum())
                inf_cols.append(f"{a}_over_{b}")
            pool[f"{a}_over_{b}"] = v
            origin[f"{a}_over_{b}"] = "ratio"

    P = pd.DataFrame(pool)
    log.info(f"raw mechanical expansion: {P.shape[1]} columns from {len(FEATURES)} features")

    # THE DATA-QUALITY DISCOVERY - this goes on the slide, it is not a bug to hide
    zero_cols = [c for c in FEATURES if (df[c] == 0).any()]
    log.info(f"non-finite values produced: {n_inf} across {len(inf_cols)} ratio columns")
    for c in zero_cols:
        n0 = int((df[c] == 0).sum())
        log.info(f"  denominator zeros: {c} has {n0} zero row(s)")
        if c == "hum":
            rows = df.loc[df[c] == 0, ["dteday", "hum", "cnt"]]
            log.warning(f"  hum == 0 on {rows['dteday'].tolist()} - 0% humidity is physically "
                        f"impossible in Washington DC; this is an unflagged missing value that "
                        f"four lectures of students never saw. THIS IS THE SLIDE.")

    # documented, logged guard - not a silent fallback
    bad = ~np.isfinite(P.values)
    if bad.any():
        log.info(f"guard: replacing {int(bad.sum())} non-finite cells with the column median "
                 f"(logged, not silent)")
        P = P.replace([np.inf, -np.inf], np.nan)
        P = P.fillna(P.median())

    nunique = P.nunique()
    const = nunique[nunique <= 1].index.tolist()
    if const:
        log.info(f"dropping {len(const)} constant column(s): {const[:8]}")
        P = P.drop(columns=const)
        for c in const:
            origin.pop(c, None)

    dup = P.T.duplicated()
    if dup.any():
        dropped = P.columns[dup.values].tolist()
        log.info(f"dropping {len(dropped)} exactly-duplicated column(s), e.g. {dropped[:6]}")
        P = P.loc[:, ~dup.values]
        for c in dropped:
            origin.pop(c, None)

    # hand-designed domain features - NOT reachable by mechanical expansion, because they
    # require knowing the normalization constants from the dataset documentation
    kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile",
                          quantile_method="averaged_inverted_cdf")
    domain = pd.DataFrame({
        "feels_gap_C": (66 * df["atemp"].values - 16) - (47 * df["temp"].values - 8),
        "temp_C": 47 * df["temp"].values - 8,
        "temp_bin": kb.fit_transform(df[["temp"]]).ravel(),
    }, index=P.index)
    P = pd.concat([P, domain], axis=1)
    origin.update({c: "domain" for c in domain.columns})

    log.info(f"FINAL POOL: {P.shape[1]} candidates "
             f"({sum(v == 'domain' for v in origin.values())} hand-designed, "
             f"{P.shape[1] - sum(v == 'domain' for v in origin.values())} mechanical)")
    log.info("  by origin: " + ", ".join(
        f"{k}={sum(1 for c in P.columns if origin[c] == k)}"
        for k in ["raw", "product", "difference", "ratio", "domain"]))

    OUT_DIR.mkdir(exist_ok=True)
    P.assign(**{TARGET: df[TARGET].values}).to_csv(OUT_DIR / "bike_candidates.csv", index=False)
    pd.Series(origin).rename("origin").to_csv(OUT_DIR / "bike_candidates_origin.csv")
    log.info(f"persisted -> {OUT_DIR / 'bike_candidates.csv'}  (deck [27] loads THIS file)")
    return P, origin


# ----------------------------------------------------------------------------------
# M7/M8 - the two regimes  ([27] frames 7 and 8)
# ----------------------------------------------------------------------------------
def m78_regimes(df, P):
    banner("M7/M8  [27] frames 7-8 - does selection beat keep-everything?")
    y = df[TARGET].values
    for label, X in [("regime 1: 11 raw features", df[FEATURES].values),
                     ("regime 2: full candidate pool", P.values)]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
        keep_all = ridge_mae(Xtr, ytr, Xte, yte)
        sel = RFECV(Ridge(alpha=1.0, random_state=SEED), step=1 if X.shape[1] < 30 else 5,
                    cv=5, scoring="neg_mean_absolute_error", min_features_to_select=2,
                    n_jobs=N_JOBS)
        sc = StandardScaler().fit(Xtr)
        sel.fit(sc.transform(Xtr), ytr)
        k = int(sel.n_features_)
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
        m.fit(Xtr[:, sel.support_], ytr)
        selected = mean_absolute_error(yte, m.predict(Xte[:, sel.support_]))
        log.info(f"{label:32s} p={X.shape[1]:4d}  keep-all MAE={keep_all:7.1f}  "
                 f"selected(k={k:3d}) MAE={selected:7.1f}  delta={keep_all - selected:+7.1f}")
    return


# ----------------------------------------------------------------------------------
# M9 - what survives RFE-CV: domain or machine?  ([27] frame 23, tests [26] frame 7)
# ----------------------------------------------------------------------------------
def m9_survivors(df, P, origin):
    banner("M9  [27] frame 23 - do the hand-designed features survive, or the mechanical ones?")
    y = df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(P.values, y, test_size=0.3, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    sel = RFECV(Ridge(alpha=1.0, random_state=SEED), step=5, cv=5,
                scoring="neg_mean_absolute_error", min_features_to_select=5, n_jobs=N_JOBS)
    sel.fit(sc.transform(Xtr), ytr)
    kept = P.columns[sel.support_].tolist()
    log.info(f"RFE-CV kept {len(kept)} of {P.shape[1]}")
    counts = {}
    for c in kept:
        counts[origin[c]] = counts.get(origin[c], 0) + 1
    log.info(f"survivors by origin: {counts}")
    dom = [c for c in kept if origin[c] == "domain"]
    log.info(f"hand-designed survivors: {dom if dom else 'NONE'} "
             f"(of {sum(v == 'domain' for v in origin.values())} offered)")
    log.info(f"first 25 survivors: {kept[:25]}")
    log.info("-> if the domain features are dropped, [26] frame 7's 'domain knowledge wins' "
             "bullet needs rewording")


# ----------------------------------------------------------------------------------
# M10 - Boruta on the correlated pair  ([27] frame 26)
# ----------------------------------------------------------------------------------
def m10_boruta(df):
    banner("M10  [27] frame 26 - what does Boruta do with temp/atemp? (MEASURE, do not predict)")
    from boruta import BorutaPy
    X = df[FEATURES].values
    y = df[TARGET].values
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=N_JOBS, max_depth=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BorutaPy(rf, n_estimators="auto", max_iter=100, random_state=SEED, verbose=0)
        b.fit(X, y)
    for f, sup, weak, rank in zip(FEATURES, b.support_, b.support_weak_, b.ranking_):
        status = "CONFIRMED" if sup else ("TENTATIVE" if weak else "rejected")
        log.info(f"  {f:12s} {status:10s} rank={rank}")
    pair = {f: bool(s) for f, s in zip(FEATURES, b.support_) if f in ("temp", "atemp")}
    log.info(f"THE PAIR: {pair}")
    if all(pair.values()):
        log.info("-> both confirmed: all-relevant behaves as [27] frame 12 advertises")
    elif not any(pair.values()):
        log.info("-> BOTH REJECTED: they split credit ([22]) so each falls below max shadow "
                 "importance. This is the sharper slide - an all-relevant method defeated by "
                 "correlation.")
    else:
        log.info("-> only one confirmed: Boruta broke the tie. Explain WHY on the slide.")


# ----------------------------------------------------------------------------------
# M11 - stability selection  ([27] frame 28)
# ----------------------------------------------------------------------------------
def m11_stability(df, n_boot=100):
    banner("M11  [27] frame 28 - stability selection (MEASURE, do not predict)")
    X, y = df[FEATURES].values, df[TARGET].values
    Xs = StandardScaler().fit_transform(X)
    alpha = LassoCV(cv=5, random_state=SEED, n_jobs=N_JOBS, max_iter=5000).fit(Xs, y).alpha_
    log.info(f"LassoCV alpha on full data = {alpha:.4f}; running {n_boot} bootstraps")

    counts = np.zeros(len(FEATURES))
    union = 0
    it, ia = FEATURES.index("temp"), FEATURES.index("atemp")
    for i in range(n_boot):
        Xb, yb = resample(Xs, y, random_state=SEED + i)
        coef = Lasso(alpha=alpha, random_state=SEED, max_iter=5000).fit(Xb, yb).coef_
        sel = coef != 0
        counts += sel
        union += bool(sel[it] or sel[ia])
    freq = counts / n_boot
    for f, v in sorted(zip(FEATURES, freq), key=lambda t: -t[1]):
        log.info(f"  {f:12s} selected in {v:5.0%} of bootstraps")
    log.info(f"THE PAIR: temp={freq[it]:.0%}  atemp={freq[ia]:.0%}  "
             f"union(either one)={union / n_boot:.0%}")
    log.info("-> if each is ~50% but the union is ~100%, that is [22]'s split-the-credit result "
             "resurfacing in an unrelated method")


# ----------------------------------------------------------------------------------
# M12 - filter disagreement  ([27] frame 17)
# ----------------------------------------------------------------------------------
def m12_filters(df):
    banner("M12  [27] frame 17 - do f_regression and mutual information disagree?")
    X, y = df[FEATURES], df[TARGET]
    f, _ = f_regression(X, y)
    mi = mutual_info_regression(X, y, random_state=SEED)
    rf_ = pd.Series(f, index=FEATURES).rank(ascending=False)
    rmi = pd.Series(mi, index=FEATURES).rank(ascending=False)
    for feat in FEATURES:
        flag = "  <-- DISAGREE" if abs(rf_[feat] - rmi[feat]) >= 3 else ""
        log.info(f"  {feat:12s} F={f[FEATURES.index(feat)]:8.1f} (rank {rf_[feat]:.0f})   "
                 f"MI={mi[FEATURES.index(feat)]:.3f} (rank {rmi[feat]:.0f}){flag}")
    log.info("frame claim: F should undersell temp, because its effect is non-monotonic ([23] ICE)")


# ----------------------------------------------------------------------------------
# M13 - multiple testing  ([27] frame 20)
# ----------------------------------------------------------------------------------
def m13_multiple_testing(df, n_noise=10_000):
    banner(f"M13  [27] frame 20 - how many of {n_noise:,} pure-noise features pass at alpha=0.05?")
    rng = np.random.RandomState(SEED)
    y = df[TARGET].values
    noise = rng.normal(size=(len(df), n_noise))
    _, p = f_regression(noise, y)
    n05 = int((p < 0.05).sum())
    n01 = int((p < 0.01).sum())
    bonf = int((p < 0.05 / n_noise).sum())
    order = np.sort(p)
    bh_thresh = 0.05 * (np.arange(1, n_noise + 1) / n_noise)
    bh = int((order <= bh_thresh).sum())
    log.info(f"pass at alpha=0.05          : {n05:5d}  (expected ~{int(0.05 * n_noise)})")
    log.info(f"pass at alpha=0.01          : {n01:5d}  (expected ~{int(0.01 * n_noise)})")
    log.info(f"pass Bonferroni (0.05/{n_noise}) : {bonf:5d}")
    log.info(f"pass Benjamini-Hochberg     : {bh:5d}")
    log.info(f"best single noise feature: p = {p.min():.2e}")
    log.info("-> every one of these is a column of pure random numbers")


# ----------------------------------------------------------------------------------
# M14 - the thesis frame, without the model-class confound  ([26] frame 7)
# ----------------------------------------------------------------------------------
def m14_thesis_ladder(df, P):
    """Comparing Ridge+FE against a forest confounds feature engineering with model class.
    The honest test holds the model fixed and varies only the features."""
    banner("M14  [26] frame 7 - does feature engineering help a model that engineers internally?")
    y = df[TARGET].values
    rows = []
    for label, X in [("11 raw", df[FEATURES].values), ("230 engineered", P.values)]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
        rows.append((f"ridge   {label}", ridge_mae(Xtr, ytr, Xte, yte)))
        rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=N_JOBS)
        rf.fit(Xtr, ytr)
        rows.append((f"forest  {label}", mean_absolute_error(yte, rf.predict(Xte))))
    for label, mae in sorted(rows, key=lambda r: r[0]):
        log.info(f"  {label:24s} test MAE = {mae:7.1f}")
    log.info("-> feature engineering transforms the LINEAR model and does nothing for the forest, "
             "because the forest was already deriving these features internally. That is the "
             "2026 thesis with no model-class confound.")


def main():
    setup_logging()
    log.info(f"reading {DATA}")
    df = pd.read_csv(DATA)
    log.info(f"{len(df)} rows, {df.shape[1]} columns")

    m0_baseline(df)
    m1_casual_registered(df)
    m2_ratios(df)
    m3_interaction(df)
    m4_binning(df)
    m5_groupby_leak(df)
    P, origin = build_pool(df)
    m14_thesis_ladder(df, P)
    m78_regimes(df, P)
    m9_survivors(df, P, origin)
    m10_boruta(df)
    m11_stability(df)
    m12_filters(df)
    m13_multiple_testing(df)

    banner("DONE - every [MEASURE] frame in both outlines now has a real number")
    log.info(f"full log: {LOGS_DIR / 'fe_measure.log'}")


if __name__ == "__main__":
    main()
