"""Generate the real-data figures for the [18] Random Forests deck.

Produces four PDFs into ``ml/04_trees/fig/`` from the Titanic dataset:
  1. rf_instability.pdf   -- % of test predictions that change when a single
                             (unrestricted) tree is refit on bootstrap resamples.
                             Motivates the whole deck: one tree is high-variance.
  2. rf_n_estimators.pdf  -- test accuracy + OOB score vs n_estimators -> the
                             "more trees plateau, they do NOT overfit" curve.
  3. rf_importance.pdf    -- default impurity-based RF feature_importances_, with
                             value labels on the bars (per repo CLAUDE.md).
  (plus oob_fraction, variance_floor, oob_vs_cv, rf_max_features, rf_boundary_smoothing.)

Run with the project venv (repo CLAUDE.md -- do NOT spin up an ephemeral env):
    ./ma/Scripts/python.exe ml/04_trees/py_src/make_rf_figures.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours for multi-line plots, value labels on bars, n_jobs=1 to
avoid pegging all cores on this machine.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.datasets import fetch_openml, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 509

# Armenian flag palette (per CLAUDE.md) -- blue, red, orange.
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]               # ml/04_trees
REPO_ROOT = HERE.parents[3]            # repo root
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l10_figures")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "make_rf_figures.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def load_titanic(logger: logging.Logger):
    """Return (X, y, feature_names) -- same tiny preprocessing as the [17] script."""
    df = fetch_openml("titanic", version=1, as_frame=True).frame
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    df = df[features + ["survived"]].copy()
    df["sex"] = (df["sex"] == "female").astype(int)        # female=1, male=0
    df["pclass"] = df["pclass"].astype(float)
    df["age"] = df["age"].fillna(df["age"].median())        # impute (median)
    df["fare"] = df["fare"].fillna(df["fare"].median())
    X = df[features].astype(float).to_numpy()
    y = df["survived"].astype(int).to_numpy()
    logger.info(f"Titanic: {X.shape[0]} rows, {X.shape[1]} features, "
                f"survived rate = {y.mean():.3f}")
    return X, y, features


def fig_instability(X, y, feature_names, logger, n_resamples=15):
    """How twitchy is one tree? Refit an unrestricted tree on bootstrap resamples
    of the training set and count the fraction of TEST predictions that flip vs the
    tree trained on the original sample."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    base = DecisionTreeClassifier(random_state=SEED).fit(X_tr, y_tr)
    base_pred = base.predict(X_te)

    rng = np.random.default_rng(SEED)
    flips = []
    for _ in range(n_resamples):
        idx = rng.choice(len(X_tr), len(X_tr), replace=True)   # bootstrap
        t = DecisionTreeClassifier(random_state=SEED).fit(X_tr[idx], y_tr[idx])
        flips.append(np.mean(t.predict(X_te) != base_pred) * 100.0)
    flips = np.array(flips)
    logger.info(f"instability: single-tree flip rate mean={flips.mean():.1f}% "
                f"(min {flips.min():.1f}, max {flips.max():.1f}) over "
                f"{n_resamples} resamples")

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(np.arange(1, n_resamples + 1), flips, color=ARM_BLUE, alpha=0.85)
    ax.axhline(flips.mean(), color=ARM_RED, ls="--", lw=1.8,
               label=f"mean {flips.mean():.0f}% of predictions flip")
    ax.set_xlabel("refit on a different bootstrap resample")
    ax.set_ylabel("% of test predictions\nthat changed")
    ax.set_xticks(np.arange(1, n_resamples + 1))
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, flips.max() * 1.35)          # headroom so the legend clears the bars
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "rf_instability.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")
    return dict(mean_flip=float(flips.mean()))


def fig_n_estimators(X, y, feature_names, logger):
    """Test accuracy + OOB score vs number of trees -> rises then plateaus."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    ns = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300]
    test_acc, oob = [], []
    for n in ns:
        rf = RandomForestClassifier(n_estimators=n, oob_score=True,
                                    bootstrap=True, random_state=SEED, n_jobs=1)
        rf.fit(X_tr, y_tr)
        test_acc.append(rf.score(X_te, y_te))
        # oob_score_ is ill-defined / warns for very few trees -> NaN below ~10
        oob.append(rf.oob_score_ if n >= 10 else np.nan)
    test_acc = np.array(test_acc)
    logger.info(f"n_estimators: test acc 1-tree={test_acc[0]:.3f}, "
                f"300-tree={test_acc[-1]:.3f}, max={test_acc.max():.3f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(ns, test_acc, "-o", color=ARM_RED, lw=2, ms=3.5, label="test accuracy")
    ax.plot(ns, oob, "-s", color=ARM_BLUE, lw=2, ms=3.5, label="OOB score")
    ax.axvspan(100, 300, color=ARM_ORANGE, alpha=0.12)
    ax.annotate("plateau: more trees\ndon't overfit", xy=(180, test_acc[-1]),
                xytext=(120, test_acc.min() + 0.25 * (test_acc.max() - test_acc.min())),
                color=ARM_ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xscale("log")
    ax.set_xlabel("number of trees (n_estimators, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "rf_n_estimators.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")
    return dict(acc_1=float(test_acc[0]), acc_max=float(test_acc.max()))


def fig_importance(X, y, feature_names, logger):
    """Default impurity-based RF feature importances, sorted, with value labels."""
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                n_jobs=1).fit(X, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)                       # ascending -> barh bottom-up
    disp = ["gender" if f == "sex" else f for f in feature_names]   # relabel sex -> gender
    names = [disp[i] for i in order]
    vals = imp[order]
    logger.info("importance: " + ", ".join(
        f"{feature_names[i]}={imp[i]:.3f}" for i in np.argsort(imp)[::-1]))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.barh(names, vals, color=ARM_BLUE, alpha=0.85, edgecolor=ARM_BLUE)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xlabel("impurity-based importance")
    ax.set_xlim(0, max(vals) * 1.18)              # right padding for labels
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "rf_importance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_max_features(logger):
    """The decorrelation knob, made visible. As max_features drops, the trees'
    predictions get less correlated (rho falls -- the rho*sigma^2 floor term the deck
    derives); test error is a gentler tradeoff. Synthetic set with one dominant feature,
    so plain bagging would otherwise make every tree split on it and correlate.
    (Titanic has too few features to show this cleanly; the error effect is genuinely
    small and data-dependent, so we show the mechanism rho, not a staged error drop.)"""
    rng = np.random.default_rng(SEED)
    n = 2500
    Xs = rng.normal(size=(n, 25))
    logit = (1.7 * Xs[:, 0]                       # one dominant feature
             + 0.55 * Xs[:, 1:5].sum(axis=1)
             + 0.35 * Xs[:, 5:9].sum(axis=1))
    ys = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, ys, test_size=0.3, random_state=SEED, stratify=ys)

    mfs = [1, 2, 3, 5, 8, 12, 16, 20, 25]
    rho, err = [], []
    for mf in mfs:
        rf = RandomForestClassifier(n_estimators=150, max_features=mf,
                                    random_state=SEED, n_jobs=1).fit(X_tr, y_tr)
        P = np.array([t.predict_proba(X_te)[:, 1] for t in rf.estimators_])
        C = np.corrcoef(P)
        rho.append(float(C[np.triu_indices_from(C, k=1)].mean()))
        err.append(1.0 - rf.score(X_te, y_te))
    logger.info(f"max_features rho={[round(r,3) for r in rho]}")
    logger.info(f"max_features err={[round(e,3) for e in err]}")

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(mfs, rho, "-o", color=ARM_BLUE, lw=2, ms=4,
            label=r"tree-to-tree correlation $\rho$")
    ax.axvline(5, color=ARM_ORANGE, ls="--", lw=1.5)
    ax.annotate(r"sqrt$\approx$5 (default)", xy=(5, rho[3]), xytext=(6.5, 0.12),
                color=ARM_ORANGE, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xlabel("max_features (features tried per split, of 25)")
    ax.set_ylabel(r"mean tree-to-tree correlation $\rho$", color=ARM_BLUE)
    ax.tick_params(axis="y", labelcolor=ARM_BLUE)
    ax.set_ylim(0, 0.32)
    ax2 = ax.twinx()
    ax2.plot(mfs, err, "-s", color=ARM_RED, lw=2, ms=4, label="test error")
    ax2.set_ylabel("test error (1 - accuracy)", color=ARM_RED)
    ax2.tick_params(axis="y", labelcolor=ARM_RED)
    lines = ax.get_lines()[:1] + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left",
              fontsize=8, frameon=False)
    for a in (ax, ax2):
        a.spines[["top"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "rf_max_features.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_boundary_smoothing(logger):
    """One tree's blocky axis-aligned boundary vs a forest's smoother averaged one,
    on the same 2D data (make_moons). Pairs with the [17] staircase figure."""
    Xs, ys = make_moons(n_samples=400, noise=0.30, random_state=SEED)
    tree = DecisionTreeClassifier(random_state=SEED).fit(Xs, ys)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1).fit(Xs, ys)
    x0min, x0max = Xs[:, 0].min() - 0.4, Xs[:, 0].max() + 0.4
    x1min, x1max = Xs[:, 1].min() - 0.4, Xs[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x0min, x0max, 400), np.linspace(x1min, x1max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharex=True, sharey=True)
    for ax, model, title in zip(axes, [tree, rf],
                                ["single tree: blocky", "random forest (300): smoother"]):
        zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.65)
        ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=10, color=ARM_BLUE,
                   edgecolor="k", linewidths=0.2, alpha=0.85)
        ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=10, color=ARM_RED,
                   edgecolor="k", linewidths=0.2, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = FIG_DIR / "rf_boundary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_oob_fraction(logger):
    """(1 - 1/n)^n -> 1/e: the fraction of rows a bootstrap leaves out converges to
    ~37% as n grows. Explains the OOB '37%' number."""
    import math
    ns = np.unique(np.round(np.logspace(np.log10(3), np.log10(500), 60)).astype(int))
    frac = (1 - 1 / ns) ** ns
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(ns, frac, color=ARM_BLUE, lw=2, zorder=3)
    ax.axhline(1 / math.e, color=ARM_RED, ls="--", lw=1.5,
               label=r"$1/e \approx 0.368$")
    for nn in (6, 20, 100):
        f = (1 - 1 / nn) ** nn
        ax.scatter([nn], [f], color=ARM_ORANGE, s=42, zorder=4)
        ax.annotate(f"n={nn}: {f:.3f}", (nn, f), textcoords="offset points",
                    xytext=(7, -3), fontsize=8, color="0.25")
    ax.set_xscale("log")
    ax.set_xlabel("n (training rows)")
    ax.set_ylabel(r"fraction left out  $(1-1/n)^n$")
    ax.set_ylim(0.28, 0.40)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "oob_fraction.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_variance_floor(X, y, logger):
    """Empirical Var(f_bar) = rho*sigma^2 + (1-rho)/M * sigma^2: retrain the forest on many
    independent training samples and watch the variance of its prediction fall with M, then
    flatten at a positive floor (rho*sigma^2). Uses the first-M-trees trick, so one fit per
    run gives the whole curve."""
    from sklearn.model_selection import train_test_split
    X_pool, X_te, y_pool, y_te = train_test_split(
        X, y, test_size=300, random_state=SEED, stratify=y)
    rng = np.random.default_rng(SEED)
    B, n_trees = 40, 60
    Ms = np.array([1, 2, 3, 5, 8, 13, 21, 34, 60])
    per_run = []                                   # each: (n_trees, n_test) survival proba
    for b in range(B):
        idx = rng.integers(0, len(X_pool), len(X_pool))          # independent training sample
        rf = RandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                    random_state=b, n_jobs=1).fit(X_pool[idx], y_pool[idx])
        per_run.append(np.array([t.predict_proba(X_te)[:, 1] for t in rf.estimators_]))
    per_run = np.array(per_run)                    # (B, n_trees, n_test)
    var_by_M = np.array([per_run[:, :M, :].mean(axis=1).var(axis=0).mean() for M in Ms])
    floor = var_by_M[-1]
    logger.info("variance_floor: sigma^2(M=1)=%.4f -> floor(M=%d)=%.4f",
                var_by_M[0], Ms[-1], floor)

    fig, ax = plt.subplots(figsize=(5.9, 3.9))
    ax.plot(Ms, var_by_M, "-o", color=ARM_BLUE, lw=2, ms=4, zorder=3,
            label=r"Var of the forest's prediction")
    ax.axhline(floor, color=ARM_RED, ls="--", lw=1.5,
               label=r"floor $\approx \rho\sigma^2$")
    ax.annotate(r"one tree ($\sigma^2$)", (Ms[0], var_by_M[0]),
                textcoords="offset points", xytext=(8, -2), fontsize=8, color="0.25")
    ax.set_xscale("log")
    ax.set_xlabel("M (trees in the forest)")
    ax.set_ylabel("variance of the prediction")
    ax.set_ylim(0, var_by_M[0] * 1.1)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "variance_floor.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_oob_vs_cv(X, y, logger):
    """OOB error tracks 5-fold CV error across max_features -- 'free validation'."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    # NB: shuffle the folds -- OpenML Titanic is ordered by pclass, so the default
    # non-shuffled folds are badly biased (and OOB, being order-agnostic, would not match).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    P = X.shape[1]
    mf_vals = [1, 2, 3, 4, P]                       # features tried per split
    oob_err, cv_err = [], []
    for mf in mf_vals:
        rf = RandomForestClassifier(n_estimators=150, max_features=mf, oob_score=True,
                                    random_state=SEED, n_jobs=1).fit(X, y)
        oob_err.append(1 - rf.oob_score_)
        base = RandomForestClassifier(n_estimators=150, max_features=mf,
                                      random_state=SEED, n_jobs=1)
        cv_err.append(1 - cross_val_score(base, X, y, cv=cv).mean())
    logger.info("oob_vs_cv: oob=%s cv=%s",
                [f"{e:.3f}" for e in oob_err], [f"{e:.3f}" for e in cv_err])

    fig, ax = plt.subplots(figsize=(5.9, 3.9))
    ax.plot(mf_vals, oob_err, "-o", color=ARM_BLUE, lw=2, ms=5, label="OOB error (free)")
    ax.plot(mf_vals, cv_err, "-s", color=ARM_RED, lw=2, ms=5, label="5-fold CV error")
    ax.set_xlabel("max\\_features (features tried per split)")
    ax.set_ylabel("error rate")
    ax.set_xticks(mf_vals)
    ax.set_ylim(0.10, 0.30)                         # avoid a misleading zoom: they nearly coincide
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "oob_vs_cv.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_forest_meme(logger):
    """Outline gag: a forest photo + the Armenian caption 'it's all trees'. Baked into
    the image because pdflatex (this preamble) has no Armenian; the Outline frame then
    hyperlinks the whole image to the video. Armenian written as \\u escapes to keep the
    source ASCII-clean."""
    import matplotlib.image as mpimg
    img = mpimg.imread(str(FIG_DIR / "forest_meme_raw.png"))
    h, w = img.shape[:2]
    # U+25B6 play + "էսի սաղ ծառ ա"
    caption = "▶ էսի սաղ ծառ ա"
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * h / w + 0.7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=21, color=ARM_BLUE, fontweight="bold", pad=8)
    fig.tight_layout()
    out = FIG_DIR / "forest_meme.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def fig_superdataset(logger):
    """The 'superdataset' table: one full training set, and how three trees each grab a
    different RANDOM SLICE of it -- some ROWS (bootstrap, with replacement -> repeats +
    out-of-bag) and, at the root split, some COLUMNS (a random feature menu, sqrt(P)).
    The faded/opaque intersection = the rows x columns the tree's root split is chosen
    from. Draws are a seeded rng sample (509): this IS the real bagging + feature-subset
    mechanism, not a staged result, so no accuracy is claimed. Concrete anchor for the
    otherwise-abstract 'two dice rolls' frame."""
    from matplotlib.patches import Rectangle

    # --- 6 real Titanic rows to make the table concrete (real values, no cherry-pick) ---
    df = fetch_openml("titanic", version=1, as_frame=True).frame
    df = df[["sex", "age", "fare", "pclass", "survived"]].dropna(subset=["age", "fare"])
    rng = np.random.default_rng(SEED)
    samp = df.iloc[rng.choice(len(df), 6, replace=False)].reset_index(drop=True)
    gender = ["F" if s == "female" else "M" for s in samp["sex"]]
    age = [f"{a:.0f}" for a in samp["age"]]
    fare = [f"{f:.0f}" for f in samp["fare"]]
    pcls = [f"{int(p)}" for p in samp["pclass"]]
    surv = [f"{int(s)}" for s in samp["survived"]]
    feat_headers = ["gender", "age", "fare", "class"]
    feat_vals = [gender, age, fare, pcls]                 # column-major: [col][row]
    NR, NF = 6, len(feat_headers)

    # --- the two dice per tree: bootstrap the rows, random root-split feature menu ---
    accents = [ARM_BLUE, ARM_ORANGE, "#0A7D3B"]           # blue / orange / green (no red = warning)
    trees = []
    for _ in range(3):
        draws = rng.integers(0, NR, NR)                   # bootstrap rows, with replacement
        counts = np.bincount(draws, minlength=NR)         # how many times each row was drawn
        menu = sorted(rng.choice(NF, 2, replace=False))   # sqrt(4)=2 features tried at the root
        trees.append((counts, menu))
    logger.info("superdataset rows: " + "; ".join(
        f"r{i+1}={g}/{a}/{fa}/{p}/y{sv}" for i, (g, a, fa, p, sv)
        in enumerate(zip(gender, age, fare, pcls, surv))))
    for k, (counts, menu) in enumerate(trees, 1):
        logger.info(f"  tree{k}: in-bag counts={list(counts)} "
                    f"(OOB rows {[i+1 for i in range(NR) if counts[i]==0]}), "
                    f"root menu={[feat_headers[j] for j in menu]}")

    def draw_block(ax, headers, cols_vals, extra=None, counts=None, menu=None, accent=None):
        """Render one matrix. Master: counts=None (all rows solid, `extra`=(header,vals)
        appends a y column). Tree: counts/menu given -> fade OOB rows and non-menu columns,
        highlight the menu-column headers + the in-bag x menu intersection cells."""
        allh = list(headers) + ([extra[0]] if extra else [])
        allv = list(cols_vals) + ([extra[1]] if extra else [])
        nc = len(allh)
        ax.set_xlim(-1.15, nc + 0.05)
        ax.set_ylim(-0.15, NR + 1.15)
        ax.axis("off")
        ax.set_aspect("equal")
        top = NR + 1                                       # header band sits at y in [NR, NR+1)

        for c, h in enumerate(allh):
            is_menu = (menu is not None and c < NF and c in menu)
            is_y = (extra is not None and c == nc - 1)
            hc = accent if is_menu else ("0.80" if is_y else "0.90")
            ax.add_patch(Rectangle((c, NR), 1, 1, facecolor=hc, edgecolor="white", lw=1.4))
            ax.text(c + 0.5, NR + 0.5, h, ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="white" if is_menu else "0.15")

        for r in range(NR):
            yb = NR - 1 - r                                # row r's y-bottom (row 0 on top)
            oob = (counts is not None and counts[r] == 0)
            # row label (left gutter)
            lbl = f"r{r+1}"
            if counts is not None and counts[r] > 1:
                lbl += f" x{counts[r]}"
            ax.text(-0.12, yb + 0.5, lbl, ha="right", va="center", fontsize=8,
                    color="0.55" if oob else "0.15",
                    fontweight="normal" if oob else "bold")
            if oob:
                ax.text(-0.12, yb + 0.5, "", ha="right", va="center")  # keep spacing
            for c in range(nc):
                is_menu = (menu is not None and c < NF and c in menu)
                if counts is None:                         # master: all solid
                    fc = "#eef1f6" if (extra is not None and c == nc - 1) else "white"
                    alpha, tcol = 1.0, "0.15"
                elif oob:                                  # tree: row not drawn
                    fc, alpha, tcol = "0.94", 1.0, "0.62"
                elif is_menu:                              # in-bag AND a considered column
                    fc, alpha, tcol = accent, 0.16, "0.10"
                else:                                      # in-bag, column not at this split
                    fc, alpha, tcol = "white", 1.0, "0.45"
                ax.add_patch(Rectangle((c, yb), 1, 1, facecolor=fc, alpha=alpha,
                                       edgecolor="0.80", lw=0.8))
                ax.text(c + 0.5, yb + 0.5, allv[c][r], ha="center", va="center",
                        fontsize=8, color=tcol)
            if oob:                                        # "OOB" tag past the last column
                ax.text(nc + 0.02, yb + 0.5, "OOB", ha="left", va="center",
                        fontsize=6.5, style="italic", color="0.55")

    fig = plt.figure(figsize=(12.2, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.28, 1.0, 1.0, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    draw_block(ax0, feat_headers, feat_vals, extra=("y", surv))
    ax0.set_title("the full training set\n(the \"superdataset\")", fontsize=9.5, color="0.15")
    for k, (counts, menu) in enumerate(trees):
        axk = fig.add_subplot(gs[k + 1])
        draw_block(axk, feat_headers, feat_vals, counts=counts, menu=menu, accent=accents[k])
        axk.set_title(f"Tree {k+1}", fontsize=9.5, color=accents[k], fontweight="bold")

    fig.text(0.5, 0.075,
             "colored cells = this tree's root-split feature menu     "
             "faded rows = out-of-bag (not drawn)     \"x2\" = row drawn twice",
             ha="center", va="bottom", fontsize=8, color="0.35")
    fig.text(0.5, 0.02,
             "Rows differ AND the column menu differs  =>  the trees disagree  =>  the "
             r"correlation $\rho$ falls.",
             ha="center", va="bottom", fontsize=8.5, color="0.20", fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.18)
    out = FIG_DIR / "rf_superdataset.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"wrote {out.name}")


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    fig_forest_meme(logger)
    fig_superdataset(logger)
    X, y, feats = load_titanic(logger)
    inst = fig_instability(X, y, feats, logger)
    ne = fig_n_estimators(X, y, feats, logger)
    fig_importance(X, y, feats, logger)
    fig_max_features(logger)
    fig_boundary_smoothing(logger)
    fig_oob_fraction(logger)
    fig_variance_floor(X, y, logger)
    fig_oob_vs_cv(X, y, logger)
    logger.info("=== SUMMARY for the slides ===")
    logger.info(f"single-tree flip rate ~ {inst['mean_flip']:.0f}% of test predictions")
    logger.info(f"RF acc: 1 tree={ne['acc_1']:.2f} -> plateau ~{ne['acc_max']:.2f}")
    logger.info("done.")


if __name__ == "__main__":
    main()
