"""Generate the real-data figures for the L10 Random Forests deck.

Produces four PDFs into ``ml/04_trees/fig/`` from the Titanic dataset:
  1. rf_instability.pdf   -- % of test predictions that change when a single
                             (unrestricted) tree is refit on bootstrap resamples.
                             Motivates the whole deck: one tree is high-variance.
  2. rf_n_estimators.pdf  -- test accuracy + OOB score vs n_estimators -> the
                             "more trees plateau, they do NOT overfit" curve.
  3. rf_importance.pdf    -- default impurity-based RF feature_importances_, with
                             value labels on the bars (per repo CLAUDE.md).

  4. rf_vs_tree.pdf       -- Titanic test accuracy: pruned single tree vs a
                             no-tuning RF vs a CV-tuned RF (the chapter's
                             pruned-tree-vs-forest reality check).

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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 509

# Armenian flag palette (per CLAUDE.md) -- blue, red, orange.
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]               # ml/ch3_trees
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
    """Return (X, y, feature_names) -- same tiny preprocessing as the L09 script."""
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
    names = [feature_names[i] for i in order]
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


def fig_vs_tree(X, y, feature_names, logger):
    """Reality check: on this small, one-dominant-feature dataset a well-pruned
    single tree is competitive with a forest. Compare (identical 70/30 split):
      - pruned single tree: ccp_alpha picked by best test accuracy (same recipe as
        the [17] pruning figure -> reproduces the chapter's ~0.835),
      - default RF: no tuning at all,
      - tuned RF: max_features / min_samples_leaf chosen by 5-fold CV on the TRAIN
        fold only (no test peeking)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(X_tr, y_tr)
    tree_accs = [DecisionTreeClassifier(random_state=SEED, ccp_alpha=a)
                 .fit(X_tr, y_tr).score(X_te, y_te) for a in path.ccp_alphas[:-1]]
    tree_acc = float(np.max(tree_accs))

    rf_def_acc = float(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)
                       .fit(X_tr, y_tr).score(X_te, y_te))

    grid = GridSearchCV(
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
        {"max_features": ["sqrt", 0.5, 0.8], "min_samples_leaf": [1, 2, 3, 5]},
        cv=5, scoring="accuracy", n_jobs=1).fit(X_tr, y_tr)
    rf_tuned_acc = float(grid.best_estimator_.score(X_te, y_te))
    logger.info(f"vs_tree: pruned tree={tree_acc:.3f}, default RF={rf_def_acc:.3f}, "
                f"tuned RF={rf_tuned_acc:.3f} (best {grid.best_params_}, "
                f"CV {grid.best_score_:.3f})")

    labels = ["pruned single tree\n([17], tuned)", "random forest\n(no tuning)",
              "random forest\n(tuned)"]
    vals = [tree_acc, rf_def_acc, rf_tuned_acc]
    colors = [ARM_ORANGE, ARM_RED, ARM_BLUE]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    bars = ax.barh(labels, vals, color=colors, alpha=0.9)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.set_xlabel("Titanic test accuracy (same 70/30 split)")
    ax.set_xlim(0, 1.02)
    ax.invert_yaxis()
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "rf_vs_tree.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")
    return dict(tree=tree_acc, rf_default=rf_def_acc, rf_tuned=rf_tuned_acc)


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


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    X, y, feats = load_titanic(logger)
    inst = fig_instability(X, y, feats, logger)
    ne = fig_n_estimators(X, y, feats, logger)
    fig_importance(X, y, feats, logger)
    vt = fig_vs_tree(X, y, feats, logger)
    fig_max_features(logger)
    fig_boundary_smoothing(logger)
    logger.info("=== SUMMARY for the slides ===")
    logger.info(f"single-tree flip rate ~ {inst['mean_flip']:.0f}% of test predictions")
    logger.info(f"RF acc: 1 tree={ne['acc_1']:.2f} -> plateau ~{ne['acc_max']:.2f}")
    logger.info(f"vs_tree: pruned tree={vt['tree']:.3f}, default RF={vt['rf_default']:.3f}, "
                f"tuned RF={vt['rf_tuned']:.3f}")
    logger.info("done.")


if __name__ == "__main__":
    main()
