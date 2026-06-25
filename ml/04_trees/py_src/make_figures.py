"""Generate the real-data figures for the L09 Decision Trees deck.

Produces three PDFs into ``ml/ch3_trees/fig/`` from the Titanic dataset:
  1. titanic_tree.pdf     -- a shallow (depth-3) decision tree, plot_tree.
  2. titanic_overfit.pdf  -- train vs CV-test accuracy across tree depth
                             (shows a single unrestricted tree overfits).
  3. titanic_pruning.pdf  -- cost-complexity (ccp_alpha) pruning path.

Run from anywhere, e.g. (no persistent venv needed):
    uv run --with scikit-learn --with pandas --with matplotlib \
        ml/ch3_trees/py_src/make_figures.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours for multi-line plots.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 509

# Armenian flag palette (per CLAUDE.md) -- blue=train, red=test/CV, orange=marker.
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
    logger = logging.getLogger("l09_figures")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "make_figures.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def load_titanic(logger: logging.Logger):
    """Return (X, y, feature_names) with a tiny, explicit preprocessing."""
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


def fig_tree(X, y, feature_names, logger):
    """A readable depth-3 tree for the slide."""
    clf = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    clf.fit(X, y)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    plot_tree(
        clf, feature_names=feature_names, class_names=["died", "survived"],
        filled=True, rounded=True, impurity=False, proportion=True,
        fontsize=8, ax=ax,
    )
    fig.tight_layout()
    out = FIG_DIR / "titanic_tree.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name} (depth-3 tree, train acc={clf.score(X,y):.3f})")


def fig_overfit(X, y, feature_names, logger):
    """Train accuracy vs CV-test accuracy across depth -> overfitting."""
    depths = list(range(1, 17))
    train_acc, cv_acc = [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=SEED)
        clf.fit(X, y)
        train_acc.append(clf.score(X, y))
        cv_acc.append(cross_val_score(clf, X, y, cv=5).mean())

    # the unrestricted tree (no depth limit)
    full = DecisionTreeClassifier(random_state=SEED).fit(X, y)
    full_train = full.score(X, y)
    full_cv = cross_val_score(DecisionTreeClassifier(random_state=SEED), X, y, cv=5).mean()
    best_d = depths[int(np.argmax(cv_acc))]
    logger.info(f"unrestricted tree: depth={full.get_depth()}, "
                f"leaves={full.get_n_leaves()}, train acc={full_train:.3f}, "
                f"CV acc={full_cv:.3f}")
    logger.info(f"best CV depth={best_d} (CV acc={max(cv_acc):.3f}); "
                f"train acc there={train_acc[best_d-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(depths, train_acc, "-o", color=ARM_BLUE, lw=2, ms=3, label="train accuracy")
    ax.plot(depths, cv_acc, "-o", color=ARM_RED, lw=2, ms=3, label="test accuracy (5-fold CV)")
    ax.axvline(best_d, color=ARM_ORANGE, ls="--", lw=1.8)
    ax.annotate("sweet spot", xy=(best_d, max(cv_acc)),
                xytext=(best_d + 1.5, max(cv_acc) - 0.08),
                color=ARM_ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xlabel("max tree depth (complexity)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.6, 1.02)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "titanic_overfit.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")
    return dict(full_depth=full.get_depth(), full_leaves=full.get_n_leaves(),
                full_train=full_train, full_cv=full_cv,
                best_d=best_d, best_cv=max(cv_acc))


def fig_pruning(X, y, feature_names, logger):
    """Cost-complexity pruning path: accuracy vs ccp_alpha."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(X_tr, y_tr)
    alphas = path.ccp_alphas[:-1]            # drop the trivial root-only tree
    train_acc, test_acc = [], []
    for a in alphas:
        clf = DecisionTreeClassifier(random_state=SEED, ccp_alpha=a).fit(X_tr, y_tr)
        train_acc.append(clf.score(X_tr, y_tr))
        test_acc.append(clf.score(X_te, y_te))
    best_i = int(np.argmax(test_acc))
    best_alpha = alphas[best_i]
    logger.info(f"best ccp_alpha={best_alpha:.5f} "
                f"(test acc={test_acc[best_i]:.3f}, train acc={train_acc[best_i]:.3f})")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(alphas, train_acc, "-o", color=ARM_BLUE, lw=2, ms=3, label="train accuracy")
    ax.plot(alphas, test_acc, "-o", color=ARM_RED, lw=2, ms=3, label="test accuracy")
    ax.axvline(best_alpha, color=ARM_ORANGE, ls="--", lw=1.8)
    ax.annotate("best pruned tree", xy=(best_alpha, test_acc[best_i]),
                xytext=(best_alpha + 0.004, test_acc[best_i] - 0.12),
                color=ARM_ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xlabel(r"ccp_alpha (cost per leaf)")
    ax.set_ylabel("accuracy")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "titanic_pruning.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out.name}")
    return dict(best_alpha=best_alpha, best_test=test_acc[best_i])


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    X, y, feats = load_titanic(logger)
    fig_tree(X, y, feats, logger)
    over = fig_overfit(X, y, feats, logger)
    prune = fig_pruning(X, y, feats, logger)
    logger.info("=== SUMMARY for the slides ===")
    logger.info(f"unrestricted: depth={over['full_depth']}, leaves={over['full_leaves']}, "
                f"train={over['full_train']:.2f}, CV={over['full_cv']:.2f}")
    logger.info(f"best depth={over['best_d']} (CV={over['best_cv']:.2f}); "
                f"best ccp_alpha={prune['best_alpha']:.4f} (test={prune['best_test']:.2f})")
    logger.info("done.")


if __name__ == "__main__":
    main()
