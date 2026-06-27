"""Real figures for the L14 Imbalanced Learning deck.

Generates (house style, Armenian palette) into ml/03_classification/fig/:
  imb_resampling_2d.pdf  -- original vs ROS vs RUS vs SMOTE on one overlapping 2D blob.
  imb_tomek.pdf          -- Tomek links: borderline majority points flagged for removal
                            (house-style replacement for the third-party source image).
  imb_decalibration.pdf  -- reliability before vs after SMOTE: resampling decalibrates.
  imb_benchmark.pdf      -- PR-AUC of baseline vs ROS/RUS/SMOTE: gains are small.

Run with the project venv:
    ./ma/Scripts/python.exe ml/03_classification/py_src/imbalanced_figs.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l14_imbalanced")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "imbalanced_figs.log"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def blob_2d(n_maj=600, n_min=40):
    """One overlapping 2D imbalanced blob (majority 0, minority 1)."""
    rng = np.random.RandomState(SEED)
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    X0 = rng.multivariate_normal([0.0, 0.0], cov, n_maj)
    X1 = rng.multivariate_normal([1.9, 1.9], cov, n_min)
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n_maj, int), np.ones(n_min, int)]
    return X, y


def _scatter(ax, X, y, title):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=8, c=ARM_BLUE, alpha=0.35, label="majority")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, c=ARM_RED, alpha=0.8,
               edgecolors="white", linewidths=0.3, label="minority")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def fig_resampling_2d(logger):
    X, y = blob_2d()
    Xr, yr = RandomOverSampler(random_state=SEED).fit_resample(X, y)
    Xu, yu = RandomUnderSampler(random_state=SEED).fit_resample(X, y)
    Xs, ys = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(X, y)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    _scatter(axes[0, 0], X, y, f"Original  ({(y==0).sum()} / {(y==1).sum()})")
    _scatter(axes[0, 1], Xr, yr, f"ROS  ({(yr==0).sum()} / {(yr==1).sum()}: copies of {(y==1).sum()})")
    _scatter(axes[1, 0], Xu, yu, f"RUS  ({(yu==0).sum()} / {(yu==1).sum()}: majority dropped)")
    _scatter(axes[1, 1], Xs, ys, f"SMOTE  ({(ys==0).sum()} / {(ys==1).sum()}: synthetic)")
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "imb_resampling_2d.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"resampling_2d: orig {(y==1).sum()} min -> ROS/SMOTE {(ys==1).sum()}, RUS {(yu==0).sum()} maj")
    logger.info("wrote imb_resampling_2d.pdf")


def fig_tomek(logger):
    """House-style Tomek-links illustration (replaces the third-party image)."""
    X, y = blob_2d(n_maj=160, n_min=60)
    tl = TomekLinks()
    Xt, yt = tl.fit_resample(X, y)
    # majority points removed = those in X (y==0) not kept in Xt
    kept = set(map(tuple, np.round(Xt, 8)))
    removed = np.array([p for p, yy in zip(X, y) if yy == 0 and tuple(np.round(p, 8)) not in kept])
    minority = X[y == 1]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, c=ARM_BLUE, alpha=0.4, label="majority")
    ax.scatter(minority[:, 0], minority[:, 1], s=20, c=ARM_RED, alpha=0.85,
               edgecolors="white", linewidths=0.3, label="minority")
    # draw each removed majority point to its nearest minority neighbour (the Tomek partner)
    for p in removed:
        j = int(np.argmin(((minority - p) ** 2).sum(1)))
        ax.plot([p[0], minority[j, 0]], [p[1], minority[j, 1]], color=ARM_ORANGE, lw=1.2, zorder=1)
    if len(removed):
        ax.scatter(removed[:, 0], removed[:, 1], s=90, facecolors="none",
                   edgecolors=ARM_ORANGE, linewidths=1.8, label="Tomek → remove (majority)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Tomek links: drop the borderline majority point of each\nopposite-class nearest-neighbour pair")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "imb_tomek.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"tomek: removed {len(removed)} majority points of {(y==0).sum()}")
    logger.info("wrote imb_tomek.pdf")


def imbalanced_data(pos_rate=0.07):
    X, y = make_classification(
        n_samples=6000, n_features=10, n_informative=5, n_redundant=2,
        weights=[1 - pos_rate, pos_rate], class_sep=0.9, flip_y=0.01, random_state=SEED)
    return train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)


def _ece(yv, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    return sum((m.sum() / len(p)) * abs(yv[m].mean() - p[m].mean())
               for m in (idx == b for b in range(n_bins)) if m.sum())


def fig_decalibration(logger):
    Xtr, Xte, ytr, yte = imbalanced_data()
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    base = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
    p_base = base.predict_proba(Xte_s)[:, 1]

    Xres, yres = SMOTE(random_state=SEED).fit_resample(Xtr_s, ytr)
    smote = LogisticRegression(max_iter=2000).fit(Xres, yres)
    p_smote = smote.predict_proba(Xte_s)[:, 1]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1.3, label="perfect")
    for name, p, c in [("baseline (no resampling)", p_base, ARM_BLUE),
                       ("after SMOTE", p_smote, ARM_RED)]:
        frac, mean = calibration_curve(yte, p, n_bins=10, strategy="uniform")
        ax.plot(mean, frac, "-o", color=c, lw=2, ms=4, label=f"{name}  (ECE {_ece(yte, p):.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Resampling decalibrates the probabilities")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "imb_decalibration.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"decalibration: ECE base {_ece(yte,p_base):.3f}, SMOTE {_ece(yte,p_smote):.3f}; "
                f"mean pred base {p_base.mean():.3f} / SMOTE {p_smote.mean():.3f} (true {yte.mean():.3f})")
    logger.info("wrote imb_decalibration.pdf")


def fig_benchmark(logger):
    Xtr, Xte, ytr, yte = imbalanced_data()
    def lr():
        return LogisticRegression(max_iter=2000)
    configs = {
        "baseline": ImbPipeline([("sc", StandardScaler()), ("clf", lr())]),
        "ROS": ImbPipeline([("sc", StandardScaler()), ("s", RandomOverSampler(random_state=SEED)), ("clf", lr())]),
        "RUS": ImbPipeline([("sc", StandardScaler()), ("s", RandomUnderSampler(random_state=SEED)), ("clf", lr())]),
        "SMOTE": ImbPipeline([("sc", StandardScaler()), ("s", SMOTE(random_state=SEED)), ("clf", lr())]),
    }
    aps = {}
    for name, pipe in configs.items():
        pipe.fit(Xtr, ytr)
        aps[name] = average_precision_score(yte, pipe.predict_proba(Xte)[:, 1])

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    names = list(aps)
    vals = [aps[n] for n in names]
    colors = [ARM_BLUE] + [ARM_RED] * 3
    bars = ax.bar(names, vals, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.axhline(yte.mean(), ls="--", color="0.5", lw=1.2)
    ax.text(3.4, yte.mean() + 0.01, f"base rate {yte.mean():.2f}", fontsize=8, color="0.4", ha="right")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("PR-AUC (average precision) on test")
    ax.set_title("Resampling barely moves the ranking (PR-AUC)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "imb_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"benchmark PR-AUC: {dict((k, round(v,3)) for k,v in aps.items())}")
    logger.info("wrote imb_benchmark.pdf")


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    fig_resampling_2d(logger)
    fig_tomek(logger)
    fig_decalibration(logger)
    fig_benchmark(logger)
    logger.info("done.")


if __name__ == "__main__":
    main()
