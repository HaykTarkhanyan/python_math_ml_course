"""Real metrics figures for the L07 Classification Metrics deck.

Generates 4 PDFs into ml/ch2_classification/fig/ from a synthetic but
realistic imbalanced "cheese factory" dataset (positive = bad batch). Every
curve is a real sweep over a fitted model's scores -- no invented coordinates.

  cm_roc.pdf        -- ROC curve, shaded AUC, with the AUC value.
  cm_pr.pdf         -- precision-recall curve, base-rate line, with AP.
  cm_lift.pdf       -- cumulative gain curve + lift at top 20%.
  cm_roc_vs_pr.pdf  -- two models: ROC (both look great) vs PR (one collapses).

Run with the project venv (see repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml/ch2_classification/py_src/cheese_metrics.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

SEED = 509
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]              # ml/ch2_classification
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l07_metrics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "cheese_metrics.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def cheese_scores():
    """One fitted model on a ~3%-positive cheese dataset -> (y_test, scores).

    Clean logistic DGP: labels drawn from sigmoid(Xw + b). Signal strength
    ||w|| sets the achievable AUC (monotonic, unlike make_classification's
    cluster geometry), and b sets the ~3% prevalence.
    """
    np.random.seed(SEED)
    n = 8000
    X = np.random.normal(size=(n, 5))
    w = np.array([0.85, 0.7, -0.6, 0.5, 0.35])   # ||w|| ~ 1.4 -> AUC ~ 0.88
    b = -4.2                                       # -> ~3% positive
    p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    y = (np.random.uniform(size=n) < p).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return yte, clf.predict_proba(Xte)[:, 1]


def _finish(ax, out, logger):
    ax.spines[["top", "right"]].set_visible(False)
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / out, bbox_inches="tight")
    plt.close(ax.figure)
    logger.info(f"wrote {out}")


def fig_roc(yte, s, logger):
    fpr, tpr, _ = roc_curve(yte, s)
    auc = roc_auc_score(yte, s)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1.2, label="random")
    ax.plot(fpr, tpr, color=ARM_RED, lw=2.4)
    ax.fill_between(fpr, tpr, color=ARM_RED, alpha=0.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR (false alarm rate)")
    ax.set_ylabel("TPR (recall)")
    ax.set_title(f"ROC curve  ---  AUC = {auc:.2f}")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    logger.info(f"single-model: ROC AUC={auc:.3f}")
    _finish(ax, "cm_roc.pdf", logger)


def fig_pr(yte, s, logger):
    prec, rec, _ = precision_recall_curve(yte, s)
    ap = average_precision_score(yte, s)
    base = float(yte.mean())
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(rec, prec, color=ARM_RED, lw=2.4)
    ax.axhline(base, ls="--", color="0.6", lw=1.2, label=f"base rate = {base:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve  ---  AP = {ap:.2f}")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    logger.info(f"single-model: AP={ap:.3f}, base rate={base:.3f}")
    _finish(ax, "cm_pr.pdf", logger)


def fig_lift(yte, s, logger):
    order = np.argsort(-s)
    y_sorted = yte[order]
    n = len(yte)
    total_pos = int(y_sorted.sum())
    frac = np.arange(1, n + 1) / n
    gain = np.cumsum(y_sorted) / total_pos
    base = total_pos / n
    k = max(1, int(0.20 * n))
    gain20 = gain[k - 1]
    lift20 = gain20 / 0.20

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1.2, label="random")
    ax.plot([0, base, 1], [0, 1, 1], color=ARM_ORANGE, lw=1.6, label="perfect")
    ax.plot(frac, gain, color=ARM_RED, lw=2.4, label="our model")
    ax.axvline(0.20, ls="--", color="0.5", lw=1.2)
    ax.annotate(f"top 20% catches {gain20*100:.0f}%\n(lift {lift20:.1f}$\\times$)",
                xy=(0.20, gain20), xytext=(0.30, 0.45),
                fontsize=9, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("fraction of batches inspected (ranked by score)")
    ax.set_ylabel("fraction of bad batches caught")
    ax.set_title("Cumulative gain")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    logger.info(f"lift: top20 gain={gain20:.3f}, lift={lift20:.2f}, base={base:.3f}")
    _finish(ax, "cm_lift.pdf", logger)


def fig_roc_vs_pr(logger):
    """Two models on 1%-positive data: ROC both look great, PR diverges."""
    np.random.seed(SEED)
    n, n_pos = 20000, 200
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    s_a = np.where(y == 1, np.random.normal(3.0, 1.0, n), np.random.normal(0.0, 1.0, n))
    s_b = np.where(y == 1, np.random.normal(2.2, 1.0, n), np.random.normal(0.0, 1.0, n))
    base = y.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for s, c, lab in [(s_a, ARM_BLUE, "Model A"), (s_b, ARM_RED, "Model B")]:
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
        ax1.plot(fpr, tpr, color=c, lw=2, label=f"{lab}  (AUC {auc:.2f})")
        ax2.plot(rec, prec, color=c, lw=2, label=f"{lab}  (AP {ap:.2f})")
        logger.info(f"roc_vs_pr Model {lab[-1]}: ROC AUC={auc:.3f}, AP={ap:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="0.6", lw=1.0)
    ax1.set_title("ROC: ``both look great''")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.legend(loc="lower right", fontsize=8, frameon=False)
    ax2.axhline(base, ls="--", color="0.6", lw=1.0)
    ax2.set_title("PR: Model B collapses")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cm_roc_vs_pr.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote cm_roc_vs_pr.pdf")


def _pr_f1_at(yte, s, t):
    pred = s >= t
    tp = int(np.sum(pred & (yte == 1)))
    fp = int(np.sum(pred & (yte == 0)))
    fn = int(np.sum((~pred) & (yte == 1)))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, fp, fn


def fig_threshold_metrics(yte, s, logger):
    ts = np.linspace(0.01, 0.99, 197)
    P, R, F = [], [], []
    for t in ts:
        prec, rec, f1, _, _ = _pr_f1_at(yte, s, t)
        P.append(prec); R.append(rec); F.append(f1)
    P, R, F = np.array(P), np.array(R), np.array(F)
    t_f1 = ts[int(np.argmax(F))]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(ts, R, color=ARM_RED, lw=2.2, label="recall")
    ax.plot(ts, P, color=ARM_BLUE, lw=2.2, label="precision")
    ax.plot(ts, F, color=ARM_ORANGE, lw=2.2, label="F1")
    ax.axvline(0.5, ls=":", color="0.5", lw=1.3)
    ax.axvline(t_f1, ls="--", color=ARM_ORANGE, lw=1.6)
    ax.annotate(f"F1-best\n$c={t_f1:.2f}$", xy=(t_f1, F.max()),
                xytext=(t_f1 + 0.12, 0.7), fontsize=9, color=ARM_ORANGE,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.text(0.5, 1.02, "default 0.5", fontsize=8, color="0.4", ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("threshold $c$")
    ax.set_ylabel("metric value")
    ax.set_title("Every metric is a function of the threshold")
    ax.legend(loc="center right", fontsize=9, frameon=False)
    logger.info(f"threshold-metrics: F1-best c={t_f1:.3f}, F1={F.max():.3f}")
    _finish(ax, "cm_threshold_metrics.pdf", logger)


def fig_cost_curve(yte, s, logger):
    c_fn, c_fp = 100, 10
    ts = np.linspace(0.01, 0.99, 197)
    cost = np.array([c_fn * _pr_f1_at(yte, s, t)[4] + c_fp * _pr_f1_at(yte, s, t)[3]
                     for t in ts])
    t_star = ts[int(np.argmin(cost))]
    cost_star = cost.min()
    cost_half = c_fn * _pr_f1_at(yte, s, 0.5)[4] + c_fp * _pr_f1_at(yte, s, 0.5)[3]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(ts, cost, color=ARM_RED, lw=2.4)
    ax.axvline(t_star, ls="--", color=ARM_ORANGE, lw=1.8)
    ax.axvline(0.5, ls=":", color="0.5", lw=1.3)
    ax.annotate(f"cost-optimal\n$c^*={t_star:.2f}$", xy=(t_star, cost_star),
                xytext=(t_star + 0.12, cost_star + 0.35 * (cost.max() - cost_star)),
                fontsize=9, color=ARM_ORANGE,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.text(0.5, cost_half, "  default 0.5", fontsize=8, color="0.4", ha="left", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("threshold $c$")
    ax.set_ylabel(r"expected cost  ($C_{FN}{=}100,\ C_{FP}{=}10$)")
    ax.set_title("Cost-optimal threshold is far below 0.5")
    logger.info(f"cost-curve: c*={t_star:.3f} (cost {cost_star:.0f}); "
                f"cost at 0.5 = {cost_half:.0f}")
    _finish(ax, "cm_cost_curve.pdf", logger)


def fig_multiclass_confusion(logger):
    """3-class cheese (good / moldy / dry) confusion-matrix heatmap."""
    X, y = make_classification(
        n_samples=3000, n_features=10, n_informative=6, n_redundant=2,
        n_classes=3, n_clusters_per_class=1, class_sep=1.1, random_state=SEED)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    cm = confusion_matrix(yte, clf.predict(Xte))
    labels = ["good", "moldy", "dry"]

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("3-class confusion matrix")
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cm_multiclass.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"multiclass confusion (rows=actual good/moldy/dry):\n{cm}")
    logger.info("wrote cm_multiclass.pdf")


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    yte, s = cheese_scores()
    logger.info(f"cheese test set: n={len(yte)}, positives={int(yte.sum())} "
                f"({yte.mean()*100:.1f}%)")
    fig_roc(yte, s, logger)
    fig_pr(yte, s, logger)
    fig_lift(yte, s, logger)
    fig_roc_vs_pr(logger)
    fig_threshold_metrics(yte, s, logger)
    fig_cost_curve(yte, s, logger)
    fig_multiclass_confusion(logger)
    logger.info("done.")


if __name__ == "__main__":
    main()
