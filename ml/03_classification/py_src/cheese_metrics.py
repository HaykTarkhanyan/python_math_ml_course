"""Real metrics figures for the L07 Classification Metrics deck.

Generates 4 PDFs into ml/ch2_classification/fig/ from a synthetic but
realistic imbalanced "cheese factory" dataset (positive = bad batch). Every
curve is a real sweep over a fitted model's scores -- no invented coordinates.

  cm_roc.pdf         -- ROC curve, shaded AUC, with the AUC value.
  cm_pr.pdf          -- precision-recall curve, base-rate line, with AP.
  cm_lift.pdf        -- cumulative gain curve + lift at top 20%.
  cm_roc_vs_pr.pdf   -- two models: ROC (both look great) vs PR (one collapses).
  cm_score_dist.pdf  -- the two score distributions + a sliding cutoff -> TP/FP/FN/TN.
  cm_f1_heatmap.pdf  -- F1 vs arithmetic mean over (precision, recall).
  (also: cm_threshold_metrics, cm_cost_curve, cm_youden, cm_recall_floor, cm_multiclass.)

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
from scipy.stats import gaussian_kde
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


def fig_lift_decile(yte, s, logger):
    """Lift per decile: rank by score, split into 10 equal groups, bad-rate / base rate."""
    order = np.argsort(-s)
    y_sorted = yte[order]
    base = float(yte.mean())
    deciles = np.array_split(y_sorted, 10)
    lift = [float(d.mean()) / base for d in deciles]
    fig, ax = plt.subplots(figsize=(5.9, 4.0))
    bars = ax.bar(range(1, 11), lift, color=ARM_BLUE, edgecolor="white")
    ax.axhline(1.0, ls="--", color="0.6", lw=1.3, label="random (lift 1)")
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("decile (1 = top-scored 10%)")
    ax.set_ylabel("lift = decile bad-rate / base rate")
    ax.set_title("Lift by decile")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    logger.info("lift by decile (1..10): " + ", ".join(f"{l:.1f}" for l in lift))
    _finish(ax, "cm_lift_decile.pdf", logger)


def fig_pr_ap(yte, s, logger):
    """PR curve with the area (AP) shaded -- for the 'PR AUC = area under PR' explanation."""
    prec, rec, _ = precision_recall_curve(yte, s)
    ap = average_precision_score(yte, s)
    base = float(yte.mean())
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.fill_between(rec, prec, color=ARM_RED, alpha=0.18, step="post")
    ax.plot(rec, prec, color=ARM_RED, lw=2.4)
    ax.axhline(base, ls="--", color="0.6", lw=1.2, label=f"base rate = {base:.2f} (random AP)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"AP = area under PR = {ap:.2f}")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    logger.info(f"pr_ap: AP={ap:.3f}, base={base:.3f}")
    _finish(ax, "cm_pr_ap.pdf", logger)


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


def fig_youden(yte, s, logger):
    """ROC-based threshold: the point maximizing J = TPR - FPR (cost-free)."""
    fpr, tpr, thr = roc_curve(yte, s)
    J = tpr - fpr
    k = int(np.argmax(J))

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1.2, label="random")
    ax.plot(fpr, tpr, color=ARM_BLUE, lw=2.2)
    # J is the vertical gap from the chosen ROC point down to the diagonal
    ax.plot([fpr[k], fpr[k]], [fpr[k], tpr[k]], color=ARM_ORANGE, lw=2.6)
    ax.plot(fpr[k], tpr[k], "o", color=ARM_ORANGE, ms=8)
    ax.annotate(f"max $J$ = {J[k]:.2f}\n($c$ = {thr[k]:.2f})",
                xy=(fpr[k], tpr[k]), xytext=(fpr[k] + 0.16, tpr[k] - 0.22),
                fontsize=9, color=ARM_ORANGE,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR (false-alarm rate)")
    ax.set_ylabel("TPR (recall)")
    ax.set_title("Youden's J: farthest point above the diagonal")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    logger.info(f"youden: J={J[k]:.3f} at c={thr[k]:.3f}, TPR={tpr[k]:.3f}, FPR={fpr[k]:.3f}")
    _finish(ax, "cm_youden.pdf", logger)


def fig_recall_floor(yte, s, logger, floor=0.80):
    """Constraint-based threshold: highest c that still meets recall >= floor."""
    ts = np.linspace(0.001, 0.999, 400)
    P, R = [], []
    for t in ts:
        prec, rec, _, _, _ = _pr_f1_at(yte, s, t)
        P.append(prec); R.append(rec)
    P, R = np.array(P), np.array(R)
    ok = np.where(R >= floor)[0]          # recall decreases with c -> low-c indices
    k = int(ok[-1])                        # highest threshold still meeting the floor
    c = ts[k]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(ts, R, color=ARM_RED, lw=2.2, label="recall")
    ax.plot(ts, P, color=ARM_BLUE, lw=2.2, label="precision")
    ax.axhline(floor, color="0.5", ls=":", lw=1.4)
    ax.text(0.985, floor + 0.02, f"recall floor {floor:.2f}", fontsize=8,
            color="0.4", ha="right")
    ax.axvline(c, color=ARM_ORANGE, lw=2.2, ls="--")
    ax.plot(c, P[k], "o", color=ARM_ORANGE, ms=7)
    ax.annotate(f"highest $c$ with recall $\\geq$ {floor:.2f}\n"
                f"$c$ = {c:.2f}, precision = {P[k]:.2f}",
                xy=(c, P[k]), xytext=(c + 0.10, P[k] + 0.28),
                fontsize=8.5, color=ARM_ORANGE,
                arrowprops=dict(arrowstyle="->", color=ARM_ORANGE))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("threshold $c$")
    ax.set_ylabel("metric value")
    ax.set_title("Recall floor: meet it, then maximize precision")
    ax.legend(loc="center right", fontsize=9, frameon=False)
    logger.info(f"recall-floor {floor}: c={c:.3f}, recall={R[k]:.3f}, precision={P[k]:.3f}")
    _finish(ax, "cm_recall_floor.pdf", logger)


def fig_three_cutoffs(yte, s, logger):
    """All three operating-point rules on ONE threshold axis.

    Cost-optimal, Youden's J and a recall floor each pick a *different* cutoff --
    and all three sit far below the default 0.5. The recall/precision curves give
    context: a lower cutoff buys recall at the price of precision.
    """
    VIOLET, GREEN = "#7832A0", "#008C46"
    ts = np.linspace(0.001, 0.30, 400)
    P = np.array([_pr_f1_at(yte, s, t)[0] for t in ts])
    R = np.array([_pr_f1_at(yte, s, t)[1] for t in ts])

    tfull = np.linspace(0.001, 0.999, 400)
    cost = np.array([100 * _pr_f1_at(yte, s, t)[4] + 10 * _pr_f1_at(yte, s, t)[3]
                     for t in tfull])
    c_cost = tfull[int(np.argmin(cost))]
    fpr, tpr, thr = roc_curve(yte, s)
    c_youden = thr[int(np.argmax(tpr - fpr))]
    Rfull = np.array([_pr_f1_at(yte, s, t)[1] for t in tfull])
    c_floor = tfull[int(np.where(Rfull >= 0.80)[0][-1])]

    # (cutoff, colour, name, text-anchor) -- text fans down-right so short
    # non-crossing arrows reach the three closely spaced lines.
    cuts = [
        (c_floor, GREEN, "recall floor $\\geq 0.80$", (0.105, 0.93)),
        (c_youden, VIOLET, "Youden's $J$", (0.175, 0.78)),
        (c_cost, ARM_ORANGE, "cost-optimal $c^*$", (0.250, 0.63)),
    ]

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.plot(ts, R, color=ARM_RED, lw=2.0, label="recall")
    ax.plot(ts, P, color=ARM_BLUE, lw=2.0, label="precision")
    for (c, col, name, (xt, yt)), yhead in zip(cuts, (0.83, 0.66, 0.50)):
        prec, rec, _, fp, fn = _pr_f1_at(yte, s, c)
        ax.axvline(c, color=col, ls="--", lw=2.0)
        ax.annotate(f"{name}\n$c={c:.2f}$:  R={rec:.2f}, P={prec:.2f}",
                    xy=(c, yhead), xytext=(xt, yt), fontsize=8, color=col,
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.1))
    ax.text(0.298, 0.90, "default 0.5 is\nfar to the right", fontsize=8,
            color="0.45", ha="right", va="center", style="italic")
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("threshold $c$")
    ax.set_ylabel("metric value")
    ax.set_title("One score, three rules $\\to$ three cutoffs (all $\\ll 0.5$)")
    ax.legend(loc="upper right", fontsize=9, frameon=False,
              bbox_to_anchor=(1.0, 0.52))
    logger.info(f"three-cutoffs: floor c={c_floor:.3f}, youden c={c_youden:.3f}, "
                f"cost c={c_cost:.3f}")
    _finish(ax, "cm_three_cutoffs.pdf", logger)


def fig_score_dist(logger, c=0.5):
    """Two score distributions and a movable cutoff -> the TP/FP/FN/TN regions.

    The master picture behind the whole metrics lecture: the confusion cells,
    the precision/recall tradeoff and ROC all fall out of two overlapping score
    distributions and where you put the cut. Balanced and illustrative so both
    distributions span [0,1] and overlap legibly -- the imbalance lives in the
    other figures. Scores are drawn in logit space (a logistic model's natural
    output) with a tuned signal/overlap.
    """
    rng = np.random.RandomState(SEED)
    n = 2500
    s_good = 1.0 / (1.0 + np.exp(-rng.normal(-1.4, 1.25, n)))  # centred ~0.20
    s_bad = 1.0 / (1.0 + np.exp(-rng.normal(1.4, 1.25, n)))    # centred ~0.80

    grid = np.linspace(0, 1, 500)
    d_good = gaussian_kde(s_good)(grid)
    d_bad = gaussian_kde(s_bad)(grid)
    ymax = max(d_good.max(), d_bad.max())
    left, right = grid <= c, grid >= c

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(grid, d_good, color=ARM_BLUE, lw=2)
    ax.plot(grid, d_bad, color=ARM_RED, lw=2)
    ax.fill_between(grid, d_good, color=ARM_BLUE, alpha=0.12)
    ax.fill_between(grid, d_bad, color=ARM_RED, alpha=0.12)
    # the two errors at this cutoff:
    ax.fill_between(grid[left], d_bad[left], color=ARM_RED, alpha=0.45)     # bad, scored low = FN
    ax.fill_between(grid[right], d_good[right], color=ARM_BLUE, alpha=0.40)  # good, scored high = FP
    ax.axvline(c, color=ARM_ORANGE, lw=2.4, ls="--")

    ax.text(c + 0.01, ymax * 1.12, "cutoff $c$", color=ARM_ORANGE, fontsize=9, ha="left")
    ax.text(0.04, ymax * 1.02, "predict good", color="0.45", fontsize=8)
    ax.text(0.96, ymax * 1.02, "predict bad", color="0.45", fontsize=8, ha="right")
    ax.text(0.20, ymax * 0.45, "TN", color=ARM_BLUE, fontsize=9, ha="center")
    ax.text(0.80, ymax * 0.45, "TP", color=ARM_RED, fontsize=9, ha="center")
    ax.annotate("FN\n(bad, missed)", xy=(c - 0.11, ymax * 0.10),
                color=ARM_RED, fontsize=8, ha="center")
    ax.annotate("FP\n(false alarm)", xy=(c + 0.12, ymax * 0.10),
                color=ARM_BLUE, fontsize=8, ha="center")

    ax.set_ylim(0, ymax * 1.26)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("predicted score  $\\hat p$(bad)")
    ax.set_ylabel("density (each class normalised)")
    ax.set_title("Two score distributions and a cutoff")
    ax.legend([plt.Line2D([], [], color=ARM_BLUE, lw=2),
               plt.Line2D([], [], color=ARM_RED, lw=2)],
              ["good batches", "bad batches"], loc="center",
              bbox_to_anchor=(0.5, 0.66), fontsize=8, frameon=False)
    logger.info(f"score-dist (illustrative balanced): good median={np.median(s_good):.3f}, "
                f"bad median={np.median(s_bad):.3f}, cutoff c={c}")
    _finish(ax, "cm_score_dist.pdf", logger)


def fig_f1_heatmap(logger):
    """F1 (harmonic) vs arithmetic mean over the (precision, recall) square."""
    g = np.linspace(0.001, 1, 250)
    PP, RR = np.meshgrid(g, g)
    f1 = 2 * PP * RR / (PP + RR)
    arith = (PP + RR) / 2
    levels = np.linspace(0, 1, 11)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    for ax, Z, title in [(a1, f1, "F1 = harmonic mean"),
                         (a2, arith, "arithmetic mean")]:
        cf = ax.contourf(PP, RR, Z, levels=levels, cmap="viridis")
        ax.contour(PP, RR, Z, levels=levels, colors="white", linewidths=0.4, alpha=0.5)
        ax.set_xlabel("precision")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
    a1.set_ylabel("recall")
    # the deck's four table rows, marked on the F1 panel
    pts = [(1.0, 0.0), (0.8, 0.8), (0.5, 0.9), (0.9, 0.5)]
    a1.scatter([p for p, _ in pts], [r for _, r in pts], color=ARM_RED, s=34,
               edgecolors="white", linewidths=0.7, zorder=5)
    cb = fig.colorbar(cf, ax=(a1, a2), fraction=0.046, pad=0.03)
    cb.set_label("score")
    logger.info("f1-heatmap: F1 contours bow into the corners; arithmetic stay straight")
    fig.savefig(FIG_DIR / "cm_f1_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote cm_f1_heatmap.pdf")


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
    fig_pr_ap(yte, s, logger)
    fig_lift(yte, s, logger)
    fig_lift_decile(yte, s, logger)
    fig_roc_vs_pr(logger)
    fig_threshold_metrics(yte, s, logger)
    fig_cost_curve(yte, s, logger)
    fig_youden(yte, s, logger)
    fig_recall_floor(yte, s, logger)
    fig_three_cutoffs(yte, s, logger)
    fig_score_dist(logger)
    fig_f1_heatmap(logger)
    fig_multiclass_confusion(logger)
    logger.info("done.")


if __name__ == "__main__":
    main()
