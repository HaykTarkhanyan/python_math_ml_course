"""Real calibration figures for the L13 Calibration deck.

Generates 3 PDFs into ml_new/ch2_classification/fig/ from a binary dataset:
  cal_reliability.pdf    -- reliability diagram: well-calibrated logistic vs
                            over-confident naive Bayes (+ Brier scores).
  cal_hist.pdf           -- histograms of predicted probabilities: over-confident
                            scores pile up near 0 and 1.
  cal_before_after.pdf   -- naive Bayes before vs after isotonic calibration;
                            the curve snaps onto the diagonal.

Run with the project venv (see repo CLAUDE.md -> Python Environment):
    ./ma/Scripts/python.exe ml_new/ch2_classification/py_src/calibration_demo.py

Conventions (repo CLAUDE.md): logging to console + logs/, seed 509, f-strings,
Armenian-flag colours.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

SEED = 509
ARM_BLUE = "#0033A0"
ARM_RED = "#D90012"
ARM_ORANGE = "#F2A800"

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("l13_calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "calibration_demo.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def ece(y, p, n_bins=10):
    """Expected calibration error: support-weighted |accuracy - confidence| per bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            total += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return total


def get_data():
    # Many redundant (correlated) features -> naive Bayes double-counts evidence
    # and becomes over-confident, while logistic regression stays calibrated.
    X, y = make_classification(
        n_samples=12000, n_features=22, n_informative=5, n_redundant=14,
        weights=[0.65, 0.35], class_sep=1.0, flip_y=0.02, random_state=SEED)
    return train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)


def fig_reliability(probs, yte, logger):
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1.3, label="perfectly calibrated")
    for name, p, c in [("logistic regression", probs["lr"], ARM_BLUE),
                       ("naive Bayes", probs["nb"], ARM_RED)]:
        frac, mean = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        e = ece(yte, p)
        ax.plot(mean, frac, "-o", color=c, lw=2, ms=4,
                label=f"{name}  (ECE {e:.3f})")
        logger.info(f"reliability: {name} ECE={e:.4f}, Brier={brier_score_loss(yte, p):.4f}")
    ax.annotate("below the line\n= over-confident", xy=(0.78, 0.62),
                xytext=(0.42, 0.78), fontsize=9, color=ARM_RED,
                arrowprops=dict(arrowstyle="->", color=ARM_RED))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency of positives")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cal_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote cal_reliability.pdf")


def fig_hist(probs, logger):
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    bins = np.linspace(0, 1, 26)
    ax.hist(probs["lr"], bins=bins, color=ARM_BLUE, alpha=0.55, label="logistic regression")
    ax.hist(probs["nb"], bins=bins, color=ARM_RED, alpha=0.55, label="naive Bayes")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("count")
    ax.set_title("Over-confident scores pile up at 0 and 1")
    ax.legend(loc="upper center", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cal_hist.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote cal_hist.pdf")


def fig_before_after(Xtr, Xte, ytr, yte, logger):
    raw = GaussianNB().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    cal = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)
    cal.fit(Xtr, ytr)
    cal_p = cal.predict_proba(Xte)[:, 1]
    e_raw, e_cal = ece(yte, raw), ece(yte, cal_p)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1.3, label="perfectly calibrated")
    for name, p, c in [(f"before  (ECE {e_raw:.3f})", raw, ARM_RED),
                       (f"after isotonic  (ECE {e_cal:.3f})", cal_p, ARM_BLUE)]:
        frac, mean = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "-o", color=c, lw=2, ms=4, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency of positives")
    ax.set_title("Calibrating naive Bayes (isotonic)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cal_before_after.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"before/after: ECE raw={e_raw:.4f} -> isotonic={e_cal:.4f}; "
                f"Brier {brier_score_loss(yte, raw):.4f} -> {brier_score_loss(yte, cal_p):.4f}")
    logger.info("wrote cal_before_after.pdf")


def main():
    logger = setup_logging()
    FIG_DIR.mkdir(exist_ok=True)
    Xtr, Xte, ytr, yte = get_data()
    probs = {
        "lr": LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1],
        "nb": GaussianNB().fit(Xtr, ytr).predict_proba(Xte)[:, 1],
    }
    logger.info(f"data: n_test={len(yte)}, positive rate={yte.mean():.3f}")
    fig_reliability(probs, yte, logger)
    fig_hist(probs, logger)
    fig_before_after(Xtr, Xte, ytr, yte, logger)
    logger.info("done.")


if __name__ == "__main__":
    main()
