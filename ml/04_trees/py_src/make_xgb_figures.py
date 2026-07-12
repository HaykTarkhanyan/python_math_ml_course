"""Figures for the Advanced Boosting ([20]) deck.

Figures for the deck:
  - 12_xgb_histogram_split.pdf : percentile bin edges as candidate splits (synthetic)
  - 12_xgb_earlystop.pdf       : train/val logloss vs rounds, early-stop point (synthetic)
  - 12_xgb_importance.pdf      : gain-based feature importance on REAL data (Titanic)
  - xgb_monotonic.pdf          : unconstrained (wiggly) vs monotone-constrained (clean) fit
  - xgb_stacking.pdf           : RF / LightGBM / logistic vs a stacked meta-learner (receipt)

Run with the ma venv:  ./ma/Scripts/python.exe ml/04_trees/py_src/make_xgb_figures.py
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lightgbm as lgb
from sklearn.datasets import fetch_openml, make_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/make_xgb_figures.log")],
)
log = logging.getLogger(__name__)

SEED = 509
np.random.seed(SEED)
FIG = Path(__file__).resolve().parents[1] / "fig"
FIG.mkdir(exist_ok=True)
ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"

# one synthetic dataset, reused everywhere
X, y = make_classification(
    n_samples=4000, n_features=12, n_informative=6, n_redundant=2,
    n_clusters_per_class=2, class_sep=1.0, random_state=SEED,
)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
log.info("data: X_tr=%s X_val=%s", X_tr.shape, X_val.shape)


def fig_histogram_split():
    x = X_tr[:, 0]
    edges = np.quantile(x, np.linspace(0, 1, 11))   # 10 percentile bins -> 11 edges
    chosen = edges[6]                               # pretend this edge maximises the gain
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=45, color="0.82", edgecolor="white")
    for e in edges:
        ax.axvline(e, color=ARM_BLUE, lw=1.0, alpha=0.7)
    ax.axvline(chosen, color=ARM_RED, lw=2.6)
    ax.set(title="Approximate split finding: candidates = percentile bin edges",
           xlabel="feature value", ylabel="count")
    ax.legend(handles=[
        Line2D([0], [0], color=ARM_BLUE, lw=1.0, label="percentile edges (candidate splits)"),
        Line2D([0], [0], color=ARM_RED, lw=2.6, label="chosen split (max gain)"),
    ], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "12_xgb_histogram_split.pdf")
    plt.close(fig)
    log.info("wrote 12_xgb_histogram_split.pdf")


def fig_earlystop():
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = dict(objective="binary:logistic", eval_metric="logloss",
                  eta=0.3, max_depth=5, seed=SEED)
    res = {}
    booster = xgb.train(params, dtr, num_boost_round=400,
                        evals=[(dtr, "train"), (dval, "val")],
                        early_stopping_rounds=20, evals_result=res, verbose_eval=False)
    tr, va = res["train"]["logloss"], res["val"]["logloss"]
    best = booster.best_iteration
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tr, color=ARM_BLUE, lw=2, label="train logloss")
    ax.plot(va, color=ARM_RED, lw=2, label="validation logloss")
    ax.axvline(best, color=ARM_ORANGE, ls="--", lw=2, label=f"early stop @ round {best}")
    ax.set(title="Early stopping: validation logloss stops improving",
           xlabel="boosting round", ylabel="logloss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "12_xgb_earlystop.pdf")
    plt.close(fig)
    log.info("wrote 12_xgb_earlystop.pdf (best_iteration=%d, val rounds=%d)", best, len(va))


def _load_titanic():
    df = fetch_openml("titanic", version=1, as_frame=True).frame
    feats = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    d = df[feats + ["survived"]].copy()
    d["sex"] = (d["sex"] == "female").astype(int)
    d["pclass"] = d["pclass"].astype(float)
    d["age"] = d["age"].fillna(d["age"].median())
    d["fare"] = d["fare"].fillna(d["fare"].median())
    return d[feats].astype(float).to_numpy(), d["survived"].astype(int).to_numpy(), feats


def fig_importance():
    """Gain importance on a REAL dataset (Titanic), so the bars carry meaning -- unlike
    synthetic f0..f11."""
    Xt, yt, feats = _load_titanic()
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.3,
                            importance_type="gain", random_state=SEED, eval_metric="logloss")
    clf.fit(Xt, yt)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    disp = ["gender" if f == "sex" else f for f in feats]   # relabel sex -> gender
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar([disp[i] for i in order], imp[order], color=ARM_BLUE)
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    ax.set(title="Gain-based feature importance (XGBoost on Titanic)",
           xlabel="feature", ylabel="importance (gain, normalized)")
    ax.margins(y=0.15)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "12_xgb_importance.pdf")
    plt.close(fig)
    log.info("wrote 12_xgb_importance.pdf (Titanic: %s)",
             ", ".join(f"{feats[i]}={imp[i]:.2f}" for i in order))


def fig_monotonic():
    """Unconstrained (wiggly) vs monotone-constrained (clean) fit on a noisy, genuinely
    increasing feature -- the constraint buys trust and acts as free regularization."""
    rng = np.random.default_rng(SEED)
    n = 800
    xf = rng.uniform(0, 10, n)
    yn = np.sqrt(xf) + rng.normal(0, 0.7, n)          # monotone signal + noise
    Xf = xf.reshape(-1, 1)
    grid = np.linspace(0, 10, 400).reshape(-1, 1)
    common = dict(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=SEED)
    unc = xgb.XGBRegressor(**common).fit(Xf, yn)
    con = xgb.XGBRegressor(monotone_constraints=(1,), **common).fit(Xf, yn)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.scatter(xf, yn, s=8, color="0.72", alpha=0.5, label="noisy data")
    ax.plot(grid, unc.predict(grid), color=ARM_RED, lw=2.2, label="unconstrained (wiggly)")
    ax.plot(grid, con.predict(grid), color=ARM_BLUE, lw=2.2, label="monotone constraint (clean)")
    ax.set_xlabel("feature (should only ever help)")
    ax.set_ylabel("prediction")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "xgb_monotonic.pdf")
    plt.close(fig)
    log.info("wrote xgb_monotonic.pdf")


def fig_stacking():
    """The receipt: does a stacked meta-learner actually beat its base models? Diverse
    bases (RF, LightGBM, logistic) + a logistic meta-learner, validation accuracy."""
    base = [("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)),
            ("lgbm", lgb.LGBMClassifier(n_estimators=300, random_state=SEED, verbose=-1)),
            ("logreg", LogisticRegression(max_iter=2000))]
    accs = {}
    for name, m in base:
        accs[name] = m.fit(X_tr, y_tr).score(X_val, y_val)
    stack = StackingClassifier(estimators=base,
                               final_estimator=LogisticRegression(max_iter=2000),
                               cv=5, n_jobs=1)
    accs["stack"] = stack.fit(X_tr, y_tr).score(X_val, y_val)
    log.info("stacking accs: %s", {k: round(v, 4) for k, v in accs.items()})
    labels = ["random forest", "LightGBM", "logistic", "stacked"]
    vals = [accs["rf"], accs["lgbm"], accs["logreg"], accs["stack"]]
    colors = [ARM_BLUE, ARM_BLUE, ARM_BLUE, ARM_ORANGE]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    bars = ax.barh(labels, vals, color=colors, alpha=0.9)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.set_xlabel("validation accuracy")
    ax.set_xlim(0, 1.02)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "xgb_stacking.pdf")
    plt.close(fig)
    log.info("wrote xgb_stacking.pdf")


def fig_lightgbm_meme():
    """Outline gag: 'me and LightGBM' -- a lone warrior charging with sword and shield.
    Armenian caption baked in (pdflatex has no Armenian); the Outline frame hyperlinks it."""
    import matplotlib.image as mpimg
    img = mpimg.imread(str(FIG / "12_me_and_lightgbm.png"))
    h, w = img.shape[:2]
    x0, x1 = int(0.16 * w), int(0.84 * w)          # center-crop the very wide frame a bit
    img = img[:, x0:x1]
    hh, ww = img.shape[:2]
    caption = "LightGBM ♥: էլ ուրիշ մոդելի անուն չգիտե՞ս"
    fig, ax = plt.subplots(figsize=(6.4, 6.4 * hh / ww + 0.7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=13.5, color=ARM_BLUE, fontweight="bold", pad=7)
    fig.tight_layout()
    out = FIG / "lightgbm_meme.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    log.info(f"wrote {out.name}")


def fig_goss():
    """GOSS illustrated: most rows have a small |gradient| (already well fit) -- keep the top
    a% with the largest gradients, randomly sample b% of the rest, drop the others."""
    rng = np.random.default_rng(SEED)
    n = 2000
    g = np.abs(rng.normal(0, 1, n)) ** 1.6          # skewed: many small, few large
    a, b = 0.2, 0.1
    thr = np.quantile(g, 1 - a)
    bins = np.linspace(0, g.max(), 40)
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.hist(g[g < thr], bins=bins, color="0.78",
            label=f"small $|$grad$|$: keep only {int(b * 100)}% (random), drop the rest")
    ax.hist(g[g >= thr], bins=bins, color=ARM_RED, alpha=0.9,
            label=f"large $|$grad$|$: keep ALL (top {int(a * 100)}%)")
    ax.axvline(thr, color="0.25", ls="--", lw=1.5)
    ax.text(thr, ax.get_ylim()[1] * 0.92, " threshold", fontsize=8, color="0.3", va="top")
    ax.set_xlabel(r"$|$gradient$|$  (how wrong the model still is on that row)")
    ax.set_ylabel("number of rows")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "goss.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out.name} (threshold at top {a:.0%})")


if __name__ == "__main__":
    fig_histogram_split()
    fig_earlystop()
    fig_importance()
    fig_monotonic()
    fig_stacking()
    fig_lightgbm_meme()
    fig_goss()
    log.info("done")
