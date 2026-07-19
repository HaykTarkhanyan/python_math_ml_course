"""Figure + worked numbers for the counterfactual add to Interpretability Deck 3.

Trains a classifier on the ch3 bank-marketing data (drops the leakage feature `duration`),
picks one client predicted "won't subscribe" near the boundary, and uses DiCE to find a
SPARSE, ACTIONABLE counterfactual that flips the prediction. Logs the original vs CF feature
values + predicted probabilities, and draws:
  - cf_flip_bank.pdf : the before/after probability of subscribing, crossing the 0.5 decision
                       threshold, with the one changed feature called out (the smallest change
                       that flips the decision).

Actionability is enforced: we freeze immutable / past-fact features (age, job, education,
poutcome, pdays, previous) and only let the bank-controllable / financial-state features vary.

Run: ./ma/Scripts/python.exe ml/05_interpretability/py_src/interp_counterfactual_figs.py
Conventions: logging to console + logs/, seed 509, f-strings, Armenian-flag colours. dice_ml in ma.
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dice_ml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"
ARM_GREEN = "#008C46"
HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
BANK = CH_DIR.parent / "03_classification" / "data" / "bank-full.csv"
LOGS_DIR = REPO_ROOT / "logs"

NUM = ["age", "balance", "day", "campaign", "pdays", "previous"]
# bank-controllable levers only (relatable recourse story); financial state / past facts frozen
ACTIONABLE = ["contact", "month", "campaign", "housing", "loan"]


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("cf"); log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "interp_cf.log")):
        h.setFormatter(fmt); log.addHandler(h)
    return log


def fig_flip(best, feats, log):
    """Before/after probability bar: the smallest change crossing the 0.5 decision line."""
    ncols, _, rank, p0, p1, cols, q, cf_row = best
    changes = [f"{c}: {q.iloc[0][c]} $\\to$ {cf_row.iloc[0][c]}" for c in cols]

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.barh([1], [p0], color=ARM_RED, height=0.55)
    ax.barh([0], [p1], color=ARM_GREEN, height=0.55)
    ax.axvline(0.5, color="0.3", ls="--", lw=1.4)
    ax.text(0.5, 1.75, "decision threshold (0.5)", ha="center", va="center",
            fontsize=8, color="0.3")
    ax.text(p0 + 0.02, 1, f"p(subscribe) = {p0:.2f}", va="center", fontsize=10)
    ax.text(p1 + 0.02, 0, f"p(subscribe) = {p1:.2f}", va="center", fontsize=10)
    ax.set_yticks([1, 0], ["original\n(won't subscribe)", "counterfactual\n(will subscribe)"],
                  fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 2.1)
    ax.set_xlabel("model's predicted probability of subscribing", fontsize=9)
    label = "  ".join(changes)
    ax.set_title(f"The smallest change that flips it: {label}", fontsize=10.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "cf_flip_bank.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote cf_flip_bank.pdf (client {rank}: {label}; p {p0:.2f}->{p1:.2f})")


def main():
    log = setup_logging()
    df = pd.read_csv(BANK, sep=";")
    df = df.drop(columns=["duration"])                 # leakage: known only after the call
    df["y"] = (df["y"] == "yes").astype(int)
    feats = [c for c in df.columns if c != "y"]
    cats = [c for c in feats if c not in NUM]
    log.info(f"bank: {df.shape}, subscribe rate = {df['y'].mean():.3f}; dropped 'duration'")

    Xtr, Xte, ytr, yte = train_test_split(df[feats], df["y"], test_size=0.3,
                                          random_state=SEED, stratify=df["y"])
    pipe = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats)],
                                   remainder="passthrough")),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED,
                                      class_weight="balanced", n_jobs=-1)),
    ]).fit(Xtr, ytr)

    # clients clearly predicted "no" (room for a meaningful flip, not a coin-flip nudge)
    p = pipe.predict_proba(Xte)[:, 1]
    mask = (yte.values == 0) & (p > 0.25) & (p < 0.45)
    idx = np.where(mask)[0]
    cand = idx[np.argsort(-p[idx])]
    log.info(f"{len(idx)} candidate 'no' clients in p(yes) in (0.25, 0.45)")

    train_df = Xtr.copy(); train_df["y"] = ytr.values
    d = dice_ml.Data(dataframe=train_df, continuous_features=NUM, outcome_name="y")
    m = dice_ml.Model(model=pipe, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    found = []
    for rank in cand[:30]:
        q = Xte.iloc[[rank]]
        p0 = pipe.predict_proba(q)[0, 1]
        try:
            res = exp.generate_counterfactuals(q, total_CFs=6, desired_class="opposite",
                                               features_to_vary=ACTIONABLE, random_seed=SEED)
        except Exception:
            continue
        cfdf = res.cf_examples_list[0].final_cfs_df
        if cfdf is None or len(cfdf) == 0:
            continue
        cfX = cfdf[feats]
        for i in range(len(cfX)):
            cf_row = cfX.iloc[[i]]
            p1 = pipe.predict_proba(cf_row)[0, 1]
            cols = [c for c in feats if str(cf_row.iloc[0][c]) != str(q.iloc[0][c])]
            if p1 >= 0.50 and 1 <= len(cols) <= 3:
                found.append((len(cols), -p1, rank, p0, p1, cols, q, cf_row))

    found.sort(key=lambda t: (t[0], t[1]))              # sparsest, then most decisive flip
    log.info(f"collected {len(found)} clean actionable flips (p1>=0.5, <=3 changes)")
    for ncols, negp1, rank, p0, p1, cols, q, cf_row in found[:5]:
        log.info(f"=== CLIENT {rank}: p(yes) {p0:.2f} -> {p1:.2f} ; changed {ncols}: {cols} ===")
        for c in feats:
            o, n = q.iloc[0][c], cf_row.iloc[0][c]
            mark = "  <-- CHANGED" if c in cols else ""
            log.info(f"    {c:<11} {str(o):>12}  ->  {str(n):>12}{mark}")

    if not found:
        raise RuntimeError("no clean actionable flip found - cannot build cf_flip_bank.pdf")
    fig_flip(found[0], feats, log)


if __name__ == "__main__":
    main()
