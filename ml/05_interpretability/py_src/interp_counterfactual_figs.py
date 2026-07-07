"""Worked numbers for the counterfactual add to Interpretability Deck 3.

Trains a classifier on the ch3 bank-marketing data (drops the leakage feature `duration`),
picks one client predicted "won't subscribe" near the boundary, and uses DiCE to find a
SPARSE, ACTIONABLE counterfactual that flips the prediction. Logs the original vs CF feature
values + predicted probabilities so the deck's before/after table uses REAL numbers.

Actionability is enforced: we freeze immutable / past-fact features (age, job, education,
poutcome, pdays, previous) and only let the bank-controllable / financial-state features vary.

Run: ./ma/Scripts/python.exe ml/05_interpretability/py_src/interp_counterfactual_figs.py
Conventions: logging to console + logs/, seed 509, f-strings. dice_ml already in ma.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import dice_ml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 509
HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
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


if __name__ == "__main__":
    main()
