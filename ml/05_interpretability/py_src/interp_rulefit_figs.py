"""Worked numbers for the RuleFit add to Interpretability Deck 1 (rule-based models).

Fits an imodels RuleFitRegressor on the bike data (deck 1's anchor) and logs the rules
it keeps (rule string, coefficient, support) so the deck's scorecard table uses REAL
numbers. No figure file - the deck shows a LaTeX tabular of these rules.

Run: ./ma/Scripts/python.exe ml/05_interpretability/py_src/interp_rulefit_figs.py
Conventions: logging to console + logs/, seed 509, f-strings.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from imodels import RuleFitRegressor
from sklearn.metrics import r2_score

SEED = 509
HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
DATA = CH_DIR / "data" / "bike-day.csv"
LOGS_DIR = REPO_ROOT / "logs"
FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed"]


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("rulefit"); log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "interp_rulefit.log")):
        h.setFormatter(fmt); log.addHandler(h)
    return log


def main():
    log = setup_logging()
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["cnt"].values
    log.info(f"bike: X={X.shape}, mean(cnt)={y.mean():,.0f} (rough scorecard baseline)")

    rf = RuleFitRegressor(random_state=SEED, max_rules=15)
    rf.fit(X.values, y, feature_names=FEATURES)
    log.info(f"train R^2 = {r2_score(y, rf.predict(X.values)):.3f}")

    rules = rf._get_rules()
    kept = rules[rules.coef != 0].copy()
    kept["abscoef"] = kept["coef"].abs()
    kept = kept.sort_values("importance", ascending=False)
    log.info(f"kept {len(kept)}/{len(rules)} terms (rules + linear)")
    log.info("TOP TERMS by importance (rule | type | coef | support):")
    for _, r in kept.head(12).iterrows():
        log.info(f"  [{r['type']:<6}] coef={r['coef']:+9.1f}  support={r['support']:.2f}  | {r['rule']}")


if __name__ == "__main__":
    main()
