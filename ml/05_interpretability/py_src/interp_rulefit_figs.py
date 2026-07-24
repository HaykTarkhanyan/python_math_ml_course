"""Figure + worked numbers for the RuleFit add to Interpretability Deck 1 (rule-based models).

Fits an imodels RuleFitRegressor on the bike data (deck 1's anchor), logs the rules it keeps
(rule string, coefficient, support), and draws:
  - rulefit_scorecard.pdf : the kept rules as a diverging horizontal bar (the scorecard made
                            visual; positive weights push rentals up, negative push down).

Run: ./ma/Scripts/python.exe ml/05_interpretability/py_src/interp_rulefit_figs.py
Conventions: logging to console + logs/, seed 509, f-strings, Armenian-flag colours.
"""
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imodels import RuleFitRegressor
from sklearn.metrics import r2_score

SEED = 509
ARM_BLUE, ARM_RED, ARM_ORANGE = "#0033A0", "#D90012", "#F2A800"
HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
DATA = CH_DIR / "data" / "bike-day.csv"
LOGS_DIR = REPO_ROOT / "logs"
FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed"]


def _pretty(rule, rtype):
    """Compact, wrapped label for a rule/linear term (slide-legible, no truncation)."""
    if rtype == "linear":
        return f"{rule} (linear term)"
    # round the split thresholds (0.34851 -> 0.35) for legibility
    rule = re.sub(r"\d+\.\d+", lambda m: f"{float(m.group()):.2f}", rule)
    # tighten operators, then wrap on the conjunctions so nothing is cut off
    s = (rule.replace(" and ", " & ").replace(" <= ", "≤").replace(" > ", ">")
             .replace(" >= ", "≥").replace(" < ", "<"))
    parts, line, out = s.split(" & "), "", []
    for p in parts:
        cand = f"{line} & {p}" if line else p
        if len(cand) > 26 and line:
            out.append(line); line = p
        else:
            line = cand
    out.append(line)
    return "\n".join(out)


def fig_scorecard(kept, baseline, log):
    top = kept.head(6).iloc[::-1]                        # biggest at top of the bar
    labels = [_pretty(r["rule"], r["type"]) for _, r in top.iterrows()]
    coefs = top["coef"].values
    colors = [ARM_BLUE if c > 0 else ARM_RED for c in coefs]

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    bars = ax.barh(range(len(coefs)), coefs, color=colors)
    ax.set_yticks(range(len(coefs)), labels, fontsize=10, fontfamily="monospace")
    ax.axvline(0, color="0.3", lw=1)
    ax.bar_label(bars, labels=[f"{c:+.0f}" for c in coefs], fontsize=9.5, padding=3)
    ax.set_xlabel("weight added to the baseline when the rule fires (rentals)", fontsize=9.5)
    ax.set_title(f"RuleFit scorecard on bike (baseline $\\approx$ {baseline:,.0f} rentals)",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.18)                                   # room for the +/- labels
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rulefit_scorecard.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("wrote rulefit_scorecard.pdf")


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

    FIG_DIR.mkdir(exist_ok=True)
    fig_scorecard(kept, float(y.mean()), log)


if __name__ == "__main__":
    main()
