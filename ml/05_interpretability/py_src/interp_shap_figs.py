"""SHAP figures for Interpretability Deck 3, all generated on the bike RF (shap library).

  shap_waterfall_bike.pdf : one prediction (a MIXED instance: some features up, some down).
  shap_bar.pdf            : global importance, mean |SHAP|.
  shap_beeswarm.pdf       : global, one dot per row, colour = feature value.
  shap_dependence.pdf     : SHAP value of temp vs temp (effect shape), normalized temp (0-1).

(Reused LMU CC-BY in the deck: the Shapley fair-payout framing image, the LIME husky/robustness.)

Run: ./ma/Scripts/python.exe ml/05_interpretability/py_src/interp_shap_figs.py
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

SEED = 509
HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FIG_DIR = CH_DIR / "fig"
DATA = CH_DIR / "data" / "bike-day.csv"
LOGS_DIR = REPO_ROOT / "logs"
F = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
     "weathersit", "temp", "atemp", "hum", "windspeed"]


def _save(name):
    fig = plt.gcf()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def main():
    LOGS_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("shap")

    df = pd.read_csv(DATA)
    X, y = df[F], df["cnt"].values
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X, y)
    sv = shap.TreeExplainer(rf)(X)

    # --- waterfall: pick the MOST MIXED instance (largest balance of + and - SHAP) ---
    phi = sv.values
    pos = np.where(phi > 0, phi, 0).sum(1)
    neg = -np.where(phi < 0, phi, 0).sum(1)
    i = int(np.argmax(np.minimum(pos, neg)))
    s = pd.Series(phi[i], index=F)
    log.info(f"waterfall instance {i}: pred={rf.predict(X.iloc[[i]])[0]:.0f}, base={sv.base_values[i]:.0f}")
    log.info(f"  pushes UP:   {s[s>0].sort_values(ascending=False).head(3).round(0).to_dict()}")
    log.info(f"  pushes DOWN: {s[s<0].sort_values().head(3).round(0).to_dict()}")
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=9, show=False)
    plt.gcf().set_size_inches(7.4, 4.6)
    _save("shap_waterfall_bike.pdf")
    log.info("wrote shap_waterfall_bike.pdf")

    # --- global plots ---
    plt.figure()
    shap.plots.bar(sv, max_display=11, show=False)
    plt.gcf().set_size_inches(6.4, 4.4)
    _save("shap_bar.pdf"); log.info("wrote shap_bar.pdf")

    plt.figure()
    shap.plots.beeswarm(sv, max_display=11, show=False)
    plt.gcf().set_size_inches(6.8, 4.6)
    _save("shap_beeswarm.pdf"); log.info("wrote shap_beeswarm.pdf")

    plt.figure()
    # color=sv lets shap auto-pick the strongest-interacting feature for the colour axis,
    # so the "colour reveals interactions" claim on the slide is actually shown (+ a colorbar).
    shap.plots.scatter(sv[:, "temp"], color=sv, show=False)
    plt.gcf().set_size_inches(6.4, 4.4)
    _save("shap_dependence.pdf"); log.info("wrote shap_dependence.pdf")


if __name__ == "__main__":
    main()
