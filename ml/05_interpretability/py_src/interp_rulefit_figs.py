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


def tree_rules(tree, X, feature_names):
    """Every root-to-node path of a fitted sklearn tree, as (rule string, boolean mask).

    RuleFit's step 1: a path is a conjunction of split conditions, and the rule is 1 when a
    row satisfies all of them. The root itself is skipped (it is the trivial "always true" rule).
    """
    t = tree.tree_
    out = []

    def walk(node, conds, mask):
        if conds:
            out.append((" and ".join(conds), mask.copy()))
        if t.children_left[node] == -1:
            return
        f, thr = feature_names[t.feature[node]], t.threshold[node]
        col = X[:, t.feature[node]]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.2f}"], mask & (col <= thr))
        walk(t.children_right[node], conds + [f"{f} > {thr:.2f}"], mask & (col > thr))

    walk(0, [], np.ones(len(X), dtype=bool))
    return out


def fig_step1_rules(df, log):
    """Step 1: one shallow tree -> one rule per root-to-node path (deck 1's own tree)."""
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    X, y = df[FEATURES].values, df["cnt"].values
    tree = DecisionTreeRegressor(max_depth=2, random_state=SEED).fit(X, y)
    rules = tree_rules(tree, X, FEATURES)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.8),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    plot_tree(tree, feature_names=FEATURES, filled=True, rounded=True, impurity=False,
              proportion=True, fontsize=8, precision=2, ax=a1)
    a1.set_title("one shallow tree from the ensemble", fontsize=11)

    supports = [m.mean() for _, m in rules]
    labels = [r.replace(" and ", " &\n") for r, _ in rules]
    order = np.argsort(supports)
    bars = a2.barh(range(len(rules)), [supports[i] for i in order], color=ARM_BLUE, height=0.62)
    a2.set_yticks(range(len(rules)), [labels[i] for i in order],
                  fontsize=8, fontfamily="monospace")
    a2.bar_label(bars, labels=[f"{supports[i]:.0%}" for i in order], fontsize=8.5, padding=3)
    a2.set_xlabel("support (share of days where the rule fires)", fontsize=9)
    a2.set_xlim(0, 1.15)
    a2.set_title(f"every path becomes a binary rule ({len(rules)} from this tree)", fontsize=11)
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rulefit_step1_rules.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"STEP 1: depth-2 tree yields {len(rules)} rules")
    for r, m in rules:
        log.info(f"    support={m.mean():.2f}  {r}")
    log.info("wrote rulefit_step1_rules.pdf")


def rule_matrix(rules, df, log):
    """Evaluate rule strings into a 0/1 design matrix. Fails loudly on an unparseable rule."""
    cols = {}
    for i, r in enumerate(rules):
        try:
            cols[f"r{i + 1}"] = df.eval(r).astype(float)
        except Exception:
            log.error(f"could not evaluate rule {i + 1}: {r!r}")
            raise
    return pd.DataFrame(cols, index=df.index)


def fig_step2_design(df, rules, log):
    """Step 2: the design matrix grows - original columns PLUS one column per rule."""
    from sklearn.preprocessing import StandardScaler
    n_days, n_rules = 40, 14
    # Sample days ACROSS the two years, not the first 40 (those are all winter 2011, so every
    # rule column comes out constant and the picture shows nothing).
    rows = np.linspace(0, len(df) - 1, n_days).astype(int)
    # ...and show the rules that actually vary (mid support), for the same reason.
    R_all = rule_matrix(rules, df, log)
    supp = R_all.mean()
    varied = supp[(supp > 0.2) & (supp < 0.8)].index[:n_rules]
    if len(varied) < n_rules:
        log.error(f"only {len(varied)} rules with mid support; expected >= {n_rules}")
        raise RuntimeError("not enough varied rules for the design-matrix figure")
    R = R_all.loc[rows, varied]
    Xs = pd.DataFrame(StandardScaler().fit_transform(df[FEATURES]),
                      columns=FEATURES).iloc[rows]
    show = varied

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.6),
                                 gridspec_kw={"width_ratios": [len(FEATURES), len(show)]})
    a1.imshow(Xs.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    a1.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=7.5)
    a1.set_ylabel(f"{n_days} days spread across both years", fontsize=9)
    a1.set_title("original features (continuous)", fontsize=10.5)

    a2.imshow(R.values, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    a2.set_xticks(range(len(show)), list(show), rotation=90, fontsize=7.5)
    a2.set_yticks([])
    a2.set_title(f"+ rule features (binary, {len(rules)} of them)", fontsize=10.5)
    fig.suptitle("Step 2: rules join the design matrix as extra 0/1 columns", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rulefit_step2_design.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"STEP 2: design matrix = {len(FEATURES)} originals + {len(rules)} rule columns")
    log.info("wrote rulefit_step2_design.pdf")


def fig_step3_sparsity(df, rules, log):
    """Step 3: Lasso over [originals | rules] - sparsity and accuracy as one dial."""
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    R = rule_matrix(rules, df, log)
    D = pd.concat([df[FEATURES].reset_index(drop=True), R.reset_index(drop=True)], axis=1)
    y = df["cnt"].values
    Xtr, Xte, ytr, yte = train_test_split(D.values, y, test_size=0.3, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    cv = LassoCV(cv=5, random_state=SEED, max_iter=50000).fit(Xtr, ytr)
    alphas = np.logspace(np.log10(cv.alpha_ * 0.3), np.log10(cv.alpha_ * 120), 40)
    n_terms, r2 = [], []
    for a in alphas:
        m = Lasso(alpha=a, max_iter=50000).fit(Xtr, ytr)
        n_terms.append(int(np.sum(m.coef_ != 0)))
        r2.append(r2_score(yte, m.predict(Xte)))

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(alphas, n_terms, color=ARM_BLUE, lw=2.2, label="terms kept")
    ax.set_xscale("log")
    ax.set_xlabel("Lasso penalty $\\alpha$  (stronger $\\rightarrow$)", fontsize=10)
    ax.set_ylabel("non-zero terms (rules + linear)", fontsize=10, color=ARM_BLUE)
    ax.tick_params(axis="y", labelcolor=ARM_BLUE)

    ax2 = ax.twinx()
    ax2.plot(alphas, r2, color=ARM_RED, lw=2.2, ls="--", label="test $R^2$")
    ax2.set_ylabel("test $R^2$", fontsize=10, color=ARM_RED)
    ax2.tick_params(axis="y", labelcolor=ARM_RED)

    k_cv = int(np.sum(cv.coef_ != 0))
    r2_cv = r2_score(yte, cv.predict(Xte))
    ax.axvline(cv.alpha_, color="0.35", ls=":", lw=1.5)
    ax.annotate(f"CV pick: {k_cv} terms, $R^2$={r2_cv:.2f}",
                xy=(cv.alpha_, k_cv), xytext=(0.30, 0.88), textcoords="axes fraction",
                fontsize=9.5, arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))
    ax.set_title(f"Step 3: the penalty decides how many of {D.shape[1]} candidate terms survive",
                 fontsize=11.5)
    ax.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rulefit_step3_sparsity.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"STEP 3: {D.shape[1]} candidate terms ({len(FEATURES)} linear + {len(rules)} rules)")
    log.info(f"  LassoCV alpha={cv.alpha_:.2f} -> keeps {k_cv} terms, test R^2={r2_cv:.3f}")
    for a, k, r in zip(alphas[::8], n_terms[::8], r2[::8]):
        log.info(f"    alpha={a:8.2f} -> {k:3d} terms, test R^2={r:.3f}")
    log.info("wrote rulefit_step3_sparsity.pdf")


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

    # --- step-by-step figures for the "how RuleFit works" frames ---
    # The scorecard above is a deliberately SMALL model (max_rules=15 -> 13 readable terms).
    # The mechanism frames use a full-size run so step 3 shows a real sparsity dial: a big
    # ensemble mines ~100 candidate rules, and Lasso decides how many survive.
    big = RuleFitRegressor(random_state=SEED, max_rules=200)
    big.fit(X.values, y, feature_names=FEATURES)
    big_rules = [r for r, t in zip(big._get_rules()["rule"], big._get_rules()["type"])
                 if t == "rule"]
    log.info(f"candidate rules mined by the full-size run: {len(big_rules)}")
    fig_step1_rules(df, log)
    fig_step2_design(df, big_rules, log)
    fig_step3_sparsity(df, big_rules, log)


if __name__ == "__main__":
    main()
