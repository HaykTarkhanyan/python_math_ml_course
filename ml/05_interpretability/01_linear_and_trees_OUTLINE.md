# Interpretability Deck 1 — Interpreting Linear Models & Trees (built-in) — Outline (v3)

Design doc for the **first deck of `ml/05_interpretability/`**. House style + SLIDE_STYLE. Intro-level.

> Chapter: (1) **this deck**; (2) PFI + feature effects (PDP/ICE); (3) SHAP + LIME.
> Deck file `LNN_*.tex`, `NN` continues the playlist.
> v3 = instructor interview folded in (see changelog): **bike-sharing** anchor (reuse LMU CC-BY
> figures), **ch4 trees assumed done → brief callback**, **Lasso/L1 added** as a feature-selection
> angle, **worked info-gain example** added on the tree side.

## Scope / thesis
**Some models you can just *read* — interpretability is free, before any method.** Two glass-box
families: **linear models** (coefficients / odds ratios / Lasso selection) and **a single tree**.
Honest counterpoint: a **forest is *not* readable**, and its **built-in importance is biased** —
motivating decks 2–3.

## Locked decisions (from interview)
- **Anchor dataset: bike-sharing** — reuse LMU's ready CC-BY figures where possible (least new work).
  *Note:* bike-sharing is **regression**, so the tree importance uses **variance reduction**; the
  worked split example below uses **information gain on a tiny classification toy** (the canonical
  "info gain"), with a one-line note that variance reduction is its regression analog.
- **ch4 (trees) is assumed delivered** → impurity / info gain referenced with a **1-line reminder**,
  not a full recap.
- **Linear side includes Lasso/L1 as built-in feature selection** (extra frame): L1 shrinks
  coefficients to exactly zero → surviving features are the "selected" ones (callback to ch2
  regularization).
- **Tree side includes a worked info-gain example** (by hand) + aggregation into importance + biases.
- **In scope:** coef / odds ratios + scaling (L11/L12 recap, brief); Lasso selection; single-tree
  reading; forest impurity importance + worked info-gain + biases; the "methods disagree" punch.
- **Out of scope:** GLM/GAM/EBM; PDP (deck 2); SHAP (deck 3); the "≠ causation" *example*
  (deck 2, referenced here).
  > **Note (post-v3):** RuleFit was later folded in as its own `\section{Rule-based models}` (2 content
  > frames + a `rulefit_scorecard.pdf` figure) - see `ADDITIONS_OUTLINE.md`. This outline predates that add.
- **Style:** hook, `[plain]` transitions, a worked-numbers frame, Recap + "Next:" box, no HW frame.
  File `LNN_linear_and_trees.tex`.

## Source / callbacks
- **ch4 trees** (info gain / impurity, RF importance), **L11** (odds ratios), **L12** (standardize;
  correlated-coef instability), **ch2 [07] regularization** (Lasso/L1). `lecture_iml/interpretable-models`
  for framing only.
- VERIFY at build: sklearn `feature_importances_`, `plot_tree`, `Lasso`/`LogisticRegression(penalty="l1")`;
  impurity-importance biases (high-cardinality / correlated; Strobl et al.).

## Frame-by-frame (~11 frames)
### Hook
1. **Two routes to interpretability** — *use a glass-box model you can read* (this deck) vs *explain
   a black box afterwards* (decks 2–3).
2. **Predict-first:** a linear model and a random forest score the same — which can you actually
   *read*? (Reveal: linear model & a *single* tree yes; a *forest* no.)
### Outline
### Section 1 — Linear models
3. **Coefficients & odds ratios as importance** — sign = direction, magnitude = effect; `exp(coef)`
   = odds ratio (L11, brief); **standardize** to compare across features (L12).
4. **Linear caveats** — only **additive/linear** effects; **correlated features** split & destabilize
   the credit.
5. **Lasso / L1 = built-in feature selection** — L1 drives coefficients to **exactly zero**; the
   non-zero ones are "selected" → a sparsity-based importance (callback ch2 regularization). Caveat:
   among correlated features L1 keeps one ~arbitrarily.
### Section 2 — Trees
6. **A single tree is a glass box; a forest is not** — read a tree's root-to-leaf path as a rule;
   but **forests/boosting are black boxes** you can't read whole. (figure: small tree, bike)
7. **How a split is chosen — worked, both criteria** (A+B) — one frame, two columns: **left**
   classification **information gain** on a tiny toy (parent entropy/Gini − weighted child impurity);
   **right** **variance reduction** on a real bike split (parent variance − weighted child variance).
   Same idea, two impurity measures; the best-reduction split is chosen. (split into two frames if cramped)
8. **Forest built-in importance** — sum each feature's gain (info gain / variance reduction) over its
   splits, averaged over trees → a number *even when the forest is unreadable*. (reuse bike importance fig)
9. **Its biases** — favors **high-cardinality / continuous** features; **dilutes** across correlated
   features; computed on **train** (can reflect overfit) → don't trust blindly.
### Section 3 — Punchline + bridge
10. **Different methods, different answers** — linear-coef importance vs tree-impurity importance on
    the *same* data **disagree** because they measure different things (not because one is wrong);
    importance is **method-dependent**. (one generated figure)
11. **When built-in is enough — and the bridge** — fast & free, but model-specific and biased; black
    boxes / fair comparison need **model-agnostic** tools, and none of this is *causation* (deck 2).
    Recap + "Next:" → PFI & feature effects.

## Figures
- **Reuse (CC-BY, credit):** LMU bike importance figure(s); a tree plot if available.
- **Generate (light, on bike-sharing — download UCI bike data like bank-marketing):** the "methods
  disagree" bar (standardized linear coef vs RF impurity); a small `plot_tree`; the worked info-gain
  is a hand-drawn TikZ/table (tiny classification split, no data needed).
- Dependency: none beyond `ma` (sklearn). Need the bike-sharing CSV for the generated bars.

## Open decisions
1. Worked split example: keep **info-gain on a classification toy** (as planned) **or** switch to
   **variance-reduction on bike** for full consistency with the regression anchor? (flagged tension)
2. One sklearn code frame (`feature_importances_` + `plot_tree` + `Lasso`)? Lean yes.

## Changelog
**v2 → v3 (interview):** bike-sharing anchor + reuse LMU figures; ch4 assumed done (callback, no full
recap); **added Lasso/L1 feature-selection frame** (5); **added worked info-gain frame** (7) with the
regression/variance-reduction note; flagged the regression-vs-info-gain tension as open decision 1.
**v1 → v2 (self-review):** tree-vs-forest distinction; compressed linear half; ch4 dependency locked;
"≠ causation" example moved to deck 2.
