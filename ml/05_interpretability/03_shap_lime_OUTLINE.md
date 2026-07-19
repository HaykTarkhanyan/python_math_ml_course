# Interpretability Deck 3 — SHAP (plots & how to read them) + LIME — Outline (v2)

Design doc for the **third deck of `ml/05_interpretability/`** — the chapter's payoff deck.
House style + SLIDE_STYLE. Intro-level, intuition-first.

> v2 = self-review folded in (see changelog): **LIME trimmed** (it was over-weighted vs the
> "especially SHAP plots" steer), **SHAP-plot frames given more room**, a one-line "what makes
> Shapley fair," a **chapter-finale decision guide**, and figure plan reduced to generating only
> the missing waterfall.

## Scope / thesis
**SHAP is the modern standard for explaining individual predictions, and its practical value is
the *plots*. The deck's weight goes to the plot types and how to read them** — then **LIME** as the
lighter local cousin. No game-theory derivations.

## Locked decisions
- **In scope (SHAP, the bulk):** the fair-payout *intuition* (incl. one line on *what is averaged*);
  the **local** decomposition (`prediction = base value + Σ SHAP`); and the **plot zoo with reading
  instructions** — **waterfall/force** (one prediction), **bar** (global mean|SHAP|),
  **beeswarm/summary** (signature global plot), **dependence/scatter** (effect shape + interaction);
  **TreeSHAP** (why SHAP is the default for trees/boosting); SHAP unifies PFI + PDP + local.
- **In scope (LIME, light — ~2 frames):** local-surrogate idea (perturb → weight by proximity → fit
  a sparse linear model); the **husky-vs-wolf** image example; instability/locality caveats;
  SHAP-vs-LIME in one line.
- **Out of scope:** Shapley axioms/formula, KernelSHAP weighting math, SAGE.
  > **Note (post-v2):** counterfactuals were later folded in as their own `\section{Counterfactual
  > explanations}` (transition + 3 frames + a `cf_flip_bank.pdf` figure, DiCE on bank-marketing) - see
  > `ADDITIONS_OUTLINE.md`. A global-surrogate frame and a calibration one-liner were also added later.
  > This outline predates those adds.
- **Prereq:** TreeSHAP frame calls back to **ch4 trees/boosting** (1-line recap if ch4 not yet built).
- **Style:** house style; reuse LMU CC-BY plots, generate **only** the missing waterfall.
  File `LNN_shap_lime.tex`.

## Source (`lecture_iml`, CC-BY 4.0) / reusable assets (credit on slide)
- SHAP: `slides02-shapley-ml.tex`, `slides03-shap.tex`, `slides05-shap-global.tex`. Assets:
  `Shapley_1..9.png` (fair-payout animation), `shapley-bike.pdf`, `global_shap_fi.pdf` (**bar**),
  `global_shap_jitter.pdf` (**beeswarm**), `global_shap_depend*.pdf` (**dependence**).
  **Gap:** no waterfall/force asset → generate one with the `shap` library.
- LIME: `slides04-le-lime.tex`; assets `lime-wolfhusky*.png`, `lime_credit.pdf`, `lime_robustness*`.
- VERIFY: `shap` API (`Explainer`, `TreeExplainer`, `plots.waterfall/beeswarm/bar/scatter`);
  consistency property (state, don't prove).

## Callbacks
- **Decks 1 & 2**: built-in + PFI/PDP were *global*; SHAP adds a principled **local** view and a
  global view that agrees (SHAP **bar** ≈ PFI; SHAP **dependence** ≈ PDP).
- **ch4 trees/boosting**: TreeSHAP fast & exact → default there.
- **bank project / L11**: explain a *single* customer's prediction.

## Frame-by-frame (~16 frames; SHAP-heavy, LIME light)
### Hook
1. **The model says THIS customer churns — why?** Global importance can't answer per-instance →
   need a **local** explanation.
### Outline
### Section 1 — SHAP: the idea (2)
2. **Fair-payout intuition** — split a prediction's "credit" among features fairly, like a team
   bonus among players. *What makes it fair (one line):* average each feature's contribution over
   the different orders it could be added in. (reuse `Shapley_1..9` animation; no formula)
3. **Local decomposition** — `prediction = base value + Σ (per-feature SHAP)`; + pushes up, −
   pushes down. **Every SHAP plot visualizes this.**
### Section 2 — The SHAP plot zoo, how to read each (the heart, 4 frames)
4. **Waterfall / force (one prediction)** — from base value, add each feature's SHAP to reach this
   prediction. Read: which features moved *this* case, and which way. (generate)
5. **Bar (global)** — mean |SHAP| per feature = global importance; read like a PFI bar.
   (reuse `global_shap_fi`)
6. **Beeswarm / summary (the signature plot)** — one dot per instance per feature; x = SHAP value,
   **colour = feature value**. Read: spread = how much it matters; colour pattern = direction
   (e.g. high value → red → pushes up). (reuse `global_shap_jitter`)
7. **Dependence / scatter** — SHAP value vs feature value (per-instance dots); reveals **effect
   shape** (≈ PDP) and **interactions** (colour by a second feature). (reuse `global_shap_depend`)
### Section 3 — SHAP in practice (2)
8. **TreeSHAP** — exact & fast for trees/boosting → why SHAP is the **default** for tabular models;
   KernelSHAP for the rest (slow). One line, no math. (ch4 callback)
9. **SHAP unifies the chapter** — bar ≈ PFI (importance), dependence ≈ PDP (effect), **plus** a
   consistent local view. Caveat: still model-behaviour, not causation; easy to over-read.
### Section 4 — LIME (light, 2)
10. **[plain] LIME: the lighter local cousin** — fit a simple **local surrogate** (perturb around
    the instance, weight by proximity, fit a sparse linear model); the **husky-vs-wolf** "it learned
    *snow*" example shows what it catches. (reuse `lime-wolfhusky`, `lime_credit`)
11. **LIME caveats + SHAP vs LIME** — LIME is **unstable** (re-run → different) and its
    **neighbourhood is arbitrary**; SHAP has theory + better plots but is slower. SHAP = default,
    LIME = quick/simple. (reuse `lime_robustness`)
### Wrap-up
12. **Which method when? (chapter decision guide)** — glass-box model → *read it* (deck 1);
    black-box global importance → *PFI / SHAP-bar*; effect shape → *PDP / SHAP-dependence*; one
    prediction → *SHAP-waterfall / LIME*. (the chapter finale)
13. **Recap** — read the four SHAP plots; SHAP unifies local+global; LIME is the simpler cousin;
    all explain the *model*, not the world — don't over-read. "Next:" box (future: counterfactuals).

## Figures
- **Reuse (CC-BY, credit):** `Shapley_1..9`, `global_shap_fi`, `global_shap_jitter`,
  `global_shap_depend`, `lime-wolfhusky*`, `lime_credit`, `lime_robustness`.
- **Generate (only the gap):** a **waterfall** on a course model (the one plot with no LMU asset);
  optionally a beeswarm on course data for continuity.
- **Dependency:** add `shap` to `ma` (`uv pip install --python ./ma/Scripts/python.exe shap`) for
  the waterfall; LIME stays reuse-only (no `lime` dep needed).

## Open decisions
1. Generate the waterfall (needs `shap`) vs reuse-only (no local-plot example)? Lean generate (it's the gap).
2. Which course model for the generated plot — bank-marketing classifier or rent regression?
3. Keep the chapter decision guide here (deck 3 finale) — confirm this is the right home.

## Changelog (v1 → v2)
1. **LIME trimmed** from a 5-frame section to **2 frames** (10–11), matching the "especially SHAP
   plots, *maybe* LIME" steer; the reclaimed room goes to the SHAP plot frames.
2. **SHAP fair-payout intuition** gains a one-line **"what is averaged"** (contribution over feature
   orderings) so it isn't a black-box explanation of a black-box explainer.
3. **Chapter-finale decision guide added** (frame 12) — "which method when?" across all three decks.
4. **Figure plan reduced** — generate only the missing **waterfall**; reuse the rest (bar/beeswarm/
   dependence) → minimal new work + only the `shap` dep.
5. TreeSHAP frame flagged as a **ch4 callback** (1-line recap fallback).
