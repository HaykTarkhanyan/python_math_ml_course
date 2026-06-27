# Interpretability Deck 2 — PFI & Feature Effects (PDP / ICE) — Outline (v2)

Design doc for the **second deck of `ml/05_interpretability/`**. House style + SLIDE_STYLE. Intro-level.
Builds on deck 1 (built-in importance), feeds deck 3 (SHAP).

> v2 = self-review folded in (see changelog): CFI demoted to a one-liner, the chapter's single
> concrete **"importance ≠ causation"** example added here, extrapolation explained once + cross-ref'd.

## Scope / thesis
**Model-agnostic interpretability — works on *any* fitted model.** Two complementary questions:
**how much** does a feature matter (**PFI**) and **how** does it act (**PDP / ICE**). Honest
throughline (from deck 1): these describe the **model's behaviour, not causation**, and both break
under **correlated features** (extrapolation).

## Locked decisions
- **In scope:** PFI (permute → error rise; train vs test; error bars; the **correlation/
  extrapolation trap**); a single concrete **importance ≠ causation** example (the chapter's one
  instantiation of the slogan); **PDP** (effect shape) and **ICE** (per-instance; PDP = mean of ICE;
  ICE reveals what PDP averages away); ALE as a one-liner.
- **Out of scope** (mention only): **CFI / conditional sampling** (demoted to one line — the *trap*
  matters more than the *fix* at intro level), SAGE, LOCO, PIMP, knockoffs, causal-DAG formalism.
- **Depth:** intuition + how to read the plots. "No need to go super deep."
- **Style:** house style; reuse the LMU **PFI step animation** (CC-BY, credit), like L12/L14 reuse
  LMU pages. File `LNN_pfi_and_effects.tex`.
- **Prereq:** model-agnostic, so no hard ch4 dependency — but examples use a fitted (tree/forest)
  model; a 1-line "any trained model" note suffices.

## Source (`lecture_iml`, CC-BY 4.0) / reusable assets (credit on slide)
- PFI: `slides02-fi-pfi.tex`; assets `pfi_demo2.pdf` (animation), `pfi_test_vs_train.pdf`,
  `pfi_hexbin_extrapolation.pdf`, `pfi_interactions.pdf`, `bike-sharing02.png`,
  `conditional_sampling.pdf` (CFI one-liner).
- Effects: `slides02-fe-ice.tex`, `slides03-fe-pdp.tex`, `slides05/06-fe-ale*`; assets
  `pdp_bike.pdf`/`PD.pdf` (PDP), `ICE.pdf`/`ice_motivation.pdf` (ICE), `pdp_xor.pdf` (PDP hides
  interaction), `ale_vs_pdp_*` (ALE).
- VERIFY: sklearn `permutation_importance`, `PartialDependenceDisplay(kind="both")`.

## Callbacks
- **Deck 1**: impurity importance was biased & train-based → PFI is the model-agnostic fix.
- **L14**: the **permutation** idea returns (there: rebalance; here: destroy a feature's info).
- **L12**: loss/error; train-vs-test (leakage logic).

## Frame-by-frame (~12 frames)
### Hook
1. **A black box scores well — which features matter, and how do they act?** Model-agnostic tools
   answer both, for any model (bridge from deck 1).
### Outline
### Section 1 — Permutation Feature Importance (how much)
2. **PFI idea** — "make a feature useless (permute it), see how much error rises":
   `risk(permuted) − risk(original)`, averaged over repetitions.
3. **PFI step-by-step** — reuse `pfi_demo2` animation (perturb → predict → aggregate); bike PFI bar.
4. **Train vs test** — PFI on train exposes overfit features; PFI on test = what generalizes.
   (`pfi_test_vs_train`)
5. **The correlation/extrapolation trap** — permuting a correlated feature creates **unrealistic
   inputs** → inflated importance (hexbin figure). *One line:* "conditional variants (CFI) fix this
   by permuting within groups." (trap = the lesson; CFI = a mention)
6. **Importance ≠ causation — a concrete example** — the chapter's one instantiation: a feature
   **correlated with the true cause** (e.g. ice-cream sales ↔ drownings via temperature, or a proxy
   of a real driver) gets high PFI because the *model* uses it — yet intervening on it changes
   nothing. PFI = the model's reliance, **not** the world. (one clear toy, referenced from deck 1)
### Section 2 — Feature effects (how)
7. **[plain] From "how much" to "how"** — importance is one number; effects show the shape.
8. **PDP** — average prediction as one feature varies (others marginalized); read the shape
   (flat / monotone / non-linear). *Same extrapolation caveat as PFI — cross-reference frame 5, don't
   re-explain.* (`pdp_bike`)
9. **ICE** — one curve per instance; **PDP = the average of the ICE curves**; ICE shows
   heterogeneity/interactions the PDP hides. **Predict-first:** PDP looks flat — does the feature
   not matter? (Reveal: opposing per-instance effects cancel; ICE exposes it.) (`ice_motivation`/`pdp_xor`)
10. **ALE one-liner** — the correlation-robust alternative to PDP (uses local changes); `ale_vs_pdp`
    figure. *(Cut if the deck runs long — open decision.)*
### Wrap-up
11. **Recap** — PFI = global importance, PDP/ICE = global effect shape; both model-behaviour, both
    correlation-sensitive. "Next:" → **SHAP** unifies local + global, with great plots (deck 3).

## Figures
- **Reuse (CC-BY, credit):** the assets above (animation + PFI/PDP/ICE).
- **Generate (light, optional):** PFI bar + PDP+ICE on a course model
  (`permutation_importance`, `PartialDependenceDisplay(kind="both")`) for continuity; the
  "≠ causation" toy (frame 6) on synthetic correlated data.
- Dependency: none beyond `ma`.

## Open decisions
1. Reuse LMU bike figures vs regenerate PFI/PDP/ICE on a course dataset.
2. Keep the ALE one-liner+figure or cut ALE entirely.
3. One sklearn code frame (`permutation_importance` + `PartialDependenceDisplay`)? Lean yes.

## Changelog (v1 → v2)
1. **CFI demoted** from a full frame to a one-liner inside the trap frame (5) — the *trap* is the
   intro-level lesson; the conditional-sampling *fix* is graduate-ish.
2. **Added the chapter's single concrete "importance ≠ causation" example** (frame 6), so the
   throughline is *shown* once rather than sloganized across decks.
3. **Extrapolation explained once** (frame 5) and **cross-referenced** from PDP (frame 8) instead of
   re-explained.
