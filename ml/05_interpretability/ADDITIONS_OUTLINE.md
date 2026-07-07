# Chapter 5 additions outline: RuleFit (deck 1) + Counterfactuals (deck 3)

Folded into existing decks (no new deck). Both are "a bit" / not-super-deep adds.
Decisions locked with instructor: fold-in; rule mining = RuleFit (intrinsic glass-box);
counterfactuals on bank-marketing (ch3 callback); depth = concept + properties + 1 example + code.

---

## Deck 1 - `01_linear_and_trees.tex`: new section "Rule-based models"

**Placement:** new `\section{Rule-based models}` AFTER "Reading trees", BEFORE "So which
features matter?" - rules are the synthesis of the two glass-box families just taught.

**Also touch:** hook TikZ glass-box label gets "+ rules"; recap gets a rules bullet.

**Frames (1 transition + 2 content):**

1. `[plain]` transition - popblue "Rule-based models" + line
   "Rules you can read - a tree's logic with a linear model's sparsity."

2. **"RuleFit: rules as features"** (the idea)
   - Single tree: readable but brittle. Forest: accurate but unreadable. RuleFit wants both:
     1. Extract many candidate rules from a tree ensemble - each root-to-node path is a
        binary rule ("is this row in that region?", e.g. `temp>0.43 AND yr=1`).
     2. Fit a sparse linear model (Lasso) over [original features + all rule features].
     3. Non-zero weights = the handful of rules/terms the model keeps.
   - armblue!8 callback box: RuleFit = trees (rules, this deck) + Lasso selection (linear
     section, this deck) combined into one glass box.

3. **"Reading a RuleFit model"** (worked example, bike - REAL numbers from imodels fit)
   - tabular: top rules/terms by |coef|, e.g.
     | rule / term            | weight |
     | temp > 0.43 AND yr = 1 |  +NNN  |
     | hum > 0.80             |  -NNN  |
     | yr  (linear term)      |  +NNN  |
   - "Read it as a scorecard: start at the baseline, add the weight of every rule that
     fires for this row."
   - armorange!12 watch-out: more rules -> more accurate but less readable (same
     depth/readability tradeoff as a tree); overlapping rules share credit (correlation again).

---

## Deck 3 - `03_shap_lime.tex`: new section "Counterfactual explanations"

**Placement:** new `\section{Counterfactual explanations}` AFTER "LIME", BEFORE the
"Same data, four rankings" wrap-up.

**Also touch:** "Which method when?" table gets a row
("what would change this outcome -> counterfactuals"); recap gets a counterfactual bullet.

**Frames (1 transition + 3 content):**

1. `[plain]` transition - popblue "Counterfactual explanations" + line
   "Not why - but what would change the outcome."

2. **"The smallest change that flips it"** (idea)
   - SHAP/LIME answer "why this prediction?" Counterfactuals answer a different, often more
     actionable question: "what is the smallest change to the input that flips the prediction?"
   - Needs a decision -> switch to the ch3 bank-marketing model (will this client subscribe?).
     Flag the dataset switch (bike was regression; "flip" needs a class).
   - Plain-language example: "This client is predicted *won't subscribe*. Counterfactual:
     contacted in a different month with a prior-campaign success, the model flips to
     *will subscribe*." Tells you what to change, not just what mattered.

3. **"What makes a good counterfactual?"** (5 properties + worked example - REAL CF from DiCE)
   - Validity - it actually flips the prediction.
   - Proximity - close to the original (small change).
   - Sparsity - few features changed (easy to act on and to explain).
   - Plausibility - stays in realistic data, no impossible rows (callback: deck 2 correlation
     trap / extrapolation).
   - Actionability - only change features the user controls (raise balance, not lower age;
     never protected attributes). The recourse / fairness angle.
   - before/after tabular (bank, real): original (pred: no, p=0.NN) vs counterfactual with
     changed cells highlighted (pred: yes, p=0.NN).
   - note: usually several valid counterfactuals exist - prefer sparse + actionable.

4. `[fragile]` **"Counterfactuals in code"** (DiCE)
   - dice_ml snippet: Data + Model -> Dice -> generate_counterfactuals(query, total_CFs=3,
     desired_class="opposite").
   - one-liner: libraries DiCE (diverse CFs), Alibi.

---

## Figures / scripts

No new PDF figures - both adds are LaTeX tabulars filled with REAL numbers logged by:

- `py_src/interp_rulefit_figs.py` - fit imodels RuleFit on bike (deck 1's data); log top
  rules + coefficients + a baseline. (needs `imodels`, not yet installed)
- `py_src/interp_counterfactual_figs.py` - train a classifier on bank-marketing (drop the
  leakage feature `duration`); use DiCE to generate a sparse counterfactual for one "no"
  client; log original/CF feature values + predicted probabilities. (`dice_ml` already in `ma`)

New dependency to install in `ma`: **imodels** (RuleFit). `dice_ml` already present.
