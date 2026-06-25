# L06 Classification + Logistic Regression — Deck Outline / Design (v2)

Design doc for building the **Classification + Logistic Regression** deck from
scratch, house style of
`ml/02_main_concepts/06_overfitting_cross_validation.tex` (the
validation/CV deck, formerly `L01d_validation_and_cv.tex`).

> v2 = independent pedagogical review folded in (see changelog). The companion deck
> **L07 classification metrics is ALREADY BUILT**
> (`ml/ch2_classification/L07_classification_metrics.tex`, 23 frames) — second
> house-style reference + source of the chapter running example + handoff target.

## Locked decisions
- **Scope:** L06 only — Classification basics + Logistic Regression (binary).
  Metrics = L07 (built). Multiclass = L08 (separate, later). Outline-first.
- **Style:** house style (L01d / L07) — hand-drawn TikZ/pgfplots, Armenian-flag
  palette, `\fcolorbox` callout boxes, `\pause` predict-first frames, hook frames
  before `\tableofcontents`, Recap + HW, provenance block. `\input{../preamble}`;
  `listings` `\lstset`.
- **Pedagogy:** lead with the **odds -> log-odds -> line -> invert -> sigmoid**
  INTUITION arc, then teach **log-loss INTUITION-first** (plot the penalty; the
  Bernoulli-likelihood derivation is an OPTIONAL boxed aside — the audience has not
  formally seen MLE). Include a **worked-numbers frame** (compute z, sigma(z), odds,
  odds-ratio) so the deck is operable, not just recitable.
- **Running example: the cheese factory (MATCH L07).** From L07's setup frame:
  1000 wheels/day, **~3% bad (positive = contaminated/bad batch)** vs ~97% good;
  model outputs a risk *score/probability*; asymmetric cost. L06 BUILDS the
  classifier L07 EVALUATES.
  - For the clean S-curve (frame 8) and 2D boundary (frame 9), use ONE strong feature
    / a balanced view and **label the plot "illustrative, balanced for clarity — real
    factory data is 3% positive (L07)."** Do not let it imply 0.5 is the natural
    operating point. Align exact feature names with L07 at build time.
- **File:** write to `ml/ch2_classification/L06_classification_logreg.tex`,
  overwriting the skeleton (7 frames / 5 TODOs). Compile from `ch2_classification/`.

## Sources
- **ml_old**: `Chapter 2 Classification/PDFs/C2 01 Classification + Logistic
  Regression.pdf` (32 pp).
- **Built reference**: `ml/ch2_classification/L07_classification_metrics.tex`.
- **Upstream (raw)**: `ml_old/slides-i2ml-1-300.pdf` (classification chapter).
- VERIFY at build: sklearn `LogisticRegression` defaults (`penalty="l2"`, `C=1.0`,
  `solver="lbfgs"`, `max_iter`), `predict_proba`, `class_weight="balanced"`.

## Callbacks / cross-references
- **ch1**: linear regression / ERM; gradient descent (how we fit log-loss);
  regularization L03 (separable -> regularize); scaling `03_data_preprocessing.tex`
  (logreg with GD needs scaled features).
- **Forward**: L07 metrics (threshold, ROC/PR, asymmetric cost — same cheese);
  L08 multiclass.

## Frame-by-frame outline (~20 frames)

### Hook
1. **From regression to classification** — you predicted rent (a number); now predict
   a CATEGORY: is this cheese batch bad? Target categorical. Quick example zoo here
   (spam, credit, churn, illness, cheese) as motivation. (v2: examples merged in from
   the old standalone frame.)
2. **Predict-first:** what goes wrong if we fit a plain line to 0/1 labels? Reveal:
   predictions <0 / >1, fake "probabilities", outlier-sensitive -> motivates logreg.

### Outline frame (`\tableofcontents`)

### Section 1 — Classification basics
3. Binary vs multiclass distinction; this deck = binary; multiclass -> L08. (v2:
   trimmed — example zoo moved to the hook.)
4. Probabilistic classifiers + thresholding — output a SCORE/probability, threshold
   at 0.5 to get a class; the threshold is a CHOICE (forward-ref L07).
   **Misconception pre-empt (v2):** it's a *score*; whether it's a *calibrated*
   probability is a separate question (calibration) — L07 uses it as a ranking score.

### Section 2 — The sigmoid via odds (the heart)
5. **Odds** — p/(1-p); p=0.75 -> 3:1; odds in (0, inf).
6. **Log-odds (logit)** — log(p/(1-p)) maps (0,1) -> (-inf,+inf). Keep to the
   ADDITIVITY intuition (effects add on the log-odds scale — this is exactly what
   makes coefficients interpretable in frame 17); box the symmetry argument. (v2)
7. **Model log-odds with a line** — assume logit(p) = z = theta . x.
   **Predict-first:** if the log-odds is linear in x, what shape is p(x)?
8. **Invert -> the sigmoid** — p = 1/(1+e^-z). The S-curve; maps the real line to
   (0,1). LR's hypothesis. 1-feature cheese S-curve, **labelled illustrative/balanced**.

### Section 3 — Logistic regression as a model
9. **LR = linear model + sigmoid** — h(x) = sigma(theta . x); a LINEAR classifier
   (boundary theta . x = 0). 2D cheese: boundary + probability gradient (illustrative).
   **Misconception pre-empt (v2):** "why is it called *regression*?" It models a
   continuous log-odds, then thresholds — the *fit* is regression-like, the *task* is
   classification.
10. Intercept shifts, weights control slope/direction; boundary at sigma=0.5 <=>
    theta . x = 0. **"0.5 is the DEFAULT threshold, not a law — L07 shows why you'd
    move it" (v2 forward-ref; removes the L06<->L07 contradiction).**
11. **Worked numbers (NEW v2)** — e.g. theta=(-4, 0.5), aging temp x=10 -> z=1,
    p=sigma(1)~0.73, odds~2.7, and exp(0.5)~1.65 = "one unit of temp multiplies the
    odds by 1.65." Anchors the sigmoid arithmetic + boundary AND pre-loads frame 17.

### Section 4 — The loss: how we fit it
12. **Predict-first:** why not squared loss on sigma? Reveal: non-convex + penalizes
    confident-wrong too weakly.
13. **Log-loss INTUITION-first (v2 split)** — plot `-log p` (for y=1) and `-log(1-p)`
    (for y=0); confident-wrong is punished toward infinity. Present
    `L = -[y log p + (1-y) log(1-p)]` as "one formula that picks the right branch via
    y." Same loss reused in NNs/boosting (callback L11). No likelihood needed here.
14. **(OPTIONAL box) Where log-loss comes from** — Bernoulli likelihood -> log ->
    negate, with a one-line plain gloss ("likelihood = the probability the model
    assigns to the labels we actually saw"). Skippable; satisfies the math-depth policy.
15. **Fitting LR** — convex (no closed form) -> **gradient descent** (callback ch1);
    convex => GD finds the global optimum. Separable data -> unbounded -> regularize
    (callback L03). (v2: Newton-Raphson CUT from the body — distractor for a
    GD-trained audience; at most a parenthetical "solvers use faster variants".)

### Section 5 — Practice & bridge
16. **sklearn `LogisticRegression`** — `predict_proba`, `C` (inverse regularization),
    `penalty`, `solver`, `class_weight="balanced"` (cheese ~3% positive). Scale
    features (callback L01c). `[fragile]` code on cheese.
17. **Interpreting coefficients** — coef = effect on the LOG-ODDS; exp(coef) = odds
    ratio (callback to the worked-numbers frame 11). Logreg's interpretability superpower.
18. **Strengths/weaknesses + bridge** — interpretable, probabilistic, fast, strong
    linear baseline / only linear boundaries (needs feature engineering), assumes
    log-odds linearity. Bridge: now HOW do we evaluate it? -> L07 metrics (same
    cheese). Multiclass -> L08.

### Wrap-up
19. **Recap** + paramgreen workflow box.
20. **HW (v2: MODELING/INTERPRETATION focus, de-overlapped from L07)** — cheese: fit
    logistic regression, interpret coefficients as odds ratios, plot the sigmoid + 2D
    boundary, compute ONE prediction by hand (z -> p -> odds). Evaluation
    (thresholds/metrics) is explicitly DEFERRED to L07 (whose HW already fits a
    logreg + tries thresholds). Bonus: `C` sweep; polynomial features for a non-linear
    boundary; `class_weight` on the imbalanced full data.

## Open decisions (pending)
1. Single feature for the S-curve (frame 8) + worked numbers (frame 11): aging
   temperature, pH, or moisture? (Align with whatever feature set L07 implies.)

## Changelog (v1 -> v2, pedagogical review)
1. Frame 12 (log-loss) split: intuition-first body frame (13) + OPTIONAL boxed
   likelihood derivation (14). The audience has not formally seen MLE.
2. Added a worked-numbers frame (11): z -> sigma(z) -> odds -> odds-ratio. The deck
   was all derivation, no "turn the crank" (L07 has worked numbers, L06 didn't).
3. Cut Newton-Raphson from the fitting frame (15) — distractor for a GD audience.
4. S-curve (8/9) labelled illustrative/balanced; "0.5 is the default, not a law"
   forward-ref on frame 10 — removes the only real L06<->L07 contradiction.
5. Two misconception pre-empts: "why it's called regression" (9) and "score vs
   calibrated probability" (4) — also tightens the L07 score/ranking vocabulary.
6. Frame 3 trimmed (example zoo merged into the hook); frame 6 kept to additivity
   intuition.
7. HW (20) refocused on modeling/interpretation; evaluation deferred to L07 to avoid
   duplicating L07's own logreg-fitting HW.
