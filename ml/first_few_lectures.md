# First Few Lectures — Plan & Review

Last updated: 2026-05-29
Scope: L01–L05b (regression block, weeks 1–6 of `syllabus.csv`)

## What this file is

Planning doc for the first chunk of the ML course. Two halves:

1. Review of `ml/Chapter 1 Regression Main Concepts/PDF/L01 Linear Regression.pdf` (the deck you actually like) with concrete improvement suggestions.
2. Outline for L02–L05b, including what each lecture has to land before the next can begin.

Source of truth for the full 30-week arc lives in `ml/syllabus.csv`. This file is the zoomed-in view of the first six weeks.

---

## L01 — Intro + Linear Regression

Status: **approved as the first lecture**. Keep delivering from the existing PDF for now. Improvements below apply to the next revision (porting into `ml/ch1_regression/L01_linear_regression.tex`, which is already mostly there).

### What works (don't break these)

- **The forest / path / hungriness analogy.** This is the single most valuable thing in the deck. It pre-loads "model = curve through data", "risk = how bad", "GD = follow the slope" without notation. Keep it.
- **Tabular comparison of `f1..f5` with intercept/slope/SSE.** Concrete and tactile. Students see ERM as a search before any calculus shows up.
- **Hypothesis space → parametrization → "search the parameter surface".** The progression from `f` to `theta` is clean. The PDF makes this jump in 4 frames; that's the right pace.
- **GD as 4 progressive frames (`[0]`, `[1]`, `[2]`, `^theta`).** The iterative build is good. Don't collapse it.
- **Learning rate visualization with three side-by-side plots (good / poor / divergent).** Hits the intuition without algebra.
- **Worker productivity example to motivate "hand-crafted rules don't scale".** Direct and convincing. Keep it.

### What to fix

Numbered in priority order.

1. **i.i.d. assumption is hidden.** The PDF mentions `P_xy` and says we have `n` i.i.d. points, but never says *what i.i.d. means* or *why we assume it*. Add one frame before "Empirical Risk":
   - "We assume the data we have looks like the data we'll see later (i.i.d.)."
   - "If your training set is 2010–2020 apartment prices and you predict 2024, that assumption is broken — that's distribution shift, more on it in week 24."
   - Tie to Yerevan: pre-COVID vs post-COVID rent data.

2. **Why squared error?** Currently the answer is "easy to optimize + classical stats". Both are true, but neither is the *deep* reason. Add one frame:
   - Geometric: residual squared = squared distance, so OLS = orthogonal projection of `y` onto column space of `X`.
   - Statistical: assuming Gaussian noise, MLE = OLS. (Don't derive; just name-drop. We come back to it in L02 / week on probability foundations.)
   - Robustness caveat: squared error punishes outliers a lot; L1 doesn't. (Pre-announces L02.)

3. **No predict-first frames** (per `feedback_predict_first_slides.md`). Counter-intuitive moments worth pausing on with `\pause`:
   - **Before** showing the learning rate divergence panel: "Predict — what happens if alpha is too big?" then `\pause` reveals the diverging plot.
   - **Before** the ERM table reveal of `f5` with SSE 5.88: "Can you eyeball a better model than `f1..f4`?" then `\pause` reveals the fit.
   - **Before** the worker productivity slide: "If I gave you 3 features and asked you to write down a productivity formula, how long until you give up?" then `\pause` reveals the doomed hand-rule attempt.

4. **No Armenian local examples** (per `feedback_armenian_examples.md`). Western examples in the PDF: Google Translate, Siri, DeepMind, Go. Add alongside (not replacing):
   - **Search engines:** "yandex.am / kanaberd.am search ranking".
   - **Recommender:** "Tashir Pizza app suggesting your usual order based on history".
   - **RL:** "AI playing hide-and-seek" is already great; keep it.
   - **Regression motivator:** "Predict marshrutka journey time from origin, hour-of-day, weather" — this is also the project hook.
   - **The worker productivity example** can be replaced with rent prediction (apartment in Yerevan: features = rooms, floor, district, year built, balcony y/n; target = monthly rent in AMD). That also dovetails with `House_Rent_Dataset.csv` you use in the lab.

5. **Notation density spike at the "Loss / Risk" frames.** `L: Y x R^g -> R`, `R(f) = E_{xy}[L(y, f(x))] = ∫ L dP_xy`, and `R_emp` all show up in three consecutive slides. Two cushions:
   - Add a *notation cheat sheet* frame right before, listing what `x`, `y`, `f`, `theta`, `L`, `R`, `R_emp` mean in one column and what each one *does* in another. Reference back to it.
   - For the integral `∫ L dP_xy`: most students haven't seen measure-theoretic notation. Either drop it (just write `E[L]`) or pause once and say "this is fancy `E[L]` notation, you can read it as average over the true distribution".

6. **Hessian + positive-definite condition (slide 4/11 of GD section).** Too advanced for L01. The PDF spends a frame on stationarity + Hessian PSD. Students who haven't seen multivariable calculus (which is everyone in week 1) lose the thread. Defer the Hessian to L02 (when we cover GD properly) or to the optimization chapter. Replace with: "If the gradient is zero AND the function curves up, we're at a minimum — we'll formalize this next week".

7. **OLS derivation jumps straight to `(X^T X)^{-1} X^T y`.** The PDF gives the formula but not the *step*. Either:
   - Skip the derivation entirely in L01 (mention "there's a closed form, here it is, derivation as HW bonus" — which the homework already does).
   - OR walk through `∇ ||X theta - y||^2 = 0 → X^T X theta = X^T y → theta = (X^T X)^{-1} X^T y` in one frame.
   The current middle ground (formula shown, derivation hand-waved) leaves students unable to reproduce it themselves.

8. **The `Frankenstein` resources slide.** Move it to the end (or the qmd page), not the front. Starting with credits dilutes the hook. Open with "ML is changing our world" — that's the energy you want.

9. **"FAST Foundation Ingredients of Machine Learning 19 October 2020 7 / 38" footer.** This footer text bleeds through onto many slides from the FAST insert pages (visible in the extracted text). Either re-export the FAST pages with the footer cropped or replace those slides with native LaTeX. Not urgent but ugly.

10. **No homework preview at the end of the lecture.** Currently the qmd has the HW link separate. Add one closing frame: "HW01: implement GD from scratch on `data_lin_reg.csv` — instructions on the course site". Concrete close > vague "questions?".

### What NOT to add to L01

These belong later; don't cram them in:

- Bias-variance tradeoff (L03 / W3)
- Train/test split, evaluation metrics (L04 / W4)
- Regularization (L03 / W3)
- Logistic regression / classification (L06 / W8)
- Probabilistic interpretation / MLE / Bayes (math chapter, currently not in checklist)
- Convexity formalism (mention informally only)

### Improvement budget

If you only do three of the above before next semester:
- (1) i.i.d. frame
- (3) predict-first on learning rate
- (4) Armenian local examples (rent / Tashir / yandex.am)

These are cheap, high-leverage, and don't require restructuring the deck.

---

## L02 — Regression Losses + Gradient Descent

Status: **skeleton in `ml/`, needs porting**.

### Learning goals

- Distinguish theoretical risk from empirical risk (callback to L01 — students often conflate them).
- See GD as one of many optimization choices; pick a learning rate sensibly.
- Compare L1 vs L2 loss with a concrete outlier-heavy example.
- Introduce polynomial regression as "linear regression with engineered features" — sets up overfitting for L03.

### Frame outline

1. Recap: L01 loss / risk / ERM in three bullets.
2. Loss vs theoretical risk vs empirical risk — same diagram, three highlights.
3. Why GD, why not always closed-form. (Mention: OLS has closed form; almost nothing else does.)
4. GD algorithm formally — 1 frame, including stop condition.
5. Three learning rates panel (good / poor / divergent) — **predict-first**: ask students which one is which.
6. Stochastic GD as 1-sentence preview ("we'll come back to this in NN chapter").
7. L1 vs L2 loss intuition (penalty curves side-by-side).
8. L1 with an outlier in the data — visual: OLS line shifts dramatically, L1 line barely moves. **Predict-first**.
9. Subgradient hand-wave for L1 (don't derive — just "the slope isn't defined at 0, we pick something").
10. Polynomial regression as `x → [1, x, x^2, ...]` feature map. Show degree 1, 3, 9 on noisy sine data.
11. Overfitting visible on the degree-9 fit. **Predict-first**: which degree generalizes?
12. Closing: next week we formalize the overfitting trade-off and add a penalty term to control it.

### Prerequisites students must have from L01

- `f`, `theta`, `H`, `L`, `R_emp` notation.
- "Search the parameter surface" intuition.
- That OLS has a closed form (`theta = (X^T X)^{-1} X^T y`).

### Homework hook

Extend the L01 GD code to support intercept + multiple learning rates; derive the normal equation as the bonus (deferred from L01 to make it tractable).

### Practical / lab

- Plot the risk surface for a 2-parameter linear regression and overlay GD trajectory.
- Fit polynomial degree 1, 3, 9 on `y = sin(x) + noise`.

---

## L03 — Regularization (Ridge & Lasso)

Status: **skeleton in `ml/`, needs porting**.

### Learning goals

- See the bias–variance tradeoff concretely (not formally — that's a later math chapter).
- Recognize the regularized ERM template `R_emp + lambda * J(theta)`.
- Tell Ridge and Lasso apart by what they do to coefficients (shrinkage vs sparsity).
- Read a coefficient path plot.

### Frame outline

1. Recap: polynomial degree-9 overfitting from L02 — set up the problem.
2. Bias-variance intuition with the dartboard picture. Don't derive expected MSE decomposition (defer to math chapter).
3. The regularized ERM pattern. Sentence: "we add a price tag on complexity".
4. Ridge regression: `R_emp + lambda * ||theta||_2^2`. Closed form: `(X^T X + lambda * I)^{-1} X^T y`.
5. Geometric picture: contour of `R_emp` + ball-shaped constraint = solution shrinks toward origin.
6. Lasso regression: `R_emp + lambda * ||theta||_1`. No closed form.
7. Geometric picture: diamond-shaped constraint pushes solution to corners → sparsity. **Predict-first**: where does the solution land?
8. Coefficient path plot: as lambda grows, Ridge shrinks all coefficients smoothly; Lasso zeros them out one by one.
9. Elastic Net as a one-line aside.
10. `sklearn`'s `alpha` parameter == `lambda` here — flag this naming mismatch explicitly.
11. Closing: now we have a model with a knob (lambda) — how do we choose lambda? → L04 / L05.

### Prerequisites from L01–L02

- The ERM framework.
- Polynomial features as a way to overfit.
- That a closed form `(X^T X)^{-1} X^T y` exists for OLS.

### Homework hook

Implement Ridge from scratch via `(X^T X + lambda * I)^{-1} X^T y`. Reproduce `sklearn.linear_model.Ridge`. Bonus: coordinate descent for Lasso.

### Armenian terms (per memory)

- ճշգրտում / regularization
- շեղում / bias
- ցրվածություն / variance

---

## L04 — Regression Evaluation Metrics

Status: **skeleton in `ml/`, needs porting**.

### Why this is here, not earlier

You might be tempted to teach train/test split in L01 alongside risk. Don't. L01–L03 keep students focused on "fit a model that minimizes empirical risk". L04 is where we say "actually, empirical risk on the training set lies — here's why and what to do about it".

### Frame outline (sketch — fill in during porting)

1. Apparent error vs generalization error: why training MSE is optimistic.
2. Hold-out split — simplest fix.
3. The U-shape of test error as model complexity grows: underfitting → sweet spot → overfitting.
4. Metrics: MSE, MAE, RMSE, MAPE, R^2. When each one lies.
5. Spearman's rho for ranking tasks (1 frame).
6. Data leakage demo: scale-before-split inflates test scores. **Predict-first**: what's wrong with this pipeline?
7. Closing: hold-out is noisy on small data → L05 introduces CV.

### Difficulty drops here

Per syllabus notes, this lecture is intuitive (🧀). Students who survived L02/L03 will breathe here. Use the breather to set up L05.

---

## L05 + L05b — Cross-Validation, Tuning, Nested Resampling

Status: L05b **already ported** (in `ml/ch1_regression/L05b_nested_resampling.tex` — that's the deck I built from the i2ml provenance).

### L05 outline (planned, not yet ported)

1. Why one hold-out split is noisy.
2. k-fold CV — diagram first, **predict-first** the algorithm.
3. Stratified k-fold for classification (forward-reference).
4. Bootstrap and out-of-bag as alternative.
5. Group / leave-one-object-out CV — when records aren't independent.
6. Hyperparameters vs parameters — distinguish clearly.
7. Grid search vs random search. **Predict-first**: which is more efficient when most hyperparameters don't matter? (Bergstra–Bengio answer.)
8. Bayesian optimization + Hyperband as 1-line previews.

### L05b is already done

The "untouched test set" principle + nested CV diagram + cost formula + leakage checklist are all there. Mark it as the optional advanced track for the regression block.

### Note on compute budget

Per global CLAUDE.md, this laptop is 16 GB / Intel Iris Xe. Students may run grid search on their own machines; warn them in L05 not to launch 1000-config grid search on a small laptop. Random search with a fixed budget = good default.

---

## Open questions / decisions to make

1. **Cutover plan.** Do you teach from the existing `ml/.../PDF/*.pdf` decks this semester and from `ml/*.tex` next semester? Or run a parallel pilot? I'd say keep delivering from the PDFs you trust, port to `ml_new` between sessions, switch at a chapter boundary (next semester).

2. **Probability prerequisites.** Several improvements above (MLE rationale for L2, i.i.d. framing) point to a "probability foundations" mini-lecture that doesn't yet exist. Either (a) defer all probabilistic interpretation to a math-chapter lecture later, or (b) bolt a 20-minute "ML probability glossary" onto L01. Recommend (a) so L01 stays tight.

3. **Forest analogy length.** The current PDF spends 2 frames on it. Worth expanding to 3–4 frames with a build-up? My read: keep at 2 — it's a hook, not a lecture.

4. **Worker productivity vs apartment rent as the running example.** Pick one and use it across L01–L04. Right now L01 uses workers and the project uses apartment rent — mild context-switching cost. Recommend apartment rent (Yerevan), which also dovetails with the project dataset.

5. **HW01 difficulty.** The current HW (implement GD from scratch on `data_lin_reg.csv`) is right for week 1. The bonus (derive normal equation) is hard — many week-1 students will skip it. That's fine; revisit in L02 / L03 as a worked solution.

---

## Migration status (regression block → `ml/`)

| Lecture | PDF (delivery) | `ml_new` tex (port) | Notes |
|---|---|---|---|
| L01 | exists, polished, you like it | done, richer | Apply improvements 1–10 above to tex during next revision |
| L02 | exists | skeleton | Port next |
| L03 | exists | skeleton | Port after L02 |
| L04 | exists | skeleton | Port after L03 |
| L05 | exists | skeleton | Port after L04 |
| L05b | partial in old deck | **done** | Already ported from i2ml; ready to use |

---

## Next concrete action

Pick one of:

- **(A)** Apply L01 improvements 1, 3, 4 (i.i.d., predict-first, Armenian examples) to the existing PDF source — fastest, biggest pedagogical bang.
- **(B)** Port L02 fully into `ml_new` so we have two real lectures source-controlled.
- **(C)** Build the L01 → L05 lab notebooks (separate from slides) — the syllabus mentions practical/lab work each week but the notebooks don't exist yet.

My recommendation: **(B)** — porting L02 unblocks the chapter and forces decisions about notation consistency you'll need anyway. Then loop back to (A) before the next iteration of L01.
