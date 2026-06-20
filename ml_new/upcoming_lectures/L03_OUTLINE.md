# L03 Regularization — implementation outline

Working plan to address the **critical gaps** from the pedagogical review (cold-open, surviving L01d2 concepts, sklearn API), while deferring double descent to the neural networks chapter.

Current deck: `L03_regularization.tex`, 14 frames, 18 PDF pages.
Target deck after edits: ~24 frames, ~29 PDF pages. Still under 45 min.

Scripts live in: `py_files/`
Generated plots land in: `fig/`
Existing legacy shared figures stay in: `figures/` (untouched)

---

## Final frame order

### Section: Cold open (NEW — replaces "Recap: overfitting")

**Reuse the DGP from `ml_old/.../02_Regression_Main_Concepts.ipynb` cell 218** for continuity:
- True function: `y = 1 + 2x + 3x² + 4x³ + N(0, 20)`
- `n=100`, `x ∈ [-3, 3]`, train/test split `0.3`
- Model class: `PolynomialFeatures(degree=20)` + Ridge
- SEED = 509

1. **Frame 1 — The degree-20 poly with no penalty.**
   λ=0. Show train scatter (blue) + test scatter (red) + true cubic (black dashed) + fitted degree-20 polynomial (thick). Annotate train MSE, test MSE.
   Figure: `fig/l03_open_1_lambda0.pdf`

2. **Frame 2 — Dial λ to the sweet spot.**
   λ=1 (picked because it gives the lowest test MSE in the sweep). Curve hugs the true cubic. Train MSE rises slightly. Test MSE drops dramatically.
   Figure: `fig/l03_open_2_lambda1.pdf`

3. **Frame 3 — Crank λ way up.**
   λ=10000. Curve is flat-ish (over-regularized, near-zero coefs). Both train and test MSE explode.
   Figure: `fig/l03_open_3_lambda10000.pdf`

   **Punchline frame text:** *"Regularization is a continuous knob. Too little → memorize noise. Too much → underfit. There's a sweet spot. How do we find it?"*

   *(The "how do we find it" sets up cross-validation as the answer — already covered in L01d, so we can callback rather than re-teach.)*

### Section: Why regularize?

4. **Frame 4 — What is regularization?** *(unchanged from current deck)*
   Bullet list explicit / implicit / structured. Keep.

5. **Frame 5 — Regularized empirical risk minimization** *(simplify notation slightly)*
   Current frame is fine but reduces some dense notation. Keep the boxed formula `R(θ) = R_emp + λ·J(θ)`.
   **Closing bridge line** *(per Fix 5, sets up next section):* *"That penalty term raises bias and lowers variance. To see why that's a trade worth making, we need a vocabulary for the two errors at play."*

6. **Frame 6 — NEW. The noise floor (irreducible error / Bayes risk).** *(moved earlier per Fix 2)*
   - Restate: `MSE = Bias² + Var + σ²`
   - **σ² is the noise floor**. Even an oracle model couldn't beat it.
   - Visual: same cold-open data, with TRUE cubic curve drawn through it. Shade ±σ band. Annotation: *"this is the best ANY model can do — `Bayes risk`."*
   - Punchline: *"regularization closes the gap above the floor, not below it."*
   Figure: `fig/l03_bayes_risk.pdf`

### Section: Bias-Variance Tradeoff *(relabel using canonical names)*

7. **Frame 7 — Bias and variance, intuitively** *(unchanged + 2 new bullets)*
   The text frame on bias = mean offset, variance = wiggle-across-datasets. Keep.
   **Add 2 closing bullets** *(per Fix 3, fold approximation/estimation naming here so we can lighten Frame 8):*
   - *"These two also have model-theoretic names: **approximation error** = best model in our class still misses truth; **estimation error** = finite data can't pin down even the best-in-class."*
   - *"Regularization trades approximation error UP for estimation error DOWN. Often a win."*

8. **Frame 8 — RENAMED. Visualizing the model spaces** *(lead with dartboard callback per Fix 3)*
   **Open with explicit L01d callback:** *"Remember the dartboard from last week? H₀ is everywhere the dart could land. H is your aiming pattern (your model class). H_R is the tightened, regularized pattern."* THEN show the Venn diagram with the relabeled distances:
   - `d(f*_0, f*)` = **approximation error**
   - `d(f*, f^_R)` = **estimation error**
   Figure: `figures/bv_anim_5.pdf` (legacy, reuse)

9. **Frame 9 — Estimation variance shrinks under regularization** *(unchanged)*
   Keep as-is. Figures: `figures/bv_anim_3.pdf`, `bv_anim_2.pdf` (legacy).

### Section: Ridge regression (L₂)

10. **Frame 10 — NEW. The U-curve: test MSE vs λ.** *(moved here per Fix 2, callback promoted per Fix 4)*
    **Open with explicit L01d callback as the lead:** *"Remember the validation curve from L01d? Same diagnostic — but for λ instead of degree d."* THEN show the plot.
    Single line plot from the same DGP: x-axis log(λ) over `[0.001, ..., 10000]`, two lines: train MSE (rising) and test MSE (U-shaped, with the minimum near λ=1). Vertical line marker at the optimum.
    - *"The U-shape is bias-variance made visible: low λ = high variance side; high λ = high bias side."*
    Figure: `fig/l03_mse_vs_lambda.pdf`

11. **Frame 11 — Ridge regression** *(unchanged math + NEW mechanism bullets per Fix 1)*
    Closed-form derivation, boxed `(XᵀX + λI)⁻¹Xᵀy`. Keep.
    **Add 3 new bullets BEFORE the closed-form, explaining the mechanism:**
    - *"Without a penalty, fitting a high-capacity model on finite data lets coefficients grow huge — they're chasing noise, not signal."*
    - *"The L₂ penalty `λ‖θ‖²` resists being dragged. Big coefs pay a quadratic cost; small ones pay almost nothing."*
    - *"Net effect: less variance (we don't chase noise), a sliver of bias (we shrink the signal a tiny bit too). Usually a win."*

12. **Frame 12 — Ridge as constrained optimization** *(unchanged)*
    L₂ ball geometry. Figure: `figures/ridge_perspectives_03.png` (legacy).

13. **Frame 13 — Ridge solution path** *(unchanged)*
    Toy 2D DGP, contour + path. Figure: `figures/lin_model_regu_02.png` (legacy).

14. **Frame 14 — Why ridge helps: polynomial example** *(unchanged)*
    Figure: `figures/poly_ridge_02.png` (legacy).

15. **Frame 15 — NEW. Scale features before regularizing (predict-first / promoted from review).**
    The most common bug students hit when they first run Ridge. Frame format: `Questions:` / `Answers:` (no `\pause`, per current style).
    **Questions:**
    - You fit `Ridge(alpha=1)` on house data. `bathrooms ∈ [1, 4]`, `square_meters ∈ [50, 500]`. Both genuinely predict price. Which feature does Ridge penalize harder?
    - Same setup but you `StandardScaler()` first. What changes?
    **Answers:**
    - *Unscaled:* `bathrooms` gets penalized harder. Its coefficient HAS to be ~100× larger to have comparable effect on `y` — and Ridge's L₂ penalty punishes large coefficients quadratically. The genuinely-predictive feature gets shrunk toward zero on a technicality.
    - *Scaled:* both features have unit variance. The penalty is fair — Ridge picks based on signal strength, not on units.
    - **Rule: ALWAYS scale features before Ridge/Lasso. Use `StandardScaler()` or `MinMaxScaler()`. Fit on TRAIN ONLY, then `.transform()` the test set.**
    Pure text frame, no figure.

16. **Frame 16 — NEW. Using Ridge in sklearn.**
    Argument table (NOT full code) matching L01d's `cross_val_score` style:
    ```
    Ridge(alpha=, fit_intercept=, max_iter=, tol=, solver=, random_state=)
    ```
    Table mapping each argument to meaning, default, and "when to change it." One row noting `fit_intercept=True` does NOT regularize the intercept — common gotcha. Pure text frame, no figure.
    **Closing bridge line** *(per Fix 5, sets up Lasso):* *"Ridge shrinks every coefficient but zeros none. Sometimes you want a model that throws features away entirely. That's the next penalty."*

### Section: Lasso regression (L₁)

17. **Frame 17 — Lasso regression** *(unchanged)*
    The L₁ formulation, no closed form, sparsity teaser. Keep.

18. **Frame 18 — Lasso solution path** *(unchanged)*
    Figure: `figures/lin_model_regu_01.png` (legacy).

19. **Frame 19 — Lasso vs ridge: geometric intuition** *(unchanged)*
    The diamond-vs-ball frame. Figure: `figures/lasso_contour_cases.png` (legacy).

20. **Frame 20 — Soft thresholding** *(unchanged)*
    Figure: `figures/soft_thresholding.png` (legacy).

21. **Frame 21 — NEW. Using Lasso in sklearn.**
    Mirror of frame 16. Argument table for:
    ```
    Lasso(alpha=, fit_intercept=, max_iter=, tol=, selection=, warm_start=)
    ```
    Note `selection='random'` for speed on wide data. Pure text.

### Section: Comparing L₁ vs L₂

22. **Frame 22 — Solution paths side-by-side** *(unchanged)*
    Figure: `figures/solution_paths_01.png` (legacy).

23. **Frame 23 — When to use which + Elastic Net** *(unchanged)*
    Decision rules + Elastic Net mention. Keep.

24. **Frame 24 — Wrap-up** *(EDIT)*
    Remove the outdated "Next: L04" pointer (that content is now in L01d which is delivered). Replace forward pointer with something accurate based on what comes next — TBD when we decide. For now: *"Coming next: classification (logistic regression) — same machinery, different output."*

---

## Frames being CUT (replaced by cold-open)

- **Current frame 1 ("Recap: overfitting")** — replaced by the 3 cold-open frames.

## Frames being DEFERRED (per user)

- **Double descent** — would have been a closing "modern wrinkle" frame after wrap-up. Pushed to neural networks chapter per `DEFERRED_TODO.md`. Frames stay drafted in `L01d2_bias_variance.tex`.

---

## Python scripts to create

All scripts:
- Use `numpy`, `scikit-learn`, `matplotlib`
- Default SEED = 509
- Save PDFs to `../fig/` (relative to `py_files/`)
- Use Armenian flag color palette where 3+ colors are needed: red `#D90012`, blue `#0033A0`, orange `#F2A800`
- Logging to `logs/<script_name>.log` per CLAUDE.md

**Source of truth for the DGP** — adapted directly from `ml_old/Chapter 1 Regression Main Concepts/Code/02_Regression_Main_Concepts.ipynb` cell 218:

```python
np.random.seed(509)
N = 100
X = np.linspace(-3, 3, N).reshape(-1, 1)
y = 1 + 2*X.ravel() + 3*X.ravel()**2 + 4*X.ravel()**3 + np.random.normal(0, 20, size=N)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
poly = PolynomialFeatures(degree=20)
# Ridge over alpha_values = [0.001, 0.01, 0.1, 1, 10, 100, 10000]
```

### Script 1: `py_files/cold_open_ridge_demo.py`

Generates **4 figures**, all sharing the DGP above:

- For each `λ ∈ {0, 1, 10000}` (= overfit, sweet spot, underfit):
  - Fit `Ridge(alpha=λ)` on the degree-20 polynomial features
  - Plot: scatter train (blue) + scatter test (red) + true cubic (black dashed) + fitted curve (thick orange `#F2A800`)
  - Textbox top-right: `train MSE = ..., test MSE = ...`
  - Same x-axis range, common y-axis range across all 3 plots so they read as a sequence
- Additionally, sweep the full `alpha_values` list and plot train+test MSE vs log(λ):
  - Two lines (train rising, test U-shape). Armenian blue / red.
  - Vertical dashed line at the minimum of the test curve
  - Title: "Train and test MSE vs regularization strength"
- Outputs:
  - `fig/l03_open_1_lambda0.pdf`
  - `fig/l03_open_2_lambda1.pdf`
  - `fig/l03_open_3_lambda10000.pdf`
  - `fig/l03_mse_vs_lambda.pdf`

### Script 2: `py_files/bayes_risk_illustration.py`

Generates **1 figure** for the Bayes risk frame, using the SAME DGP for continuity:

- Same cubic true function + same σ=20 noise
- Scatter `N=100` data points
- Overlay the TRUE cubic as a thick black dashed line — labeled "Bayes-optimal predictor"
- Shaded band of `±σ` around the true curve — labeled "irreducible noise (σ = 20)"
- Annotation: `"Best possible MSE ≈ σ² = 400"` callout pointing at the band
- Output: `fig/l03_bayes_risk.pdf`

### Total scripts: 2 | Total new figures: 5

No script needed for:
- Frame 15 (scaling predict-first — pure LaTeX text)
- Frame 16 / 21 (sklearn API tables — pure LaTeX text)
- Frame 8 (rename only, reuse `figures/bv_anim_5.pdf`)
- Frames using legacy figures already in `figures/` (unchanged)

---

## Implementation order

1. Write the 2 Python scripts in `py_files/`.
2. Run them, verify 4 figures land in `fig/`.
3. Edit `L03_regularization.tex`:
   - Cut current frame 1 ("Recap: overfitting")
   - Insert 3 cold-open frames at the top
   - Insert Bayes risk frame after "Regularized ERM"
   - Relabel the "model spaces" frame to use approx/estimation error names
   - Insert Ridge sklearn API frame after frame "Why ridge helps"
   - Insert Lasso sklearn API frame after frame "Soft thresholding"
   - Edit wrap-up forward pointer
4. Compile, check page count.
5. Spot-check for overflow.

---

## What this outline does NOT cover (intentionally — flagged for later)

These showed up in the review as "high-leverage adds" but the user asked for CRITICAL only:

- ~~Scaling-before-regularization predict-first frame~~ — promoted to Frame 15 after independent review (2026-06-19)
- Intercept regularization gotcha frame
- Connection to L01d's three-curve diagnostic *(partially addressed by Frame 10's callback line)*
- Armenian cultural analogy for ridge vs lasso
- Math notation lightening throughout

If any of these become must-haves after the critical pass, treat as a follow-up edit. Don't pre-add.
