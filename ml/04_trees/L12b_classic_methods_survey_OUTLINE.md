# L12b Classic Methods — SVM & the kernel trick (deep) · KNN · Naive Bayes · LDA/QDA · Gaussian Processes (DRAFT v5)

**SVM-centered deck.** SVM + the kernel trick is the centerpiece and gets the real
treatment (margin derivation -> hinge/ERM view -> dual sketch -> feature-map blow-up ->
kernel trick); the other four classic methods stay intuition-first survey depth. House style of
`ml/02_main_concepts/06_overfitting_cross_validation.tex`.

> STATUS: DRAFT v5. v4 expanded SVM with a derivation + dual sketch. **v5 folds in the LMU SL
> SVM chapters** (`_reference/lecture_sl/slides/linear-svm` + `nonlinear-svm`): adopts their clean
> margin derivation and **theta/theta_0 notation** (matches our preamble), adds the **ERM/hinge-loss
> frame** (SVM = regularized ERM; hinge->log-loss = logistic regression), and the **feature-map
> blow-up motivation** (explicit maps explode -> kernels are *necessary*, not just clever). SVM now
> 9 frames. See the "LMU SL borrow list" at the bottom.

## The organizing idea: three families, SVM is the deep one
1. **Learn by similarity** -> KNN
2. **Model how each class is generated** (generative classifiers) -> Naive Bayes -> LDA/QDA
3. **Margins & kernels** -> **SVM (deep)** -> Gaussian Processes

Two payoff adjacencies: **LDA/QDA = "Naive Bayes that allows feature correlations"**;
**GP = "the kernel trick again, but it hands you calibrated uncertainty."** The kernel through-line
(SVM dual -> kernel trick -> GP) is the deck's spine and climax.

## Locked decisions (v5)
- **Notation: theta, theta_0** (not w, b) — matches LMU SL and our preamble `\thetav`/`\thx`.
  f(x) = theta^T x + theta_0; predict sign(f).
- **SVM depth = derive + hinge/ERM view + dual sketch (NOT full KKT).** Geometry -> the max-margin
  QP -> soft margin (C) -> the ERM/hinge reinterpretation -> a SKETCH of the Lagrangian dual (state
  it, don't grind KKT) so students see (a) alpha_i>0 only for support vectors and (b) data enters
  only through inner products -> feature maps blow up -> the kernel trick is the shortcut. Matches
  the course's derive-in-full precedent (XGBoost structure score, EM, sigmoid MLE) without SMO.
- **One deck, SVM-centered.** ~9 SVM frames of ~34 total; the rest is the survey tail.
- **The other four keep full v3 survey depth:** KNN 4 / NB 4 / LDA-QDA 3 / GP 3. Intuition-first.
- **Unifying device:** the SAME 2D toy carved by each method (hook teaser AND synthesis); circles
  for the kernel trick; a 1D regression toy for the GP uncertainty band.
- **Style:** house style — real matplotlib for data plots, TikZ for schematics, Armenian-flag
  palette, `\fcolorbox` callouts, `\pause` predict-first frames, section-transition `[plain]`
  slides, hook before `\tableofcontents`, Recap + light HW, provenance block.
- **Placement / numbering:** OPEN, not a priority per instructor. Suggested rename
  `L12b_svm_and_classic_methods.tex`.

## SVM correctness cribsheet (verify all at build via Context7/sklearn; notation = theta, theta_0)
- Decision fn f(x) = theta^T x + theta_0; predict sign(f). Signed distance of a point to the
  hyperplane: d(f, x_i) = y_i f(x_i) / ||theta||.
- **Margin derivation (LMU chain):** max_gamma s.t. y_i(theta^T x_i + theta_0)/||theta|| >= gamma
  -> multiply by ||theta|| -> the hyperplane rep is non-unique (scale by c), so make the **reference
  choice ||theta|| = 1/gamma** -> constraints become y_i(theta^T x_i + theta_0) >= 1 -> substitute
  gamma = 1/||theta|| -> **max 1/||theta|| = min (1/2)||theta||^2**. Result:
  **min (1/2)||theta||^2  s.t.  y_i(theta^T x_i + theta_0) >= 1** (hard-margin, convex QP). Margin
  (distance to nearest point) gamma = 1/||theta||; full street width = 2/||theta||.
- **Soft margin (two goals -> weighted sum):** slack xi_i>=0, y_i f(x_i) >= 1 - xi_i,
  **min (1/2)||theta||^2 + C sum(xi_i)**. C = trade-off / **regularization / complexity** knob;
  large C -> hard margin as a special case.
- **ERM/hinge view:** at the optimum xi_i = max(0, 1 - y_i f(x_i)), so
  **min (1/2)||theta||^2 + C sum max(0, 1 - y_i f_i)** = **L2-regularized ERM with the hinge loss**.
  Hinge = convex upper bound on 0-1 loss. **Swap hinge -> log-loss = L2-regularized logistic
  regression**; swap -> squared hinge = least-squares SVM. Sparsity (support vectors) comes from the
  hinge's flat region; smooth losses give no SVs. (Ties to L02 ERM template + L06 logreg + reg deck.)
- **Support-vector taxonomy (soft margin):** non-SV alpha_i=0 (outside margin, y f > 1, removable);
  on-margin SV 0<alpha_i<C (y f = 1); violator SV alpha_i=C (has slack, y f < 1, maybe misclassified).
- **Dual (box the result; don't derive KKT):** Lagrangian L = (1/2)||theta||^2 - sum alpha_i[y_i(theta^T x_i+theta_0)-1];
  stationarity -> **theta = sum alpha_i y_i x_i**, sum alpha_i y_i = 0. Dual:
  **max_alpha sum(alpha_i) - (1/2) sum_ij alpha_i alpha_j y_i y_j <x_i, x_j>**, s.t. sum alpha_i y_i = 0,
  **0 <= alpha_i <= C** (soft; hard = alpha_i>=0). Complementary slackness -> **alpha_i>0 <=> support
  vector**. Prediction: **f(x) = sum_over_SV alpha_i y_i <x_i, x> + theta_0** — a weighted vote of
  **inner products (similarities)**; theta_0 from any margin SV: theta_0 = y_i - <theta, x_i>.
- **Why kernels are needed (not just clever):** explicit degree-d monomial map phi: R^p -> R^K has
  K = C(d+p, d) - 1 features — explodes (even 16x16 images are infeasible). But the dual + prediction
  touch data ONLY through inner products, so:
- **Kernel trick:** replace <phi(x), phi(x')> with k(x, x') computed directly, no phi built. Kernels:
  **linear** k = x^T x'; **polynomial** k = (x^T x' + b)^d, b>=0, d in N; **RBF/Gaussian**
  k = exp(-||x-x'||^2 / 2 sigma^2) = exp(-gamma ||x-x'||^2), gamma>0 (implicitly infinite-dim).
  Kernelized dual stays convex (Gram matrix K is PSD). Prediction f(x) = sum alpha_i y_i k(x_i, x) + theta_0.
  A valid (Mercer) kernel = symmetric + PSD Gram matrix (one-liner, not a frame). **RBF = a similarity
  bump around each support vector** -> prediction = weighted vote of similarities to SVs (ties to KNN).
- **Practice:** needs **scaling** (inner-product/distance based, like KNN); ~O(n^2)-O(n^3), no big
  data; `SVC(kernel="rbf", C=..., gamma=...)`, tune C & gamma by CV; `probability=True` for
  Platt-scaled probs else uncalibrated. Multiclass = one-vs-one in sklearn's `SVC`.

## Sources
- **LMU SL (local, CC BY 4.0):** `_reference/lecture_sl/slides/linear-svm/{erm, hard-margin,
  hard-margin-dual, soft-margin, optimization}` + `nonlinear-svm/{featuregen, kernel-trick,
  kernel-poly, kernel-rbf, modelsel, rkhs-repr, uniapprox}`. The derivation and figures below come
  from here. Skip `rkhs-repr` + `uniapprox` (too deep for this course).
- ISLR ch.9 (max-margin/support-vector classifier/kernels); ESL ch.12; sklearn SVM user guide
  (C, gamma, kernels, one-vs-one). CS229 SVM notes for the dual. Verify RBF formula/gamma at build.
- KNN/NB/LDA-QDA/GP as in v3 (Cornell CS4780 L5; sklearn docs; Raschka PCA-vs-LDA; d2l.ai GP).

## Callbacks / cross-references
- **L01c** (`03_data_preprocessing.tex`): KNN + SVM need scaling; KNN imputation.
- **L01d** (`06_overfitting_cross_validation.tex`): KNN's k, SVM's C, LDA-vs-QDA are bias-variance
  knobs; choose by CV.
- **L02 ERM template + regularization deck** (`07_regularization.tex`): SVM = regularized ERM;
  C = regularization knob; **hinge loss joins the loss thread** (squared/absolute/log-loss seen before).
- **L06 logistic regression**: generative (NB, LDA/QDA) vs discriminative (logreg, SVM);
  **hinge vs log-loss -> SVM vs logreg are the same regularized-ERM recipe**; "score vs calibrated
  probability" (NB/SVM uncalibrated; GP native uncertainty).
- **L13b PCA / dim reduction**: LDA = supervised counterpart to PCA.
- **L05 CV & tuning**: GPs power Bayesian optimization.
- **Trees/boosting (L09-L12)**: what wins tabular today (honest placement).
- **Interpretability / retrieval**: KNN in embedding space = vector search / RAG.

---

## Frame-by-frame outline (~34 frames)

### Hook / framing [2]
1. **Classic ideas worth knowing** — teaser: the same 2D data carved several ways (KNN jagged,
   LDA blob-boundary, SVM max-margin/kernel curve, GP band).
2. **A map: three families** — similarity / generative / margins & kernels. Pedagogy line:
   **"SVM gets the real derivation; the rest are ideas to recognize."**

### FAMILY 1 — Learn by similarity  `[plain]` transition
#### KNN [4]
3. **Lazy learning** — no training; predict = vote of k nearest neighbors. "Cost moved, not removed."
4. **Boundary + k = the bias-variance knob** — jagged Voronoi boundary; k=1 overfit, large k underfit;
   CV (L01d). **Predict-first:** k=1 on noisy data?
5. **Three gotchas** — scaling (L01c), curse of dimensionality, inference cost (kd/ball-trees, ANN/FAISS).
6. **Where it still helps + transferable** — neighbors in an EMBEDDING space = retrieval (vector
   search, RAG, recommenders); KNN imputation; anomaly detection; quick baseline.

### FAMILY 2 — Model how each class is generated  `[plain]` transition
#### Naive Bayes [4]
7. **Spam hook** — P(spam|words) via Bayes' rule: posterior ~ likelihood x prior; argmax. (Armenian SMS.)
8. **The "naive" assumption + variants** — conditional independence -> multiply per-feature probs;
   obviously false. Boxed: Gaussian/Multinomial/Bernoulli NB; Laplace smoothing.
9. **Why it works anyway** — **Predict-first:** false assumption, should it fail? Only the argmax must
   be right; dodges curse of dim; tiny data; fast.
10. **Generative vs discriminative** — NB models how data is generated; logreg/SVM model the boundary
    (L06). NB is poorly calibrated.

#### LDA / QDA [3]
11. **The generative idea made concrete** — each class = a Gaussian blob; Bayes turns blobs into a
    boundary. NB's richer cousin: ALLOWS feature correlations (full covariance). Same Bayes formula;
    only the covariance changes.
12. **LDA vs QDA (one knob: covariance)** — shared covariance -> LINEAR boundary (more bias); per-class
    -> QUADRATIC (more variance). Visual: class contours + both boundaries. Boxed unification:
    **Gaussian NB = diagonal-covariance special case; LDA/QDA = full-covariance.** Bias-variance again.
13. **Transferable + where it helps** — LDA = **supervised dim reduction** (project to axes that best
    SEPARATE classes, <= classes-1 dims) vs PCA (unsupervised, max variance; L13b). Low-data baseline.
    Predict-first: "PCA vs LDA — which axis on a 2-class blob?"

### FAMILY 3 — Margins & kernels  (THE CENTERPIECE)  `[plain]` transition
#### SVM [9] — derive it
14. **Maximum margin: the widest street** — many lines separate; SVM picks the widest-margin one
    (largest safety margin -> most robust). **Support vectors** = the few points on the margin edge;
    the rest don't matter. **Predict-first:** which separating line generalizes best? Figure:
    candidate lines -> max-margin one + margin band + circled SVs. (LMU `linear_classif_1/2`, `svm_geometry`.)
15. **From "widest street" to a QP** — signed distance d(f,x_i) = y_i f(x_i)/||theta||; the safety
    margin gamma = min_i d. Derive (LMU chain): max gamma -> reference choice ||theta|| = 1/gamma ->
    constraints y_i(theta^T x_i + theta_0) >= 1 -> **min (1/2)||theta||^2** (convex QP, the "primal").
    THE derivation frame — geometry left, the min-problem boxed right.
16. **Support vectors: only the touching points matter** — points with y_i f = 1 (active constraints)
    at distance 1/||theta||; delete a non-SV -> boundary unchanged; delete an SV -> it moves. Hence
    *Support Vector* Machine; sparse in the data. (LMU `support_vectors`.)
17. **Soft margin (C): real data isn't separable** — hard margin has an empty feasible region on
    non-separable data. Two contradictory goals (big margin vs few violations) -> slack xi_i, minimize
    **(1/2)||theta||^2 + C sum xi_i**, s.t. y_i f >= 1 - xi_i. **C = regularization/complexity knob**
    (large C = punish violations, narrow margin; small C = wide margin, more bias); hard margin =
    C->inf special case. SV taxonomy (non-SV / on-margin / violator). (LMU `non_separable_data`,
    `margin_violations`, `soft_margin_svs`.)
18. **SVM = regularized ERM with the hinge loss (NEW, LMU erm)** — at the optimum xi_i = max(0, 1-y f),
    so the whole thing is **min (1/2)||theta||^2 + C sum max(0, 1 - y_i f_i)** = L2-regularized ERM,
    **hinge loss** (convex upper bound on 0-1). Payoff: **swap hinge -> log-loss = regularized logistic
    regression** (SVM and logreg are the SAME recipe, different loss); the hinge's flat part is what
    creates support vectors. Callbacks: L02 ERM template, L06 logreg, regularization deck. Figure:
    hinge vs log vs squared-hinge (LMU `soft_margin_losses` / `other_losses`).
19. **The dual (sketch): what the algorithm actually touches** — box the Lagrangian ->
    theta = sum alpha_i y_i x_i, sum alpha_i y_i = 0 -> the dual
    (max sum alpha_i - (1/2) sum_ij alpha_i alpha_j y_i y_j <x_i,x_j>, 0<=alpha_i<=C). Two facts to
    SEE: (1) **alpha_i>0 only for support vectors** (frame 16's sparsity, now algebraic); (2) **data
    enters only through inner products <x_i,x_j>** — and so does prediction f(x) = sum_SV alpha_i y_i
    <x_i,x> + theta_0 (a weighted vote of similarities). Land the hook: "the model touches data only
    through dot products — remember that."
20. **Non-linear data -> feature maps -> a wall (predict-first)** — concentric circles: no line
    separates. **Predict-first:** can a line EVER separate them? Lift with phi(x1,x2) =
    (x1, x2, x1^2+x2^2) -> a plane separates in 3D -> curved boundary back in 2D. BUT explicit maps
    **explode**: degree-d monomials = C(d+p,d)-1 features (even 16x16 images infeasible). "Dead end?"
    (LMU `circles_ds/feature_map/boundary`, `n_monomials`.)
21. **The kernel trick (the beautiful idea)** — the dual + prediction need only inner products, so
    replace <phi(x),phi(x')> with **k(x,x') computed directly, no phi built**. Kernels: linear /
    polynomial (x^T x'+b)^d / **RBF exp(-gamma||x-x'||^2)** (implicitly infinite-dim). Kernelized dual
    stays convex (K is PSD). Prediction f(x) = sum alpha_i y_i k(x_i,x) + theta_0. **RBF = a similarity
    bump around each SV** -> weighted vote of similarities (ties back to KNN!). THE payoff frame.
    (LMU shortcut diagram + `svm_rbf_kernel`.)
22. **SVM in practice + where it still wins** — **scale first** (L01c); ~O(n^2)-O(n^3), no big data;
    `SVC(kernel="rbf", C, gamma)`, tune by CV; probabilities Platt/uncalibrated. Strong on
    high-dim / text / small-n / clear margin. **Honest:** trees & boosting usually win tabular now
    (L09-L12) — SVM's lasting gift is the **kernel trick** (next: GP uses it for uncertainty).

#### Gaussian Processes [3] — "the kernel idea again, now with uncertainty"
23. **A distribution OVER functions** — keep all functions consistent with the data and average them;
    a GP returns a mean AND calibrated uncertainty everywhere. THE plot: mean line + band that PINCHES
    at data and BALLOONS away. **Predict-first:** where is it most uncertain? (Far from data.) 1D rent toy.
24. **How it works (intuition)** — the **kernel = similarity = which points influence each other**
    ("the kernel from SVM again — now it buys uncertainty"); prior over smooth functions + data ->
    posterior; length-scale = wiggliness.
25. **Where it helps + limits** — principled uncertainty (calibration thread; cf. conformal); the
    engine of **Bayesian optimization** for HP tuning (L05); small data / expensive experiments /
    scientific ML. Limit: O(n^3), like kernel SVM — no big data.

### Synthesis [3]
26. **Same data, several ways** — the 2D toy: KNN (jagged), NB & LDA/QDA (blobs), SVM (max-margin/
    kernel curve), GP (mean + band).
27. **When would you reach for each** — table across all five: scaling?, boundary shape, calibrated?,
    scales?, reach-for-it-when. What beats them today (trees/boosting tabular; NNs perception/text).
28. **The transferable ideas (the real point)** — similarity -> embeddings/retrieval (KNN); generative
    modeling (NB, LDA/QDA); the **kernel idea** (the through-line of SVM AND GP, now *derived*);
    Bayesian uncertainty (GP).

### Wrap-up [2]
29. **Recap** — one line per method by family + paramgreen "ideas to keep" box (max-margin; the dual
    touches only dot-products; SVM = regularized ERM/hinge = logreg's cousin; the kernel trick;
    generative-vs-discriminative; similarity).
30. **HW (light / exploratory)** — CORE: run `KNeighborsClassifier`, `GaussianNB`,
    `LinearDiscriminantAnalysis`/`QuadraticDiscriminantAnalysis`, `SVC` (linear vs rbf) on
    cheese/Titanic; plot boundaries (KNN k=1 vs k=50; linear vs RBF SVM on moons/circles); scale
    first and show it matters for KNN/SVM. 2-3 sentences on the idea that surprised you most.
    Bonus: sweep SVM `C` and `gamma`; `GaussianProcessRegressor` on the 1D rent toy (uncertainty
    band); `MultinomialNB` on a spam mini-set; LDA as a 2D projection vs PCA.

## LMU SL — what to borrow (all CC BY 4.0; attribute "Source: LMU SL, CC BY 4.0")
Derivations (adopt near-verbatim, restyled to house palette + theta notation):
- **Margin derivation** — `linear-svm/slides-linsvm-hard-margin.tex` (the max gamma -> ||theta||=1/gamma
  reference-choice -> min (1/2)||theta||^2 chain; the non-unique-representation subtlety). Frame 15.
- **Dual + support vectors** — `slides-linsvm-hard-margin-dual.tex` (Lagrangian -> stationarity ->
  dual; complementary slackness -> alpha_i>0 <=> SV; prediction = weighted sum of dot-products =
  similarity). Frames 16, 19.
- **Soft margin / C / SV taxonomy** — `slides-linsvm-soft-margin.tex` (two-goals weighted sum; box
  constraint 0<=alpha<=C; three SV types; C as regularization; hard margin as special case). Frame 17.
- **ERM / hinge loss / logreg link** — `slides-linsvm-erm.tex` (xi = max(0,1-yf) -> hinge; hinge->log
  = regularized logistic regression; hinge->squared = LS-SVM; sparsity from hinge). Frame 18. **Best
  single borrow — it wires SVM into our ERM + logreg + regularization threads.**
- **Feature maps blow up** — `nonlinear-svm/slides-nonlinsvm-featuregen.tex` (circles phi=(x1,x2,
  x1^2+x2^2); monomial count C(d+p,d)-1; 16x16 infeasible; "dead end? -> kernels"). Frame 20.
- **Kernel trick + kernels + prediction** — `slides-nonlinsvm-kernel-trick.tex` (shortcut diagram;
  kernelized dual stays convex; linear/poly/RBF definitions; f(x)=sum alpha y k(x_i,x)+theta_0; Mercer
  = symmetric+PSD one-liner). Frame 21. RBF depth: `slides-nonlinsvm-kernel-rbf.tex`.
Figures (reuse-with-attribution OR regenerate in house style — prefer regenerating the DATA plots,
reuse the geometry schematics):
- Geometry/schematic (reuse OK): `svm_geometry.png`, `support_vectors.png`, `soft_margin_svs.png`,
  the kernel-shortcut TikZ (redraw in our palette).
- Data plots (regenerate in matplotlib, house style): the circles lift + boundary (`circles_*`),
  hinge-vs-log-vs-squared loss (`soft_margin_losses`/`other_losses`), linear/poly/RBF decision
  boundaries (`svm_*_kernel`), margin-with-violations. Optional real numbers: the MNIST kernel error
  table (linear .134 / poly-d2 .119 / RBF .12) from the kernel-trick deck.
Skip (too deep for this course): `slides-nonlinsvm-rkhs-repr` (RKHS / representer theorem),
`slides-nonlinsvm-uniapprox` (universal approximation), full KKT/Lagrangian-duality theory (LMU keeps
it commented out too), `slides-linsvm-optimization` (solver internals).

## Open decisions (pending instructor)
1. Rename deck file `..._survey` -> `..._svm_and_classic_methods`?
2. Reuse LMU figures with attribution, or regenerate all data plots in house style? (Recommend:
   regenerate the boundary/loss/circles plots; reuse the geometry schematics with a source line.)
3. GP `GaussianProcessRegressor` band in the body (frame 23) or HW-only? (Currently body.)

## Changelog
- **v5 (LMU fold-in):** switched to theta/theta_0 notation; adopted LMU's margin derivation (frame
  15) and dual (frame 19); added the **ERM/hinge-loss frame** (18, SVM<->logreg via loss swap) and
  the **feature-map blow-up motivation** (20, why kernels are necessary); added the SV taxonomy to the
  soft-margin frame; added a cribsheet + LMU borrow list. SVM 8 -> 9 frames; total ~34.
- **v4:** re-weighted to SVM-centered; expanded SVM 5 -> 8 with the margin derivation + dual sketch +
  kernel trick; one deck; other four kept at full survey depth.
- **v3:** co-equal 5-method survey (no derivations), three-families structure, added LDA/QDA + GP.
