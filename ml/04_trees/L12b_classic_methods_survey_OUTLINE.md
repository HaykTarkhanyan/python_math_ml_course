# L12b Classic Methods Survey — KNN · Naive Bayes · LDA/QDA · SVM · Gaussian Processes (DRAFT v3)

A breadth-not-depth **survey deck**: classic supervised methods the course will NOT
cover in depth, but that students should recognize. House style of
`ml/02_main_concepts/06_overfitting_cross_validation.tex`.

> STATUS: DRAFT v3. v2 covered KNN/NB/SVM; v3 adds **LDA/QDA** and **Gaussian
> Processes** (each a proper 3-frame mini-section, not a one-liner) and restructures
> the deck into THREE FAMILIES so the additions integrate. Renamed from
> `L12b_svm_nb_knn_OUTLINE.md`.

## The organizing idea (v3): three families, not five disconnected methods
1. **Learn by similarity** -> KNN
2. **Model how each class is generated** (generative classifiers) -> Naive Bayes -> LDA/QDA
3. **Margins & kernels** -> SVM -> Gaussian Processes

Two payoff adjacencies: **LDA/QDA = "Naive Bayes that allows feature correlations"**;
**GP = "the kernel trick again, but it hands you uncertainty."**

## Locked decisions
- **Scope:** KNN, Naive Bayes, LDA/QDA, SVM, Gaussian Processes — survey/intuition
  only; trades depth for breadth. Each method = its one beautiful idea + decision
  boundary + where it still helps + the transferable insight.
- **Order:** KNN -> NB -> LDA/QDA -> SVM -> GP (the three-family arc above).
- **Pacing:** KNN 4 / NB 4 / LDA-QDA 3 / SVM 5 / GP 3 frames. ~26 frames total.
- **Style:** house style — hand-drawn TikZ/pgfplots, Armenian-flag palette,
  `\fcolorbox` callout boxes, `\pause` predict-first frames (question -> pause ->
  reveal), hook before `\tableofcontents`, Recap + light HW, provenance block.
- **Pedagogy:** intuition-first, no derivations; formulas (Bayes rule, margin, GP
  kernel) boxed/optional. State up front: "breadth deck — we trade depth for ideas."
- **Unifying device:** the SAME 2D toy carved by each method, used as the hook teaser
  AND the synthesis. (GP uses a 1D regression toy for its uncertainty-band plot.)
- **Placement / numbering (OPEN):** filed as `L12b` (after L12, before clustering L13);
  could be a mini-chapter or end of ch2.
- **File:** outline `L12b_classic_methods_survey_OUTLINE.md`; deck would be
  `L12b_classic_methods_survey.tex` (location pending).

## Sources (web research; no existing ml_old deck for these)
- KNN: lazy learner (no training-time optimization, O(n.d) PER-QUERY cost; kd/ball-
  trees, ANN/FAISS at scale); k = bias-variance knob; scaling; curse of dimensionality.
- Naive Bayes: Bayes' rule + conditional independence; works despite the false
  assumption (only the argmax must be right); variants Gaussian/Multinomial/Bernoulli;
  Laplace smoothing; spam/text. (sklearn NB; Cornell CS4780 L5.)
- **LDA/QDA (v3):** all three (LDA, QDA, Gaussian NB) come from the SAME Bayes formula;
  they differ ONLY in the covariance assumption — QDA = per-class full covariance
  (quadratic boundary), LDA = shared covariance (linear boundary), Gaussian NB =
  diagonal covariance (the independence special case). LDA also = SUPERVISED
  dimensionality reduction (maximize class separation; <= classes-1 components) vs PCA
  (unsupervised, maximize variance). (sklearn LDA/QDA docs; Raschka PCA-vs-LDA.)
- SVM: max-margin "widest street"; support vectors; soft margin C; needs scaling;
  scales poorly to large n; kernel trick (inner products via a kernel WITHOUT building
  the space; RBF = similarity / bump per support vector). (IBM SVM; Medium guides.)
- **Gaussian Processes (v3):** a DISTRIBUTION OVER FUNCTIONS; returns mean + calibrated
  uncertainty everywhere (uncertainty grows away from data); kernel = covariance =
  similarity (length-scale = wiggliness); the surrogate model behind Bayesian
  optimization; O(n^3), no big data. (d2l.ai GP intro; practical GP guides.)
- VERIFY at build (Context7/sklearn): `KNeighborsClassifier`; `GaussianNB`/
  `MultinomialNB`/`BernoulliNB`; `LinearDiscriminantAnalysis`/
  `QuadraticDiscriminantAnalysis`; `SVC`; `GaussianProcessClassifier`/`Regressor`.
  Confirm the exact GNB = diagonal-covariance QDA statement.

## Callbacks / cross-references
- **L01c** (`03_data_preprocessing.tex`): KNN + SVM need scaling; KNN imputation.
- **L01d** (`06_overfitting_cross_validation.tex`): KNN's k, SVM's C, LDA-vs-QDA are
  bias-variance knobs; choose by CV.
- **L06** (logistic regression): generative (NB, LDA/QDA) vs discriminative (logreg);
  "score vs calibrated probability" (NB/SVM uncalibrated; GP gives real uncertainty).
- **L13b** (PCA / dim reduction): LDA as the SUPERVISED counterpart to PCA.
- **L05** (CV & tuning): GPs power Bayesian optimization for hyperparameter search.
- **Math probability module**: Bayes' theorem.
- **Trees / boosting (L09-L12)**: what wins tabular today, to place these in context.

## Frame-by-frame outline (~26 frames; pacing 4/4/3/5/3)

### Hook / framing
1. **Classic ideas worth knowing** — we won't go deep; each has one beautiful,
   transferable idea. Teaser: the same data carved several ways.
2. **A map: three families** — learn by similarity / model how classes are generated /
   margins & kernels. (Orients the whole deck; breadth not depth.)

### FAMILY 1 — Learn by similarity
#### KNN [4 frames]
3. **Lazy learning** — no training-time optimization; predict = vote of the k nearest
   neighbors. "The cost is moved, not removed": zero training, EXPENSIVE prediction.
4. **Boundary + k = the bias-variance knob** — local jagged (Voronoi-like) boundary;
   k=1 overfit, large k underfit; pick by CV (callback L01d). **Predict-first:** k=1 on
   noisy data?
5. **Three gotchas** — scaling (callback L01c), curse of dimensionality, inference cost
   (O(n.d)/query -> kd/ball-trees, ANN/FAISS).
6. **Where it still helps + lead transferable** — nearest neighbors in an EMBEDDING
   space = retrieval (vector search, RAG, recommenders); KNN imputation; anomaly
   detection; quick baseline.

### FAMILY 2 — Model how each class is generated (generative classifiers)
#### Naive Bayes [4 frames]
7. **Spam hook first** — is this message spam, from its words? (Armenian spam SMS.)
   We want P(spam | words) -> Bayes' rule: posterior ~ likelihood x prior; argmax.
8. **The "naive" assumption + variants** — features conditionally independent given the
   class -> multiply per-feature probs. Obviously false ("free"+"money" co-occur).
   Boxed: GaussianNB (continuous), Multinomial/BernoulliNB (counts/text); Laplace smoothing.
9. **Why it works anyway** — **Predict-first:** the assumption is false, should it fail?
   Reveal: only the argmax must be right; dodges curse of dimensionality; little data; fast.
10. **Generative vs discriminative** — NB models how data is generated; logreg models
    the boundary (callback L06). NB is poorly calibrated (a score, not a probability).

#### LDA / QDA [3 frames — NEW v3]
11. **The generative idea, made concrete** — model each class as a Gaussian BLOB in
    feature space; Bayes' rule turns the blobs into a boundary. This is Naive Bayes'
    richer cousin: it ALLOWS feature correlations (full covariance) instead of assuming
    independence. (Same Bayes formula as NB; only the covariance assumption changes.)
12. **LDA vs QDA (one knob: the covariance)** — shared covariance across classes ->
    LINEAR boundary (fewer params, more bias); per-class covariance -> QUADRATIC curved
    boundary (more params, more variance). Visual: class Gaussian contours + the two
    boundary shapes. Boxed unification: **Gaussian NB = diagonal-covariance special
    case; LDA/QDA = the full-covariance versions.** Another bias-variance instance.
13. **The transferable + where it helps** — LDA doubles as **supervised dimensionality
    reduction**: project onto the axes that best SEPARATE classes (<= classes-1 dims),
    vs PCA which is unsupervised and maximizes variance (callback L13b). Strong low-data
    baseline; assumes ~Gaussian features. Predict-first option: "PCA vs LDA — which axis
    would each pick on a 2-class blob?"

### FAMILY 3 — Margins & kernels
#### SVM [5 frames]
14. **Maximum margin** — pick the WIDEST-margin separating line (most robust).
    **Support vectors** = the few points that touch the margin; the rest don't matter.
15. **Soft margin (C) + practical limits** — allow violations (C = regularization knob,
    bias-variance); SVM needs feature scaling (inner-product based, like KNN) and scales
    poorly to large n (~O(n^2)-O(n^3)).
16. **The problem: non-linear data** — concentric circles: no line separates.
    **Predict-first:** can a straight line ever separate these circles? What if we add
    the right third feature?
17. **The kernel trick (the beautiful idea)** — lift to higher-dim where it IS
    separable, but compute inner products via a **kernel WITHOUT building the space**
    (never "project back"). **RBF = a similarity function / a bump around each support
    vector** (ties to KNN); "implicitly infinite-dim" as the wow-aside.
18. **Where it still helps** — high-dim / text / small-n; the kernel trick is the
    transferable idea. Honest: trees/boosting usually win tabular now (callback L09-L12).

#### Gaussian Processes [3 frames — NEW v3]
19. **The idea: a distribution OVER functions** — don't fit one function; keep all
    functions consistent with the data and average them. A GP returns a mean prediction
    AND calibrated uncertainty everywhere. THE plot: mean line + a confidence band that
    PINCHES at data points and BALLOONS away from them. **Predict-first:** where is the
    model most uncertain? (Far from data.) Use the synthetic Yerevan rent-vs-area 1D toy.
20. **How it works (intuition)** — the **kernel = similarity = which points influence
    each other** ("the kernel idea from SVM again, now it gives you uncertainty");
    Bayesian prior over smooth functions + data -> posterior; the kernel's length-scale
    sets how wiggly the functions can be.
21. **Where it helps + limits** — principled uncertainty (ties to the calibration thread;
    cf. conformal prediction); GPs are the engine of **Bayesian optimization** for
    hyperparameter tuning (callback L05); great for small data / expensive experiments /
    scientific ML. Limit: O(n^3) — like kernel SVM, no big data.

### Synthesis
22. **Same data, several ways** — KNN (jagged/local), NB & LDA/QDA (probabilistic
    blobs), SVM (max-margin / kernel curve), GP (mean + uncertainty band). The payoff.
23. **When would you reach for each** — table across all five; calibration column
    (NB/SVM uncalibrated; GP native uncertainty); what beats them today (trees/boosting
    tabular, NNs perception/text).
24. **The transferable ideas (the real point)** — similarity -> embeddings/retrieval
    (KNN); generative modeling / "model the classes" (NB, LDA/QDA); the **kernel idea**
    (the through-line of SVM AND GP); Bayesian uncertainty (GP). These recur across ML.

### Wrap-up
25. **Recap** — one line per method, grouped by the three families + paramgreen
    "ideas to keep" box.
26. **HW (light / exploratory)** — CORE: run `KNeighborsClassifier`, `GaussianNB`,
    `LinearDiscriminantAnalysis` (and `QuadraticDiscriminantAnalysis`), and `SVC` on
    cheese/Titanic; compare to your trees/logreg; plot the decision boundaries (KNN
    k=1 vs k=50 for bias-variance). 2-3 sentences on which idea surprised you most.
    Bonus: `GaussianProcessRegressor` on the 1D rent-vs-area toy — plot the uncertainty
    band; `MultinomialNB` on a text/spam mini-set; kernel `SVC` on moons/circles;
    `LinearDiscriminantAnalysis` as a 2D projection vs PCA. No theory.

## Datasets
- 2D toy(s) for boundary visuals (blobs; concentric-circles/moons for the kernel trick).
- **1D regression toy (reuse synthetic Yerevan rent-vs-area from L09/L11)** for the GP
  uncertainty-band plot — continuity.
- Reuse cheese / Titanic for the "run them in practice" HW.
- An (Armenian) spam-SMS snippet for the NB hook.

## Open decisions (pending)
1. Placement/numbering: `L12b` after trees, a mini-chapter, or end of ch2?
2. ~26 frames is now a substantial survey — OK, or split into two (classifiers vs
   kernel-methods/GP)?
3. GP depth: pure conceptual (current) or include the 1D `GaussianProcessRegressor`
   demo in the body rather than as an HW bonus?

## Changelog (v2 -> v3)
1. Added **LDA/QDA** as a 3-frame mini-section (frames 11-13): generative Gaussian
   idea, LDA-vs-QDA via the covariance knob, GNB-as-special-case, LDA-as-supervised-
   dim-reduction vs PCA. Web-verified.
2. Added **Gaussian Processes** as a 3-frame mini-section (frames 19-21): distribution
   over functions, mean+uncertainty, kernel-as-similarity (callback SVM), Bayesian
   optimization (callback L05). Web-verified.
3. Restructured into THREE FAMILIES (similarity / generative / margins+kernels) so the
   additions integrate; two payoff adjacencies (NB->LDA/QDA, SVM->GP).
4. New callbacks: L13b (LDA vs PCA), L05 (GP -> Bayesian optimization).
5. Synthesis + recap + HW updated for all five methods. File renamed to
   `L12b_classic_methods_survey_OUTLINE.md`.
