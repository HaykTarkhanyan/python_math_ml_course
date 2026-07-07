# Curriculum gaps before the neural-net block

Date: 2026-07-07 (Claude). Scope: **classical / tabular ML only** - everything the course teaches
*before* neural networks (syllabus weeks ~1-20). Method: inventoried what is actually **built**
as a compiled deck in `ml/` (not just planned in `syllabus.csv`), then cross-checked against the
canonical classical-ML curricula: **ISLR 2e** (statistical-learning standard), **Hands-On ML 3e /
Geron, Part I** (applied standard), and current university intro-ML syllabi (Tufts CS135 2025,
Oxford ML, common 2025 course outlines). This is a completeness lens, distinct from the older
`MISSING_TOPICS.md` (which was framed around LMU-upstream availability in 2026-05).

## Bottom line

The classical block is unusually complete and, in places, *ahead* of the standard texts. There is
exactly **one core gap**, and the instructor already knows about it (it exists as an unbuilt
outline): the **classic non-tree supervised methods** - SVM, k-NN, Naive Bayes, LDA/QDA. Beyond
that, two genuinely-standard topics are absent and easy to overlook (splines/GAMs, and a dedicated
workflow/leakage deck). Everything else is either built or a reasonable deliberate cut.

---

## What is already built and solid (for context)

Linear regression + normal equation + polynomial; preprocessing (scale/encode/impute); overfitting
& CV; regularization (ridge/lasso/elastic net); hyperparameter tuning; regression metrics; logistic
regression (binary + softmax/multiclass); classification metrics; calibration; imbalanced learning;
decision trees; random forests/bagging; boosting; XGBoost/LightGBM/CatBoost + stacking; clustering
(k-means, GMM/EM, DBSCAN, HDBSCAN, hierarchical, k-medoids); PCA + t-SNE/UMAP; a full 3-deck
interpretability chapter (glass-box, PFI/PDP/ICE, SHAP/LIME/counterfactuals); feature engineering;
feature selection. Built but parked in `deferred/`: GLMs, regression inference, causal inference.

---

## The gaps, tiered

### Tier 1 - core, universally taught, currently UNBUILT (the classic-methods survey)

These are the one real hole. In ISLR they are Chapters 4 (LDA/QDA/NB/kNN) and 9 (SVM); in Geron
they are Chapters 4-5 (neighbor methods, SVM); Tufts makes "kernel methods" and "neighbor methods"
their own units. The instructor already outlined all of this as **`04_trees/L12b_classic_methods_survey_OUTLINE.md`** (three families: similarity -> kNN; generative -> NB, LDA/QDA; margins/kernels -> SVM, GP) - but no deck was ever built.

1. **SVM - linear + the kernel trick.** The single most important missing topic. Every standard
   curriculum gives it a *full chapter*, not a survey slot. Margin/hinge-loss/support-vectors, the
   dual, soft margin (C), and the kernel trick (RBF/poly) are foundational - the kernel trick alone
   reappears in Gaussian processes, kernel ridge, and is a canonical interview topic. **Recommend
   promoting SVM out of the survey into its own real deck (or a 2-part linear+kernel block).**
2. **k-NN.** The simplest learner and the baseline that "surprisingly competes." Non-parametric,
   ties directly to the distance/curse-of-dimensionality material already in the clustering and PCA
   decks. One short deck (or a shared deck with NB/LDA).
3. **Naive Bayes.** The canonical generative classifier and text-classification baseline; ISLR 2e
   specifically expanded its NB treatment. Pairs naturally with LDA/QDA.
4. **LDA / QDA.** Generative classifiers, decision-boundary geometry, the Gaussian-class-conditional
   assumption; the natural "Naive Bayes that allows feature correlations" (the survey outline's own
   framing). ISLR Chapter 4.

Note: **Gaussian Processes** is also in that survey outline - keep it as the advanced tail (it is
*not* in ISLR/Geron; it is a nice-to-have, not a core gap).

### Tier 2 - standard, absent, easy to miss

5. **Moving beyond linearity: splines & GAMs** (ISLR Chapter 7). The repo teaches **polynomial**
   regression and stops - splines, natural/smoothing splines, and **Generalized Additive Models**
   are absent (only a one-word passing mention of "basis splines" exists). This is the honest
   "flexible but still interpretable nonlinear" family, and it connects directly to the strong
   interpretability chapter: a GAM is the glass-box nonlinear model, and **EBM** ("glass-box
   boosting") is literally boosted GAMs. Easy to think "polynomial covers nonlinearity" - it does
   not cover the additive-smooth-function family. Medium priority; 1 deck or a section.
6. **A dedicated ML-workflow / pipeline / data-leakage deck** (Geron Chapter 2, "End-to-End ML
   Project"). Leakage, `sklearn.Pipeline`/`ColumnTransformer` discipline, train/val/test hygiene,
   and "select among candidate models" are currently **scattered** across the preprocessing, CV,
   and tuning decks rather than taught as one workflow. For an applied, project-driven course
   (this one has project milestones every week) the end-to-end workflow + "how leakage sneaks in
   through preprocessing" deserves its own frame-set. High applied value, low build cost.

### Tier 3 - optional / mostly covered / reasonable to defer

7. **Anomaly / outlier detection as its own topic** - partly covered (DBSCAN noise, GMM likelihood
   as a clustering "bonus"), but **Isolation Forest** and **One-Class SVM** are the standard
   dedicated methods and are absent. One-Class SVM would naturally live in the SVM deck; Isolation
   Forest in the trees chapter. Cheap add, high recognition value.
8. **Learning curves as an explicit diagnostic** (train/val score vs training-set size). The
   overfitting deck has the depth/complexity U-curve but not the sample-size learning curve, which
   is the standard "do I need more data or a better model?" tool. One frame.
9. **Recommender systems / matrix factorization.** Tufts and many syllabi give this a unit; the
   syllabus here marks it out of scope. Reasonable to defer (often its own course) - flag it as a
   deliberate cut, not an oversight.
10. **Gaussian Processes** - see Tier 1 note; advanced tail of the survey, optional.

---

## Where the repo is AHEAD of the standard curricula (do not over-invest elsewhere)

- **Interpretability**: a full 3-deck chapter (SHAP/LIME/PDP/ICE/counterfactuals). ISLR and Geron
  barely touch this; most courses do not teach it at all.
- **Calibration, imbalanced learning, modern boosting libraries** (XGBoost/LightGBM/CatBoost with
  real internals - GOSS/EFB/ordered boosting), **stacking**: deeper than ISLR/Geron.
- **Clustering breadth** (HDBSCAN, k-medoids, GMM/EM with the full E/M math): beyond most intros.

The gaps are narrow precisely because the built material is strong. The priority is to close the
Tier 1 classic-methods hole (SVM first) rather than add more breadth elsewhere.

---

## Recommended priority order

| # | Add | Why | Effort |
|---|-----|-----|--------|
| 1 | **SVM deck** (linear + kernel trick), promoted out of the survey | Universal core topic given a full chapter everywhere; kernel trick is foundational | 1 deck |
| 2 | **k-NN + Naive Bayes + LDA/QDA** (the rest of the survey, as outlined) | ISLR Ch.4 staples; kNN is the baseline that competes | 1-2 decks |
| 3 | **Workflow / pipeline / data-leakage deck** | Applied course needs the end-to-end + leakage discipline in one place | 1 deck |
| 4 | **Splines & GAMs** (moving beyond linearity) | ISLR Ch.7; the interpretable-nonlinear family; ties to EBM in interpretability | 1 deck / section |
| 5 | **Anomaly detection** (Isolation Forest + One-Class SVM) + **learning curves** frame | Cheap, high-recognition adds folded into existing chapters | sections |

## Sources

- [An Introduction to Statistical Learning, 2nd ed. (statlearning.com)](https://www.statlearning.com/) - chapters: LinReg, Classification (logistic/LDA/QDA/NB/kNN), Resampling, Linear Model Selection & Regularization, Moving Beyond Linearity (splines/GAMs), Trees, SVM, Unsupervised (PCA/clustering).
- [Hands-On Machine Learning, 3rd ed., ageron/handson-ml3 (GitHub)](https://github.com/ageron/handson-ml3) - Part I: landscape, end-to-end project, classification, training linear models, **SVM**, decision trees, ensembles/RF, dimensionality reduction, unsupervised.
- [Tufts CS135 Intro to ML, 2025](https://www.cs.tufts.edu/cs/135/2025s/index.html) - units: regression (linear+neighbor), classification (linear+neighbor), NN, trees & ensembles, **kernel methods**, recommenders + dim reduction.
- [Oxford Machine Learning 2024-2025](https://www.cs.ox.ac.uk/teaching/courses/2024-2025/ml/); general 2025 syllabi (Scaler/AppliedAI) confirming SVM as a pre-deep-learning classical staple.
