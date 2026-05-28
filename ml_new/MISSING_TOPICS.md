# Missing Topics — Gap Analysis vs Current `ml_new/` Outline

Date: 2026-05-26.
Method: cross-referenced the current 16-lecture skeleton against (a) what's available in `_reference/lecture_i2ml/` and `_reference/lecture_sl/`, and (b) standard intro-ML / deep-learning curricula from Stanford CS229, Stanford CS231n, DeepLearning.AI Deep Learning Specialization, and common public syllabi.

Status legend (column "In upstream?"):

- **i2ml** — source `.tex` exists in `_reference/lecture_i2ml/slides/`
- **sl** — source `.tex` exists in `_reference/lecture_sl/slides/`
- **none** — not in either cloned repo; needs original content or a 3rd upstream

---

## Category 1: classic intro-ML topics missing from current outline

These are commonly covered in CS229-style intro ML courses. Upstream sources exist for most.

| Topic | In upstream? | Upstream files | Suggested placement |
|---|---|---|---|
| **k-Nearest Neighbors (k-NN)** | i2ml | `lecture_i2ml/slides/knn/slides-knn.tex` | New lecture between L06 (LogReg) and L07 (metrics), OR as 1 frame in L06. Classic non-parametric baseline. |
| **Naive Bayes** | i2ml | `lecture_i2ml/slides/supervised-classification/slides-classification-naivebayes.tex` | New section in L06, OR new lecture (NB + LDA/QDA together). |
| **Discriminant Analysis (LDA / QDA)** | i2ml | `lecture_i2ml/slides/supervised-classification/slides-classification-discranalysis.tex` | Same as above. Often paired with Naive Bayes as "generative classifiers". |
| **Support Vector Machines (linear)** | sl | `lecture_sl/slides/linear-svm/*.tex` (5 files: ERM view, hard margin, hard margin dual, optimization, soft margin) | New lecture in `ch2_classification/`, e.g. `L08b_linear_svm.tex`. Big omission for an intro ML course. |
| **Kernel SVM / Nonlinear SVM** | sl | `lecture_sl/slides/nonlinear-svm/*.tex` (7 files: feature gen, kernel trick, poly + RBF kernels, RKHS, model selection, univ approx) | Pair with linear SVM as 2-lecture mini-block. |
| **Nested resampling** | i2ml | `lecture_i2ml/slides/nested-resampling/*.tex` (3 files: intro, nested CV, train/valid/test) | Add to **L05 CV & Tuning** as additional sections — current L05 covers basic CV but not nested CV (critical for honest HPO). |
| **Principal Component Analysis (PCA)** | **none** | not in cloned repos | New lecture, e.g. `L13b_pca_dimreduction.tex`. Could fit in Chapter 4 (Clustering / Unsupervised) since PCA is unsupervised. |
| **t-SNE / UMAP** | none | not in cloned repos | Pair with PCA, or add as section. Standard for high-dim data visualization. |

## Category 2: deep learning topics (user explicitly asked: CNN, RNN, "other stuff")

**Important:** the cloned upstream repos (`lecture_i2ml`, `lecture_sl`) do **NOT cover modern deep learning**. They stop at multilayer FNNs. No CNN, RNN, LSTM, Transformer, Attention, BatchNorm, modern optimizers — none of it. These topics would need original content or a different upstream source (e.g., NYU DLSP21 by Yann LeCun, DeepLearning.AI specialization, Stanford CS231n, fast.ai).

| Topic | In upstream? | Notes / suggested placement |
|---|---|---|
| **Convolutional Neural Networks (CNNs)** | none | Needs new lecture. Convolution operation, pooling, classical architectures (LeNet, AlexNet, VGG, ResNet). Place after L15 (backprop/dropout), e.g. `L15b_cnns.tex` in `ch5_neural_networks/`. |
| **Recurrent Neural Networks (RNNs)** | none | New lecture. Sequential data, vanilla RNN, vanishing gradient problem. |
| **LSTM and GRU** | none | New lecture or extended RNN lecture. Gating mechanisms (input, forget, output gates), how they fix vanishing gradients. |
| **Attention Mechanism** | none | New lecture, bridges seq2seq with Transformers. Soft lookup interpretation. |
| **Transformers** | none | New lecture. "Attention is all you need" architecture, self-attention, positional encoding, multi-head attention. Current L16 (GenAI) has a placeholder section for transformers but at conceptual level only — needs a real deep dive. |
| **Modern optimizers** | none | SGD with momentum, RMSprop, Adam, AdamW. Could be a section in L15 (which currently is just "HPO for NNs"). |
| **Batch Normalization** | none | Section in L15 or new mini-lecture. Often paired with Dropout (already in L15). |
| **Skip connections / ResNets** | none | Section in CNN lecture. |
| **Initialization (Xavier / He)** | none | Section in L14 or L15. |
| **Embeddings** | none | Section in RNN or Transformer lecture (word2vec, GloVe, learned embeddings). |
| **Modern deep learning for vision** | none | Optional advanced lecture: object detection (YOLO, Faster R-CNN), segmentation (U-Net), vision transformers (ViT). |

## Category 3: generative AI topics (L16 is currently a stub)

L16 GenAI Intro currently has 4 placeholder sections (disc vs gen, LM basics, transformers conceptual, practical). Real coverage of generative AI needs more:

| Topic | In upstream? | Notes |
|---|---|---|
| **Generative Adversarial Networks (GANs)** | none | Generator-discriminator setup, mode collapse, conditional GANs. |
| **Variational Autoencoders (VAEs)** | none | Latent variable models, reparameterization trick, ELBO. |
| **Diffusion Models** | none | Stable Diffusion / DDPM forward + reverse process. Practically dominant for image generation. |
| **Large Language Models** | none | Pretraining, scaling laws, fine-tuning, RLHF, alignment. |
| **Prompting / Retrieval-Augmented Generation (RAG)** | none | Practical use. Already alluded to in L16 placeholder. |

## Category 4: theory / foundational topics in upstream that aren't in our outline

These are in `lecture_sl/` and could be pulled in if depth-of-theory is desired:

| Topic | In upstream? | Upstream files | Notes |
|---|---|---|---|
| **Advanced Risk Minimization** | sl | `lecture_sl/slides/advriskmin/*.tex` (18 files: bias-variance decomp, Bernoulli/Brier/L1 losses, MLE, proper scoring rules, pseudo-residuals) | Mostly deep theory. Could enrich L03 (regularization) bias-variance section and L11 (boosting) pseudo-residual discussion. Probably not a standalone lecture for intro course. |
| **Curse of Dimensionality** | sl | `lecture_sl/slides/cod/slides-cod.tex`, `slides-cod-examples.tex` | 1-2 frames, fits in L01 or L04. |
| **Feature Selection** | sl | `lecture_sl/slides/feature-selection/*.tex` (5 files: filters, wrappers, intro, motivating examples) | New lecture in `ch1_regression/`, or fold into L05. Lasso/L1 already touches it in L03. |
| **Information Theory** | sl | `lecture_sl/slides/information-theory/*.tex` (13 files: entropy, KL, MI, cross-entropy, source coding) | Probably overkill for an intro course. But cross-entropy/KL would enrich L06 (logistic loss) and L15 (multiclass NN). |
| **Gaussian Processes** | sl | `lecture_sl/slides/gaussian-processes/*.tex` (6 files) | Advanced. Optional standalone or skip. |

## Category 5: practical / applied topics often in intro ML

| Topic | In upstream? | Notes |
|---|---|---|
| **Class imbalance / SMOTE / class weights** | none | Often a section in classification metrics lecture. Could go in L07. |
| **Calibration (Platt scaling, isotonic regression)** | none | Section in L07 or original. Important for probabilistic predictions. |
| **Time series forecasting** | none | New mini-block (ARIMA, exponential smoothing, Prophet, modern RNN/Transformer for TS). Out of scope for short course. |
| **Anomaly detection** | none | Brief section in clustering lecture (isolation forest, one-class SVM, autoencoders). |
| **Recommender systems** | none | Standalone topic. Collaborative filtering, matrix factorization. Often a separate course. |
| **Reinforcement learning** | none | Standalone topic. Usually a separate course. Worth a 1-frame "what this is" mention in L01. |
| **A/B testing / hypothesis testing for ML** | none | Could fit in `math/` lectures (stat). Already partially covered there. |
| **Data Leakage** | none | Listed in current `Chapter 1 Regression/Todo.md` as done. Worth a frame in L05 (CV) explaining what counts as leakage. |
| **Transfer learning / fine-tuning** | none | Section in CNN or Transformer lecture. |
| **Self-supervised learning** | none | Brief mention paired with embeddings or generative AI. |
| **MLOps / pipelines / deployment** | none | Usually a separate course. Hyperparameter pipelines fits in L05. |

## Recommended additions, ranked by importance for an intro ML course

If you only added a handful, do them in this order:

### Tier 1 — strongly recommended (intro ML staples)

1. **SVM (linear + kernel)** — L08b + L08c, source from `lecture_sl/linear-svm` + `nonlinear-svm`. Big classic gap.
2. **PCA / dimensionality reduction** — fold into Chapter 4 clustering as L13b, original content.
3. **k-NN** — short lecture, port from `lecture_i2ml/slides/knn`. Or as section in L06.
4. **Naive Bayes + LDA/QDA** — short joint lecture, port from `lecture_i2ml/slides/supervised-classification`. Or sections in L06.
5. **Nested resampling** — extend L05 from `lecture_i2ml/slides/nested-resampling`.

### Tier 2 — important if course covers deep learning seriously

6. **CNNs** — new `L15b_cnns.tex`. Original content, ref CS231n.
7. **RNN / LSTM** — new `L15c_rnn_lstm.tex`. Original content, ref CS230 cheatsheet.
8. **Modern optimizers (Adam etc.) + BatchNorm** — extend L15 or new section.
9. **Attention + Transformers (deep dive)** — new `L15d_transformers.tex`. Bridges to L16 (GenAI). Original content, ref "Attention Is All You Need" + Karpathy's nanoGPT lectures.

### Tier 3 — fills out GenAI lecture

10. **VAEs / GANs / Diffusion** — split L16 into 2-3 lectures (or keep as L16 + L16b for image gen).
11. **LLMs deep dive (pretraining, RLHF)** — section in expanded L16.

### Tier 4 — optional / advanced

12. Feature selection, calibration, class imbalance, curse of dim — sections, not standalone lectures.
13. Time series, anomaly detection, RL, recommenders — separate mini-courses.

## How to incorporate

For Tier 1 (upstream source exists):

- Create new skeleton `.tex` in the appropriate `chN_*/` chapter folder with the same template as L01-L16.
- Update `ml_new/README.md` status table.
- Update `docs/superpowers/specs/2026-05-26-ml-consolidation-design.md` provenance map.

For Tier 2-3 (no upstream):

- Decide whether to clone a 3rd upstream (e.g., Yann LeCun's NYU DLSP21 repo, or write notes from CS231n / CS230 / Karpathy's nanoGPT series).
- Mark skeleton's provenance comment as ORIGINAL or list the external references.

---

## Update (2026-05-26): partial execution of addendum

After this gap analysis, two more shallow clones were added and 3 new skeletons + 1 renumber landed:

- Cloned `_reference/lecture_i2dl/` (intro DL) and `_reference/lecture_dl4nlp/` (DL for NLP). `lecture_iml` deliberately NOT cloned per instructor.
- Added `ml_new/ch5_neural_networks/L16_cnns.tex`, `L17_rnns_lstm.tex`, `L18_optimizers_init.tex` — all sourced from `lecture_i2dl/slides/{cnn1,cnn2,rnn,opt1,opt2,regu}`.
- Renumbered the original GenAI lecture: `ml_new/ch6_genai/L16_genai_intro.tex` → `L19_genai_intro.tex`.
- Fixed `L15_backprop_dropout_hpo.tex` provenance: backprop and dropout DO have upstream sources in `lecture_i2dl/slides/{opt1,regu}`, contrary to the first-draft claim that they were original-only.
- All 5 affected files compile cleanly.

**Tier 1 partial fill (2026-05-26, third pass):**

- `ml_new/ch1_regression/L05b_nested_resampling.tex` — sourced from `lecture_i2ml/nested-resampling`.
- `ml_new/ch4_clustering/L13b_pca_dim_reduction.tex` — mostly original; partial refs in `lecture_i2dl/ae`.

**Still pending** (from the addendum below):

- Interpretable ML (IML) chapter — explicitly skipped per instructor.
- Remaining Tier 1 gaps: SVM, k-NN, Naive Bayes, LDA/QDA — not added per instructor scope.
- Tier 4 advanced (survival, online, multi-target, BO) — not added.
- Upstream non-slide assets (exercises, quizzes, cheatsheets) — not addressed.
- `misc/dl4nlp/` decks — left in place; not moved into `ml_new/`.
- Process gaps (frozen SHAs, build automation, Quarto integration, migration cutover) — not addressed.

---

## Addendum (2026-05-26, second pass): things missed in the first pass

### A. Inaccuracies in Category 2 (deep learning topics)

I claimed "neither upstream covers DL" — that's **only true for the two repos I cloned**. The slds-lmu organization has several more:

| Topic claimed "no upstream" | Actual upstream | Notes |
|---|---|---|
| CNN | `slds-lmu/lecture_i2dl` | "Introduction to Deep Learning" — not yet cloned |
| RNN / LSTM | `slds-lmu/lecture_i2dl` | Same |
| BatchNorm, modern optimizers | `slds-lmu/lecture_i2dl` | Same |
| Attention / Transformers | `slds-lmu/lecture_dl4nlp` + **`misc/dl4nlp/` in THIS repo** | The instructor has already authored ~10 transformer/NLP decks in `misc/dl4nlp/` (transformers, tokenization, decoding, prompting, hallucinations, MoE, etc.). Way more advanced than what I'd skeleton in L16. |
| Embeddings / word2vec | `slds-lmu/lecture_dl4nlp` + `misc/dl4nlp/` | Same |

**Action:** clone `lecture_i2dl` and `lecture_dl4nlp` into `_reference/`. Decide whether `misc/dl4nlp/` decks move into the new `ml_new/ch6_genai/` (or a new `ch7_dl_nlp/`) chapter.

### B. Brand new category: Interpretable Machine Learning (IML)

Completely missing from the first pass. The instructor already has an `ml/interpretabilty_project/` folder (notebook + data + profiling report) — clear signal IML is part of the course.

Upstream: `slds-lmu/lecture_iml` (not yet cloned). Confirmed by `lecture_i2ml/latex-math/ml-interpretable.tex` macro file.

Topics: feature importance (covered shallowly in L10 forests), partial dependence plots (PDP), individual conditional expectation (ICE), accumulated local effects (ALE), SHAP, LIME, surrogate models, counterfactual explanations, fairness/bias.

**Suggested placement:** new chapter `ch7_interpretability/` or wedge between L12 (boosting) and L13 (clustering). 2-3 lectures' worth.

### C. Other slds-lmu repositories worth knowing about

| Repo | What it covers | Relevance |
|---|---|---|
| `slds-lmu/lecture_i2dl` | Deep Learning intro: CNN, RNN, LSTM, optimizers | Fills DL gap |
| `slds-lmu/lecture_iml` | Interpretable ML: SHAP, LIME, PDP, ICE, ALE | Fills IML gap (instructor has IML project) |
| `slds-lmu/lecture_dl4nlp` | DL for NLP: transformers, attention, embeddings | Aligns with existing `misc/dl4nlp/` |
| `slds-lmu/lecture_advml` | Advanced ML: imbalanced learning, cost-sensitive | Fills Category 5 (class imbalance) |
| `slds-lmu/lecture_appml` | Applied ML | Practical pipelines, deployment |

### D. Topics hinted at by `lecture_i2ml/latex-math/ml-*.tex` macro files

These macros are defined in i2ml's shared math library, suggesting the upstream ecosystem covers them somewhere (likely in `lecture_advml` or `lecture_iml`):

- `ml-mbo.tex` → **Bayesian / Model-Based Optimization** (often a deep-tuning topic, paired with HPO)
- `ml-survival.tex` → **Survival analysis** (Cox regression, Kaplan-Meier; niche but used in medical ML)
- `ml-multitarget.tex` → **Multi-target / multi-output regression** (one model predicts multiple targets)
- `ml-online.tex` → **Online learning / streaming** (incremental learning)
- `ml-interpretable.tex` → **Interpretability** (confirms category B)

Add to Tier 4 (optional / advanced).

### E. Non-slide assets from upstream `lecture_i2ml` we ignored

I focused only on `slides/`. There are 3 more goldmines:

| Asset | Path | What it is | Use |
|---|---|---|---|
| **Exercise sheets** | `_reference/lecture_i2ml/exercises/` | Full Quarto-based exercise sheets per chapter (ml-basics, supervised-regression, supervised-classification, evaluation, tuning, trees, forests, NN, kNN, nested-resampling). Each has its own `.qmd`, build Makefile, `_quarto.yml`, and produces `*_ex.pdf` + `*_all.pdf` (with solutions). | Current `ml/Chapter 1/Homeworks/` is much smaller. Could mine exercises directly into new homework set. |
| **Quizzes** | `_reference/lecture_i2ml/quizzes/` | Kahoot-style CSVs per chapter (Loss Functions, Linear Reg, Poly Reg, k-NN, Linear Class, LogReg, KNN, Intro to Eval, Regression Metrics, etc.) | Could be imported into a quiz/poll tool. |
| **Cheatsheets** | `_reference/lecture_i2ml/cheatsheets/` | 9 printable cheatsheets in `.tex` + `.pdf` (notation, regression, classification, evaluation, eval metrics, CART, random forest, NN, tuning) | Distribute as student references. Style is LMU mlr theme — would need restyling to match project. |

**Action:** spec out a Phase 1.5 to inventory + selectively port these into `ml_new/exercises/`, `ml_new/quizzes/`, `ml_new/cheatsheets/`.

### F. Materials already in THIS repo that the consolidation plan didn't address

| Path | What it is | Status in current plan |
|---|---|---|
| `misc/dl4nlp/*.tex` | ~10 source-controlled DL/NLP slide decks (transformers, tokenization, decoding strategies, eval, early models, pretraining/finetuning, prompting, hallucinations, MoE, ...) | Untouched. These should bridge into `ml_new/` — either move into `ch6_genai/` or create `ch7_dl_nlp/`. |
| `misc/pre_math/` | Pre-course math material | Out of scope but worth confirming. |
| `misc/data_arms/` | Unknown — needs survey | Out of scope but worth confirming. |
| `misc/pearson_correlation_and_regression_slope.tex` | Standalone slide source | Could fold into L01/L02 of `ml_new/ch1_regression/`. |
| `ml/Chapter 6 GenAI/gen_ai.ipynb` | Only existing artifact for L16 | Confirmed in spec but no migration path defined. |
| `ml/Datasets/bzbz.ipynb` | Single notebook, unclear purpose | Out of scope. |
| `ml/interpretabilty_project/` | Notebook + Kaggle wellbeing data + profiling report | **NOT in current outline.** Would slot into the new IML chapter (Category B). |

### G. Structural / process items missing from the spec

| Gap | Impact |
|---|---|
| **No frozen upstream commit SHAs** | Spec mentions as "open question" but never executed. If upstream changes, provenance comments could become wrong. Should record SHAs in `_reference/UPSTREAM_SHAS.md` (gitignored or tracked?). |
| **No build/render automation for `ml_new/`** | Currently each deck is compiled by hand. `math/Lectures/` has the same gap. Could add a top-level `Makefile` target like `make ml_new` or a CI workflow. |
| **No Quarto integration plan** | The site at `hayktarkhanyan.github.io/python_math_ml_course/` is built from `.qmd` files. Current `ml/Chapter 1/01_intro__lin_reg.qmd` embeds PDFs. When `ml_new/L01.pdf` is content-complete, the qmd needs to swap `data="PDF/L01 Linear Regression.pdf"` → `data="../ml_new/ch1_regression/L01_linear_regression.pdf"`. No checklist for this swap. |
| **No migration / cutover plan** | When does `ml_new/L01.pdf` replace `ml/Chapter 1 ...PDF/L01 Linear Regression.pdf`? Per-deck as content lands? Big-bang at end of Phase 2? Spec doesn't say. |
| **`ml_new/` is not under any test / lint** | `make` and CI don't catch a broken `.tex` until you compile manually. Could add `pdflatex --halt-on-error` to CI for the chapter dirs. |
| **`README.md` in `ml/` claims content is "mostly based on LMU Munich i2ml"** | Now partly inaccurate — L13 clustering, L16 GenAI, future CNN/RNN won't be from i2ml. README should be updated when Phase 2 lands. |

## Sources consulted

- [Stanford CS229: Machine Learning](https://cs229.stanford.edu/)
- [DeepLearning.AI Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/)
- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/)
- [CS 230 Recurrent Neural Networks Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/)
- [NYU Deep Learning Spring 2021 (Yann LeCun)](https://atcold.github.io/NYU-DLSP21/)
- [Coursera Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks)
- [Aman's AI Journal: Deep Learning Architectures Comparative Analysis](https://aman.ai/primers/ai/dl-comp/)
- Upstream repos inventory: `_reference/lecture_i2ml/slides/`, `_reference/lecture_sl/slides/`, plus `exercises/`, `quizzes/`, `cheatsheets/`, `latex-math/` (browsed 2026-05-26).
- [slds-lmu GitHub organization (113 repos)](https://github.com/orgs/slds-lmu/repositories) — additional sources: `lecture_i2dl`, `lecture_iml`, `lecture_dl4nlp`, `lecture_advml`, `lecture_appml`.
- This repo's existing source: `misc/dl4nlp/*.tex` (10 transformer/NLP decks), `ml/interpretabilty_project/` (notebook + dataset), `ml/Chapter 6 GenAI/gen_ai.ipynb`.
