# ml_new/ — Source-Controlled Slide Decks

Parallel rewrite of `ml/` slides as project-styled Beamer `.tex` files.

**Phase 1 (this folder, today):** scaffolding only — every lecture has a skeleton with titlepage, outline frame, and `% TODO`-marked section frames pointing at the upstream files to mine for content.

**Phase 2 (later):** port content from `_reference/lecture_i2ml/` and `_reference/lecture_sl/` (gitignored shallow clones) into these skeletons one deck at a time.

Old `ml/` stays untouched until each `ml_new/` deck is ready to replace it.

## How to compile

Every `.tex` file uses `\input{../preamble}` and must be compiled from inside its chapter folder:

```bash
cd ml_new/ch1_regression
pdflatex -interaction=nonstopmode L01_linear_regression.tex
```

Or compile all of one chapter:

```bash
for f in ml_new/ch1_regression/L*.tex; do
  (cd "$(dirname "$f")" && pdflatex -interaction=nonstopmode "$(basename "$f")")
done
```

## Status table

| #   | Chapter            | Title                              | Status   | Primary upstream                                | Topics                                                                       |
| --- | ------------------ | ---------------------------------- | -------- | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| L01 | ch1_regression     | Linear Regression                  | skeleton | lecture_i2ml/ml-basics + supervised-regression  | What is ML, Tasks, Data, Models, Linear regression                           |
| L02 | ch1_regression     | Regression Losses                  | skeleton | lecture_i2ml/ml-basics + supervised-regression  | ERM, Gradient descent, L1 vs L2, Polynomial regression                       |
| L03 | ch1_regression     | Regularization                     | skeleton | lecture_sl/regularization                       | Bias-variance, Ridge, Lasso, Geometric intuition                             |
| L04 | ch1_regression     | Regression Evaluation              | skeleton | lecture_i2ml/evaluation                         | Train/test, Generalization, Overfitting, MSE/MAE/R^2                         |
| L05 | ch1_regression     | CV and Tuning                      | skeleton | lecture_i2ml/evaluation + tuning                | k-fold CV, Bootstrap, HPO intro, Grid/Random search                          |
| L05b| ch1_regression     | Nested Resampling                  | skeleton | lecture_i2ml/nested-resampling                  | Overtuning, Train/Valid/Test, Nested CV. Companion to L05.                   |
| L06 | ch2_classification | Classification + LogReg            | skeleton | lecture_i2ml/supervised-classification          | Classification tasks, Linear classifiers, LogReg, Bernoulli loss             |
| L07 | ch2_classification | Classification Metrics             | skeleton | lecture_i2ml/evaluation                         | Confusion matrix, F1, ROC, PR, AUC, Threshold tuning                         |
| L08 | ch2_classification | Multiclass                         | skeleton | lecture_sl/multiclass                           | OvR/OvO, Codebooks, Softmax regression (optional; foldable into L06)         |
| L09 | ch3_trees          | Decision Trees                     | skeleton | lecture_i2ml/cart                               | Tree growing, Split criteria, Stopping/Pruning, Predictions                  |
| L10 | ch3_trees          | Random Forests                     | skeleton | lecture_i2ml/forests                            | Bagging, OOB, Feature importance, Proximities                                |
| L11 | ch3_trees          | Boosting                           | skeleton | lecture_sl/boosting                             | AdaBoost, Gradient boosting, GBM with trees, Regularization                  |
| L12 | ch3_trees          | XGBoost / LightGBM                 | skeleton | lecture_sl/boosting                             | XGBoost, XGBoost deep dive, LightGBM, CatBoost                               |
| L13 | ch4_clustering     | Clustering                         | skeleton | **original** (no upstream)                      | k-means, DBSCAN, Hierarchical, Cluster evaluation                            |
| L13b| ch4_clustering     | Dimensionality Reduction           | skeleton | partial: lecture_i2dl/ae + original             | PCA, t-SNE, UMAP. Bulk original; PCA/manifold framing from AE chapter.       |
| L14 | ch5_neural_networks| Neural Networks                    | skeleton | lecture_i2ml/neural-networks                    | Single neuron, Hidden layer, XOR, Multilayer FNN, Matrix notation            |
| L15 | ch5_neural_networks| Backprop / Dropout / HPO           | skeleton | lecture_i2ml/NN + lecture_i2dl/opt1 + regu      | Softmax+CE, Univ approx, Backprop, Early stopping, Dropout                   |
| L16 | ch5_neural_networks| Convolutional Neural Networks      | skeleton | lecture_i2dl/cnn1 + cnn2                        | Convolution, Pooling, LeNet/AlexNet/VGG, ResNet, Dilated/Separable conv      |
| L17 | ch5_neural_networks| Recurrent Neural Networks          | skeleton | lecture_i2dl/rnn                                | Vanilla RNN, BPTT, LSTM/GRU, Attention, Applications                         |
| L18 | ch5_neural_networks| Optimization, Init, Activations    | skeleton | lecture_i2dl/opt1 + opt2                        | Adam/RMSprop, He/Xavier init, ReLU variants, BatchNorm, Hardware             |
| L19 | ch6_genai          | GenAI Intro                        | skeleton | **original** + misc/dl4nlp/ + lecture_dl4nlp    | High-level survey. Deep dives live in misc/dl4nlp/ (18 decks already done)   |
| —   | deferred           | Regression Inference               | done     | original (already authored)                     | OLS inference (existing content)                                             |
| —   | deferred           | Generalized Linear Models          | done     | original (already authored)                     | GLM framework (existing content)                                             |
| —   | deferred           | Causal Inference                   | done     | original (already authored)                     | Causal inference primer (existing content)                                   |

**Legend:**

- `skeleton` — empty deck that compiles; needs Phase 2 content port.
- `done` — content already written (the 3 deferred decks moved over from `ml/deferred_lectures/`).

## Provenance

The full mapping of which upstream `.tex` files feed each skeleton is in `docs/superpowers/specs/2026-05-26-ml-consolidation-design.md`. Each skeleton also carries its own provenance comment block at the bottom.

The upstream repos live (gitignored, all `--depth 1` shallow) at:

- `_reference/lecture_i2ml/` — https://github.com/slds-lmu/lecture_i2ml (intro, supervised, ~210 MB)
- `_reference/lecture_sl/` — https://github.com/slds-lmu/lecture_sl (advanced supervised, ~274 MB)
- `_reference/lecture_i2dl/` — https://github.com/slds-lmu/lecture_i2dl (intro to deep learning: CNN, RNN, optimizers, autoencoders, GANs)
- `_reference/lecture_dl4nlp/` — https://github.com/slds-lmu/lecture_dl4nlp (DL for NLP: transformers, BERT, GPT, RLHF)

Snapshots from 2026-05-26.

## See also: existing transformer/NLP source in this repo

`misc/dl4nlp/` already contains 18 source-controlled .tex decks the instructor authored using the same Beamer style (dove theme, popblue palette):

`01_pre_transformer`, `02_transformers`, `03_tokenization`, `04_decoding_strategies`, `05_evaluation`, `06_early_notable_models`, `07_pretraining_finetuning`, `08_prompting`, `09_hallucinations`, `10_mixture_of_experts`, `11_inference_optimization`, `12_rag`, `13_scaling_laws`, `14_agents_tool_use`, `15_reasoning_test_time`, `16_long_context_attention`, `17_emergence`, `18_reinforcement_learning`.

These are NOT moved into `ml_new/` yet — they live where they are. For Phase 2, you can either:

1. Leave them in `misc/dl4nlp/` and cross-reference from L19, or
2. Move them into `ml_new/ch6_genai/` (renumber as L20+, replace each file's inline preamble with `\input{../preamble}`, add `array` package and `matrix` tikzlibrary to `preamble.tex`).
