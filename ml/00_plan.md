# Course plan

Last updated: 2026-07-29

## Already delivered ✅

- **[01] ML intro + linear regression**
- **[02] Design matrix, normal equation, polynomial regression**
- **[03] Data preprocessing — missing values, categorical encoding, scaling**
- **[04] Linear regression from scratch — practical** (HW1)
- **[05] Linear regression — predicting house rent** (HW2)
- **[06] Model evaluation — overfitting and cross-validation**
- **[07] Regularization** — Ridge, Lasso, early stopping
- **[08] Hyperparameter tuning** — grid / random / Bayesian
- **[09] Regression metrics** — MSE, R², diagnostic plots
- **[10] Practical — find the errors**
- **[11] Logistic regression** — binary + multiclass, log-loss, odds ratios
- **[12] Classification metrics** — precision/recall/F1, ROC-AUC, PR-AUC, lift
- **[13] Threshold tuning** — cost-optimal cutoff, Youden's J, `TunedThresholdClassifierCV`
- **[14] Calibration** — reliability diagrams, Brier/ECE, Platt + isotonic
- **[15] Imbalanced learning** — class weights, resampling, SMOTE
- **[16] Classification practical — bank marketing**
- **[17] Decision trees** — Gini/entropy, CART, pruning
- **[18] Random forests** — bagging, OOB, the ρσ² floor
- **[19] Boosting** — AdaBoost, gradient boosting in function space
- **[20] Advanced boosting** — XGBoost / LightGBM / CatBoost, stacking
- **[21] Trees practical** — wine + census income
- **[22] Interpreting linear models & trees** — glass boxes, split credit, RuleFit
- **[23] PFI & feature effects** — PFI/CFI/LOCO/SAGE, fANOVA, ICE→PDP→M-plot→ALE, Friedman's H

## Built but not yet delivered

- **[24] SHAP & LIME** — Shapley values, the four SHAP plots, LIME step by step, counterfactuals
- **[25] Interpretability practical** — explaining startup success (leakage audit + glass boxes + PFI/PDP)

## Next chapter

**Feature engineering / selection** (`ml/06_feature_engineering`) — both decks exist but date from
2026-07-07 and predate the current style conventions, so they need a polish pass before recording.
After that: **classic methods** (`ml/07_classic_methods` — KNN, Naive Bayes, LDA/QDA, SVM, GP),
which is deliberately placed right before the neural-network block so the margin/kernel material
sets it up. Then **time series** (`ml/08_time_series`), below.

## Time series — built, not delivered (`ml/08_time_series`)

Two decks, reviewed and reworked 2026-07-31. They were originally numbered 07/08, which collided
with the delivered [07] Regularization and [08] Hyperparameter tuning; renumbered to **30/31**,
leaving **28–29 free for classic methods** (chapter 07 comes first in course order).

- **[30] Time series — classical methods** — trend/season/noise, STL, stationarity + differencing
  (ADF *and* KPSS), ACF/PACF read on our own series, AR/MA/ARMA/ARIMA/SARIMA, Holt-Winters,
  naive baselines and MASE worked by hand.
- **[31] Time series — the ML approach** — forecasting as supervised learning, time-aware splits
  and leakage, lag/rolling/calendar/exogenous features, `TimeSeriesSplit` (incl. `gap`),
  the tree extrapolation trap, recursive vs direct, prediction intervals via conformal,
  M4/M5 and why the winners were hybrids, deep + foundation models.

Open items before recording: no real dataset appears in either deck (everything is one synthetic
monthly series); the AMD/Dram exchange rate is name-dropped in the cold open but never shown.

## Next lecture — Tuesday 2026-06-23

Videos [07], [08], [09]:

- **[07] Regularization** — Ridge, Lasso, Elastic Net
- **[08] Hyperparameter tuning** — Grid / Random / Optuna
- **[09] Regression losses** — MSE, MAE, Huber, quantile

## Practical — Thursday 2026-06-25

Video [10]. Hands-on covering L01d (overfitting + CV) + the three Tuesday lectures.

**Suggested dataset:** `insurance.csv` (Kaggle "Medical Cost Personal Datasets", ~1300 rows, target = `charges`).

Why:

- Mixed types — `sex`, `smoker`, `region` force one-hot / ordinal decisions
- Small enough for fast CV + HP search iteration
- `smoker × age` interaction rewards polynomial features + regularization
- Skewed, asymmetric residuals (smokers) motivate MAE / Huber over MSE
- Single file, no API, no auth — students download once and go

Alt if we want continuity: reuse `data/House_Rent_Dataset.csv` (already used in HW2 — students know it).

## Classification — Saturday/Sunday 2026-06-27/28

Videos [11], [12], [13], [14]:

- **[11] Logistic regression** — incl. multiclass (softmax / one-vs-rest) and log-loss (built: `11_classification_logreg`)
- **[12] Classification metrics** — accuracy → precision/recall/F1 → ROC AUC → PR AUC → lift (built: `12_classification_metrics`)
- **[13] Threshold tuning** — cost-sensitive cutoff (`c* = C_FP/(C_FP+C_FN)`), Youden's J, recall floor, `TunedThresholdClassifierCV` (built: `13_threshold_tuning`; not yet lectured, split out of [12])
- **[14] Calibration** — reliability diagrams, Brier score, ECE, Platt / isotonic, `CalibratedClassifierCV` (building: `14_calibration`)

> Note (2026-06-21): the planned [12] "classification losses" and [13] "multiclass" were folded into [11]; metrics + threshold tuning became [12]; calibration is its own [13]. Threshold tuning moved out of [14].

## Imbalanced learning — next lecture (Tue 2026-06-30, tentative)

Video [15]:

- **[15] Imbalanced learning** — class weights, resampling (over/under-sampling, SMOTE), the precision/recall tradeoff when the positive class is rare

## Next lectures (sequence, dates TBD)

- **[15] Feature engineering**
- **[16] Decision trees**
- **[17] Random forests + bagging**
- **[18] Boosting** — AdaBoost, GBM
- **[19] XGBoost / LightGBM / CatBoost** — include monotonic + interaction constraints (~3-4 frames; required for credit / insurance / healthcare)
- **[20] Stacking / blending**
- **[21] Feature selection** — filter / wrapper / embedded, tree-based importances, SHAP
- **[22] Other classic models** — KNN, SVM, Gaussian processes, Naive Bayes, LDA, QDA
- **[23] Neural networks** — likely multi-lecture (perceptron → MLP → backprop → frameworks)
- **[24] Experiment tracking** — MLflow / W&B
- **[25] TabPFN + tabular foundation models** — awareness lecture; in-context prior-fitted transformer that beats default XGBoost on small data (≤10k rows). Short, students will see this in interviews.
  
## Other topics — want to cover, timing TBD

- Data leakage
- Model interpretability
- Error analysis
- Data drift
- Model deployment
- Cost-sensitive learning
