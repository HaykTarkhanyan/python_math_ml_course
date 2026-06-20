# Course plan

Last updated: 2026-06-19

## Already delivered ✅

- **[01] ML intro + linear regression**
- **[02] Design matrix, normal equation, polynomial regression**
- **[03] Data preprocessing — missing values, categorical encoding, scaling**
- **[04] Linear regression from scratch — practical** (HW1)
- **[05] Linear regression — predicting house rent** (HW2)
- **[06] Model evaluation — overfitting and cross-validation**

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

Videos [11], [12], [13]:

- **[11] Logistic regression**
- **[12] Classification losses**
- **[13] Multiclass** — softmax, one-vs-rest, one-vs-one

## Imbalanced + thresholds + calibration — next lecture (Tue 2026-06-30, tentative)

Video [14]:

- **[14] Threshold tuning + imbalanced learning + calibration** — class weights, resampling, precision/recall tradeoff, reliability diagrams, Brier, Platt / isotonic, `CalibratedClassifierCV`

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
