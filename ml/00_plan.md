# Course plan

Last updated: 2026-08-08

**Cadence: Wednesday / Friday / Sunday**, three sessions a week (changed from the earlier
Tuesday/Thursday pattern). Times on the "Metric" Google Calendar.

**Projected finish: mid-to-late November 2026** — see [Projected finish](#projected-finish).

---

## Delivered ✅ — [01] through [24], plus [26] and [27]

Last delivered: **[26] Feature engineering** and **[27] Feature selection** on 2026-08-12,
notes exported the same day. **[28] classic methods was cut for time** and needs a new slot.

| # | Lecture |
|---|---|
| [01] | ML intro + linear regression |
| [02] | Design matrix, normal equation, polynomial regression |
| [03] | Data preprocessing — missing values, categorical encoding, scaling |
| [04] | *Practical* — linear regression from scratch (HW1) |
| [05] | *Practical* — predicting house rent (HW2) |
| [06] | Model evaluation — overfitting and cross-validation |
| [07] | Regularization — Ridge, Lasso, early stopping |
| [08] | Hyperparameter tuning — grid / random / Bayesian |
| [09] | Regression metrics — MSE, R², diagnostic plots |
| [10] | *Practical* — find the errors |
| [11] | Logistic regression — binary + multiclass, log-loss, odds ratios |
| [12] | Classification metrics — precision/recall/F1, ROC-AUC, PR-AUC, lift |
| [13] | Threshold tuning — cost-optimal cutoff, Youden's J, `TunedThresholdClassifierCV` |
| [14] | Calibration — reliability diagrams, Brier/ECE, Platt + isotonic |
| [15] | Imbalanced learning — class weights, resampling, SMOTE |
| [16] | *Practical* — bank marketing |
| [17] | Decision trees — Gini/entropy, CART, pruning |
| [18] | Random forests — bagging, OOB, the ρσ² floor |
| [19] | Boosting — AdaBoost, gradient boosting in function space |
| [20] | Advanced boosting — XGBoost / LightGBM / CatBoost, stacking |
| [21] | *Practical* — wine + census income |
| [22] | Interpreting linear models & trees — glass boxes, split credit, RuleFit |
| [23] | PFI & feature effects — PFI/CFI/LOCO/SAGE, fANOVA, ICE→PDP→M-plot→ALE, Friedman's H |
| [24] | SHAP & LIME — Shapley values, the four SHAP plots, LIME, counterfactuals |
| [26] | Feature engineering — same row / many rows / outside the data, target-encoding leakage |
| [27] | Feature selection — filter / wrapper / embedded, Boruta, stability selection |

Videos for [22]–[24] are recorded but not yet published. [26] and [27] are not yet recorded.

---

## Scheduled — the classic ML track closes out

| Date | Day | Session | Material |
|---|---|---|---|
| ~~**Aug 12**~~ | Wed | ✅ Feature engineering + feature selection — **delivered**, notes exported. Classic methods was cut for time. | `26_feature_engineering` · `27_feature_selection` |
| **next slot** | — | Classic methods — **slipped from Aug 12**, needs rescheduling | `28_svm_and_classic_methods` |
| **Aug 14** | Fri | Time series lecture | `30_classical_time_series` · `31_ml_time_series` |
| **Aug 16** | Sun | *Practical* — Armenia electricity / gas usage | `32_electricity_forecast_solution.ipynb` |
| **Aug 19** | Wed | Clustering | `L13_clustering` |
| **Aug 21** | Fri | *Practical* — compress a Saryan painting | `solution_image_compression.ipynb` |
| **Aug 23** | Sun | Dimensionality reduction | `L13b_dimensionality_reduction` |

After Aug 23 the classic ML track (chapters 01–08 plus clustering and dim reduction) is fully
delivered, and everything remaining is the deep-learning half.

---

## Projected — the deep learning half

Extrapolated Wed/Fri/Sun from Aug 26 with no breaks. Dates are the **low end** of each range;
every extra session pushes everything after it by 2–3 days.

| Block | Sessions | Dates | Built material |
|---|---|---|---|
| Dim reduction practical | 1 | Aug 26 | `solution_eigenfaces.ipynb` |
| Neural networks | 2–3 | Aug 28, 30 | `L14`, `L15` |
| NN practical | 1–2 | Sep 2 | `nn_practical_solution.ipynb` |
| CNN | 3 | Sep 4, 6, 9 | `L16`–`L19` |
| RNN | 2 | Sep 11, 13 | `L20`, `L21` |
| Autoencoders | 3 | Sep 16, 18, 20 | `L22`, `L23`, `HW1_sae_rnn` |
| GANs | 2 | Sep 23, 25 | `L23b`, `L23c` |
| **Attention + LLMs** | **8** | Sep 27, 30 · Oct 2, 4, 7, 9, 11, 14 | `L24` only — **see build dependencies** |
| Diffusion | 2–3 | Oct 16, 18 | `L27`–`L31` + PANIR project |
| Reinforcement learning | 2–3 | Oct 21, 23 | `L32` + tic-tac-toe project |
| VLM | 2 | Oct 25, 28 | `L33`, `L34` |
| Audio | 2 | Oct 30, Nov 1 | `L35`, `L36` |
| VLA | 2 | Nov 4, 6 | `L38` only |
| Tabular foundation models | 2 | Nov 8, 11 | `L37` only |

### Projected finish

- **Wed 11 November 2026** — 40 sessions from Aug 12, if every "2–3" lands on 2.
- **Fri 20 November 2026** — 44 sessions, if all four ranges (NN lectures, NN practical,
  diffusion, RL) land on 3.

---

## Build dependencies — what the schedule needs that does not exist yet

Ordered by how soon the date arrives.

### 1. Attention + LLMs — ~7 sessions of material, first needed **Sep 27**

The largest block in the course and the least built. `ml/ch9_attention/` holds **L24 only**
(53pp); the planned **L25 and L26 do not exist**. `ATTENTION_CHAPTER_PLAN.md` carries the
approved outline for all three.

The rest of the block would come from **`misc/dl4nlp/`** — 18 decks, ~442 pages, already in the
house palette — which is currently outside `ml/` and unregistered in `_quarto.yml`. Folding it
into `ch9_attention` (or a new LLM chapter) is a decision, not a mechanical move: it changes
chapter boundaries and the numbering.

**Lead time: ~7 weeks from 2026-08-08.**

### 2. VLA and tabular FM — one session each with nothing to teach

Both blocks are scheduled for 2 sessions but hold **one deck each** (`L38` 36pp, `L37` 39pp) and
no practical. Either build a practical for each, or cut both to 1 session and finish ~4 days
earlier. Candidate already identified: the **nanoTabPFN "change the prior" lab** cut from ch14
(~60 s pre-training per iteration, verified against the paper).

### 3. Homework gaps

No homework exists for **ch7 (RNN), ch8b (GAN), ch12 (VLM), ch13 (audio), ch14 (tabular FM),
ch15 (VLA)**. Only ch14 and ch15 need one to fill a scheduled slot; the others are lecture-only
blocks, so their homework is optional relative to this schedule.

---

## Open questions

1. **Sunday Aug 9 is unassigned**, and **[25] the interpretability chapter project**
   (startup success — the solution notebook is written and the data is in the repo) has no slot
   anywhere in this schedule. Dropped, or does it go somewhere?
2. **RL was listed twice** when this schedule was set (2–3 sessions each time). Counted once
   here. If it is genuinely two blocks, add 2–3 sessions and the finish moves to
   late November.
3. **Numbering.** The two tracks now collide: `24_shap_lime` vs `L24_attention`,
   `30_classical_time_series` vs `L30_diffusion_conditioning`. This bites the moment the
   deep-learning decks get YouTube playlist numbers — which starts Aug 26. See the
   housekeeping entry in `DEFERRED_TODO.md` for the two honest resolutions.
4. **Duplicate feature-engineering decks.** `ml/06_feature_engineering/` holds both
   `26_`/`27_` (Jul 2026) and legacy `L01g_`/`L01h_` (Jun 2026) versions of the same two
   topics. The Aug 12 lecture should use `26_`/`27_`; the legacy pair needs archiving.

---

## Not scheduled

### `ml/deferred/` — three compiled decks, deliberately parked

GLMs (29pp), causal inference (37pp), regression inference (32pp). All three compile; all three
stay unregistered by instructor decision (2026-08-08) — the statistics-background reason still
holds. GLMs' stated revisit condition is already met. See `DEFERRED_TODO.md`.

### Topics wanted, no slot

Experiment tracking (MLflow / W&B), data drift, model deployment, error analysis.

### Deliberately out of scope

Recommender systems, MLOps, survival analysis, online learning, multi-target regression.
Each is a separate course.

**Gaussian processes** were on this list until 2026-08-12. They are *not* a chapter, but [28]
gives them a short survey section (distribution over functions, the conditioning identity,
Bayesian optimization) because the kernel through-line from SVM makes them nearly free to teach
there. Out of scope as a topic; in scope as one of five classic ideas.
