# PROGRESS.md

> **Archived 2026-07-22.** Session logs now live in `_work_sessions/` (one TOML per
> session). This file is kept read-only for history - do not add new entries here.

Session log: what was done, what is in progress, what comes next. Newest entry first.
Entries are point-in-time snapshots - for anything old, verify against `git log`
before acting on it.

## 2026-07-19 - New chapter: Time Series (ml/07_time_series)

### Done

Built a two-lecture time-series chapter from scratch (instructor asked for classic + ML split, full build):

- **`ml/07_time_series/`** new chapter folder: `py_src/`, `fig/`, `data/`.
- **Lecture 07 (classical, 22 frames)** `07_classical_time_series.tex/.pdf` - what makes TS different, STL decomposition, stationarity + differencing (ADF), ACF/PACF, AR/MA/ARMA/ARIMA/SARIMA, exponential smoothing (Holt-Winters), baselines + MASE.
- **Lecture 08 (ML, 25 frames)** `08_ml_time_series.tex/.pdf` - forecasting as supervised learning, leakage + time-aware split, lag/rolling/calendar features, `TimeSeriesSplit`, gradient boosting + the trend-extrapolation trap (honest: GBM MAE 19 raw -> 6.7 differenced, MASE 0.39), model comparison (M5 vs M3/M4), deep + foundation models (TimesFM/Chronos-2/Moirai-2/TimeGPT/Time-MoE, from a 2026 web search).
- **14 Python figures** (`classical_figs.py` + `ml_figs.py`, seed 509, one shared synthetic monthly series). No TikZ, no external images. Both decks compile clean (0 `!`), overflow-checked page by page; the two worst overfulls (stationarity 76pt, differencing 17pt) fixed.
- **`07_time_series.qmd`** written per CONVENTIONS template, registered in `_quarto.yml` after `04_trees` (exact case, YAML validated). Videos / notes / Google Form / random image are `TBD` (not delivered yet).
- `concepts_checklist.csv` TS rows marked with deck coverage; two `_OUTLINE.md` files added.

### Pending / flags

- Not-yet-covered TS checklist rows left blank: **state-space / Kalman** (221) and **Granger causality** (224). Prophet is awareness-only.
- Local build used a throwaway `.tsvenv` (numpy/pandas/matplotlib/sklearn/statsmodels) because the Windows `ma/` venv isn't present in the Linux web container; figure scripts still document `ma` as the intended interpreter. TeX Live was apt-installed in-container.
- `.tsvenv/` is git-ignored via a scratch path, not committed; regenerate figures on the instructor machine with `ma` if needed.

### Next

- Record lectures 07/08, then add YouTube + `_notes` PDF links + Google Form to `07_time_series.qmd`; add a `00_random_image/07_*.jpg`; optionally add a solution notebook for the forecaster assignment.

## 2026-07-10 - Trees chapter (ml/04_trees) full rebuild from REVIEW.md

### Done

Addressed all of `ml/04_trees/REVIEW.md` in 7 phases (plan + status: `ml/04_trees/WORK_PLAN.md`),
12 commits `0377a55`..`0b208aa`:

- **Renumbered** trees decks L09-L12 -> **[17]-[20]** (real video slots; ch3 bank project stays [16]); swept every rendered cross-reference; fixed the REVIEW bug tier.
- **Content:** section transitions, cardinality-bias + one-hot/CART-binary frames, native-missing-values line (Context7-verified, sklearn >= 1.3), OOB caveat, calibration + imbalanced callbacks, worked XGBoost-on-Yerevan bridge, and 3 predict-first frames in [20] (was 0).
- **14 figures** across all four decks (staircase, extrapolation, XOR, readable depth-2 tree; RF boundary smoothing, max_features rho-curve, legend fix; AdaBoost weights, log-odds->sigmoid, round-50 overlay; monotonic constraint, real-data importance, stacking receipt; 4 TikZ schematics).
- **Titanic reality-check** (honest): pruned tree 0.835 > tuned RF 0.827 > untuned RF 0.809 - owned in a [18] frame, not papered over.
- **Chapter infra:** HW moved into `04_trees.qmd` (HW1-3), page built, `data/titanic.csv` pinned, `trees_project.ipynb` starter added.
- All four decks compile clean (17=44pp, 18=29pp, 19=31pp, 20=44pp); every new/edited frame overflow-checked.
- Earlier same session (ch3): decks 13-15 recompiled + clean/notes PDFs linked + 3 videos (exact titles, fixed swapped [11]/[12] links); bank project renamed to `16_bank_marketing_*`.

### Pending / flags

- `ml/00_plan.md` still lists trees as videos [16]-[19]; now disagrees with delivered [17]-[20] (left untouched per instruction).
- Deferred: hoist `\sectiontransition` into `ml/preamble.tex` (cross-chapter - also touches ch2 `09_regression_metrics` + ch4b `L12b_svm_and_classic_methods`).
- `04_trees.qmd`: Google Form + Random-image section are TBD; no video/notes links yet (lectures not recorded).

### Next

- Record trees lectures [17]-[20], then add video + `_notes` PDF links to `04_trees.qmd`.
- Reconcile the global video numbering in `ml/00_plan.md`.

## 2026-07-07 - Baseline snapshot (created retroactively from git log + memory)

### Current focus

ML course (`ml/`): chapters built out through neural networks; feature-engineering folder started.

### Recently completed (from git log)

- NN chapter: L15 lecture + practical (`ml/ch5_neural_networks/`)
- Classification (`ml/03_classification/`): deck renumbering, threshold-tuning split, SVM deck, imbalanced-learning deck [14], bank-marketing project, real figures
- Interpretability (`ml/05_interpretability/`): 3 decks + outlines + figures (bike data)
- Chapter 2 finalized, including the rent-audit solution with review fixes and Bayesian-opt appendix
- ML decks reorganized into numbered chapter folders

### Known pending (last tracked May 2026 - re-verify before acting)

- `math/Lectures/stat/10_stat.tex` (Classical Tests & LRT) lecture not delivered; HW 26/27 have problems but no solutions
- HW 23 (MLE/MAP): 0/11 inline solutions
- Optimization homework modules 09-15 still skeleton
- No stat homework `.qmd` for lectures 11-14 (ANOVA/A-B, regression inference, GLMs, causal inference)
- Stat lectures 11-16 not yet pedagogically reviewed
- ML curriculum gaps tracked in `ml/MISSING_TOPICS.md` and `ml/CURRICULUM_GAPS_PRE_NN.md`
- Deferred topics parking lot: `DEFERRED_TODO.md` (older scratch list: `debt.md`)

### Next

- (fill in at the next session)
