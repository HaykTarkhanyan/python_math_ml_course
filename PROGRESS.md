# PROGRESS.md

Session log: what was done, what is in progress, what comes next. Newest entry first.
Update at the end of every session (`wrap-session` skill). Entries are point-in-time
snapshots - for anything old, verify against `git log` before acting on it.

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
