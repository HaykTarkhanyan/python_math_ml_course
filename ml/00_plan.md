# Course plan

Last updated: 2026-10-15 (the "Next sessions" section only; everything below it is as of 2026-08-08)

**Cadence: Wednesday / Friday / Sunday**, three sessions a week (changed from the earlier
Tuesday/Thursday pattern). Times on the "Metric" Google Calendar.

**Projected finish: mid-to-late November 2026** — see [Projected finish](#projected-finish).
**Stale as of 2026-10-14:** that projection had LLMs on Sep 27 - Oct 14; they now start Oct 21.

---

## Next sessions (set 2026-10-14, renumbered 2026-10-15)

Every session is 20:00-22:00 Yerevan time (18:00-20:00 Munich), per the "Metric" calendar
(recurring event "MA"). Playlist numbers after [51] are projected, not assigned yet. On
2026-10-15 the CNN decks moved down one number to match the playlist (DECISIONS #83): [48]
foundations, [49] architectures, [50] transfer learning, [51] vision tasks.

| Date | Day | # | Session | Material | State |
|---|---|---|---|---|---|
| ~~**Oct 14**~~ | Wed | [50] + [51] | ✅ Transfer learning + Vision tasks - **delivered**, notes exported | `12_cnn/50_transfer_learning` (32 frames) · `12_cnn/51_vision_tasks` (49 frames) | Clean PDFs recompiled 2026-10-15; both notes PDFs linked on the chapter page |
| **Oct 16** | Fri | [52] | *Practical* - Gesture Snake | `12_cnn/gesture_snake/52_gesture_snake_solution.ipynb` · `STUDENT_GUIDE.md` | Built and executed 2026-10-01 |
| **Oct 18** | Sun | [53] + [54] | RNN foundations + The road to attention | `13_rnns/L20_rnn_foundations` (30 frames) · `13_rnns/L21_road_to_attention` (25 frames) | Built, last edited Oct 3-5, on GitHub. Rename to `53_`/`54_` at delivery |
| **Oct 21** | Wed | [55] | LLMs, session 1 | `14_llms/LLM1_tokenization` (40 frames), if the chapter order holds - see risk 5 | Built 2026-10-04, not student-reviewed (LLM-3 to LLM-5 were) |

### Risks and open questions (critique of this plan)

1. **Oct 14 was the heaviest session yet** (81 frames in 120 minutes; [48] alone was 73).
   Delivered. The notes PDFs show where the lecture actually got to - worth checking before
   planning the next double session.
2. ~~**The CNN chapter students see online is still L16-L19.**~~ Resolved 2026-10-15: the
   chapter (decks, notes PDFs, notebooks, chapter page) committed and pushed.
3. **Friday's risk is setup, not content.** Each student needs a webcam, torch + opencv from
   `requirements.txt`, and 500 recorded crops before any training (training is ~3 min on CPU).
   Send the STUDENT_GUIDE install section ahead and ask everyone to run `record_gestures.py`
   once before Friday. Still open in `DEFERRED_TODO.md`: ship the instructor's 500 crops
   (2.2 MB) as starter data for students without a webcam (burst 1 has the planted sleeve
   shortcut). The order is right: Grad-CAM and the frozen ResNet-18 come from [50] two days
   earlier.
4. **Sunday: two decks fit, the RNN practical does not.** 55 frames is lighter than Oct 14. But
   `13_rnns/xx_rnn_memory` ("How far back can a network remember?") gets no slot in this plan:
   cut it, make it homework, or give it a session. The RNN decks have not had the kind of
   instructor review the CNN decks got this week (13 change requests for [51] alone). If they
   need one, do it Thursday or Saturday, not on Sunday.
5. **"Wednesday we do transformers" conflicts with the chapter as built.** The LLM chapter runs
   in forward-pass order: LLM-1 tokenization, LLM-2 embeddings and position, LLM-3 attention,
   LLM-4 the transformer block (DECISIONS #67). L21's closing Next box promises tokenization
   next, and LLM-3 builds on LLM-2's embeddings. Starting at attention means skipping LLM-1/2 or
   reversing #67. Recommended: Oct 21 = LLM-1 tokenization, as L21 promises.
6. **There is no course end date, and the old one is gone.** The LLM chapter is 14 decks
   (DECISIONS #73 removed the session cap), plus RAG (L41-L43) and agents (L44): 18 decks. At one
   deck per session and no practicals, LLM-14 lands on Fri Nov 20 and agents on Sun Nov 29.
   Still unscheduled after that, from the 08-08 plan: diffusion, RL, VLM, audio, VLA, tabular
   FMs (~12-15 sessions), which pushes past mid-December. If there is a hard end date, section 8
   of `14_llms/LLM_CHAPTER_PLAN.md` has a 10-session cut.
7. **Autoencoders and GANs dropped out of the order.** The 08-08 plan put them between RNN and
   LLMs; this plan goes straight from RNN to LLMs. Cut, or after LLMs? Their decks assume RNN
   came right before (`L23_vae`'s Next box and `L23c`'s king-queen callback point at L21), and
   [51]'s last frame teases GANs.
8. **The class moves to 21:00 Yerevan on Sun Oct 25.** The recurring "MA" event is pinned to
   Europe/Berlin at 18:00. Europe leaves summer time on Oct 25 and Armenia does not, so the
   calendar shows Oct 23 at 20:00+04 and Oct 25 at 21:00+04. If the class should stay at 20:00
   in Yerevan, change the event's time zone to Asia/Yerevan (then it is 17:00 in Munich).
9. **Small:** `[49]` (architectures) has no `_notes.pdf` on disk, so its chapter-page line has
   no notes link. Export it if it exists.

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
| **Aug 14** | Fri | Time series lecture | `29_classical_time_series` · `30_ml_time_series` |
| **Aug 16** | Sun | *Practical* — Armenia electricity / gas usage | `31_electricity_forecast_solution.ipynb` |
| **Aug 19** | Wed | Clustering — **ch09** | `32_clustering` |
| **Aug 21** | Fri | *Interlude* — color spaces, then the practical | `33_color_spaces` |
| **Aug 21** | Fri | *Practical* — Lake Sevan land cover — **delivered** | `34_land_cover_solution.ipynb` |
| **Sep 2** | Tue | *Practical* — Saryan painting, colour quantization — **delivered** | `35_image_compression_solution.ipynb` |
| **unscheduled** | — | *Practical* — the Armenian word tree. Built and executing clean; carries an `xx_` prefix until a session number is known. | `xx_semantic_tree_solution.ipynb` |
| ~~**Aug 23**~~ | Sun | ✅ Dimensionality reduction — **ch10** — **delivered**, notes exported, video [36] live, qmd linked | `35_dimensionality_reduction` |
| ~~**Aug 26**~~ | Tue | ✅ UMAP in depth — **delivered**, notes exported, video [37] live, qmd linked | `36_umap` |

After the UMAP lecture the classic ML track (chapters 01–10) is fully delivered, and everything
remaining is the deep-learning half. **✅ Reached — as of 2026-09-03 both ch10 lectures are
delivered, with notes and videos linked on the chapter page. Next up: the ch10 practical
(eigenfaces + LFW + CLIP photo-grouping), then neural networks.**

**Numbering note (2026-09-03).** The YouTube playlist numbers the two ch10 lectures **[36]** and
**[37]** — the ch09 practicals were published as separate videos [34] (Sevan) and [35] (image
compression) — while the ch10 files on disk are `35_*`/`36_*`/`37_*`. Disk numbering now trails
the playlist by one from dimensionality reduction onward (and `35_image_compression` in ch09
shares its number with `35_dimensionality_reduction` in ch10). The chapter qmd labels the
bullets by the *video* numbers. Whether to renumber the ch10 files to 36/37/38 is an open
decision — per DECISIONS #22 this was the anticipated "recorded as separate sessions" case.

**Numbering note (2026-08-16).** Clustering and dimensionality reduction were renumbered to
continue the global sequence: folders `ml/09_clustering/` and `ml/10_dimensionality_reduction/`,
files `32`–`36`, replacing the legacy `ch4_clustering` / `ch4b_...` and `L13`/`L13b`/`L13c` names.
The clustering practicals share one number because they are one practical session, matching the
`21_*` precedent in ch04. Careful: `ml/10_dimensionality_reduction/` is **not** `ml/ch10_diffusion/`
— the remaining `chNN_` folders are legacy and still need their own renumbering pass.

**Numbering note (2026-08-20).** The new colour-spaces interlude took `33`, so everything after it
shifted by one: the three clustering practicals `33_*` → `34_*`, `34_dimensionality_reduction` →
`35_`, `35_umap` → `36_`, `36_eigenfaces_solution` → `37_`. Both chapter folders were still
**untracked** in git at the time, so the rename cost no history — but it also means `git grep`
could not see them. The reference sweep had to use ripgrep: `git grep` found 4 files, ripgrep
found 13. See `_learnings/2026-08-20-1900_git-grep-cannot-see-untracked-work.md`.

---

## Projected — the deep learning half

Extrapolated Wed/Fri/Sun from Aug 26 with no breaks. Dates are the **low end** of each range;
every extra session pushes everything after it by 2–3 days.

| Block | Sessions | Dates | Built material |
|---|---|---|---|
| Dim reduction practical | 1 | Aug 28 | `xx_genes_geography_solution.ipynb` (instructor-led, 1000 Genomes; built 2026-09-03, DECISIONS #32) · `37_eigenfaces_solution.ipynb` + the LFW recognition project · homework = Project 3 (CLIP photo map) |
| Neural networks | 2–3 | Aug 30, Sep 1 | ✅ **all seven lecture decks delivered** ([40], [41_1]/[41_2] on Aug 30 / Sep 1, [42] and [43] on 2026-09-17, [44] and [45] on 2026-09-20), notes exported, videos not uploaded yet: `40_intro_history` · `41_1_neuron_to_network` · `41_2_multilayer_nets` · `42_training_backprop` · `43_nn_regularization` · `44_optimization` · `45_optimization_init_activations` (folder `ml/11_neural_networks/`, renamed from `ch5_neural_networks` 2026-09-16). Remaining in the chapter: the practical (`47_name_inventor_solution.ipynb`) |
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

The largest block in the course and the least built. `ml/14_llms/` holds **L24 only**
(53pp); the planned **L25 and L26 do not exist**. `ATTENTION_CHAPTER_PLAN.md` carries the
approved outline for all three.

The rest of the block would come from **`misc/dl4nlp/`** — 18 decks, ~442 pages, already in the
house palette — which is currently outside `ml/` and unregistered in `_quarto.yml`. Folding it
into `14_llms` (or a new LLM chapter) is a decision, not a mechanical move: it changes
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
