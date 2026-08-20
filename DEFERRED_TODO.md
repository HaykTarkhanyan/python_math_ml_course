# DEFERRED TOPICS — DO NOT FORGET

Things explicitly punted out of the main lecture flow. Park here so they don't get lost. Revisit when course pace allows or when student level is ready.

Last updated: 2026-07-25

---

## dl4nlp slides: mine Karpathy "Deep Dive into LLMs" for stage examples + high-level explanations

Instructor ask (2026-07-25). Source: Andrej Karpathy, "Deep Dive into LLMs like ChatGPT" (3h31m) — https://www.youtube.com/watch?v=7xTGNNLPyMI . This is the general-audience companion to his tokenizer video we already mined for deck 03. Full 24-chapter list is in the 2026-07-25 session log; key timestamps below.

**Goal:** across the dl4nlp decks, add more of Karpathy's concrete examples and his intuition-first "what is actually happening" framing for **every training stage** — pretraining, post-training (SFT), RLHF/RL. Our decks are currently more definitional than illustrative; the user wants the same motivation-first arc we gave deck 03.

Primary home: `07_pretraining_finetuning.tex`; also touch `01`, `06`, `09`, `18` where a stage already has a deck. **Interview the instructor on scope/depth first** (slide-style new-content workflow), then outline, then build. Cross-link deck 03 for the tokenization callbacks instead of duplicating.

Per-stage examples/framings to pull (timestamps into the video):
- **Pretraining data (1:00)** — internet as corpus; FineWeb-scale filtering funnel (dedup, language ID, quality, PII); "the model is a lossy compression of the internet."
- **Neural net I/O + internals + inference (14:27–31:09)** — next-token prediction *is* the whole objective; network as a fixed function with tunable knobs; sampling = one token at a time. Keep high-level, minimal math.
- **Pretraining → post-training (59:23)** — why a base model isn't a usable chatbot; the shift in both data and objective.
- **Post-training = conversations (1:01)** — SFT on human-written ideal assistant replies; the conversation/token format (ties to our `<|im_start|>` chat-tokens frame in deck 03); InstructGPT-style labelers + labeling instructions; "the assistant is a statistical imitation of a helpful human labeler."
- **Hallucinations, tool use, working vs long-term memory (1:20)** — why models make things up; "weights = vague memory, context tokens = working memory"; tool use (search, code) to reduce hallucination.
- **Knowledge of self (1:41)** — "who are you" answers are trained/spoofable, not intrinsic.
- **Models need tokens to think (1:46)** — spread reasoning across tokens; why "explain then answer" beats "answer then explain"; mental-arithmetic example.
- **Tokenization revisited / spelling (2:01)** — count-the-r's, reversing strings. Cross-link deck 03, don't duplicate.
- **SFT → RL, RLHF, DeepSeek-R1, AlphaGo (2:07–3:09)** — RL as "practice problems with verifiable answers"; emergence of reasoning traces; RLHF reward model + its gameability; AlphaGo "move 37" analogy. Home: `18_reinforcement_learning.tex`.

## dl4nlp reminder: show a base model MEMORIZING and NOT being an assistant yet

Explicit instructor ask (2026-07-25). At/just before the pretraining→post-training transition, add frame(s) that make the base-vs-assistant distinction visceral (Karpathy does this ~42:52, "Llama 3.1 base model inference"):
- **Memorization/regurgitation:** prompt a base model (Llama 3.1 base / GPT-2) with the opening of a famous Wikipedia article or poem → it continues *verbatim*. It has memorized chunks of the training set.
- **Not an assistant yet:** ask a *base* model a question → it does NOT answer; it autocompletes more questions or rambles like a web document. The helpful "assistant" only appears after post-training (SFT).
- **It's a document simulator:** base model "dreams internet documents"; few-shot prompting can coax useful behavior, but there's no built-in Q&A persona.
- Delivery: live demo (Hyperbolic/Together base-model endpoint, or a small local base model) vs annotated screenshots — decide in the interview. Raster screenshots are fine (deck 03 already embeds PNGs).
- Home: `07_pretraining_finetuning.tex`, at the base-model / "why post-training" moment.

---

## From bias-variance deck (`ml/upcoming_lectures/L01d2_bias_variance.tex`)

The full deck got cut. Status of the 3 surviving concepts (as of 2026-06-19):

1. **Irreducible error / Bayes risk** — adopted into L03 regularization (see `L03_OUTLINE.md`). Becomes the "noise floor" frame in the "Why regularize?" section. ✅
2. **Approximation vs Estimation error** — adopted into L03 regularization. Replaces the existing "model bias / estimation bias" frames using the canonical names. ✅
3. **Double descent** — **DEFERRED FURTHER.** Originally planned for L03, but user wants it pushed back. Natural home: the neural networks chapter, where over-parameterization is the rule and double descent stops feeling like a wrinkle and starts feeling like the regime. Frames stay drafted in `L01d2_bias_variance.tex` under `\section{Modern wrinkle: double descent}` — lift them when ch5 is built.

**Action:** after L03 lifts concepts 1 and 2, archive the rest of `L01d2_bias_variance.tex` — only the double descent section needs to survive for later.

---

## From `ml/deferred/` folder

Pre-built decks parked because they need more statistics background than the current cohort has. Each is mostly compile-ready.

> **Re-evaluated 2026-08-08 and deliberately kept parked.** All three were test-compiled and
> **all three build cleanly** - "mostly compile-ready" is now "compiles". They are still not
> registered in `_quarto.yml`, by instructor decision: the statistics-background reason above
> still holds, and being finished is not the same as being ready to teach.
>
> Note for the next revisit: **GLMs' stated condition is already met** - logistic regression is
> taught and delivered in the classification chapter. That one is unblocked whenever the cohort
> is ready; the other two still need the placement question answered.

### `deferred_glms.tex` — Generalized Linear Models (29 frames)

Sections: Why GLMs / The GLM Framework / OLS as GLM / Logistic as GLM / Poisson regression / Estimation & inference (IRLS, deviance) / Family tree / Choosing the right GLM / Practical Python.

**When to revisit:** after logistic regression is taught in the classification chapter. GLMs are the natural unification — "logistic and linear are two faces of the same thing." Needs MLE comfort.

### `deferred_causal_inference.tex` — Causal Inference (37 frames)

Sections: Correlation != causation / RCTs / Potential outcomes (ATE) / DAGs (chain, fork, collider) / Observational strategies (regression, matching, propensity score, IVs) / Doing vs seeing / Backdoor criterion / Ladder of causation / Practical checklist.

**When to revisit:** could be a standalone bonus lecture once the prediction chapter is solid. High value, very different mental model from prediction. Pair with a Simpson's-paradox demo.

### `deferred_regression_inference.tex` — Regression Inference / Coefficient Significance (31 frames)

Sections: Linear model assumptions / Sampling distribution of beta-hat / t-test for coefficients / Confidence intervals / F-test / Prediction intervals vs confidence intervals / Diagnostics for assumptions.

**When to revisit:** this is the *statistics* side of regression (p-values, CIs, hypothesis tests) that we're currently skipping. Ties to the broader stat lectures in `math/Lectures/stat/`. Could insert between L01c and L01d as an optional sidebar, OR fold into a dedicated "regression for inference vs regression for prediction" lecture.

---

## Other deferred ideas (collected over sessions)

- **1-SE rule** (Tibshirani): pick simplest model within 1 SE of best CV score. Removed from L01d. Reintroduce when regularization is taught.
- **LOOCV** (leave-one-out CV): removed from L01d. Mention as the k=n boundary case when discussing CV bias-variance.
- **Out-of-bag (OOB) error**: parallel to CV via bootstrap. Natural to introduce alongside random forests.
- **Bootstrap as practical bias/variance measurement.** Closes the loop between theory and lab.
- **Repeated K-fold motivation:** brief frame on "when CV scores are jumpy, repeat the CV."
- **Hyperband / successive halving** (`HalvingGridSearchCV`): visualize the bracket. For L01e.
- **Quantile / pinball loss** for non-mean prediction. For L01f.
- **MASE** for time-series metrics. For L01f.
- **Binning / discretization** as cheap nonlinearity for linear models. For L01g.
- **Mutual information filter** (`mutual_info_regression`). For L01h.
- **mRMR** feature selection. For L01h.
- **Selection vs PCA distinction** — interpretable vs not. For L01h.

---

## Cross-deck recurring idea

A **single worked example threaded across L01d / L01d2 / L01e / L01f / L01h** (rental prices? bike-share counts?). Each deck references back: "remember the bias-variance plot from L01d2 — this is what the validation curve from L01d looks like for that case." Heavy refactor, defer until decks are otherwise stable.

---

## Housekeeping deferred

- **Migrate or delete the auto-memory folder.** As of 2026-06-19 we switched to "all persistence in repo files, never write memory." But ~13 pre-existing memory files still live at `~/.claude/projects/C--Users-hayk--OneDrive-Desktop-01-python-math-ml-course/memory/` (user role, feedback rules, pedagogy notes, course completion status, etc.). Decide per file: copy still-useful content into `CLAUDE.md` / `CONVENTIONS.md` / `LEARNINGS.md`, then delete the memory folder. Until done, future sessions may still load those memories as context.

- **Decide the DL-track file naming, then make `CONVENTIONS.md` true.** (Found 2026-08-04.) The convention file says lecture slides are `NN_topic.tex` matching playlist position, and that "No `L01`, `L01b`, `L01c` style prefixes — those are legacy and being phased out." But **nine chapters** use `LNN_topic.tex` inside `chN_name/` folders: `09_clustering`, `10_dimensionality_reduction`, `ch5_neural_networks`, `ch6_cnn`, `ch7_rnn`, `ch8_autoencoders`, `ch8b_gans`, `ch9_attention`, `ch10_diffusion`. (`07_classic_methods/L12b` was on this list until 2026-08-12, when it was renamed to `28_svm_and_classic_methods` — its playlist number became knowable once the feature-engineering lectures were fixed at 26/27.) Their chapter pages are also bare (`09_clustering.qmd`, `gans.qmd`) rather than the documented `NN_chapter_topic.qmd`. Two honest resolutions: (a) update `CONVENTIONS.md` to describe the real two-track scheme — playlist-numbered `NN_` for the delivered classic-ML track, `LNN_` for the not-yet-scheduled DL track — or (b) renumber the DL track once its playlist order is known. **(a) is cheap now; (b) gets more expensive with every deck.** Note the DL lecture numbers currently *collide* with the classic ones (`24_shap_lime` vs `L24_attention`).

- **Resolve the duplicate feature-engineering decks.** (Found 2026-08-04.) `ml/06_feature_engineering/` holds two compiled versions of the same two topics: `26_feature_engineering` + `27_feature_selection` (Jul 2026) and `L01g_feature_engineering` + `L01h_feature_selection` (Jun 2026). Pick the live pair, move the other to an archive folder per the repo-structure tiers. The chapter has no `.qmd`, so neither is on the site yet.

- **One learnings home, not two.** `LEARNINGS.md` (root, legacy) and `_learnings/` (the current per-file convention) both exist. Split the former into dated files and delete it.

---

## Planned: Diagnostic plots section (for `[12]` metrics, or a standalone)

A "what to actually plot when debugging a classifier" section. Some plots already live in L12
(ROC, PR, lift, threshold-metric curves, confusion matrices) and calibration is in `[13]`; this
section would consolidate the visual toolkit and add the missing pieces. Proposed frames (each a
real figure via a `py_src/diagnostics_demo.py` on the cheese model):

1. **Score distribution by class** — histogram/KDE of predicted scores split by true label.
   Shows class separation/overlap and whether any threshold cuts cleanly. *(new figure)*
2. **Confusion-matrix heatmaps** — binary + the `K×K` `cm_multiclass.pdf` (already built). *(reuse)*
3. **Per-class metrics bar chart** — precision/recall/F1 per class with support; surfaces the
   weak class. *(new figure)*
4. **Error analysis** — the most confident *wrong* predictions / hardest examples (small table or
   annotated scatter): "look at what it gets wrong." *(new figure/table)*
5. **Learning curve** — train vs validation score vs training-set size → over/underfitting
   diagnosis (more data vs more model). *(new figure)*
6. Cross-reference the curves already in the deck (ROC / PR / lift / threshold) and reliability in `[13]`.

Scope ~5 frames + a transition. Decide whether it lives at the end of `[12]` or as its own short deck.

---

## Cut from ch12 (vision-language models), 2026-08-07

Both cut by the "figures only, no training" decision (`DECISIONS.md` #9), not because they are
bad ideas. Either would make a strong homework for the chapter, which currently has none.

1. **The AR-vs-diffusion head-to-head on the ՊԱՆԻՐ letters.** Train a VQ-VAE plus a small
   autoregressive transformer on `ml/ch10_diffusion/data/mashtots_panir_24.npz`, generate the
   word one token at a time, and put it next to ch10's diffusion samples. This is the
   Chameleon-vs-Transfusion argument reduced to a dataset students already know, on a machine
   with no GPU - both models are tiny (24x24 inputs, 6x6 token grids, minutes on CPU). L34
   currently *asserts* the tradeoff and cites published work; this would demonstrate it.
2. **A real CLIP zero-shot run on the Armenian letters.** Ask CLIP to name Պ / Ա / Ն / Ի / Ր
   from sentence prompts. Expected to fail - CLIP has barely seen Armenian script - and the
   failure is the teachable part: it shows concretely what "zero-shot" is bounded by.
   Blocked on a dependency choice (`open_clip` vs `transformers`) plus a ~350 MB download,
   which is the instructor's call, not Claude's.

---

## Cut from ch16 (JEPA and world models), 2026-08-08

Both cut by instructor decision at plan time, not because they are weak. Plan:
`ml/ch16_jepa/JEPA_CHAPTER_PLAN.md`, decisions 2 and 3. Research: `_knowledge/jepa/`.

1. **The toy collapse experiment.** "Not yet" - deferred, not cancelled. Two small MLP encoders on
   2-D synthetic data, trained with the JEPA objective **with and without** the EMA teacher,
   plotting **embedding variance against step** with the loss curve alongside. The no-EMA run's
   variance goes to zero while its loss looks excellent, which is the whole point: **the loss does
   not tell you it broke.** Seconds of CPU, no GPU, no large arrays - the machine-load rule is not
   in play here, this was a scope decision.

   Why it is worth doing later: L39 currently *asserts* that a low loss is compatible with a
   worthless representation, and argues it from the frame-13 degenerate solution. That is sound but
   it is arithmetic, not evidence. The chapter's central danger stays abstract without this.

   **It slots in as L39 frame 15b and renumbers nothing** - the plan was written to absorb it.

2. **The LeWorldModel term project.** The one genuinely reproducible artifact in the whole chapter:
   a compact JEPA world model trained end to end from raw pixels, **two loss terms, one
   hyperparameter, a single GPU and a few hours** (arXiv 2603.19312, id from the Welch Labs
   interview - paper not yet read). It exercises everything the chapter teaches in one artifact -
   latent prediction, anti-collapse regularisation, action conditioning, and goal-conditioned CEM
   planning.

   Everything else in ch16 is a model no student can reproduce (1M hours of video, 16 A100s). This
   one is a term project, and that gap is exactly why it is worth revisiting. Note the machine
   constraint: this needs a GPU, so it is a Colab job, not a laptop job.

   Project ideas stay recorded in `_knowledge/jepa/06_teaching_notes.md` as research.

---

## ch6 CNN — trim L16's colour section now that deck 33 exists

**Deferred 2026-08-20.** `ml/09_clustering/33_color_spaces.tex` now teaches cones, RGB, HSV and
grayscale roughly three weeks before `L16_cnn_foundations.tex` reaches them. `L16` Section 1 still
carries its own four frames plus its own copies of `eye_cones.py`, `rgb_channels.py` and
`hsv_space.py` — the instructor's call was **copy, not move**, so `L16` still compiles standalone.

The open question is whether `L16`'s Section 1 should shrink to a single recap frame pointing back
at deck 33. Arguments both ways:

- **Trim it:** four duplicated frames is four frames of a three-lecture CNN block spent on material
  already delivered, and the duplicate figure scripts can drift apart.
- **Keep it:** `L16` is currently self-contained and can be lifted into another course; the CNN
  chapter is not scheduled until ~Sep 4, and by then the colour material will be three weeks stale
  for the students.

Decide when the CNN chapter is next touched, not before. If it gets trimmed, the astronaut-based
figures in `ch6_cnn/fig/` can go with it; deck 33's copies are Saryan-based and independent.

---

## How to use this file

- Top-level of repo means it appears in every `ls` and every `git status`. Visible by design.
- When a deferred topic gets adopted into a real lecture, DELETE that line (don't strike through — keeps the file short).
- Add new deferred items here the moment they get cut from a deck. The longer you wait, the more you forget the context.
