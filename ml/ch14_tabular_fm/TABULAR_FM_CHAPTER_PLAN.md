# Chapter plan — Tabular Foundation Models (L37)

**Status:** REVISED DRAFT after independent pedagogical review, awaiting approval.
**Revision 2, 2026-08-08.** Rev 1 aimed the chapter's thesis at the wrong deck; see below.

**Research:** `_knowledge/tabular_foundation_models/` (9 files), distilled from the instructor's
own `~/OneDrive/Desktop/mech_interp/` repo (39 title-verified TabPFN papers) plus web checks.
**Review:** `_knowledge/tabular_foundation_models/review_pedagogy.md`.

## Instructor decisions (2026-08-07)

1. **Placement: ch14 at the end, plus back-pointers.** `ml/ch14_tabular_fm/`, deck **L37**.
   Back-pointer callouts get added to the **ch04** and **ch06** qmd pages.
2. **No mechanistic interpretability.** Collected for a different purpose; stays in `_knowledge/`
   as research, not course content.
3. **Theory: one or two frames, stated not derived.** Deliberate deviation from
   `ml/SLIDE_STYLE.md`, matching ch11 and ch12.
4. **No practical, no project, no training.** Explanatory slides only.

**Inferred and stated:** with 2 and 4 applied there is **one deck of material, not two.**

## What changed in revision 2 (all from the review, all verified before accepting)

**1. The thesis was aimed at the wrong chapter.** Rev 1 claimed this deck undercuts
`02_main_concepts/08_hyperparameter_tuning.tex`. Reading that deck, it does not: it is a
model-agnostic deck about *how to search a space*, it never claims tuning is mandatory, and it
already concedes the limits itself - *"on a small 2-HP space no method works magic"* (RMSE 55-57
across grid/random/Optuna) and *"Tune on 10% of training data... Often within 1% of
fully-tuned."* A model with no knobs falls outside that deck's scope rather than contradicting
it. A student who watched it would notice.

The real collision is **ch04**, and it is much sharper. `04_trees/20_advanced_boosting.tex`
closes on a `paramgreen` takeaway box: **"Trees + ensembles dominate tabular data."** Also
*"reach for boosting on tabular data"* and *"Often the top performer on tabular benchmarks - if
properly tuned."* Those sentences are what this chapter bounds. Quote the box verbatim, then
scope it to small n. Real argument, real citation, and true.

**2. Frame 25 argued with a strawman.** Rev 1 "defended" ch06 against an accusation nobody made.
`26_feature_engineering.tex` already says *"feature engineering is not dead --- it moved"* and
already measured FE at **+81.5 MAE to Ridge and -4.3 to a forest** - it already shows feature
engineering *hurting* a tree ensemble. Reframed as a callback to ch06's own number.

**3. Evidence now comes before mechanism.** Rev 1 spent ten frames on architecture before
substantiating a claim students have been taught to disbelieve. The result and its scope
condition now land in the cold open.

**4. The prior gets three frames and a figure**, not one. Rev 1 called frame 8 "the frame to
slow down on" and then gave it one slide and no figure. If the prior does not land, half the
deck fails.

**5. Added:** a predict-first frame (house style, previously zero), a worked-numbers frame
(house style, previously none), a canonical code snippet, and a misconception pre-empt.

**6. Section 4 reduced to one content frame**, titled by its checkable consequence. The causal
prior-bias frame was cut to a single sentence - one frame with no setup would have lost the room.

## Why this chapter exists

Chapter 4 concludes that trees and ensembles dominate tabular data. That was true when it was
written and it is still true in most of the space. This chapter shows where it stops being true:
a transformer pre-trained once on **synthetic** data, with no fitting to your dataset at all,
that wins on small tables.

It is different in kind from ch10-ch13, which each *added* a capability. This one **bounds a
claim the course already made**. Write it as an argument, not a survey.

## Deck outline (~30 frames)

### Cold open
1. **Predict first.** A model that has never seen a single real dataset, against a properly
   tuned gradient-boosted ensemble, on 1,000 rows. Which wins? Let them commit. *(This converts
   frame 2 from an assertion into a bet the student has already lost.)*
2. **The result, with its scope condition attached immediately.** The Nature 2025 numbers, and
   what "small data" means precisely. **Not** a schematic - see build blocker 1.
3. **What ch04 told you.** Quote the closing box verbatim: *"Trees + ensembles dominate tabular
   data."* True, and it now has a boundary. That boundary is this lecture.
4. Outline.

### Section 1 - The inversion
5. `[plain]`: *Stop fitting the model to the data.*
6. **Ordinary supervised learning vs a prior-data fitted network.** *Figure: `pfn_inversion`.*
7. **The recipe in four steps.** Write a prior over datasets; sample millions of synthetic ones;
   train one transformer; ship it and never train again.
8. **At inference there is no fitting.** Your training table *is* the context. Signposted to
   in-context learning from `llm_training`.
9. **The prior, part 1 - what is in it.** Structural causal models (SCM) and Bayesian neural
   networks (BNN), sampled. No real tables. *Figure: `prior_samples` - actual draws from a
   prior like this, so "synthetic dataset" stops being an abstraction.*
10. **The prior, part 2 - the prior IS the inductive bias.** Everything later in the deck
    follows from this one idea: drift needing its own prior, bias that does not wash out,
    Real-TabPFN's continued pre-training.
11. **Misconception pre-empt: "no training" is not "no cost."** The pre-training happened once,
    expensively, and somebody else paid for it. What moved is *who* pays and *when*.

### Section 2 - The architecture
12. `[plain]`: *A table is not a sentence.*
13. **Each cell gets its own representation** - not each row, not each token.
14. **Two attentions per layer:** over features (across a row), then over samples (down a
    column), then an MLP. *Figure: `two_d_attention`.*
15. **Predict first:** permute the columns of the test table. Does the prediction change?
    Reveal: no, and the invariance is **architectural, not learned**. Callback to ch6 - same
    design idea as a CNN's translation equivariance, different symmetry.
16. **Worked numbers - why the factorisation is not just tidy.** Flattening all `n . m` cells
    into one sequence costs `(n . m)^2`. The two-way factorisation costs `O(n^2 + m^2)`. At
    n = 10,000, m = 100 that is `10^12` against `10^8` - **about 10,000x**. Memory is `O(n . m)`,
    linear. *Figure: `context_cost`, real axes.* **The only number in the deck the student
    derives rather than receives.**
17. **So the ceiling is rows, and here is where it sits.** Nature: wins up to **10,000
    samples**; TabPFN-2.5 reaches 50,000 x 2,000. Separately, the paper reports *running* on up
    to 50 million cells on an H100 - **capacity and superiority are different claims**, and the
    frame says so.

### Section 3 - The landscape
18. `[plain]`: *Not one model, and not one idea.*
19. **The model line**, as a `\modeltransition` card rather than a frame: v1 (2022) -> v2
    (Nature 2025) -> 2.5 -> 3.
20. **It is not a monopoly**, on the two axes that actually differ - where the pre-training data
    comes from, and how many rows fit: TabICL/TabICLv2, TabDPT, LimiX. Folds in the
    "it generalises" list (time series, drift, wide, generation).
21. **How you would actually use it.** One canonical snippet, `fit` / `predict`, three lines.
    Without this the lecture is unactionable.

### Section 4 - What is it doing? (one content frame, per decision 3)
22. `[plain]`: *Nobody agrees, including the authors.*
23. **"Why does it keep improving on datasets bigger than anything it pre-trained on?"** - the
    dispute, titled by its checkable consequence rather than by the two camps. Bayesian
    (Müller, Hollmann & Hutter - the TabPFN authors) vs frequentist (Nagler). Stated, not
    derived. Closing sentence carries the causal result in plain words: **the prior does not
    wash out as data grows**, which lands because frame 10 already said the prior is the model.

### Section 5 - Where it stops winning
24. `[plain]`: *The honest half.*
25. **Six papers exist only to make it bigger.** Sketching, distillation, TuneTables, Unleashed,
    chunking, Wide. That is the shape of the method, not a frontier. *Figure:
    `scaling_timeline`.*
26. **Distribution shift needed its own model.** Drift-Resilient TabPFN fixes it by putting
    shift *into the prior* - elegant, and it tells you the base model does not handle it.
27. **The benchmark problem.** A model pre-trained on a prior over structural causal models will
    look good on benchmarks that resemble that prior. *Realistic Evaluation in Open
    Environments.* Ties to `math/27_stat_how_to_lie`.
28. **Feature engineering: ch06 already measured this.** FE is worth **+81.5 MAE to Ridge and
    -4.3 to a forest**; ch06's own honest summary is *"not dead - it moved."* Same direction,
    now at scale. A callback, not a correction.
29. **The licence.** Research and internal evaluation only; commercial and production use
    excluded, **outputs included**. TabICLv2 is open. A real engineering decision.

### Wrap-up (`\section*{Wrap-up}`, outside Section 5)
30. **Recap** + a **"what to do with this"** box pointing back at ch04 with the scope condition,
    since this is the last deck in the L-sequence and an empty "Next:" ritual would be worse
    than none.

## Figure budget (Python, `py_src/` → `fig/`, nothing trained)

| Figure | Frame | What it shows |
|---|---|---|
| `pfn_inversion` | 6 | fit-to-your-data vs fit-to-a-prior-over-datasets |
| `prior_samples` | 9 | real draws from an SCM-like prior, so "synthetic dataset" is concrete |
| `two_d_attention` | 14 | a table grid with row-attention and column-attention overlaid |
| `context_cost` | 16 | cells in context and quadratic attention cost, real axes |
| `scaling_timeline` | 25 | the six scaling papers over time |

Nothing is downloaded, installed or run - consistent with decision 4 and with the licence.

## House-style fixes carried into the build

- Recap goes under `\section*{Wrap-up}`, not inside Section 5 (pattern:
  `20_advanced_boosting.tex`, `26_feature_engineering.tex`).
- **Expand on first use:** SCM, BNN, TabICL, TabDPT, LimiX. PFN is expanded at frame 6.
- Back-pointers in ch04/ch06 are **callouts carrying the scope condition** ("under roughly 10k
  rows"), not bare sentences - a naked "see ch14" reads as contradicting the lecture the student
  just watched.
- Google Form quiz: **yes.** With no homework it is the only assessment surface left.

## Why no practical, recorded rather than assumed

ch11 has a project because reinforcement learning is a method students should run. ch12, ch13
and ch14 are **literacy chapters** - both `vlm.qmd` and `audio.qmd` currently carry
`Տնային: TBD` too, so this is a consistent line rather than an exception.

The cost is real and worth naming: students *measured* tuning in ch04 and *measured* feature
engineering at +81.5/-4.3 in ch06. A chapter that bounds those claims using only citations is
epistemically weaker than the chapters it bounds. Frame 16's context arithmetic is the
mitigation - one number the student derives, no download, no licence problem.

## Build blockers - BOTH RESOLVED 2026-08-08 from the primary paper

Read out of `nature2025_*.pdf` (Nature vol 637, 9 January 2025, p.319; Hollmann, Müller,
Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister & Hutter).

**Blocker 1 - frame 2's numbers. Resolved, and the paper is better than hoped.** Verbatim:

> "gradient-boosted decision trees have **dominated tabular data for the past 20 years**...
> Here we present the Tabular Prior-data Fitted Network (TabPFN), a tabular foundation model
> that outperforms all previous methods on datasets with **up to 10,000 samples** by a wide
> margin, using substantially less training time. **In 2.8 s, TabPFN outperforms an ensemble of
> the strongest baselines tuned for 4 h** in a classification setting."

The paper's own opening sentence states ch04's claim ("dominated tabular data for the past 20
years") and then bounds it. That is frame 3's citation, from the challenger's own mouth.
**2.8 seconds vs 4 hours tuned** is the cold open. **10,000 samples** is the scope condition,
and it is the number that goes in the ch04/ch06 back-pointer callouts.

**A distinction that is easy to get wrong and must not be:** "up to 10,000 samples" is a
*quality* claim (where it wins). The paper separately reports running on **up to 50 million
cells, for example 5 million rows x 10 features, on a single H100**, at "less than 1,000 bytes
per cell". Capacity and superiority are different numbers. Do not merge them.

**Blocker 2 - frame 16's arithmetic. Resolved, and MY PLANNED VERSION WAS WRONG.**

Rev 2 said "n rows x d features = cells in context, and attention is quadratic in that", i.e.
O((nm)^2). The paper says:

> "The compute requirements of this architecture scale quadratically with the number of samples
> (n) and the number of features (m), that is **O(n^2 + m^2)**, and the memory requirements
> scale linearly in the dataset size, **O(n . m)**."

Quadratic in each dimension **separately**, not in their product. That is the whole point of the
factorised two-way attention, and it makes frame 16 a stronger frame than planned:

- **Naive**: flatten all `n . m` cells into one sequence -> `(n . m)^2` attention pairs.
- **Factorised** (what TabPFN does): `O(n^2 + m^2)`.
- Worked, with n = 10,000 and m = 100: naive is `(10^6)^2 = 10^12`; factorised is
  `10^8 + 10^4 ~ 10^8`. **Roughly 10,000x cheaper.**

So the architecture is not only about invariance - **it is what makes the cost survivable at
all.** Frame 14 and frame 16 now tell one story instead of two, and the student derives the
10,000x themselves.

Also confirmed from the paper, for frames 13-15:

> "an architecture that assigns a separate representation to **each cell** in the table... a
> **two-way attention** mechanism, with each cell attending to the other features in its row
> (that is, its sample) and then attending to the same feature across its column (that is, all
> other samples). This design enables the architecture to be **invariant to the order of both
> samples and features**."

The invariance is stated by the authors as a design consequence, which is exactly what frame
15's predict-first reveal claims.

## Open questions

1. Frame 3 quotes ch04's closing box verbatim. Confirm that is the intended reading of that box
   and that bounding it is welcome.
2. Frame 1's predict-first needs a concrete pairing (which dataset, which baseline) - invent a
   plausible one, or use a real row from a published comparison?
