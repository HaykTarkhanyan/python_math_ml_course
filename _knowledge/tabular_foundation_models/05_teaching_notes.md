# Teaching notes: turning this into a chapter

Not an approved outline. This is the raw shape, for the interview step of the deck workflow
(`ml/SLIDE_STYLE.md`: interview -> outline -> approval -> build).

## The one-sentence pitch

**Chapters 1-8 taught you to fit, engineer and tune. Here is a model that does none of those and
wins anyway - and here is exactly where it stops winning.**

## Why this chapter is different from ch10-ch13

Diffusion, RL, VLM and audio all *added* capability. This one **contests** material already
delivered. That is rarer and more valuable: it forces a student to re-examine the first third of
the course rather than append to it. It should be written as an argument, not a survey.

Direct collisions worth naming on slides:

| Course material | What a tabular foundation model claims |
|---|---|
| `08_hyperparameter_tuning.tex` (whole deck) | No tuning. |
| ch04 trees and boosting | Beaten on small data by a model that never saw your data. |
| ch06 feature engineering | Largely absorbed - though `TabPrep` (2026) disputes this. |
| ch02 cross-validation discipline | Still needed, and now the *only* thing left to do. |
| `math/23` MLE vs MAP vs Bayes | The live dispute about what the model is doing. |
| ch08 time series | `From Tables to Time` extends it to forecasting. |
| ch05 interpretability | `Interpretable ML for TabPFN`, plus six mechanistic studies. |

## The practical: nanoTabPFN

**This is the strongest asset in the whole topic.** nanoTabPFN pre-trains in **one minute on a
single consumer GPU** (160,000x faster than TabPFN v2 pre-training), and is explicitly an
educational reimplementation. Code: `https://github.com/automl/nanoTabPFN` (AutoML group).

**Verified against the paper 2026-08-07, including the qualifiers, which matter:** in 60 seconds
it reaches ROC AUC **comparable to traditional ML baselines** on **subsampled small** datasets
from TabArena. It does **not** reach TabPFN v2 in that minute. So the lab promises "one minute
of pre-training gets you to random-forest territory", which is still a remarkable claim and is
the honest one.

**Two hard constraints on dataset choice:** nanoTabPFN supports **neither categorical features
nor missing values** - the user must preprocess first. Any dataset picked for this practical
must be numeric and complete, or ship with the preprocessing written.

That makes the central abstract idea into a lab exercise:

1. Pre-train nanoTabPFN on the default synthetic prior. Evaluate on a real small table.
2. **Change the prior** - narrow it, break a structural assumption, add noise.
3. Re-pre-train. One minute.
4. Watch the inductive bias move.

A student can *see* that the prior is the model's inductive bias, by intervening on it. No other
foundation model in this course can be pre-trained in a classroom, and the course already has
the Colab-via-WSL workflow for the GPU minute.

Fits the house pattern exactly: ch10 trained a diffusion model on Armenian letters, ch11 trained
a tic-tac-toe agent. This would be the third chapter project where students train the real thing.

Sanity check before promising it: confirm nanoTabPFN's actual runtime and dependencies on a T4,
and confirm its licence (it is a reimplementation, so it may not carry the TabPFN-2.5
restriction - **verify, do not assume**).

## Candidate arc

Two decks, mirroring ch12's seeing/drawing split:

**Deck A - "A model that does not learn your data"**
Cold open: a benchmark table where an untuned forward pass beats a tuned ensemble. The PFN idea.
The synthetic prior. The two-dimensional attention architecture (attention over features, then
over samples) and the architectural invariances. Why "no training" means the limit is context
length, not epochs.

**Deck B - "What is it actually doing, and when does it fail"**
The Bayesian-vs-frequentist dispute as the centrepiece. What the mechanistic studies found -
especially Biloš et al.'s result that different architectures reach the same accuracy through
*different* readouts with *different* failure modes. Then the honest limits: size, shift,
prior-induced bias, the evaluation-culture problem, and the licence.

Figures would come from real runs on real small tables, per house style - and unusually for this
course, the "measure it yourself" figure is cheap, because inference is a forward pass.

## Open questions for the instructor

1. **Where does it sit?** Placing it at ch14 (after audio) is chronological but weak: its
   argument is with chapters 1-8. A pointer from ch04 or ch06 forward to it may matter more than
   its position.
2. **Does it need the mechanistic-interpretability half at all**, or is that the instructor's
   research interest rather than a student need? Deck B is the more original material and the
   more demanding.
3. **How much theory?** The Nagler-vs-Bayes dispute can be one honest frame or a whole section.
4. **Is the nanoTabPFN practical a chapter project** (like ch10 and ch11) or a demo inside the
   lecture?
