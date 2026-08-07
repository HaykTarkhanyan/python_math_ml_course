# The model line, and the competitors

Dates from `papers_manifest.json` (title-verified 2026-08-06) unless marked web-checked.

## The core line

| When | What | Why it matters |
|---|---|---|
| 2021/12 | **Transformers Can Do Bayesian Inference** | The founding idea. PFNs before they were tabular. |
| 2022/07 | **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second** | v1. Classification only, tiny datasets. Proof of concept. |
| 2025 | **Accurate predictions on small data with a tabular foundation model** (Hollmann et al., **Nature**) | v2. The one that made the field pay attention: a Nature paper claiming a pretrained transformer beats tuned tree ensembles on small tabular data. |
| 2025/11 | **nanoTabPFN: A Lightweight and Educational Reimplementation** | Pre-trains in **one minute on a single GPU**, reported as ~160,000x faster than TabPFN v2 pre-training. Explicitly educational. |
| 2025/11 | **TabPFN-2.5** | Up to **50,000 samples and 2,000 features** (web-checked). |
| 2026/05 | **TabPFN-3: Technical Report** | Current line at the time these notes were written. |
| 2026/06 | **TabPFN-3-Plus** | Web-checked only, no paper in the local collection. |

**nanoTabPFN is the single most useful entry here for teaching.** A one-minute pre-training run
means students can *change the prior and watch the inductive bias change* - the whole point of
the chapter made into an experiment, on a rented Colab GPU, in the length of a coffee break. No
other foundation model in this course can be pre-trained in a classroom.

## Competitors and alternatives

Not a Prior Labs monopoly. From the manifest plus web checks:

- **TabICL** (2025/02) - tabular foundation model for in-context learning on *large* data,
  attacking TabPFN's row limit. **TabICLv2** (2026/02) advertises itself as "better, faster,
  scalable, and **open**" - the licensing contrast is the point, see `04_limits_and_licensing.md`.
- **TabDPT** (2024/10) - scales tabular foundation models on **real** data rather than purely
  synthetic priors. The natural counterexample to "the prior is everything".
- **Mitra** - appears as a third architecture in the Biloš et al. mechanistic comparison.
- **LimiX-2M** (2026/06, web-checked) - addresses low-rank collapse and attention bottlenecks.

For a lecture, three names is enough: TabPFN (synthetic prior, closed weights), TabICL (large
data, open), TabDPT (real-data pre-training). They differ on the two axes that actually matter -
**where the pre-training data comes from** and **how many rows fit**.

## The extension literature, as evidence of generality

The `extension` category is worth one slide as a list, because it shows the idea is not
confined to small-n classification:

- **Time series**: *From Tables to Time* extends TabPFN-v2 to forecasting. Directly relevant to
  ch08, which currently teaches classical and ML time series with no foundation-model option.
- **Distribution shift**: *Drift-Resilient TabPFN* puts temporal shift **into the prior**.
- **Generation**: *TabPFGen* uses it to synthesise tabular data.
- **Wide data**: *TabPFN-Wide* handles extreme feature counts via continued pre-training.
- **Interpretability**: *Interpretable Machine Learning for TabPFN* (2024/03) - connects
  straight to ch05, which already teaches SHAP, LIME and PFI.
- **Multitask**: *TabPFN-MT* (2026/05).
- **Text features**: *Towards Pretraining Text Encoders for TabPFN* (2026/06).

## The scaling problem, stated honestly

The headline result is for **small** data, and the whole `scaling` category exists because that
is the binding limit:

- Sketching and feature selection (2023/11)
- In-context data distillation (2024/02)
- TuneTables: context optimization (2024/02)
- TabPFN Unleashed (2025/02)
- Chunked TabPFN, for long context (2025/08)
- TabPFN-Wide, for extreme feature counts (2025/10)
- *Prior-Fitted Networks Scale to Larger Datasets When Treated as Weak Learners* (2025/03) -
  filed under theory, and a nice bridge to ch04's boosting: use PFNs as the weak learner.

Six papers to make one model handle more rows is itself the lesson. If a student takes away
"just use TabPFN", they have missed that the method has a hard shape: brilliant inside its
regime, awkward outside it.
