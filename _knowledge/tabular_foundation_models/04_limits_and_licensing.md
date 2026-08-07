# Limits, evaluation, and the licensing trap

The honest-limits material. Every chapter in this course has one; here the material is unusually
strong because critical papers exist in the same collection as the promotional ones.

## The licensing trap - check before teaching

**Web-checked 2026-08-07.** TabPFN-2.5 weights ship under `tabpfn-2.5-license-v1.1`. It is
permissive for research and limited internal evaluation - testing, evaluation and internal
benchmarking are explicitly allowed - but **the model, its derivatives, and its outputs may not
be used for any commercial or production purpose**. Commercial use goes through
`sales@priorlabs.ai`.

Three reasons this belongs on a slide rather than in a footnote:

1. **"Its outputs" is unusually broad.** A student who uses TabPFN to generate a feature, or to
   label data, has produced an output covered by the restriction.
2. **It makes the open alternatives a real decision**, not a footnote. TabICLv2 markets itself
   on being open. That is a live engineering tradeoff of exactly the kind this course asks
   students to reason about.
3. **Students will hit it.** Many of them work, or will work, in industry. "Beats gradient
   boosting" and "cannot be used in production without a commercial licence" belong on the same
   slide.

This is a factual statement about a licence, not a legal caveat about materials the instructor
already has the right to use - it is course content, because it changes which tool a student
should reach for.

## Where it actually fails

- **Realistic Evaluation of TabPFN v2 in Open Environments** (2025/05) is the designated
  counterweight paper. "Open environments" means the assumptions the benchmark quietly makes -
  clean features, no shift, closed label set - stop holding.
- **Size.** The headline claim is for *small* data. TabPFN-2.5 reaches 50,000 samples and 2,000
  features; beyond that you are in the six-paper `scaling` literature (sketching, distillation,
  chunking, TuneTables), which is a strong signal that the method has a shape rather than a
  frontier.
- **Distribution shift.** Needed its own model - `Drift-Resilient TabPFN` - which fixes the
  problem by putting shift *into the prior*. That is elegant, and it also means the base model
  does not handle it.
- **Prior-induced bias that does not vanish.** For causal estimands, Melnychuk et al. (2026)
  show the prior is not asymptotically overwritten by data. A student who has just learned
  consistency in `math/22_stat_estimators.qmd` can appreciate exactly how unusual that is.
- **Feature engineering is not obviously dead.** *TabPrep: Closing the Feature Engineering Gap
  in Tabular Benchmarks* (2026/06, web-checked) exists, which matters because ch06 is an entire
  feature-engineering chapter. Worth reading before claiming the chapter is obsolete.
- **The interpretability is shallower than the accuracy.** Steering fails to transfer across
  samples; causal structure is decodable but not shown to be used; only two datasets in the one
  causal-localisation study.

## The evaluation problem, one level up

A recurring theme worth its own frame: **tabular benchmarks are contested**. A model pre-trained
on synthetic data drawn from a prior over structural causal models will do well on benchmark
datasets that look like that prior. Whether the standard suites are representative of real
messy tables is exactly what *Realistic Evaluation* and *TabPrep* are arguing about.

The course already has the vocabulary for this from ch02 (train/test discipline, leakage) and
`math/27_stat_how_to_lie.qmd`. This is that lesson applied to a benchmark culture rather than to
a single experiment.

## What would change the picture

Recording this so a future revision knows what to re-check:

- If **TabPFN-3 / TabPFN-3-Plus** moves the row and feature ceiling by an order of magnitude,
  the "small data only" framing needs rewriting.
- If the licence changes, the open-vs-closed slide changes with it.
- If the Bayesian-vs-frequentist question gets settled empirically - which the instructor's
  research notes argue mechanistic interpretability could do - the chapter's centrepiece
  becomes a resolved story rather than a live argument, and should be retold that way.
