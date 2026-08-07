# Tabular foundation models (TabPFN and relatives)

Reference material for a possible `ml/` chapter. Gathered 2026-08-07.

## In three sentences

A **tabular foundation model** is a transformer pre-trained once on millions of *synthetic*
datasets, which then predicts on a new real dataset **without any training at all**: you put the
entire training table into its context window and read the prediction out of a single forward
pass. On small tabular data it beats a gradient-boosted tree ensemble that has been tuned for
hours. That is a direct challenge to the workflow this course teaches in chapters 1 through 8.

## Why this is worth a chapter here

Chapters 1-8 teach the classical tabular pipeline: fit a model, engineer features, tune
hyperparameters, cross-validate. `ml/02_main_concepts/08_hyperparameter_tuning.tex` exists as a
whole deck. A model whose headline claim is *no hyperparameter tuning, no gradient steps, one
forward pass* is not a new algorithm to bolt on at the end - it is a live challenger to the
premise of the first third of the course.

That makes it pedagogically stronger than another architecture chapter. It also lands on
prerequisites the course already has: transformers (ch9), attention, in-context learning
(`llm_training`), Bayesian inference and MAP/MLE (`math/23`), and the bias-variance framing
(ch2).

## Provenance

Two sources, kept distinct on purpose:

1. **`~/OneDrive/Desktop/mech_interp/`** - the instructor's own research repo. Contains
   **39 title-verified TabPFN papers** (155 MB) with `manifest.json`, a systematic mechanistic
   interpretability literature review (53/53 references verified), and
   `research_ideas/tabpfn_mech_interp.md`, which is the single best summary of what the
   interpretability literature has and has not established. Most of the technical content in
   these notes is distilled from there.
   **The PDFs were deliberately NOT copied** into this repo - 155 MB against a `.git` that is
   already 1.9 GB, and `papers_manifest.json` here carries every arXiv URL, so
   `scripts/download_tabpfn_papers.py` in that repo rebuilds the collection.
2. **Web search on 2026-08-07** - used to check that the local material is still current. It
   was not, in two places: TabPFN-2.5's exact limits and license, and the existence of
   TabPFN-3-Plus. Anything web-checked is marked as such in `sources.md`.

## Files

| File | Contents |
|---|---|
| `01_the_idea.md` | Prior-data fitted networks, the synthetic prior, and the two-dimensional attention architecture |
| `02_the_model_line.md` | TabPFN 2022 -> Nature 2025 -> 2.5 -> 3, plus TabICL, TabDPT and other competitors |
| `03_what_it_actually_learned.md` | The analysis and interpretability findings, and the live Bayesian-vs-frequentist dispute |
| `04_limits_and_licensing.md` | Where it fails, how it is evaluated, and the licensing trap |
| `05_teaching_notes.md` | How this becomes a chapter, including the nanoTabPFN practical |
| `sources.md` | Every source, marked local / web-verified / unverified |
| `papers_manifest.json` | The 39-paper manifest copied from `mech_interp`, with URLs |
