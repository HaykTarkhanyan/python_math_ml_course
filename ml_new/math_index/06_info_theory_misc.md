# Information Theory + Misc — Math Reference

## Information Theory

### Homework / Quarto modules (`math/`)

- [28_info_theory_entropy_kl.qmd](../../math/28_info_theory_entropy_kl.qmd) — **Entropy, cross-entropy, KL divergence** — Shannon entropy, joint and conditional entropy, cross-entropy, KL divergence (definition, asymmetry, properties), connection to log-likelihood.
- [29_info_theory_mle_maxent_mi.qmd](../../math/29_info_theory_mle_maxent_mi.qmd) — **MLE, MaxEnt, mutual information** — MLE ↔ KL minimization, maximum entropy principle, mutual information, information gain, data processing inequality (sketched).

### Lecture decks

No dedicated info-theory slide deck exists. The qmds are the canonical lecture material.

### Homework write-ups

- `hw_18_info_theory.pdf` — modules 28 + 29

## Misc — Preliminaries and Curse of Dimensionality

### Homework / Quarto modules (`math/`)

- [00_sets_comb_funcs.qmd](../../math/00_sets_comb_funcs.qmd) — **Preliminaries** — Sets, set operations, basic combinatorics (permutations, combinations), functions, function composition, injective/surjective/bijective.
- [30_curse_of_dimensionality.qmd](../../math/30_curse_of_dimensionality.qmd) — **Curse of dimensionality** — Volume concentration in high-d, distance concentration, sparsity, nearest-neighbor breakdown, why dimensionality reduction matters.

### Lecture PDFs (`math/Lectures/`)

- [Intro.pdf](../../math/Lectures/Intro.pdf) — Course intro deck (whole math block).
- [L00_Sets__Combinatorics__Functions.pdf](../../math/Lectures/L00_Sets__Combinatorics__Functions.pdf) — Sets, comb, functions (module 00).
- [curse_of_dimensionality/](../../math/Lectures/curse_of_dimensionality/) — Supporting material for module 30 (folder).

### Homework write-ups

- `hw_00_sets_comb_funcs.pdf` — module 00

## Use when teaching (ML hooks)

### Info theory

| ML concept | Pull from |
|---|---|
| Cross-entropy loss (any classifier) | module 28 |
| Decision tree splits (entropy / info gain) | modules 28, 29 |
| KL divergence (VAE, distillation, label smoothing, RLHF, DPO) | module 28 |
| Mutual information (feature selection, contrastive learning, info bottleneck) | module 29 |
| MaxEnt / softmax derivation | module 29 |
| Wasserstein vs KL trade-off (GANs, diffusion) | module 28 baseline + future material |

### Preliminaries

| ML concept | Pull from |
|---|---|
| Counting hypothesis space size (combinatorics) | module 00 |
| Set notation for train / test / validation | module 00 |
| Function notation `f: X -> Y` used throughout | module 00 |
| Bijection ↔ invertible function (normalizing flows hint) | module 00 |

### Curse of dimensionality

| ML concept | Pull from |
|---|---|
| Why kNN fails in high-d | module 30 |
| Why PCA / dimensionality reduction is needed | module 30 + module 03 (eigen, SVD) |
| Why high-d data needs more samples (sample complexity) | module 30 |
| Embedding intuition (low intrinsic dim of natural data) | module 30 (counterpoint discussion) |

## Notes

- Info theory has **no slide deck** — qmds are the lecture material. When the ML course gets to entropy / KL / MI (decision trees in L08, cross-entropy in classification, KL in VAE chapter), point students at the qmds rather than expecting a polished deck.
- Module 30 (curse of dimensionality) is treated as a *capstone* of the math block. It glues linear algebra + probability + intuition together.
- The merged "Short Math for ML" PDF (`math/Lectures/Կարճ մաթեմ մեքենայական ուսուցման համար.pdf`) is a single bundle of selected slides spanning all sections — handy as a one-link reference for students who want a single download.
