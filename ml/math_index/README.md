# Math Index — Reference Map for Older Materials

Generated: 2026-05-29
Purpose: When the ML course needs to reference, link back to, or build on math material that already exists in `math/`, look here first. This index points to the source files; the section files have one-line summaries per module.

## How to use this

- If a future ML lecture needs to assume / cite / link back to math content, read the relevant section file in this folder, find the matching qmd or tex, then link to it from the ML slides/qmd.
- "When to reference" headers in each section file suggest which ML lectures naturally pair with that math content.
- Don't duplicate math content into ML decks. Link back instead.

## Folder map

| Section | File | Coverage | qmd modules | Slide decks |
|---|---|---|---|---|
| Linear Algebra | [01_linear_algebra.md](01_linear_algebra.md) | Vectors, matrices, SLE, eigenstuff, SVD | 01-03 | `Lectures/L01`-`L05` |
| Calculus | [02_calculus.md](02_calculus.md) | Limits, derivatives, extrema, Taylor, integrals, multivar | 04-07 | `Lectures/L06`-`L08` |
| Optimization | [03_optimization.md](03_optimization.md) | Univariate, GD, momentum/Adam, second order, derivative-free, evolutionary, Bayesian, multicriteria | 08-15 | `Lectures/optim/01`-`06` |
| Probability | [04_probability.md](04_probability.md) | Basics, RVs, expectation, covariance, distributions, convergence, LLN/CLT | 16-20 | `Lectures/L09`-`L14` |
| Statistics | [05_statistics.md](05_statistics.md) | Foundations, estimators, MLE/MAP, CIs, hypothesis testing, classical tests, common lies | 21-27 | `Lectures/stat/01`-`12` |
| Information Theory + Misc | [06_info_theory_misc.md](06_info_theory_misc.md) | Entropy, KL, MI, MaxEnt + preliminaries + curse of dimensionality | 00, 28-30 | (none separate) |

## Where each artifact type lives

- **Homework qmds (bilingual Armenian + English):** `math/00_*.qmd` through `math/30_*.qmd`. Each has Material / Lecture / Practical / Homework sections, problems with difficulty stars (🧀 = 1, 🧀🧀 = 2, 🧀🧀🧀 = 3), and bonus problems (🎁).
- **Homework solutions (when written):** Quarto profile `solution` — same qmd, gated content. Some have separate `_solutions.md` files (e.g., `22_stat_estimators_solutions.md`).
- **Homework write-ups (handwritten on tablet):** `math/Homeworks/hw_*.pdf` (paired with `.xopp` source).
- **Lecture PDFs (early modules, 00-14):** `math/Lectures/L00`-`L14` are single PDFs without source.
- **Lecture decks with TeX source (optim + stat):** `math/Lectures/optim/*.tex` and `math/Lectures/stat/*.tex`. These compile via `pdflatex`. Beamer dove theme. Stats decks also have `*_notes.pdf` (slides with speaker notes baked in).
- **Stat lecture outline (canonical reference):** `math/Lectures/stat/00_lecture_outline.md` — topic-by-topic, includes homework breakdowns and cross-references back to modules.
- **Older/legacy material:** `math/unstructured/` — earlier qmd versions, practicals, solutions. Treat as deprecated; pull from main 00-30 series.
- **Armenian notes (textbook):** `math/armenian_notes/` — referenced by some qmds (e.g., `07_calc_multivar.qmd` cites "Section 7.7" of the notes).
- **Lectures plan + summary:** `math/Lectures/00_lecture_plan.md`, `math/Lectures/apr_slack_messages.md`.
- **Single merged math PDF:** `math/Lectures/Կարճ մաթեմ մեքենայական ուսուցման համար.pdf` (Armenian: "Short Math for ML") — produced from `merged_slides.pdf`.

## What's complete vs partial

Per `course-completion-status.md` memory:
- Modules 00-20 (linear algebra, calculus, optim, probability): qmd + delivery PDFs exist.
- Modules 21-27 (statistics): qmds exist; stat lecture decks (`Lectures/stat/01`-`10`) delivered; `11_stat.tex` and `12_stat.tex` exist but `12_stat`+ are deferred (`ml/deferred_lectures/`).
- Modules 28-29 (info theory): qmds exist; no separate slide deck (lectures pull from qmds directly).
- Module 30 (curse of dimensionality): qmd + handout-style content; `Lectures/curse_of_dimensionality/` has supporting material.
- Optimization slide decks (`Lectures/optim/01`-`06`): 6 decks for modules 08-13 approximately. Bayesian (14) + multicriteria (15) may not yet have dedicated source decks.

## Cross-reference to ML course

The 30-week ML syllabus (`ml/syllabus.csv`) implicitly assumes much of this math. Direct dependencies:

| ML lecture / topic | Math prerequisite |
|---|---|
| L01 linear regression | Modules 01-03 (vectors, matrices), 04-05 (derivatives, extrema) |
| L02 gradient descent | Modules 07 (multivar calc), 09 (GD intro) |
| L03 regularization | Modules 01 (norms), 05 (convexity) |
| L04 evaluation metrics | Module 17 (expectation, variance) |
| L05 cross-validation | Modules 21-22 (estimator properties) |
| L06 logistic regression | Modules 19 (Bernoulli), 23 (MLE) |
| Bias-variance | Modules 17, 22, stat L3 |
| Naive Bayes | Modules 16, 19, 23 |
| PCA | Module 03 (eigen, SVD) |
| K-means / EM | Stat L5 (MLE), Module 18 (mixtures) |
| Decision trees | Module 28 (entropy, info gain) |
| Neural networks | Modules 07 (gradients), 10 (Adam), 11 (Hessian) |
| Transformers | Module 28 (cross-entropy), Module 03 (attention as matrix ops) |
| RLHF / DPO | Stat L6 (MAP), Module 14 (Bayesian opt is closest) |

## Conventions used in these files

- Each module entry: `[NN file_name.qmd](path) — Title — one-line topic list`.
- For slide decks: `[deck_name.tex](path) — frame count + one-line summary` where frame count is available (stat outline supplies it; optim approximate).
- "Use when teaching" hints are at the bottom of each section file.
