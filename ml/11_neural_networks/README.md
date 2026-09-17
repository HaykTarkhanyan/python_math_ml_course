# Chapter 11 - Neural Networks

## Delivered lectures and their `_notes` PDFs (2026-09-16)

Taught so far: `40_intro_history` (lecture 40), `41_1_neuron_to_network` and
`41_2_multilayer_nets` (lecture 41, two parts). Renamed the same day from `dl_*`; the undelivered
decks carry an `xx_` prefix until they get a lecture number.

Each delivered deck has **two PDFs**:

- `NN_topic.pdf` - the **clean** build, compiled from the `.tex`. Recompiling writes only this file.
- `NN_topic_notes.pdf` - the same slides **hand-annotated live in the lecture** and exported
  afterwards. **Not reproducible: never overwrite, regenerate, or delete a `_notes.pdf`.**

The chapter page links both (clean slides, notes PDF, video).

This folder holds **two parallel treatments** of the same neural-networks material:

1. **House-style decks** - `L14_neural_networks` and `L15_training_neural_networks`. Fully
   re-authored in our own style, every essential figure Python-generated. These are the
   canonical, self-contained decks.
2. **Embed-based deck set** - the `dl_*` decks. These **embed the LMU "Introduction to Deep
   Learning" slides verbatim** (via `pdfpages`) and add our own slides ("sprinkles") on top.
   This README documents that set, because the workflow is not obvious from the files alone.

The two coexist on purpose. `L14`/`L15` are the tight house version; the `dl_*` set is a
comprehensive, reference-grade walk through the full LMU course with our commentary woven in.

---

## The embed-based deck set (7 decks, LMU weeks 16-20)

All 4:3 (`aspectratio=43`) to match the embedded LMU pages. Built 2026-07-18.

| # | Deck (`.tex`/`.pdf`) | LMU week / chunks | Pages | Our added slides |
|---|---|---|---|---|
| 1 | `40_intro_history` | wk16: intro + brief-history | 54 | modern-applications gallery, state-of-field, industry, recent-timeline, adoption chart, key people (godfathers / figures / educators), case studies (Bun rewrite) |
| 2 | `41_1_neuron_to_network` | wk16: single-neuron, xor, single-hidden-layer | 50 | two-moons, activation gallery, interactive tools, recap/bridge |
| 3 | `41_2_multilayer_nets` | wk17: matrix-notation, multilayer-FNNs, multiclass, univ-approx | 74 | matrix-shapes, folding, features-stack, softmax bar chart, sum-of-bumps, Sherlock-Holmes UAT joke |
| 4 | `42_training_backprop` | wk18: basic-training, comp-graphs, backprop 1&2, hardware | 78 | arithmetic erratum, `f-y` / `p-y` residual + push-and-pull, vanishing gradients, `loss.backward()`, leaky abstraction, hardware 2026, renting a GPU, experiment tracking |
| 5 | `43_nn_regularization` | wk19: basic-reg, early-stopping, ensemble-dropout-augmentation | 53 | Ridge callback, training-curves, predict-first random labels, memorizing noise, double descent, scaling laws, dropout scaling, "which knob when" table |
| 6 | `xx_optimization` | wk20: challenges, advanced-optim | 94 | "why SGD works anyway", optimizer cheat-sheet, warmup + decay, AdamW in three frames, LayerNorm, bridge to part 2 |
| 7 | `xx_optimization_init_activations` | wk20: initialization, activations | 44 | He derivation + measurement, loss at init (predict-first), activations as equations, "which activation", GELU, calibration, sanity checks, full-chapter recap |

Decks 6 and 7 are the two halves of LMU week 20 (its source is 116 pages, so the combined deck
was split). Deck 7 carries the chapter-closing recap.

---

## How the embed works

Each deck is:

```latex
\documentclass[aspectratio=43]{beamer}
\input{../preamble}
\usepackage{pdfpages}
\begin{document}
  ... our title frame ...
  \includepdf[pages=-]{lmu_src/01_introduction.pdf}   % LMU chunk, verbatim
  ... our sprinkle frame(s) ...
  \includepdf[pages=-]{lmu_src/02_brief-history.pdf}
  ...
\end{document}
```

**Sprinkle mid-deck, not only at the ends.** To drop a frame after LMU page N inside a chunk,
split the include:

```latex
\includepdf[pages=1-6]{lmu_src/04_univ-approx-theorem.pdf}
\begin{frame}{...our slide...}...\end{frame}     % lands right after LMU page 6
\includepdf[pages=7-21]{lmu_src/04_univ-approx-theorem.pdf}
```

(Used in `41_2_multilayer_nets` to put the sum-of-bumps intuition between LMU's theorem statement
and its examples.)

---

## Building / rebuilding a deck

From inside this folder (so `\input{../preamble}` and the relative `lmu_src/` path resolve):

```bash
pdflatex -interaction=nonstopmode -halt-on-error 40_intro_history.tex   # run twice
pdflatex -interaction=nonstopmode -halt-on-error 40_intro_history.tex
```

Then clean aux files (never deletes `.pdf`, never touches any `_notes.pdf`):

```bash
./ma/Scripts/python.exe clean_latex.py ml/11_neural_networks/40_intro_history.tex
```

Figures are regenerated with the repo venv, e.g.:

```bash
./ma/Scripts/python.exe ml/11_neural_networks/py_src/uat_bumps.py
```

---

## Assets

- **`lmu_src/`** - unprotected copies of the LMU chunk PDFs. The originals under
  `ml/deep_learning/moodle_s26_course/slides/weekNN_*/` are owner-password-locked, which breaks
  `\includepdf`; these copies have the password stripped so the decks build standalone. Regenerate
  a copy with PyMuPDF:
  ```python
  import fitz
  d = fitz.open(src_pdf)
  if d.needs_pass: d.authenticate("")
  d.save(dst_pdf, garbage=4, deflate=True)
  ```
- **`fig/`** - Python-generated figures. Embed-set additions: `recent_timeline`, `adoption_speed`,
  `softmax_bars`, `uat_bumps`. Reused from the house decks: `two_moons_*`, `activations`,
  `activation_grads`, `folding_*`, `lr_curves`, `init_variance`, `training_curves_4cases`.
- **`img/`** - raster images fetched from Wikimedia (researcher/educator portraits, robot,
  protein, scroll, balloon, AI art, logos).
- **`py_src/`** - the figure-generation scripts (seed 509, Armenian-flag palette, `ax.bar_label`,
  logging to `logs/`).

---

## Two kinds of sprinkle

1. **Ours** - modern applications, key people, practical-synthesis cheat-sheets, illustrations.
2. **Improving LMU's own pages** where they were text-only or abstract: matrix-shapes for their
   notation deck, sum-of-bumps for their theorem, `init_variance` for their text-only init slide,
   the vanishing-gradient consequence for their backprop recursion.

All modern-AI factual claims were web-verified (July 2026) and are listed in each deck's Sources
frame / provenance comment.

---

## Conventions and open items

- **Naming:** these use `dl_<topic>` names, NOT the playlist `NN_topic` convention (see
  `../../CONVENTIONS.md`). Rename once each gets a YouTube slot.
- **Not registered** in `_quarto.yml` / `11_neural_networks.qmd` yet - they will not appear on the
  site until slotted in.
- **Reproducibility caveat:** unlike the house decks, an embed deck is not reproducible from
  `.tex` alone - it depends on `lmu_src/` (and `fig/`, `img/`) being present. All of those are
  tracked, so a fresh clone builds.
- **Delivery:** these are comprehensive/reference-grade; subset per lecture as needed.

## Next chapters

The CNN (`ch6`) and RNN (`ch7`) LMU material (weeks 23+) can be built the same way: decrypt the
week's chunks into `lmu_src/` (or a sibling `ch6_cnn/lmu_src/`), embed with `pdfpages`, and
sprinkle. This folder's decks are the working template.
