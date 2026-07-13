# slides_tex — LaTeX sources for the s26 Deep Learning decks

Purpose: hold the **`.tex` sources** of the Moodle slide decks so their content
(frame titles, bullet text, math, figure captions) can be read directly, without
parsing the compiled PDFs.

## Provenance

Copied from the I2DL lecture repo (cloned into `../../_reference/`):

- Slides: `slds-lmu/lecture_i2dl` @ `acc641a` (2026-06-03)
- Math macros: `slds-lmu/latex-math` (cloned into the repo; not vendored upstream)

## What's here

```
slides_tex/
├── slides/<topic>/*.tex     # deck sources for the 10 topics the s26 course uses
├── style/                   # shared preamble.tex, common.tex, lmu-lecture.sty, color/
└── latex-math/*.tex         # math-macro definitions the decks \input
```

The two-levels-deep layout (`slides/<topic>/deck.tex` with `style/` and `latex-math/`
as siblings of `slides/`) is deliberate: each deck does
`\input{../../style/preamble}` and `\input{../../latex-math/basic-math}` etc., so those
paths resolve unchanged. No `.tex` was edited.

### Topic folders (10) → Moodle weeks

| topic | Moodle content |
|---|---|
| `intro` | Introduction, History |
| `mlps` | Single Neuron, XOR, Single Hidden, Matrix Notation, Multilayer, Multiclass, Univ. Approx. |
| `opt1` | Basic Training, Comp. Graphs, Backprop 1 & 2, Hardware/Software |
| `regu` | Basic Regularization, Early Stopping, Dropout & Augmentation |
| `opt2` | Challenges, Advanced Optim, Initialization, Activations |
| `cnn1` | Intro to CNNs, Conv2d, Properties, Components, Conv-vs-CrossCorr (math), Application |
| `cnn2` | Convolution Types, Dilated & Transposed, Separable & Flattened |
| `cnn3` | Modern CNN I & II |
| `rnn` | Intro to RNN, Backprop-Through-Time, Modern RNN, Applications, **Attention/Transformers** |
| `ae` | Unsupervised Learning, Manifold Learning, AutoEncoders (+ Regularized, Specific), VAE |

Note: `attention` lives in `rnn/`, and the RNN intro is `rnn/slides-introduction.tex`
(there is a different `slides-introduction.tex` in the repo's `genmod/` folder — not
copied, not part of this course).

## Coverage

All **43** decks downloaded from Moodle have their `.tex` source here (verified by exact
filename). The folders also contain **3** extra sibling decks not used on Moodle
(e.g. `slides-mlps-mlps-as-predictor`) — 46 `.tex` total.

## What's intentionally NOT here

- **Figures** (`figure/`, `figure_man/` PNG/JPG trees) and **compiled PDFs** — omitted to
  keep this git-committable folder small and text-focused. They remain in
  `../../_reference/lecture_i2dl/slides/<topic>/` if a full local compile is ever needed.
- Because figures are absent, these sources **read** correctly but will **not compile**
  as-is (missing `\includegraphics` targets). Compile from `_reference/` instead.

To refresh: re-pull `_reference/lecture_i2dl` and re-run the copy (10 topic folders'
`*.tex` + `style/` + `latex-math/*.tex`).
