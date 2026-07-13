# slides_tex — LaTeX sources for the DL4NLP lecture decks

Purpose: hold the `.tex` sources of the DL4NLP lecture decks so their content can be read
directly, without parsing PDFs.

## Provenance

Copied from the DL4NLP lecture repo (cloned at repo root `_reference/lecture_dl4nlp/`):

- Slides: `slds-lmu/lecture_dl4nlp` @ `9527645` (2026-05-13)
- Math macros: `slds-lmu/latex-math` (deck submodule, pinned `81d03ba`). The submodule was
  not initialized in the shallow clone, so `basic-math.tex` / `basic-ml.tex` here were taken
  from the identical `slds-lmu/latex-math` clone used for I2DL. Same files.

## Scope — which chapters (from the Moodle "Lecture Material" weekly URLs)

The Moodle course maps weeks 1–8 to the slds-lmu web chapters; **weeks 9–13 are Prof.
Schütze's own materials on `cis.lmu.de/~hs/teach/26s/dl4nlp/` and are NOT in this repo.**
So `slides_tex` covers chapters 01–08 only:

| Week | Moodle chapter URL | Repo folder here |
|---|---|---|
| 1 | `chapters/01_introduction/` | `chapter01-basics` |
| 2 | `chapters/02_dl_basics/` | `chapter02-deeplearningbasics` |
| 3 | `chapters/03_transformer/` | `chapter03-transformer` |
| 4 | `chapters/04_bert/` | `chapter04-bert` |
| 5 | `chapters/05_bert_based/` | `chapter05-bert-based` |
| 6 | `chapters/06_post_bert_t5/` | `chapter06-post-bert-t5` |
| 7 | `chapters/07_gpt/` | `chapter07-gpt` |
| 8 | `chapters/08_decoding/` | `chapter08-decoding` |
| 9–13 | Schütze's page (external) | — not in repo |

The upstream repo also has `chapter09-llm`, `10-rlhf`, `11-training-llms`,
`12-multilinguality` (and setup/deprecated folders). They exist in
`../../../../_reference/lecture_dl4nlp/slides/` if ever needed, but are left out here because
this course's weeks 9–13 use Schütze's external slides instead.

## Layout

```
slides_tex/
├── slides/chapterNN-*/*.tex   # 44 decks, chapters 01-08
├── style/                     # preamble_new.tex, preamble.tex, common*, lmu-lecture*.sty
└── latex-math/                # basic-math.tex, basic-ml.tex, ...
```

Two-levels-deep layout is deliberate: decks do `\input{../../style/preamble_new}` and
`\input{../../latex-math/basic-math.tex}`, which resolve unchanged. Decks use the custom
`\begin{vbframe}` frame environment (same as I2DL). No `.tex` was edited.

## Not included here (but see sibling folder)

- **Compiled slide PDFs** for these same chapters 01–08 are in **`../slides_weeks1-8/`** (36 PDFs).
  This `slides_tex/` folder is the *text-readable* form; `slides_weeks1-8/` is the *viewable* form.
- **Figures** (`figure/` PNG/JPG trees) are omitted to keep this folder small and text-focused;
  they remain in `_reference/lecture_dl4nlp/slides/<chapter>/`, so these `.tex` read fine but
  won't compile standalone.
