# Deep Learning

Working notes for a deep-learning module, built around the **Introduction to Deep Learning (I2DL)**
reference slides from compstat-LMU (LMU Munich).

- **Reference slides live at:** `_reference/lecture_i2dl/` (LaTeX in `slides/`, compiled PDFs in `slides-pdf/`).
- **Site:** https://compstat-lmu.github.io/lecture_i2dl/
- These docs were last regenerated on **2026-06-22**.

## Files in this folder

| File | What it is | Use it when |
|---|---|---|
| [`i2dl_slides_summary.md`](i2dl_slides_summary.md) | Deck-level overview: the 53 decks across 13 topic folders, each with its chapter title, learning goals, key frames, plus coverage tables, the 14-block course arc, and reuse gotchas. | You want the big picture: what topics exist and where, and how the decks map to lectures. |
| [`i2dl_course_outline.md`](i2dl_course_outline.md) | Slide-by-slide outline: every frame in presentation order with short content gists, ordered along the course arc. (~1170 lines.) | You want the detail: what each individual slide covers. |
| [`additional_resources.md`](additional_resources.md) | Curated external resources (books, courses, blogs, papers, interactive tools), mapped to the same topics and web-search-verified. | You want to go deeper than the slides or point students to other material. |

## How to read these together

1. Start with **`i2dl_slides_summary.md`** to see the topic landscape and pick a section.
2. Drill into **`i2dl_course_outline.md`** for the frame-by-frame detail of that section.
3. Pull from **`additional_resources.md`** for textbook chapters, videos, and papers on the same topic.

## How they were generated

- The summary and outline were extracted by parsing the deck `.tex` sources (`\lecturechapter`, `\learninggoals`, frame titles). Math/formulas are stripped from the outline gists for readability, so the deck PDFs remain the source of truth for exact equations.
- The resource list was compiled from web searches and the key links were verified.
