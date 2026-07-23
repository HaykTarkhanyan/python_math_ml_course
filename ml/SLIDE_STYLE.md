# Slide style guide — `ml` Beamer decks

Single source of truth for how the instructor wants slide decks built in this repo.
The `slide-style` skill points here. **Read this before creating or editing a deck.**

Captured from building the classification chapter (L11 logreg, L12 metrics, L13
calibration) and the regression decks (05 regularization, 06 hyperparameter tuning,
07 regression metrics). Update this file when a preference changes.

---

## Workflow

**New deck** (the instructor's preferred order):

1. **Interview first.** Ask about content for *this* deck — scope, what to include vs cut, which examples, how deep. Do not start from assumptions.
2. **Outline next.** Draft the section + frame list (an outline `.md`, e.g. `L03_OUTLINE.md`) and get approval.
3. **Then build.** Write the `.tex`, compile, clean, open the PDF for review.

**Editing an existing deck:** make the change, recompile, verify — no outline step needed.

**Compile loop:**

- `pdflatex -interaction=nonstopmode -halt-on-error FILE.tex` run **twice** (so `\tableofcontents`/Outline and cross-refs resolve).
- Then clean: `./ma/Scripts/python.exe clean_latex.py` (or `clean_latex.py PATH`).
- Beamer **clips overflow silently** (no overfull-vbox warning). Verify by rendering pages to PNG and eyeballing, or run the `/beamer-overflow-check` skill.
- Open a finished PDF for review with PowerShell `Start-Process "msedge.exe" -ArgumentList '"...pdf"'`, one at a time (~0.5s apart).

---

## Boilerplate

- `\documentclass[aspectratio=169]{beamer}` then `\input{../preamble}` (decks sit one level under `ml/`).
- End each `.tex` with a `% Provenance:` comment block — source material, which figures, key pedagogical decisions, and the forward pointer to the next deck. (Matches every existing deck.)

---

## Palette & callouts

Defined in `ml/preamble.tex`:

| Color | RGB | Use for |
|---|---|---|
| `popblue` | 52,101,164 | theory / definitions |
| `armred` / `sampred` | 200,30,40 / 204,0,0 | data, warnings |
| `paramgreen` | 0,140,70 | parameters, takeaways |
| `armorange` / `orange1` | 230,160,30 | watch-outs, highlights |
| `violet1` | 120,50,160 | accents |

Callout boxes via `fcolorbox{COLOR}{COLOR!8}{\parbox...}`:

- `armblue!8` — key point / definition
- `armred!8` — warning / trap / "this breaks"
- `paramgreen!8` — takeaway / "Next:" forward pointer
- `armorange!12` — watch-out / footgun

---

## Deck skeleton (defaults — include unless told otherwise)

- **Cold-open hook** before the Outline: bridge from the previous lecture or pose the motivating problem.
- **Outline** frame (`\tableofcontents`).
- One `\section` per major topic, each preceded by a **`[plain]` transition slide**: `popblue` bold title + one short motivation line.
- **Recap** frame at the end + a `paramgreen` **"Next:"** box pointing to the next lecture.
- **No HW frame in the deck.** Homework lives on the chapter `.qmd` page, not in the slides.
- **Long decks are fine — no fixed length, and don't split or trim a deck just because it's long.** One idea per frame; split only a single *dense* frame into two. A single deck may cover several subtopics in one file (e.g. binary + multiclass logistic regression).
- **Full-bleed / full-screen frames — an approved default, not an exception (added 2026-07-21).** When a frame's payload is one strong visual — a big architecture diagram, a photo, an animation still, a borrowed video still, a rich generated figure — prefer a `[plain]` **full-screen** frame that lets the image fill the slide over the usual title+body+footer structure. Use it **sometimes**, by judgment: go full-bleed when the picture should do the talking (visualization-rich moments); keep the standard structure when the frame needs surrounding prose. Good practice: set up a full-bleed still with a short *preceding* normal frame, and keep only a small source/caption line (and optionally suppress the footer for the cleanest look).
  - **Mechanism.** 16:9 deck: `\usebackgroundtemplate{\includegraphics[width=\paperwidth,height=\paperheight]{fig/...}}` wrapped around a `[plain]` frame (see the welchlabs full-bleed pattern in `ml/ch5_neural_networks`). 4:3 deck (e.g. the `dl_*` LMU-embed set): letterbox the image on black so it is not distorted — see the `\bbslide` macro in `ml/ch6_cnn/dl_cnn_conv_math.tex`. Put the attribution/caption in a small `tikz` overlay node.

---

## Content conventions

- **Language:** slide body and labels in **English**. Armenian only where natural (local examples, the `.qmd` page section headers).
- **Tone:** mostly straight delivery, **occasional light touch**. A running example is welcome; don't force humor or Armenian cultural references — use sparingly.
- **Running example:** thread one concrete scenario through a chapter where it helps (e.g. the cheese factory across classification).
- **Predict-first frames:** add them where a result is **counter-intuitive** (instructor's judgment, no need to ask) — e.g. "97% accuracy, 0% recall". Not on every frame.
- **Worked-numbers frame:** include one "by hand" example when the topic has computable mechanics (e.g. logistic `z -> p -> odds`).
- **Math:** show **full step-by-step derivations**, usually in a boxed/structured frame (e.g. the sigmoid derivation, MLE -> log-loss).
- **Code:** **minimal** — at most one canonical "in `sklearn`" snippet per topic. Teach the concept, not the API.
- **Misconception pre-empts** where a confusion is common (e.g. "0.5 is the default threshold, not a law"; "score vs calibrated probability").
- **Cite the source (convention).** For each named method/algorithm, cite the originating paper as **author(s) + year** (e.g. "AdaBoost (Freund & Schapire, 1997)", "gradient boosting (Friedman, 2001)"). For a **library/model**, give it a transition **card** (`\modeltransition` in `20_advanced_boosting.tex`) showing **year · company/authors · GitHub repo** (e.g. "LightGBM — 2017 · Ke et al. (Microsoft) · github.com/microsoft/LightGBM"). Verify years by web search before baking them in.
- **Abbreviations (convention).** Spell the **full name on first use, then introduce the abbreviation**, e.g. "Gradient-based One-Side Sampling (GOSS)", "Exclusive Feature Bundling (EFB)", "DART (Dropouts meet Multiple Additive Regression Trees)". After that, the abbreviation alone is fine.

Common math macros (from `preamble.tex`): `\xv \yv \thetav \thx \fh \fxh \sumin \argmin \risk \riske`. Reuse them, don't redefine.

---

## Figures

- **Every essential figure is Python-generated (matplotlib), not TikZ** - anything data-driven (curves, distributions, benchmarks, confusion matrices) and any diagram that carries the frame's core message. **TikZ only for small throwaway visuals** - quick boxes-and-arrows, tiny annotations, decoration the frame could live without.
- Generation scripts live in a sibling **`py_src/`**; output PDFs/PNGs to a sibling **`fig/`**. Run them with the **`ma` venv** (`./ma/Scripts/python.exe py_src/script.py`).
- **No enforced figure palette** — sensible per-figure defaults. (Global fallback for 3+ color charts: Armenian-flag red `#D90012`, blue `#0033A0`, orange `#F2A800`.)
- **Embedded third-party figures:** keep a source/attribution line on the frame (e.g. "Source: LMU i2ml, CC BY 4.0"). Verify the license first.

---

## Prose style (from the global rules)

- Plain prose: no em-dashes (use `-`), no curly quotes (use `"` and `'`).
- Fail loud in any helper scripts; no silent fallbacks.
