# Slide style guide — `ml_new` Beamer decks

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

- `\documentclass[aspectratio=169]{beamer}` then `\input{../preamble}` (decks sit one level under `ml_new/`).
- End each `.tex` with a `% Provenance:` comment block — source material, which figures, key pedagogical decisions, and the forward pointer to the next deck. (Matches every existing deck.)

---

## Palette & callouts

Defined in `ml_new/preamble.tex`:

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
- **No fixed deck length.** One idea per frame; split a dense frame into two.

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

Common math macros (from `preamble.tex`): `\xv \yv \thetav \thx \fh \fxh \sumin \argmin \risk \riske`. Reuse them, don't redefine.

---

## Figures

- **Real (matplotlib / generated) for anything data-driven** — curves, distributions, benchmarks, confusion matrices. **TikZ only for conceptual schematics** (boxes, arrows, flow, geometry).
- Generation scripts live in a sibling **`py_src/`**; output PDFs/PNGs to a sibling **`fig/`**. Run them with the **`ma` venv** (`./ma/Scripts/python.exe py_src/script.py`).
- **No enforced figure palette** — sensible per-figure defaults. (Global fallback for 3+ color charts: Armenian-flag red `#D90012`, blue `#0033A0`, orange `#F2A800`.)
- **Embedded third-party figures:** keep a source/attribution line on the frame (e.g. "Source: LMU i2ml, CC BY 4.0"). Verify the license first.

---

## Prose style (from the global rules)

- Plain prose: no em-dashes (use `-`), no curly quotes (use `"` and `'`).
- Fail loud in any helper scripts; no silent fallbacks.
