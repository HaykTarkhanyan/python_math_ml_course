# CONVENTIONS

Recurring decisions made across this course. Codify here so future-me and Claude don't reinvent them per file.

This is course-specific only. Tooling conventions (uv, logging, plot colors, etc.) live in `CLAUDE.md`. Recurring deferrals live in `DEFERRED_TODO.md`.

Last updated: 2026-08-07

---

## Naming

### Lecture slides — `NN_topic_name.{tex,pdf}`

- `NN` is a two-digit ordinal matching the YouTube playlist position (e.g. `01`, `02`, `04`).
- Topic name in lowercase with underscores: `01_intro_linear_regression.tex`, `06_overfitting_cross_validation.tex`.
- No `L01`, `L01b`, `L01c` style prefixes — those are legacy and being phased out as decks are reviewed.

### Notes PDFs — `NN_topic_name_notes.pdf`

- Sits next to the slide PDF.
- Hand-annotated version exported after class, not auto-generated.

### Homework notebooks — `NN_HWX_solution.ipynb`

- `NN` is the YouTube playlist number for THAT homework video.
- `HWX` is the homework number within the chapter (`HW1`, `HW2`).
- Example: `04_HW1_solution.ipynb` = playlist video #4 = solution to homework 1 of chapter 1.

### Chapter folders — `NN_short_descriptor/`

- Under `ml/`. Number prefix gives global course order.
- Example: `ml/01_regression_intro/`, `ml/02_main_concepts/`.

### Chapter QMD — `NN_chapter_topic.qmd`

- Lives inside the chapter folder. Number matches the chapter folder number.
- Example: `01_regression_intro/01_regression_intro.qmd`.

### Random images — `ml/00_random_image/NN_descriptive.jpg`

- One per chapter, used in the QMD's Random section.
- Numbered to match the chapter that references it.

---

## YouTube link text

**Link text MUST match the YouTube video title exactly.**

- No invented `[NN]` prefixes if the actual YouTube title doesn't have one.
- No translation, paraphrasing, or "cleanup" of the title.
- If YouTube says `Գերհարմարեցում, թերհարմարեցում, Cross-Validation`, the QMD link says `Գերհարմարեցում, թերհարմարեցում, Cross-Validation`. Period.
- Reason: students search by title, and a mismatch makes the link harder to verify.

---

## QMD structure

Each chapter's QMD follows this template (see `01_regression_intro/01_regression_intro.qmd` for the canonical example):

```markdown
---
title: "NN Chapter Title"
resources:
  - data/file.csv   # only if the page resources need to be served
---

# 🎲 Random

- [Optional weekly link](https://...)

![image.png](../00_random_image/NN_descriptive.jpg)

# 📚 Նյութը

- [📺 Exact YouTube title](https://youtu.be/...), [🎞️ Սլայդեր](NN_slides.pdf), [📝 Նշումներով](NN_slides_notes.pdf)
- ... one bullet per lecture video / homework video ...

📝 **Թեմայի վերաբերյալ հարցաշար (Google Form):** [հղում](https://forms.gle/...)

# 🏡 Տնային

... assignment descriptions, one per HW ...

<flag-counter HTML>
```

If a section's content doesn't exist yet, mark it `TBD` rather than omitting — visible TBDs prevent silent gaps.

---

## Per-chapter folder layout

Inside each `NN_chapter_topic/`:

```
NN_chapter_topic.qmd       # the page
NN_lecture_1.{tex,pdf}     # slide source + render
NN_lecture_1_notes.pdf     # hand-annotated
NN_HW1_solution.ipynb      # homework solution notebook
data/                      # datasets used by this chapter's notebooks
gform/                     # Apps Script + question CSVs for the Google Form
```

- **figures/** subfolder only if `.tex` decks include external images (TikZ-only decks don't need it).
- **gform/** keeps form scaffolding out of the main lecture flow.

---

## Borrowed video material — `_reference_<slug>/` in, `fig/borrowed/<source>/` out

Used twice now (3Blue1Brown in `ml/ch9_attention`, Welch Labs in `ml/ch16_jepa`), so it is a
convention rather than a one-off. Fetch with the `youtube-reference` skill.

```
ml/<chapter>/_reference_<slug>/
  transcript.txt      # cleaned + timestamped   (committed)
  README.md           # beat map + re-fetch command, written by hand (committed)
  meta.txt            # one-line metadata       (committed)
  description.txt     # chapters + links        (committed)
  frames/             # fNN_HH-MM-SS.jpg        (committed, small)
  video.mp4, *.vtt    # GIT-IGNORED via the folder's own .gitignore
ml/<chapter>/fig/borrowed/<source>/
  descriptive_name.jpg   # the stills a deck actually uses, renamed
```

- **Never commit the video.** `yt_fetch.sh` writes the `.gitignore`; keep it. ch16 carries ~355 MB
  of video locally and ~6 MB in git. The README's re-fetch command is the recovery path.
- **Frames get descriptive names when they enter `fig/borrowed/`** (`jepa_core_arch.jpg`, not
  `f18_00-31-20.jpg`). The reference folder keeps the timestamped originals so a frame can always
  be traced back.
- **The README is written by hand**, after reading the transcript — the script cannot summarise
  content. It carries a timestamped beat map, which becomes the shot list when building the deck.
- **Attribution goes on every borrowed slide**, as a small corner node. Full-bleed mechanism and
  the 16:9 / 4:3 macros are in `ml/SLIDE_STYLE.md`.
- **Check each still is *settled* and actually shows what its slide claims.** Nothing automated
  catches a mid-animation grab or a mismatched illustration — see
  `_learnings/2026-08-08-2300_borrowed-video-stills-need-a-settled-state-check.md`.

---

## Training runs — tag every artifact, never overwrite

Any script that trains a model and can be run more than once with different settings carries a
`TAG` constant, and **every** path it writes is built from it. Canonical example:
`ml/ch10_diffusion/py_src/train_panir_ddpm.py`.

```python
DATA_SIZE = 24
TAG = ""          # "" is the shipped run; "_lvl1", "_res32" etc. for experiments

def paths():
    d, suf = CHAPTER / "data", f"{DATA_SIZE}{TAG}"
    return (d / f"mashtots_panir_{DATA_SIZE}.npz", d / f"panir_ddpm_{suf}.pt",
            d / f"panir_ddpm_{suf}_ckpt.pt", d / f"panir_progression_{suf}.npz")
```

Rules:

- One `paths()` function returns every output path. No path literals scattered through the script — that is how a rerun silently clobbers the good weights.
- Weights, checkpoint, figures and any progression/metrics file all take the same suffix, so a run's artifacts are greppable as a set and comparable side by side.
- The untagged name (`TAG = ""`) is the **shipped** artifact — the one the qmd and notebooks load. Experiments always get a tag.
- Save weights as CPU tensors (`{k: v.cpu() for k, v in state_dict.items()}`). A checkpoint saved on a GPU raises on load for every student who does not have one.
- Store the config *inside* the checkpoint (`ch`, `levels`, `size`, `n_classes`, `letters`) and check for missing keys on load. A stale checkpoint that loads with mismatched keys produces plausible-looking garbage, which reads like a modelling bug and costs hours.

Reason: during ch10 four runs were compared (2-level, 1-level, 24px, 32px). Without tagging, each would have overwritten the last and no comparison would have been possible.

---

## Special top-level files

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project-level instructions for Claude Code (build, LaTeX, Quarto gotchas). |
| `WORKFLOWS.md` | Task playbook for AI assistants: task -> workflow/skill/source-of-truth map, definition of done, hard rules. |
| `CONVENTIONS.md` | This file. Naming, structure, YouTube title rules. |
| `DEFERRED_TODO.md` | Parked topics (bias-variance bits, GLMs, causal inference, etc.). |
| `_learnings/*.md` | Non-obvious lessons, one file per lesson (`YYYY-MM-DD-HHMM_slug.md`). **New lessons go here.** |
| `LEARNINGS.md` | Read-only archive of pre-2026-08-08 lessons. Do not append; see its header. |
| `_work_sessions/*.toml` | Per-session logs (one TOML per session); `PROGRESS.md` is the pre-2026-07-22 archive. |

---

## Other holding areas

- **Orphan / not-yet-delivered decks** — keep in a dedicated staging folder; don't bury them in numbered chapter folders until they're part of a delivered lecture. The former `ml/upcoming_lectures/` held feature engineering + selection and was promoted to `ml/06_feature_engineering/` (2026-07-07).
- **`ml/deferred/`** — pre-built decks parked for "later in the course" (currently: GLMs, causal inference, regression inference). See `DEFERRED_TODO.md`.

---

## When to update this file

- Whenever a decision gets made twice — that's a convention waiting to be written.
- Whenever you catch yourself re-explaining a structural choice to Claude or to a future-you.
- Delete any convention here that's been abandoned. Stale conventions cause more confusion than no convention.
