# 33 — Color Spaces (outline, for approval)

**Status:** BUILT and REVIEWED 2026-08-20. `33_color_spaces.tex`, **24 frames / 30 PDF pages**
(three `\pause` steps). Both mechanical gates clean: `detect_clipped_slides.py` 0 flagged,
`detect_footer_collisions.py` 1 flag on the hue-threshold frame, verified by eye as band
occupancy and not a collision. Acronym check passes.

**Note on scope drift.** This started as a 15-20 frame interlude, went to 22 at the interview,
23 after the student review, and 24 (30 pages) after the instructor asked for depth perception,
task-first guidance and the wider zoo of spaces. It is now a full lecture, not an interlude, and
the "Where should a 90-minute session end" answer in the student review is worth re-reading with
that in mind — the reviewer suggested stopping after the grayscale section if the practical has
to start in the same slot.
**Slot:** interlude between the clustering lecture (`32`) and the image practicals (now `34_*`).
**Length:** 22 frames. Instructor asked for 15-20 but kept all four extra topics and accepted 22.

**Instructor decisions taken during the interview:** number it `33` and renumber everything after
(not `32b`); all four extra topics in (gamma, Lab, YCbCr, colour histograms); Saryan painting
everywhere, including in the three scripts copied from `ch6_cnn`; leave `L16` alone for now.

---

## Why this deck exists

Three things in the course currently assume color knowledge that is taught nowhere:

1. `32_clustering.tex` ends by clustering pixels "in RGB space" and repainting them with the
   centroid color. Nothing has said what those three numbers are or why Euclidean distance
   between them should mean anything.
2. The image-compression practical has a whole section (**"Does the color space matter? RGB vs
   HSV"**) that asks students to reason about hue circularity and cone encoding cold.
3. The photo-grouping practical's task 1 asks students to cluster a **color histogram** without
   ever defining one.

The material that *does* exist sits in `ml/ch6_cnn/L16_cnn_foundations.tex` Section 1 — roughly
three weeks too late in the schedule.

---

## Renumbering (prerequisite)

`33` is currently the practicals. Everything from there shifts by one:

| Now | Becomes |
|---|---|
| *(new)* | `ml/09_clustering/33_color_spaces.{tex,pdf}` |
| `33_image_compression_solution.ipynb` | `34_image_compression_solution.ipynb` |
| `33_land_cover_solution.ipynb` | `34_land_cover_solution.ipynb` |
| `33_land_cover_OUTLINE.md` | `34_land_cover_OUTLINE.md` |
| `33_image_clusters_solution.ipynb` | `34_image_clusters_solution.ipynb` |
| `34_dimensionality_reduction.{tex,pdf}` + `_OUTLINE.md` | `35_dimensionality_reduction.*` |
| `35_umap.{tex,pdf}` + `_OUTLINE.md` | `36_umap.*` |
| `36_eigenfaces_solution.ipynb` | `37_eigenfaces_solution.ipynb` |

**Both folders are untracked in git** (`?? ml/09_clustering/`, `?? ml/10_dimensionality_reduction/`),
so the rename costs no history and `git mv` does not apply. It also means **`git grep` cannot see
them** — the sweep in `_learnings/2026-08-12-1615_use-git-grep-for-rename-sweeps.md` says to use
`git grep` as authoritative, which is wrong for this case. `git grep` found 4 files; ripgrep over
`ml/` found 13.

Live references to update (13 files):

- `ml/00_plan.md` (lines 59-61, 82) — schedule table, plus a new row for the color deck
- `ml/MISSING_TOPICS.md` (34, 165)
- `ml/ch10_diffusion/DIFFUSION_CHAPTER_PLAN.md`
- `ml/09_clustering/09_clustering.qmd`, `REVIEW.md`, `32_clustering.tex`, `32_clustering_OUTLINE.md`,
  `33_land_cover_OUTLINE.md`
- `ml/10_dimensionality_reduction/10_dimensionality_reduction.qmd`, `REVIEW.md`,
  `34_dimensionality_reduction.tex`, `35_umap.tex`, `py_src/umap_demos.py`

`DECISIONS.md` mentions the old numbers in entries #20 and the ch09/ch10 rename records. Those are
**historical and stay as written** (the "do not edit history" rule); a new entry gets added instead.

---

## Frame list

### Cold open (1 frame)

**"You already clustered these"** — the Saryan k-means palette from the end of `32`. It rests on
two unexamined claims: a pixel is a point in a 3D space, and distance in that space means
"different color". Both are shakier than they look. Predict-first seed: *is pixel value 128 half
as bright as 255?*

### Outline (1 frame)

### Section 1 — What a pixel actually stores (4 frames)

| # | Frame | Figure | Source |
|---|---|---|---|
| 1 | `[plain]` transition: "A pixel is not a color, it is three numbers" | — | new |
| 2 | How your eye sees color — S/M/L cones, three numbers per point. Extended with **metamerism**: two different light spectra hitting the same three cone responses look identical, which is *why* three numbers are enough. | `eye_cones.pdf` | **copied** from `ch6_cnn/py_src/eye_cones.py` |
| 3 | A pixel is three numbers — RGB triples, three stacked grids | `rgb_channels.pdf` | **copied** from `ch6_cnn/py_src/rgb_channels.py` |
| 4 | **Predict-first: gamma.** Pixel 128 is **21.6%** of the light of 255, not 50%. Half the light is pixel **188**. The sRGB curve and why it exists (8 bits spent where the eye is sensitive). Second panel: averaging in gamma space darkens gradients, which is why naive image resizing looks wrong. | `gamma_curve.pdf` (new, 2 panels) | new |

Measured, `ma` venv: `lin(128/255) = 0.2159`; `255 * (1.055 * 0.5^(1/2.4) - 0.055) = 187.5`.
sRGB transfer function verified against the ICC spec: `x/12.92` for `x <= 0.04045`, else
`((x+0.055)/1.055)^2.4`.

### Section 2 — Same pixel, different axes: HSV (4 frames)

| # | Frame | Figure | Source |
|---|---|---|---|
| 5 | `[plain]` transition: "RGB says how to make the color, not what it is" | — | new |
| 6 | HSV axes — hue (angle), saturation (radius), value (height). Carries the existing measured argument: subtract a constant in RGB and hue drifts by a measured number of degrees with N% of pixels crushed to black; scale V and hue *provably* cannot move (the script asserts `max gap < 1e-6`). | `hsv_space.pdf` | **copied** from `ch6_cnn/py_src/hsv_space.py` |
| 7 | **The trap: hue is circular.** 0.99 and 0.01 are both red and maximally far apart in Euclidean distance. Hand k-means naive HSV and hue coherence collapses to ~0. The cone encoding `(S·cos 2πH, S·sin 2πH, V)` fixes it completely. | `hue_seam.pdf` (new) | new figure, **numbers ported** from the image-compression practical, section 10 |
| 8 | When HSV earns its keep — thresholding one object by hue survives a lighting change that breaks an RGB threshold. | `hsv_threshold.pdf` (new) | new |

### Section 3 — Throwing color away: grayscale (4 frames)

| # | Frame | Figure | Source |
|---|---|---|---|
| 9 | `[plain]` transition: "One number instead of three" | — | new |
| 10 | **Predict-first:** should `(R+G+B)/3` be the answer? No — green carries most of the perceived brightness (cone density plus the luminous-efficiency peak near 555 nm). Worked by hand on pure red / green / blue. | `gray_weights.pdf` (new) | new |
| 11 | **Two standards, and your libraries disagree.** Rec.601 `(0.299, 0.587, 0.114)` vs Rec.709 `(0.2126, 0.7152, 0.0722)`. Measured in the `ma` venv on the astronaut crop: `PIL.convert("L")` → mean **79.285** (601); `skimage.rgb2gray` → mean **77.741** (709). Same input, two different grayscale images. | `gray_weights.pdf` panel 2 | new, measured |
| 12 | The gotcha nobody fixes: both formulas get applied to **gamma-encoded** values, which makes them *luma* (Y′), not *luminance* (Y). Correct order is linearize → weight → re-encode. Almost no code does it. | table frame | new |

By-hand numbers for frame 10: pure blue → 29.1 (601) / 18.4 (709) / 85.0 (naive mean);
pure green → 149.7 / 182.4 / 85.0.

### Section 4 — Distance should mean difference (5 frames)

| # | Frame | Figure | Source |
|---|---|---|---|
| 13 | `[plain]` transition: "Distance should mean difference" | — | new |
| 14 | **Lab and perceptual uniformity.** Measured, seed 509, 4000 random pairs at a fixed RGB distance of 20: perceived difference ranges from ΔE **4.3** (5th pct) to **16.2** (95th) — a **3.8×** spread, against a just-noticeable threshold of ΔE ≈ 2.3. Lab is built so Euclidean distance ≈ perceived difference. Ties straight back to k-means measuring distance in whatever space you hand it. | `lab_deltae.pdf` (new) | new, measured |
| 15 | **YCbCr and chroma subsampling.** Split brightness from color, then store *less color*. 4:4:4 vs 4:2:0, and the asymmetry demo: subsample the chroma and you barely see it; subsample the luma instead and it falls apart. | `chroma_subsample.pdf` (new) | new |
| 16 | **Color histograms as features.** What one is, and the failure the photo-grouping practical is about to hand them: a histogram groups sunsets with tomatoes, because it discards every bit of spatial structure. | `color_histogram.pdf` (new) | new |
| 17 | **Which space when** — decision table. RGB for reconstruction and anything a network consumes; cone-encoded HSV for coherent palettes and lighting-robust thresholds; Lab when distance must match perception; grayscale when color is genuinely irrelevant and you want 3× fewer numbers; YCbCr when you are compressing. Callback: the Sevan practical, where "RGB" is 6 bands and one of them is infrared, so the whole premise of three numbers is a convention, not a law. | table frame | new |

### Recap (1 frame)

Recap + `paramgreen` **Next:** box pointing to the image practicals (`34_*`).

---

## Figure work

Ten scripts in `ml/09_clustering/py_src/`, all writing to `../fig/`. Shared helpers live in
**`color_common.py`** (Saryan loader, logging setup, sRGB transfer functions, palette) rather than
being pasted into each script.

**Copied** from `ml/ch6_cnn/py_src/` (originals stay put; `L16` is untouched and still compiles):

| Script | Output | Change from the original |
|---|---|---|
| `eye_cones.py` | `eye_cones.pdf` | re-pointed to `color_common`; **new metamerism panel** |
| `rgb_channels.py` | `rgb_channels.pdf` | Saryan instead of astronaut; **new zoomed 4x4 patch** with the actual (R,G,B) triples printed on the pixels |
| `hsv_space.py` | `hsv_space.pdf` | Saryan instead of astronaut; measurements unchanged |

**New:**

| Script | Output |
|---|---|
| `color_gamma.py` | `gamma_curve.pdf` (sRGB curve + the 128/188 markers; the black+white averaging panel) |
| `color_hue_seam.py` | `hue_seam.pdf` (number line vs circle, then k-means in three spaces with measured seam overlap) |
| `color_hsv_threshold.py` | `hsv_threshold.pdf` (one region, two lightings, RGB vs hue threshold, IoU) |
| `color_grayscale.py` | `gray_weights.pdf` (primaries under three recipes, the image under each, and the 601-minus-709 difference map) |
| `color_lab.py` | `lab_deltae.pdf` (ΔE distribution at fixed RGB distance + the least/most visible pair) |
| `color_ycbcr.py` | `chroma_subsample.pdf` (Y/Cb split, then 4:2:0 on chroma vs the same on luma, zoomed) |
| `color_histogram.py` | `color_histogram.pdf` (painting vs shuffled pixels, identical histograms) |
| `color_depth.py` | `depth_shading.pdf` (three Lambertian spheres split into H/S/V, then the destroy test: flatten V and depth vanishes, discard colour and it survives; plus a specular sphere as the exception) |

All follow the house pattern: `ma` venv, seed 509, `logging` to console + `logs/`, fail loud,
assert the claims the slides make rather than trusting them. Note `np.trapz`, not `np.trapezoid` —
the `ma` venv is on numpy 1.26.4.

---

## Also needs writing

- **`DECISIONS.md`** — new entry: why a color-spaces interlude exists, why it went at 33 rather
  than staying folded into the CNN chapter, why the practicals shifted.
- **`ml/00_plan.md`** — new schedule row + the shifted numbers.
- **`ml/09_clustering/09_clustering.qmd`** — link the new deck, update the practical filenames.
- **`DEFERRED_TODO.md`** — the `L16` trim pass (its Section 1 now duplicates this deck; decide
  later whether it shrinks to one recap frame).

---

## What changed during the build

- **`eye_cones.py` gained a metamerism panel.** The first version solved for two spike weights to
  match a target spectrum's cone responses; it failed its own assertion (109% mismatch on S),
  because two unknowns cannot satisfy three equations. Rewritten with **three** primaries and an
  exact solve, which is both correct and a better slide: this is literally why a screen has three
  primaries. Matches to solver precision.
- **The Lab claim was rewritten after measuring it.** See `DECISIONS.md` #24.
- **Four frames were clipping and the detector caught all four.** Pages 6, 7, 13 and 21 lost their
  callout boxes to too-narrow left columns in `columns[T]` layouts. Compile log was silent, page
  count correct, exit 0 - exactly the failure mode
  `_learnings/2026-08-20-1745_the-clipped-slide-detector-works-nobody-ran-it.md` documents. Fixed
  by widening each left column ~0.04 and trimming prose.
- **One "false positive" was a real style violation.** The detector kept flagging the HSV frame
  because it was written `\textbf{H}ue`, `\textbf{S}aturation`, `\textbf{V}alue` - which
  `ml/SLIDE_STYLE.md` explicitly forbids, for exactly this reason: the markup splits the words so
  neither a grep nor a reader searching the PDF can find them. Rewritten as the approved form
  (plain phrase, then the acronym in parentheses); the flag went away on its own.
- **The chroma-subsampling frame needed a zoom.** At full-page scale the chroma-damaged and
  luma-damaged images look identical and the frame proves nothing. It now crops to the 96x96
  window with the most chroma detail, chosen from the data by summed-area table.

## What the student review changed

Applied:

- **New frame 9, "Linearize, compute, re-encode".** The deck quoted gamma numbers and prescribed
  "linearize -> compute -> re-encode" without ever showing the sRGB function, so a student could
  recognise the bug but not fix it. The reviewer hit this from five directions independently (an
  exam question they could not answer, two self-efficacy failures, an office-hours question, and
  the reason the "everyone does this wrong" framing read as a disclaimer rather than a tool). The
  frame gives both transfer functions and then *derives* the 187.5 and the 0.216 that earlier
  frames assert.
- **The luma weights now point back at the cone curves.** They had been presented as standards
  handed down, with the cone-sensitivity figure sitting unused two sections earlier.
- **IoU is now defined, not just expanded.**

Not applied:

- **Reviewer's top cut was YCbCr / 4:2:0**, arguing none of the three practicals touch JPEG or
  video. Sound argument, but the instructor was offered exactly that cut during the interview and
  chose to keep it. Recorded here rather than acted on.

Worth keeping in view:

- The reviewer would rather gamma and YCbCr were **folded into the practical**, motivated live by
  "why does my resize look muddy", than pre-loaded here in the abstract. That is a real structural
  alternative for this interlude if it ever gets revisited.
- **The running example was the single strongest structural choice** - one painting threaded
  through every frame, with the shuffled-histogram bookend (cold open, then the histogram frame)
  landing specifically because the image was already familiar.

Adding one frame re-broke two others, exactly as `WORKFLOWS.md` step 4 predicts: the IoU
definition pushed the hue-threshold frame past the bottom edge, and the cone sentence pushed the
grayscale callout into the page number. Both found by re-running the detectors, both fixed.

## Measured numbers the deck quotes

Every one is computed by the figure script that draws it; the scripts assert rather than assume.

| Claim | Value | Script |
|---|---|---|
| light at pixel 128 | 21.6% | `color_gamma.py` |
| pixel carrying half the light | 187.5 | `color_gamma.py` |
| naive black+white average | code 127.5 = 21.4% light, 57% too dark | `color_gamma.py` |
| RGB subtract-110 hue drift | 24.8 deg mean, 13.8% crushed to black | `hsv_space.py` |
| V-scale vs uniform RGB multiply | identical, max gap 3.2e-16 (asserted) | `hsv_space.py` |
| seam overlap, k=8 | RGB 93.1%, naive HSV 0.0%, cone HSV 98.8% | `color_hue_seam.py` |
| threshold IoU after dimming to 55% | RGB 19.5%, hue 99.3% | `color_hsv_threshold.py` |
| PIL vs skimage grayscale | 4.28 levels mean, 23.8 max | `color_grayscale.py` |
| luma vs true luminance | 4.1 mean, 48.1 max | `color_grayscale.py` |
| ΔE spread at fixed RGB distance 20 | 4.37 (p05) to 15.96 (p95), 3.7x | `color_lab.py` |
| 4:2:0 chroma vs luma halving | RMSE 5.84 vs 11.10 | `color_ycbcr.py` |
| histogram after pixel shuffle | bit-identical (asserted to 0.0) | `color_histogram.py` |
| hue/saturation across a lit Lambertian sphere | H circular std 2.4e-9, S std 2.3e-16 | `color_depth.py` |
| V against true shading | corr = 1.000000 on all three spheres | `color_depth.py` |
| saturation under a specular highlight | falls from 0.78 to 0.00 | `color_depth.py` |
