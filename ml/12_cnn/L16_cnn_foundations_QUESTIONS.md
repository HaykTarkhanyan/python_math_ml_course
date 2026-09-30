# L16 CNN Foundations - implementer questions (Opus, 2026-07-13)

I read the L16 outline, the master `CNN_CHAPTER_PLAN.md`, `ml/SLIDE_STYLE.md`, the
WORKFLOWS "new slide deck" row, and verified the referenced assets on disk before writing
these. Only listing things the repo itself can't answer.

**Verdict on the plan:** solid and buildable. I checked every `[worked-numbers]` example
and all the arithmetic is correct (2D conv -> `[[10,4],[5,13]]`; multi-channel sum = 5;
param counts 448 / 76 / 80 / 1168 / 7850 / 9098; output-size 28/24/15; the 109,386-param
MLP baseline; 150M-weight cold open). I do not need to change any numbers.

---

## A. Blocking - I need your answer before these parts can be finished

### A1. The pomegranate photo (plan open question 5)

`ml/ch6_cnn/fig/src_pomegranate.jpg` does **not exist yet**. Three figures depend on it and
are written to fail loudly if it's missing: `pixel_shuffle.py`, `kernel_zoo.py` (both L16),
and `task_zoo.py` (L18). How do you want to proceed?

- **(a)** You drop the photo in now and I build everything against it.
- **(b)** I build the whole deck + all photo-independent figures first, and leave the 3
  photo-dependent figures as the final step once you supply it. (My default if I don't
  hear back - keeps momentum.)
- **(c)** I temporarily wire in a permissively-licensed stand-in fruit photo so the deck
  compiles end-to-end, clearly marked TEMP, and we swap in your photo later.

For the L16 kernel zoo the photo is used in **grayscale** and wants clear edges/texture (a
single pomegranate is ideal). Note the L18 task-zoo frame later wants **several fruit
visible** - so you may eventually want two shots, but L16 needs only the one.

### A2. Scope of this build pass

The outline is the **deck only** (no HW frame in it - the homework hook says "lives on
cnn.qmd"). The master plan's per-lecture bundle is deck **+** `cnn.qmd` chapter page **+**
`HW1_solution.ipynb`. Which do you want in this pass?

- **(a)** Just the L16 `.tex` deck (compiled, overflow-checked, figures generated). My
  default read of "implement the L16 outline."
- **(b)** Deck + create `cnn.qmd` (register in `_quarto.yml`) + HW1 solution notebook.

I'll do (a) unless you say otherwise, since the plan's build order is "one deck at a time,
approved before the next."

---

## B. Defaults I'll take unless you object (not blocking)

### B1. Kernel zoo: 6 or 7 kernels?

Internal inconsistency in the plan. The **outline body** names **6** (identity, box blur,
sharpen, Sobel X, Sobel Y, emboss - no Gaussian). The **build notes + chapter plan** list
**7** (they add a Gaussian blur = 1/16 * [[1,2,1],[2,4,2],[1,2,1]]).

**Default: 7 (include Gaussian).** It's the authoritative build-notes list, and box blur
vs Gaussian nicely shows two *different* blurs. Say the word if you want the leaner 6.

### B2. The contingency training-loop recap frame (section 5)

The outline marks it "include ONLY if L15 is not fresh at delivery time." I can't know your
delivery schedule.

**Default: include it.** It's one cheap recap frame, trivially cut live if L15 just ran.
Tell me if L16 runs immediately after L15 and you'd rather drop it.

### B3. "CNNs in the wild" montage - already-decided, just confirming

I'll ship the 4 whitelisted LMU images (`tesla_autopilot.jpg`, `coronatrack.jpeg`,
`cityscapes_visual.png`, `colorization.png`) with a per-image original-source citation,
exactly as your COPY-IMG whitelist specifies. All 4 files are present. Flag only if you've
reconsidered - a couple originate from third parties (Tesla, a colorization paper), so
LMU's CC BY covers LMU's own figures but not those; per-image paper citations handle it,
and it's a free course, so I think it's fine.

---

## C. Verified, no action needed (so you know what I checked)

- All 4 COPY-IMG files present in `_reference/lecture_i2dl/slides/cnn1/figure/`.
- COPY-SLIDE PDF present: `slides/week23_cnns_intro/03_cnn-properties-of-convolution.pdf`
  (fallback for the sparse-interactions build). I'll prefer fresh TikZ for the 16-vs-36
  connection diagram - it's cheap - and only reach for COPY-SLIDE if a multi-overlay
  recreation proves costly.
- ch5 practical is already **Fashion-MNIST** (784-128-64-10 MLP), so the CNN-vs-MLP payoff
  and HW1B reuse it cleanly (resolves plan open question 3 - the "switch" is already true).
- `ma` venv has torch, numpy, matplotlib, sklearn, PIL, tensorflow. **torchvision is
  missing** but L16 doesn't need it - I'll load Fashion-MNIST via `keras.datasets` (like
  ch5) and build the tiny CNN in raw torch. (torchvision only matters for L17's
  pretrained-filters figure.)
- LMU `cnn1/*.tex` sources all present for REDERIVE.

---

## Next step

Once A1 and A2 are answered I'll build the L16 deck per the outline (2x pdflatex ->
0 `!` lines -> clean aux -> overflow check -> open for review), generating the
photo-independent figures first. B1-B3 I'll proceed on my defaults unless you push back.

---

## Answers (reviewer, 2026-07-13)

### A1 - RESOLVED: the photo exists now

`ml/ch6_cnn/fig/src_pomegranate.jpg` was dropped in after you wrote this (1280x960 JPEG,
RGB - verified it opens with PIL and inspected it visually: a cracked-open pomegranate
with exposed arils on a wooden table, sharp, high-contrast). It is very good for L16:
strong edges for Sobel/emboss, dense aril texture for blur/sharpen. Proceed with your
option (a) - build everything against it.

Two notes:
1. **Provenance:** if this is not the instructor's own shot, it needs a source line
   wherever it appears. Ask in your next questions file OR add a `TODO photo credit`
   comment next to the `\includegraphics` so it is not forgotten. Do not block on it.
2. **L18:** this is a single fruit; the L18 task-zoo frame will still need a multi-fruit
   shot later. Not your problem for L16.

### A2 - (a): deck only

This pass = the L16 `.tex`, its figures, compiled + overflow-checked + aux-cleaned, PDF
opened for review. `cnn.qmd` (+ `_quarto.yml` registration) and the HW1 solution
notebook are a separate pass after the deck is approved - matching the plan's
"one deck at a time" build order.

### B1 - agreed: 7 kernels (Gaussian included)

Good catch; that was an inconsistency in the outline. The outline body and figure list
are now fixed to 7 (edited 2026-07-13), so the build notes, outline, and plan agree.
Box blur vs Gaussian side by side is exactly the right contrast.

### B2 - agreed: include the recap frame

Include it. It also survives as insurance for students who skipped/forgot L15.

### B3 - confirmed as specified

Ship the 4 montage images with per-image original-source citations (Tesla and the
colorization figure cite their original sources, not LMU). Instructor has additionally
approved full-slide embeds chapter-wide (COPY-SLIDE mechanism in the plan) with the
up-front credit line - so your TikZ-first stance with COPY-SLIDE as fallback is right.

### On your C-notes

- Fashion-MNIST via `keras.datasets` + raw torch for L16: fine, matches ch5 practice.
- torchvision: install into `ma` BEFORE starting L17 (`uv pip install --python
  ./ma/Scripts/python.exe torchvision`) - needed for `pretrained_filters.py`. Not now.
- Thanks for verifying the arithmetic; noted that plan open question 3 (Fashion-MNIST
  for HW1B) is de facto resolved since ch5's practical already uses it.

**Green light: build L16.**
