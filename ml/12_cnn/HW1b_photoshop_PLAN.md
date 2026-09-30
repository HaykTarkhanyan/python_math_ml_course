# HW1b - "Photoshop from scratch" - plan (for approval)

Drafted 2026-08-02. **Plan only - nothing built yet.**

A standalone, convolution-only homework attached to **L16 CNN Foundations**. It does not
replace HW1 ("Build your own Photoshop, then let the network design the kernels") from
`CNN_CHAPTER_PLAN.md` - HW1 stays exactly as specced, including its Fashion-MNIST Part B.
This is an additional assignment, so the chapter grows from 4 homeworks to 5.

Instructor decisions taken 2026-08-02 (three questions, three answers):

1. **Standalone extra homework.** HW1 untouched.
2. **Biopsy images downloaded from Wikimedia Commons.** Both candidates verified live
   (resolution, license and visual suitability all checked before this plan was written).
3. **Notebook first, `cnn.qmd` wraps it.** The notebook is the real artifact; the chapter
   page finally gets created and registered in `_quarto.yml`.

## Why this homework exists (the one-line pitch)

HW1 asks students to convolve. This one asks them to **ship three things people actually
pay for**: a music-thumbnail generator, a colour-picker, and the first stage of a digital
pathology pipeline - all with kernels they type in by hand. Then it shows them all three
are brittle, which is the argument for the rest of the chapter.

No PyTorch. No dataset download. No GPU. Target runtime end to end on a laptop CPU:
**under 2 minutes**, except for the one cell that is *designed* to be too slow (see Part 2).

## The three anchors the instructor asked for

| Anchor | What it is | Where it lands |
|---|---|---|
| Photoshop features | named kernels mapped to real Photoshop menu items | Part 1 |
| Lo-fi music thumbnail | the blurred-background layout from `youtu.be/ydIFRGlXGy4` | Part 2 |
| HSV colour space | motivated by the thumbnail's darkening step, paid off on the biopsy | Part 3 |
| Liver biopsy | H&E and trichrome slides, filtered and quantified | Part 4 |

### The thumbnail, decomposed

Reference image: `https://img.youtube.com/vi/ydIFRGlXGy4/maxresdefault.jpg` (Alis,
"Ten more times" - a standard lo-fi music upload layout, re-fetchable from that URL, no
need to store the video). Reviewed 2026-08-02. The whole aesthetic is one convolution
plus one HSV operation:

- **background**: the square cover art, upscaled to fill 1280x720, hit with a large-sigma
  Gaussian blur, then darkened
- **foreground**: the untouched sharp original, square, centred, about 25% of canvas width,
  sitting slightly above the vertical centre
- **text**: bold white title, lighter grey artist name below, centred sans-serif

This is a better convolution exercise than "apply Sobel to a photo" for one reason: a
visually convincing blur needs a kernel around 150-180 px wide, at which point the naive
loop from Part 0 does not finish. The failure is the lesson.

## Part-by-part breakdown

### Part 0 - the engine 🧀

- Implement **two** versions in pure NumPy. No `scipy.signal`, no `cv2`. The gap between
  them is the whole point, and it is measured in Part 2.
  - `convolve2d_naive` - loop over output pixels, inner `k x k` window sum. The direct
    transcription of the definition on the L16 slides.
  - `convolve2d_fast` - loop over the `k x k` kernel **taps**, vectorised over the whole
    image (shift and accumulate). Same answer, `k^2` NumPy passes instead of `H*W` Python
    iterations.
- Three correctness checks shipped with the notebook: the identity kernel returns the input
  exactly, a hand-computed 3x3 example from the L16 slides matches cell for cell, and the
  two implementations agree to `1e-12`.
- Handle the three L16 padding cases (`valid` / `same` / stride) so the output-size formula
  `o = floor((i - k + 2p)/s) + 1` is something they have implemented, not just seen.
- **Time both** on a small image across a few kernel sizes, fit the per-tap-per-pixel cost,
  and *extrapolate* rather than wait. Part 2 cashes the extrapolation in.

### Part 1 - the kernel zoo as a Photoshop menu 🧀

Each kernel is introduced by the menu item it corresponds to, not by its matrix:

| Photoshop menu | Kernel |
|---|---|
| Filter > Blur > Box Blur | `ones((k,k)) / k**2` |
| Filter > Blur > Gaussian Blur | sampled 2D Gaussian, normalised |
| Filter > Blur > Motion Blur | 1-D line kernel rotated to an angle |
| Filter > Sharpen > Sharpen | unsharp mask, see below |
| Filter > Stylize > Emboss | asymmetric diagonal 3x3 |
| Filter > Stylize > Find Edges | Sobel magnitude |
| Filter > Other > Custom | students design their own |

The hook worth stating explicitly in the notebook: **Photoshop's `Filter > Other > Custom`
dialog is literally a 5x5 kernel entry grid with a scale and offset field.** Students are
not simulating Photoshop here, they are reimplementing the exact feature.

Two things to derive rather than hand over:

- **Unsharp mask**: `sharp = orig + amount * (orig - blur(orig))`. Ask *why* subtracting a
  blur sharpens. The answer connects straight to the L16 edges-as-derivatives arc: `orig -
  blur` is a high-pass filter, so adding it back amplifies exactly the frequencies the blur
  removed. Also worth noting: this is why the "Sharpen" slider produces halos when pushed -
  the same fact L16's Fourier aside sets up.
- **Emboss vs Sobel**: both are asymmetric difference kernels. Ask what the offset (+128)
  in the Photoshop dialog is doing, and why an edge kernel needs one.

Deliverable: a labelled grid figure, all kernels applied to the same image.

### Part 2 - the thumbnail generator 🧀🧀

Input: a square cover image. **Recommendation: reuse `fig/src_pomegranate.jpg`, centre-
cropped square**, with an invented track title. That keeps the chapter's pomegranate
running example threaded into the homework and means zero new assets to source.

Steps, in order:

1. Upscale and centre-crop the square to 1280x720 to make the background layer.
2. Gaussian blur it hard (sigma about 30 px).
3. Darken it (Part 3 does this properly; Part 2 does it naively in RGB first).
4. Composite the sharp original at 25% width, centred, slightly above the vertical centre.
5. Draw the title (bold white) and artist (grey) below it.

The cost ladder, and the rescue. **Corrected 2026-08-02 after measuring** - the earlier draft
said the naive loop "will not finish", which is only true of one of the two implementations.
The real story is a three-rung ladder, which is a better exercise than a binary:

A Gaussian truncated at ±3σ needs `k ≈ 6σ + 1`. At σ = 30 that is a **181x181** kernel:
32,761 taps per pixel per channel, times 1280x720x3 pixels, is about **9x10^10** multiply-adds.

| rung | implementation | σ=30 on 1280x720x3 |
|---|---|---|
| 1 | `convolve2d_naive` - loop over output pixels | **~8 hours** |
| 2 | `convolve2d_fast` - loop over kernel taps, vectorised | **~3.5 min** |
| 3 | separable - two 1D passes | **~2.3 s** |

Measured on this machine (`ma` venv, numpy 1.26.4) at 2.29 ns per tap per pixel for the tap
loop, and 1.04 s for a 7x7 on 256x256 for the pixel loop. Rungs 1 and 2 are **extrapolated
in the notebook, never actually run** at full size - students fit the cost model on small
kernels and predict, which is the point. Only rung 3 runs for real.

- **Separability**: a 2D Gaussian factors as `G(x,y) = G(x)·G(y)`, so one horizontal pass
  plus one vertical pass gives the identical result in `2k = 362` taps per pixel instead of
  `k² = 32,761`. That is exactly **90x** fewer taps. Students implement it, assert the output
  matches the full 2D version at small σ to numerical tolerance, and report the measured
  wall-clock speedup against rung 2.
- The jump from rung 1 to rung 2 is **not** an algorithmic improvement - both do the same
  9x10^10 multiply-adds. It is purely where the loop runs (Python vs NumPy's C). Rung 2 to
  rung 3 **is** algorithmic: genuinely fewer operations. Students should be asked to say
  which is which, because conflating the two is the standard beginner error.
- 🎁 **Bonus**: three successive box blurs approximate a Gaussian (central limit theorem),
  and a box blur can be done in O(1) per pixel with a running sum regardless of `k`. This
  is how real-time blurs are actually implemented. Measure it against the separable version.

Forward pointer to write into the notebook: this separability argument reappears in **L19**
as depthwise-separable convolutions, where the same factorisation trick is what makes
MobileNet run on a phone.

Deliverable: their generated 1280x720 thumbnail, side by side with the reference.

### Part 3 - HSV, motivated by step 3 above 🧀🧀

Opens by calling back to **L16 page 8, "Three numbers, but which three?"** (built 2026-08-02,
see below) rather than re-deriving the axes from cold.

Then straight to the problem, not the definition: *you need the background darker. How?*

- **Naive RGB attempts**: subtract a constant (saturated channels clip, hue shifts, blacks
  crush), or multiply all three channels (works, but you cannot express "same colour, less
  bright" without knowing that multiplying is the right operation and subtracting is not).
- **HSV**: convert, scale V alone, convert back. H and S are untouched by construction, so
  "same colour, less bright" is exactly one number changing.

Geometry to teach, briefly and with a figure:

- RGB is a **cube**. Its axes are three physical primaries; brightness is spread across all
  three, so no single axis means "how bright" and no axis means "which colour".
- HSV is a **cylinder** over that cube's diagonal. **H** = angle, 0-360°, which colour.
  **S** = radius, how far from grey. **V** = height, how bright.
- The conversion is short enough to write out: `V = max(R,G,B)`, `S = (max - min)/max`
  (0 when max is 0), and H is determined by which channel is the max and by how much the
  other two differ. Students implement V and S themselves; H's piecewise case analysis can
  be given, or implemented as a 🎁 bonus.
- **The one sentence students should keep**: RGB entangles *which colour* with *how bright*;
  HSV separates them. That is why every colour picker ever shipped is HSV or HSL, and none
  are RGB.

Dependency note: `matplotlib.colors.rgb_to_hsv` / `hsv_to_rgb` exist, so no new package is
needed. Have students check their own implementation against matplotlib's rather than
importing it blind.

Three demos, each mapped to a real Photoshop adjustment:

| Task | Photoshop equivalent |
|---|---|
| add 30° to H | Image > Adjustments > Hue/Saturation, Hue slider |
| set S to 0 | Image > Adjustments > Hue/Saturation, Saturation to -100 |
| keep one hue band in colour, grey the rest | the "red dress in a black-and-white film" effect |

Then go back and redo the thumbnail's darkening step in HSV. Show both results.

### Part 4 - the liver biopsy 🧀🧀🧀

Where it stops being a toy. Two slides, both verified on Wikimedia Commons 2026-08-02:

| File | Source | Size | What it shows |
|---|---|---|---|
| `hw_liver_he_cirrhosis.jpg` | Commons `Histopathology of chronic alcoholic cirrhosis.jpg`, CC0 | 2048x1532 | textbook **H&E**: pink eosinophilic cytoplasm, purple/blue hematoxylin nuclei, bright magenta red blood cells, white sinusoids, one fibrous septum running diagonally |
| `hw_liver_trichrome_nafld.jpg` | Commons `Non-alcoholic fatty liver disease1.jpg`, CC BY-SA 3.0 (Nephron) | 3460x2572 | **Masson trichrome + Verhoeff**, NAFLD: red hepatocytes, green fibrosis, white round macrovesicular fat vacuoles, purple nuclei |

Direct URLs are recorded in the "Assets to fetch" section below. Both were downloaded and
visually reviewed before this plan was written - they are not guesses.

Tasks:

1. **Blur as denoising** (H&E). A small Gaussian removes stain speckle. Push σ up until
   adjacent nuclei merge into one blob. Find the σ where it breaks. The tradeoff is the
   point: blur removes noise and signal at the same rate, and only the structure you want
   decides which σ is "right".
2. **Sobel on tissue** (H&E). Run Sobel-X and Sobel-Y separately on the same crop. The
   fibrous septum runs diagonally, so **neither alone captures it** and the magnitude
   `sqrt(Gx² + Gy²)` does. This is the cleanest concrete demonstration in the whole
   homework of why an edge detector needs both orientations - much better than asserting it.
3. **HSV nuclei selection** (H&E) - the hero task. Nuclei are purple, cytoplasm is pink.
   In RGB these are *close* (both high in R and B), so an RGB threshold either misses nuclei
   or grabs half the cytoplasm - students should try and fail at this first. In HSV the two
   separate on hue. Threshold on H (with an S floor to reject the white sinusoids, which
   have arbitrary hue at near-zero saturation), get a nuclei mask, clean it with a blur and
   re-threshold (a poor man's morphological opening).
4. **Count the nuclei** (H&E). `scipy.ndimage.label` on the cleaned mask. **scipy 1.13.1 is
   already in the `ma` venv and is on Colab**, so this needs no install. Then the honest
   part: **this undercounts**, because touching nuclei merge into one component. Make that
   a written question, not a hidden flaw. It motivates watershed / instance segmentation,
   which is exactly L19's territory.
5. **Quantify the trichrome slide** (image B). Plot the hue histogram: it has clearly
   separated modes for red hepatocytes and green fibrosis, while the white fat vacuoles fall
   out as low-saturation regardless of hue. Threshold each and report **% fibrosis area** and
   **% steatosis area**. These are real quantities pathologists grade on. Frame them as
   "the shape of the pipeline", not a diagnostic output - the numbers depend entirely on the
   thresholds students picked.
6. **The honesty question** (written answer, no code). Re-run one of your thresholds on the
   *other* slide. It fails. Why? Different stain, different protocol, different scanner,
   different white balance - every constant you tuned is tied to this one image.

That last question is the whole chapter's thesis in one exercise, and it is the documented
misconception target from `CNN_CHAPTER_PLAN.md`: students arrive thinking kernels are
hand-engineered. Here they hand-engineer a pipeline, watch it work, and watch it break.

### Part 5 - what you built vs what a network learns 🧀

Close the arc. Show the seven kernels students typed in next to resnet18's learned conv1
filters. Discussion prompt: which of your kernels did the network apparently discover on its
own, without being told to? Point to L16's learned-kernel frames and to HW1 Part B for the
version where they train it.

**Asset note (corrected 2026-08-02):** the existing `fig/pretrained_filters.pdf` is a PDF and
will not render inline in a notebook. Options, cheapest first: (a) drop the side-by-side and
make this a pure discussion cell pointing at the L17 slide, (b) have `pretrained_filters.py`
also emit a `.png`, (c) render the first-layer filters live, which pulls torchvision into an
otherwise dependency-free notebook and is not worth it. **Recommendation: (b)** - a two-line
change to the existing script, and the PNG is reusable on `cnn.qmd`.

## Assets to fetch (nothing exists yet)

Into **`ml/ch6_cnn/data/`** - not `fig/borrowed/`. Corrected 2026-08-02: `CONVENTIONS.md`
reserves `data/` for "datasets used by this chapter's notebooks", while `fig/` holds deck
figures produced by `py_src/`. These are notebook inputs, so they belong in `data/`.

```
https://upload.wikimedia.org/wikipedia/commons/c/c5/Histopathology_of_chronic_alcoholic_cirrhosis.jpg
  -> hw_liver_he_cirrhosis.jpg          (2048x1532, CC0)

https://upload.wikimedia.org/wikipedia/commons/8/83/Non-alcoholic_fatty_liver_disease1.jpg
  -> hw_liver_trichrome_nafld.jpg       (3460x2572, CC BY-SA 3.0)
```

Both need a browser-style User-Agent on `curl`; Wikimedia returns a 2 KB error page to a
bare curl UA. Downscale both to about 1400 px on the long edge before committing so the
notebook stays light and the naive convolution stays tractable; keep the full-size originals
out of git.

Thumbnail task input: `fig/src_pomegranate.jpg`, already in the repo, centre-cropped square.
Style reference for comparison: `https://img.youtube.com/vi/ydIFRGlXGy4/maxresdefault.jpg`.

No new Python packages. Verified present in the `ma` venv: numpy 1.26.4, matplotlib 3.10.0,
PIL 11.3.0, scipy 1.13.1, skimage 0.25.2, imageio 2.37.0. (`cv2` is absent and not needed.)

## L16 support frame (done)

**Built 2026-08-02, ahead of the homework** so Part 3 has a slide to call back to. This is
the only thing from this plan that exists; everything else below is still unbuilt.

- New frame **"Three numbers, but which three?"**, L16 page 8, in Section 1 between the
  RGB-channels frame and the grid-of-numbers frame. L16 went 63 -> 64 pages.
- New figure `py_src/hsv_space.py` -> `fig/hsv_space.pdf`. Top row: the astronaut portrait
  (same source image as `rgb_channels.pdf`) split into H, S and V. Bottom row: the payoff -
  original, RGB subtract, HSV V-scale, HSV hue rotation.
- **The slide's numbers are measured by the script, not asserted in prose.** Subtracting 110
  shifts hue by a mean of 20.4 degrees across the 48.3% of pixels with S > 0.2, and crushes
  30.9% of pixels to pure black.
- The script also **asserts** that scaling V is exactly a uniform RGB multiply (max gap
  2.5e-16). This matters for how the homework must be worded: HSV is not "the only way to
  darken correctly" - multiplying in RGB is the identical operation. The real point is that
  RGB leaves "darker" **ambiguous** between subtract and multiply, while in HSV brightness is
  an axis and moving V cannot do anything else. Part 3 must make that distinction, not the
  sloppier "RGB breaks colour" version.
- One adjacent line was edited by necessity: the next frame's "We just saw color as three
  stacked grids" became "RGB or HSV, color is three stacked grids".
- Verified: 2x pdflatex, 0 `!` lines, 0 overfull vboxes, page 8 rendered and eyeballed
  (the first attempt silently clipped the figure's bottom row and the whole callout box -
  fixed by dropping the figure's suptitle and tightening the frame text), aux cleaned.
- Logged in the deck's `% Provenance:` block as the **v6 revision**.

A useful side effect for Part 4: the H panel visibly goes to noise across the white
spacesuit, because hue is numerically meaningless as saturation approaches zero. That is the
exact trap the biopsy task hits with white sinusoids, so the slide can be pointed at directly
when students ask why their nuclei mask needs an S floor.

## Restructure (2026-08-02, instructor direction)

The single compact notebook was **split in two and expanded**. Standing instruction, which
overrides the "under 2 minutes end to end" target in the pitch above: *"the goal is clear
demo and understanding, not compact notebooks."* Build for teaching, not for brevity.

| Notebook | Covers | Cells | Figures | Runtime |
|---|---|---|---|---|
| `HW1b_photoshop_solution.ipynb` | engine, kernel zoo, thumbnail, HSV | 66 | 40 | 2m23s |
| `HW1c_biopsy_solution.ipynb` | the liver biopsy, end to end | 29 | 16 | 51s |

Four specific changes, all instructor-requested:

1. **Split.** HW1b builds the tools; HW1c points them at real data and watches them fail.
   HW1c repeats HW1b's helpers in one setup cell so it stands alone.
2. **2D explanations of the sliding kernel, not 1D applied twice.** New Task 0.2 walks a
   Sobel X kernel across a readable 6x6 image with a vertical edge in it, showing four
   panels per stop: the window, the kernel, their cell-by-cell product, and the output cell
   being written. Padding (zero vs edge) and stride are also shown as 2D number grids rather
   than described. Separability is now introduced as a **2D** fact - the 2D Gaussian *is* the
   outer product of two 1D ones, verified to 1e-15 and shown as three heatmaps side by side -
   rather than as a 1D trick asserted twice.
3. **No more all-at-once kernel zoo.** Each filter gets its own section, its kernel drawn as
   a labelled heatmap, and a **sweep of its own knob**: box blur at k=3/9/21/41, Gaussian at
   sigma=1/2/4/8 plus a matched-width box-vs-Gaussian comparison, motion blur across four
   angles and three lengths, unsharp mask across four amounts and three detail scales, emboss
   in four rotations, Sobel X/Y/magnitude with an orientation breakdown, and a Custom section.
4. **The thumbnail is built one layer at a time**, each step a viewable image: cover on a flat
   canvas -> add text -> add the unblurred background (visibly too busy, which is what earns
   the blur) -> blur swept at sigma 5/15/30/60 -> darkening swept at 1.0/0.75/0.55/0.35 ->
   final. The cost/separability comparison follows as a task rather than leading.

Two presentation bugs caught by eyeballing the rendered figures, both of which would have
taught the opposite of the intended lesson:

- the walkthrough's "output so far" grid printed `0` in cells that had not been computed yet,
  indistinguishable from a genuine zero result. Uncomputed cells are now blank.
- the outer-product `|difference|` panel autoscaled ~1e-16 rounding noise into a vivid
  pattern, so two provably identical arrays *looked* different. It now shares the kernel's
  colour scale and prints the max difference in its title.

## Build status (2026-08-02)

**Built, reviewed and verified:** `HW1b_photoshop_solution.ipynb` - 51 cells (26 markdown,
25 code), 14 figures, executes top to bottom on a cold kernel in **58s** with **0 errors**
(`nbconvert --execute`). The solution notebook was built first deliberately: a task notebook
full of blanks cannot be executed, so it cannot be verified, and every number in the prose
had to come from a real run.

### Content review (2026-08-02) - eight defects found and fixed

Four were genuine correctness problems, and the first is the kind that survives a green test
run indefinitely:

1. **The most important assertion was hollow.** The check that the engine reproduces the L16
   worked example ran against a throwaway `_valid_2x2` helper defined in the same cell, not
   against `convolve2d_naive`. It validated six lines written next to it and proved nothing
   about the student's function. Now calls `convolve2d_naive(I, W, padding="valid")` directly.
2. **A conclusion contradicted its own metric.** The Sobel task printed "the diagonal septum is
   under-detected by either axis alone" off a test of `|gx| < 0.5*mag` - which flags
   *axis-aligned* edges, since a perfect diagonal gives `|gx| = |gy| = 0.707*mag` and passes.
   Replaced with an orientation histogram (31% near-vertical, 38% diagonal, 31% near-horizontal)
   plus the fraction of edge energy each single axis recovers (~64% each).
3. **"Would take hours" survived in the prose** after the measurement had already corrected it
   to minutes.
4. **An unverifiable claim stated as fact**: "a nucleus is 5-10 microns; at this magnification
   that is a few hundred pixels". Neither image ships a scale bar or an objective, so the
   pixel conversion was invented. The task now names the missing metadata as the reason that
   check is unavailable - which is a better lesson than the fake number was.

Plus four consistency fixes: a docstring claiming `convolve2d_fast` asserts square kernels
(it does not - it pads `kh//2` on both axes, which is what actually breaks); a dead `cmaps`
parameter on the `show()` helper carrying a latent `titles.index()` bug; an intro promising
"you will watch all three break" when only the pathology pipeline breaks; and two absolute
timing claims hardcoded in prose, now computed.

**Cache bit a third time.** Part 0's cost model predicted ~7 min for the tap loop where the
measurement says ~3 min, and the notebook now *uses* that 2.8x miss: the model fit its own
training points well and still mispredicted, because per-tap cost depends on how much of the
image stays in cache. The lesson written into the deck is that absolute times are
order-of-magnitude only, and the trustworthy quantity is the **ratio between implementations
measured at the same size**.

**Runtime** came down from 5m30s to 58s by benchmarking one channel instead of three at
k=181 - the colour channels only repeated identical work. The k=181 comparison now lands at
**0.99x** (tap loop vs pixel loop), which is the cleanest possible statement of the point.

**Assets:** `data/liver_he_cirrhosis.jpg` (1400x1047) and `data/liver_trichrome_nafld.jpg`
(1400x1041), downscaled from the Commons originals, originals not kept.

### What measuring changed (three corrections the plan did not anticipate)

1. **The three-rung ladder is wrong at large `k`, and the truth is a better lesson.** The plan
   assumed pixel loop < tap loop < separable, monotonically. Measured at `k=181`, the tap
   loop's advantage **completely disappears**: 26.9 s vs 25.6 s on a 480x270x3 canvas, a
   ratio of **1.05x**, against roughly **100x** at the `k=3..19` sizes of Part 1. The tap loop
   sweeps the whole padded image once per tap - 32,761 passes over several MB - so it goes
   memory-bound, while the pixel loop reuses a small cached window. Separability then measures
   **100x**, slightly *beating* the 90x its tap count predicts. The notebook now teaches:
   vectorising is a constant-factor trick that stops paying as the problem grows, and only the
   algorithmic change survives. All three are **run for real** at reduced resolution - the
   earlier extrapolation-in-`k` was itself unsound, because per-tap cost depends on cache.
2. **The nuclei hue band in the plan would not have worked.** Measured, every stained pixel in
   the H&E slide sits in a narrow arc: 1st percentile 277 deg, median 309 deg, 99th 320 deg.
   A "purple vs pink" band of 0.72-0.92 covers essentially all of it and flags **24%** of the
   field. The real structure is a haematoxylin shoulder at 275-295 deg against an eosin peak
   at 305-320 deg, with the valley near 300. The notebook cuts at **295 deg with V < 0.78**,
   giving **4.8%** of the field and **90** components - a plausible nuclei mask.
3. **Instructor request, 2026-08-02: the threshold search became a task of its own** (Task
   4.3b). Rather than handing over the constants, students sweep a 5x4 grid of
   (hue, V) and read area / component count / median size out of each cell, then run a
   stability check. The teaching point is the contrast with **[08] Hyperparameter tuning**:
   the mechanics are identical, but there is **no validation set and no score to maximise**,
   so "best" is not available - only "plausible", argued from outside the image. The stability
   check earns its place: nudging `V` by 0.04 moves the count by **-67%**, while +/-5 deg of
   hue moves it 6-16%. The number students are about to report rests on a knob picked by eye.

## Task notebooks and chapter page (2026-08-02) - DONE

**Task notebooks are derived, not hand-written.** `scratchpad/build_tasks.py` reads each
solution notebook and mechanically replaces the bodies of the functions students should write,
keeping the signature, the docstring, every downstream `assert`, all markdown, and all
plotting infrastructure; then strips outputs. It **fails loudly** if a named function or block
is not found, so a rename in a solution generator cannot silently produce an un-blanked task
notebook. One source of truth: edit the solution generator, re-run it, re-run the deriver.

| File | Blanked |
|---|---|
| `HW1b_photoshop.ipynb` | 7 functions (`convolve2d_naive`, `convolve2d_fast`, `box_kernel`, `gaussian_kernel`, `gaussian_1d`, `gaussian_blur_separable`, `rgb_to_hsv_mine`) + the Custom-kernel block |
| `HW1c_biopsy.ipynb` | `nuclei_stats` + the two threshold-choice blocks (nuclei hue/V cuts, slide-B fat/collagen cuts) |

Deliberately **given, not blanked**: `_pad`, `convolve2d_rect`, `motion_blur_kernel`,
`grid_plot`, `show`, `load_image`, `to_gray`, `center_square`, `resize`, `render`, and all of
HW1c's setup cell (which states it is HW1b's code). These are infrastructure or fiddly
geometry - blanking them costs time without teaching convolution.

**`cnn.qmd` created and registered** in `_quarto.yml` at `ml/ch6_cnn/cnn.qmd`, between ch5 and
ch8. Armenian section headers per `CONVENTIONS.md`, all four deck PDFs linked plus the
`dl_cnn_conv_math` appendix, both homeworks linked in task + solution form, the chapter-level
LMU CC BY 4.0 credit line, and HW2-HW4 listed as TBD. This closes the standing gap where
**all four compiled decks were invisible on the site**.

Verified: `_quarto.yml` parses and all 80 chapter entries resolve; no broken link targets in
`cnn.qmd`; no missing blank line before a markdown block element; and
`quarto render ml/ch6_cnn/cnn.qmd` (Quarto 1.6.41) produces HTML with no errors.

## Deliverables and file layout

```
ml/ch6_cnn/
  HW1b_photoshop_PLAN.md          <- this file
  HW1b_photoshop.ipynb            <- student version: tasks, empty cells, all assertions
  NN_HW1b_solution.ipynb          <- solution; NN = playlist number, assigned at delivery
  cnn.qmd                         <- NEW chapter page (does not exist yet)
  fig/borrowed/hw_liver_*.jpg     <- the two slides
  py_src/hw1b_teaser.py           <- OPTIONAL: one figure for cnn.qmd
```

Naming follows `CONVENTIONS.md` (`NN_HWX_solution.ipynb`, `NN` = YouTube playlist number
for that homework's video). `HW1b` marks it as the second homework hanging off L16.

`cnn.qmd` is worth building here regardless of this homework: **`ml/ch6_cnn` is currently
absent from `_quarto.yml` entirely**, so all four compiled decks are invisible on the site.
The page should link the L16-L19 clean PDFs, the `dl_cnn_conv_math` appendix, both L16
homeworks, and carry the chapter-level LMU CC BY 4.0 credit line required by
`CNN_CHAPTER_PLAN.md`.

## Definition of done

- [ ] The solution notebook runs top to bottom on a cold kernel, CPU only, verified with
      `nbconvert --execute` (nbconvert 7.17.1 + ipykernel are in the `ma` venv)
- [ ] **Images resolve on Colab as well as locally.** Corrected 2026-08-02: the earlier
      "no downloads at runtime, images committed" bullet was wrong, because on Colab the repo
      is not present and a committed file is unreachable. The notebook uses a `load_image()`
      helper that prefers the local `data/` copy and otherwise downloads from the pinned
      Wikimedia URL, caching it - and **raises loudly if both paths fail**, per the repo's
      no-silent-fallback rule
- [ ] Every numeric claim in the notebook is computed in a cell, not typed in prose -
      especially the separability speedup and the area percentages
- [ ] Part 0's correctness checks are `assert`s that fail loudly, per the repo's no-silent-
      fallback rule
- [ ] Difficulty markers (🧀 / 🧀🧀 / 🧀🧀🧀) and 🎁 bonus markers set per task
- [ ] `cnn.qmd` created and registered in `_quarto.yml` with an exact-case path
- [ ] Blank line before every list, fence and blockquote in the qmd (Quarto gotcha)
- [ ] Biopsy images downscaled and committed; full-size originals not committed
- [ ] A `DECISIONS.md`-style log of anything this plan left open, written at build time

## Open questions for the instructor

1. **Name.** `HW1b` (second homework off L16) or `HW5` (fifth in the chapter)? `HW1b` says
   what it is; `HW5` keeps a flat sequence. Recommendation: `HW1b`.
2. **Thumbnail input.** Recommendation is the existing pomegranate photo, centre-cropped, so
   the chapter's running example continues and nothing new needs sourcing. Alternative: let
   students supply their own photo, which is more fun but makes the solution notebook's
   output non-reproducible.
3. **Solution notebook now or later?** The other chapter homeworks have solution notebooks
   specced but not written. Build `HW1b_photoshop.ipynb` alone first, or both at once?
4. ~~Should Part 4 use both slides or just the H&E one?~~ **Resolved 2026-08-02: use both.**
   Part 4 keeps all six tasks - the H&E slide carries denoising, Sobel orientation, HSV
   nuclei masking and counting; the trichrome slide carries the three-mode hue histogram and
   the fibrosis / steatosis area quantification. Both files are named in "Assets to fetch".
5. ~~HSV depth - should L16 gain an HSV frame?~~ **Resolved 2026-08-02: yes, and it is
   built.** See "L16 support frame (done)" below. Part 3 of this homework should open by
   calling back to it rather than re-deriving the axes.
6. **Armenian text on the thumbnail?** The title/artist text is a free choice. An Armenian
   track title fits the course's local-flavour convention, but needs a font that renders
   Armenian in matplotlib - worth 10 minutes of checking, not worth a fight.
