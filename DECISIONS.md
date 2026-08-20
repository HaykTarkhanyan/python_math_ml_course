# Decisions

Significant design choices for this repo, newest first. Each entry records what was decided, when,
why, what was rejected, and what would justify revisiting it. Superseded entries stay - the fact
that we changed our mind, and why, is the valuable part.

Deep supporting research lives in the relevant chapter's `_reference_*/` or `research/` folder;
this file holds the choice and a pointer.

---

## #24 - A colour-spaces interlude becomes deck 33, and everything after it shifts by one

**Date:** 2026-08-20 · **Status:** active

**Decision.** New deck `ml/09_clustering/33_color_spaces.tex` (22 frames), sitting between the
clustering lecture and the image practicals. It covers what a pixel stores (cones, metamerism,
RGB), gamma/sRGB encoding, HSV and the circular-hue trap, grayscale conversion and its two
competing standards, and then Lab, YCbCr and colour histograms with a "which space when" table.
The three clustering practicals moved `33_*` → `34_*`, and the dimensionality-reduction chapter
moved `34`/`35`/`36` → `35`/`36`/`37`.

**Why.** Three places in the course already assumed this material and none taught it:
`32_clustering` ends by clustering pixels "in RGB space"; the image-compression practical has a
whole section comparing RGB against naive and cone-encoded HSV; and the photo-grouping practical's
first task asks students to cluster a colour histogram. The only colour-space content in the repo
sat in `ml/ch6_cnn/L16_cnn_foundations.tex` Section 1 - roughly three weeks later in the schedule
than the practicals that depend on it.

**What was copied rather than rebuilt.** `eye_cones.py`, `rgb_channels.py` and `hsv_space.py` were
copied from `ch6_cnn/py_src/` (instructor: copy, do not move - `L16` stays self-contained), then
re-pointed from skimage's astronaut to the Saryan painting so the deck shows the same pixels the
practical clusters. Trimming `L16`'s Section 1 to a recap frame is parked in `DEFERRED_TODO.md`.

**Alternatives rejected.**
- *Number it `32b` and leave the practicals at `33`.* Instructor's call: take `33` and renumber.
  Cheap here because both chapter folders were still untracked in git.
- *Its own chapter folder `09b_color_spaces/`.* Rejected: a 22-frame interlude does not justify a
  new `_quarto.yml` entry and chapter page, and the deck exists to serve the practicals that sit
  in the same folder. Promoting it later is a one-folder move.
- *Cut YCbCr and Lab to hit the 15-20 frame interlude target.* Instructor kept all four extra
  topics and accepted 22 frames.
- *The usual "equal RGB steps look perceptually unequal" framing for Lab.* Rejected on
  measurement: a single equal step near black vs near white only moves ΔE from 10.6 to 13.4,
  because Lab's cube-root `L*` partly cancels sRGB's gamma. The deck instead quotes the measured
  spread across the whole cube - ΔE 4.4 to 16.0 at a fixed RGB distance of 20, a 3.7× range
  against a just-noticeable threshold of ~2.3.

**What would change this.** If the CNN chapter is ever taught before clustering, this deck should
move with it rather than being duplicated. If the interlude grows past ~30 frames or gets its own
practical, promote it to a real chapter.

---

## #23 - The clustering deck gets a running example, and citations for every named method

**Date:** 2026-08-20 · **Status:** active

**Decision.** Six of eleven items from a pedagogical review of `32_clustering.tex` were applied
(instructor picked the six). The two structural ones:

1. **A running example.** A toy supermarket loyalty-card dataset (`customers()` in
   `py_src/cluster_demos.py` - age, monthly spend in drams, visits, online share) now appears
   **twice**: as the scaling-trap predict-first early in the k-means section, and as the
   centroid-profiling frame near the end. Deck went 66 -> 72 pages, 47 -> 52 frames.
2. **Originating citations** (author + year) on every named method, per `ml/SLIDE_STYLE.md`.
   All ten years web-verified on the day rather than recalled.
3. **A border-vs-noise frame** (added after an instructor question the same day). The deck defined
   core/border/noise but never showed *why* two adjacent non-core points get different verdicts.
   The frame zooms on the closest such pair: a border point needs a **core** point in its
   eps-ball, and sitting beside another border point earns nothing, because reachability
   propagates only through core points. Deck ends at 53 frames / 73 pages.

**Why.** The deck taught six algorithms well but never showed what a practitioner does with the
labels afterwards, and the hook promised "customer segments" that never arrived. Profiling
centroids and naming segments is the step students will actually be paid for. Separately, the
deck's most consequential practical warning (scale first) was a text box asserting a rule the
students never saw bite; now it is a measured demonstration - ARI against the true segments is
**0.42 raw vs 0.84 standardized**, with spend's standard deviation (63,790) against visits' (7.3)
explaining exactly why.

**Alternatives rejected.**
- *A real public dataset instead of synthetic customers.* Rejected: the scaling trap needs
  features whose units differ by four orders of magnitude AND a known ground truth to score
  against. Synthetic gives both; no tidy public dataset does.
- *Adding the segmentation content as a bullet on an existing frame.* Rejected - the point is
  that reading clusters is a distinct step, and burying it would repeat the original defect.
- *Leaving the scale-first box where it was.* It had to be **reworded to pose the question**
  rather than answer it, otherwise the new predict-first frame two lines later is spoiled. This is
  the same defect flagged (and left unfixed) on the DBSCAN predict-first frame - see below.

**Also fixed, found while verifying:** the Lloyd animation frame had been **silently clipping its
own footnote** ("converges, but only to a local optimum") on all six overlay pages since the deck
was written - the figure at `0.6\textwidth` was too tall, confirmed by bisecting variants of the
pristine original from git (title and wording changed nothing; width alone did it). Figure reduced
to `0.49`. Beamer reports **zero** overfull-vbox warnings, but
`non_essential/detect_clipped_slides.py` flags it by name in seconds - it had simply never been run
on this deck, which is why the bug outlived two reviews. See
`_learnings/2026-08-20-1745_the-clipped-slide-detector-works-nobody-ran-it.md`.

**What would change this.** If the practical (`33_*`) grows its own segment-profiling section, the
deck frame could shrink to a pointer. If the instructor prefers a real dataset for the scaling
trap, the synthetic customers can be swapped out - but keep a known ground truth, or the raw-vs-
standardized comparison becomes an assertion again.

**Not applied** (the other five review items, still open): the DBSCAN predict-first still prints
its answer above the question; the elbow figure shows an unrealistically clean elbow while the
prose calls elbows "often fuzzy"; the curse-of-dimensionality plot is still a fabricated TikZ
curve `3.2/sqrt(d)` dressed as measurement (also `REVIEW.md` #9); the hook's "how many groups?"
never pays off; and the two closing frames overlap.

---

## #22 - Clustering and dim reduction join the global numbering; syllabus.csv is deleted

**Date:** 2026-08-16 · **Status:** active

**Decision.** The last two classic-ML chapters stop using the legacy `chN`/`LNN` scheme and
continue the sequence the delivered chapters already use:

| was | now |
|---|---|
| `ml/ch4_clustering/` | `ml/09_clustering/` |
| `ml/ch4b_dimensionality_reduction/` | `ml/10_dimensionality_reduction/` |
| `clustering.qmd` | `09_clustering.qmd` |
| `dim_reduction.qmd` | `10_dimensionality_reduction.qmd` |
| `L13_clustering.{tex,pdf}` | `32_clustering.{tex,pdf}` |
| `solution_image_compression / land_cover / image_clusters.ipynb` | `33_*_solution.ipynb` |
| `L13b_dimensionality_reduction.{tex,pdf}` | `34_dimensionality_reduction.{tex,pdf}` |
| `L13c_umap.{tex,pdf}` | `35_umap.{tex,pdf}` |
| `solution_eigenfaces.ipynb` | `36_eigenfaces_solution.ipynb` |

Numbers 02-31 were already taken (28 = classic methods, 29-31 = time series and its practical),
so the sequence resumes at 32 with no collision.

**Why 33 is shared by three notebooks.** The three clustering practicals are one practical
session, not three lectures, and the number tracks the video slot. This follows the existing
`21_adult_lightgbm.ipynb` / `21_adult_lightgbm_solution.ipynb` / `21_trees_project.ipynb`
precedent in ch04. If they are ever recorded as separate sessions, they need 33/34/35 and
everything downstream shifts.

**`ml/syllabus.csv` deleted** at the instructor's instruction ("i dont care about it, we just use
the 00_plan"). It had drifted badly - it still carried `L13_clustering`, `L13b_pca_dim_reduction`
and a week ordering that no longer matched delivery. Two competing schedules is worse than one.
`ml/00_plan.md` is now the single registry, and its time-series numbers were corrected from
30/31/32 to the 29/30/31 actually on disk.

**Cross-references were swept, not just the filenames.** 32 files referenced the old paths.
Student-visible callbacks in `ch8_autoencoders/L22_autoencoders.tex` said "(ch4b)"; they now say
"the dimensionality-reduction chapter" rather than "ch10", because **`ml/ch10_diffusion/` still
exists** and a bare "ch10" would be ambiguous until the remaining `chNN_` folders get the same
treatment.

**Alternatives rejected.** *Give each clustering practical its own number* - over-commits to a
schedule that has not happened. *Renumber every remaining `chNN_` folder in the same pass* - far
beyond what was asked, and the deep-learning chapters are still being written. *Keep syllabus.csv
as a historical artifact* - it reads as current, which is exactly how it caused confusion.

**What would change this.** When the deep-learning chapters are delivered, they need the same
pass, and the `ch10_diffusion` / `10_dimensionality_reduction` ambiguity should be resolved then.

---

## #21 - Project 2 assigns a diagnosis, and the diagnosis was verified before it was set

**Date:** 2026-08-16 · **Status:** active

**Decision.** `10_dimensionality_reduction` gains a second, harder homework project: run the Project 1 eigenfaces pipeline
on **LFW** instead of Olivetti, watch it collapse, find out why, and fix it. Chosen by the
instructor over two alternatives (an intrinsic-dimension/embedding-distortion measurement project,
and a CLIP semantic-image-atlas project).

Before assigning it, the entire arc was **run** (`py_src/non_essential/validate_lfw_project.py`),
because the project asks students to discover a specific causal story and it would be indefensible
to assign that story unverified. It holds, and more cleanly than expected:

| | accuracy |
|---|---|
| Olivetti, PCA(150) + 1-NN (Project 1) | 0.920 |
| LFW, identical pipeline | 0.571 |
| majority-class baseline (Bush, 530/1288) | 0.411 |
| drop first 3 PCs | 0.655 |
| standardize pixels first | 0.575 |
| Fisherfaces (PCA -> LDA) | 0.820 |
| drop 3 + LDA | 0.820 |

**corr(PC1 score, image mean brightness) = +0.998**, and PC1 carries 20.4% of total variance.
PC2-PC5 correlate at |r| < 0.05. The fattest direction in a face dataset is literally *how bright
the photograph is*. That is the deck's "PCA is unsupervised, so the discriminative direction can
sit in a low-variance component" slide, demonstrated rather than asserted.

**Why this project over the other two.** It reuses Project 1 as its own control, so the collapse is
measured against the student's own prior number rather than an abstract baseline. It also has a
real intellectual payoff (supervised DR beats unsupervised DR when the nuisance variance dominates)
instead of ending at a picture.

**Two deliberate traps, both verified.** Standardizing pixels barely helps (0.575) because it
equalizes each *pixel* across the dataset and does nothing about a per-*image* brightness offset -
students who assume the two fixes are equivalent get contradicted by their own table. And
`drop 3 + LDA` exactly equals plain LDA, because once labels are in play LDA already assigns the
brightness axis no weight; task 10 asks students to explain that.

**Alternatives rejected.** *Intrinsic dimension + embedding distortion* - quantitatively the
richest, but it never fails at anything, and "measure a number" is a weaker arc than "your model
broke, find out why". *CLIP semantic atlas* - heavy overlap with the image-clustering practical
just built on the identical data (#17). *Extend Project 1 in place* - would leave the course with
one project where the lesson deserves a contrast between a lab dataset and a real one.

**What would change this.** If the ~200 MB LFW download proves a real barrier for students, swap to
a smaller in-the-wild face set and re-run the answer key. If an sklearn change moves the numbers
materially, update the key rather than the prose - the arc, not the digits, is the assignment.

---

## #20 - UMAP gets its own lecture (35_umap), ported from the instructor's LMU deck

**Date:** 2026-08-16 · **Status:** active

**Decision.** `10_dimensionality_reduction` becomes a two-lecture chapter. `34_dimensionality_reduction` keeps answering *what DR gives you and
when to use which method*; the new **`35_umap.tex`** (29 pages) answers *how UMAP actually
works*. It is a port of the instructor's own LMU student-assistant deck, copied to
`_reference_umap_lmu/` and rebuilt in course style.

**Why.** The course deck gave UMAP a single frame naming two hyperparameters, which is thin for the
algorithm students will use most. The LMU deck already covered the mechanism properly. The
instructor chose the full port over "figures only" and "grow to ~5 frames".

**Port decisions.**

- **Overlap compressed, not repeated.** Four LMU frames (motivation/curse, PCA recap, t-SNE recap,
  the three-way comparison table) are covered in more depth by 34_dimensionality_reduction, delivered immediately before.
  They collapse into one "Where we left off" bridge frame that keeps only the genuinely new claim:
  `KL(P||Q)` punishes tearing neighbours apart but not collapsing distant points together, so
  t-SNE's objective leaves global structure unprotected. That claim then pays off on the loss
  frame, where UMAP's cross-entropy is split into its attraction and repulsion halves.
- **The toy example runs at k=3, not k=2.** The LMU deck states `sigma_A ~ 0.4`. That is not what
  the binary search returns: at `k=2` the nearest neighbour alone contributes exactly
  `log2(2) = 1`, so the target is already met and `sigma -> 0` (verified: `sigma=0.01` gives a sum
  of 1.0035). At `k=3` the target 1.585 gives a genuine `sigma_A = 0.1532`.
- **The triangle apexes moved off-centre** (0.45/5.45, not 0.50/5.50). A centred apex puts two
  neighbours at exactly `rho`, pinning the sum at >= 2 against a target of 1.585 - also unsolvable.
  Found because the figure script *raises* on non-convergence instead of returning a fallback.
- **Hand-drawn hyperparameter grids became real runs.** The LMU deck sketched `n_neighbors` and
  `min_dist` effects as scattered TikZ dots; `py_src/umap_demos.py` now runs UMAP at those actual
  settings on Fashion-MNIST. Per `ml/SLIDE_STYLE.md`, and it is also just more honest. TikZ is kept
  only for the manifold sketch, the directed-edge pair and springs-and-magnets.
- **Every toy number on the slides is printed by the figure script** to `logs/umap_demos.log`, so
  the slides cannot drift from the math.

**Alternatives rejected.** *Figures only* - cheapest, but leaves the mechanism untaught. *Grow to
~5 frames* - the balanced option, rejected by the instructor in favour of depth. *Keep the LMU
recap frames for standalone use* - costs ten minutes re-teaching material from the previous slot.

**What would change this.** If the two lectures end up delivered weeks apart rather than back to
back, restore the compressed recap frames so 35_umap stands alone.

---

## #19 - 10_dimensionality_reduction's running dataset moves from 8x8 digits to Fashion-MNIST

**Date:** 2026-08-16 · **Status:** active

**Decision.** The dimensionality-reduction deck's running dataset changes from sklearn's 8x8
digits (64-D) to **Fashion-MNIST** (28x28, 784-D), cached as a committed 12,000-image stratified
subsample (`data/fashion_mnist.npz`, 5.3 MB, built by `py_src/fetch_fashion_mnist.py`). A single
extra frame uses the **CLIP embeddings already committed for 09_clustering** (#17) to show DR on a
real 512-D embedding space, drawn with the actual photographs.

**Why.** Instructor: "8x8 already looks quite terrible." It is not one bad figure - the digits were
the spine of the whole deck (hook, scree, reconstruction, denoising, t-SNE, UMAP, comparison). The
reconstruction frame was the worst case: at 8x8, `k=5` and `k=50` are both grey mush, so the
compression lesson was a claim rather than a demonstration. At 28x28 an ankle boot is unmistakable
at `k=50` and unrecognizable at `k=5`.

The numbers also teach better. Regenerated and verified:

- PC1 = **29.0%**, PC1+PC2 = **46.8%** (digits: 14.9% / 28.5%)
- 95% variance needs **184 of 784** components - a far stronger predict-first than "29 of 64",
  because students reliably guess "two or three"
- reconstruction at k = 5 / 20 / 50 keeps 61.7% / 78.6% / 86.3% of variance

**Also swept `REVIEW.md` (2026-07-07), which had never been applied.** All ten items addressed
except #10. Notably its item 1 - the deck stated "first PC ~12%, first two ~22%" while its own
scree figure showed 0.148 - was still live in the `.tex` thirteen months later. Item 10 (a 2x2
characteristic-polynomial worked example) is deliberately not done: the by-hand PCA frame already
carries real covariance and eigenvalue numbers, and a determinant derivation would push a
37-page deck longer for mechanics the linear-algebra course covers.

**Alternatives rejected.** *MNIST-784* - same resolution win, but retells the digits story and
wastes the second domain. *Olivetti faces* - the most dramatic reconstruction, but it is the
Project 1 dataset, and the review specifically praised deck and homework using different data.
*CLIP embeddings as the main dataset* - the most modern framing, but embeddings cannot carry the
compression/denoising half of the deck, since there is no image to rebuild. Hence: Fashion-MNIST
throughout, CLIP for exactly one frame.

**What would change this.** If the 5.3 MB committed npz becomes a problem, drop `PER_CLASS` in
`fetch_fashion_mnist.py` - PCA's explained-variance numbers are stable well below 12,000 samples,
but the slides quote them, so regenerate the figures and the `.tex` numbers together.

---

## #18 - ch20 is a deliberate retelling of one video, but its central experiment is re-run here

**Date:** 2026-08-14 · **Status:** active

**Decision.** `ml/ch20_subliminal_learning` (deck `L48`) follows Welch Labs' *These Numbers Can
Make AI Dangerous* beat for beat, at the instructor's request ("basically retell the video, don't
add too much"). The scope rule written into the chapter plan is: **if it is not in the video, it
needs a reason to be here.** Exactly three things were added, all of them corrections or
verifications rather than new material:

1. The MNIST experiment is **measured on this machine**, not quoted. `py_src/subliminal_mnist.py`
   writes `results/subliminal_mnist.json`; every figure derives from that file.
2. A **different-initialisation control**, which the video only implies. It is the falsifiable half
   of the argument and the thing that makes the GPT-4.1/GPT-4o anomaly land.
3. The token-entanglement source is corrected: it is a **blog post**, not an arXiv paper, and its
   mechanism (the softmax bottleneck) is named. The video says neither.

**Why re-run it.** Same reason as ch19: a measured number the instructor can defend beats a quoted
one, and it costs about a CPU-minute. It paid for itself immediately - the guard asserting the
auxiliary head receives zero gradient returned **exactly `0.000e+00`**, which is the single claim
the whole lecture rests on, and the theorem check returned **0/200 negative cosines with a shared
init against 91/200 without one**.

**The uncomfortable part, recorded rather than hidden.** The paper's headline (>50% MNIST accuracy)
and its most striking variant (distilling on *pure noise*) **did not reproduce at this scale**. We
get 10.0% -> 20.4% with a shared init against 11.6% -> 13.7% for the control, and only 14.4% on
noise. The paper does not publish the learning rate or schedule. The deck carries a frame saying
exactly this rather than quoting a number we did not obtain.

**Alternatives rejected.** *Quote the paper's 50% and show no run of our own* - cheapest, and it is
what the video does, but it gives up the one thing this repo can add. *Keep tuning until we hit
50%* - unbounded search against unpublished hyperparameters, on a laptop, for a number that is not
load-bearing; the qualitative effect and its dependence on shared initialisation are what the
lecture actually needs. *Drop the MNIST section* - it is the bridge between the language-model
result and the proof, and removing it would leave the algebra unmotivated.

**What would change this.** If the paper's code or hyperparameters become available and a short run
reproduces >50%, replace the measurements and delete the caveat frame. If a student review finds
the "what did not reproduce" frame reads as a failure rather than as method, reframe it - but do
not remove it.

---

## #17 - A third clustering practical uses CLIP as a black box, in ch4 rather than later

**Date:** 2026-08-14 · **Status:** active

**Decision.** `09_clustering` gains `33_image_clusters_solution.ipynb`: 2000 Imagenette photos encoded
with **CLIP ViT-B/32**, clustered with k-means, displayed as a self-contained interactive HTML map
with thumbnail-on-hover. The encoder is **explicitly a black box** at this point in the course, with
a stated promise that chapters 6 and 9 explain it. Embeddings, thumbnails, labels and 40 text
vectors are precomputed into `data/imagenette_clip.npz` (3.2 MB) by
`py_src/embed_images_clip.py`; the student notebook needs numpy, sklearn and plotly only.

**Why.** It supplies the result the chapter otherwise lacks. Image compression has no labels, and
Sevan scored ARI 0.48 that collapsed to 0.16 once the lake was removed. Here the *same* k-means on
the *same* photos scores **0.048 on raw pixels and 0.939 on CLIP embeddings** - a twentyfold
difference from the representation alone. That measurement is the chapter's thesis, and nothing
else in it states the case as sharply.

CLIP specifically, over DINOv2 or a small CNN, because text shares the embedding space: each
cluster **names itself** by finding the nearest of 40 candidate English words, and it got 10 out
of 10 right with no labels involved. That turns the manual naming step from the Sevan practical
into an automatic one, and the runner-up words (church/clock, golf ball/parachute) are the
clearest available picture of what "distance" means in a learned space.

**Alternatives rejected.** *Wait for ch6 or ch9, where the encoder could be explained* - it would
strand the clustering chapter without this result for two months, and using a pretrained encoder
as a component is exactly how it is done in practice. *DINOv2* - slightly better pure-vision
features and a smaller download, but no text tower, so no self-naming. *No deep learning
(histograms, HOG)* - that is the baseline the notebook uses to demonstrate failure, not a
substitute for the payoff. *An unlabelled photo album* - loses ARI, and the chapter's evaluation
thread is what ties the three practicals together.

**What would change this.** If ch6 or ch9 later wants an image-embedding practical of its own,
this one should be checked for overlap rather than duplicated. If the CLIP download becomes a
problem for students, note that they never need it - only the instructor re-running the script does.

---

## #16 - The land-cover practical ships a committed 20 m npz, not a live data pull

**Date:** 2026-08-13 · **Status:** active

**Decision.** `ml/09_clustering/py_src/fetch_sevan_scene.py` is **instructor-side and run once**.
It queries Earth Search for a cloud-free Sentinel-2 L2A scene, crops a 1000x1000 window at
**20 m**, reprojects ESA WorldCover onto the same grid with nearest neighbour, and writes
`data/sevan_s2_crop.npz` (9.2 MB, committed). The student notebook opens that file with plain
`np.load` and needs **numpy, sklearn and matplotlib only**.

**Why.** The `ma` venv had **no geospatial stack at all** - `rasterio`, `rioxarray`,
`pystac-client` and `geopandas` were all missing, and installing them is a GDAL-shaped dependency
chain on every student machine an hour before class. A practical that can fail at `import
rasterio` has a failure mode unrelated to anything being taught. The npz also removes the network
from the critical path: the session works with the wifi down.

20 m rather than the native 10 m because at 10 m a 20 km square is 4M pixels and ~50 MB in git.
At 20 m the scene covers the same ground for 9.2 MB, k-means on the full cube runs in ~10 s on
the laptop, and **B11/B12 arrive at their native resolution** instead of being upsampled - and
those two bands are what separate bare soil from built-up from dry grass, so the trade buys
accuracy rather than costing it.

**Alternatives rejected.** *Students query the STAC API themselves* - teaches real data
acquisition and lets them pick their own region, but costs a geospatial install per machine,
class time on setup, and a hard network dependency; kept as a homework bonus instead, since the
fetch script is in the repo. *Copernicus Data Space* - needs an account; the AWS
`sentinel-2-l2a-cogs` bucket is free, unauthenticated and not requester-pays. *10 m with a
smaller footprint* - a 10 km square loses the steppe and most of the class variety.

**What would change this.** A repo-wide geospatial stack arriving for some other chapter, which
would make the live-fetch version nearly free. Or the crop needing to change often, which would
make a committed binary the wrong place to keep it.

---

## #15 - Clustering gets a second practical: land cover, not a second image task

**Date:** 2026-08-13 · **Status:** active

**Decision.** `09_clustering` gains a second practical, **unsupervised land-cover mapping of
Lake Sevan** (`33_land_cover_solution.ipynb`), alongside the existing k-means image-compression
project. Design in `33_land_cover_OUTLINE.md`. Not yet scheduled - `00_plan.md` still
shows Aug 21 as the image-compression slot, and which one takes it is an open call.

**Why.** Image compression exercises roughly a quarter of a 47-frame deck: k-means, mini-batch,
elbow, silhouette. The rest is unreachable *by construction*, not by omission - RGB pixels have no
labels, so the deck's entire external-evaluation section (ARI, AMI, the label-permutation
problem) cannot be practised, and quantization never asks what a cluster *is*, because the
clusters are colours about to be thrown away. Land cover reaches all of it: naming clusters from
mean spectra, GMM soft assignment with a physical meaning, DBSCAN failing for a stateable reason,
and ARI/AMI against ESA WorldCover.

**Alternatives rejected.** *Gaia star clusters with HDBSCAN* - the strongest fit for the deck
(95 % of points are correctly noise, which k-means cannot express) and a genuinely current
published method, but the instructor chose the locally-grounded option. *NBA hidden positions* -
best story, weakest visual payoff. *A tabular customer-segmentation exercise* - covers the same
concepts with none of the visual result. *Extending the image practical instead* - would not have
produced labels, which is the whole point.

**What would change this.** If the schedule can only fit one clustering practical, this one
covers strictly more of the deck than image compression and should take the slot; the choice is
then which to demote to homework-only.

---

## #14 - Superposition and SAEs stay in ch8; ch19 gets them as a callback only

**Date:** 2026-08-13 · **Status:** active

**Decision.** The new mechanistic-interpretability chapter (`ml/ch19_mech_interp`) **does not
re-teach** superposition, polysemanticity, dictionary learning, L0, feature splitting, dead
features, or the SAE objective. All of it stays where it already is - `ml/ch8_autoencoders/L22`,
section *"Sparse autoencoders and interpretability"*, plus the `HW1_sae_rnn.ipynb` lab where the
students implement one. ch19's L47 opens by naming the callback (*"you built one of these"*) and
spends its frames only on what is new: **using** a pretrained SAE at LLM scale, transcoders,
attribution graphs, and steering. Working rule written into the chapter plan: **if an L47 frame
duplicates an L22 frame, cut the L47 one.**

**Why.** The overlap was discovered while outlining, not after building. `L22` already covers
superposition, the decoder-columns dictionary picture, L0, ablation-as-evidence, Golden Gate
Claude, the *"SAEs Do Not Find Canonical Units of Analysis"* caveat, **and** a closing frame that
forward-points to attribution graphs. A chapter arriving ~2 months later and re-deriving that
material would spend roughly a third of a session telling students something they already
implemented in homework. Framing L47 as the delivery of a promise L22 already made is both
cheaper and a better story.

**Alternatives rejected.** *Move the SAE material out of ch8 and into ch19* - it belongs in the
autoencoder chapter pedagogically (an SAE **is** an autoencoder, and that is the cleanest moment
to teach it), and ch8 is already delivered, so moving it would strand the existing homework.
*Re-teach it briefly in ch19 for students who missed ch8* - this is what produces the duplicate
half-explanations the acronym rule already fights; signpost instead.

**What would change this.** ch8 being cut or restructured, or the SAE section moving out of L22.
Either would leave ch19's L47 standing on a callback to nothing, and section 1 would have to grow
from three recap frames into a real treatment.

---

## #13 - ch19 deck numbers are L45-L47, build order, not delivery order

**Date:** 2026-08-13 · **Status:** active

**Decision.** The mech-interp decks are numbered **L45, L46, L47** - continuing from L44 (agents)
- even though the chapter is scheduled for delivery in mid-October, directly after **L26**
(transformers). The L-number is a **build-order identifier**. It does not encode when a deck is
taught.

**Why.** The correspondence between L-number and delivery order was already broken before this
chapter existed: L37 (tabular FM) and L38 (VLA) are swapped relative to the schedule, and the
whole L41-L44 range (RAG, agents) does not appear in `ml/00_plan.md`'s schedule table at all.
Numbering this chapter by delivery order would fix the ordering for one chapter while making the
global inconsistency harder to reason about, because two conflicting conventions would then be in
use simultaneously.

**Alternatives rejected.** *`L26b/c/d`* - preserves delivery order locally, but wedges three decks
into a gap and implies they are sub-parts of the transformer chapter, which they are not.
*Renumber the whole DL half by delivery order* - correct in principle, but it would rewrite
filenames, `_quarto.yml` paths, and YouTube playlist numbers across ~20 delivered decks, and the
YouTube numbers are already published.

**What would change this.** A general renumbering of the deep-learning half, which
`DEFERRED_TODO.md` already carries as a housekeeping item. If that happens, this chapter is
renumbered with everything else, not before.

---

## #12 - L13's agglomerative animation uses Ward, not centroid linkage

**Date:** 2026-08-09 · **Status:** active

**Decision.** `fig_agglo_anim` in `ml/09_clustering/py_src/cluster_demos.py` builds its
dendrogram with **Ward** linkage. The frame's story changes from "merge the two clusters whose
midpoints are closest" to "merge the two clusters that cost the least extra spread."

**Why.** Centroid linkage is the one linkage that can produce non-monotonic merges, and on these
seven toy points it did. The merge distances ran `0.58 0.64 0.65 0.67 2.62 2.50` - the **root
merge (2.50) sat below its own child (2.62)**, so the red root bar was drawn *underneath* the
black bar it was supposed to span. That directly contradicts the next frame, which teaches
"the largest vertical gap is a natural place to cut," a rule that only holds if merge height
never falls. Ward gives `0.58 0.64 0.67 0.75 3.70 4.63` - strictly increasing. It also matches
the deck's own claim two frames later that Ward is "the common default," and `fig_dendrogram`
already used Ward, so the two dendrograms in the deck were previously built with different
linkages.

**Cost accepted.** The "distance between the midpoints you can see on the left" reading is gone;
Ward's merge cost is not a distance between the two `x` markers. The markers still show where
each cluster sits and the dashed line still shows which two merge, so the visual survives, but
the number in the title is no longer something the student can measure off the scatter.

**Alternatives rejected.** *Nudging the seven toy points* so centroid linkage happens to stay
monotonic - keeps the midpoint story, but leaves an inversion-capable linkage in a figure that
teaches monotonicity, so it fixes the symptom on this data only. *Leaving it and adding a
warning* - turns a bug into a caveat, but the next frame still teaches a rule the picture breaks.

**Guard added.** The generator now asserts `np.all(np.diff(Z[:, 2]) >= 0)` and raises if the
linkage ever inverts again, so this cannot come back silently.

**What would change this.** Wanting the midpoint-distance reading back badly enough to redesign
the toy points around single linkage (also monotonic) instead.

---

## #11 - ch16 borrows 33 full-bleed video stills, at a density that is deliberately high

**Date:** 2026-08-08 · **Status:** active

**Decision.** Stills pulled from the Welch Labs LeCun documentary go into L39/L40 as
**full-bleed frames with no caption**, attributed with a small corner node, at roughly **one page
in three**. They are never redrawn into house style.

**Why.** The instructor's framing settled it: *"I'm fine with guided screening, the important
thing is student content, not author."* I had argued the opposite - that a deck cutting to
someone else's artwork every third slide stops feeling like ours - and that objection was
correctly overruled. Optimising for the deck feeling ours is a worse objective than the room
understanding JEPA, and the borrowed architecture diagrams are better than what I would draw.

**The line that was held.** Borrowed stills carry **architecture and narrative**; Python carries
**every number and every measurement**. A still is someone else's explanation; the I-JEPA
ablation bars are our evidence. Three planned figures were cut because a still did the job
better; the three-panel architecture figure was *kept despite* having a still, because the deck
points back to it five times and it must be in our visual language.

**Alternatives rejected.** Redrawing everything in house style (weeks of figure work, worse
diagrams); ~16 stills, architecture only (my recommendation - overruled); captions under each
still (breaks full-bleed, and the surrounding frames already carry the argument).

**What would change this.** A rendered-slide review reporting that the deck reads as a screening
rather than a lecture, or a licensing situation that makes 33 borrowed frames untenable.

---

## #10 - ch16 (JEPA) ships as two decks, explanatory only, and refuses to pick a side

**Date:** 2026-08-08 · **Status:** active

**Decision.** New chapter `ml/ch16_jepa/`, decks **L39** (the objective) and **L40** (world
models), registered after `ch15_vla`. **No model is trained; every plotted number is transcribed
from a paper.** No student project. The chapter is written to leave the LeCun-versus-LLMs
question **open**.

**Why.** The chapter's subject is the *objective*, not an architecture - the question the course
had been answering implicitly for fifteen chapters without asking: what should a model be asked
to predict? The evidence is unusually clean, from I-JEPA Table 7: same architecture, same
masking, **66.9** predicting representations vs **40.7** predicting pixels, with the pixel run
getting **60% more** training.

Two decks because there is a real conceptual boundary: L39 has no time axis at all (I-JEPA works
on one still image), and time plus actions is what turns the objective into a world model.

**Explanatory only** follows the ch14 precedent. One exception was proposed - a toy collapse
demo, seconds of CPU, showing embedding variance going to zero while the loss looks excellent -
and the instructor deferred it. L39 frame 15 now argues it in words instead, and the plan is
written so the figure can drop in as frame 15b without renumbering anything.

**Refusing to pick a side is load-bearing, not politeness.** The chapter's own scoreboard is two
green, two red: good efficient encoders (supported), planning with a video world model
(supported, slowly), intuitive physics (**not** supported - IntPhys 2), replacing next-token
prediction (**not** demonstrated). Both student reviewers independently reached that verdict
before the deck stated it.

**One figure refuses to draw a number.** `physics_gap.pdf` plots the human range as a **shaded
85-95% band** rather than three bars, because Meta reports it as a band with no per-benchmark
figures, and model performance only qualitatively as "at or near chance". Inventing three human
bars would have made the one chart intended to keep the chapter honest the only one built on
fabricated numbers. The limitation is printed on the figure itself.

**Alternatives rejected.** One combined deck (drops either the ablations or the planning loop -
and after the L40 restructure it would orphan the hierarchy section that answers the chapter's
sharpest criticism); running the collapse experiment (deferred, in `DEFERRED_TODO.md`); the
LeWorldModel term project (deferred - it is the only reproducible model in the chapter, single
GPU and a few hours, so it will be worth revisiting); deriving energy-based models properly
(that is its own lecture - stated with one picture instead, per the ch11/ch12/ch14 precedent).

**What would change this.** V-JEPA-style physical reasoning clearing the IntPhys 2 gap would make
the "not supported" row wrong and require rewriting L40's close. A JEPA-based language model
competitive with frontier LLMs would do the same to the fourth row. Either would be a reason to
revisit, not a reason to soften the current text.

---

## #9 - ch12 (vision-language models) ships as two decks with figures only, no trained model

**Date:** 2026-08-07 · **Status:** active

**Decision.** New chapter `ml/ch12_vlm/`, decks **L33** (how a model sees) and **L34** (how a
model draws), registered between `ch11_rl` and `llm_training`. **Intuition-first**, not full
derivations. **No neural network is trained anywhere in the chapter.**

**Why.** The course could explain transformers (ch9), autoencoders (ch8) and diffusion (ch10)
but not how a chat model reads a pasted photo - the single most visible AI capability to a
non-specialist, and the natural convergence point of four earlier chapters. Two decks because
"seeing" is a settled engineering recipe while "drawing" is an open architectural argument;
that is a real conceptual boundary, not an arbitrary split at 47 frames.

Intuition-first is a **deliberate deviation from `ml/SLIDE_STYLE.md`** (which asks for full
step-by-step derivations), matching the precedent set by `ch11_rl`. It shows in exactly two
places: the contrastive loss and the VQ straight-through estimator, both described in words.

**The chapter still measures something.** `fig/vq_quantization.pdf` fits a k-means codebook on
ch10's 4,481 letters, which is clustering rather than network training (~8 s):

| Codebook K | 8 | 32 | 128 | 512 |
|---|---|---|---|---|
| Reconstruction MSE | 0.01819 | 0.01064 | 0.00844 | **0.00654** |

Two results are taught from it: sharply diminishing returns (128 -> 512 buys 22% for 4x the
vocabulary), and visible stroke breakup even at K=512 because each patch is quantized with no
knowledge of its neighbours. The second is **#8's finding again** - 1-2 px strokes are what
every compression scheme destroys first.

**Alternatives rejected.**
- *Train a VQ-VAE plus an autoregressive generator on the ch10 letters and race it against the
  diffusion model* - the strongest idea in the plan, and cut by the instructor as too much
  build. The chapter now cites published comparisons instead of running its own.
- *A real CLIP zero-shot run on the Armenian letters* - cut. It needs a ~350 MB download and a
  new dependency (`open_clip` or `transformers`), which is a dependency choice that was not on
  the table. Parked in `DEFERRED_TODO.md`.
- *One long deck* (the ch11 shape) and *three decks* - rejected for the boundary reason above.

**What would change this.** If the chapter gets a homework slot, the cut AR-vs-diffusion
project is the obvious candidate and would give the chapter a project matching ch10's and
ch11's. If GPT-4o's architecture is ever published, the L34 "known vs inferred" frame needs
rewriting - it is currently the one frame in the chapter that could teach something false.

---

## #8 - The ՊԱՆԻՐ denoiser is a ONE-level UNet at ch=96; #7's two-level design was the bug

**Date:** 2026-08-07 · **Status:** active · **supersedes #7**

**Decision.** `LEVELS = 1` (24 -> 12 -> 24, a single halving) at `ch=96`, **1.50M params**,
10000 steps. Trained on a rented T4 via the Colab CLI, not locally.

**Why.** #7 assumed capacity was the constraint and went from 266k to 7.03M params. It was
wrong, and the measurement is unambiguous:

| arch | params | steps | final loss | samples |
|---|---|---|---|---|
| 2 levels, 24x24 | 7.03M | 20000 | 0.038 | fragments |
| 2 levels, 32x32 | 7.03M | 10000 | **0.9994** | diverged |
| **1 level, 24x24** | **1.50M** | **10000** | **0.0269** | **legible letters** |

A **4.7x smaller** model produced the best loss of any run and the first readable ՊԱՆԻՐ.
The cause is the same property that made crop-to-ink mandatory in #6: **the strokes are 1-2 px
wide.** Halving twice (24 -> 12 -> 6) leaves them sub-pixel in the deep layers, so the extra
capacity models a representation from which the letter has already been erased. Halving once
keeps them. The 20000-step run also plateaued by ~step 4000, ruling out training length.

**Alternatives rejected.**
- *More capacity* (#7's answer). Falsified above.
- *More steps.* The progression figure shows no change from step 4000 to 20000.
- *32x32.* Not rejected - **untested**. That run diverged (loss 0.9994 = predicting zero)
  because `lr=2e-3` is too hot for 7.03M params; the same run at 24x24 had already shown a
  27.32 loss spike over its first 250 steps. Retest with a lower LR before concluding anything.

**Cost accepted.** The UNet is now built from `ModuleList`s with `levels` as a parameter, so
**state-dict keys changed** and every checkpoint predating this entry is unloadable. Given all of
them produced unusable samples, nothing of value was lost.

**What would change this.** If a 32x32 run at a lower LR beats this, revisit - more pixels is the
other way to stop downsampling from destroying strokes. `pack_mashtots.py` takes a size argument
and `TAG` keeps experiment artifacts apart, so that test is ~10 min on a T4.

---

## #7 - The ՊԱՆԻՐ denoiser is a two-level UNet at ch=64, not digits_ddpm's TinyUNet

**Date:** 2026-08-06 · **Status:** active, **outcome pending** (6000-step run in flight)

**Decision.** `train_panir_ddpm.py` uses its own **two-level** conditional UNet
(24 -> 12 -> 6 -> 12 -> 24, skips at both scales, conditioning injected at all three),
**ch=64, 3.13M params**, rather than reusing `digits_ddpm.py`'s TinyUNet.

**Why.** The first full run *did* converge - loss 1.2 -> 0.0344 - but the samples were
malformed and **Ի effectively failed to render** (per-class ink 0.059 against 0.099-0.123 for
the others; 0.024 in the generated word). Loss went flat at **step ~800** and the remaining
5,200 steps bought 0.005. Flat loss plus bad samples is a capacity limit, not undertraining,
and TinyUNet is 266k params with a single down/up level - built for 8x8 digits, not 24x24
cursive across 5 classes. Notably the *thinnest* input class (ink 0.131 vs 0.16-0.21) became
the failed output class.

**Why ch=64 specifically.** Measured at 4 threads: **ch=48 -> 1274 ms/step, ch=64 -> 1288,
ch=96 -> 3245.** ch=64 buys 1.8x the parameters of ch=48 for ~1% more time - the step is
memory-bound at this size, so the capacity is nearly free - while ch=96 costs 2.5x.

**Alternatives rejected.**
- *More steps on TinyUNet.* The loss curve was flat for 5,200 steps. Nothing there to gain.
- *Drop to 16x16*, which is what #6 prescribed for trouble. Rejected because resolution was not
  what bound the first run; the same architecture would simply fail faster.
- *ch=96.* 5.4 h per run for capacity this dataset almost certainly does not need.

**Cost accepted.** The step-timing probe (a tight loop over one cached batch) predicted 1288 ms;
the real loop runs at **3.86 s/step**, so a 6000-step run is ~6.4 h rather than ~2 h. The probe
did not model per-step data indexing or memory pressure and should not be trusted for future
estimates without a real-loop check. The run was left at BelowNormal priority regardless, per the
freeze-safety rule in `diffusion_lib.py:23`.

**What would change this.** If the letters are still malformed after this run, capacity is *not*
the binding constraint and the next suspects are the data volume (~900 images/class) and the
per-glyph size normalization from #6 - not a still-larger model.

---

## #6 - The diffusion homework trains on five Armenian letters at 24x24, vendored as one .npz

**Date:** 2026-08-05 · **Status:** active

**Decision.** `ml/ch10_diffusion` gets a homework after all (reversing the "lectures only" call in
`DIFFUSION_CHAPTER_PLAN.md`), built on **five** classes of the Kaggle *Mashtots Dataset v2* -
**Պ Ա Ն Ի Ր**, which spell **ՊԱՆԻՐ** - preprocessed to **24x24** and committed as a single
**1.25 MB `.npz`** (`data/mashtots_panir_24.npz`). Students never touch Kaggle.

**Why.**
- *Five letters, not 78.* The word is the payoff: generate each letter class-conditionally, paste
  them side by side, and the result is visibly wrong because every letter comes from a different
  hand. That failure *is* the lesson about global coherence. ՊԱՆԻՐ also happens to be this course's
  difficulty unit. Five classes give ~4,481 images, against `digits_ddpm.py`'s 1,797.
- *24x24.* Measured, throttled to 4 threads on a loaded machine: **16x16 = 22.5 min/run,
  24x24 = 35.2 min, 32x32 = 109.1 min** for 6,000 steps. 32x32 is 3.1x the time of 24x24 for 1.8x
  the pixels - superlinear, so it is disqualified. 16 -> 24 costs only 1.56x and the glyphs are
  visibly better (`mashtots_letters.html` shows both).
- *Crop to the ink box before resizing.* Not an optimization - required. The glyph fills only
  ~34-40 px of the 64 px frame, so a naive resize applies a 4x reduction to 1-2 px strokes:
  ink fraction **0.133 vs 0.258** at 16x16, peak brightness 134 vs 154. A font-rendered probe missed
  this entirely because font strokes are 5-8x thicker than this handwriting.
- *Vendored `.npz`.* The source is a **competition**, so raw access needs an account, an API token
  and accepting the rules. Every other dataset in this course is a one-liner.

**Alternatives rejected.**
- *All 78 classes.* ~900 images/class either way, but 78-way conditioning on a CPU budget buys
  nothing the word demo needs.
- *64x64 native.* Hours per run. The chapter's own `digits_ddpm.py` docstring already made this call
  for MNIST, though note its "hours" figure is for 60,000 images, not our 4,481.
- *Pretrained Stable Diffusion via `diffusers`.* Teaches none of L27-L30 and is minutes per image on
  an Iris Xe. `diffusers` is not even installed.
- *Font-rendered letters (Sylfaen + augmentation).* Zero download and fully reproducible, but real
  handwriting is the better story and makes the per-writer inconsistency genuine. Kept as a fallback.

**Cost accepted.** Per-glyph cropping normalizes every letter to the same size, discarding the
natural ~3x size spread (19-57 px), so the model cannot generate size variation. Stroke weight and
slant survive, which is enough for the inconsistency lesson.

**What would change this.** If a training run at 24x24 fails to converge in ~35 minutes, drop to
16x16 rather than adding steps. If the letters Ի and Ր turn out to be confusable at 24x24 (they are
near-twins in cursive), swap one and re-pack - `extract_mashtots.py` and `pack_mashtots.py` are
parameterized by a single `LETTERS` list and the raw zip is kept.

---

## #5 - GANs get two decks in the generative thread, not a chapter after diffusion

**Date:** 2026-08-03 · **Status:** active

**Decision.** New chapter `ml/ch8b_gans/` with **L23b** (the adversarial game) and **L23c**
(applications and evaluation), delivered between L23 (VAE) and L24 (attention). The generative
thread now runs **L22 -> L23 -> L23b/L23c -> L27-L31**.

**Why.** GANs were referenced by three delivered decks and taught by none: L19 shows StyleGAN faces,
L23's comparison table calls GANs "unstable", and L28 had to teach mode collapse from scratch so its
"diffusion is just an MSE" argument would land. Verified by grep: "generative adversarial" appeared
in **zero** built decks. Placing the material *before* diffusion converts L28's improvised teaching
into a genuine callback, which has now been done.

**Alternatives rejected.**
- *L32, a chapter after diffusion.* Cleanest numbering, no suffixes. Rejected because the diffusion
  chapter spends five lectures comparing against a model students would not yet have met.
- *Fold into `ch8_autoencoders` as L23b.* Least churn, and the VAE-vs-GAN table already lives there.
  Rejected because the folder name would stop describing its contents.

**Cost accepted.** A `b`/`c` suffix in the lecture numbering, following the existing `L13b`
precedent. Renumbering L24 onward was never considered - it would break every cross-reference in
four chapters.

**What would change this.** If the generative material is ever reorganized into one large chapter,
these two decks and `ch10_diffusion` should merge rather than stay adjacent.

---

## #4 - L30 builds cross-attention itself; the merge-to-4-decks fallback is withdrawn

**Date:** 2026-08-03 · **Status:** active

**Decision.** (a) The diffusion chapter does **not** depend on `ch9_attention`'s unwritten decks:
L30 builds cross-attention in one frame from the Q/K/V material L24 already teaches. (b) The
"merge L27 into L28 to get back to four decks" fallback recorded in #3 is withdrawn.

**Why.** An adversarial review of the chapter plan checked both claims against the repo and both
failed. `L24_attention.tex` contains zero occurrences of "cross-atten", and **L25 and L26 do not
exist** - `ml/ch9_attention/` holds one deck of a planned three. So the original "the only place the
transformer chapter is load-bearing" note pointed at material nobody has written. Building the
concept locally costs one frame and removes the ordering constraint between two chapters entirely.

On (b): the fallback claimed a merged deck would run "~32 frames". The actual arithmetic is
17 + 20 = 37 numbered frames, and once the mandatory `[plain]` section-transition and Outline frames
are counted (`SLIDE_STYLE.md:63`) it is ~56 pages - larger than any deck in the course
(measured: L24 = 53, L17 = 47, L22 = 43). It was not a compression, it was two lectures relabelled.

**Alternatives rejected.**
- *Ship L26 first, then L30.* Correct dependency order, but it blocks a chapter the instructor asked
  for on a backlog item with no date.
- *Keep the merge option "just in case".* Rejected because the number in it was wrong; an escape
  hatch nobody has checked is worse than none.

**What would change this.** If L26 ships before L30 is built, the cross-attention frame becomes a
recap instead of new teaching. If the calendar forces a cut, cut scope (end the chapter at latent
diffusion, drop flow matching and video) rather than merging decks.

---

## #3 - Diffusion chapter scope: full derivation, through latent diffusion and video

**Date:** 2026-08-03 · **Status:** active

**Decision.** `ml/ch10_diffusion/` covers diffusion from the forward process through to video
models, deriving the DDPM loss in full rather than asserting it. Planned as **five decks
(L27-L31)**, not four.

**Why.** The instructor chose "full derivation" and "add latent diffusion + video" when asked. The
five-deck count follows from that pair: the ELBO -> L2 -> epsilon-prediction chain is a deck on its
own, and latent diffusion + flow matching + video is another. Four decks would have meant either
compressing the derivation (contradicting the first choice) or dropping the video material
(contradicting the second). The instructor's stated fallback was "build the full one, and maybe
later make it smaller," so the plan is built at full size with the merge points marked.

**Alternatives rejected.**
- *Intuition-only (no ELBO).* Fastest and most visual, but the loss function arrives unexplained -
  and students have already met the ELBO in L23 (VAE), so the machinery is not new to them.
- *DDPM only, stop at MNIST.* Would leave out conditioning and guidance, i.e. the part that makes
  text-to-image actually work, and would not pay off the L23 "diffusion is today's SOTA" claim.
- *Four decks.* Rejected as dishonest packaging of the chosen scope rather than a real reduction.

**What would change this.** If L27-L31 overruns the calendar, merge L27 (forward process) into L28
(the loss) - the merge point is marked in `ml/ch10_diffusion/DIFFUSION_CHAPTER_PLAN.md`. If students
stall on the ELBO derivation in delivery, demote it to an appendix deck and teach the vector-field
route as the main line.

---

## #2 - Two reference videos, deliberately chosen to be complementary

**Date:** 2026-08-03 · **Status:** active

**Decision.** The chapter is sourced from **two** videos, not one: Welch Labs / 3Blue1Brown
"But how do AI images and videos actually work?" (37:20) for intuition, and Deepia
"Diffusion Models: DDPM" (32:05) for the derivation. Both fetched at 1080p into
`ml/ch10_diffusion/_reference_*/`.

**Why.** Neither covers the chapter alone, and the gap is structural rather than a matter of taste:
Welch Labs never writes down the ELBO, and Deepia never covers conditioning or guidance (he defers
score-based/SDE to a later video). Welch Labs supplies *why* DDPM adds noise during sampling - shown
geometrically on a 2D spiral - which Deepia only gets to algebraically. Deepia's closing frame puts
FFHQ diffusion samples beside VAE samples, which closes the L23 VAE "blurry" cliffhanger with
evidence rather than assertion.

**Alternatives rejected.**
- *Welch Labs alone.* Was the original single-video request; leaves the loss unexplained, which is
  incompatible with decision #3.
- *Lilian Weng's blog / the DDPM paper directly.* Both are better references but neither yields
  slide-ready visuals, and the chapter's figure budget is the binding constraint.

**What would change this.** Deepia's promised score-based/SDE follow-up, if it lands, would be a
better source for the DDIM material than the Welch Labs treatment currently planned for L29.

---

## #1 - Diffusion gets its own chapter; `ch9_genai` narrowed to `ch9_attention`

**Date:** 2026-08-03 · **Status:** active

**Decision.** Created `ml/ch10_diffusion/` as a standalone chapter and renamed
`ml/ch9_genai/` -> `ml/ch9_attention/`. `ch9` keeps attention, transformers, and the LLM-training /
RLHF track; generative models move to `ch10`.

**Why.** Four planning documents (`ch6_cnn/L17`, `ch6_cnn/L19`, `ch6_cnn/CNN_CHAPTER_PLAN.md`,
`ch5_neural_networks/CNN_BLOCK_DESIGN.md`) handed diffusion off to "the GenAI chapter," and the ch9
plan itself said "later GenAI parts (generative models) extend the same chapter." That would have
made ch9 carry attention + transformers + LLM training + diffusion + GANs in one folder. Diffusion's
prerequisite spine is the VAE (L23) and CLIP, not the transformer stack, so it does not depend on
most of ch9. Renaming ch9 at the same time stops the folder name from implying it owns all of GenAI,
which is what caused the ambiguity in the first place.

**Alternatives rejected.**
- *Diffusion inside `ch9_genai`.* What the existing plans literally said. Rejected because the
  folder was already the largest in the course and the two topics share few prerequisites.
- *New `ch10` but leave `ch9_genai` named as-is.* Least churn, but preserves the misleading name -
  the next person adding GAN or RLHF material faces the same ambiguity again.

**Cost accepted.** The rename touched 76 tracked files (`git mv`, history preserved) plus five
files carrying the literal string. Decision #1 in `ml/ch9_attention/ATTENTION_CHAPTER_PLAN.md` is
marked superseded rather than rewritten. `_quarto.yml` did not reference ch9, so the site build is
unaffected.

**What would change this.** If the LLM-training track grows large enough to want its own chapter,
`ch9_attention` should split again rather than absorb it.
