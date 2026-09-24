# Decisions

Significant design choices for this repo, newest first. Each entry records what was decided, when,
why, what was rejected, and what would justify revisiting it. Superseded entries stay - the fact
that we changed our mind, and why, is the valuable part.

Deep supporting research lives in the relevant chapter's `_reference_*/` or `research/` folder;
this file holds the choice and a pointer.

---

## #42 - ch19 causal tracing corrupts with 2x the embedding noise and restores a 3-layer window

**Date:** 2026-09-24 · **Status:** active

**Decision.** `facts_tracing.py` adds Gaussian noise of **2x** the typical embedding size to the
subject tokens (the ROME paper uses 3x) and restores MLP/attention outputs over a **window of 3
layers** centred on the restored one, 3 noise draws per fact. The deck says so on its own frame
("The noise is a choice") with the measured table, so students see that the picture depends on it.

**Why.** Measured on the Jordan prompt (`results/facts_tracing.json`, `noise_check_jordan`):

| noise | P(basketball) left | stream at " Jordan" L1 / L4 / L8 | MLPs L0-2 | attention L9-11 at "of" |
|---|---|---|---|---|
| 1x | 0.202 | 85% / 81% / 63% | 91% | 67% |
| 2x | 0.041 | 54% / 59% / 37% | 70% | 58% |
| 3x | 0.010 | 39% / 23% / 21% | 20% | 36% |

At 3x GPT-2 small loses the sentence entirely and no single restore brings much back - the picture
goes blank. At 1x half the answer survives the corruption, so there is little to restore. 2x is
the smallest noise that removes the fact (0.60 -> 0.028 averaged over the 13 known facts) while
leaving the localisation visible. Single-layer restores were too weak on this small model for the
MLP site to show at all.

**Alternatives rejected.** *The paper's 3x* - washes the picture out on a 124M model (table).
*Swapping the subject for another name (clean-vs-corrupt patching, as in the core deck)* - needs a
second fact of the same token length per subject; noise needs none and is what the literature
calls causal tracing. *Single-layer restore* - too weak, see above.

**What would change this.** Moving the deck to a larger model (GPT-2 XL, GPT-J) - then use the
paper's 3x and window 10 and redraw.

---

## #41 - ch19 add-on heavy runs go to a Colab GPU; figures stay CPU-drawable from JSON

**Date:** 2026-09-24 · **Status:** active · **Revisits** #36 ("everything stays CPU-reproducible")

**Decision.** The five heavy ch19 add-on scripts (`facts_tracing.py`, `facts_rome.py`,
`vision_features.py`, `vision_attribution.py`, `diffing_finetune.py`) pick `cuda` when present
(`mi_common.pick_device`) and were run on a Colab T4 through the `colab` CLI in WSL. Every script
keeps a `--plot-only` path that redraws its figures on CPU from `results/*.json`, so rebuilding a
deck never needs a GPU. Core decks and the circuits add-on still run end to end on CPU.

**Why.** Running causal tracing locally took ~4.5 min per fact before batching; with feature
visualization and fine-tuning queued behind it the laptop sat at 100% CPU / 97% RAM and the
instructor asked (2026-09-24, just after midnight) to stop local heavy runs and use the Colab CLI instead.

**Alternatives rejected.** *Keep grinding on CPU at low priority* - tried; the machine stayed
unusable. *Shrink the experiments to fit the CPU* (fewer facts, fewer feature-viz steps) - the
13-fact average and the 512-step feature images are what make the figures trustworthy. *RunPod* -
paid; Colab free tier was enough.

**What would change this.** If a script's `--plot-only` path ever needs the model (it must not),
fix the script. If Colab access is lost, the JSONs in `results/` are the artifact of record - the
decks rebuild without rerunning anything.

---

## #40 - The RNN chapter gets one CPU-light practical, merging the two June homework designs

**Date:** 2026-09-23 · **Status:** active · **Reverses** the "No homework this chapter" lock in
`ml/ch7_rnn/RNN_CHAPTER_PLAN.md` (2026-07-13)

**Decision.** `ml/ch7_rnn/xx_rnn_memory_solution.ipynb` (+ the derived task version
`xx_rnn_memory.ipynb`), built by `py_src/build_rnn_practical_nb.py` and
`py_src/build_rnn_practical_tasks.py`. Four parts, each measuring one lecture claim: (1) the RNN
step by hand, reproducing L20's slide numbers and `nn.RNN`; (2) autograd confirming lambda^T and a
real RNN's fading sensitivity; (3) a first-token-recall race, vanilla RNN vs LSTM with the forget
gate half-shut vs open; (4) a char-GRU on the ch11 surnames, same split, compared position by
position with the ch11 window MLP. LSTM/GRU stay black boxes. ~70-140 s on 2 CPU threads.

**Why.** The instructor asked for an RNN practical on 2026-09-22 (answered "Build an RNN
practical" when asked). The June `RNN_BLOCK_DESIGN.md` had already specified the two halves (HW1
"build an RNN, watch the gradient vanish, vanilla fails long / LSTM succeeds"; HW2 "char-level
model on Armenian names"); merging them avoids a second notebook. Measured before writing the
prose (scratchpad, 2026-09-23): GRU 1.605 vs MLP K=3 1.673 nats/char; the recall race at T=40 is
100% (open gate, 100 steps) vs 26% / 25% (RNN, half-shut LSTM, 800 steps).

**Alternatives rejected.** *A char-RNN that writes Armenian text* - duplicates the ch8 SAE
homework's Part 0 ("train a tiny RNN that spells") and needs a corpus the July scope change
declined. *Two notebooks (HW1 + HW2 as designed)* - more to maintain for the same four lessons.
*Colab/GPU* - unnecessary; every run is seconds on CPU.

**What would change this.** If Part 3's race stops separating the runners on another machine or
PyTorch version (it is single-seed by design; the notebook invites a `seed=510` rerun), move it to
2 seeds like the deck figure. If the practical is assigned as homework rather than run in class,
rename to the `NN_HW1_` scheme on delivery.

---

## #39 - L21's "Does it work?" frame now shows a measured run - reopening the July "no real training runs" scope

**Date:** 2026-09-23 · **Status:** active · **Reopens** the 2026-07-13 instructor scope change
logged in `ml/ch7_rnn/L21_DECISIONS.md` for this one frame

**Decision.** The frame embeds `fig/memory_highway.pdf` from `py_src/memory_highway.py` ->
`results/memory_highway.json`: (left) sensitivity of the final state to the input k steps back
at initialisation, RNN vs LSTM with forget-gate bias 0 / 1 / 3; (right) first-token-recall
accuracy vs T for RNN, LSTM bias 0, LSTM bias 3, 2 seeds, <=1500 steps each. The old
`gradient_flow_comparison.pdf` stays on disk, unembedded.

**Why - the new evidence.** The July quick run showed *no* LSTM advantage and the frame said so
honestly, so the slide asked "does it work?" and could not answer it. L21_DECISIONS.md's own open
question #2 asked whether a clearer illustration would be better. The new measurement explains
the old null result instead of hiding it: PyTorch starts the forget gate half-shut (bias ~0), and
such an LSTM forgets as fast as a vanilla RNN (40 steps back: RNN 7.6e-11, LSTM bias 0 2.7e-9,
bias 3 3.7e-2). Cost: ~15 min on 2 CPU threads, one-off, checkpointed per run.

**Alternatives rejected.** *A purely schematic figure* - allowed by the July message, but the
course's line is "measured, not asserted", and the schematic cannot teach the forget-bias point.
*Tuning the old run until it flatters* - explicitly ruled out in July, rightly.

**What would change this.** If the instructor re-affirms "no training runs in L21", revert the
one `\includegraphics` line and drop the new frames; the practical (#40) carries the same lesson.

---

## #38 - L16's colour section collapses to one recap frame, deferring to deck [33]

**Date:** 2026-09-22 · **Status:** active · Resolves the `DEFERRED_TODO.md` item "ch6 CNN - trim
L16's colour section now that deck 33 exists" (parked 2026-08-20)

**Decision.** L16's three frames "How your eye sees color", "A pixel is three numbers" and "Three
numbers, but which three?" become one frame, "Color, recalled from [33]", keeping only what the
CNN lecture needs (a colour photo is a 3 x H x W stack = input channels). The figures and their
`py_src/` scripts stay on disk; the originals are recoverable with `git show 6f0f7cd:...`.

**Why.** [33] Color Spaces was delivered on 2026-08-20 and teaches cones, RGB and HSV in depth.
The DEFERRED entry said to decide "when the CNN chapter is next touched"; the pre-delivery pass
of 2026-09-22 was that moment. The Sonnet student review of the new L16 judged the one-slide recap
"enough, and well-placed" for material a month old. HW1b's pointer to the HSV frame was moved to
[33] in the same pass.

**Alternatives rejected.** *Keep all three frames* (the "L16 stays self-contained" argument) -
costs a CNN session three frames of re-teaching. *Delete with no recap* - loses the bridge from
"three stacked grids" to input channels, which Section 4 needs.

**What would change this.** If L16 is ever taught without [33] before it (another course, a
reordered schedule), restore the three frames from git.

---

## #37 - ch19 decks drop the L45-L47 names for `xx_` names until delivery

**Date:** 2026-09-22 · **Status:** active · **Supersedes #13**

**Decision.** The three v1 mech-interp decks are renamed `L45_opening_the_box` ->
`xx_opening_the_box`, `L46_does_it_actually_do_that` -> `xx_does_it_actually_do_that`,
`L47_features` -> `xx_features`; the five new decks of the extension (#36) are `xx_` too. Each gets
its `NN_` delivery number when taught, like every other delivered deck.

**Why - the new evidence that reopens #13.** #13 kept L-numbers as build-order ids and said the
chapter would only be renumbered as part of a general renumbering. That renumbering has started
piecemeal: on 2026-09-21 the ch11 decks were renamed to their delivery numbers `44_optimization`
and `45_optimization_init_activations`. "L45" and "45" now name two different lectures, and
`CONVENTIONS.md` already calls L-prefixes legacy. Adding five decks to a chapter whose order is
not settled (#36: "build it all, decide later") also makes any fixed number wrong on arrival.

**Alternatives rejected.** *Keep L45-L47 and number the new decks L48-L52* - extends a scheme the
repo is moving away from, and the numbers would imply an order that #36 deliberately leaves open.
*Assign delivery numbers now* - the schedule slot is not decided (`ml/00_plan.md`).

**What would change this.** Nothing short of a repo-wide naming rule change; on delivery each
deck is renamed to `NN_topic` per `CONVENTIONS.md`.

---

## #36 - ch19 grows from 3 decks to 8: four core decks plus four droppable add-ons

**Date:** 2026-09-22 · **Status:** active · revisited 2026-09-24 (#41: heavy add-on runs moved to Colab)

**Decision.** Mechanistic interpretability becomes an 8-deck chapter: a new **intro** deck plus the
three v1 decks (revised to be easier) form a complete 4-session **core**; four new **add-on**
decks cover the gaps found in v1 - vision interpretability (saliency, integrated gradients,
feature visualization, curve detectors), transformer circuits read off the weights (QK/OV,
composition), where facts live (MLPs, causal tracing, ROME-style editing), and personas and model
diffing. Each add-on depends only on core decks before it, so any subset can be dropped at
scheduling time. Models: GPT-2 small (CPU) for language, TransformerLens `attn-only-1l/2l` for the
circuits deck, torchvision ResNet-18 for vision, and a GPT-2 small fine-tuned here as the diffing
model organism. Spec and build log: `ml/ch19_mech_interp/EXTENSION_PLAN.md`.

**Why.** Instructor's ask (2026-09-22): extend the LMU-based interpretability material to mech
interp, make it easier with more examples, fill gaps, add an intro. The survey found LMU has no
mech interp at all (upstream checked), and ch19 v1 had one hard running example (IOI), no toy
network, no weights-based circuits, almost no MLPs, no vision and nothing from 2025-26 on
personas/diffing. All choices in the interview table of the spec were the instructor's, including
"build everything, decide later what to teach" - which is what forces the core + add-on shape.

**Alternatives rejected.** *LMU-style short chunks* - offered, instructor chose house decks.
*One linear 8-deck chapter* - less recap per deck, but dropping a deck later would break callbacks
downstream. *A bigger model (Gemma-2-2B on Colab)* for real SAE and attribution-graph figures -
declined; everything stays CPU-reproducible. *Papers-only diffing* - declined in favour of a model
organism with known ground truth.

**What would change this.** When the schedule is set: any add-on that does not get a session is
dropped from the qmd, not deleted. If a build-risk gate fails (spec table G1-G7), that section is
redesigned around what was measured.

---

## #35 - The ch6 barcode project ships as one walkthrough notebook, decoding EAN-13 from scratch

**Date:** 2026-09-06 · **Status:** active

**Decision.** New standalone practical `ml/ch6_cnn/project_barcode.ipynb` (instructor-chosen
format: single reference walkthrough, no student/TODO version; instructor-chosen placement:
standalone project in ch6, not HW1d). Scope, per the instructor's ask: EAN-13 bars-to-digits
decoding, check-digit validation, gradient-based orientation detection, and localization -
everything explained step by step, OpenCV allowed (contrast with HW1b/HW1c where cv2 was
banned). Test data: self-generated synthetic barcodes (own encoder, perfect ground truth,
staged degradations) plus 6 real product photos from Wikimedia Commons committed under
`data/barcode/` (downscaled). All prose numbers were measured in scratch scripts BEFORE the
notebook was written (per the write-after-measuring rule, `_learnings/2026-08-13-2015`).

**Why.** The chapter thesis needed a counterweight: HW1c shows hand-designed filters failing on
organic data; the barcode is the opposite pole - a pattern DESIGNED for machine reading, where
classical CV is the right tool and no training is needed. Key measured facts driving the
notebook's arc: adaptive thresholding fixes uneven light (0->30/30) but breaks the noise cases
global thresholding handled (30->0/30), so the pipeline is a checksum-gated cascade; a sliding
59-run window fixes quiet-zone speckle (1/30 -> 30/30 on ramp+noise); band-averaging crushes
noise (sigma=120: 30/30); blur ~ module width is fatal at 3 px modules (0/30) but trivial at
6 px (30/30) - resolution beats cleverness; the classic |gx|-|gy| localizer is accidentally
rotation-symmetric (convertScaleAbs takes abs) yet dies near 45 deg (7/15) while the
structure-tensor coherence localizer is rotation-proof (40/40); end-to-end 5/6 real photos vs
3/6 for cv2.barcode.BarcodeDetector on the same six. The one failure (barcode wrapped around
the bottle's curved side) and one caught false positive (checksum passes ~1 in 10 garbage;
killed by a min-3-vote gate) are kept in the notebook as the honest-failure act. Reviewed
2026-09-06: an inline factual pass (4 fixes, e.g. the cv2.barcode WeChat attribution was
wrong) plus one adversarial Sonnet subagent that independently re-derived the math (0
mismatches) and flagged 9 findings; all but one applied (declined: empirically printing
derivable arithmetic like sqrt(15)), notebook re-executed, all numbers stable.

**Alternatives rejected.** HW1d naming (instructor picked standalone-project naming); a
student+solution pair (double build cost, and the ask was explicitly a walkthrough);
python-barcode / treepoem for generating test images (writing the encoder ourselves IS the
lesson on EAN-13 structure and gives exact ground truth); synthetic-only data (no real-photo
payoff); UPC-A (EAN-13 is its superset, has the parity-encoded 13th digit story, and 485 =
Armenia's GS1 prefix).

**What would change this.** If students need a task version, derive one by stripping solution
cells (the walkthrough was written with clean per-step sections to make that split cheap). If
the 6 photos bloat the repo, re-point the loader at the Commons URLs (kept in the notebook).

---

## #34 - opencv-python pinned at 4.11.0.86 in ma; 5.x forbidden while numpy is 1.26

**Date:** 2026-09-06 · **Status:** active

**Decision.** `ma` gets `opencv-python==4.11.0.86` (for the ch6 barcode project). opencv-python
5.0.0.93 is NOT allowed: installing it silently upgraded the shared venv's numpy 1.26.4 ->
2.4.6, which risks breaking every compiled dependency in this repo (scipy 1.13.1, sklearn
1.7.1, matplotlib all built/tested against numpy 1.x here). Rolled back immediately;
cv2 4.11 + numpy 1.26.4 verified importing side by side.

**Why.** The shared venv serves ~30 notebooks and dozens of figure scripts; a silent numpy
major bump is exactly the "fresh sync pulls a breaking major" failure the pin-exact-versions
rule exists for.

**Alternatives rejected.** opencv-python 5.0 with numpy 2.x (would mean revalidating the whole
repo for one chapter's practical); a separate venv just for cv2 (violates the one-`ma`-venv
convention).

**What would change this.** A deliberate, repo-wide numpy 2.x migration - do it as its own
tested change, never as a side effect of installing something else.

---

## #33 - Project 3's reference walkthrough runs on an own web-fetched demo set, not imagenette

**Date:** 2026-09-05 · **Status:** active

**Decision.** 37_image_clusters_solution.ipynb (the Project 3 "Build your own Google Photos"
reference) was rebuilt on data/demo_photos/: 92 Wikimedia Commons photos in 11 categories
(instructor's 5: Khustup, shawarma, cheese, potato, folk dance; plus duduk, Sevan, khachkar,
pomegranate, Cascade; plus khorovats planted as a deliberately-unseparable trap). Fetched by
py_src/fetch_project_demo_images.py, manually curated off contact sheets (132 -> 92; logos,
maps, archival b/w, dupes dropped; _sources.json maps each file to its Commons title), then
embedded with the SAME given scripts students use (embed_my_photos.py -> demo_photos_clip.npz).
The notebook gained a CLIP multimodality playground (instructor request): text-to-photo search,
image-minus-image ~ text arithmetic, photo+word retrieval, before any clustering.

**Why.** The reference should mirror the student experience (own folder through the given
tooling), and imagenette's 10 clean classes hid the interesting failure modes. Measured before
writing (per the write-after-measuring rule): pixels ARI 0.063 vs embeddings 0.647 (k=11);
silhouette prefers k=7 (0.253) over the true 11 (0.243) because the trap works - k-means fuses
all 10 khorovats with 8/9 shawarma (centroid cosine 0.897, the #1 pair); mean(sevan) -
mean(khustup) -> "sea, lake, water" and mean(shawarma) - mean(khorovats) -> "lavash, sandwich,
wrap"; zero-shot naming 97.8% with sentence prompts vs 65% bare words. All numbers printed by
the executed notebook match the pre-registered measurements.

**Alternatives rejected.** Keeping imagenette (too clean, no multimodal story on our terms, and
"instead of imagenet" was the explicit instruction); students' own genres like memes/screenshots
(not reproducible as a committed dataset); a new notebook file alongside the old one (two
near-identical walkthroughs to maintain; git history keeps the imagenette version).

**What would change this.** If the ~20 MB data/demo_photos/ folder proves too heavy for the
repo, keep only demo_photos_clip.npz (0.5 MB) + fetch script committed and drop the raw JPEGs;
thumbnails inside the npz are enough to re-render every figure except the 12-photo sample grid.

---

## #32 - The ch10 live practical is "genes mirror geography" on 1000 Genomes chr22

**Date:** 2026-09-03 · **Status:** active

**Decision.** The instructor-conducted ch10 practical (not homework - the assigned homework is
Project 3, the CLIP photo map) reproduces the Novembre et al. (2008, Nature) "genes mirror
geography" result on open data: PCA of raw genotype counts from 1000 Genomes phase 3,
chromosome 22 only (2,504 people, 26 populations). Instructor chose it over two lower-prep
alternatives (eigen-patches / "PCA invented JPEG" on the Saryan painting; LSA semantic search
over Armenian Wikipedia) for wow factor and because every step exercises lecture 36's frames:
curse of dimensionality, the scaling decision (Patterson allele-frequency scaling), variance =
structure, out-of-sample projection. Armenian samples are explicitly optional ("its fine if no
armenians in the data") - the Human Origins merge is parked, and the AJHG 2024 Armenian-PCA
figure can close the session with attribution if wanted.

**Honesty constraint baked in.** The exact Novembre figure cannot be reproduced from open data -
POPRES (their dataset: 1,387 Europeans at country resolution, ~197k SNPs after QC from a 500k chip) is dbGaP controlled-access. The
open-data version shows (a) continental structure worldwide and (b) the coarse European gradient
across 1000G's five EUR populations (FIN/CEU/GBR/IBS/TSI). Per the write-after-measuring rule
(_learnings 2026-08-13-2015) and the LFW precedent (#21), py_src/non_essential/
validate_genes_geography.py measures the story (EVR, silhouette, k-means ARI, per-pop medians)
BEFORE the practical notebook is written; the practical is built only on what the validation
shows.

**Data mechanics.** py_src/fetch_1000g_genotypes.py streams the ~205 MB chr22 VCF (URL
HEAD-verified 2026-09-03; v5a does not exist, v5b does) plus the sample panel from the EBI FTP,
keeps biallelic SNPs with 0.05 <= AF <= 0.95, every 6th passing variant, and commits
data/genomes_1000g_chr22.npz (int8, numpy-only for students). Genotype parse fails loudly on
any unexpected genotype string rather than imputing.

**Alternatives rejected.** *Whole-genome or multi-chromosome* - chr22 alone carries the
structure and keeps the download and the committed npz small. *scikit-allel / plink tooling* -
a pure-python streaming parse keeps the fetch script dependency-free and readable as course
material. *POPRES application* - controlled access, weeks of lead time, and the course does not
need country-level Europe to make the point.

**Validation outcome (2026-09-03, same day).** The pre-registered condition fired. Measured on
the committed 16k npz vs a full-density 97k scratch matrix (EUR subset, Patterson scaling,
monomorphic-in-subset SNPs dropped): silhouette over the five EUR populations on PC1-2 is 0.012
vs 0.014, and the IBS-TSI median separation is 0.06 vs 0.15 within-pop-std units - the ratio
grew by exactly sqrt(6), as signal-averaging predicts, and is still invisible on a projector.
So: the worldwide panel leads (it is textbook - silhouette 0.588, k-means ARI 0.87), the
within-Europe beat is FIN-vs-mainland only, the Novembre Figure 1 (free PMC copy, attributed)
closes as "country-level sampling + ~197k QC'd chip SNPs buys THIS", and the committed npz
stays at 16k / 6.0 MB - the 6x bigger matrix buys nothing visible, so no instructor-side large
matrix either. The sqrt(m) scaling of separation is itself now a planned teaching beat.

**What would change this.** If validation shows the EUR-only panel is too weak to read on a
projector (five populations may just blob), the practical leads with the worldwide panel and
the Novembre figure is shown as the borrowed "with ~197k QC'd chip SNPs and country-level
sampling you get THIS" closer. If the committed npz exceeds ~20 MB, raise KEEP_EVERY and regenerate.

---

## #31 - Ch10 review round 2 applied; UMAP deck stops overselling the repulsion-term story

**Date:** 2026-09-02 · **Status:** active

**Decision.** A second content review of `35_dimensionality_reduction.tex` / `36_umap.tex` was
applied in full, together with four instructor-requested pedagogy additions (why maximize
variance; covariance-matrix refresher; what "linear" means; SVD factor anatomy). The one
substantive content change: **36_umap no longer claims UMAP's extra global structure comes from
the repulsion term alone.** A new "Where does the layout start?" frame teaches the spectral
(Laplacian-eigenmaps) initialization and carries an honesty box: Kobak & Linderman (2021, Nature
Biotechnology) showed much of UMAP's measured global-structure advantage over t-SNE disappears
when both methods get an informative start, and sklearn's t-SNE has defaulted to PCA init since
v1.2 (both facts web-verified 2026-09-02). The loss-frame payoff box and the recap were softened
to "part of the story" accordingly.

**Why.** The repulsion-term narrative is the UMAP paper's own framing and is pedagogically clean,
but presenting it as settled would leave students with a claim the literature has specifically
tested and largely overturned. The deck's own theme is "which parts of the picture are you
allowed to believe" - it cannot itself oversell.

**Also in this pass** (deck 35): three new frames (why-variance with a projected |A-B| = 12.6 vs
0.7 demonstration; covariance refresher with the Var(Xw) = w'Sigma w identity; linearity via a
grid that a matrix cannot bend), an annotated SVD block diagram, an eigen-garments frame (PC1
29%, PC2 18%, sets up Project 1), a "New points?" column in the decision table (t-SNE has no
out-of-sample map), the L13b/L13c stale references the 2026-08-16 renumber sweep missed, the
SLIDE_STYLE.md acronym check finally run mechanically (t-SNE, UMAP, LSA, EVR, DR were all
unexpanded), and one-liners: elbow callback, PCA-inside-the-pipeline leakage note, crowding
problem explained, LDA/Fisherfaces pointer to Project 2.

**Alternatives rejected.** *Footnote-only for the init caveat* - rejected: the initialization is
part of the mechanism (the deck otherwise never says where SGD starts), so it earns a frame, not
an apology. *Reopening the 2x2 characteristic-polynomial by-hand example* - stays closed per the
2026-08-16 decision; the linear-algebra course covers it.

**What would change this.** If delivery shows the honesty box confuses more than it clarifies,
demote it to a spoken remark and keep only the spectral-init frame. Review details:
`ml/10_dimensionality_reduction/REVIEW.md`, round-2 section.

---

## #30 - The photo-grouping project moves from clustering to dimensionality reduction

**Date:** 2026-08-27 · **Status:** active · **Supersedes the placement in #17**

**Decision.** `34_image_clusters_solution.ipynb` moves out of `ml/09_clustering/` and becomes
`ml/10_dimensionality_reduction/37_image_clusters_solution.ipynb` (Project 3 of that chapter),
taking `imagenette_clip.npz`, `embed_images_clip.py` and its five `out/` artifacts with it. The
number changes from 34 to 37 because the prefix is the practical-session number and 37 is ch10's,
shared with `37_eigenfaces_solution.ipynb`.

**Why.** The project's actual subject is **representation**, not partitioning: its headline
measurement is k-means scoring ARI 0.048 on raw pixels against 0.939 on CLIP embeddings, with the
algorithm held fixed. That is a dimensionality-reduction argument. It also fits ch10's existing
material - `py_src/dimred_demos.py` was already reaching across chapters into
`ml/09_clustering/data/imagenette_clip.npz` for its 512-d demo, which the move removes - and its
task 5 (project 512 dimensions to 2 for an interactive map) is a DR exercise sitting in a
clustering chapter. Ch09 was also carrying four projects against ch10's two.

**Alternatives rejected.** Leaving it and cross-linking from ch10 - rejected because the
cross-chapter data reference was already awkward and the chapter balance was wrong. Duplicating it
in both - rejected outright; two copies of a notebook diverge.

**What would change this.** If ch10 runs long in delivery, this is the project to cut first: it is
the only one of the three whose lesson is also made elsewhere (ch09's Sevan practical makes the
representation argument too, on features rather than embeddings).

**Follow-on edits.** Resource lists in both chapter qmds; `dimred_demos.py` CLIP path; the
photo-grouping comment in `ml/09_clustering/py_src/color_histogram.py`; the Aug 21 row of
`ml/00_plan.md`; and the ch10 page title, which still said "06 Dimensionality Reduction".

---

## #29 - The land-cover practical drops its GMM act; DBSCAN carries section 4 alone

**Date:** 2026-08-26 · **Status:** active

**Decision.** Cut the Gaussian-mixture half of `34_land_cover_solution.ipynb` at the instructor's
request: the soft-assignment fit, the max-responsibility confidence map, the k-means-vs-GMM ARI
comparison, and the closing "the mixture model knew" reveal. Section 4 is now DBSCAN alone,
reframed as "an honest failure". The conclusion table lost its *soft assignment* row and its
*k-means assumptions* row, the latter having had no remaining evidence once the GMM comparison
went.

**Why.** Instructor's call on session length. The notebook had grown past a 90-minute slot, and
Act 0 gained substantial new material the same day (provenance, per-band physics, reflectance,
and the ground-truth reveal), which had to come from somewhere.

**Alternatives rejected.** Keeping GMM and cutting the DBSCAN act instead - rejected because
DBSCAN is the only place in the practical where an algorithm from the deck is shown *failing* for
a stateable reason, and that is the harder lesson to get elsewhere. Trimming both to half length -
rejected as leaving two thin acts rather than one solid one.

**What would change this.** If the session runs short in delivery, or if the clustering deck's
GMM/EM section needs a practical anchor it currently lacks. The act was good and is recoverable
verbatim from commit `684b4cf`; `34_land_cover_OUTLINE.md` records where it sat.

---

## #28 - Reflectance conversions are validated against physics, not taken from metadata

**Date:** 2026-08-26 · **Status:** active

**Decision.** `ml/09_clustering/py_src/fetch_sevan_scene.py` no longer trusts the `scale` and
`offset` advertised in a scene's STAC `raster:bands` metadata. A new `verify_reflectance()` checks
the advertised conversion against a physical invariant - **surface reflectance is a ratio of light
out to light in and cannot be negative** - drops the offset with a loud `log.warning` if applying
it drives more than 0.1 % of values below zero, and raises if neither variant is physical.

**Why.** The committed cube had been converted with `DN * scale + offset` where the metadata
advertised `offset = -0.1`. That is correct for a raw post-baseline-04.00 L2A product but wrong
for Element84's `sentinel-2-l2a` COGs, which are already baseline-harmonised. The offset was
applied twice. **60.04 % of the cube was negative**, open water sat at about -0.09 in all six
bands, and the error had been shipped and taught. Measured on the real cube, the new guard reads
60.04 % negative with the offset and 0.0000 % without - decisive either way.

The cost was not cosmetic. NDVI is computed after `np.clip(refl, 0, None)`, so red clipped to zero
over vegetation and NDVI saturated to **1.0** for tree cover and grassland alike, turning the
vegetation index into a water mask. That artifact produced the notebook's headline finding
("indices only, k=4, ARI 0.589, the best result in the notebook"). Corrected, the top three
feature sets tie at 0.483 / 0.482 / 0.479, and the deck's "scale first" rule - which the buggy
version had appeared to *refute* - comes out vindicated.

**Alternatives rejected.** Hardcoding `offset = 0.0` for this collection - rejected because it is
silently wrong for any genuinely unharmonised product a student might fetch with the same script,
and the qmd's bonus task invites exactly that. Asserting hard and refusing to run - rejected
because the correct action here is unambiguous and recoverable, so a warning plus the right answer
beats a crash. Leaving it and documenting the quirk - rejected: the notebook already *had* a
plausible-sounding explanation for the negative values, and that is precisely what stopped anyone
looking.

**What would change this.** If a future scene legitimately needs the offset, the guard applies it
automatically - it only intervenes when the result would be unphysical. If ESA or Element84 change
their harmonisation policy, the 0.1 % threshold is the knob. Revisit if the warning ever fires on
a product known to be unharmonised.

**See also.** `_learnings/2026-08-26-2145_reflectance-metadata-lied-and-honest-measurement-did-not-catch-it.md`
and `ml/09_clustering/34_land_cover_OUTLINE.md`, "What the rebuild found (2026-08-26)".

---

## #27 - A fourth ch09 project: the semantic tree of Armenian words

**Date:** 2026-08-21 · **Status:** active

**Decision.** New project `xx_semantic_tree` (🧀🧀🧀) in `ml/09_clustering/`: 108 Armenian words
and phrases across 12 semantic families, embedded with the instructor's own
`Metric-AI/armenian-text-embeddings-2-large`, clustered hierarchically and read as a dendrogram.
Dataset in `py_src/armenian_words.py`, embeddings pre-computed into `data/armenian_words.npz`
(563 KB) so students download nothing.

**Why a fourth project.** The existing three all ask *which representation wins* and all answer
it with pixels. This one fixes the representation and asks what else decides the clusters -
metric, linkage, and tokenizer. It is also the only project whose clusters can be **read**:
you cannot eyeball a pixel cluster, but `մայր, հայր, քույր, եղբայր` checks itself.

**The finding it is built around, measured not assumed.** `ձու` (egg) and `ձի` (horse) are the
**closest pair in the entire space** (0.505), closer than every planted synonym except one.
The cause is not spelling but a shared **subword token**: all of ձու/ձի/ձուկ/ձյուն tokenize as
`▁ձ` + suffix, while `ձեռք` is a single token `▁ձեռք` and sits 2.8× further away despite the
same first letter. Six such words - 6% of the vocabulary - cost **22% of the ARI**
(0.287 → 0.350 when removed). Phrase forms recover about half of that.

**Correction (2026-08-30).** Measured by true Euclidean distance rather than dendrogram merge
height (which is where the numbers above came from): `ձու`/`ձի` is the *second* closest pair in
the space at 0.505, behind the planted synonym `ուրախ`/`երջանիկ` at 0.490, and `ձեռք` sits
1.6× further from `ձի` (0.793), not 2.8× (that was the cophenetic height, 1.399). The 22% is the
ARI *lift* from removing the six words, 0.287 → 0.350. The notebook body already used the
corrected numbers; its conclusion, the qmd tasks, and the outline were brought into line.

**Related:** decision **#26** (Armenian glitch-token hunt) attacks the same phenomenon head-on
across 8 tokenizers. This project is where a student meets it as a *consequence* rather than a
subject; the two should cross-reference.

**Alternatives rejected.**
- *Bilingual hy+en "does meaning survive translation".* My original pitch; instructor chose
  Armenian-only and a dendrogram-first, structure-discovery shape instead.
- *Documents or sentences as leaves.* Unreadable on a tree; words plus short phrases keep the
  figure legible and make the words-vs-phrases comparison intrinsic.
- *The normalization trap as a bonus.* Does not exist by this route - the model returns
  unit-norm vectors through `sentence-transformers` regardless of the flag (std 0.0000).
  Replaced by an anisotropy/centering bonus.

**What would change this.** ARI is only 0.287 with 12 families, partly because the probes are
designed to damage it. If students read that as "embeddings do not work", merge `մարմին` and
`բնություն` into the other families and re-baseline.

---

## #26 - Armenian glitch token hunt: 8 tokenizers, wiki corpus, Gradio playground, ungated mirrors

**Date:** 2026-08-21 · **Status:** active

**Decision.** New standalone mini-project `ml/claude_projects/armenian_glitch_token_hunt/`
(intended for a later live-coding session): token-level analysis of Armenian across 8 tokenizers
(XLM-R deep dive + mBERT, GPT-2/cl100k/o200k via tiktoken, Llama 3, Gemma 2, Qwen 2.5), corpus =
`wikimedia/wikipedia` `20231101.hy` 20M-char sample with a 5M-char `en` baseline. Experiment
notebook writes JSONs to `out/`; `build_report.py` derives the HTML report; `tokenizer_app.py` is
a Gradio playground accepting any HF repo. All choices confirmed by the instructor in the kickoff
Q&A.

**Why.** Gradio: one file, HF tokenizers and tiktoken behind the same code path, colored spans
built in. Wikipedia: clean one-line streaming download, reproducible sample. Llama 3 / Gemma 2 go
through ungated mirrors (`Xenova/llama3-tokenizer`, `unsloth/gemma-2-9b`) because this machine has
no local HF token and the official repos are gated (verified: `whoami` raises
`LocalTokenNotFoundError`).

**Alternatives rejected.** Streamlit (more boilerplate for colored token spans) and static
HTML + transformers.js (no tiktoken support) for the app; OSCAR/CC-100 webtext as corpus
(messier, some mirrors gated - wiki gives cleaner headline numbers); official gated repos
(no local token).

**What would change this.** A live session wanting messier glitch material -> add an OSCAR `hy`
sample as a second corpus; local `hf auth login` -> swap mirrors for the official repos;
tiktokenizer-style UI polish mattering -> revisit the static-HTML option.

## #25 - The image-compression project grows to three cheeses and exercises both new decks

**Date:** 2026-08-21 · **Status:** active

**Decision.** `35_image_compression_solution.ipynb` gains four sections (11-14) and the qmd task
grows from 6 items to 10, moving 🧀🧀 → 🧀🧀🧀. The additions, one per assumption the original
task left unexamined:

1. **Gamma** (task 7) - the original clustered gamma-encoded values. A centroid is an *average*,
   and averaging encoded values does not average light, so every palette it produced was biased
   dark. Measured: at `k=2` the image loses **10.76%** of its light; clustering in linear light
   loses **0.00%**, and reconstruction error in linear light improves 9-36%.
2. **k-medoids** (task 8) - **0 of 8** k-means centroids are a colour that occurs anywhere in the
   image. Medoids are real pixels by construction, at a measured **+7.7%** reconstruction cost.
3. **Hierarchical nesting** (task 9) - k-means `k=8` is *not* a coarsening of its `k=16`
   (verified `False`); one Ward tree cut at 8/16/32 is (verified `True`, both levels).
4. **Density** (task 10) - DBSCAN swings from 14 clusters (51% noise) to 1 as `eps` moves by 5x,
   and noise pixels have no centroid, so the student must decide what colour to paint them.

**Why.** Three of the four are the only places where the new colour-spaces deck and the clustering
deck's non-k-means algorithms become *consequential* rather than decorative. The gamma one is the
strongest teaching move available here because it does not add a variant - it shows the solution
the students just wrote is measurably wrong, in a direction predicted by Jensen's inequality.

**Alternatives rejected** (all offered to the instructor, all declined for this pass):
- *Lab + ΔE as the error metric.* Strong on paper - the deck measured a 3.7x spread in perceived
  difference at fixed RGB distance - but kept as a bonus to hold the task's size down.
- *Floyd-Steinberg dithering.* Would close task 6's banding dead end. Bonus instead.
- *Bit-budget duel against 4:2:0 chroma subsampling.* Bonus instead.
- *GMM soft assignment.* Cut; overlaps dithering and adds a third palette variant.

**What would change this.** If the task starts taking students more than a session, the density
section (task 10) is the one to drop - it is the most interesting *negative* result but the least
transferable skill. If ΔE ever moves from bonus to core, drop k-medoids rather than adding an
eleventh task.

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

**Date:** 2026-08-14 · **Status:** active, but **relocated by #30** - the notebook and its
data now live in `ml/10_dimensionality_reduction/` as `37_image_clusters_solution.ipynb`.
The reasoning below still holds; only the chapter changed.

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

**Date:** 2026-08-13 · **Status:** superseded by #37 (2026-09-22)

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
