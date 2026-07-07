# L13 Clustering (ml/ch4_clustering) - content & teaching review

Reviewed 2026-07-07 (Claude). Scope: `L13_clustering.tex` (46 frame-numbers / 63 physical
pages), `L13_clustering_OUTLINE.md`, all figures in `fig/`, generator `py_src/cluster_demos.py`.
Method: close read of the `.tex`, visual pass on every rendered page (7 contact sheets), and
up-close renders of the figures I was unsure about (HDBSCAN, agglo dendrogram, GMM, gallery).
Pure content/pedagogy - infrastructure, numbering, and qmd wiring deliberately ignored.

**Verdict up front:** this is a strong, well-sequenced deck - the best-built of the ml decks I've
seen. The arc (distance -> k-means mechanics -> GMM as the soft generalization -> density/hierarchy
-> evaluation with no ground truth -> curse of dim -> PCA bridge) is textbook-grade, the two
by-hand numerics (k=2 iteration and Rand/ARI) are both **arithmetically correct** (I recomputed
them), the callouts pre-empt exactly the right confusions, and the two 6-frame animations
(k-means, DBSCAN) earn their space. The problems are all fixable and mostly cosmetic-to-medium:
(1) the agglomerative animation has a **real dendrogram inversion** baked in by its linkage choice
and a caption that **collides with the page number** on all 7 overlays; (2) the HDBSCAN figure's
titles undercut its own message; (3) the "wider zoo" frame is a raw image dump that under-delivers
against its own outline; (4) the single most memorable result in clustering (k-means invents
clusters in pure noise) is buried as a bullet instead of being a predict-first frame; (5) a couple
of small figure/scope nits. Nothing here is a conceptual error in the teaching - the math is sound.

---

## 1. Bugs, contradictions, and things to fix first

1. **Agglomerative caption collides with the page number (visual bug, confirmed).** Frame
   "Agglomerative, step by step" (25/46, all **7** overlay pages). The bottom `\footnotesize`
   caption "Each merge distance $d$ (between midpoints, here *centroid* linkage) becomes that
   bar's height in the dendrogram." wraps to two lines and the trailing word "dendrogram." runs
   straight under the "25 / 46" page number on every one of the 7 states. Confirmed on contact
   sheets 5 and 6. Fix: shorten to one line (e.g. "Each merge's distance is that bar's height in
   the dendrogram (centroid linkage).") and/or drop the figure width from `0.82` to `~0.78` and
   add a small `\vskip`. This is the instructor's suspected collision - it is real.

2. **The agglomerative animation contains a dendrogram inversion (correctness/clarity).** The
   figure uses **centroid linkage**, and centroid linkage is the one linkage that can produce
   *non-monotonic* merges. With the chosen toy points the merge distances are
   `0.58, 0.64, 0.65, 0.67, 2.62, 2.50` - the **root merge (2.50) is lower than its child merge
   (2.62)**. I confirmed this both from the coordinates and from the rendered `clu_agglo_anim_5/6`:
   the red root bar is drawn *below* the black bar beneath it. This directly contradicts the mental
   model the very next frame teaches ("The dendrogram: cut the tree" - low cut = many clusters,
   "the largest vertical gap is a natural place to cut"), which assumes merge height only ever
   rises. It also clashes with the deck's own claim two frames later that **Ward is "the common
   default"** and with `clu_dendrogram.pdf`, which *is* built with Ward. A sharp student will notice
   the root bar dipping and be confused.
   - **Cleanest fix:** rebuild the animation with **Ward** (monotonic, never inverts, and matches
     the stated default). Keep the centroid `x` markers as "where each cluster sits" and the dashed
     line as "which two merge"; only relabel the numeric `d` from "distance between midpoints" to
     "Ward merge cost." You lose the pure midpoint-distance story but gain correctness + consistency.
   - **Minimal fix:** keep centroid linkage but nudge the 7 toy points so the last two merges stay
     monotonic (move the top pair closer / the two bottom groups farther apart so the final root
     merge is clearly the tallest). Then the "midpoints closest" story survives intact.

3. **HDBSCAN figure titles fight the teaching point (`clu_hdbscan.pdf`, frame 22/46).** Both panels
   are titled "**-> 3 clusters**" ("DBSCAN, one eps=0.4 -> 3 clusters (sparse group lost to noise)"
   vs "HDBSCAN -> 3 clusters (adapts to each density)"). The headline a student reads is "same
   count, so same result." What the scatter actually shows (I rendered it) is the real lesson:
   DBSCAN captures only a **tiny green core** of the sparse top group and throws the rest to noise
   (`x`), while HDBSCAN swallows the whole top group as one clean cluster. The identical "3
   clusters" numbers actively hide that. Fix: retitle so the contrast is in the words - e.g. DBSCAN
   panel "sparse group mostly lost to noise", HDBSCAN "all three groups recovered" - or tune
   `eps`/`min_samples` so DBSCAN visibly returns a *different* (worse) count. Also the cluster
   colors are permuted between the two panels (top group is green on the left, blue on the right),
   which makes the eye work harder; fix labels to a stable color if easy.

4. **Comparison table: k-medoids "Noise? somewhat" is misleading (frame 39/46).** k-medoids assigns
   **every** point to a cluster; it has no noise/outlier category. It is only *more robust to
   outliers* than k-means (a medoid isn't dragged by extremes). Under a column literally headed
   "Noise?" (where DBSCAN/HDBSCAN get a bold "yes" for genuinely flagging noise), "somewhat"
   conflates robustness with noise-detection. Change to "no (robust)" or similar.

---

## 2. Per-section content review

### Hook + positioning (1-2/46) - good
Cold-open "how many groups?" scatter + the supervised/unsupervised map is exactly right. The
"You can have labels and still cluster" box (sub-structure, label QA, cluster-id features,
semi-supervised) is a genuinely useful framing and it correctly seeds the external-metrics section.
Keep.

### K-means (3-12/46) - the strongest run in the deck
- **Distance metrics frame (5/46):** Euclidean/Manhattan/cosine + the `armred` "scale first" trap.
  The trap is the real payload and is well placed. Minor: Manhattan/cosine are introduced here but
  not *used* until k-medoids ~13 frames later, so there's a small "why are we told this now?"
  Acceptable - it establishes "the metric shapes the clusters." "Manhattan robust to large
  single-feature gaps" is defensible (L1 penalizes a big one-feature gap linearly, L2 quadratically).
- **Objective (6/46):** WCSS formula is correct; the "inertia is not normalized" `armorange` box is
  the right misconception pre-empt and per the outline it's stated exactly once. Good.
- **Lloyd's animation (7/46, 6 overlays):** clear, centroids visibly move, ends at "converged," and
  the "local optimum, same idea as EM" footnote forward-links cleanly. Strong.
- **Worked k=2 by hand (8/46):** I recomputed it - assignments (P1,P2,P5->mu1; P3,P4->mu2) and
  updates (mu1=(1.67,1.33), mu2=(4.5,3.5)) are **all correct**. Good worked-numbers frame.
- **Predict-first: init matters (9/46):** correct use of the device; the two-panel k-means++ vs bad
  random init with inertia numbers (876 vs 1055) makes it concrete. Keep.
- **Elbow (10/46):** honest ("often fuzzy", defers to silhouette). Figure is a clean elbow at k=4.
- **Failure modes (11/46):** the 2-panel figure (elongated + unequal-variance) is the right picture
  and the "reach for k-means when" line closes it well.
- **Mini-batch (12/46):** appropriately one frame. Fine.

### GMM & EM (13-16/46) - correct and appropriately deep
- k-means-as-GMM-special-case is **correctly qualified** ("spherical, equal-variance, hard
  assignments") per the outline fix. The E-step responsibility formula and the M-step
  mean/covariance/weight updates are all standard and correct; "each is the ordinary statistic,
  responsibility-weighted" is a nice way to say it. BIC/AIC box correctly notes it needs a
  likelihood model (GMM, not plain k-means). Two frames of dense formulas, but the audience has seen
  MLE, so this is appropriate depth. No changes needed.

### Other algorithms (17-28/46)
- **k-medoids (18/46):** clear; "any metric / robust / interpretable center" is the right trio.
  Correctly flags it's not in core sklearn (`sklearn_extra`). Placement note: k-medoids is
  conceptually closest to k-means (swap mean for medoid) yet sits after GMM in "other"; defensible
  given the deliberate hard->soft GMM adjacency, but a one-line callback ("remember k-means? just
  swap the mean for a real point") would help.
- **DBSCAN (19-21/46):** the three-point-type frame, the 6-frame growth animation, and the
  eps/min_samples frame (k-distance knee) are all excellent and internally consistent - the
  k-distance knee (~0.15) is the same eps used in the circles figure. The embedded predict-first
  ("what can DBSCAN do that k-means/GMM cannot?") is well placed. min_samples ~ 2xD rule of thumb is
  the standard Sander heuristic. Solid.
- **HDBSCAN (22-23/46):** concept frame good aside from the figure-title issue in (3). The
  "how it works (sketch)" frame (core distance -> mutual reachability -> MST -> condense -> stability)
  is **correct** (mutual reachability = max of the two core distances and the actual distance is the
  right definition) but is the densest, most jargon-heavy frame in the deck and is borderline
  over-scoped for a *first* clustering pass. It's honest ("sketch"), so keep it, but consider a
  one-line framing at the top - "you won't implement this; here's the intuition for *why* it's
  stable" - so students know they're allowed to not fully absorb it.
- **Hierarchical + dendrogram + linkage (24-28/46):** the intro, the 7-frame build-up animation, the
  cut-the-tree frame, the linkage taxonomy, and the two-moons linkage comparison are a very good run
  - except for the inversion/collision issues in (1) and (2), which live in the animation. The
  two-moons figure (single follows crescents; complete/average/Ward cut compact halves) is the
  canonical, memorable picture. Keep the whole run once the animation is fixed.

### Evaluation (29-36/46) - conceptually the highlight
- "No accuracy for clustering" `armred` box and internal-vs-external split are exactly the right
  framing. Silhouette formula `s=(b-a)/max(a,b)` is correct, and the little TikZ a-vs-b schematic is
  a good intuition pump.
- **Rand/ARI worked example (34-35/46):** I recomputed the entire thing. Contingency table (5/2/1/3),
  pairs same-both = 14, same-class = 27, same-cluster = 25, different-both = 17, **Rand = 31/55 =
  0.56**, expected index = 27*25/55 = 12.3, max = 26, **ARI = (14-12.3)/(26-12.3) = 0.13** - **all
  correct**, and the "Rand looks okay but ARI shows barely-above-chance" punchline is the perfect
  reason to prefer adjusted indices. This is the best single teaching frame in the deck. The
  separate "data behind the Rand index" scatter (34/46) that shows the 11 points before the table is
  a thoughtful touch. Keep untouched.
- "Adjusted vs raw NMI rewards more clusters" and "plain accuracy fails - labels are a permutation"
  are both correct and well-stated.

### In practice & beyond (37-46/46)
- **Wider zoo (38/46):** see section 3 below - under-delivers vs its own outline.
- **Comparison table (39/46):** useful; fix the k-medoids "Noise?" cell per (4).
- **Anomaly detection (40/46):** good short bonus.
- **Curse of dimensionality (41/46):** conceptually right, but the plot is a hand-drawn analytic TikZ
  curve `3.2/sqrt(d)` dressed up with axes as if it were measured. Per SLIDE_STYLE ("real/generated
  for anything data-driven; TikZ only for conceptual schematics") this should be a *computed* curve.
  It's ~10 lines to sample uniform points in d dimensions and plot mean `(max-min)/min` vs d -
  cheap, honest, and it lands the point harder because the numbers are real. Low priority but noted.
- **Cluster-after-PCA (42/46):** clean bridge to the next deck. Good.
- **sklearn snippet (43/46):** one canonical `fit_predict` block, per house style. Fine.
- **Caveats (44/46):** the "k-means will happily split pure noise into k tidy clusters" bullet is
  buried here - see illustrations section, it deserves to be a predict-first frame.
- **Image-quantization teaser (45/46):** Saryan before/after (196,680 -> 16 colors) is a great,
  culturally-grounded teaser for the practical. Keep.
- **Recap + Next box (46/46):** complete and correct.

---

## 3. Missing / under-delivered vs the outline

- **Spectral clustering is never mentioned in text.** The outline (frame 23) promised "one line on
  the rest (Spectral for non-convex, MeanShift, AffinityProp, BIRCH, OPTICS)" and "annotate the
  columns we taught." The delivered "wider zoo" frame is the **raw 66-panel sklearn gallery + a
  source line and nothing else**. A student sees a wall of tiny panels with 11 unfamiliar column
  headers and no idea which ones were covered or what Spectral/MeanShift/OPTICS/BIRCH/AffinityProp
  even do. Add a short `armblue` callout beside/under the image: one clause each for the 4-5
  algorithms *not* taught (e.g. "Spectral - graph cut, great for non-convex; MeanShift - finds modes,
  no k; OPTICS - DBSCAN over all eps; BIRCH - streaming/huge-n; Affinity Propagation - picks
  exemplars") and a note like "the columns you already know: MiniBatch KMeans, Ward/Agglomerative,
  DBSCAN/HDBSCAN, Gaussian Mixture." This is the single biggest gap between the outline and the deck.
- **Silhouette "worked by hand" (outline frame 20) was not delivered.** The deck shows the formula
  and a silhouette *plot* but never computes an `s` by hand, even though ARI gets a full by-hand
  treatment. A tiny 3-number example for one point (pick a point, `a=`mean intra distance,
  `b=`mean nearest-other distance, `s=(b-a)/max`) would mirror the ARI frame and cement the metric.
  Optional but cheap and on-brand.

## Missing topics - fine to skip (agree with the cuts)
- Gap statistic / X-means for choosing k (elbow + silhouette are enough for a first pass; a
  one-line "gap statistic also exists" is optional).
- Spectral-clustering *mechanics* (graph Laplacian eigenvectors) - correctly out of scope; a
  one-liner in the zoo is all that's owed.
- DBSCAN having no `.predict` for new points - a real gotcha but fine to leave to the practical; a
  one-line watch-out on the caveats frame would be a bonus, not a requirement.
- Empty-cluster handling in k-means, k-means|| init details, full PAM/CLARA cost - all correctly
  omitted for a first pass.

---

## 4. Illustrations to add or fix

1. **Predict-first: "k-means on pure noise" (new frame, high value).** This is the most
   counter-intuitive and most memorable fact in the whole subject and it's currently a throwaway
   bullet on the caveats frame (44/46). Promote it to a `\pause` predict-first frame near the end of
   evaluation or in the caveats section: "Run k-means (k=4) on **uniformly random** points. What do
   you get?" -> reveal: four tidy, confident, meaningless clusters. Back it with a tiny 1-panel
   figure (uniform scatter, colored into 4 Voronoi wedges with centroids). It drives home "clustering
   always returns *something*; validation is not optional" better than any sentence can. The
   generator already has all the pieces.
2. **Fix the HDBSCAN figure titles** (section 3 of Bugs) so the DBSCAN-fails / HDBSCAN-adapts
   contrast is legible from the titles, not just inferable from the dots.
3. **Fix the agglomerative animation** (Bugs 1+2): kill the caption collision and the dendrogram
   inversion (switch to Ward, or re-place the toy points).
4. **Annotate the wider-zoo gallery** (section 3): a one-line-per-unfamiliar-algorithm callout +
   a "columns you know" note.
5. **Curse-of-dimensionality: replace the analytic TikZ curve with computed points** (~10 lines).
   Optional, low priority, but matches house style and is more convincing.

---

## 5. What is working - do not touch

- **The overall arc and sequencing.** distance -> objective -> Lloyd -> worked -> init -> k selection
  -> failure modes -> GMM (soft generalization) -> density/hierarchy -> evaluation -> curse -> PCA.
  Each idea builds on the last with no leaps. This is the deck's biggest strength.
- **Both by-hand numerics are real and correct** (k=2 iteration; Rand=0.56 / ARI=0.13). The ARI
  frame in particular - Rand-looks-fine-but-ARI-says-chance - is the standout teaching moment.
- **The two 6-frame animations** (k-means Lloyd, DBSCAN growth) - paced right, honest titles,
  internally consistent (DBSCAN eps matches the k-distance knee).
- **The honest callouts**: "inertia is not normalized," "scale first," "no accuracy for clustering,"
  "adjusted vs raw NMI," k-means-as-GMM-limit-qualified. This is what separates the deck from a
  tutorial - keep every one.
- **The two-moons linkage figure** and the **k-means-vs-DBSCAN-on-circles** figure - the two
  canonical "why shape matters" pictures, both present and clean.
- **The Saryan image-quantization teaser** - concrete, local, and a perfect hook into the practical.
- **Section transitions + cold-open hook + recap/Next box** - all present per house style.

---

## 6. Suggested priority order

| # | Item | Effort | Where |
|---|------|--------|-------|
| 1 | Fix agglomerative caption/page-number collision | minutes | frame 25/46 (fig + caption) |
| 2 | Kill the dendrogram inversion (Ward, or re-place toy points) | small | `fig_agglo_anim` |
| 3 | Retitle HDBSCAN figure so DBSCAN-fails/HDBSCAN-adapts reads from the titles | small | `fig_hdbscan` |
| 4 | Add "k-means on pure noise" predict-first frame (+ tiny figure) | small | new frame + generator |
| 5 | Annotate the wider-zoo gallery (taught columns + 1 line on Spectral/MeanShift/etc.) | small | frame 38/46 |
| 6 | Fix k-medoids "Noise? somewhat" cell in the comparison table | minutes | frame 39/46 |
| 7 | Add a one-point by-hand silhouette (mirror the ARI frame) | small | frame 32/46 area |
| 8 | Frame the HDBSCAN mechanism sketch as optional depth (1 line) | minutes | frame 23/46 |
| 9 | Replace curse-of-dim analytic curve with computed points | small | `fig` (new) + frame 41/46 |
