# Practical: land cover of Lake Sevan — outline (approved 2026-08-13)

Second practical for `09_clustering`, alongside the existing k-means image-compression project.
Brainstormed and approved 2026-08-13; built the same day.

## Why a second practical

`35_image_compression_solution.ipynb` exercises roughly a quarter of `32_clustering` (47 frames):
k-means, mini-batch, elbow, silhouette. It cannot exercise the rest, and not by accident —
pixels in an RGB cube have **no labels**, so external evaluation (a full deck section: ARI, AMI,
the label-permutation problem) is unreachable, and quantization never asks *what is this cluster*
because the clusters are colours you are about to throw away.

This practical closes that. Same "cluster the pixels" entry point, so students start from
something familiar, but the clusters are now **things on the ground** that must be named,
argued about, and scored against a real land-cover map.

| Deck concept | Image compression | Land cover |
|---|---|---|
| k-means, k-means++, mini-batch | yes | yes |
| Elbow, silhouette | yes | yes |
| "Scale first" trap | no (RGB is already commensurate) | **yes, and it breaks the map** |
| Cluster interpretation / naming | no | **yes, from mean spectra** |
| ~~GMM, soft assignment~~ | no | *cut 2026-08-26, instructor's call* |
| DBSCAN and its limits | no | **yes, as an honest failure** |
| External metrics (ARI / AMI) | impossible, no labels | **yes, vs ESA WorldCover** |
| Label-permutation problem | no | **yes** |
| Representation choice (NDVI/NDWI) | no | **yes** |

Hierarchical clustering, HDBSCAN and k-medoids do not fit the session and go to the homework
bonus.

## Confirmed decisions (instructor, 2026-08-13)

- **Scene: Lake Sevan only.** Ararat valley was offered as a second, harder scene and declined.
- **ESA WorldCover ground truth is the closing act**, not a bonus.
- **Data ships cached in the repo.** Students never install a geospatial stack.
- **20 m, not 10 m** (see below).
- Kept out of scope for now: which slot this takes on the schedule. `00_plan.md` still shows
  Aug 21 as the image-compression practical; this one is unscheduled pending that call.

## Session arc — 6 acts, ~90 minutes

> Revised 2026-08-26: Act 0 gained the provenance / channels / reflectance material and the
> ground-truth reveal; Act 4 lost the GMM half. Act numbering is unchanged.


**Act 0 — where the data came from, and what the eye cannot see (20 min).**
Where the scene comes from (Earth Search STAC, windowed COG read, the 20 m and window-selection
decisions) and what each of the six channels physically measures. Then reflectance properly:
radiance vs reflectance, what Level-2A means, `DN * scale`, and the offset that the metadata
advertises but physics rejects. Draw true colour, then false colour with near-infrared in the red
channel — vegetation glows, the lake goes black. Finally show the WorldCover ground truth **once**
and put it away until Act 5, so the students know what is down there without being able to peek.

**Act 1 — from image to table (15 min).**
Reshape `(H, W, B)` to `(H·W, B)`. A "point" is now 20 m of Armenia. Scale — and show the map
you get without scaling, where the short-wave infrared bands hijack the partition. The `armred`
trap from the deck, rendered as a broken map.

**Act 2 — k-means and the naming step (20 min).**
k by elbow + silhouette (silhouette on a 10k subsample; it is O(n²) in pairs). Fit, predict all
pixels, reshape labels back to `(H, W)`, draw the map. Then **name each cluster from its mean
spectrum**: near-zero NIR is water, an NIR spike is vegetation, a flat rising curve is bare
ground. Students leave with a legend they wrote themselves.

**Act 3 — representation is the model (15 min).**
Add NDVI and NDWI, recluster, watch the map change. No held-out score can say which feature set
is right, so in unsupervised work the choice of representation *is* the modelling decision.
Callback to ch6 feature engineering.

**Act 4 — an honest failure (10 min).**
DBSCAN on a subsample, where it does badly, and say why: spectral space here is a continuous
gradient with no density valleys, and at ~10⁶ points the O(n²) memory cost rules it out anyway.
Reporting an algorithm that did not work is part of the job.

*(The GMM half — soft assignment, the confidence map, and the "the mixture model knew" reveal —
was cut on 2026-08-26 at the instructor's request. It was a good act; if the session ever needs
a soft-clustering beat, it is recoverable from git history at commit `684b4cf`.)*

**Act 5 — the ground truth (15 min).**
Load ESA WorldCover for the same pixels. Show that accuracy is meaningless (cluster 3 is not
class 3), score with **ARI and AMI**, then read the contingency table: which real classes did the
clustering merge? Closing discussion — *is the clustering wrong, or is the taxonomy asking for a
distinction the spectra do not contain?*

## Files

| File | Role |
|---|---|
| `py_src/fetch_sevan_scene.py` | One-off, instructor-side. Earth Search STAC → crop → WorldCover reprojection → npz. Needs `rasterio` + `pystac-client`. Students never run it. |
| `data/sevan_s2_crop.npz` | The cached crop. Committed. |
| `34_land_cover_solution.ipynb` | Solution notebook. Seed 509, numpy + sklearn + matplotlib only. |
| `09_clustering.qmd` | Gains a second project section. |

## Data spec

- **Source:** Sentinel-2 L2A COGs on AWS (`sentinel-2-l2a-cogs`), via the Earth Search STAC API.
  Free, no account, no requester-pays.
- **Bands:** B02, B03, B04 (blue/green/red), B08 (NIR), B11, B12 (SWIR). Six bands.
- **Resolution: 20 m.** At 10 m a 20 km square is 4M pixels and ~50 MB in the repo. At 20 m the
  scene covers the same ground, the file fits, k-means runs in seconds, and B11/B12 arrive at
  their **native** resolution instead of being upsampled — and those two bands are what separate
  bare soil from built-up from dry grass. The trade buys accuracy rather than costing it.
- **Window:** 1000 × 1000 px = 20 km square over Lake Sevan.
- **Ground truth:** ESA WorldCover v200 (2021), 10 m, 11 classes, CC-BY 4.0, public S3.
  Reprojected onto the Sentinel-2 UTM grid with **nearest neighbour** — anything else invents
  classes that do not exist.

**Scene selection criterion (checkable, not vibes):** summer, < 5 % scene cloud cover, and the
crop must contain **at least 5 WorldCover classes each covering > 2 % of pixels**. If the first
window fails, move the window, not the criterion.

## Compute budget

Sized for a 16 GB laptop with no GPU, per `CLAUDE.md`.

| Step | Size | Cost |
|---|---|---|
| k-means, k ≤ 12 | 10⁶ × 6–8 float32 | seconds |
| Silhouette | 10k subsample | seconds (O(n²) pairs) |
| GMM, full covariance | fit on 100k, predict on 10⁶ | tens of seconds |
| DBSCAN | 20k subsample only | seconds; the point is that it fails |

Nothing here is run in parallel, and nothing needs a GPU.

## Homework (in `09_clustering.qmd`)

Re-run the pipeline with their own choices and defend them: pick k with an argument, name every
cluster, report ARI/AMI for at least two feature sets, find a place where the map is wrong and
explain why. Bonus: agglomerative on a subsample with a dendrogram, `MiniBatchKMeans` on a
larger crop, HDBSCAN, or fetching their own Armenian scene with the script in `py_src/`.

## Verify at build

- Cloud-free summer scene actually exists over Sevan in the archive.
- The chosen window passes the 5-classes-over-2 % test.
- npz lands under ~10 MB.
- WorldCover reprojection is nearest neighbour and the class codes survive it.

---

## What the build found (2026-08-13) — SUPERSEDED, see the 2026-08-26 section below

> **Do not act on the four numbered findings below.** They were measured on a cube whose
> reflectance conversion was wrong (the BOA offset was applied twice). Three of the four
> reversed once it was fixed. They are kept verbatim because the record of what a broken
> pipeline can make you believe is worth more than a tidy file.

All four checks passed: scene `S2A_38TNK_20240902_0_L2A`, 0 % cloud, window `martuni_south`
(40.14 N, 45.30 E) with 5 classes over 2 % — grassland 48.8 %, water 28.3 %, cropland 13.5 %,
tree cover 5.3 %, built-up 3.9 % — and a 9.2 MB npz.

**Four claims in the design above were wrong, and the notebook teaches the measured result
instead.** Recorded here because the wrong versions are the intuitive ones and will be proposed
again.

1. **"Scaling breaks the map."** It does not. On the six bands alone, standardising moves ARI by
   about +0.03; the six features are all reflectance in the same unit, so nothing dominates
   unfairly. Where it *does* bite is once NDVI/NDWI join the table: unscaled they are 4-20x wider
   than the bands and dominate every distance — and that **raises** ARI from 0.487 to 0.559,
   because they are the more informative features. The lesson became *scaling equalises
   influence, which only helps when the loud features were not the useful ones.*
2. **"ARI peaks at 5, matching the 5 classes."** It peaks at **4 or 5** depending on feature set,
   and the single best score in the sweep is `k=4`. Consistent with grassland and cropland being
   spectrally inseparable, so four is the honest number of *findable* classes.
3. **"GMM beats k-means."** A tie at `k=5` (0.487 vs 0.483). It wins clearly at `k=6`
   (0.545 vs 0.446). The notebook shows both.
4. **"Shoreline pixels are the uncertain ones."** The shoreline gets its own component. The
   uncertainty concentrates in **Martuni and the field mosaic**, and — measured after the labels
   are revealed — in grassland and cropland, the exact pair the clustering cannot separate. This
   turned into a better act than the designed one: the GMM flags the weak distinction *before*
   any label is loaded. Built-up is the honest counterexample, confidently assigned and wrong.

**The best result in the whole notebook uses no raw bands at all:** three indices, `k=4`,
unscaled, ARI 0.589, against 0.484 for the six-band model students build first.

**The closing act is stronger than designed.** ARI ≈ 0.48 collapses to ≈ 0.16 when the water is
removed, and refitting on land only recovers it to 0.19 at best. Nearly all of the headline score
was one easy split worth 28 % of the pixels.

---

## What the rebuild found (2026-08-26)

Two changes: the GMM act was cut on the instructor's request, and **a reflectance bug was found
and fixed**. The second one invalidated most of the section above.

### The bug

`fetch_sevan_scene.py` read `scale` and `offset` from the scene's STAC `raster:bands` metadata
and applied `reflectance = DN * scale + offset`. The metadata advertises `offset = -0.1`, correct
for a raw post-baseline-04.00 L2A product. But Element84's `sentinel-2-l2a` COGs are **already
baseline-harmonised**, so the offset had been folded in, and applying it again shifted every band
down by 0.1.

The evidence, medians per WorldCover class:

| class | B04 / B08 with offset | B04 / B08 without |
|---|---|---|
| Tree cover | **-0.069** / 0.149 | 0.031 / 0.249 |
| Grassland | -0.014 / 0.128 | 0.086 / 0.228 |
| Water | **-0.086 / -0.090** | 0.014 / 0.010 |

**60.0 % of all values in the cube were negative.** Reflectance is a ratio of light out to light
in; it cannot be negative. The no-offset column matches textbook spectra for all five classes at
once.

The npz itself was never wrong — only the conversion applied to it — so no refetch was needed.
`verify_reflectance()` now checks the advertised conversion against physics before trusting it,
and refuses to proceed if neither variant is physical. It would have caught this on day one.

### What the bug had been teaching

NDVI is computed after `np.clip(refl, 0, None)`. With the offset applied, red clipped to zero on
vegetated ground, so NDVI collapsed to `NIR / NIR` = **1.0** for tree cover *and* grassland alike.
The "vegetation index" was a saturated water/land mask. Correct values: 0.78 and 0.45.

### Corrected findings — these replace 1-4 above

1. **Scaling.** The old finding said standardising *hurt* once indices were added, and concluded
   the deck's "scale first" rule was too simple. **Backwards.** Scaling now helps at every `k` on
   the mixed set (+0.035 to +0.079 ARI), does nothing where columns already share a unit (RGB
   moves ≤0.002), and lands in between on the six bands (-0.015 to +0.036). **The deck's `armred`
   trap frame is correct as written** and needs no revision: scaling matters in proportion to how
   unequal the spreads were.
2. **`k`.** Unchanged and still true: ARI peaks at 4 or 5 in every row, the elbow's range, not the
   silhouette's `k=2`.
3. **Feature sets tie.** The old "indices only, `k=4`, unscaled, ARI 0.589, the best result in the
   notebook" was the saturated-NDVI artifact. Corrected, the top three are **0.483 / 0.482 /
   0.479** — six bands, three indices, and all nine columns, in a dead heat. What separates a good
   row from a bad one is whether infrared is present at all (RGB trails by 0.04-0.10), not how it
   is recombined.
4. **Silhouette vs features.** The old claim that the silhouette ranked feature sets in ARI's
   order was false even before the fix (it compared the scaled `k=5` silhouette against
   best-over-sweep ARI). It is false now too. The honest statement: the real gaps are ~0.004 ARI,
   below what any internal metric resolves.

### Unaffected, and verified bit-identical after the refit

Adding a constant to every band is invariant under Euclidean distance and under standardisation,
so `Xs` never changed. Cluster shares, all five mean spectra (shifted +0.1, NIR-red steps
identical), cluster naming, the elbow and silhouette curves, the DBSCAN table, water precision
0.999 / recall 0.997, the contingency table, and the closing collapse (0.484 → 0.155 on land only)
all came back unchanged. **The strongest act in the notebook was never at risk.**

### Also changed

- **GMM cut** (Act 4 is now DBSCAN alone). Removed the confidence map, the "mixture model knew"
  act, and the k-means-vs-GMM comparison. The conclusion table lost its *soft assignment* and
  *k-means assumptions* rows, the latter having had no evidence left without the GMM.
- **New up-front sections**: where the scene comes from and how it was cropped; a per-band table
  of what each of the six channels measures; a proper account of reflectance; and a ground-truth
  cell showing WorldCover in ESA's official class colours before any clustering is done.
- **NDBI is not a wide feature.** At std 0.139 it is only 1.2x B11's 0.114. Only NDVI and NDWI
  dominate unscaled distances.

### Rule this reinforces

`_learnings/2026-08-13-2015_write-the-practical-after-measuring-not-before.md` said design docs
state what to *investigate*, not what will be *found*. This round adds the harder half:
**measuring is not enough if the measurement itself is unvalidated.** Every one of those four
findings was honestly measured. They were measured on bad data, and nothing in the write-prose-
after-measuring discipline catches that. Physical sanity checks on the input belong upstream of
the analysis, in the fetch script, where they now are.

