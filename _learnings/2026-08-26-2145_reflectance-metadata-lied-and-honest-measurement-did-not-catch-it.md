# Product metadata can be wrong, and measuring honestly does not catch it

**Symptom.** The Sevan land-cover practical shipped with `reflectance = DN * scale + offset` and
taught four measured findings from it. Every value in the cube was 0.1 too low. Open water came
out at about **-0.09 in all six bands**, and 60.0 % of the whole cube was negative. Three of the
four findings reversed once it was fixed.

**Cause.** `fetch_sevan_scene.py` read `scale` and `offset` from the scene's STAC `raster:bands`
metadata, which advertises `offset = -0.1`. That is correct for a **raw** post-baseline-04.00
Sentinel-2 L2A product. It is wrong for Element84's `sentinel-2-l2a` COGs on AWS, which are
already **baseline-harmonised** — the offset has been folded into the stored DNs. Applying it
again subtracts 0.1 twice.

The metadata was not lying about the product family. It was lying about *these files*.

**Evidence.** Median DN converted both ways, per WorldCover class:

| class | B04 / B08 with offset | B04 / B08 without |
|---|---|---|
| Tree cover | **-0.069** / 0.149 | 0.031 / 0.249 |
| Grassland | -0.014 / 0.128 | 0.086 / 0.228 |
| Built-up | 0.011 / 0.139 | 0.111 / 0.239 |
| Water | **-0.086 / -0.090** | 0.014 / 0.010 |

The no-offset column matches textbook spectra for all five classes simultaneously. That is not a
coincidence you can get by accident.

**Why nobody noticed.** The notebook *rationalised it*. It carried a markdown cell explaining
that slightly negative values over dark water are expected, because atmospheric correction is an
estimate that can overshoot on an almost-black surface. That is a true sentence about real data.
It is also exactly the sentence that makes a systematic -0.1 shift look like a known quirk. **A
plausible explanation for an artifact is more dangerous than no explanation**, because it stops
the search.

**What it had been teaching.** NDVI is computed after `np.clip(refl, 0, None)`. With the offset
applied, red clipped to zero over vegetated ground, so NDVI collapsed to `NIR / NIR` = **1.0** for
tree cover and grassland alike. The "vegetation index" was a saturated water/land mask, its std
inflated to 0.436. Correct values: 0.78 for trees, 0.45 for dry September grass, std 0.318. That
one artifact produced the notebook's headline result — "throw the bands away, cluster on three
ratios, ARI 0.589, the best score in the whole sweep" — which does not survive: corrected, the
top three feature sets tie at 0.483 / 0.482 / 0.479.

**The blast radius was smaller than it looked, and provably so.** Adding a constant to every band
is invariant under Euclidean distance *and* under standardisation, so `Xs` never changed. Cluster
labels, mean-spectrum shapes, the elbow, DBSCAN, the contingency table and the closing ARI
collapse all came back bit-identical. Only the index-derived features moved. Worth deriving that
before re-running everything in a panic — it turns "every number is suspect" into "these two rows
are suspect".

**Fix.** `verify_reflectance()` in `fetch_sevan_scene.py` now checks the advertised conversion
against physics before trusting it: reflectance is a ratio of light out to light in and cannot be
negative, so if the advertised offset drives more than 0.1 % of values below zero it is dropped,
with a loud `log.warning`. If *neither* variant is physical it raises. Tested against the real
cube: 60.04 % negative with, 0.0000 % without.

**Rule.** `2026-08-13-2015_write-the-practical-after-measuring-not-before.md` says design docs
state what to *investigate*, not what will be *found*. This is the harder half:

> **Measuring honestly is not enough if the measurement itself is unvalidated.**

All four original findings were honestly measured, prose-diffed against real printed outputs, and
wrong. No amount of write-prose-after-measuring discipline catches a bad input. Validation has to
sit **upstream of the analysis**, in the ingestion code, as an assertion about a physical
invariant — not as a sentence in a notebook explaining why the weird thing is fine.

Cheap invariants worth asserting at ingestion: reflectance and probabilities in `[0, 1]`,
counts non-negative, rates in `[0, 1]`, distances non-negative and symmetric, an angle in range,
a total that must equal its parts. If a quantity has a physical range, assert it where the data
enters, and fail loudly.

See `ml/09_clustering/34_land_cover_OUTLINE.md`, "What the rebuild found (2026-08-26)", and
`DECISIONS.md` #28.
