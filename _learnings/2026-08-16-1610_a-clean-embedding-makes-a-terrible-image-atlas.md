# A clean embedding makes a terrible image atlas

**Symptom.** Building the "here is what DR is actually for" frame in 34_dimensionality_reduction: a UMAP of 2000
Imagenette photos through CLIP, drawn with the photographs themselves instead of dots. Asked for
220 thumbnails at random positions. About 15 were visible. Raising the count to 281 changed
nothing.

**Cause.** Two compounding problems, both caused by the embedding being *good*.

1. **Random sampling stacks.** CLIP separates ten Imagenette classes almost perfectly, and UMAP at
   the default `min_dist=0.1` packs each class into a near-point. 220 random thumbnails landed on
   about ten spots, and matplotlib drew all of them - each hidden under the last. Nothing was
   missing; 205 of them were underneath.
2. **A grid over the bounding box does not fix it on its own.** Switching to one thumbnail per
   occupied grid cell only lifted the count to 39 of 468 cells, because a handful of outliers
   stretch the raw min/max range so the ten balls fall in a few cells. Percentile bounds (1st/99th)
   barely helped: 39 -> 39. The balls really are that tight.

**Fix, both halves needed:**

- **Loosen the embedding for the picture.** `n_neighbors=30, min_dist=0.8` spreads each class from
  a dot into a visible patch: 252 of 816 cells occupied on a 34x24 grid, against 39 before. This is
  legitimate rather than a cheat - `min_dist` changes only how tightly points may pack, not who is
  whose neighbour, which is exactly the point the 35_umap `min_dist` frame makes.
- **Select on a grid, not at random**, so coverage is even across the map.

Also: **drop the legend.** The thumbnails cover the coloured scatter completely, so the colour key
maps to nothing the viewer can see - and the whole point of an atlas is that you read the groups
off the images without a key.

**Consequences.** Any "plot the data as thumbnails" figure needs the embedding tuned *for
legibility*, separately from the embedding used to make an analytical point. The settings that
best show cluster separation are the settings that make the worst atlas: the better the
separation, the smaller and denser each clump, and thumbnails need area.

Numbers measured 2026-08-16, `ml/10_dimensionality_reduction/py_src/dimred_demos.py:fig_clip_atlas`.

Related: [[2026-08-16-1545_umap-sigma-search-has-no-solution-at-k-2]],
[[2026-08-14-0250_verify-interactive-html-in-a-browser-not-by-grep]]
