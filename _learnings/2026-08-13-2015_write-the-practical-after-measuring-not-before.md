# Write a practical's prose after measuring, not before

**Symptom.** The approved design for the Sevan land-cover practical named four teaching moments
up front. When the pipeline was built and run, **four of them were false** and one was backwards.
Had the notebook been written from the design, it would have shipped confident prose asserting
things the code printed the opposite of, three cells below.

**Cause.** Every one of the four was the *intuitive* reading of the lecture, applied to a dataset
nobody had looked at yet:

| Designed claim | What the data said |
|---|---|
| "Scaling breaks the map" (the `armred` trap) | Scaling barely moves ARI on six reflectance bands (+0.03). It only matters once NDVI/NDWI are added — and there it **hurts**, -0.10, because those indices are 4-20x wider than the bands and dominating is the *right* thing for them to do |
| "ARI will peak at k=5, matching the 5 WorldCover classes" | Peaks at k=4 or 5 by feature set; the single best score is k=4, because grassland and cropland are not spectrally separable |
| "GMM beats k-means" | Tie at k=5 (0.487 vs 0.483); wins only at k=6 (0.545 vs 0.446) |
| "The shoreline pixels are the uncertain ones" | The shoreline gets its own component. Uncertainty concentrates in the town and the field mosaic |

**What worked.** Two exploratory scripts in the scratchpad before a single markdown cell was
written, then a rule for the notebook itself: **every number in the prose is printed by a cell in
the same notebook**, and the prose only makes qualitative claims that were checked against a real
run. After executing, dump every text output and re-read the markdown against it. That pass
caught three more overstatements the exploration had not (`built-up mostly disappears` when it
was a 43/38 split; `refitting on land does better` when it only does at k=3).

**Consequences.** The measured versions were better teaching than the designed ones. "Scale
first" as a rule is weaker than "scaling equalises influence, and equal influence only helps when
the loud features were not the useful ones" — and the second one cannot be invented at a desk.
The GMM finding got a whole extra act out of it: the confidence map flags the grass/crop pair as
shaky *before* any label is loaded, which is the closest thing to validation that exists when you
have no labels.

**Rule.** For any practical built on real data: explore first, write prose second, execute, then
diff the prose against the outputs. Design documents state what to *investigate*, not what will
be *found*. The record of what the design got wrong belongs in the outline file too — the wrong
versions are the intuitive ones and will be proposed again.

See `ml/09_clustering/33_land_cover_OUTLINE.md`, section "What the build found".
