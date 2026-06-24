# L13 Clustering — Deck Outline (FINAL v4, post-review 2026-06-23)

House style per `ml/SLIDE_STYLE.md`. `slide-style` skill workflow: interview (4 rounds)
+ sklearn-docs research + independent review -> this outline -> build after approval.

## Confirmed decisions

- **Focus: k-means**; **GMM/EM moved to right after k-means** (hard-vs-soft contrast while fresh;
  GMM generalizes k-means) but kept brief. **k-medoids** one frame (`sklearn_extra`). **DBSCAN** full
  + **HDBSCAN** short frame. **Hierarchical** full. **Spectral** = one line in the "wider zoo".
- **Keep all frames** (no trims - instructor's call). ~31 content frames + 5 transitions.
- Every algorithm frame ends with a one-line **"Reach for it when..."**.
- **"You can have labels and still cluster"** on the positioning frame, linked to external eval.
- **Evaluation** = full section; prefer chance-adjusted **ARI/AMI** over raw NMI; name FMI.
- Examples: **per-frame toy 2D blobs** (no single threaded example) + the full **sklearn gallery**
  image, framed as "the wider zoo". PCA = one pointer frame (`L13b`). Caveats frame kept.
- Code: one `fit_predict` snippet (note `random_state`). Practical teaser: image compression.
- File: overwrite `L13_clustering.tex`.

## Review fixes folded in
- GMM moved up (own short section after k-means).
- "k-means = special case of GMM" qualified: **spherical, equal-variance, hard-assignment limit**.
- Frame "objective" no longer says "pick k" (choosing k is its own frame); "inertia not normalized"
  stated **once**.
- BIC/AIC frame notes it needs a likelihood model (GMM), not plain k-means.
- "Scale first" promoted to an `armred` trap callout.
- Figures: Lloyd's shows **multiple iterations** (centroids moving); DBSCAN figure shows **k-means vs
  DBSCAN on the same moons**; k-means-fail is **2-panel** (elongated + unequal-size).
- sklearn gallery: BSD-3, add source line + URL (confirm at build).

## Frame-by-frame

### Hook + positioning
1. **Cold-open.** Unlabeled toy scatter - "how many groups?". Bridge: supervised had `y`; not here.
2. **Positioning (ML map).** Supervised vs unsupervised; clustering = groups, dim-reduction = axes.
   **You can have labels and still cluster:** sub-structure within a class, label QA / mislabel
   detection, cluster-id features for a supervised model, semi-supervised. (Links to frame 21 external eval.)

[Outline frame]

### Section 1 - k-means (focus)  [transition]
3. **Distance metrics.** Euclidean / Manhattan / cosine; metric matters. `armred` trap: **scale first**.
4. **The objective.** WCSS / inertia; lower is better but **inertia is not normalized** (relative only). (Choosing k -> frame 8.)
5. **Lloyd's = coordinate descent on WCSS.** assign <-> update; figure: **several iterations, centroids moving**.
6. **Worked-by-hand iteration.** ~5 points, k=2.
7. **Init & local optima.** *Predict-first:* two seeds -> same answer? -> k-means++. Figure: good vs bad init.
8. **Choosing k.** Elbow on inertia (silhouette teased -> frame 20).
9. **Assumptions & failure modes.** Convex/isotropic/equal-size; figure: **2-panel** (elongated + unequal-size).
   *Reach for k-means when:* large n, round/equal clusters, can pick k, fast baseline.
10. **Mini-batch k-means.** Scaling to large data. *When:* same as k-means but n huge.

### Section 2 - from hard to soft: GMM & EM (brief)  [transition]
11. **GMM (soft clustering).** k-means: "you ARE in A"; GMM: "70% A". Gaussians (mean+cov) -> elliptical,
    different sizes. **k-means = the spherical, equal-variance, hard-assignment limit of GMM.**
    Figure: GMM ellipses vs k-means Voronoi. *When:* overlapping/elliptical clusters, soft memberships, density estimation.
12. **EM in one frame.** E-step (responsibilities) <-> M-step (re-fit mean/cov/weight); converges to a local
    optimum. Choose #components by **BIC/AIC** (a likelihood criterion - works for GMM, not for plain k-means).

### Section 3 - other algorithms  [transition]
13. **k-medoids / PAM.** Medoid = real point; any metric; robust to outliers. (`sklearn_extra`.)
    *When:* non-Euclidean metric, or outlier-robust / interpretable centers.
14. **DBSCAN.** core/border/noise; `eps`+`min_samples`; arbitrary shapes, no `k`, flags outliers.
    Figure: **k-means vs DBSCAN on the same two-moons**. *Predict-first.* *When:* odd shapes, unknown #clusters, want outliers.
15. **DBSCAN: choosing `eps` + limits.** k-distance elbow; varying density; O(n^2) memory.
16. **HDBSCAN (short).** Fixes both DBSCAN pains: no `eps`, handles variable density; hierarchy of densities.
    Core sklearn (1.3+). *When:* DBSCAN-style but density varies / eps untunable.
17. **Hierarchical.** Agglomerative + dendrogram; cut the tree. Figure: dendrogram.
    *When:* nested structure / a dendrogram, explore #clusters, connectivity constraints; smaller n.
18. **Linkage.** single / complete / average / Ward - figure: all four on same data.

### Section 4 - cluster evaluation (attention)  [transition]
19. **No ground truth.** Internal (no labels) vs external (labels exist - the frame-2 cases).
20. **Internal metrics.** Silhouette (worked by hand), Davies-Bouldin, Calinski-Harabasz, elbow. Figure: silhouette plot.
21. **External metrics.** When labels exist: prefer chance-adjusted **ARI / AMI** over raw NMI; V-measure, FMI;
    why plain accuracy fails (label permutation).
22. **In practice.** Pick k by silhouette vs elbow; check stability; scale.

### Section 5 - practice & beyond  [transition]
23. **The wider zoo.** sklearn 6x11 gallery (BSD-3, attribute + link); annotate columns we taught
    (k-means/MiniBatch, DBSCAN/HDBSCAN, Ward/Agglomerative, GMM); one line on the rest
    (Spectral for non-convex, MeanShift, AffinityProp, BIRCH, OPTICS).
24. **Comparison table.** algorithm x (needs k? / shape / handles noise / scalability / use it when).
25. **Anomaly detection.** DBSCAN noise + GMM low-probability points as outlier flags.
26. **Curse of dimensionality.** Distances stop discriminating in high-D -> PCA first (sklearn-recommended).
27. **Cluster after PCA.** Reduce then cluster; pointer to `L13b`.
28. **One canonical sklearn snippet.** Shared `fit_predict` pattern (set `random_state`).
29. **Clustering is treacherous (caveats).** Depends on scaling / k / metric / algorithm; no ground truth;
    easy to over-read structure that is not there (k-means on uniform noise still returns k "clusters").
30. **Next time - the practical: shrinking image file size with k-means.** Color quantization: a photo has
    up to ~16M colors; run k-means on the pixels in RGB, repaint each with its centroid -> only k colors ->
    smaller palette / file. Built end-to-end in the practical. Figure: original vs k=16 (before/after).
31. **Recap - when to use which** + `paramgreen` **Next:** box (points to the image-compression practical).

## Figures (`py_src/cluster_demos.py` -> `fig/`, toy blobs, `ma` venv)
- clu_hook_unlabeled, clu_kmeans_steps (multi-iteration), clu_kmeans_init, clu_elbow,
  clu_kmeans_fail (2-panel), clu_dbscan_moons (k-means vs DBSCAN), clu_dendrogram,
  clu_linkage (4-up), clu_gmm_vs_kmeans, clu_silhouette, clu_image_quantization (orig vs k=16)
- Download: `clu_sklearn_gallery.png` (sklearn comparison; BSD-3, attribute + link)

## Sources / verify at build
- sklearn clustering docs: https://scikit-learn.org/stable/modules/clustering.html
- gallery: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
- HDBSCAN in core since 1.3; KMedoids = `sklearn_extra.cluster`; ARI/AMI chance-adjusted.
- API: `KMeans`/`MiniBatchKMeans`, `DBSCAN`/`HDBSCAN`, `AgglomerativeClustering` +
  `scipy.cluster.hierarchy`, `GaussianMixture` (.bic/.aic), `silhouette_score/samples`,
  `davies_bouldin_score`, `calinski_harabasz_score`, `adjusted_rand_score`, `adjusted_mutual_info_score`.
