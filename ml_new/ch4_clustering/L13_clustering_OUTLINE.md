# L13 Clustering — Deck Outline / Design (DRAFT SEED)

Planning seed for the **Clustering** deck, house style of
`ml_new/02_main_concepts_continued/04_overfitting_cross_validation.tex`.

> STATUS: DRAFT SEED. This deck has NOT had the full research + review pass the
> trees/classification outlines got. The non-GMM sections below are placeholder
> bullets (from the README scope: k-means / DBSCAN / hierarchical / evaluation);
> the **GMM + EM** section is the detailed addition requested. Do a full pass before
> building.

## Scope (from README)
Unsupervised clustering: k-means, **Gaussian Mixture Models + EM (added)**, DBSCAN,
hierarchical, cluster evaluation. (L13b PCA / dim reduction is a separate deck.)

## Frame-by-frame seed

### Hook / framing
1. What is clustering — unsupervised, no labels; find structure/groups in data.
   Examples (customer segments, image colors, document topics).

### Section 1 — k-means (placeholder, needs full pass)
2. Centroids + Lloyd's algorithm (assign -> update -> repeat).
3. Choosing k (elbow / inertia); assumptions: spherical, equal-size clusters; must
   pick k; sensitive to init (k-means++).

### Section 2 — Gaussian Mixture Models + EM (DETAILED — the requested addition)
4. **Soft clustering (the interesting idea)** — k-means says "you ARE in cluster A";
   GMM says "70% A, 30% B." Model each cluster as a **Gaussian** with its own mean AND
   covariance -> clusters can be **elliptical and different sizes**, unlike k-means'
   round, equal blobs. **Predict-first:** run k-means on two overlapping elliptical
   blobs — what goes wrong? (It forces round, hard boundaries; GMM fixes both.)
5. **The EM algorithm (the transferable gem)** — chicken-and-egg: you need cluster
   assignments to estimate the Gaussians, but you need the Gaussians to assign points.
   Solve by alternating: **E-step** (given current Gaussians, compute each point's soft
   "responsibilities") <-> **M-step** (given responsibilities, re-estimate each
   Gaussian's mean/covariance/weight), repeat until convergence. EM recurs across ML
   (any latent-variable model) — that is the reusable idea, not just GMM.
6. **k-means is a special case of GMM** — hard assignments + equal spherical covariance
   = k-means. ("k-means = GMM with hard edges and round clusters.") Choosing the number
   of components: GMM is probabilistic, so use **BIC / AIC** (vs k-means' elbow).
   Where it helps: overlapping / elliptical / different-size clusters; density
   estimation; anomaly detection (low-probability points). Limits: assumes Gaussian
   clusters; sensitive to init; can over/under-fit covariance.
   Visual: GMM's elliptical contours + soft boundary vs k-means' Voronoi cells.

### Section 3 — DBSCAN (placeholder, needs full pass)
7. Density-based: finds arbitrary-shaped clusters, no k needed, labels noise/outliers;
   params eps + minPts; predict-first: "what does DBSCAN do that k-means/GMM can't?"
   (non-convex shapes, automatic outlier flagging).

### Section 4 — Hierarchical (placeholder, needs full pass)
8. Agglomerative (bottom-up) merging; the dendrogram; linkage (single/complete/average/
   Ward); cut the tree to choose #clusters.

### Section 5 — Evaluation (placeholder, needs full pass)
9. Internal metrics (no labels): silhouette score, Davies-Bouldin, Calinski-Harabasz;
   inertia/elbow. The hard part: clustering has no ground truth.

### Wrap-up
10. Recap — when to use which (k-means fast/round; GMM soft/elliptical; DBSCAN
    shapes+noise; hierarchical dendrogram) + paramgreen box.
11. HW — run k-means, GMM, DBSCAN on a 2D toy + a real dataset; compare boundaries;
    show k-means failing on elliptical/overlapping blobs where GMM succeeds; pick k via
    elbow vs BIC; bonus: GMM for anomaly detection (flag low-probability points).

## Callbacks
- **L01c** scaling (distance-based clustering needs it). **L01d** model selection
  (choosing k by a criterion). **L12b** (GP/LDA Gaussians; EM as a probabilistic idea).
  **L13b** PCA (dim-reduce before clustering high-dim data).

## Sources (GMM/EM, web-verified)
- GMM = soft clustering via per-cluster Gaussians; EM alternates E-step
  (responsibilities) and M-step (re-fit params); k-means = hard/spherical special case;
  use BIC/AIC for #components. (sklearn mixture docs; standard references.)
- VERIFY at build (sklearn): `GaussianMixture` (n_components, covariance_type,
  `.bic()`/`.aic()`, `.predict_proba`), `KMeans`, `DBSCAN`, `AgglomerativeClustering`.

## Changelog
- SEED created; **GMM + EM added as a detailed Section 2** (soft clustering, the EM
  alternation, k-means-as-special-case, BIC, where-it-helps). Rest is placeholder
  pending a full research + review pass.
