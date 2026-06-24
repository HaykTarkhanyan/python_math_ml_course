# Dimensionality Reduction — Deck Outline (v6, 2026-06-23)

House style per `ml_new/SLIDE_STYLE.md`. `slide-style` workflow: interview (4 rounds) +
two independent reviews -> outline -> build after approval. v5 folded in the clustering-build
lessons; v6 applies a second independent review.

## v6 review fixes
- **PCA animation is a teaching SWEEP, not PCA's mechanism.** Frame 4 reframed: sweep a direction
  and show `w^T Σ w` peak; caption that PCA does NOT iterate/search -- eigenvectors give the axes
  directly. Avoids students importing the k-means "iterate to converge" model onto PCA.
- **Keep top-k, components are sorted.** Frame 4 notes we keep the top-k axes (2D->1D here; 64-D->2D
  for digits); frame 7 notes PCs come out ordered by variance.
- **Reconstruction bridge stated** on frame 7 (max-variance == min reconstruction error) so frame 13
  doesn't use "reconstruction" cold.
- **Digits dataset introduced** on frame 1 before the embedding frames depend on it.
- **Perplexity figure**: keep n >= ~300; verify the three perplexities actually differ (low end clumps
  most); cite Wattenberg et al. (2016).

## Lessons from the clustering build, folded in
- **Step-by-step animation** (you liked the k-means/DBSCAN overlays): animate **PCA finding the
  max-variance axis and projecting onto it** (overlay frames `dr_pca_anim_*`).
- **On-slide citations** (like the DBSCAN SE link): t-SNE (van der Maaten & Hinton 2008), UMAP
  (McInnes et al. 2018), caveats (distill.pub), curse of dim (Beyer et al. 1999) -- small source lines.
- **Show, don't just tell, the gotcha**: a **t-SNE perplexity-sensitivity** figure (same data, 3
  perplexities) on the caveats frame -- the analogue of the clustering failure-mode figures.
- **Worked example paired with a figure** (like the k-means by-hand): the by-hand PCA frame keeps
  the 2x2 covariance numbers next to the axes figure.
- **Plain language** (no big words); separated/clean toy data for the geometry figures.

## Confirmed scope (from interview + review)
PCA workhorse (full derivation + SVD + eigen refresher); t-SNE & UMAP viz-only WITH key formulas;
dedicated curse-of-dim frame; dedicated viz-caveats frame; dedicated "when NOT to reduce" frame;
kernel-PCA + autoencoder = one-liners; by-hand PCA = covariance->axis (no eigen-by-hand); toy + digits;
skip big-data PCA variants + PCA-preinit tip; no app closer. UMAP via `umap-learn` (approved).
Do NOT reference i2dl manifold slides. Deck file: supersede the `L13b_pca_dim_reduction` stub here.

## Frame-by-frame (~26 content + outline + 5 transitions; animations add overlay pages)

### Hook + why
1. **Cold-open.** "How do you *look* at 64-dimensional data?" digit=8x8=64; gene expr ~20k; embeddings ~768.
   Bridge from clustering: unsupervised - clustering groups *rows*, DR compresses *columns*.
   **Running dataset:** sklearn's 1797 handwritten digits (8x8 grayscale, labels 0-9), used for the embeddings later.
2. **Why reduce dimensions.** Visualization, preprocessing/compression, denoising (drop low-variance noise, *often*).
3. **Curse of dimensionality.** *Predict-first* (`\pause`): "in 100-D, how much closer is the nearest point
   than the farthest?" -> almost the same; distances **concentrate** (max/min ratio -> 1), distance-based
   methods degrade. Figure: distance distribution vs dimension. {\scriptsize cite Beyer et al. 1999.}

[Outline frame]

### Section 1 - PCA: the idea & derivation  [transition]
4. **PCA intuition (animated).** Overlay sweeping a direction through the 2D cloud, showing the
   **projected variance `w^T Σ w` rise and fall with the angle**, peaking at the max-variance axis;
   then **project** onto it. Caption: *PCA does not search or iterate -- the eigenvectors give these
   axes directly (next frames); the sweep just shows what ``maximum variance'' means.* We keep the
   **top-k** axes (here 2D->1D; for 64-D digits we keep 2). Figures `dr_pca_anim_1..5`.
5. **Eigen refresher.** `A v = lambda v` (direction unchanged, length scaled by `lambda`); a **symmetric**
   matrix (like a covariance) has real eigenvalues and **orthogonal eigenvectors** - that is *why* PCs are
   orthogonal. No computation. TikZ schematic.
6. **Derivation (1): max variance.** **Step 0 - center the data** (without it `X^T X` measures spread about
   the origin, not the data). Variance along unit `w` = `w^T Σ w` (`Σ` = covariance of centered data);
   maximize s.t. `||w||=1` -> Lagrange -> **`Σ w = lambda w`**: PCs = eigenvectors of covariance,
   `lambda` = variance captured.
7. **Derivation (2): eigenvalues -> explained variance.** Eigenvalues sorted **descending** -> PCs come
   out **ordered by variance**; top-k eigenvectors = the PCs; **explained-variance ratio**
   `lambda_i / sum lambda` (a small worked number). Note: maximizing variance **is** minimizing
   **reconstruction error** (project then lift back loses the least) -- the bridge to frame 13.
8. **The SVD connection.** PCA = **SVD of centered data** (`X = U S V^T`, cols of `V` = PCs); numerically
   better. **TruncatedSVD / LSA** for sparse text - it does NOT center, so it preserves sparsity.
9. **By-hand 2D PCA.** Columns: the actual **2x2 covariance matrix (numbers)** + figure with **both
   eigenvector arrows scaled by `sqrt(lambda)`** on the scatter (paired numbers+picture, like the k-means worked frame).

### Section 2 - PCA in practice  [transition]
10. **Centering vs scaling.** PCA **always centers** (sklearn does it for you); it does **not** standardize
    - your call. `armred` trap: variance is scale-dependent, so standardize unless features share units.
11. **Choosing #components.** Scree plot + cumulative explained variance (keep ~95%). Figure.
12. **Interpreting components.** Loadings + **biplot** (samples + feature arrows); what a component
    "means". Figure must be clearly readable -- else fold loadings into frame 11.
13. **Reconstruction, compression & denoising.** Project to top-k, then lift back. Figure: digits at
    k=5/20/50 (compression) + a denoise panel.
14. **Pitfalls.** Not feature selection (PCs are *combinations*); **linear only**; outlier/scale sensitive;
    components can be uninterpretable.

### Section 3 - nonlinear DR for visualization: t-SNE & UMAP  [transition]
15. **Why nonlinear.** PCA is linear - can't unfold a curved manifold. **2-panel figure:** PCA flattens the
    swiss roll (color gradient scrambled) vs **UMAP unrolls it** (gradient preserved). Goal: preserve
    **local neighborhoods**.
16. **t-SNE.** High-D similarities **`p_ij = (p_{j|i}+p_{i|j})/2n`**; low-D **`q_ij`** with a **Student-t**
    kernel (heavy tail fixes the "crowding problem"), normalized over all pairs; minimize **KL(P||Q)**;
    iterative/stochastic; **perplexity** ~ effective #neighbors (typ. 5-50). Figure: digits t-SNE.
    {\scriptsize cite van der Maaten & Hinton 2008.}
17. **UMAP.** **Fuzzy nearest-neighbor graph** laid out by minimizing a graph **cross-entropy**;
    **`n_neighbors`** (local<->global), **`min_dist`** (cluster tightness). Faster, more global. Figure:
    digits UMAP. {\scriptsize cite McInnes et al. 2018.}
18. **Caveats (dedicated) - show it.** Figure: **same digits, t-SNE at three perplexities** (low / 30 /
    high) -> the picture changes a lot (the low end clumps most strikingly). *Build: keep n >= ~300 so
    perplexity < n; verify the three actually differ, else use 5 / 30 / 50.* So: **cluster sizes &
    distances not meaningful**; **stochastic**; perplexity/`n_neighbors` sensitivity; **don't feed
    embeddings to downstream models**. {\scriptsize cite Wattenberg et al., distill.pub "How to Use
    t-SNE Effectively" (2016).}

### Section 4 - choosing a method  [transition]
19. **PCA vs t-SNE vs UMAP on digits.** Three 2D embeddings side by side, colored by digit. Figure:
    shared legend, small alpha markers, subsample ~700 pts for legibility.
20. **Decision table.** linear? / preserves global? / deterministic? / scalable? / use when.

### Section 5 - connections & wrap  [transition]
21. **Kernel PCA (short).** Nonlinear PCA via the kernel trick.
22. **Autoencoders (forward-pointer).** Nonlinear DR with neural nets - neural-nets chapter.
23. **DR as preprocessing (callback).** Reduce -> then cluster / model (cluster-after-PCA); denoise before modeling.
24. **When NOT to reduce (honesty).** DR loses information; **PCA is unsupervised - it ranks directions by
    variance with no knowledge of `y`, so the discriminative direction can sit in a low-variance component
    PCA discards** (thin signal axis vs a high-variance nuisance axis); tree models often don't need it;
    don't reduce reflexively - try without first.
25. **Recap.** One-line takeaways: PCA = linear, fast, preprocessing+compression; t-SNE/UMAP = nonlinear,
    visualization, handle with care. `paramgreen` **Next:** box.

## Figures (`py_src/dimred_demos.py` -> `fig/`, `ma` venv)
- dr_pca_anim_1..5 (animated: cloud -> candidate axis -> max-variance axis -> projection)
- dr_curse_distances (distance concentration vs dim), dr_byhand_pca (2x2 covariance + sqrt(lambda) arrows),
  dr_scree, dr_biplot, dr_reconstruction (digits k=5/20/50 + denoise),
  dr_swiss_roll (2-panel: PCA collapse vs UMAP unroll),
  dr_tsne_digits, dr_umap_digits, dr_tsne_perplexity (3-panel: perplexity 5/30/100),
  dr_compare_digits (PCA | t-SNE | UMAP, shared legend, subsample ~700, alpha)
- (Eigen refresher = TikZ schematic.) Data: `sklearn.datasets.load_digits` (1797 x 64) + toy 2D/3D.
- Colors: clusters/labels via tab10 (>3 classes); single-series plots via flag blue/red (slide-style rule).

## Sources / verify at build
- PCA: variance-max ≡ min reconstruction error; eigen of covariance; SVD of centered X; sklearn centers
  but does not scale; explained_variance_ratio_ = lambda_i/sum. (scikit-learn PCA docs; Bishop PRML Ch.12.)
- t-SNE: symmetric p_ij, Student-t q over all pairs, KL objective, perplexity (van der Maaten & Hinton 2008;
  distill.pub "How to Use t-SNE Effectively"). UMAP: fuzzy graph + cross-entropy (McInnes et al. 2018).
- Curse of dimensionality: distance concentration, max/min -> 1 (Beyer et al. 1999).
- sklearn: `PCA`, `TruncatedSVD`, `KernelPCA`, `TSNE`; `umap-learn` for UMAP.
- Do NOT reference the i2dl manifold-learning slides - content is original.
