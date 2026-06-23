# Dimensionality Reduction — Deck Outline (FINAL v4, post-review 2026-06-23)

House style per `ml_new/SLIDE_STYLE.md`. `slide-style` workflow: interview (4 rounds) +
independent review -> this outline -> build after approval.

## Review fixes folded in
- **Centering made explicit and separated from scaling.** Frame 6 adds "Step 0: center" with reason;
  frame 10 reframed as "PCA always centers (sklearn auto); it does NOT standardize - your call".
- **"High variance != signal"** (frame 24) reframed via the real mechanism: PCA is **unsupervised**,
  ranks by variance ignoring `y`, so the discriminative direction can sit in a low-variance component.
- **t-SNE formula precision** (frame 16): symmetric `p_ij=(p_{j|i}+p_{i|j})/2n`, Student-t `q_ij`
  normalized over ALL pairs, heavy tail resolves the crowding problem; perplexity ~ 5-50.
- **Swiss-roll figure now 2-panel** (PCA collapses it vs UMAP unrolls it) - a single panel didn't show recoverability.
- **By-hand PCA figure** must show the actual 2x2 covariance numbers + both eigenvector arrows scaled by `sqrt(lambda)` (else it duplicates the intuition figure).
- **Reconstruction** moved entirely to frame 13 (was split into frame 7).
- **Predict-first beat added to the curse-of-dimensionality frame** (house-style ask).
- **compare-digits figure**: shared legend, small/alpha markers, subsample ~700 pts for legibility.
- Eigen refresher states symmetric -> orthogonal eigenvectors (the link that justifies PC orthogonality).
- TruncatedSVD note: doesn't center -> preserves sparsity. perplexity/n_neighbors typical ranges on 16/17.

## Confirmed scope (from interview)
PCA workhorse (full derivation + SVD + eigen refresher); t-SNE & UMAP viz-only WITH key formulas;
dedicated curse-of-dim frame; dedicated viz-caveats frame; dedicated "when NOT to reduce" frame;
kernel-PCA + autoencoder = one-liners; by-hand PCA = covariance->axis (no eigen-by-hand); toy + digits;
skip big-data PCA variants + PCA-preinit tip; no app closer. UMAP via `umap-learn` (approved).
Do NOT reference i2dl manifold slides. Deck file: supersede the `L13b_pca_dim_reduction` stub here.

## Frame-by-frame (~25 content + outline + 5 transitions ~= 32 pages)

### Hook + why
1. **Cold-open.** "How do you *look* at 64-dimensional data?" digit=8x8=64; gene expr ~20k; embeddings ~768.
   Bridge from clustering: unsupervised - clustering groups *rows*, DR compresses *columns*.
2. **Why reduce dimensions.** Visualization, preprocessing/compression, denoising (drop low-variance noise, *often*).
3. **Curse of dimensionality.** *Predict-first* (`\pause`): "in 100-D, how much closer is the nearest point
   than the farthest?" -> almost the same; distances **concentrate** (max/min ratio -> 1), distance-based
   methods degrade. Figure: distance distribution vs dimension. Callback to clustering.

[Outline frame]

### Section 1 - PCA: the idea & derivation  [transition]
4. **PCA intuition.** New orthogonal axes capturing the most variance; keep the top few. Figure: 2D cloud + axes.
5. **Eigen refresher.** `A v = lambda v` (direction unchanged, length scaled by `lambda`); a **symmetric**
   matrix (like a covariance) has real eigenvalues and **orthogonal eigenvectors** - that is *why* PCs are
   orthogonal. No computation. TikZ schematic.
6. **Derivation (1): max variance.** **Step 0 - center the data** (subtract feature means; without it
   `X^T X` measures spread about the origin, not the data). Variance along unit `w` = `w^T Σ w` (`Σ` =
   covariance of centered data); maximize s.t. `||w||=1` -> Lagrange -> **`Σ w = lambda w`**: PCs =
   eigenvectors of covariance, `lambda` = variance captured.
7. **Derivation (2): eigenvalues -> explained variance.** Sort eigenvalues; top-k eigenvectors = PCs;
   **explained-variance ratio** `lambda_i / sum lambda`.
8. **The SVD connection.** PCA = **SVD of centered data** (`X = U S V^T`, cols of `V` = PCs); numerically
   better. **TruncatedSVD / LSA** for sparse text - it does NOT center, so it preserves sparsity.
9. **By-hand 2D PCA.** Tiny dataset -> the actual **2x2 covariance matrix (numbers on the slide)** -> draw
   **both eigenvector arrows, each scaled by `sqrt(lambda)`** on the scatter. Figure.

### Section 2 - PCA in practice  [transition]
10. **Centering vs scaling.** PCA **always centers** (sklearn does it for you); it does **not** standardize
    - that is your call. `armred` trap: variance is scale-dependent, so standardize unless features share units.
11. **Choosing #components.** Scree plot + cumulative explained variance (keep ~95%). Figure.
12. **Interpreting components.** Loadings + **biplot**; what a component "means". Figure.
13. **Reconstruction, compression & denoising.** Project to top-k, then lift back. Figure: digits at
    k=5/20/50 (compression) + a denoise panel. (Reconstruction introduced here.)
14. **Pitfalls.** Not feature selection (PCs are *combinations*); **linear only**; outlier/scale sensitive;
    components can be uninterpretable.

### Section 3 - nonlinear DR for visualization: t-SNE & UMAP  [transition]
15. **Why nonlinear.** PCA is linear - can't unfold a curved manifold. **2-panel figure:** PCA flattens the
    swiss roll (color gradient scrambled / overlapping) vs **UMAP unrolls it** (gradient preserved). Goal:
    preserve **local neighborhoods**.
16. **t-SNE.** High-D similarities **`p_ij = (p_{j|i}+p_{i|j})/2n`**; low-D **`q_ij`** with a **Student-t**
    kernel (heavy tail fixes the "crowding problem"), normalized over all pairs; minimize **KL(P||Q)**;
    **perplexity** ~ effective #neighbors (typ. 5-50). Figure: digits t-SNE.
17. **UMAP.** **Fuzzy nearest-neighbor graph** laid out by minimizing a graph **cross-entropy**;
    **`n_neighbors`** (local<->global), **`min_dist`** (cluster tightness). Faster, more global. Figure: digits UMAP.
18. **Caveats (dedicated).** Can mislead: **cluster sizes & between-cluster distances not meaningful**;
    **stochastic**; perplexity/`n_neighbors` sensitivity; **don't feed embeddings to downstream models**.

### Section 4 - choosing a method  [transition]
19. **PCA vs t-SNE vs UMAP on digits.** Three 2D embeddings side by side, colored by digit. Figure:
    shared legend, small alpha markers, subsample ~700 pts for legibility.
20. **Decision table.** linear? / preserves global? / deterministic? / scalable? / use when.

### Section 5 - connections & wrap  [transition]
21. **Kernel PCA (short).** Nonlinear PCA via the kernel trick.
22. **Autoencoders (forward-pointer).** Nonlinear DR with neural nets - neural-nets chapter.
23. **DR as preprocessing (callback).** Reduce -> then cluster / model (cluster-after-PCA); denoise before modeling.
24. **When NOT to reduce (honesty).** DR loses information; **PCA is unsupervised - it ranks directions by
    variance with no knowledge of `y`, so the discriminative direction for your label can sit in a
    low-variance component PCA discards** (thin signal axis vs a high-variance nuisance axis); tree models
    often don't need it; don't reduce reflexively - try without first.
25. **Recap.** One-line takeaways (not a re-table): PCA = linear, fast, preprocessing+compression;
    t-SNE/UMAP = nonlinear, visualization, handle with care. `paramgreen` **Next:** box.

## Figures (`py_src/dimred_demos.py` -> `fig/`, `ma` venv)
- dr_curse_distances (distance concentration vs dim), dr_pca_axes (2D cloud + axes),
  dr_byhand_pca (2x2 covariance numbers + both eigenvector arrows scaled by sqrt(lambda)),
  dr_scree, dr_biplot, dr_reconstruction (digits k=5/20/50 + denoise),
  dr_swiss_roll (2-panel: PCA collapse vs UMAP unroll),
  dr_tsne_digits, dr_umap_digits, dr_compare_digits (shared legend, subsample ~700, alpha)
- (Eigen refresher = TikZ schematic.) Data: `sklearn.datasets.load_digits` (1797 x 64) + toy 2D/3D.

## Sources / verify at build
- PCA: variance-max ≡ min reconstruction error; eigen of covariance; SVD of centered X; sklearn centers
  but does not scale; explained_variance_ratio_ = lambda_i/sum. (scikit-learn PCA docs; Bishop PRML Ch.12.)
- t-SNE: symmetric p_ij, Student-t q over all pairs, KL objective, perplexity (van der Maaten & Hinton 2008;
  distill.pub "How to Use t-SNE Effectively"). UMAP: fuzzy graph + cross-entropy (McInnes et al. 2018).
- Curse of dimensionality: distance concentration, max/min -> 1 (Beyer et al. 1999).
- sklearn: `PCA`, `TruncatedSVD`, `KernelPCA`, `TSNE`; `umap-learn` for UMAP.
- Do NOT reference the i2dl manifold-learning slides - content is original.
