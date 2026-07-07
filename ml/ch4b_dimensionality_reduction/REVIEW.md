# Dimensionality-reduction deck (ml/ch4b) - content & teaching review

Reviewed 2026-07-07 (Claude). Scope: `L13b_dimensionality_reduction.tex` (29 slides / 34
rendered pages), `dim_reduction_OUTLINE.md`, all figures in `fig/`, the generator
`py_src/dimred_demos.py`, and a glance at `solution_eigenfaces.ipynb`. Method: close read of
the `.tex`, a visual pass over every rendered page, up-close inspection of the dense figures,
and a numeric re-check of the PCA explained-variance claims against sklearn digits.

**Verdict up front:** this is a strong, accurate, well-sequenced deck - one of the cleaner ones
in the course. The motivation arc (curse of dimensionality -> intuition -> mechanism ->
practice -> nonlinear -> choose) is textbook, the PCA math is correct throughout, the animated
variance-sweep with its "PCA does not iterate" caption pre-empts exactly the right misconception,
and most figures (reconstruction, perplexity-sensitivity, scree, by-hand PCA, curse) are clean and
earn their space. There is exactly **one hard content error** (frame 9's explained-variance
percentages contradict the deck's own scree figure), **two rendering defects** (the animation
footnote collides with the page number on 5 pages; the biplot's arrow labels overlap), and a
short list of **high-value, low-cost additions** (state that PCA scores are uncorrelated; a
show-don't-tell centering-vs-scaling figure; two predict-first opportunities; a
feature-selection-vs-extraction one-liner; a "choose k by CV" caveat). Nothing is over-scoped.

---

## 1. Bugs and math errors (fix first)

1. **Frame 9 "Derivation (2)": the explained-variance numbers are wrong and contradict the
   deck's own figure.** The slide says "first PC of the digits `~=` 12%; first two `~=` 22%".
   The scree figure (`fig_scree`, frame 14) is computed on the **unscaled** digits (matching its
   "29 of 64 for 95%" claim). On that same unscaled PCA the real numbers are **first PC = 14.9%,
   first two = 28.5%** (re-verified with sklearn). The scree plot two frames later visibly shows
   the first bar at ~0.148 and the cumulative curve at ~0.28 at k=2 - so 12% / 22% directly
   contradicts the picture the student is looking at. **Fix: "`~=` 15%; first two `~=` 28%".**
   (This is the only numeric error in the deck; everything else recomputes correctly, including the
   by-hand toy: cov `[[3.51,1.13],[1.13,0.83]]` -> eigenvalues 3.92, 0.42, checked.)

## 2. Rendering / overflow defects

2. **PCA-sweep animation (frame "PCA: keep the directions of greatest spread", all 5 overlay
   pages) overflows into the page number.** The bottom footnote's last line
   "...for 64-D digits we keep 2)." runs off the bottom and **collides with the "6/29" page
   number** on every overlay (confirmed on the rendered pages). The frame is over-full: title +
   3-line intro + a `0.56\textwidth` figure + a 3-line footnote. Fix any one of: shrink the figure
   to `0.5\textwidth`, move the "keep the top-k / 2-D->1-D" clause up into the intro paragraph, or
   cut the footnote to two lines. Cheap and it's the deck's most-viewed frame (5 pages).

3. **Biplot arrow labels overlap (frame 15 / `dr_biplot.pdf`).** On the right side, "sepal
   length", "petal width", and "petal length" sit almost on top of each other near the PC1 axis
   and are hard to read on a projector. Nudge the three label offsets apart (stagger the `* 1.13`
   text radius per-arrow) or shorten to "petal L / petal W / sepal L". The samples-and-loadings
   content itself is correct and the interpretation text matches the picture.

## 3. Per-section content review

**Hook / why / curse (frames 1-3).** Strong. The "clustering groups the rows, DR compresses the
columns" framing is a clean bridge, and the curse frame is correctly a predict-first (`\pause`).
One subtlety: `fig_curse_distances` plots `(max - min)/mean` over **all pairwise distances**
(`pdist`), which is a reasonable proxy but is *not* the Beyer et al. "relative contrast"
`(Dmax - Dmin)/Dmin` measured from a query point - and the slide's question is phrased as
"how much closer is your *nearest* point than your *farthest*", i.e. exactly the query-based
quantity. It still shows concentration and the point lands, so this is a minor
message-vs-figure mismatch, not an error. If you ever regenerate: use nearest/farthest from a set
of query points and divide by the nearest (or by the mean nearest) to match the wording.

**PCA idea & math (frames "intuition" through "by hand").** Excellent build and all correct:
centering as Step 0, `w^T Sigma w` maximized under `||w||=1`, Lagrange -> `Sigma w = lambda w`,
eigenvalues = variance captured, descending order -> top-k, `lambda_i / sum lambda_j`, the
max-variance == min-reconstruction-error bridge (green box), SVD of centered X with columns of V =
PCs, TruncatedSVD/LSA skipping centering for sparse data. Two small notes:
- The animated sweep reaches the maximum (variance 5.54) already on candidate frame 4, and frame 5
  relabels *the same axis* as "max variance - keep this axis". Mild redundancy; a student may
  wonder why the "candidate" and the "answer" are identical. Optional: stop the sweep one step
  short of the PC so frame 5 is a genuine reveal.
- The Lagrange step is compressed to one line ("a Lagrange multiplier gives ..."). The style guide
  asks for full step-by-step derivations; one intermediate line
  (`grad(w^T Sigma w - lambda(w^T w - 1)) = 2 Sigma w - 2 lambda w = 0`) would complete it for a
  university audience. Low priority.

**Missing fundamental fact (belongs on frame 7 or the pitfalls frame).** The deck never states
that **the PCA scores are mutually uncorrelated** - `Cov(Z) = W^T Sigma W = diag(lambda)`. That is
the precise sense in which PCA "decorrelates" the data, it is a *defining* property of PCA, and it
ties straight back to the curse/collinearity motivation ("why run PCA before a model that struggles
with correlated features"). One line earns its keep and pre-empts the loose folklore claim that
"PCA removes correlated features" (which the deck commendably avoids stating). Recommend adding.

**PCA in practice (frames 13-17).** Correct and well-ordered.
- *Centering vs scaling* is text-only (armred trap box). This is the deck's best show-don't-tell
  opportunity: a 2-panel toy (e.g. income-in-10,000s + age) where unscaled PCA gives PC1 `~=` the
  income axis, and standardized PCA gives a balanced combination. The trap is much stickier as a
  picture than as a sentence. See illustrations below.
- *How many components* - the "29 of 64 for 95%" is a natural **predict-first**: ask "for 64-D
  digits, how many PCs to reach 95%?" Students reliably guess 2-3; the answer being ~half the
  features is the counter-intuitive, memorable moment the frame currently states flat.
- *Reconstruction* is good; consider a one-line callback tying it to the scree frame ("the
  blurriness at k=5 is exactly the variance the scree plot said we threw away" - i.e.
  reconstruction error at k = `1 - cumulative EVR = sum of dropped eigenvalues"). It closes the
  loop between frames 14 and 16 for free.
- *Pitfalls* frame is correct and appropriately terse.

**Nonlinear DR (frames 19-22).** t-SNE and UMAP formulas are all stated correctly (symmetrized
`p_ij`, Student-t `q_ij` and the crowding fix, KL(P||Q), perplexity 5-50; fuzzy NN graph,
cross-entropy, `n_neighbors`/`min_dist`, faster/more-global). The perplexity-sensitivity figure is
excellent - clearly readable, the three panels genuinely differ, and it's the right analogue of the
clustering failure-mode figures. Two notes:
- The **swiss-roll figure works but under-teaches**. The PCA "roll stays folded" panel renders as a
  fairly clean rainbow spiral, so a student who has never seen a swiss roll cannot tell what PCA is
  *failing* to do - the colors still look ordered. Add the **original 3-D roll** as a small left
  panel (roll colored by position) so the sequence reads 3-D roll -> PCA flattens/overlaps ->
  UMAP unrolls to a strip. That makes "unfold a curved manifold" legible.
- The **caveats frame is the strongest predict-first candidate in the deck**: show the three
  perplexity panels one at a time behind a "same data, same algorithm - will the picture be the
  same?" prompt. The whole lesson is that students wrongly expect t-SNE to give *the* embedding.

**Choosing a method (frames 24-25).** Compare-on-digits figure and the decision table are both
correct (PCA yes/yes/yes, t-SNE no/no/no, UMAP no/some/no-seeded) and the "reach for PCA first"
rule of thumb is right. No changes.

**Connections & when-not (frames 27-29).** Excellent and honest. The "PCA is unsupervised, so the
discriminative direction can hide in a low-variance component it discards (thin signal axis vs. fat
nuisance axis)" is the single best "when not to reduce" lesson and it is exactly right. Kernel-PCA
and autoencoder one-liners are correctly scoped and correctly stated.

## 4. Missing topics

**Add (small, high value):**
- **PCA scores are uncorrelated** (decorrelation property) - see section 3. Defining property,
  currently absent.
- **Feature selection vs feature extraction.** The pitfalls frame says "not feature selection" but
  never names the *other* family. One clause - "DR is feature *extraction* (build new features); a
  separate family, feature *selection*, keeps a subset of the originals (drop low-variance /
  correlated columns, L1, etc.)" - places PCA in the taxonomy and answers a question students ask.
- **Choosing k by cross-validation when PCA feeds a model.** The scree frame teaches only the "95%"
  rule of thumb. When PCA is preprocessing for a classifier, k should be tuned by CV on the
  *downstream* metric, not by a variance threshold. One line on the "how many components" frame.

**Fine to skip (agree with the cuts):** whitening, incremental/randomized/sparse PCA, big-data
variants, kernel-PCA math, autoencoder details (all one-liners or out of scope). One to
*reconsider*: the outline deliberately cut eigen-by-hand, but a 2x2 characteristic-polynomial
worked example (`det(Sigma - lambda I) = 0 -> lambda^2 - tr*lambda + det = 0`) is the single most
"computable mechanic" in the whole topic and the audience has the linear algebra for it. The
by-hand frame currently *displays* cov and lambda without deriving them; a real derivation would be
the strongest worked-numbers frame in the deck. Judgment call, but worth a second look.

## 5. Illustrations to add / fix

1. **Centering-vs-scaling 2-panel** (new figure) - unscaled PCA where PC1 `~=` the big-range
   feature, vs standardized PCA where PC1 is balanced. Turns the text-only trap into the picture
   students remember. Highest-value new figure.
2. **Swiss-roll: add the 3-D roll panel** so "unroll" is legible (see nonlinear section).
3. **Fix the biplot label overlap** (bug 3) and **the animation footnote collision** (bug 2).

## 6. What is working - do not touch

- The motivation arc and section ordering (curse -> intuition -> mechanism -> practice ->
  nonlinear -> choose -> when-not). Textbook-grade.
- The **animated variance sweep with the "PCA does not search or iterate - the eigenvectors give
  these axes directly" caption** - this pre-empts students importing the k-means "iterate to
  converge" mental model onto PCA, which is exactly the right misconception to kill here.
- The **max-variance == min-reconstruction-error bridge** stated early (frame 9, green box) and paid
  off on the reconstruction frame - clean forward-reference design.
- The **by-hand PCA figure** (real 2x2 covariance numbers next to `sqrt(lambda)`-scaled eigenvector
  arrows) - correct and readable.
- The **reconstruction figure** (compression k=5/20/50 + a denoise row) and the **perplexity
  -sensitivity figure** - both excellent and readable at projector scale.
- The **decision table** and the **honest "when NOT to reduce" frame** (low-variance signal axis).
- The **on-slide citations** (Beyer 1999; van der Maaten & Hinton 2008; McInnes 2018; Wattenberg
  distill.pub 2016) - keep this pattern.
- The **eigenfaces project notebook** reinforces reconstruction, scree, and clustering-after-PCA on
  a *different* dataset (Olivetti faces) than the deck's digits - good transfer, and the "a face in
  k numbers" framing mirrors the deck's compression story. Nicely aligned.

## 7. Priority order

| # | Item | Effort | Where |
|---|------|--------|-------|
| 1 | Fix frame-9 numbers 12%/22% -> ~15%/~28% (contradicts own scree figure) | minutes | frame 9 |
| 2 | Fix animation footnote colliding with page number (5 pages) | minutes | PCA-sweep frame |
| 3 | Fix biplot arrow-label overlap | small | `dr_biplot.pdf` |
| 4 | Add "PCA scores are uncorrelated" line | minutes | frame 7 / pitfalls |
| 5 | Convert scree "29 of 64" and t-SNE caveats to predict-first | small | frames 14, 22 |
| 6 | Centering-vs-scaling show-don't-tell 2-panel figure | medium | frame 13 |
| 7 | Swiss-roll: add 3-D roll panel | small-medium | `dr_swiss_roll.pdf` |
| 8 | Feature-selection-vs-extraction + choose-k-by-CV one-liners | small | pitfalls / scree |
| 9 | Reconstruction<->scree callback; Lagrange gradient line | small | frames 16, "Deriv (1)" |
| 10 | (Reconsider) 2x2 eigen-by-hand worked example | medium | by-hand frame |
