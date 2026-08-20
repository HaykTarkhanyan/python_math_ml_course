# L13c — UMAP, in depth (outline v1)

Second lecture of ch4b. `L13b` answers **what** DR does and **when** to reach for each method;
`L13c` answers **how UMAP actually works**. Ported from the LMU student-assistant deck
(`_reference_umap_lmu/umap_revised_iter_2.tex`, 17 slides) into course style.

## Port decisions

**1. Overlap with L13b — compress, do not repeat.** The LMU deck opens with four frames that
L13b already covers in more depth: motivation/curse (its slide 2), PCA recap (4), t-SNE recap (5),
and a PCA/t-SNE/UMAP comparison table (14). Delivered back to back, that is ten minutes of
re-teaching. **Collapsed into one "where we left off" bridge frame**, keeping only the piece
L13b does *not* make: *why* KL divergence leaves t-SNE blind to global structure, which is the
setup for UMAP's cross-entropy.

**2. `L13b`'s UMAP frame stays.** It answers "what does it give me", which a student needs even
if they skip this lecture. Add one forward-pointer line to it.

**3. TikZ mock-ups become real figures.** The LMU deck hand-draws the hyperparameter grid
(`k=5/50`, `min_dist=0.0/0.8`) as scattered TikZ dots. Per `ml/SLIDE_STYLE.md` essential figures
must be Python-generated, and here it is also just *better*: run UMAP at those actual settings on
Fashion-MNIST and show what really happens. Same for the exponential-decay curve, which becomes a
plot of the real `p_{j|i}` for the toy points. Small connective diagrams (the directed-edge pair,
springs-and-magnets) stay TikZ - they are throwaway visuals.

**4. Toy example runs at k=3, not k=2.** The LMU version states `sigma_A ~ 0.4`, which is not what
UMAP's binary search returns: at `k=2` the nearest neighbour alone already contributes exactly
`log2(2) = 1.0`, so the search drives `sigma -> 0` (verified: `sigma=0.01` gives 1.0035).
At `k=3` the target `log2(3) = 1.585` gives a genuine `sigma_A ~ 0.106`. All toy numbers get
recomputed by the figure script rather than typed in, so the slide cannot drift from the math.

---

## Frames

### Cold open
1. **Hook** — the Google PAIR elephant: same data, four parameter settings, four different
   pictures. "You have all used this plot. Today: what is it actually doing, and which parts of
   it are you allowed to believe?"
2. **Where we left off** — one bridge frame. UMAP in one line from L13b, plus the one new
   claim: t-SNE's `KL(P||Q)` punishes tearing neighbours apart but *not* collapsing distant
   points together, so global structure is unprotected. UMAP's loss is symmetric in that respect.
3. **Outline**

### Section 1 — The manifold assumption
4. *Transition:* "What UMAP assumes about your data"
5. **What is a manifold** — locally flat, globally curved; Earth's surface; face images along
   pose/lighting. The assumption stated plainly: data is sampled uniformly from a manifold
   *with respect to a locally varying metric*.
6. **Why "locally varying" is the whole trick** — predict-first. If the sampling looks uniform
   from every point's own perspective, then a point in a sparse region must have its distances
   *shrunk* and one in a dense region *stretched*. This is what `rho_i` and `sigma_i` do, and it
   is why UMAP handles varying density where raw kNN does not.

### Section 2 — Building the high-dimensional graph
7. *Transition:* "Step 1 of 2: turn the data into a fuzzy graph"
8. **The running toy** — 6 points, two clusters, k=3. Real coordinates, drawn by matplotlib.
9. **`rho_i`: local connectivity** — distance to the nearest neighbour; guarantees every point
   is connected to something with weight 1. Table of real values.
10. **`sigma_i`: the local scale** — the binary search, target `log2(k)`. Worked for point A:
    `sigma_A ~ 0.106`. Plot of the decay curve with the toy points marked on it.
11. **By-hand frame** — compute `p_{C|A}`, `p_{B|A}`, `p_{D|A}` from the formula, real numbers.
12. **Symmetrisation** — `p_ij = p_{j|i} + p_{i|j} - p_{j|i} p_{i|j}` as a fuzzy OR, with the
    "if *either* thinks the other is a neighbour" reading. TikZ pair diagram.
13. **The finished graph** — the toy's weighted graph, edge thickness = weight.

### Section 3 — Finding the low-dimensional layout
14. *Transition:* "Step 2 of 2: lay it out in 2-D"
15. **The low-dim kernel** — `q_ij = 1/(1 + a d^{2b})`, with `a, b` fitted from `min_dist`.
    Show the fitted curve for two `min_dist` values.
16. **Cross-entropy, split in two** — attraction `p log q` and repulsion `(1-p) log(1-q)`,
    and the contrast with t-SNE's KL that has no repulsion term. This is the frame that pays off
    frame 2.
17. **Springs and magnets** — the physical reading; SGD with negative sampling as the reason
    it is not `O(n^2)`. TikZ.
18. **Watching it converge** — real UMAP embeddings of Fashion-MNIST at increasing epoch counts,
    so "the system settles" is something they see.

### Section 4 — Using it
19. *Transition:* "The two knobs, and how to read the output"
20. **`n_neighbors`** — real UMAP runs at 5 / 15 / 50 on the same data. Local detail vs global shape.
21. **`min_dist`** — real runs at 0.0 / 0.1 / 0.8. Packing only; it does not change the topology.
22. **Predict-first: does the picture change?** — same data, same settings, three seeds.
23. **Trust / do not trust** — the two-column frame. Trust: neighbourhoods, cluster membership,
    connected components. Do not trust: axes, cluster size, between-cluster distance, tiny
    clusters at low `n_neighbors`.
24. **The canonical snippet** — one `umap.UMAP(...)` call, metric choice (cosine for embeddings),
    standardize first. One frame, minimal.
25. **Common mistakes** — wrong metric, unstandardized features, trusting a single run,
    feeding the 2-D embedding to a downstream model.

### Wrap-up
26. **Recap** + paramgreen "Next:" box.

---

## Figures (all `py_src/umap_demos.py` -> `fig/`)

| File | What |
|---|---|
| `umap_toy_points.pdf` | the 6-point toy, two clusters, labelled |
| `umap_rho_sigma.pdf` | decay curve `p_{j|i}` vs `d - rho`, toy points marked, real `sigma_A` |
| `umap_toy_graph.pdf` | the symmetrised fuzzy graph, edge width = weight |
| `umap_lowdim_kernel.pdf` | `q` vs distance for two `min_dist` values, with fitted `a, b` |
| `umap_attract_repel.pdf` | the two loss terms as functions of `q`, showing where each bites |
| `umap_epochs.pdf` | Fashion-MNIST embedding at increasing epochs |
| `umap_n_neighbors.pdf` | real runs at `n_neighbors` = 5 / 15 / 50 |
| `umap_min_dist.pdf` | real runs at `min_dist` = 0.0 / 0.1 / 0.8 |
| `umap_seeds.pdf` | three seeds, identical settings |

Borrowed, with attribution: `elephant.png` (Google PAIR) on the hook and the trust frame.
