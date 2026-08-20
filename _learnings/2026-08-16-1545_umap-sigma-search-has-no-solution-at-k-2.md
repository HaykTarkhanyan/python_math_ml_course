# UMAP's sigma binary search has no solution at k=2, or with tied nearest neighbours

**Symptom.** Building the toy example for the UMAP lecture (35_umap), the per-point `sigma_i`
search either ran to its iteration limit or converged to an absurd value, depending on the toy's
geometry. The LMU source deck this was ported from states `sigma_A ~ 0.4` for a `k=2` toy, which
looked plausible and is wrong.

**Cause.** UMAP sets `sigma_i` so the neighbourhood carries a fixed fuzzy mass:

```
sum_{j in kNN(i)} exp( -max(d(x_i,x_j) - rho_i, 0) / sigma_i )  =  log2(k)
```

`rho_i` is the distance to the nearest neighbour, so that neighbour's shifted distance is exactly
`0` and its term is exactly `exp(0) = 1`, **whatever sigma is**. So the sum is bounded below by 1
(or by the number of neighbours tied at `rho_i`).

Two ways that collides with the target:

- **k=2.** Target is `log2(2) = 1.0`. The nearest neighbour alone already contributes 1.0, so the
  second neighbour's term must vanish, driving `sigma -> 0`. Measured on the toy: `sigma=0.4`
  gives a sum of 1.868, `sigma=0.1` gives 1.568, `sigma=0.01` gives 1.0035. It never reaches 1.0.
- **Tied nearest neighbours.** A symmetric toy (an isoceles triangle apex equidistant from both
  base points) puts *two* neighbours at exactly `rho_i`, so the sum is pinned at `>= 2` against a
  target of `log2(3) = 1.585`. No solution at any sigma.

**Consequences.**

- The toy runs at **k=3** with **off-centre apexes** (0.45 / 5.45 rather than 0.50 / 5.50). Target
  1.585, and point A gets a genuine `rho_A = 0.9179`, `sigma_A = 0.1532`.
- **`k=2` is not a legal illustration of UMAP's mechanism.** Any worked example needs `k >= 3` and
  distinct neighbour distances. Real datasets rarely hit the tie case; hand-built teaching toys
  hit it constantly, because we draw them symmetric on purpose.
- This was caught only because `smooth_knn` in `py_src/umap_demos.py` **raises** on
  non-convergence. A silent fallback returning the last sigma would have printed a
  plausible-looking number straight onto a lecture slide. Worth remembering next time a
  "just return something sensible" fallback looks harmless.

**Verify with:**

```python
# toy point A, k=2: sum can never reach the target of 1.0
import numpy as np
rho, d_second = 0.9434, 1.0
for s in (0.4, 0.1, 0.01):
    print(s, 1 + np.exp(-(d_second - rho) / s))   # 1.868, 1.568, 1.0035
```

Related: [[2026-08-16-1610_a-clean-embedding-makes-a-terrible-image-atlas]]
