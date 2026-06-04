# Linear Algebra — Math Reference

## Homework / Quarto modules (`math/`)

- [01_linear_algebra_vectors.qmd](../../math/01_linear_algebra_vectors.qmd) — **Vectors** — Vector basics, vector spaces, dot product, norms (L1/L2/Linf), Cauchy-Schwarz, angles, projections.
- [02_linear_algebra_matrices.qmd](../../math/02_linear_algebra_matrices.qmd) — **Matrices** — Matrix operations, transpose, identity, inverse, determinant, rank, trace, special matrix types (symmetric, orthogonal, diagonal, triangular).
- [03_linear_algebra_concepts.qmd](../../math/03_linear_algebra_concepts.qmd) — **Linear independence, basis, SLE, eigenstuff** — Linear independence, basis, dimension, change of basis, systems of linear equations (Gauss elimination), eigenvalues / eigenvectors / eigendecomposition, SVD, condition number.

## Lecture PDFs (`math/Lectures/`)

- [L01_Vectors.pdf](../../math/Lectures/L01_Vectors.pdf) — Vectors (delivery deck, no tex source).
- [L02_Angles__Vector_Spaces__Matrices.pdf](../../math/Lectures/L02_Angles__Vector_Spaces__Matrices.pdf) — Angles, vector spaces, intro to matrices.
- [L03_Inverse__Determinant.pdf](../../math/Lectures/L03_Inverse__Determinant.pdf) — Inverse + determinant.
- [L04_Linear_Independence__Basis__Dimension.pdf](../../math/Lectures/L04_Linear_Independence__Basis__Dimension.pdf) — Linear independence, basis, dimension.
- [L05_Eigenvalues__Eigendecomposition__SVD.pdf](../../math/Lectures/L05_Eigenvalues__Eigendecomposition__SVD.pdf) — Eigenvalues, eigendecomposition, SVD.

## Homework write-ups (`math/Homeworks/`)

- `hw_01_vectors.pdf` (with `.xopp` source)
- `hw_02_matrices_det_inverse.pdf`

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| L01 design matrix `X` | module 02 |
| L01 dot product `x^T theta` | module 01 |
| L03 L1 / L2 norms (Ridge / Lasso penalty) | module 01 |
| OLS `(X^T X)^{-1} X^T y` | module 02 (inverse), 03 (rank) |
| PCA | module 03 (eigen, SVD), L05 deck |
| Singular matrices / multicollinearity warnings | module 03 (rank) |
| Attention as `Q K^T` matrix products | module 02 |
| Embeddings, cosine similarity | module 01 (dot product, norms, angles) |

## Notes

- Standard notation choices throughout: column vectors, matrix-on-the-left convention (`y = X theta + epsilon`).
- The qmd modules are bilingual (Armenian section headers like `📚 Նյութը`, `🏡 Տնային`). The lecture PDFs are mostly English-text + Armenian-spoken delivery.
- For SVD specifically, the ML checklist still has it tagged as "missing" — see `concepts_checklist.csv`. When the ML course needs SVD, point to `module 03` and `L05_Eigenvalues...pdf` rather than re-deriving.
