# Calculus — Math Reference

## Homework / Quarto modules (`math/`)

- [04_calc_lim_continuity_deriv.qmd](../../math/04_calc_lim_continuity_deriv.qmd) — **Limits, continuity, derivatives** — Limits (intuitive + epsilon-delta), continuity, differentiation rules, chain rule, related rates.
- [05_calc_extrema_convexity_taylor.qmd](../../math/05_calc_extrema_convexity_taylor.qmd) — **Extrema, convexity, Taylor** — Critical points, first and second derivative tests, convex / concave functions, Jensen's inequality (1D), Taylor series and remainders.
- [06_calc_integrals.qmd](../../math/06_calc_integrals.qmd) — **Integrals** — Antiderivatives, definite integrals, FTC, substitution, integration by parts, improper integrals.
- [07_calc_multivar.qmd](../../math/07_calc_multivar.qmd) — **Multivariate calculus** — Partial derivatives, gradient, directional derivatives, Hessian, contour plots, multivariate convexity, multivariate Taylor, multivariate extrema (incl. local extrema problem set tied to the Armenian notes Section 7.7). Includes Boat in Sevan, Topless Box, TV Tower Placement problems.

## Lecture PDFs (`math/Lectures/`)

- [L06_Limit__Derivative__Extrema_of_a_Function.pdf](../../math/Lectures/L06_Limit__Derivative__Extrema_of_a_Function.pdf) — Limit, derivative, extrema (univariate).
- [L07_Taylor_Series__Integral.pdf](../../math/Lectures/L07_Taylor_Series__Integral.pdf) — Taylor series + integrals.
- [L08_Functions_of_Several_Variables.pdf](../../math/Lectures/L08_Functions_of_Several_Variables.pdf) — Multivariate functions.

## Homework write-ups (`math/Homeworks/`)

- `hw_03_calc_1.pdf` — univariate derivatives
- `hw_04_calc_2_extrema_convexity_taylor.pdf`
- `hw_05_integrals.pdf`
- `hw_06_multivar_calc.pdf`

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| L01 derivative of loss `dL/d theta` | module 04 |
| L02 gradient descent step `theta - alpha * grad R_emp` | module 07 (gradient) |
| L02 learning rate intuition | module 05 (convexity) + 07 (Hessian curvature) |
| L03 convex regularizers / convex optimization | module 05 |
| Jensen's inequality (ELBO, info theory bounds) | module 05 |
| Backprop (chain rule across layers) | module 04 (chain rule) + 07 (multivariate chain) |
| Taylor expansion for Newton's method / quadratic approximations | modules 05, 07 |
| L2-loss MSE derivative via expectation | module 06 (integrals → expectation in stats L6) |
| Delta method | module 07 (Taylor) + stat L8 |

## Notes

- Module 07 is the heaviest of the four — it's where students first see gradients and contour plots together. Almost every ML lecture from L02 onward implicitly depends on it.
- "Boat in Sevan" and "Topless Box" are the Armenian local examples for optimization problems (matches `feedback_armenian_examples.md`). Reuse them in ML decks when possible.
- The Lectures PDFs are single artefacts (no `.tex` source) — older delivery decks. For new ML content, point students at the qmd if they need to revise.
