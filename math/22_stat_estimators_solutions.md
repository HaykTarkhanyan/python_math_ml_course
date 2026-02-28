# 22: Properties of Estimators — Solutions (Problems 01–04)

---

## Recap: Key Concepts

### Exponential Family

A distribution belongs to the **one-parameter exponential family** if its PDF/PMF can be written as:

$$
f(x \mid \theta) = h(x) \exp\!\big(\eta(\theta)\, T(x) - A(\theta)\big)
$$

where:

| Symbol | Name | Depends on |
|--------|------|------------|
| $h(x)$ | Base measure | data only |
| $\eta(\theta)$ | Natural parameter | parameter only |
| $T(x)$ | Sufficient statistic | data only |
| $A(\theta)$ | Log-partition function | parameter only |

Most "named" distributions (Normal, Poisson, Bernoulli, Exponential, Gamma, Beta, …) belong to this family. The Uniform distribution is a notable **exception** (its support depends on the parameter).

### Sufficient Statistic

A statistic $T(\mathbf{X})$ is **sufficient** for $\theta$ if it captures all the information in the sample about $\theta$. Once you know $T$, the conditional distribution of the data given $T$ does not depend on $\theta$.

**Fisher–Neyman Factorization Theorem:** $T(\mathbf{X})$ is sufficient for $\theta$ iff the joint density factors as:

$$
f(\mathbf{x} \mid \theta) = g\!\big(T(\mathbf{x}),\, \theta\big) \cdot h(\mathbf{x})
$$

where $g$ depends on $\mathbf{x}$ only through $T(\mathbf{x})$, and $h$ does not depend on $\theta$.

**Exponential family shortcut:** For an i.i.d. sample from an exponential family, $\sum_{i=1}^n T(X_i)$ is always sufficient.

### Minimal Sufficiency

A sufficient statistic $T$ is **minimal sufficient** if it is a function of every other sufficient statistic — it compresses the data as much as possible without losing information about $\theta$.

**Likelihood ratio criterion:** $T(\mathbf{x})$ is minimal sufficient iff:

$$
T(\mathbf{x}) = T(\mathbf{y}) \quad \Longleftrightarrow \quad \frac{f(\mathbf{x} \mid \theta)}{f(\mathbf{y} \mid \theta)} \text{ is free of } \theta.
$$

### Bias and Unbiasedness

An estimator $\hat{\theta}$ is **unbiased** for $\theta$ if $\mathbb{E}[\hat{\theta}] = \theta$.

The **bias** is $\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$.

### Key Distributional Facts Used Below

| Distribution | Mean | Variance |
|---|---|---|
| $\text{Poisson}(\lambda)$ | $\lambda$ | $\lambda$ |
| $\text{Uniform}(0,d)$ | $d/2$ | $d^2/12$ |
| $\text{Bernoulli}(\pi)$ | $\pi$ | $\pi(1-\pi)$ |
| $\text{Binomial}(n,\pi)$ | $n\pi$ | $n\pi(1-\pi)$ |

---

## Problem 01 — Poisson Meets the Exponential Family

> $X \sim \text{Poisson}(\lambda)$, so $P(X = k) = \dfrac{\lambda^k e^{-\lambda}}{k!}$.

### Part (a): Write in exponential family form

Start with the PMF and take the logarithm to rearrange:

$$
f(x \mid \lambda) = \frac{\lambda^x e^{-\lambda}}{x!}
$$

Rewrite $\lambda^x = e^{x \ln \lambda}$:

$$
f(x \mid \lambda) = \frac{1}{x!} \cdot \exp\!\big(x \ln\lambda - \lambda\big)
$$

This is already in the exponential family form $h(x)\exp\!\big(\eta(\lambda)\,T(x) - A(\lambda)\big)$ with:

$$
\boxed{
\begin{aligned}
h(x) &= \frac{1}{x!} \\[4pt]
\eta(\lambda) &= \ln \lambda \\[4pt]
T(x) &= x \\[4pt]
A(\lambda) &= \lambda
\end{aligned}
}
$$

### Part (b): Sufficient statistic for an i.i.d. sample

For an i.i.d. sample $X_1, \dots, X_n$, the joint PMF is:

$$
f(\mathbf{x} \mid \lambda) = \prod_{i=1}^n \frac{1}{x_i!} \cdot \exp\!\left(\ln\lambda \sum_{i=1}^n x_i - n\lambda\right)
$$

By the exponential family result (or by direct application of the factorization theorem), the sufficient statistic is:

$$
\boxed{T(\mathbf{X}) = \sum_{i=1}^{n} X_i}
$$

Equivalently, $\bar{X} = T/n$ is also sufficient (since it is a one-to-one function of $T$).

---

## Problem 02 — Slit Width Estimation

> $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} \text{Uniform}(0, d)$, with unknown $d > 0$.

### Part (a): Joint density

Each $X_i$ has density $f(x_i \mid d) = \frac{1}{d}\,\mathbf{1}(0 \le x_i \le d)$.

The joint density of the sample is:

$$
f(\mathbf{x} \mid d) = \prod_{i=1}^n \frac{1}{d}\,\mathbf{1}(0 \le x_i \le d) = \frac{1}{d^n}\,\mathbf{1}(0 \le x_{(1)}) \cdot \mathbf{1}(x_{(n)} \le d)
$$

where $x_{(1)} = \min_i x_i$ and $x_{(n)} = \max_i x_i$. Since all $x_i \ge 0$ automatically when $x_{(1)} \ge 0$, we can write:

$$
\boxed{f(\mathbf{x} \mid d) = \frac{1}{d^n}\,\mathbf{1}\!\big(0 \le x_{(1)}\big)\,\mathbf{1}\!\big(x_{(n)} \le d\big)}
$$

### Part (b): Sufficiency of $X_{(n)}$ via factorization

We factor the joint density as:

$$
f(\mathbf{x} \mid d) = \underbrace{\frac{1}{d^n}\,\mathbf{1}(x_{(n)} \le d)}_{g(x_{(n)},\, d)} \;\cdot\; \underbrace{\mathbf{1}(x_{(1)} \ge 0)}_{h(\mathbf{x})}
$$

- $g(x_{(n)}, d) = d^{-n}\,\mathbf{1}(x_{(n)} \le d)$ depends on $\mathbf{x}$ only through $x_{(n)}$ and on the parameter $d$.
- $h(\mathbf{x}) = \mathbf{1}(x_{(1)} \ge 0)$ does not depend on $d$.

By the **Fisher–Neyman factorization theorem**, $X_{(n)} = \max\{X_1, \dots, X_n\}$ is sufficient for $d$. $\blacksquare$

### Part (c): Is $X_{(n)}$ unbiased?

Using the hint, the CDF of $X_{(n)}$ is:

$$
F_{X_{(n)}}(x) = \left(\frac{x}{d}\right)^n, \qquad 0 \le x \le d
$$

The PDF is:

$$
f_{X_{(n)}}(x) = \frac{n\, x^{n-1}}{d^n}, \qquad 0 \le x \le d
$$

Compute the expectation:

$$
\mathbb{E}[X_{(n)}] = \int_0^d x \cdot \frac{n\, x^{n-1}}{d^n}\, dx = \frac{n}{d^n} \int_0^d x^n\, dx = \frac{n}{d^n} \cdot \frac{d^{n+1}}{n+1} = \frac{n}{n+1}\,d
$$

Since $\mathbb{E}[X_{(n)}] = \frac{n}{n+1}\,d \neq d$, the statistic $X_{(n)}$ is **biased** — it systematically underestimates $d$.

**Unbiased correction:** Since $\mathbb{E}\!\left[\frac{n+1}{n}\,X_{(n)}\right] = d$, the estimator

$$
\boxed{\hat{d} = \frac{n+1}{n}\,X_{(n)}}
$$

is unbiased for $d$.

**Intuition:** The maximum of a uniform sample almost surely falls short of the true upper bound. The factor $\frac{n+1}{n}$ inflates it just enough to correct the bias on average.

---

## Problem 03 — Normal Variance: Minimal Sufficiency

> $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$ with **known** $\mu$, unknown $\sigma^2 > 0$.

### Part (a): Sufficiency of $T(\mathbf{X}) = \sum_{i=1}^n (X_i - \mu)^2$

The joint density is:

$$
f(\mathbf{x} \mid \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right) = (2\pi\sigma^2)^{-n/2} \exp\!\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2\right)
$$

Factor this as:

$$
f(\mathbf{x} \mid \sigma^2) = \underbrace{(2\pi\sigma^2)^{-n/2} \exp\!\left(-\frac{T(\mathbf{x})}{2\sigma^2}\right)}_{g(T(\mathbf{x}),\, \sigma^2)} \;\cdot\; \underbrace{1}_{h(\mathbf{x})}
$$

where $T(\mathbf{x}) = \sum_{i=1}^n (x_i - \mu)^2$.

The function $g$ depends on $\mathbf{x}$ only through $T(\mathbf{x})$, and $h(\mathbf{x}) = 1$ is free of $\sigma^2$. By the **factorization theorem**, $T(\mathbf{X})$ is sufficient for $\sigma^2$. $\blacksquare$

### Part (b): Minimal sufficiency via the likelihood ratio

Take two sample points $\mathbf{x}$ and $\mathbf{y}$ and form the likelihood ratio:

$$
\frac{f(\mathbf{x} \mid \sigma^2)}{f(\mathbf{y} \mid \sigma^2)}
= \frac{(2\pi\sigma^2)^{-n/2}\exp\!\big(-T(\mathbf{x})\,/\,2\sigma^2\big)}{(2\pi\sigma^2)^{-n/2}\exp\!\big(-T(\mathbf{y})\,/\,2\sigma^2\big)}
= \exp\!\left(-\frac{T(\mathbf{x}) - T(\mathbf{y})}{2\sigma^2}\right)
$$

**($\Rightarrow$) If $T(\mathbf{x}) = T(\mathbf{y})$:** The ratio becomes $e^0 = 1$, which is trivially free of $\sigma^2$. ✓

**($\Leftarrow$) If the ratio is free of $\sigma^2$:** The expression $\exp\!\left(-\frac{T(\mathbf{x}) - T(\mathbf{y})}{2\sigma^2}\right)$ is constant in $\sigma^2$ only when the exponent is zero, i.e., $T(\mathbf{x}) - T(\mathbf{y}) = 0$, which gives $T(\mathbf{x}) = T(\mathbf{y})$. ✓

We have shown:

$$
\frac{f(\mathbf{x} \mid \sigma^2)}{f(\mathbf{y} \mid \sigma^2)} \text{ is free of } \sigma^2 \quad \Longleftrightarrow \quad T(\mathbf{x}) = T(\mathbf{y})
$$

Therefore $T(\mathbf{X}) = \sum_{i=1}^n (X_i - \mu)^2$ is **minimal sufficient** for $\sigma^2$. $\blacksquare$

---

## Problem 04 — Binomial Sufficiency and Estimating $\pi^2$

> $X_i \overset{\text{i.i.d.}}{\sim} \text{Bernoulli}(\pi)$, $U(\mathbf{X}) = \sum_{i=1}^n X_i$.

### Part (a): $U/n$ is unbiased for $\pi$

By linearity of expectation:

$$
\mathbb{E}\!\left[\frac{U}{n}\right] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i] = \frac{1}{n} \cdot n\pi = \pi
$$

So $U/n = \bar{X}$ is unbiased for $\pi$. $\blacksquare$

### Part (b): $U(\mathbf{X})$ is minimal sufficient for $\pi$

The joint PMF is:

$$
f(\mathbf{x} \mid \pi) = \prod_{i=1}^n \pi^{x_i}(1-\pi)^{1-x_i} = \pi^{\sum x_i}(1-\pi)^{n - \sum x_i} = \pi^{u}(1-\pi)^{n-u}
$$

where $u = \sum_{i=1}^n x_i$. The likelihood ratio for two samples $\mathbf{x}$ and $\mathbf{y}$ with sums $u_x$ and $u_y$:

$$
\frac{f(\mathbf{x} \mid \pi)}{f(\mathbf{y} \mid \pi)} = \frac{\pi^{u_x}(1-\pi)^{n-u_x}}{\pi^{u_y}(1-\pi)^{n-u_y}} = \pi^{u_x - u_y}\,(1-\pi)^{u_y - u_x} = \left(\frac{\pi}{1-\pi}\right)^{u_x - u_y}
$$

**($\Rightarrow$) If $u_x = u_y$:** The ratio equals $\left(\frac{\pi}{1-\pi}\right)^0 = 1$, free of $\pi$. ✓

**($\Leftarrow$) If the ratio is free of $\pi$:** Since $\frac{\pi}{1-\pi}$ is a non-constant function of $\pi$ (for $\pi \in (0,1)$), we need $u_x - u_y = 0$, i.e., $u_x = u_y$. ✓

Therefore:

$$
\frac{f(\mathbf{x} \mid \pi)}{f(\mathbf{y} \mid \pi)} \text{ is free of } \pi \quad \Longleftrightarrow \quad U(\mathbf{x}) = U(\mathbf{y})
$$

So $U(\mathbf{X})$ is **minimal sufficient** for $\pi$. $\blacksquare$

### Part (c): $V(\mathbf{X}) = \frac{U(U-1)}{n(n-1)}$ is unbiased for $\pi^2$

We need to compute $\mathbb{E}[U(U-1)]$. Use the identity:

$$
U(U-1) = U^2 - U
$$

We know:
- $\mathbb{E}[U] = n\pi$
- $\text{Var}(U) = n\pi(1-\pi)$, so $\mathbb{E}[U^2] = \text{Var}(U) + (\mathbb{E}[U])^2 = n\pi(1-\pi) + n^2\pi^2$

Therefore:

$$
\mathbb{E}[U(U-1)] = \mathbb{E}[U^2] - \mathbb{E}[U] = n\pi(1-\pi) + n^2\pi^2 - n\pi
$$

Simplify:

$$
= n\pi - n\pi^2 + n^2\pi^2 - n\pi = n^2\pi^2 - n\pi^2 = n(n-1)\pi^2
$$

So:

$$
\mathbb{E}[V(\mathbf{X})] = \mathbb{E}\!\left[\frac{U(U-1)}{n(n-1)}\right] = \frac{n(n-1)\pi^2}{n(n-1)} = \pi^2
$$

$$
\boxed{\mathbb{E}[V(\mathbf{X})] = \pi^2}
$$

Therefore $V(\mathbf{X})$ is **unbiased** for $\pi^2$. $\blacksquare$

**Intuition:** Why does $U(U-1)/[n(n-1)]$ work? Think of it combinatorially. $\binom{U}{2}/\binom{n}{2}$ counts the fraction of pairs where *both* observations are 1. Each pair $(X_i, X_j)$ independently has $\mathbb{E}[X_i X_j] = \pi^2$ (for $i \neq j$), so averaging over all $\binom{n}{2}$ pairs gives an unbiased estimate of $\pi^2$.
