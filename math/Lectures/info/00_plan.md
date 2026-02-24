# Information Theory Lectures — Plan & Summary

## Overview

Three Beamer lectures on **Information Theory for ML**, covering the core concepts from Shannon entropy through mutual information, then extending to f-divergences, variational inference, and the information bottleneck.

Based on material from [slds-lmu/lecture_sl](https://github.com/slds-lmu/lecture_sl/tree/main/slides/information-theory), adapted to our course style.

---

## Lecture 1: Entropy, Cross-Entropy, KL Divergence (`info_01.tex`, 31 pages)

### Sections

| Section | Frames | Content |
|---------|--------|---------|
| Entropy | 6 | Surprisal, Shannon entropy (definition + worked example), Bernoulli entropy (pgfplots), properties table, Khinchin uniqueness, max entropy = uniform (Lagrange proof) |
| Source Coding | 5 | Fixed-length codes, variable-length codes, prefix property, prefix code tree (TikZ), optimal code length = entropy, Shannon's source coding theorem |
| Cross-Entropy | 2 | Wrong codebook table (waste column), cross-entropy definition and properties |
| KL Divergence | 6 | Derivation from cross-entropy, KL definition, fundamental identity (bar diagram), information inequality (Gibbs' inequality via Jensen), KL is not a distance (comparison table), three interpretations |
| Summary | 2 | Differential entropy (brief), big picture diagram (Entropy -> Cross-Entropy -> KL) |
| Homework | 1 | 4 problems: Bernoulli entropy, information inequality proof, full KL/CE computation, Huffman code design |

### Key Formulas Introduced

- Surprisal: $I(x) = -\log_2 p(x)$
- Shannon entropy: $H(X) = -\sum_x p(x) \log_2 p(x)$
- Cross-entropy: $H(p \| q) = -\sum_x p(x) \log_2 q(x)$
- KL divergence: $D_{\text{KL}}(p \| q) = \sum_x p(x) \log_2 \frac{p(x)}{q(x)}$
- Fundamental identity: $H(p \| q) = H(p) + D_{\text{KL}}(p \| q)$

---

## Lecture 2: Information Theory for ML (`info_02.tex`, 32 pages)

### Sections

| Section | Frames | Content |
|---------|--------|---------|
| KL and Maximum Likelihood | 5 | Recap, KL=MLE derivation (3-step proof), cross-entropy = MLE, cross-entropy loss in classification (multiclass + binary), entropy as baseline risk |
| Forward vs Reverse KL | 3 | Two directions comparison, bimodal Gaussian fitting visualization, use cases table (supervised learning vs variational inference) |
| Maximum Entropy Principle | 3 | MaxEnt principle (Jaynes), Lagrangian solution = exponential family, familiar distributions table, biased die example (bar chart) |
| Mutual Information | 7 | Joint & conditional entropy, chain rule, MI definition (= KL between joint and product of marginals), information diagram (Venn circles), properties (6 properties), worked example (joint table), MI vs correlation (scatter plots), MI for Gaussians (formula + plot), information gain in decision trees |
| Summary | 1 | Toolbox table (7 concepts with formulas and ML roles) |
| Homework | 1 | 4 problems: cross-entropy = MLE proof, MI verification, feature selection via MI, forward/reverse KL intuition |

### Key ML Connections

- **KL = MLE:** Minimizing $D_{\text{KL}}(p \| q_\theta)$ is equivalent to maximizing log-likelihood
- **Cross-entropy loss:** The standard classification loss IS cross-entropy IS negative log-likelihood
- **Forward KL:** Mass-covering, used in supervised learning
- **Reverse KL:** Mode-seeking, used in variational inference
- **MaxEnt = Exponential family:** Gaussian is MaxEnt given mean and variance
- **Information gain:** Decision tree splitting criterion = mutual information
- **MI vs correlation:** MI detects any dependence (not just linear)

---

## Lecture 3: Advanced Topics for ML (`info_03.tex`, 33 pages)

### Sections

| Section | Frames | Content |
|---------|--------|---------|
| Data Processing Inequality | 4 | Markov chains, DPI statement + proof sketch, DPI in neural networks (layers lose info), practical consequences (feature engineering, dimensionality reduction, sufficient statistics, post-processing) |
| f-Divergences | 4 | f-divergence definition (Ali-Silvey, Csiszar), family table (KL, reverse KL, TV, chi-squared, JS, Hellinger), Jensen-Shannon (symmetric, bounded, metric), f-divergences & GANs (GAN = minimize JSD, f-GAN), support mismatch & Wasserstein |
| ELBO & VAEs | 5 | Intractable posteriors (latent variable models), ELBO derivation (Jensen's inequality), ELBO = reconstruction - KL penalty, VAE architecture (encoder/reparameterization/decoder), VAE loss (closed-form KL for Gaussians, beta-VAE), ELBO landscape diagram |
| Information Bottleneck | 4 | IB principle (Tishby et al. 1999), information plane (IB curve), IB & deep learning (fitting + compression phases, Tishby & Zaslavsky 2015), debate & caveats, IB connects to everything (hub diagram) |
| Summary | 1 | Toolbox table (6 concepts with key ideas and ML applications) |
| Homework | 1 | 4 problems: DPI application, f-divergence verification, ELBO derivation, IB tradeoff analysis |

### Key Concepts

- **DPI:** Processing can only destroy information; $I(X;Z) \leq I(X;Y)$ for $X \to Y \to Z$
- **f-divergences:** Unified family including KL, TV, chi-squared, JS, Hellinger
- **Jensen-Shannon:** Symmetric + bounded KL; original GAN objective
- **Support mismatch:** Why KL and JS fail when distributions don't overlap; motivation for Wasserstein
- **ELBO:** $\log p(\mathbf{x}) = \text{ELBO} + D_{\text{KL}}(q \| p)$; reconstruction minus KL penalty
- **VAE:** Neural network implementation of ELBO with reparameterization trick
- **Information Bottleneck:** $\min I(X;T) - \beta I(T;Y)$; compress input, preserve task-relevant info
- **Information plane:** Visualizing what layers learn (fitting phase → compression phase)

---

## Prerequisites

| This lecture uses | From |
|-------------------|------|
| Probability basics, Bayes | Module 16 |
| Expectation, variance | Module 17 |
| Distributions (Bernoulli, Gaussian) | Module 19 |
| Convexity, Jensen's inequality | Module 05 |
| Lagrange multipliers | Module 05 / 07 |
| MLE, log-likelihood | Stat Lecture 5 (`05_stat.tex`) |
| Entropy, KL, cross-entropy (L3) | `info_01.tex`, `info_02.tex` |
| Forward/reverse KL (L3) | `info_02.tex` |
| Markov chains (L3) | Module 16 (conditional probability) |

## Style

- Beamer, `\usetheme{default}`, `\usecolortheme{dove}`, 16:9
- All diagrams are TikZ/pgfplots (no external images)
- Colors: popblue, sampred, paramgreen, warnred, orange1, violet1
- `fcolorbox` for key results, TikZ nodes for definitions
- Compiled with pdflatex (TeX Live 2025), zero overflow warnings
