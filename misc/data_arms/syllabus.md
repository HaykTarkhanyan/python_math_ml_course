# Probability & Statistics for Data Science — Syllabus

> Short course (~12 weeks) for undergrads heading into data science.
> One 90-min lecture + one 60-min practical per week.

---

## Semester-Long Project: "Your Data, Your Story"

At the start of the course, **choose 2 things from your daily life** to track throughout the semester. Examples:

- Bus/metro arrival times (how late is it really?)
- Daily coffee consumption & sleep quality
- Number of messages you send per day
- Time spent on your phone
- Weather temperature at a fixed time each day
- How long your phone battery lasts

**Rules:** collect data every day (or as often as makes sense), record it in a spreadsheet. By the end of the course, you'll have real data to analyze with the tools you've learned — distributions, confidence intervals, hypothesis tests, correlation, and regression. The final project will be a short analysis of your two datasets.

---

## Prerequisites & Preparation

This course assumes basic **calculus** and **set theory**. You'll need these to work with CDFs, PDFs, expectations (integrals), and probability spaces. If you're rusty, review:

### Required background
- **Set theory**: unions, intersections, complements, De Morgan's laws, cardinality
- **Functions**: domain, range, injective/surjective/bijective, inverse functions
- **Derivatives**: basic rules (product, chain, quotient), partial derivatives
- **Integrals**: definite integrals, integration by parts, substitution — needed for continuous PDFs and expectations
- **Summation notation**: sigma sums, double sums, index manipulation

### Preparation materials

| Resource | Covers |
|----------|--------|
| [Metric Academy — Lecture 0: Sets, Combinatorics, Functions](<https://youtu.be/7rIw7ocwMP4)> | Set theory, combinatorics, functions |
| [Metric Academy — Practical 1: Sets, Combinatorics, Mappings](<https://youtu.be/vBgPpeX9aKA)> | Hands-on set theory & counting |
| [Metric Academy — Lecture 7: Sequences, Limits, Convergence](<https://youtu.be/9MeOlcKH0BY)> | Limits and convergence |
| [Metric Academy — Lecture 8: Continuity, Derivative](<https://youtu.be/nQydacDiLmk)> | Derivatives |
| [Metric Academy — Lecture 11: Integral](<https://youtu.be/J2tU1cUlZok)> | Definite & indefinite integrals |
| [Metric Academy — Practical 7: Integrals, Convolution](<https://youtu.be/FRjc1KiAbaw)> | Integral practice |
| Slides: [L00_Sets__Combinatorics__Functions.pdf](../../math/Lectures/L00_Sets__Combinatorics__Functions.pdf) | Sets & functions slides |
| Slides: [L07_Taylor_Series__Integral.pdf](../../math/Lectures/L07_Taylor_Series__Integral.pdf) | Integration slides |
| Book: [Կարճ մաթեմ մեքենայական ուսուցման համար.pdf](../../math/Lectures/Կարճ մաթեմ մեքենայական ուսուցման համար.pdf) | Compact math-for-ML book (Armenian) — covers sets through probability |
| [Khan Academy — Integrals](https://www.khanacademy.org/math/integral-calculus) | Review integration |
| [3Blue1Brown — Essence of Calculus](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Visual calculus intuition |

> If you're comfortable with all of the above, you're ready. If not, spend ~1 week reviewing before the course starts.

---

## Week 1 — Probability Foundations

**Topics:** Sample spaces, events, probability axioms, combinatorics (permutations & combinations), conditional probability, Bayes' theorem, independence.

### Metric Academy Resources
| Type | Link |
|------|------|
| Lecture 23 — Probability Intro | <https://youtu.be/OhbCxYucA4s> |
| Lecture 24 — Conditional Probability, Bayes, Independence | <https://youtu.be/KkWwnFRd0YU> |
| Lecture 25 — Geometric Probability | <https://youtu.be/7WB-qkVn9lo> |
| Practical 14 — Probability Foundations | <https://youtu.be/twHQTcN_48E> |
| Slides | [L09_Probability__Independence__Bayes_Rule.pdf](../../math/Lectures/L09_Probability__Independence__Bayes_Rule.pdf) |
| Homework (qmd) | [16_probability_intro.qmd](../../math/16_probability_intro.qmd) |
| Homework (PDF) | [hw_18_prob_1.pdf](../../math/Homeworks/hw_18_prob_1.pdf) |

### External Resources
- [3Blue1Brown — Bayes' theorem](<https://youtu.be/HZGCoVF3YvM)> — best visual intuition for Bayes
- [Khan Academy — Probability & Combinatorics](https://www.khanacademy.org/math/statistics-probability/probability-library)
- [StatQuest — Probability concepts](<https://youtu.be/uzkc-qNVoOk)>
- Textbook: Blitzstein & Hwang, *Introduction to Probability* (Harvard Stat 110) — Ch. 1-2

---

## Week 2 — Random Variables, PMF, CDF, PDF

**Topics:** Discrete vs continuous random variables, probability mass function (PMF), cumulative distribution function (CDF), probability density function (PDF), uniform distribution, LOTUS.

### Metric Academy Resources
| Type | Link |
|------|------|
| Lecture 26 — Random Variables | <https://youtu.be/eV1EaNM21_Q> |
| Lecture 27 — PMF, CDF, PDF, Uniform Distribution | <https://youtu.be/oienebEeHAE> |
| Practical 15 — Density/Distribution Functions, Independence | <https://youtu.be/_c_Jlt256Lc> |
| Slides | [L10_Random_Variables.pdf](../../math/Lectures/L10_Random_Variables.pdf) |
| Homework (PDF) | [hw_19_prob_2_rvs.pdf](../../math/Homeworks/hw_19_prob_2_rvs.pdf) |

### External Resources
- [3Blue1Brown — But what is a convolution?](<https://youtu.be/KuXjwB4LzSA)> — PDFs and convolutions
- [StatQuest — Histograms](<https://youtu.be/qBigTkBLU6g)> and [CDF](<https://youtu.be/YXLVjCKVP7U)>
- [Harvard Stat 110 — Random Variables (Lecture 8-9)](https://youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo)
- Textbook: Blitzstein & Hwang — Ch. 3

---

## Week 3 — Expectation, Variance & Inequalities

**Topics:** Expected value, linearity of expectation, variance, standard deviation, Markov's inequality, Chebyshev's inequality, Jensen's inequality.

### Metric Academy Resources
| Type | Link |
|------|------|
| Lecture 28 — Expectation & Variance | <https://youtu.be/sR-HbLj-Dg8> |
| Lecture 29 — Markov & Chebyshev Inequalities | <https://youtu.be/mDPO7KC5L-Q> |
| Practical 16 — E[X], Var, Inequalities | <https://youtu.be/8fkODa_yEEM> |
| Slides | [L11_Expected_Value__Variance.pdf](../../math/Lectures/L11_Expected_Value__Variance.pdf) |
| Homework (qmd) | [17_probability_exp_var_inequalities.qmd](../../math/17_probability_exp_var_inequalities.qmd) |
| Homework (PDF) | [hw_20_prob_3_exp_var_inequalities.pdf](../../math/Homeworks/hw_20_prob_3_exp_var_inequalities.pdf) |

### External Resources
- [StatQuest — Expected Values](<https://youtu.be/KLs_7b7SKi4)>
- [Seeing Theory (Brown U) — Expectation chapter](https://seeing-theory.brown.edu/basic-probability/) — beautiful interactive visualizations
- [Khan Academy — Variance](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/variance-standard-deviation-population/v/variance-of-a-population)
- Textbook: Blitzstein & Hwang — Ch. 4

---

## Week 4 — Covariance, Correlation & Joint Distributions

**Topics:** Joint distributions, marginals, covariance, Pearson correlation, Spearman rank correlation, correlation ≠ causation.

### Metric Academy Resources
| Type | Link |
|------|------|
| Lecture 30 — Covariance & Correlation | <https://youtu.be/bLpMxKkrKTI> |
| Lecture 31 — Correlation Pitfalls, Spearman | <https://youtu.be/5FdGFaG0pXg> |
| Practical 17 — Covariance & Correlation | <https://youtu.be/XxhWp0gmb1U> |
| Slides | [L12_Covariance__Correlation.pdf](../../math/Lectures/L12_Covariance__Correlation.pdf) |
| Homework (qmd) | [18_probability_corr_cov.qmd](../../math/18_probability_corr_cov.qmd) |
| Homework (PDF) | [hw_21_prob_4_corr_cov.pdf](../../math/Homeworks/hw_21_prob_4_corr_cov.pdf) |

### External Resources
- [StatQuest — Covariance and Correlation](<https://youtu.be/qtaqvPAeEJY)>
- [Spurious Correlations](https://www.tylervigen.com/spurious-correlations) — hilarious and pedagogically valuable
- [3Blue1Brown — But what is a correlation?](<https://youtu.be/nHOBjNdCCG8)>
- Textbook: Blitzstein & Hwang — Ch. 7

---

## Week 5 — Common Distributions

**Topics:** Bernoulli, Binomial, Geometric, Poisson (discrete); Uniform, Normal, Exponential, Beta (continuous). When to use each.

### Metric Academy Resources
| Type | Link |
|------|------|
| Lecture 32 — Bernoulli Distribution | <https://youtu.be/Pg8CFfxnI5I> |
| Lecture 33 — Geometric, Binomial, Poisson | <https://youtu.be/jTA1swTAbCs> |
| Lecture 34 — Continuous Distributions | <https://youtu.be/3By51o8_-8w> |
| Practical 18 — Distributions | <https://youtu.be/YhT1p1sGFp8> |
| Practical 19 — Distribution Problems | <https://youtu.be/D9JIZXNzios> |
| Slides | [L13_Distributions.pdf](../../math/Lectures/L13_Distributions.pdf) |
| Homework (qmd) | [19_probability_distributions.qmd](../../math/19_probability_distributions.qmd) |
| Homework (PDF) | [hw_22_prob_5_distribs.pdf](../../math/Homeworks/hw_22_prob_5_distribs.pdf) |

### External Resources
- [StatQuest — Normal Distribution](<https://youtu.be/rzFX5NWojp0)>
- [StatQuest — Poisson Distribution](<https://youtu.be/LN0xtUuJsEI)>
- [3Blue1Brown — Binomial Distributions](<https://youtu.be/8idr1WZ1A7Q)>
- [Seeing Theory — Distributions chapter](https://seeing-theory.brown.edu/probability-distributions/)
- Textbook: Blitzstein & Hwang — Ch. 5-6

---

## Week 6 — Convergence, Law of Large Numbers & Central Limit Theorem

**Topics:** Modes of convergence (in probability, almost sure, in distribution), Law of Large Numbers, Central Limit Theorem, why the normal distribution is everywhere.

### Metric Academy Resources
| Type | Link |
|------|------|
| Lecture 35 — Why RVs Are Functions | <https://youtu.be/mYNYarCTZVU> |
| Lecture 36 — Convergence of Random Variables | <https://youtu.be/oqB9pTXHxFM> |
| Lecture 37 — LLN & CLT | <https://youtu.be/uvdv-7ZlZNo> |
| Practical 20 — Convergence, LLN, CLT | <https://youtu.be/xnT7eaba_UM> |
| Slides | [L14_Comvergence__LLN__CLT.pdf](../../math/Lectures/L14_Comvergence__LLN__CLT.pdf) |
| Homework (qmd) | [20_probability_convergence_modes_lln_clt.qmd](../../math/20_probability_convergence_modes_lln_clt.qmd) |
| Homework (PDF) | [hw_23_prob_6_convergence_lln_clt.pdf](../../math/Homeworks/hw_23_prob_6_convergence_lln_clt.pdf) |

### External Resources
- [3Blue1Brown — Central Limit Theorem](<https://youtu.be/zeJD6dqJ5lo)> — the definitive visual explanation
- [StatQuest — Central Limit Theorem](<https://youtu.be/YAlJCEDp768)>
- [Seeing Theory — CLT simulation](https://seeing-theory.brown.edu/compound-probability/)
- Textbook: Blitzstein & Hwang — Ch. 10

---

## Week 7 — Statistics Fundamentals & Descriptive Statistics

**Topics:** Population vs sample, parameter vs statistic, i.i.d., plug-in principle, loss functions (L1/L2/L0), measures of center (mean, median, mode), measures of spread (variance, IQR), histograms, boxplots, ECDF, Anscombe's quartet, Simpson's paradox.

### Metric Academy Resources
| Type | Link |
|------|------|
| Lecture 38 — Statistics Foundations | <https://youtu.be/ulNS3QVenYo> |
| Lecture 39 — Descriptive Statistics & Data Visualization | <https://youtu.be/ice5rtgBOcA> |
| Slides (Foundations) | [01_stat.pdf](../../math/Lectures/stat/01_stat.pdf) |
| Slides (Descriptive Stats) | [02_stat.pdf](../../math/Lectures/stat/02_stat.pdf) |
| Notes (Foundations) | [01_stat_notes.pdf](../../math/Lectures/stat/01_stat_notes.pdf) |
| Notes (Descriptive Stats) | [02_stat_notes.pdf](../../math/Lectures/stat/02_stat_notes.pdf) |
| Homework (qmd) | [21_stat_fundamentals.qmd](../../math/21_stat_fundamentals.qmd) |

### External Resources
- [StatQuest — Histograms](<https://youtu.be/qBigTkBLU6g)>
- [Seeing Theory — Basic Probability](https://seeing-theory.brown.edu/basic-probability/)
- [Khan Academy — Descriptive statistics](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data)
- Textbook: Wasserman, *All of Statistics* — Ch. 1-3

---

## Week 8 — Estimators & Their Properties

**Topics:** Point estimation, bias, variance, MSE, bias-variance tradeoff, consistency, sufficiency, exponential family, Fisher information, Cramer-Rao bound.

### Metric Academy Resources
| Type | Link |
|------|------|
| Lecture 40 — Bias, MSE, Bias-Variance Tradeoff | <https://youtu.be/lAKPMjqQ6vc> |
| Lecture 41 — Consistency, Sufficiency | <https://youtu.be/Ye0ZsTDnPx4> |
| Lecture 42 — Fisher Info, Cramer-Rao, Minimax | <https://youtu.be/DDdzHnQsyrA> |
| Practical 21 — Sufficiency, Fisher-Neyman | <https://youtu.be/fxYxoHIz-P0> |
| Slides (Estimator Properties) | [03_stat.pdf](../../math/Lectures/stat/03_stat.pdf) |
| Notes (Estimator Properties) | [03_stat_notes.pdf](../../math/Lectures/stat/03_stat_notes.pdf) |
| Slides (Fisher Info & CR) | [04_stat.pdf](../../math/Lectures/stat/04_stat.pdf) |
| Notes (Fisher Info & CR) | [04_stat_notes.pdf](../../math/Lectures/stat/04_stat_notes.pdf) |
| Homework (qmd) | [22_stat_estimators.qmd](../../math/22_stat_estimators.qmd) |
| Homework (PDF) | [hw_24_stat_1_estimator_properties.pdf](../../math/Homeworks/hw_24_stat_1_estimator_properties.pdf) |

### External Resources
- [StatQuest — Bias and Variance](<https://youtu.be/EuBBz3bI-aA)> — the classic explanation
- [StatQuest — Maximum Likelihood (intro)](<https://youtu.be/XepXtl9YKwc)>
- [ritvikmath — Cramér-Rao Bound](<https://youtu.be/i0JiSddCXMM)>
- Textbook: Wasserman — Ch. 9; Casella & Berger — Ch. 7

> Note: Fisher information and Cramer-Rao can be kept light for the data science audience — focus on the intuition (more data = more precision, some estimators hit the efficiency floor).

---

## Week 9 — Maximum Likelihood & MAP Estimation

**Topics:** Method of Moments, MLE derivation & examples (Bernoulli, Normal, Poisson), MLE properties (consistency, asymptotic normality), Bayesian estimation, priors, posteriors, conjugate priors (Beta-Binomial, Normal-Normal), MAP, connection to regularization (Ridge = Gaussian prior, Lasso = Laplace prior).

### Metric Academy Resources
| Type | Link |
|------|------|
| Lecture 43 — MoM & MLE | <https://youtu.be/xQKwAcaM410> |
| Slides (MLE) | [05_stat.pdf](../../math/Lectures/stat/05_stat.pdf) |
| Notes (MLE) | [05_stat_notes.pdf](../../math/Lectures/stat/05_stat_notes.pdf) |
| Slides (MAP/Bayesian) | [06_stat.pdf](../../math/Lectures/stat/06_stat.pdf) |
| Notes (MAP/Bayesian) | [06_stat_notes.pdf](../../math/Lectures/stat/06_stat_notes.pdf) |
| Homework (qmd) | [23_stat_mle_map.qmd](../../math/23_stat_mle_map.qmd) |

### External Resources
- [StatQuest — Maximum Likelihood (detailed)](<https://youtu.be/XepXtl9YKwc)>
- [3Blue1Brown — Bayes theorem, the geometry of changing beliefs](<https://youtu.be/HZGCoVF3YvM)>
- [StatQuest — Ridge vs Lasso](<https://youtu.be/Q81RR3Y5fQk)> — regularization = priors connection
- [ritvikmath — MLE vs MAP](<https://youtu.be/kkhdIriddSI)>
- Textbook: Wasserman — Ch. 9; Murphy, *Probabilistic Machine Learning* — Ch. 4

---

## Week 10 — Sampling Distributions, Confidence Intervals & Bootstrap

**Topics:** Sampling distribution of estimators, standard error vs standard deviation, SE of X-bar = sigma/sqrt(n), CLT for X-bar, Wald CI, t-interval, Wilson interval, Delta method, bootstrap (nonparametric & parametric), bootstrap SE and CIs (Normal, Percentile, BCa).

### Metric Academy Resources
| Type | Link |
|------|------|
| Slides (Sampling Distributions) | [07_stat.pdf](../../math/Lectures/stat/07_stat.pdf) |
| Slides (CIs & Bootstrap) | [08_stat.pdf](../../math/Lectures/stat/08_stat.pdf) |
| Homework (qmd) | [24_stat_confidence_intervals.qmd](../../math/24_stat_confidence_intervals.qmd) (stub — ToDo) |

> No Metric Academy YouTube lecture covers this topic yet. No homework PDF exists yet.

### External Resources
- [StatQuest — Confidence Intervals](<https://youtu.be/TqOeMYtOc1w)>
- [StatQuest — Bootstrapping](<https://youtu.be/Xz0x-8-cgaQ)> — must-watch
- [3Blue1Brown — Why "margin of error" is misleading](<https://youtu.be/FnCIzGknGck)>
- [Seeing Theory — Frequentist Inference](https://seeing-theory.brown.edu/frequentist-inference/)
- [Khan Academy — Confidence Intervals](https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample)
- Textbook: Wasserman — Ch. 8; Efron & Tibshirani, *An Introduction to the Bootstrap*

---

## Week 11 — Hypothesis Testing

**Topics:** Null & alternative hypotheses, test statistics, p-values (definition and common misconceptions), Type I/II errors, significance level, power, effect size (Cohen's d), permutation tests, multiple testing corrections (Bonferroni, Benjamini-Hochberg), practical vs statistical significance, common tests (z-test, t-test, chi-squared, Mann-Whitney U, Wilcoxon), Likelihood Ratio Test, decision flowchart.

### Metric Academy Resources
| Type | Link |
|------|------|
| Slides (Hypothesis Testing) | [09_stat.pdf](../../math/Lectures/stat/09_stat.pdf) |
| Slides (Classical Tests & LRT) | [10_stat.pdf](../../math/Lectures/stat/10_stat.pdf) |
| Homework (qmd) | [25_stat_hypothesis_testing.qmd](../../math/25_stat_hypothesis_testing.qmd) (stub — ToDo) |

> No Metric Academy YouTube lecture covers this topic yet. No homework PDF exists yet.

### External Resources
- [StatQuest — p-values](<https://youtu.be/vemZtEM63GY)> — the most-watched p-value video on YouTube
- [StatQuest — Hypothesis Testing](<https://youtu.be/0oc49DyA3hU)>
- [StatQuest — FDR and Benjamini-Hochberg](<https://youtu.be/K8LQSvtjcEo)>
- [StatQuest — t-tests](<https://youtu.be/0Pd3dc1GcHc)>
- [StatQuest — Chi-squared test](<https://youtu.be/ZjdBM7NO7bY)>
- [Cassie Kozyrkov — Statistical significance is not practical significance](<https://youtu.be/E4KCfcVwzyw)>
- [Seeing Theory — Hypothesis Testing](https://seeing-theory.brown.edu/frequentist-inference/)
- Textbook: Wasserman — Ch. 10

---

## Week 12 — ANOVA, A/B Testing, Regression & Causal Inference

**Topics:** One-way ANOVA (F-statistic, post-hoc tests), A/B testing (design, sample size, peeking problem, sequential testing), simple linear regression (statistical view: OLS, t-test for coefficients, F-test, R-squared), GLMs (logistic, Poisson), causal inference intro (DAGs, correlation ≠ causation, RCTs, confounders).

### Metric Academy Resources
| Type | Link |
|------|------|
| Slides (ANOVA & A/B Testing) | [11_stat.pdf](../../math/Lectures/stat/11_stat.pdf) |
| Slides (Regression Inference) | [12_stat.pdf](../../math/Lectures/stat/12_stat.pdf) |
| Slides (GLMs) | [13_stat.pdf](../../math/Lectures/stat/13_stat.pdf) |
| Slides (Causal Inference) | [14_stat.pdf](../../math/Lectures/stat/14_stat.pdf) |

> No Metric Academy YouTube lectures cover these topics yet. No homework exists yet.
> Week 12 is dense — consider splitting into two weeks if time allows.

### External Resources
- [StatQuest — Linear Regression](<https://youtu.be/nk2CQITm_eo)>
- [StatQuest — R-squared](<https://youtu.be/2AQKCzFe2xw)>
- [StatQuest — Logistic Regression](<https://youtu.be/yIYKR4sgzI8)>
- [StatQuest — One-way ANOVA](<https://youtu.be/oOuu8IICZRo)>
- [Cassie Kozyrkov — A/B Testing](<https://youtu.be/DUNk4GPZ9bw)>
- [Brady Neal — Causal Inference (lecture series)](https://youtube.com/playlist?list=PLoazKTcS0RzZ1SUgeOgc6SWt51gfT80N0) — excellent free course
- [Khan Academy — Regression](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data)
- Textbook: Wasserman — Ch. 13; Pearl, *The Book of Why* (popular science, great for intuition)

---

## Recommended Textbooks

| Book | Level | Best for |
|------|-------|----------|
| Blitzstein & Hwang, *Introduction to Probability* | Undergrad | Weeks 1-6 (probability). Gold standard. [Free PDF + Harvard Stat 110 lectures](https://projects.iq.harvard.edu/stat110) |
| Wasserman, *All of Statistics* | Undergrad/Grad bridge | Weeks 7-12 (statistics). Concise, covers everything a data scientist needs |
| Casella & Berger, *Statistical Inference* | Advanced undergrad | Deep theory (estimators, sufficiency, Neyman-Pearson). Reference, not primary |
| Murphy, *Probabilistic Machine Learning* | Grad | Bridge to ML. [Free PDF](https://probml.github.io/pml-book/) |
| Diez, Cetinkaya-Rundel & Barr, *OpenIntro Statistics* | Intro undergrad | Gentler alternative. [Free PDF](https://www.openintro.org/book/os/) |

## Recommended YouTube Channels

| Channel | Why |
|---------|-----|
| [Metric Academy](https://www.youtube.com/@MetricAcademy) | This course's own lectures (Armenian + English terms) |
| [StatQuest (Josh Starmer)](https://www.youtube.com/@statquest) | Best bite-sized stats explainers on YouTube |
| [3Blue1Brown](https://www.youtube.com/@3blue1brown) | Unmatched visual intuition for probability & linear algebra |
| [Harvard Stat 110](https://youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) | Full probability course, great problem-solving |
| [ritvikmath](https://www.youtube.com/@ritvikmath) | Short, clear explanations of intermediate topics |
| [Seeing Theory (Brown U)](https://seeing-theory.brown.edu/) | Interactive web visualizations (not YouTube, but essential) |
| [Cassie Kozyrkov](https://www.youtube.com/@kozyrkov) | Practical data science perspective on statistics |

---

*Auto-generated syllabus — YouTube links auto-extracted from Metric Academy channel. If you spot a mistake or broken link, please let me know.*
