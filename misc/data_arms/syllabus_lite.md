# Probability & Statistics for Data Science — Syllabus (Lite)

> A practical, 12-week probability & statistics course for undergrads heading into data science.
> One 90-min lecture + one 60-min practical per week.
> This is a standalone course. It cherry-picks materials from the Metric Academy math course but is independent from it.

---

## Semester-Long Project: "Your Data, Your Story"

Choose 2 things from your daily life to track throughout the semester (bus arrival times, coffee consumption, sleep, phone screen time, etc.). Collect data daily in a spreadsheet. By the end of the course, you'll analyze it using the tools you've learned.

**Checkpoints:** Week 6 (explore & describe), Week 9 (confidence intervals), Week 12 (full analysis + presentation).

See **[project-guide.md](project-guide.md)** for full details, rubric, and peer review instructions.

---

## Prerequisites

Basic **calculus** (derivatives, definite integrals) and **set theory** (unions, intersections, complements). If rusty, review:

| Resource | Covers |
|----------|--------|
| [3Blue1Brown — Essence of Calculus](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Visual calculus intuition |
| [Khan Academy — Integrals](https://www.khanacademy.org/math/integral-calculus) | Integration review |
| [Metric Academy — Lecture 0: Sets, Combinatorics, Functions](<https://youtu.be/7rIw7ocwMP4)> | Set theory basics |
| [Metric Academy — Lecture 11: Integrals](<https://youtu.be/J2tU1cUlZok)> | Definite & indefinite integrals |

---

## Supporting Materials

| File | What's in it |
|------|-------------|
| [warm-ups.md](warm-ups.md) | Per-week puzzles and in-class activities (Monty Hall, birthday problem, guess the distribution, etc.) |
| [case-studies.md](case-studies.md) | Per-week real-world stories (WWII Poisson bombs, replication crisis, spam filters, etc.) |
| [assessment.md](assessment.md) | Grading breakdown, quiz examples, homework schedule & sample problems |
| [project-guide.md](project-guide.md) | Full project guide with checkpoints, rubric, peer review, presentation tips |

---

## Week 1 — What is Probability?

**Topics:** Sample spaces, events, probability axioms, counting (permutations & combinations), conditional probability, Bayes' theorem, independence.

**Why it matters:** Every ML model outputs probabilities. Understanding what they mean is non-negotiable.

### Resources
| Type | Link |
|------|------|
| Metric Lecture 23 — Probability Intro | <https://youtu.be/OhbCxYucA4s> |
| Metric Lecture 24 — Conditional Prob, Bayes, Independence | <https://youtu.be/KkWwnFRd0YU> |
| Metric Lecture 25 — Geometric Probability *(optional)* | <https://youtu.be/7WB-qkVn9lo> |
| Metric Practical 14 | <https://youtu.be/twHQTcN_48E> |
| Slides | [L09_Probability__Independence__Bayes_Rule.pdf](../../math/Lectures/L09_Probability__Independence__Bayes_Rule.pdf) |
| Textbook (Armenian) | [Կարճ մաթեմատիկա ՄԼ-ի համար.pdf](../../math/Lectures/%D4%BF%D5%A1%D6%80%D5%B3%20%D5%B4%D5%A1%D5%A9%D5%A5%D5%B4%20%D5%B4%D5%A5%D6%84%D5%A5%D5%B6%D5%A1%D5%B5%D5%A1%D5%AF%D5%A1%D5%B6%20%D5%B8%D6%82%D5%BD%D5%B8%D6%82%D6%81%D5%B4%D5%A1%D5%B6%20%D5%B0%D5%A1%D5%B4%D5%A1%D6%80.pdf) — probability chapters (stat not covered) |
| [3Blue1Brown — Bayes' theorem](<https://youtu.be/HZGCoVF3YvM)> | Best visual explanation of Bayes |
| [StatQuest — Probability concepts](<https://youtu.be/uzkc-qNVoOk)> | Quick overview |
| Textbook | Blitzstein & Hwang, *Intro to Probability* — Ch. 1-2 |

---

## Week 2 — Random Variables & Distributions

**Topics:** Discrete vs continuous random variables, PMF, CDF, PDF, uniform distribution.

**Why it matters:** Data = realizations of random variables. You need the language to describe them.

### Resources
| Type | Link |
|------|------|
| Metric Lecture 26 — Random Variables | <https://youtu.be/eV1EaNM21_Q> |
| Metric Lecture 27 — PMF, CDF, PDF | <https://youtu.be/oienebEeHAE> |
| Metric Practical 15 | <https://youtu.be/_c_Jlt256Lc> |
| Slides | [L10_Random_Variables.pdf](../../math/Lectures/L10_Random_Variables.pdf) |
| [StatQuest — Histograms](<https://youtu.be/qBigTkBLU6g)> | Connecting data to distributions |
| Textbook | Blitzstein & Hwang — Ch. 3 |

---

## Week 3 — Expectation & Variance

**Topics:** Expected value, linearity of expectation, variance, standard deviation. Brief mention of Chebyshev's inequality (intuition only: "unlikely to be far from the mean").

**Why it matters:** Mean and variance are the two numbers you'll compute most often in data science.

> **Skipping:** Formal Markov/Jensen inequalities, LOTUS proofs. Just know the results.

### Resources
| Type | Link |
|------|------|
| Metric Lecture 28 — Expectation & Variance | <https://youtu.be/sR-HbLj-Dg8> |
| Metric Lecture 29 — Markov & Chebyshev *(optional)* | <https://youtu.be/mDPO7KC5L-Q> |
| Metric Practical 16 — E[X], Var, Inequalities | <https://youtu.be/8fkODa_yEEM> |
| Slides | [L11_Expected_Value__Variance.pdf](../../math/Lectures/L11_Expected_Value__Variance.pdf) |
| [StatQuest — Expected Values](<https://youtu.be/KLs_7b7SKi4)> | Quick and clear |
| [Seeing Theory — Expectation](https://seeing-theory.brown.edu/basic-probability/) | Interactive visualizations |
| Textbook | Blitzstein & Hwang — Ch. 4 |

---

## Week 4 — Common Distributions

**Topics:** Bernoulli, Binomial, Poisson (discrete); Normal, Exponential, Uniform (continuous). When to use which. Intuition for each.

**Why it matters:** You'll see these everywhere — A/B tests (Bernoulli), counts (Poisson), measurements (Normal), waiting times (Exponential).

> **Skipping:** Beta distribution, Geometric distribution (mention briefly). No MGFs.

### Resources
| Type | Link |
|------|------|
| Metric Lecture 32 — Bernoulli | <https://youtu.be/Pg8CFfxnI5I> |
| Metric Lecture 33 — Binomial, Poisson | <https://youtu.be/jTA1swTAbCs> |
| Metric Lecture 34 — Continuous Distributions | <https://youtu.be/3By51o8_-8w> |
| Metric Practical 18 — Distributions | <https://youtu.be/YhT1p1sGFp8> |
| Metric Practical 19 — Distribution Problems | <https://youtu.be/D9JIZXNzios> |
| Slides | [L13_Distributions.pdf](../../math/Lectures/L13_Distributions.pdf) |
| [StatQuest — Normal Distribution](<https://youtu.be/rzFX5NWojp0)> | Visual explainer |
| [StatQuest — Poisson Distribution](<https://youtu.be/LN0xtUuJsEI)> | Visual explainer |
| [Seeing Theory — Distributions](https://seeing-theory.brown.edu/probability-distributions/) | Play with parameters |
| Textbook | Blitzstein & Hwang — Ch. 5-6 |

---

## Week 5 — Correlation & the Law of Large Numbers

**Topics:** Covariance, Pearson correlation, Spearman rank correlation, correlation ≠ causation. Law of Large Numbers (intuition: averages stabilize), Central Limit Theorem (intuition: averages are approximately normal).

**Why it matters:** Correlation is probably the single most used (and misused) concept in data analysis. LLN/CLT explain why sampling works.

> **Note:** This is a dense week. Consider covering correlation in the lecture and LLN/CLT in the practical session, or letting CLT spill into Week 6.

> **Skipping:** Formal convergence modes (almost sure, in distribution, etc.). Just the intuition.

### Resources
| Type | Link |
|------|------|
| Metric Lecture 30 — Covariance & Correlation | <https://youtu.be/bLpMxKkrKTI> |
| Metric Lecture 31 — Correlation Pitfalls, Spearman | <https://youtu.be/5FdGFaG0pXg> |
| Metric Lecture 37 — LLN & CLT | <https://youtu.be/uvdv-7ZlZNo> |
| Metric Practical 17 — Covariance & Correlation | <https://youtu.be/XxhWp0gmb1U> |
| Metric Practical 20 — Convergence, LLN, CLT | <https://youtu.be/xnT7eaba_UM> |
| Slides (Correlation) | [L12_Covariance__Correlation.pdf](../../math/Lectures/L12_Covariance__Correlation.pdf) |
| Slides (CLT) | [L14_Comvergence__LLN__CLT.pdf](../../math/Lectures/L14_Comvergence__LLN__CLT.pdf) |
| [Spurious Correlations](https://www.tylervigen.com/spurious-correlations) | Hilarious and educational |
| [3Blue1Brown — Central Limit Theorem](<https://youtu.be/zeJD6dqJ5lo)> | The definitive visual explanation |
| Textbook | Blitzstein & Hwang — Ch. 7, 10 |

---

## Week 6 — Descriptive Statistics & EDA

**Topics:** Population vs sample, measures of center (mean, median, mode), measures of spread (variance, IQR, range), histograms, boxplots, ECDF, Anscombe's quartet ("always plot your data"), Simpson's paradox.

**Why it matters:** Before any fancy model, you explore data. This is the toolkit.

### Resources
| Type | Link |
|------|------|
| Metric Lecture 38 — Statistics Foundations | <https://youtu.be/ulNS3QVenYo> |
| Metric Lecture 39 — Descriptive Stats & Visualization | <https://youtu.be/ice5rtgBOcA> |
| Slides (Foundations) | [01_stat.pdf](../../math/Lectures/stat/01_stat.pdf) |
| Slides (Descriptive) | [02_stat.pdf](../../math/Lectures/stat/02_stat.pdf) |
| Notes | [01_stat_notes.pdf](../../math/Lectures/stat/01_stat_notes.pdf), [02_stat_notes.pdf](../../math/Lectures/stat/02_stat_notes.pdf) |
| [StatQuest — Histograms](<https://youtu.be/qBigTkBLU6g)> | |
| [Khan Academy — Descriptive statistics](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data) | |
| Textbook | Wasserman, *All of Statistics* — Ch. 3 |

---

## Week 7 — Estimation & the Power of Simulation

**Topics:** Point estimation, bias (is my estimate systematically off?), variance of an estimator, MSE = bias² + variance, the bias-variance tradeoff (intuition for ML). Brief: consistency (more data → better estimates). **Monte Carlo simulation** — "don't know the answer? Simulate it 10,000 times." Use simulation to verify formulas, estimate probabilities, and build intuition.

**Why it matters:** Every time you compute a mean from a sample, you're estimating. Understanding bias and variance is the foundation of all ML. And simulation is the single most useful trick for a working data scientist — if you can simulate it, you can answer it.

> **Skipping:** Sufficiency, exponential family, Fisher information, Cramér-Rao bound, minimax, admissibility. These are beautiful theory but not essential for applied data science.

### Resources
| Type | Link |
|------|------|
| Metric Lecture 40 — Bias, MSE, Bias-Variance Tradeoff | <https://youtu.be/lAKPMjqQ6vc> |
| Slides | [03_stat.pdf](../../math/Lectures/stat/03_stat.pdf) |
| Notes | [03_stat_notes.pdf](../../math/Lectures/stat/03_stat_notes.pdf) |
| [StatQuest — Bias and Variance](<https://youtu.be/EuBBz3bI-aA)> | The classic explanation |
| Textbook | Wasserman — Ch. 9 (sections 9.1-9.3 only) |

---

## Week 8 — Maximum Likelihood & Bayesian Intuition

**Topics:** MLE idea (find the parameters that make the observed data most probable), MLE examples (Bernoulli, Normal, Poisson), MLE = minimizing cross-entropy = minimizing MSE (for Gaussian noise). Brief: method of moments as a simpler alternative. **Bayesian thinking (lite, 20 min):** the idea of updating beliefs with evidence — a surgeon does 3 surgeries, all successful, is she really 100% accurate? Prior knowledge + data = better estimate. No formulas needed, just the idea.

**Why it matters:** MLE is the workhorse of modern statistics and ML. Logistic regression, neural networks — they're all doing MLE under the hood. Bayesian intuition (even without the math) helps you think about priors, regularization, and why small samples can be misleading.

> **Skipping:** Full Bayesian machinery (conjugate priors, posteriors, MAP). The math will come in the ML course when you need regularization. Here we just plant the seed.
> Also skipping: asymptotic normality proofs, regularity conditions, MLE & sufficiency connection.

### Resources
| Type | Link |
|------|------|
| Metric Lecture 43 — MoM & MLE | <https://youtu.be/xQKwAcaM410> |
| Slides | [05_stat.pdf](../../math/Lectures/stat/05_stat.pdf) |
| Notes | [05_stat_notes.pdf](../../math/Lectures/stat/05_stat_notes.pdf) |
| [StatQuest — Maximum Likelihood](<https://youtu.be/XepXtl9YKwc)> | Must-watch |
| [3Blue1Brown — Bayes theorem](<https://youtu.be/HZGCoVF3YvM)> | Visual Bayesian intuition (just watch, no math needed) |
| Textbook | Wasserman — Ch. 9 (section 9.4) |

---

## Week 9 — Confidence Intervals & Bootstrap

**Topics:** What a confidence interval actually means (and what it doesn't), Wald CI, t-interval, bootstrap (the Swiss army knife: resample → compute → repeat), bootstrap CIs. Standard error vs standard deviation. **Prediction interval vs confidence interval** — students confuse these constantly; a CI is about the mean, a prediction interval is about the next observation.

**Why it matters:** "The average is 42" is useless without "± something". CIs quantify uncertainty. Bootstrap lets you get CIs for anything.

> **Skipping:** Delta method, BCa bootstrap, Fisher information connection to CIs. Just use bootstrap when in doubt.

### Resources
| Type | Link |
|------|------|
| Slides (Sampling Distributions) | [07_stat.pdf](../../math/Lectures/stat/07_stat.pdf) |
| Slides (CIs & Bootstrap) | [08_stat.pdf](../../math/Lectures/stat/08_stat.pdf) |
| [StatQuest — Confidence Intervals](<https://youtu.be/TqOeMYtOc1w)> | Clear walkthrough |
| [StatQuest — Bootstrapping](<https://youtu.be/Xz0x-8-cgaQ)> | Must-watch |
| [Seeing Theory — Frequentist Inference](https://seeing-theory.brown.edu/frequentist-inference/) | Interactive CI demo |
| [Khan Academy — Confidence Intervals](https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample) | |
| Textbook | Wasserman — Ch. 8 |

---

## Week 10 — Hypothesis Testing & p-values

**Topics:** Null & alternative hypotheses, test statistics, p-values (what they are, what they are NOT), Type I/II errors, significance level, **power analysis** (how many samples do I need to detect an effect?), effect size (Cohen's d), one-sample & two-sample t-tests, chi-squared test, permutation tests (the bootstrap of hypothesis testing).

**Why it matters:** p-values are everywhere in science and industry. Misunderstanding them leads to bad decisions. This week is about not being one of those people. Power analysis answers the question every data scientist gets asked: "how much data do we need?"

> **Skipping:** Neyman-Pearson lemma, Likelihood Ratio Test theory, UMP tests. Focus on practical testing and interpretation.

### Resources
| Type | Link |
|------|------|
| Slides (Hypothesis Testing) | [09_stat.pdf](../../math/Lectures/stat/09_stat.pdf) |
| Slides (Classical Tests) | [10_stat.pdf](../../math/Lectures/stat/10_stat.pdf) |
| [StatQuest — p-values](<https://youtu.be/vemZtEM63GY)> | Most-watched p-value video on YouTube |
| [StatQuest — Hypothesis Testing](<https://youtu.be/0oc49DyA3hU)> | Overview |
| [StatQuest — t-tests](<https://youtu.be/0Pd3dc1GcHc)> | Practical walkthrough |
| [StatQuest — Chi-squared test](<https://youtu.be/ZjdBM7NO7bY)> | Categorical data testing |
| [Seeing Theory — Hypothesis Testing](https://seeing-theory.brown.edu/frequentist-inference/) | Interactive demo |
| Textbook | Wasserman — Ch. 10 |

---

## Week 11 — A/B Testing & Multiple Comparisons

**Topics:** A/B testing (design, sample size planning, when to stop), the peeking problem ("just one more look..."), multiple testing problem (test 20 things → one will be "significant" by chance), Bonferroni correction, false discovery rate (Benjamini-Hochberg). Practical vs statistical significance.

**Why it matters:** A/B testing is how tech companies make decisions. Getting it wrong wastes money and ships bad products.

### Resources
| Type | Link |
|------|------|
| Slides (ANOVA & A/B Testing) | [11_stat.pdf](../../math/Lectures/stat/11_stat.pdf) |
| [StatQuest — FDR and Benjamini-Hochberg](<https://youtu.be/K8LQSvtjcEo)> | Multiple testing correction |
| [Cassie Kozyrkov — A/B Testing](<https://youtu.be/DUNk4GPZ9bw)> | Industry perspective |
| Textbook | Wasserman — Ch. 10 (multiple testing section) |

---

## Week 12 — ANOVA, Causal Thinking & Course Wrap-Up

**Topics:** The multiple comparisons motivation: why not just do many t-tests? One-way ANOVA (F-statistic, SS decomposition, ANOVA table), post-hoc tests (Tukey HSD), effect size (eta-squared). Brief: correlation ≠ causation revisited, intro to causal thinking (DAGs, confounders — what to watch out for).

**Why it matters:** ANOVA is the natural next step after t-tests — comparing more than two groups at once. It's everywhere in experimental science and A/B testing with multiple variants. Causal thinking protects you from drawing wrong conclusions.

> **Skipping:** Two-way ANOVA, regression (will be covered in the ML course), GLMs. Linear regression is better introduced alongside ML where you'll actually use it with code.

### Resources
| Type | Link |
|------|------|
| Slides (ANOVA & A/B Testing) | [11_stat.pdf](../../math/Lectures/stat/11_stat.pdf) |
| Slides (Causal Inference) | [14_stat.pdf](../../math/Lectures/stat/14_stat.pdf) |
| [StatQuest — One-way ANOVA](<https://youtu.be/oOuu8IICZRo)> | Visual walkthrough |
| [StatQuest — Tukey's test](<https://youtu.be/UMgiUOGT1qg)> | Post-hoc comparisons |
| [Brady Neal — Causal Inference](https://youtube.com/playlist?list=PLoazKTcS0RzZ1SUgeOgc6SWt51gfT80N0) | If you want to go deeper |
| Textbook | Wasserman — Ch. 10 (ANOVA section) |

**Final project due:** Analyze your two semester-long datasets. What distributions fit? Are they correlated? Can you build a CI for the mean? Test a hypothesis?

---

## Bonus — How to Lie with Statistics

A fun, non-technical wrap-up lecture. How graphs, averages, and p-values get weaponized — and how to spot it.

**Topics:** Misleading axes and truncated graphs, cherry-picking data, survivorship bias, base rate neglect, Simpson's paradox revisited, p-hacking ("torture the data until it confesses"), misleading percentages ("up 200%!"), selection bias in polls and surveys, how news headlines abuse statistics.

**Why it matters:** Knowing statistics isn't just about computing things — it's about not getting fooled. This is the lecture students remember years later.

### Resources
| Type | Link |
|------|------|
| Book | Darrell Huff, *How to Lie with Statistics* (1954) — short, funny, still relevant |
| [Calling Bullshit — course videos](https://www.callingbullshit.org/videos.html) | Full university course by Carl Bergstrom & Jevin West (U of Washington) |
| [StatQuest — p-hacking](<https://youtu.be/Gx0fAjNHb1M)> | What happens when you test everything |
| [Spurious Correlations](https://www.tylervigen.com/spurious-correlations) | Correlation between Nicolas Cage films and pool drownings |
| [Survivorship bias — airplane example](../../math/Lectures/stat/survivorship_airplane.pdf) | The classic WWII bullet hole story |
| Book | Joel Best, *Damned Lies and Statistics* — how bad stats spread through media |
| Book | Tim Harford, *The Data Detective* — 10 rules for thinking clearly about numbers |

---

## What We Deliberately Skipped (and Where to Find It)

These topics are important for statisticians but not essential for a first data science course. If you're curious:

| Topic | Why skipped | Where to learn |
|-------|-------------|----------------|
| Fisher information & Cramér-Rao | Beautiful theory, but you can do great data science without it | Metric Lecture 42, Casella & Berger Ch. 7 |
| Sufficiency & exponential families | Theoretical elegance, rarely used directly in practice | Metric Lecture 41, Practical 21, Wasserman Ch. 9 |
| Formal convergence modes | You need LLN/CLT intuition, not the four types of convergence | Metric Lectures 35-36 |
| Minimax estimation | Decision theory — fascinating but niche | Metric Lecture 42 |
| Estimator properties (deep) | Consistency, sufficiency, detailed theorems | [04_stat.pdf](../../math/Lectures/stat/04_stat.pdf), Metric Lectures 41-42 |
| MAP & Bayesian estimation | Important but better introduced alongside ML (regularization) | [06_stat.pdf](../../math/Lectures/stat/06_stat.pdf), Metric Lecture 43 |
| Linear regression (statistical view) | Will be covered in the ML course with hands-on code | Stat slides 12, StatQuest, Wasserman Ch. 13 |
| GLMs (logistic, Poisson regression) | Better as the first topic of an ML course | Stat slides 13 |
| Moment generating functions | Proof technique, not a practical tool | Blitzstein Ch. 6 |

---

## Recommended Materials

### Textbooks
| Book | Level | Best for |
|------|-------|----------|
| [Կարnothing մ...](../../math/Lectures/%D4%BF%D5%A1%D6%80%D5%B3%20%D5%B4%D5%A1%D5%A9%D5%A5%D5%B4%20%D5%B4%D5%A5%D6%84%D5%A5%D5%B6%D5%A1%D5%B5%D5%A1%D5%AF%D5%A1%D5%B6%20%D5%B8%D6%82%D5%BD%D5%B8%D6%82%D6%81%D5%B4%D5%A1%D5%B6%20%D5%B0%D5%A1%D5%B4%D5%A1%D6%80.pdf) — Math for ML (Armenian) | Undergrad | Weeks 1-5 probability chapters. Stat not covered. |
| Blitzstein & Hwang, *Introduction to Probability* | Undergrad | Weeks 1-5. [Free PDF + Harvard Stat 110 lectures](https://projects.iq.harvard.edu/stat110) |
| Wasserman, *All of Statistics* | Undergrad/Grad bridge | Weeks 6-12. Concise, data-science-friendly |
| Diez, Cetinkaya-Rundel & Barr, *OpenIntro Statistics* | Intro | Gentler alternative. [Free PDF](https://www.openintro.org/book/os/) |

### YouTube Channels
| Channel | Why |
|---------|-----|
| [Metric Academy](https://www.youtube.com/@MetricAcademy) | Lecture videos referenced throughout this course (Armenian + English terms) |
| [StatQuest (Josh Starmer)](https://www.youtube.com/@statquest) | Best bite-sized stats explainers on YouTube |
| [3Blue1Brown](https://www.youtube.com/@3blue1brown) | Unmatched visual intuition |
| [Harvard Stat 110](https://youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) | Full probability course |
| [Seeing Theory (Brown U)](https://seeing-theory.brown.edu/) | Interactive visualizations |

---

*A [Data ARMs](https://www.linkedin.com/company/data-arms) course. YouTube links auto-extracted from the Metric Academy channel. If you spot a mistake or broken link, please let us know.*
