# Real-World Case Studies

> Short (5-10 min) stories to open or close a lecture. Each connects the week's theory to something students can relate to.

---

## Week 1 — How a Probability Puzzle Changed Game Shows

The Monty Hall problem became front-page news in 1990 when Marilyn vos Savant published the correct answer in *Parade* magazine. Nearly 1,000 PhD holders wrote in to say she was wrong. She wasn't.

**Connection:** Even experts get probability wrong when they rely on intuition.

**Source:** [The Time Everyone "Corrected" the World's Smartest Woman](https://priceonomics.com/the-time-everyone-corrected-the-worlds-smartest/)

---

## Week 2 — How Netflix Knows What You'll Watch Next

Netflix models your viewing as draws from a probability distribution over genres, actors, time of day, etc. Your "taste profile" is essentially a random variable — and they're estimating its PMF from your watch history.

**Connection:** Random variables aren't abstract — they're the language behind every recommendation engine.

---

## Week 3 — The Gambler's Ruin

A gambler with $100 plays a fair coin-flip game ($1 per round) against a casino with $10,000. Despite the game being "fair" (E[gain] = 0), the gambler goes broke with near-certainty. Why? The variance kills you before the mean saves you.

**Connection:** Expected value alone doesn't tell the whole story. Variance and ruin probability matter.

**Source:** Any probability textbook, or [Seeing Theory simulation](https://seeing-theory.brown.edu/)

---

## Week 4 — The Poisson Distribution Saved London in WWII

During the Blitz, Londoners noticed that some neighborhoods were hit by V-2 rockets repeatedly while others were spared. Conspiracy theories spread: the Germans must have spies targeting specific areas!

Statistician R.D. Clarke divided London into 576 squares and counted hits per square. The distribution was almost perfectly Poisson — the rockets were random. No spies, no targeting. Just Poisson.

**Connection:** The Poisson distribution describes rare, random events in fixed intervals of space or time.

**Source:** Clarke, R.D. (1946). "An Application of the Poisson Distribution."

---

## Week 5 — The Literary Digest Poll Disaster (1936)

The Literary Digest mailed 10 million survey postcards to predict the 1936 U.S. presidential election. They predicted Alf Landon would win in a landslide. Roosevelt won 61% of the vote.

What went wrong? Their mailing list came from telephone directories and car registrations — sampling the wealthy during the Great Depression. Meanwhile, George Gallup used a much smaller but representative sample and nailed the result.

**Connection:** Sample size doesn't matter if your sample is biased. Correlation between wealth and political preference created selection bias.

**Source:** [The 1936 Literary Digest Poll](https://www.math.upenn.edu/~deturck/m170/wk4/lecture/case1.html)

---

## Week 6 — Anscombe's Quartet: Always Plot Your Data

In 1973, statistician Francis Anscombe constructed four datasets with identical means, variances, correlations, and regression lines — but completely different shapes when plotted. One is linear, one is curved, one has an outlier, one is clustered with a single extreme point.

**Connection:** Summary statistics can hide the truth. Visualization is not optional.

**Activity:** Show the four datasets' summary stats first. Ask students: "Are these the same?" Then reveal the plots.

**Source:** Anscombe, F.J. (1973). "Graphs in Statistical Analysis." *The American Statistician*.

---

## Week 7 — Stein's Paradox: Sometimes Being Wrong is Better

You want to estimate the batting averages of 18 baseball players. Intuitively, you'd use each player's observed average. But James and Stein showed in 1961 that "shrinking" all estimates toward the overall mean gives lower total MSE — even though each individual estimate is biased.

**Connection:** Bias isn't always bad. The bias-variance tradeoff in action — a biased estimator can have lower MSE than an unbiased one.

**Source:** Efron, B. & Morris, C. (1977). "Stein's Paradox in Statistics." *Scientific American*.

---

## Week 8 — How Spam Filters Use Bayes' Theorem

Your email spam filter is doing Bayesian estimation hundreds of times per second. Given the words in an email, what's the probability it's spam? It starts with a prior (most emails are not spam), updates with evidence (the word "lottery" makes spam more likely), and outputs a posterior.

**Connection:** MLE says "what parameters make this data most likely?" Bayesian thinking adds: "...given what I already know." Even a lite Bayesian intuition explains why your spam filter works.

---

## Week 9 — The Resampling Revolution

Before computers, getting a confidence interval required knowing the sampling distribution analytically — which was often impossible. In 1979, Bradley Efron introduced the bootstrap: just resample your data with replacement, thousands of times. No formulas needed.

It was controversial. "You can't just make up data!" critics said. But it works — and it democratized statistical inference.

**Connection:** Bootstrap is the ultimate "simulation instead of math" tool. If you can compute a statistic, you can bootstrap a CI for it.

**Source:** Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife."

---

## Week 10 — The Replication Crisis

In 2015, the Open Science Collaboration tried to replicate 100 published psychology experiments. Only 36% replicated successfully. Causes: small sample sizes (low power), p-hacking, publication bias ("only significant results get published"), and misunderstanding of p-values.

**Connection:** This is what happens when hypothesis testing is done poorly. Understanding power, effect size, and what p-values actually mean isn't academic — it determines whether science works.

**Source:** Open Science Collaboration (2015). "Estimating the reproducibility of psychological science." *Science*.

---

## Week 11 — The Peeking Problem at Optimizely

Optimizely (A/B testing platform) found that many customers were "peeking" at results daily and stopping tests as soon as they saw a significant result. This inflated false positive rates from 5% to over 30%.

Their solution: sequential testing methods that account for repeated looks at the data.

**Connection:** A/B testing isn't just "run a t-test." When you stop matters as much as what you test.

**Source:** [Optimizely — Stats Engine whitepaper](https://www.optimizely.com/resources/stats-engine-whitepaper/)

---

## Week 12 — Chocolate and Nobel Prizes

A 2012 paper in the *New England Journal of Medicine* found a strong correlation (r = 0.79) between per-capita chocolate consumption and the number of Nobel Prize winners per country.

Does chocolate make you smarter? Of course not. Both correlate with wealth, education spending, and research funding. Switzerland tops both lists because it's a rich country with great universities (and great chocolate), not because chocolate causes genius.

**Connection:** Correlation ≠ causation. Confounders are everywhere. DAGs help you see the real causal structure.

**Source:** Messerli, F.H. (2012). "Chocolate Consumption, Cognitive Function, and Nobel Laureates." *NEJM*.

---

*Pick one per week. Don't force all of them — use whichever fits the energy of the class.*
