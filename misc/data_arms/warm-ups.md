# Weekly Warm-Ups & In-Class Activities

> 5-10 minutes at the start of each lecture. The goal is to get students thinking before the theory starts.

---

## Week 1 — The Monty Hall Problem

You're on a game show. Three doors: one has a car, two have goats. You pick door 1. The host (who knows what's behind the doors) opens door 3 — it's a goat. Should you switch to door 2?

**How to run it:** Vote first (switch vs stay). Then play 20 rounds as a class with cards. Tally results. Then explain why switching wins 2/3 of the time.

**Why here:** Forces students to confront the fact that probability is not always intuitive. Sets the tone for the whole course.

---

## Week 2 — The Birthday Problem

How many people do you need in a room before there's a >50% chance two share a birthday?

**How to run it:** Ask students to guess (most say ~180). Then calculate together: P(no match among k people) = 365/365 * 364/365 * ... Reaches 50% at just 23 people.

**Bonus:** Check if anyone in the class shares a birthday.

**Why here:** Beautiful example of how quickly probabilities compound. Connects to PMF/CDF — you're computing a CDF by hand.

---

## Week 3 — The St. Petersburg Paradox

A casino offers a game: flip a coin repeatedly. If the first heads appears on flip n, you win 2^n dollars. How much would you pay to play?

**How to run it:** Calculate E[X] = 1 + 1 + 1 + ... = infinity. Then ask: would you really pay $1,000,000 to play? Why not?

**Discussion:** Expected value isn't everything. Leads to variance, risk, and (optionally) utility theory. Daniel Bernoulli proposed this in 1738.

**Why here:** Perfect motivation for why variance matters, not just the mean.

---

## Week 4 — Guess the Distribution

Show 4-5 real datasets (just histograms, no labels). Students vote on which distribution each follows.

**Suggested datasets:**
1. Daily number of emails received → Poisson
2. Heights of adult men → Normal
3. Time between earthquakes → Exponential
4. Number of heads in 100 coin flips → Binomial (≈ Normal)
5. Dice rolls → Uniform (discrete)

**How to run it:** Project histograms one at a time. Students discuss in pairs, then vote. Reveal the answer and the real-world context.

**Why here:** Builds intuition for "which distribution fits this data?" — the core skill of Week 4.

---

## Week 5 — Spurious Correlations Game

Show a series of correlation plots. Students must decide: real causal relationship, or spurious?

**Examples:**
- Ice cream sales vs drowning deaths (confounded by summer)
- Organic food sales vs autism diagnoses (both trending up over time)
- Nicolas Cage films vs pool drownings (from tylervigen.com)
- Smoking vs lung cancer (real!)
- Shoe size vs reading ability in children (confounded by age)

**How to run it:** Project each plot. Students vote: causal / confounded / coincidence. Discuss each.

**Why here:** Correlation ≠ causation is Week 5's theme. This makes it visceral.

---

## Week 6 — Bad Chart Roast

Students (or instructor) bring a misleading graph from the news, social media, or a paper. The class dissects what's wrong.

**Common tricks to look for:**
- Truncated y-axis (making small changes look huge)
- Cherry-picked time range
- 3D pie charts (distort proportions)
- Dual y-axes that mislead
- Missing context (percentages without base rates)

**How to run it:** Project 3-4 charts. For each: what claim is being made? What's misleading? How would you fix it?

**Ongoing:** This becomes a recurring activity — students can submit "bad charts" they find for bonus participation points.

**Why here:** Descriptive stats week. The message: always plot your data, but also always question other people's plots.

---

## Week 7 — Simulate It

**The exercise:** "I claim that the sample median is a biased estimator of the population mean for skewed distributions. Prove me wrong — or right — using simulation."

Give students 10 minutes (on paper or mentally, since no Python yet):
1. Imagine drawing 1000 samples of size 30 from an Exponential(1) distribution
2. For each sample, compute the mean and the median
3. What would the average of those 1000 means be? (≈ 1, the true mean)
4. What would the average of those 1000 medians be? (≈ 0.69, which is ln(2), not 1 — biased!)

**Discussion:** This is Monte Carlo simulation in a nutshell. "Don't know the theory? Simulate 10,000 times."

**Why here:** Introduces simulation thinking alongside estimation theory. Even without code, the logic matters.

---

## Week 8 — The Surgeon Puzzle

A surgeon has performed 3 operations. All 3 were successful. What's the probability her next operation succeeds?

- **MLE answer:** 3/3 = 100%. Clearly wrong.
- **Your gut:** Maybe 85-90%? Based on... what?
- **Bayesian answer:** Combine prior knowledge ("surgeons are typically ~85% successful") with the data (3/3). Get something like 90-95%.

**How to run it:** Walk through the logic without formulas. The point is: prior knowledge + data = better estimate.

**Why here:** Plants the Bayesian seed. Students see why pure MLE can be silly with small samples.

---

## Week 9 — CI or Not?

Present a series of statements. Students decide: is this a correct interpretation of a 95% confidence interval?

1. "There's a 95% probability the true mean is in this interval" → WRONG (common misconception)
2. "If we repeated this experiment 100 times, about 95 of the intervals would contain the true mean" → CORRECT
3. "95% of the data falls in this interval" → WRONG (that's a prediction interval)
4. "We're 95% confident in our methodology" → CLOSEST to correct
5. "The interval is narrow, so our estimate is precise" → CORRECT (but doesn't mean accurate)

**Why here:** CIs are the most misunderstood concept in statistics. Nailing the interpretation matters more than the formula.

---

## Week 10 — The XKCD Jelly Bean Problem

Reference: [XKCD #882 "Significant"](https://xkcd.com/882/)

Scientists test whether jelly beans cause acne. They test 20 colors. At alpha = 0.05, one color shows a "significant" result. The headline: "GREEN JELLY BEANS LINKED TO ACNE!"

**How to run it:** Calculate: P(at least one false positive in 20 tests) = 1 - 0.95^20 = 64%. Then discuss: How should they have corrected for this? (Bonferroni, FDR.)

**Why here:** Perfect setup for p-values, multiple testing, and the replication crisis.

---

## Week 11 — Design an A/B Test

**Scenario:** You work at a streaming company. The product team wants to test a new recommendation algorithm. They claim it increases watch time by 5%.

Students must answer:
- What's H0 and H1?
- What test would you use?
- How many users do you need? (power analysis)
- How long should the test run?
- What could go wrong? (peeking, novelty effect, network effects)

**How to run it:** Small groups (3-4), 10 minutes, then present their designs. Class votes on the best one.

**Why here:** Applies everything from Weeks 10-11 to a real scenario.

---

## Week 12 — Correlation Court

A mock trial: the prosecution claims X causes Y based on a strong correlation. The defense must find alternative explanations.

**Case:** "Countries that eat more chocolate win more Nobel Prizes. Therefore chocolate makes you smarter."

**Roles:**
- Prosecution: presents the correlation, p-value, scatterplot
- Defense: proposes confounders (GDP, education spending, research funding)
- Jury: the class votes

**Why here:** Causal thinking is Week 12's theme. Making it adversarial forces students to think critically about both sides.

---

*These are suggestions — adapt based on class energy and time. The warm-up should never eat more than 10 minutes of lecture time.*
