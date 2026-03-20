# Semester Project: "Your Data, Your Story"

---

## Overview

Track 2 things from your daily life for the entire semester. Analyze them using the tools you learn. Present your findings.

This is the single most important assignment in the course — it turns abstract theory into something personal and real.

---

## Step 1: Choose Your Variables (Week 1)

Pick 2 things you can measure daily (or regularly). They should be:
- **Measurable** — a number, not a feeling (unless you use a 1-10 scale)
- **Variable** — not the same every day (tracking something constant is boring)
- **Accessible** — you can actually collect this data without heroic effort

### Good examples
| Variable 1 | Variable 2 | Potential question |
|-------------|-------------|-------------------|
| Bus arrival delay (minutes) | Weather temperature | Are buses later when it's cold? |
| Hours of sleep | Number of messages sent | Do you text more when tired? |
| Daily coffee count | Focus time (hours) | Does coffee actually help? |
| Time to fall asleep (min) | Screen time before bed | Does screen time delay sleep? |
| Steps walked | Mood (1-10 scale) | Does walking improve mood? |
| Commute time (min) | Day of the week | Is Monday commute really worse? |

### How to collect
- A simple spreadsheet (Google Sheets, Excel) with columns: Date, Variable 1, Variable 2
- Record at a consistent time each day
- Don't skip days if you can help it — missing data is real but annoying
- Be honest — don't round or "correct" your data

**Submit your chosen variables by end of Week 1** (just a one-line description, no analysis yet).

---

## Step 2: Checkpoint 1 — Explore (Week 6)

By now you should have ~5 weeks of data (~35 data points).

**Deliverable:** A short writeup (1 page max) with:
1. **Descriptive statistics** for each variable: mean, median, SD, min, max
2. **A histogram** for each variable — draw it by hand or use any tool
3. **Your guess:** What distribution does each variable follow? Why?
4. **One surprise:** Was anything unexpected in the data so far?

This is not graded harshly — it's a sanity check that you're collecting data and thinking about it.

---

## Step 3: Checkpoint 2 — Confidence Intervals (Week 9)

**Deliverable:** Add to your writeup:
1. **95% confidence interval** for the mean of each variable
2. **Interpretation:** What does the interval tell you? Is it wide or narrow? Why?
3. **Sample size reflection:** If you wanted a CI half as wide, how many more days would you need to collect?

---

## Step 4: Final Analysis (Week 12)

**Deliverable:** A full report (3-5 pages, or equivalent slides) covering:

### Required analyses
1. **Describe each variable:** distribution, summary stats, any outliers
2. **Confidence interval** for the mean of each (updated with full dataset)
3. **Hypothesis test:** Formulate and test one hypothesis. Examples:
   - "My bus is late more than 3 minutes on average" (one-sample t-test)
   - "I sleep less on weekdays than weekends" (two-sample t-test)
   - "My commute time differs by day of the week" (ANOVA)
4. **Correlation:** Is there a relationship between your two variables? Compute and interpret the correlation. Is it likely causal or confounded?
5. **One honest limitation:** What could be wrong with your data or analysis?

### Grading rubric

| Criterion | Points | Notes |
|-----------|--------|-------|
| Data quality | 15 | Consistent collection, few missing days, honest recording |
| Descriptive analysis | 20 | Summary stats, histograms, distribution identification |
| Confidence intervals | 15 | Correct computation, correct interpretation |
| Hypothesis test | 20 | Appropriate test, correct execution, honest interpretation |
| Correlation analysis | 15 | Computation + thoughtful causal discussion |
| Presentation clarity | 15 | Clear writing/slides, good visuals, honest about limitations |
| **Total** | **100** | |

---

## Step 5: Peer Review (before final submission)

Before submitting your final report:
1. **Swap reports** with a classmate
2. **Review their work:** Is the hypothesis test appropriate? Is the CI interpretation correct? Did they miss anything interesting?
3. **Write 3 comments:** one strength, one suggestion, one question
4. **Incorporate feedback** into your final version

This is worth participation points and makes everyone's work better.

---

## Step 6: Presentation (Week 12)

5-minute presentation to the class. No need for polish — focus on:
- What did you track and why?
- What did you find?
- What surprised you?
- What would you do differently?

**Audience questions:** 2-3 minutes of Q&A. Classmates should ask about methodology, alternative explanations, or what else could be analyzed.

---

## Tips

- **Start collecting data in Week 1.** Every week you delay costs you data points.
- **More data = narrower CIs.** 12 weeks of daily data gives you ~84 points. That's solid.
- **Don't fake data.** We can tell (and it defeats the purpose). Real data is messy — that's the point.
- **Simple is fine.** A well-analyzed bus delay dataset beats a poorly analyzed complex one.
- **The limitation section matters.** Acknowledging what could go wrong shows statistical maturity.

---

*This project is intentionally low-tech. You don't need Python or R (those come in a later course). Pen, paper, a calculator, and clear thinking are enough.*
