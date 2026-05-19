# Near-Term Teaching Plan

## Schedule Overview (~1.5 sessions/week, done early June)

| # | Date | Session | Type | Slides/Materials | Status |
|---|------|---------|------|------------------|--------|
| 1 | Wed Apr 8 | CI & Bootstrap | Practical | `24_stat_confidence_intervals.qmd` | [x] |
| 2 | Thu Apr 9 | Hypothesis Testing (p-values, power) | Lecture | `09_stat.tex` | [x] |
| 3 | Sat May 2 | Hypothesis Testing | Practical | `25_stat_hypothesis_testing.qmd` | [x] [video](https://youtu.be/deBUJ8pMomU) |
| 4 | TBD | Classical Tests + ANOVA & A/B | Lecture | `10_stat.tex` (merged L10+L11) | [ ] |
| 5 | TBD | Classical Tests + ANOVA | Practical | `26` + `27` qmd | [ ] |
| 6 | Sun May 17 | How to Lie with Statistics | Lecture | `12_stat.tex` | [x] [video](https://youtu.be/1KB7Riu_p_o) |
| 7 | TBD | Q&A / Review | - | - | [ ] |
| 8 | TBD | Entropy, Cross-Entropy, KL | Lecture | `info_01.tex` | [ ] |
| 9 | TBD | Info Theory for ML | Lecture | `info_02.tex` | [ ] |
| 10 | TBD | Information Theory | Practical | TBD | [ ] |
| 11 | TBD | Curse of Dimensionality | Lecture | `math/misc/cod/cod.tex` | [ ] |

---

## Details

## This Week

### Wed Apr 8 - Practical: Confidence Intervals & Bootstrap
- Exercises: `24_stat_confidence_intervals.qmd` (5 problems, ready)
- Solutions: `solutions/hw_07_08_solutions.pdf` + `.ipynb`
- Covers lectures 07, 08

### Thu Apr 9 - Lecture: Hypothesis Testing + Classical Tests (long session)
- Slides: `09_stat.tex` (38 frames) + `10_stat.tex` (30 frames)
- Part 1: H0/H1, test statistics, p-values, Type I/II errors, power analysis, permutation tests, multiple testing
- Part 2: LRT framework, t-tests (one-sample, paired, Welch), chi-squared tests, nonparametric tests

## Next Week

### Tue Apr 14 - Practical: Hypothesis Testing
- Need to create: `25_stat_hypothesis_testing.qmd`
- Topics to cover: z-test, t-test, permutation test, multiple testing simulation, power analysis
- Solutions PDF + notebook needed

## After That

### Classical Tests + ANOVA & A/B (lecture 10, merged)
- `10_stat.tex` (59 frames; L10 + former L11 merged with cuts: dropped Why $-2\log\Lambda$, LRT worked example, Two-Sample Teaching example, Two-Way ANOVA, Sequential Testing)
- Topics: LRT framework, t-tests (one-sample, paired, Welch), proportions z-test, $\chi^2$ (GoF, independence), nonparametric (Mann--Whitney, Wilcoxon), one-way ANOVA, post-hoc (Tukey, Bonferroni), $\eta^2$, A/B testing, pitfalls (peeking, multiple metrics, novelty, Simpson's, interference)

### Practical: ANOVA & A/B Testing
- Exercises TBD
- Topics to cover: one-way ANOVA, A/B test with z-test, sample size planning, peeking simulation

### How to Lie with Statistics (final stat lecture)
- `12_stat.tex` (renamed from former `16_stat.tex`)
- Wrapping up the stat block
- Topics: misleading graphs, cherry-picking, survivorship bias, p-hacking, Simpson's paradox revisited, base rate fallacy, media literacy

### Q&A / Review Session
- Open questions, recap of stat block,

### Deferred to `ml/deferred_lectures/`
- `deferred_regression_inference.{tex,pdf}` (was `12_stat`, 32 frames) — OLS regression inference, Gauss-Markov, diagnostics, logistic
- `deferred_glms.{tex,pdf}` (was `13_stat`, 29 frames) — exp family, link functions, Poisson regression, IRLS, deviance
- `deferred_causal_inference.{tex,pdf}` (was `14_stat`, 37 frames) — RCTs, potential outcomes, DAGs, propensity scores, IV
- Note: delta method (from L8) was meant to ride along with causal — now untaught. Consider weaving into Q&A session or info theory lectures.

---

## Information Theory Block (after stat)

### Lecture 1: Entropy, Cross-Entropy, KL Divergence
- `info_01.tex` (27 frames)
- Topics: surprisal, Shannon entropy, source coding, cross-entropy, KL divergence, information inequality

### Lecture 2: Information Theory for ML
- `info_02.tex` (28 frames)
- Topics: KL = MLE, forward vs reverse KL, maximum entropy principle, mutual information, MI vs correlation, info gain in decision trees

### Practical: Information Theory
- Exercises TBD
- Topics to cover: entropy computation, KL divergence, MI estimation, cross-entropy loss connection

---

## Bridge to ML

### Curse of Dimensionality
- Slides: `math/misc/cod/cod.tex` (needs iteration)
- Topics: volume concentration, distance concentration, empty space phenomenon, implications for ML

---

## Prep TODO
- [x] Create `25_stat_hypothesis_testing.qmd` exercises
- [x] Create hypothesis testing solutions PDF + notebook
- [x] Review `09_stat.tex` and `10_stat.tex` before Thu lecture
