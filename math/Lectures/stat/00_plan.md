# Near-Term Teaching Plan

## Schedule Overview (~1.5 sessions/week, done early June)

| # | Date | Session | Type | Slides/Materials | Status |
|---|------|---------|------|------------------|--------|
| 1 | Wed Apr 8 | CI & Bootstrap | Practical | `24_stat_confidence_intervals.qmd` | [x] |
| 2 | Thu Apr 9 | Hypothesis Testing (p-values, power) | Lecture | `09_stat.tex` | [x] |
| 3 | TBD | Classical Tests & LRT | Lecture | `10_stat.tex` | [ ] |
| 4 | Tue Apr 14 | Hypothesis Testing | Practical | `25_stat_hypothesis_testing.qmd` (TBD) | [ ] |
| 5 | TBD | ANOVA & A/B Testing | Lecture | `11_stat.tex` | [ ] |
| 6 | TBD | ANOVA & A/B Testing | Practical | TBD | [ ] |
| 7 | TBD | Causal Inference + delta method | Lecture | `14_stat.tex` | [ ] |
| 8 | TBD | Causal Inference | Practical | TBD | [ ] |
| 9 | TBD | How to Lie with Statistics | Lecture | `16_stat.tex` | [ ] |
| 10 | TBD | Q&A / Review | - | - | [ ] |
| 11 | TBD | Entropy, Cross-Entropy, KL | Lecture | `info_01.tex` | [ ] |
| 12 | TBD | Info Theory for ML | Lecture | `info_02.tex` | [ ] |
| 13 | TBD | Information Theory | Practical | TBD | [ ] |
| 14 | TBD | Curse of Dimensionality | Lecture | `math/misc/cod/cod.tex` | [ ] |

---

## Details

## This Week

### Wed Apr 8 - Practical: Confidence Intervals & Bootstrap
- Exercises: `24_stat_confidence_intervals.qmd` (5 problems, ready)
- Solutions: `hw_07_08_solutions.pdf` + `.ipynb`
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

### ANOVA & A/B Testing (lecture 11)
- `11_stat.tex` (34 frames)
- Topics: one-way ANOVA, post-hoc tests, A/B testing, peeking problem, sequential testing

### Practical: ANOVA & A/B Testing
- Exercises TBD
- Topics to cover: one-way ANOVA, A/B test with z-test, sample size planning, peeking simulation

### Causal Inference + Leftovers (lecture 14)
- `14_stat.tex` (37 frames)
- Topics: DAGs, potential outcomes, propensity scores, IV
- Also cover: delta method (from lecture 08, not yet taught)

### Practical: Causal Inference
- Exercises TBD
- Topics to cover: DAG reasoning, propensity scores, collider simulation, IV assumptions

### How to Lie with Statistics (final stat lecture)
- `16_stat.tex`
- Wrapping up the stat block
- Topics: misleading graphs, cherry-picking, survivorship bias, p-hacking, Simpson's paradox revisited, base rate fallacy, media literacy

### Q&A / Review Session
- Open questions, recap of stat block,

### Regression & GLMs - skipping for now
- `12_stat.tex` + `13_stat.tex` - deferred

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
- [ ] Create `25_stat_hypothesis_testing.qmd` exercises (before Apr 14)
- [ ] Create hypothesis testing solutions PDF + notebook
- [ ] Review `09_stat.tex` and `10_stat.tex` before Thu lecture
