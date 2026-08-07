# Pedagogy review - `TABULAR_FM_CHAPTER_PLAN.md` (L37)

Reviewed 2026-08-08. Teaching only; ML facts are another reviewer's job.

## 1. The central claim is aimed at the wrong chapter

This is the big one. I read the deck you say you are contesting.

`02_main_concepts/08_hyperparameter_tuning.tex` is 25 frames on **how to search a hyperparameter
space** - manual, grid, random, Bayesian, space design, pruning. It is model-agnostic. It never
claims tuning is mandatory, and it never claims tuning buys much. Line 447 already concedes
*"on a small 2-HP space no method works magic"* (RMSE 55-57 across all three methods). Line 480
already says tuning on 10% of the data lands *"within 1% of fully-tuned."* A model with no knobs
does not undercut that deck - it just falls outside its scope, alongside every LightGBM run above
50k rows and every neural net in ch5-ch13, which the deck still governs. The plan's framing
(lines 36-42) picks a fight that is not there, and a student who has actually watched L01e will
notice.

The real collision is one chapter over and the plan barely mentions it.
`04_trees/20_advanced_boosting.tex` line 928: **"Trees + ensembles dominate tabular data."**
Line 800: *"reach for boosting on tabular data."* Line 101: *"Often the top performer on tabular
benchmarks - if properly tuned."* Those are the sentences this chapter contradicts, and only in
the small-n regime. Quote line 928 verbatim on a slide, then scope it. That is a real argument
with a real citation, and it is stronger than the current one because it is true.

Frame 25 is worse: it defends ch06 against an accusation nobody made. `26_feature_engineering.tex`
line 233 already says *"feature engineering is not dead - it moved,"* and the recap (line 844)
already reports FE at **+81.5 MAE to Ridge and -4.3 to a forest** - it already shows FE *hurting*
a tree ensemble. Frame 25 as written argues with a strawman. Either cut it or flip it into a
callback: "ch06 measured this and got -4.3; here is the same effect at scale."

Retarget the thesis paragraph and frames 25 to ch04, and the chapter's premise holds.

## 2. Wrong order: evidence must come before mechanism here

Yes, 9-13 before 14-18 is wrong, and specifically because of what this deck is. Frame 1 asserts a
result the students have been taught to disbelieve, then you spend **ten frames on mechanism**
before frame 15 substantiates it. That is a long time on "trust me."

Compare L32: AlphaGo cold open, then "Does it actually work?" lands at frame 22 of 44 - halfway.
L33 gets away with mechanism-first because nothing in the course disputes "a model can see."
Here something does.

Move **15 and 16 up to directly after 2**. Frame 15's *"what small data means precisely"* is the
single most important frame in the deck and it currently arrives fourteen frames late. Right now
a student holds an oversized version of the claim for half the lecture, then watches it get
narrowed at 15 and narrowed again at 23. State the scope condition early - it makes the claim
more credible, not less. Section 3 is only 4 content frames, so the swap is cheap.

Also pull **23 (size)** up next to **13** ("the limit is context length, not epochs"). Frame 13
sets up frame 23 exactly, and they are currently ten frames apart.

## 3. Frame 8 is under-built; the arc has three holes

- **Frame 8 is the load-bearing frame of the chapter and gets one slide and no figure.** The plan
  itself calls it *"the frame to slow down on"* and then does not. If a student does not get the
  prior, frames 20, 21, 23, 26 all fail. Give it 2-3 frames and a figure. The figure budget has
  five entries and none of them is for frame 8. That is backwards.
- **No worked-numbers frame.** House style asks for one and this topic hands you the best kind:
  n rows x d features = cells in context, and what that costs at quadratic attention. This is the
  direct analogue of L33 frame 6 ("the arithmetic that governs everything downstream"), which is
  the strongest frame in that deck. Build `context_not_epochs` as a **real plot with real axes**
  rather than a schematic and you get the worked-numbers frame for free.
- **No "how do you use it" frame.** SLIDE_STYLE allows one canonical snippet. `fit`/`predict` in
  three lines. Without it the lecture is unactionable.
- **No misconception pre-empt**, and there is an obvious one: "no training" is not "no cost" -
  the pre-training happened once, expensively, and someone else paid.

Where it drags: **frame 18** ("the idea generalises", six domains as a list) is a name-drop with
no teaching content - fold into 17 or cut. **Frame 16** (v1 to v3 version history) is a
`\modeltransition` card, not a frame. **Frames 24-27** are four consecutive one-paper-each frames;
that reads as a literature review, not the argument you promised. 24 and 25 can merge.

## 4. Section 4: two frames is fine, three frames is the problem

The header says 2 frames; the outline lists 3 (19, 20, 21). Frames 19+20 work as-is - "stated,
not derived" is a legitimate choice and the dispute has a checkable consequence (Nagler explains
why accuracy improves past pre-training sizes). **Make that consequence the frame title**, not a
parenthetical. Without it, "two camps disagree" is trivia and students will correctly skip it.

**Frame 21 does not survive.** Prior-induced bias in causal estimands, with a `math/22`
consistency callback, in one frame, with no setup - students will not follow it. Two options:
cut it, or strip it to the one sentence they can use - *"the prior does not wash out with more
data"* - which lands because frame 8 already told them the prior is the model. Also, the
cross-reference is wrong: frame 20 (Bayesian vs frequentist) is `math/23` MLE-vs-MAP; `math/22`
belongs to 21.

## 5. Zero predict-first frames. Take the obvious one.

L32 and L33 have exactly one each - that is the house dose. This topic has the best candidate in
the course, sitting unused: **at frame 5/6, "a model trained on zero real datasets, against a
tuned LightGBM on 1,000 rows - better or worse?"** Everyone says worse. That converts frame 1
from an assertion into a bet the student has already lost, which is worth more than the table.
Note this only works if the reveal comes early - another reason to do the reorder in section 2.
Backup candidate: frame 12, "permute the test table's columns - does the prediction change?"

## 6. House style: the rest is mostly right

Cold open, outline at 3, `[plain]` transitions at 4/9/14/19/22, recap at 28 - all correct. Fixes:

- Recap sits **inside** Section 5. Move it under `\section*{Wrap-up}` (see
  `20_advanced_boosting.tex`, `26_feature_engineering.tex` line 838).
- The "Next:" box has no target - this is the last deck in the L-sequence. Make it a
  "what to do with this" box pointing back at ch04, not an empty ritual.
- Abbreviations: `SCM`, `BNN`, `TabICL`, `TabDPT`, `LimiX` all need first-use expansion
  (SLIDE_STYLE 84-100). `PFN` is already handled at frame 5.
- Back-pointers (open q2): callout, not a sentence, and **put the scope condition in it**
  ("under ~10k rows"). A bare "see ch14" in ch04 reads as a contradiction of the lecture the
  student just watched.

## 7. No practical: it stands, but write down why, and add one derived number

`vlm.qmd` and `audio.qmd` both currently say `Տնային: TBD`, so "every other recent chapter has
one" is not true of the two most recent. ch11 has a project because RL is a method students
should run; ch12-ch14 are literacy chapters. That is a defensible line - put it in the plan as
the stated reason instead of a bare decision.

The cost is real though: nothing here is falsifiable by the student. Frame 1 is published
numbers, frame 23 is published numbers. The students *measured* tuning in ch04's practical and
*measured* FE at +81.5/-4.3 in ch06. A chapter that contests those with only citations is
epistemically weaker than what it contests, and a sharp student will say so. The fix inside your
constraint is the context-length arithmetic from section 3 - no download, no licence problem, one
number the student derives rather than receives. I would treat that as a build blocker.

Second blocker: **open question 1 is not an open question.** Frame 1 is the thesis. If the Nature
numbers are not read out of the primary paper, the cold open is a schematic, and a schematic cold
open for a contested claim is worth nothing. Resolve before building.

Yes to the Google Form (open q3) - with no HW it is the only assessment surface left.
