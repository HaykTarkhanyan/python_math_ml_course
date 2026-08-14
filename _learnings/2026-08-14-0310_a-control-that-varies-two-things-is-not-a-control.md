# A control that varies two things measures nothing

**Context.** Reproducing the subliminal-learning MNIST experiment for `ml/ch20`. The claim
under test: a student learns the teacher's task from meaningless auxiliary outputs **only
if it shares the teacher's initialisation**. So the experiment needs a treatment/control
pair differing in exactly one thing.

**Symptom.** The first run looked like a clean result:

```
shared init    / mnist inputs / mse  ->  10.0% -> 20.4%   (treatment)
different init / noise inputs / mse  ->  11.6% -> 11.2%   (control)
```

Treatment moves, control does not. Exactly the predicted pattern - and it proves nothing.

**Cause.** The two rows differ in **two** variables: the initialisation *and* the
distillation inputs. The third row of the same run showed why that is fatal:

```
shared init    / noise inputs / mse  ->  10.0% ->  8.8%
```

Nothing happens on noise **even with a shared initialisation**. So the "control" was run
in a condition where the effect cannot appear for an unrelated reason. It was not
isolating the initialisation; it was measuring the input distribution. Re-run matched:

```
shared init    / mnist / mse  ->  10.0% -> 20.4%
different init / mnist / mse  ->  11.6% -> 13.7%   <- the real control
```

Still the right conclusion, but now it is actually evidence for it.

**Why it slipped through.** The confound was invisible *because the result came out the
way the theory predicted*. A control that fails to move is exactly what you hope to see,
so there is no moment of surprise to trigger a second look. The wrong version and the
right version produce the same headline sentence.

**The transferable lesson.** Read a treatment/control pair as a diff, and count the
variables that changed. If more than one changed, the comparison is uninterpretable no
matter how good the numbers look. Confirmation is when confounds hide best - a
disappointing result gets audited, a confirming one gets written up.

**Cheap tell.** Before plotting, print the full run configuration next to each number
(`label / inputs / loss`). The confound here was visible the instant the four runs were
listed as a table with their settings; it was invisible while they were two variables in
a figure.
