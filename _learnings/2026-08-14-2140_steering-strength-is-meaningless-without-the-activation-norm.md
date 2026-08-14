# Steering strength is meaningless without the activation's norm

**Context.** Building a measured steering figure for `ml/ch19_mech_interp/L47`: take a
difference-of-means direction, add it to GPT-2 small's residual stream at layer 8, sweep the
coefficient, and watch the output probability move.

**Symptom.** The output did not move.

```
steering: P(' wedding') 0.00000 -> 0.00000  as alpha goes 0 -> 8
```

Read at face value, that is evidence that activation steering does not work - a claim that would
have gone onto a slide.

**Cause.** The direction was normalised to unit length, and alpha swept 0 to 8. The residual
stream at that layer has norm **119**. So the largest intervention in the sweep perturbed the
stream by about 6 percent, and most of the sweep by under one percent. The method was never given
a chance to do anything.

Sweeping alpha as a **fraction of the measured stream norm** instead (0 to 1.6x) on the same
direction, same layer, same model:

```
P(' wedding')  0.048%  ->  16.04%   (336x)
top token      " first"  ->  " wedding"
```

**The transferable lesson.** A coefficient on a vector addition has no meaning on its own. It is
only interpretable relative to the magnitude of what it is added to - and in a transformer that
magnitude **grows by two orders of magnitude across the depth** (measured on the same model:
norm 4.5 at the embedding, 482 at the unembedding). So the same alpha is a different intervention
at every layer. Quote steering strengths as a multiple of the activation norm, or they do not
transfer between layers, models, or papers.

**The second, worse mistake: a guard that could not fail.** The script had a check:

```python
if probs[-1] <= probs[0]:
    raise RuntimeError("steering did not increase the target probability")
```

This passed. A rise from `1e-7` to `2e-7` satisfies it perfectly, while the plotted curve is a
flat line at zero. The guard tested the *direction* of the effect and said nothing about whether
it was large enough to be visible - which was the only thing the figure actually claimed. It now
requires both a ratio and an absolute floor:

```python
if peak < 0.002 or peak < 50 * probs[0]:
    raise RuntimeError(f"steering effect too small to plot honestly: {probs[0]:.6f} -> {peak:.6f}")
```

Same family as `_learnings/2026-08-13-1930_a-passing-check-is-only-as-good-as-its-coverage.md`:
write the guard against **the claim the artifact makes**, not against the nearest property that is
easy to compute. "It went up" is easy. "It went up enough to see" is the claim.

**Also worth keeping:** the whole thing was only caught because the figure was measured rather
than drawn. A hand-drawn schematic of steering would have looked completely convincing and taught
the reader nothing about the one detail that decides whether their own attempt works.
