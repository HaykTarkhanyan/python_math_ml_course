# Labels from `np.array([ints])` are int32 on Windows - and an augmentation hid the crash

**Symptom.** Training the gesture CNN *without* augmentation crashed at the first step:

```
RuntimeError: expected scalar type Long but found Int
```

With augmentation, the same labels trained without complaint.

**Cause.** In the `ma` venv (numpy 1.26.4, Windows) `np.array([CLASSES.index(c) for ...])` is
`int32`, and `torch.from_numpy` keeps it; `F.cross_entropy` wants int64. The augmentation did
`torch.where(flip, FLIP[yb], yb)` with an int64 `FLIP` table, which silently promoted the labels
to int64 - so every training run that went through `augment()` worked by accident.

```
y dtype in the notebook: int32
```

**Fix.** Create labels as `np.array(..., dtype=np.int64)` at the source.

**Consequences.**

- A student who removes the augmentation (a natural experiment) would have hit the crash with no
  idea why. Code that works only because of an incidental dtype promotion is a latent bug.
- When a label array crosses from numpy to torch, state its dtype explicitly.
