# A smoke test proves a notebook runs, not that its model learns - use a known-answer set

**Symptom.** The gesture practical (`ml/12_cnn/gesture_snake`) was smoke-tested on a fake dataset
built from real photos (HaGRID thumbs up/down/nothing, left/right made by rotating thumbs-up). Every
cell ran; the small CNN scored 38.8% and the ResNet-18 probe 91.6%. That read as "the fake data is
hard". It was not the data.

A second fake set was built so the answer is **obvious by construction**: a disc with a bar
pointing in the labelled direction, random colours, sizes and positions. Same notebook:

```
ResNet-18 + logistic regression: 99.6% | small CNN: 21.2%          (5 classes: chance is 20%)
train loss per epoch: 1.679, 1.63, 1.605, 1.588, ... 1.579          (ln 5 = 1.609)
```

**Cause - established by elimination.**

| Check | Result | Rules out |
|---|---|---|
| overfit 64 crops, no augmentation, 150 steps | loss 1.695 -> 0.004, 100% | a bug in model, loop or eval |
| batch 64/32/16, lr 2e-3/3e-3, one-cycle, up to 384 steps | held-out 20-25% every time | "too few steps" (my first diagnosis - wrong) |
| no augmentation, 16 epochs, GAP head | train 0.39 / held-out 0.25 | memorisation - it cannot even fit |
| same, flattened 8x8 head | train 0.94 / held-out 0.77 | - |
| full notebook, with augmentation, 20 epochs | flatten 77.2% vs GAP 24.6% | - |

The global average pool (L17's default) averages away *where* a feature fired, and the answer
here is where the thumb is relative to the fist; the small net's last features see only ~36 px.
Decision recorded as DECISIONS #57.

**Consequences.**

- For any practical that trains a model, build a known-answer synthetic set whose labels are
  obvious by construction, and run a strong baseline (frozen pretrained features + a linear
  model) next to your model. Baseline aces it, model does not -> the model is the problem.
- A fake set made from real-but-mislabelled photos cannot separate "hard data" from "broken
  model". It is still useful for timing and for catching crashes, nothing more.
- Run the overfit-a-tiny-batch check before any hyperparameter search: it split "cannot learn"
  from "learns slowly" in one minute, and saved a sweep aimed at the wrong cause.
