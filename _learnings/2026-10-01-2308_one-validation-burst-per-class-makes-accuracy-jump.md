# With one validation burst per class, validation accuracy jumps between epochs - the last number is luck

**Symptom.** The Gesture Snake small CNN, trained on the instructor's burst 1 and validated on
burst 2 (250 crops: 5 classes x 50 frames), printed "final validation accuracy 27.2%". The
per-epoch curve told a different story:

```
val accuracy per epoch: 0.20 0.47 0.36 0.45 0.40 0.40 0.42 0.55 0.56 0.32
                        0.41 0.46 0.78 0.60 0.74 0.46 0.33 0.63 0.62 0.27
train loss:             2.89 -> 0.03-0.10 from epoch 12 on
```

Best 78.4% at epoch 13, 27.2% at epoch 20, jumps of up to 35 points between neighbouring epochs.

**Cause.** The 50 frames of a burst are a tenth of a second apart - near-copies. 250 validation
crops are really **5 independent examples**, so classes flip largely as a block, and one class
flipping moves the accuracy by up to 20 points. Training loss near zero says the net had memorized its one training
burst per class, so small weight changes swung entire classes between "right" and "nothing".

**Consequences.**

- Report the curve, or the best epoch next to the last one - never only the final epoch. The
  notebook's training cell now prints both.
- Picking the best epoch (early stopping) chooses on the validation set, which then stops being a
  test: an honest score needs a third, untouched burst.
- Count **independent** units, not images. For video-like data the unit is the burst, the
  session or the person - the same reason the split is by burst in the first place.
- The instructor kept the 2-burst data on purpose: "the model fails" is the lesson, and the frozen
  ResNet-18 (64.4%) is the rescue.
