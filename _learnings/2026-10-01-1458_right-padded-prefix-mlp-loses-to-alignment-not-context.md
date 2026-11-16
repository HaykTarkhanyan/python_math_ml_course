# A right-padded whole-prefix MLP loses ~0.15 nats to alignment, not to context length

Context: the ch11 name-inventor practical (`ml/11_neural_networks/name_inventor_hy.md`). A
natural design students reach for is to feed the MLP the **whole prefix**, padded on the **right**
to the longest name: `[<START>, letters so far..., <PAD>, <PAD>, ...]`. It trains, samples, and
looks reasonable, and it scored only 0.07 better than add-1 bigram counts (1.893 vs 1.964).

**Cause.** With right padding the most recent letter lands in a different input slot for every
prefix length, and each slot has its own first-layer weights. The rule that matters most ("after
`յա` comes `ն`", 95.8% of names end in `-յան`) has to be learned separately in each of the 10
slots that names of 5-14 letters put it in, from 551 training names.

**Evidence** (same config `emb=16, [256,128,64], dropout 0.2, tanh`, Adam 0.01, batch 32,
early stopping patience 10, same 551/138 name split, 3 seeds):

```
right-padded whole prefix  : 1.905 / 1.901 / 1.866   mean 1.891
LEFT-padded whole prefix   : 1.717 / 1.755 / 1.744   mean 1.739   <- only change
last 3 letters (K=3 window): 1.752 / 1.767 / 1.743   mean 1.754
bigram counts (add-1)      : 1.964      uniform over 39: 3.664
```

Samples tell the same story (1000 at T=1): in the data `ւ` follows `ո` in 118 of 118 cases. The
right-padded model breaks that in **18.2%** of its names, the K=3 window in 1.2%, and even the
smoothed bigram table only in 6.0%.

**Disproven theory.** The first guess (the submitter's, and mine) was that the fix is a
*shorter window*, which shares what it learns across positions. The control says no: the
left-padded version still reads the whole prefix and matches K=3 within the seed noise (0.038).
What fixes it is **alignment**: the newest letter always sits in the same slot. Bengio's
fixed window gets alignment for free, which is part of why it is the right model here.

**Consequences.**

- When reviewing a ch11 submission, check the input layout first. Right padding explains a loss
  that barely beats the bigram baseline.
- A possible teaching beat for the practical or [41]/[42]: "pad on which side?" is a one-line
  decision worth 0.15 nats, and it shows why position-specific weights need aligned inputs, an
  idea that CNNs and RNNs later solve properly.
- Probe: `_submissions/11_name_inventor/*/evidence/verify_claims.py` (local only, gitignored).
