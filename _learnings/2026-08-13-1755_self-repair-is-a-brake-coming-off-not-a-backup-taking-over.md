# Self-repair in GPT-2-small is mostly a brake coming off, not a backup taking over

**Context.** Building `ml/ch19_mech_interp`. The chapter plan's L46 section 6 was written from the
literature's summary of the Hydra effect: *ablate the name-mover heads and performance barely
drops, because backup name-mover heads take over.* The plan's own build-risk gate required running
the experiment before writing the deck. Running it changed the section twice.

**Symptom.** The measured result is not "barely drops". Ablating the three heads that provably do
the task makes the model **better**:

```
baseline logit diff : +3.580
ablated  logit diff : +3.754
DLA predicted drop  : +4.438
actual drop         : -0.175
```

(`ml/ch19_mech_interp/py_src/self_repair.py`, 128 IOI prompts, GPT-2-small, mean-ablation of
`L9H9`, `L9H6`, `L10H0`. Raw numbers in `results/self_repair.json`.)

**Cause.** "Backups take over" describes less than half of the `+4.612` of compensation. Breaking
the per-head DLA change down by source:

| Source | Contribution | What happened |
|---|---|---|
| `L10H7` stops suppressing | **+2.064** (45%) | a *negative* name mover, `-2.049` -> `+0.015` |
| genuine backups ramp up | **+1.851** (40%) | `L10H10 +0.730`, `L10H2 +0.588`, `L10H6 +0.334`, `L10H1 +0.199` |

`L10H7` is a copy-suppression head: its job is to suppress whatever the name movers boost. Remove
the name movers and there is nothing left to suppress, so it goes quiet. The largest single term
in "self-repair" is therefore not repair at all - it is a brake being released at the same moment
the engine is removed. Two mechanisms with opposite characters, summarised in the literature under
one friendly name.

**Consequences.**

1. A deck teaching only "backups took over" would be teaching a measured result inaccurately,
   while showing a chart generated from the correct numbers. The chart would not catch it.
2. The pedagogical framing improves. "Performance barely drops" is a caveat; "you delete the three
   heads that do the job and the model gets *better*" is a predict-first frame that the room will
   remember, and the two-mechanism breakdown is a better lesson about interpretability than the
   one-mechanism version.
3. Ablation understates importance whenever a backup **or a suppressor** exists - the suppressor
   half is the one that gets left out of the summary.

**The transferable process point.** The plan required three experiments to run before any `.tex`
was written, on the grounds that the outline asserted results nobody had checked at this scale.
Two of the three came back different from the assumption - this one, and the L46 cold open, where
the hunt found a decoy head (`L9H8`: 0.318 attention to the answer, DLA `-0.001`) sitting in the
*same layer* as the two real name movers, which is a sharper frame than the fallback the plan had
budgeted for. Neither would have been caught by reading the deck afterwards, because both were
already written down as facts.

**Disproved along the way.** The initial worry was that the Hydra effect might not reproduce at
124M parameters and that section 6 would need a different spine. It reproduces easily. The risk
was real but pointed the wrong way: the danger was not absence of the effect, it was a
too-simple description of it.

**Validation that the setup was right.** The experiment independently recovered the canonical
Wang et al. (2022) head assignments without being given them - name movers `L9H9`, `L9H6`,
`L10H0`; negative name movers `L10H7`, `L11H10`. When a from-scratch reimplementation lands on the
published head numbers, the metric, the LayerNorm handling in the attribution and the dataset
construction are all almost certainly correct.
