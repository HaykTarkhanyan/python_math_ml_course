# Measure a real model's heads before deciding what story they tell

**Symptom.** For LLM-4's "real heads" frame, the first version of `py_src/block_figs.py` picked the
lowest- and highest-entropy heads of one GPT-2 layer, to show "one head concentrates, one spreads".
The lowest head scored entropy **0.0** - which would have been presented as "a focused head".

**Cause.** Entropy says *how much* a head concentrates, not *on what*. Measuring a few
descriptive numbers for every head (from the same saved attention weights, no new model run)
showed what the extremes actually were, on "a fluffy blue creature roamed the verdant forest":

```
top previous-token heads (layer, head, mean weight on i-1): (5, 11, 1.0), (3, 2, 0.6), (4, 7, 0.49)
creature -> fluffy, top head: layer 6 head 2, 0.62
mean first-token share per layer: [0.22, 0.42, 0.43, 0.59, 0.63, 0.81, 0.8, 0.86, 0.82, 0.88, 0.88, 0.74]
heads with first-token share > 0.9: 39 of 144
```

The entropy-0 head is a perfect **previous-token head**, and most of GPT-2's attention on this
sentence goes to the first token - **attention sinks** (Xiao et al., ICLR 2024). A
"concentrated vs spread" story would have been wrong twice: the concentrated head means something
specific, and many "spread" heads are mostly parked on token 0.

**Consequences.**

- Before choosing heads to show, compute per head: weight on the previous token, share on the
  first token, and the weight between the token pairs the lecture cares about. Choose from those,
  then assert the quoted values in the script (`GPT2_QUOTED` in `block_figs.py`).
- Say what a head does on *this* sentence; one sentence does not establish a head's job.
- Save every head's attention to `results/` once (`--gpt2`), so re-choosing costs no model run.
