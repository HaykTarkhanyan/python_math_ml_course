# A by-hand frame must add up with the numbers it prints

**Symptom.** LLM-3's "Itself" frame quoted creature's weights as 0.11, 0.11, 0.79 and its update as
(0.21, 0.11, 0.11, 1.57). A student redoing the sum with the printed weights gets
0.11 + 0.11 = **0.22**, not 0.21. Same in "what gets handed over": 0.06 x 2 = 0.12, while the slide
says 0.13. And `block_figs.py` failed its own check on the neuron:

```
AssertionError: neuron fluffy blue creature: computed [0.33], slides quote [0.34]
```

because 0.47 + 1.065 - 1.2 is 0.33499999... in floating point, which rounds to 0.33 at two
decimals, while the slide prints 0.335.

**Cause.** The script asserted the *exact* computation against the rounded results, but the slide
shows a different computation: the one with the rounded inputs. The exact weights were
0.1065, 0.1065, 0.787 and 0.4683, 0.4683, 0.0634.

**Consequences.**

- Print the inputs of a by-hand sum with enough decimals that the printed arithmetic gives the
  printed result (here 3 decimals: 0.107 + 0.107 = 0.214 -> 0.21; 0.063 x 2 = 0.126 -> 0.13), and
  assert *that* arithmetic in the script (`raw_weights3`, `learned_weights3` in
  `attention_figs.py`).
- Compare at the precision the slide prints (3 decimals for 0.335), not a fixed 2.
- Same lesson as LLM-6's top-p table (`decoding_figs.py` asserts the cumulative sums of the printed
  probabilities); now in three scripts, so it is a convention candidate.
