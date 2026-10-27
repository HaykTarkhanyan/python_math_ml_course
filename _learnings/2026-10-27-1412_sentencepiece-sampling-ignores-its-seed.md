# SentencePiece's sampled splits ignore `set_random_generator_seed`

**Symptom.** LLM-1's subword-regularization figure (`fig_unigram_samples` in
`ml/14_llms/py_src/tokenization_figs.py`) drew five sampled splits of "tokenization" with
`spm.set_random_generator_seed(509)` and `sp.sample_encode_as_pieces(word, -1, 0.1)`. Every rerun
drew a different figure. The only assert was "at least 4 distinct splits", so the script passed
each time and the drift went unnoticed until two contact sheets showed different samples.

**Test (sentencepiece 0.2.2, T5's `spiece.model`, 2026-10-27).** Each order of calls, run twice
in fresh processes:

```
seed-then-load ['▁ t ok en ization', '▁token i z a t i on', '▁token i z ation']
load-then-seed ['▁ to ke n izat i o n', '▁ to ke n i z a tion', '▁ t o k en izat i o n']
seed-then-load ['▁ t o k en i z ation', '▁ to ke n i za t i on', '▁to ke n i z ation']
load-then-seed ['▁ t ok e niz at i on', '▁ t o ken ization', '▁ t o ken i za tion']
```

Neither order gives the same output twice. The seed apparently reaches the trainer, not the
processor's sampling.

**Fix.** Sample the same distribution yourself: enumerate every split of `▁tokenization` into
vocabulary pieces (267 of them for T5), weight each by `exp(alpha * sum of piece log-probs)`
(that is P(split)^alpha, what SentencePiece samples from), and draw with
`np.random.default_rng(509).choice`. Two reruns now give identical samples. Brute-force
enumeration is fine for one word; for long text use SentencePiece and accept that it is random.

**Consequences.**
- A figure that samples must be checked for reproducibility by running the script twice and
  comparing outputs. An assert on the *shape* of the output ("at least 4 distinct") passes on
  every random draw.
- SentencePiece *training* is deterministic with `num_threads=1` (the surname Unigram model gave
  the same 1,423 -> 120 pruning sizes in all 6 logged runs); only the sampling is not.
