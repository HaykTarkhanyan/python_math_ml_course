# L20 - RNN Foundations - outline (for approval)

Drafted 2026-07-13 under the chapter reframe (road to attention, TEXT spine, heavy
visuals). L20's job: make the variable-length problem hurt (why MLPs really fail on
text), give sequences a vocabulary (recurrence, unrolling, sharing, taxonomy), then
land the chapter's one deep dive - the full BPTT / vanishing-gradient derivation.
Students should leave feeling the PROBLEM, not admiring the architecture.
Subtitle idea: "Networks with memory - and why the memory fades."

**Interview decisions (2026-07-13, structural rounds + voice directions):**
- Chapter = road to attention; RNN taught as lineage, not destination.
- TEXT spine: customer reviews of the exported fruit (sorting-line sequel).
- Variable-length pain EXPANDED: dedicated shoehorn-hacks frame (pad/truncate,
  bag-of-words) - demonstrate, don't assert.
- BPTT: FULL derivation (D-matrices, (V^T)^(t-i), eigenvalue argument) - locked.
- Heavy visuals: every content frame carries a diagram/figure/ANIM.
- NO homework this chapter.
- Bidirectional / stacked RNNs: one-liner mentions only.
- Per-deck interview still pending before build - this outline may be revised then.

Target: ~21 frames, one ~90-min session.

### Cold open (before Outline)

- **Frame - the sequel and the destination.** The ch6 classifier grades the fruit; the
  exports ship worldwide. Now the reviews arrive: "Fresh and sweet!" (3 words),
  "The pomegranates arrived fresh and sweet, best I've had since Yerevan." (12 words),
  a 40-word complaint, some in Armenian. Ops wants them read (sentiment), translated,
  answered. Zoom out: every headline AI system of the 2020s (ChatGPT included) is a
  SEQUENCE model - this chapter and the next are the road there. Predict-first: "You
  know MLPs (L14) and CNNs (ch6). Feed a review into an MLP - what breaks before you
  even train?" `\pause` The input layer has a FIXED size; reviews do not.
  `[predict-first]` `[running example]` `[visual: 3 reviews of visibly different
  lengths stacked next to a fixed-width input layer]`

### Outline frame

### Section 1: Text refuses to fit

- `[plain]` transition: "Order is information, and length is a variable."
- **The sequence zoo.** Text leads; time series (tonnage logs, exchange rates), audio
  as one-line cousins. What they share: order matters, length varies, dependencies can
  be LONG-range. `[diagram: 3-example strip, text highlighted]`
- **The shuffle test.** The review sentence vs the same words shuffled: every word
  identical, all meaning gone - and the bag-of-words histograms are IDENTICAL, so any
  order-blind model literally cannot tell them apart. Callback: L16 did this to an
  image's pixels - same demo, new axis. `[real fig: word_shuffle.pdf]`
  `[callback: L16 pixel shuffle]`
- **The shoehorn hacks.** How people force variable-length text into fixed nets, and
  the price of each: (1) pad + truncate to length 8 - the short review wastes most of
  its input, the long one loses its ending (which held the verdict!); (2) bag-of-words
  averaging - fails the shuffle test by construction; (3) separate weights per
  position - "I went to Munich in 2009" vs "In 2009, I went to Munich": the same fact
  must be relearned at every position. `[real fig: pad_truncate.pdf or TikZ panel]`
  `[armorange watch-out box]`
- **Tokens, minimally.** To feed text to ANY network, words must become vectors. For
  now, the simplest scheme: a vocabulary + one-hot vectors (callback: [03] one-hot
  encoding - same trick, new victim). The full tokenization story (and its problems)
  is L21's. `[diagram: sentence -> token boxes -> one-hot columns]` `[callback: [03]]`

### Section 2: The recurrence idea

- `[plain]` transition: "Read like a human: one word at a time, with memory."
- **One new arrow.** How a human reads: word by word, retaining a running summary -
  the STATE. From L14's z = sigma(W'x + b) to
  z[t] = sigma(V'z[t-1] + W'x[t] + b): a single new term, the hidden-to-hidden
  weights V. The state is a function of the current input AND the previous state -
  by recurrence, a summary of everything read so far. `[callback: L14 hidden layer]`
  `[popblue theory box]` `[diagram: dense net + the one red recurrent arrow added]`
- **Unrolling.** The folded loop view vs the unrolled view: same network copied per
  time-step, passing a message forward. `[ANIM (mandatory): unroll_anim - the review
  sentence fed token by token, the state annotated at each click]`
  `[OWN-TEX candidate: dl4nlp unrolled-RNN TikZ as the static summary frame]`
- **Worked numbers - a 1-d RNN reads three tokens.** Scalar everything: W=0.5, V=0.8,
  b=0, linear activation. Three tokens with 1-d embeddings [2, 1, 3]. Compute on the
  frame: h1 = 1.0, h2 = 0.8*1.0 + 0.5*1 = 1.3, h3 = 0.8*1.3 + 0.5*3 = 2.54. Now
  REVERSE the tokens [3, 1, 2]: h3 = 2.36. Same bag of words, different memory - the
  state encodes ORDER, which is exactly what the shuffle test demanded. "The memory
  is one number here; in practice a vector of 64-512." `[worked-numbers]`
  `[predict-first candidate: ask the reversed case before revealing]`
  `[running example]`
- **Parameter sharing.** The same W and V at EVERY step: parameter count is
  independent of sequence length - the 3-word and the 40-word review use the same 305
  parameters (d_x=1, d_h=16: W 16 + V 256 + b 16 + U 16 + c 1). Callback: convolution
  shares across space, recurrence shares across time - the same trick on a new axis.
  And it kills hack (3): a pattern learned at position 2 works at position 17 for
  free. `[worked-numbers]` `[callback: L16 parameter sharing]` `[armblue key box]`
  `[diagram: unrolled net with the same W,V labels highlighted at every step]`
- **The taxonomy.** seq-to-one (sentiment - our reviews), one-to-seq (image
  captioning), seq-to-seq (translation, language modeling) - vocabulary L21 needs.
  Variants exist (bidirectional, stacked) - one sentence each, no more.
  `[WEB-IMG: Karpathy's taxonomy strip, or TikZ equivalent]`

### Section 3: Training - and the fatal flaw

- `[plain]` transition: "Unroll it, and it is just a deep network."
- **Backprop through time.** The unrolled RNN is a deep feed-forward net with TIED
  weights; autograd needs nothing new - the L15 promise ("one sweep, any
  architecture") kept again. Loss sits at the final step for seq-to-one sentiment.
  `[callback: L15 comp-graph]` `[diagram: unrolled net with gradient arrows flowing
  right-to-left]`
- **Predict-first: the fading number.** Our scalar RNN again: how much does token 1
  influence the state at token 30 (a long review)? dh30/dh1 = V^29 = 0.8^29. Guess.
  `\pause` 0.0015. And if V were 1.25? 646. One multiplication, repeated, either
  starves or explodes. `[predict-first]` `[worked-numbers]`
  `[real fig: vanish_curve.pdf - log scale]`
- **The full derivation, part 1.** For vectors:
  dz[t]/dz[t-1] = diag(sigma'(...)) V^T = D[t-1] V^T, so
  dL/dz[1] = dL/dz[t] * D[t-1] D[t-2] ... D[1] (V^T)^(t-1). Every path to the past
  runs through powers of the SAME matrix. `[math frame - locked full derivation]`
- **The full derivation, part 2 - eigenvalues.** The largest eigenvalue of V decides:
  |lambda_max| < 1 -> vanishing, > 1 -> exploding. Why this is WORSE than a deep MLP:
  an MLP has a different matrix per layer (errors can partially cancel); the RNN
  multiplies by the same V at every step, so the effect compounds cleanly. The
  sigmoid's D factors only make it worse (|sigma'| <= 1/4). `[armred trap box]`
  `[callback: L15 vanishing gradients]`
- **Exploding is the easy half.** Gradient clipping: if ||grad|| > h, rescale to h.
  One formula, solved problem. (Build note: check whether L15 taught clipping; teach
  in place if not.) `[armorange watch-out box]` `[diagram: gradient vector clipped
  to a sphere - small TikZ]`
- **What vanishing means in practice.** "The cat that I saw yesterday in the garden
  near the old oak tree WAS cute" - the gradient from "was" cannot reach "cat"; the
  network cannot learn to connect them. Our version: the 40-word review whose verdict
  is in the last clause - the RNN has forgotten the product by then.
  `[OWN-TEX: dl4nlp vanishing-gradient frame]` `[running example]`
- **Empirical proof.** Gradient norm per time-step for a tiny vanilla RNN on a toy
  recall task: dead within ~15 steps. `[real fig: gradient_flow.pdf, vanilla curve
  only - the LSTM curve is L21's payoff]`

### Recap + Next

- Recap: length varies and order matters - hacks lose one or the other; state =
  running summary; one new weight matrix V; unrolled = deep net with tied weights;
  same-matrix powers -> gradients vanish (derived, not asserted); clipping fixes
  exploding, nothing yet fixes vanishing.
- `[paramgreen Next box]`: "1997's answer: a memory built from addition, not
  multiplication - the LSTM. It works. And it still will not be enough (L21)."

---

## Figures (py_src -> fig, `ma` venv, seed 509)

1. `word_shuffle.py` - review sentence vs shuffled + identical bag-of-words
   histograms side by side.
2. `pad_truncate.py` - the shoehorn panel: 3 reviews (lengths ~3/12/40) padded and
   truncated to a fixed 8, losses highlighted (TikZ acceptable if cleaner).
3. `unroll_anim.py` - ANIM flip-book `unroll_0.pdf .. unroll_N.pdf` (~6 clicks),
   using the worked-numbers values so the animation and the math frame agree.
4. `vanish_curve.py` - 0.8^k and 1.25^k vs k, log scale, both curves labeled.
5. `gradient_flow.py` - gradient norm vs time-step on a toy recall task; emits BOTH
   vanilla and LSTM curves; L20 uses the vanilla-only PDF variant.

## TikZ (small)

Folded-loop diagram (the single recurrent arrow); one-hot column diagram; clipping
sphere; taxonomy strip if not WEB-IMG.

## Build notes (for the implementing model)

- **Respect the reframe:** the derivation frames are the ONLY deep-math frames in the
  chapter. Everything before them stays light and visual.
- **Heavy-visuals rule:** every content frame carries a visual (see plan); the bullets
  in this outline name one per frame - do not drop them to save build time.
- **Worked numbers are locked** - W=0.5, V=0.8, b=0, forward [2,1,3] ->
  h=[1.0, 1.3, 2.54]; reversed [3,1,2] -> h=[1.5, 1.7, 2.36]; 0.8^29 ~ 0.0015,
  1.25^29 ~ 646. Verify with a quick script before typesetting; do not substitute
  other numbers.
- **The running review sentence** should be chosen once at build (short, concrete,
  export-flavored, e.g. "The pomegranates arrived fresh and sweet.") and reused
  verbatim in L21 - it is a chapter-level constant; record it in the Provenance block.
- **Derivation source:** LMU `rnn/slides-backprop.tex` frames "Backpropagation through
  time" and "Long-Term Dependencies" - REDERIVE in our notation (z, V, W as here),
  do not paste.
- **OWN-TEX whitelist (this deck):** the vanishing-gradient "The cat ... was" TikZ
  frame and optionally the unrolled-RNN TikZ from
  `misc/dl4nlp/01_pre_transformer.tex` (~lines 407-530). Check macros (`lightbg`,
  `-Stealth`) against `ml/preamble.tex`; patch locally if missing.
- **WEB-IMG whitelist (this deck):** Karpathy's taxonomy strip (from "The Unreasonable
  Effectiveness of Recurrent Neural Networks") if TikZ is not cleaner.
- **`gradient_flow.py`:** tiny nets (hidden 16), toy recall task (predict an early
  token after a delay), a few hundred training steps, CPU seconds not minutes; log to
  `logs/`; fail loudly on any missing input.
- **Clipping callback:** grep L15's deck for gradient clipping before writing that
  frame; cite it if present, teach it in one formula if not.
- **No homework hook** - do not add an HW section to this deck or to `rnn.qmd`.
- **Credit line** on the title/Outline frame, same LMU wording as ch6 decks.
