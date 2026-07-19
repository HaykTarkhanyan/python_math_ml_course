# GenAI chapter (ch9) - Attention & Transformers - plan + outline

Drafted 2026-07-19; **scope approved by the instructor 2026-07-19** (see "Decisions" below).
This is the opening of the big **GenAI chapter** (`ml/ch9_genai/`): the attention + transformer
decks **L24 / L25 / L26**, which stop at "the T in GPT." Later GenAI parts (LLM training / RLHF,
generative models) extend the same chapter. Follows the house new-chapter workflow
(interview -> outline -> approval -> build) and the ch6/ch7/ch8 plan format.

## Mission

Pay off the two cliffhangers the course has been building toward:
- **L21 (RNN)** ended on the seq2seq **bottleneck**: "what if the decoder could look back at
  *all* the encoder states?" - asked, not answered.
- **L23 (VAE)** ended pointing at generative models / the GenAI track.

This chapter answers the L21 question - **that "looking back" IS attention** - and then makes
the leap the field made: *if attention lets any output read any input directly, do we even need
the RNN?* No. Drop it. What's left is the **Transformer** - "the idea that ate deep learning,"
the T in GPT. The chapter ends by handing off to the LLM/GenAI track (the `dl4nlp` decks).

**Tone the instructor asked for: "explained very properly," with a *ton* of visuals** (web
images, generated figures, and screenshots from the 3Blue1Brown videos). So the pedagogy leans
hard on the 3b1b intuition-building spine, not just the equations.

## Sources (all gathered 2026-07-19)

| Source | What it gives us | Location |
|---|---|---|
| **LMU I2DL attention deck** (17pp) | The **RNN-era bridge**: Bahdanau additive attention, the MT attention heatmap, image-captioning attention, then a compact transformer intro. This is the piece that connects to L21. | `ml/deep_learning/moodle_s26_course/slides/week26_attention_autoencoders/01_attention.pdf` |
| **House `02_transformers.tex`** (28 frames, already our palette) | A **complete, modern transformer deck**: key-idea/parallelism, Q/K/V intuition, scaled dot-product + worked heatmap, coreference viz, self-vs-cross, multi-head, what-heads-learn, positional encoding (sinusoidal + RoPE/ALiBi), the block, Pre-LN/RMSNorm/SwiGLU, encoder/decoder/causal-mask, the 3 architectures, CLM vs MLM, why-it-won, the O(n^2) wall, a GPT-3/LLaMA dimensions table. **We adapt this heavily.** | `misc/dl4nlp/02_transformers.tex` |
| **3Blue1Brown Ch5** "Transformers, the tech behind LLMs" (27:14) | The big picture: embeddings as directions in meaning-space, the unembedding/softmax, temperature, what the whole network is doing. Transcript saved. | `ml/ch9_genai/research/3b1b_ch5_transformers.md` |
| **3Blue1Brown Ch6** "Attention, step-by-step" (26:09) | The **intuition spine for attention**: the mole/Eiffel polysemy motivation, the "fluffy blue creature" running example, query/key as "asking/answering," the dot-product grid -> softmax -> **attention pattern**, masking, value-vector-as-delta-e, multi-head. Transcript saved, timestamps marked. | `ml/ch9_genai/research/3b1b_ch6_attention.md` |
| **Online reference set** | Vaswani et al. 2017 "Attention Is All You Need" (the architecture figure); Jay Alammar "The Illustrated Transformer" (illustrations); Bahdanau et al. 2015 (additive attention); BertViz / Karpathy (attention-head visualizations). Pulled at build time for specific images. | web (build-time) |
| **`misc/dl4nlp/` (the rest)** | The whole downstream LLM course (tokenization, decoding, pretraining/RLHF, MoE, RAG, scaling laws, agents, ...). **Out of scope here** - this chapter *hands off* to it. | `misc/dl4nlp/*.tex` |

## Decisions (approved by instructor 2026-07-19)

All recommendations accepted, with one change on #1 (big GenAI chapter, not a dedicated one):

1. **Chapter identity: one big `ml/ch9_genai/` GenAI chapter.** Attention + transformers are its
   opening (decks **L24 / L25 / L26**); the chapter later grows to the rest of GenAI (LLM
   training/RLHF, generative models) sourced from `dl4nlp` and the reference GAN decks.
2. **Scope ceiling: STOP at "the T in GPT."** No tokenization/pretraining/RLHF/RAG in depth in
   L24/L25 - those come later in the chapter / from the `dl4nlp` track. (Mirrors L21 and L23.)
3. **Reuse `02_transformers.tex` heavily** - port to `../preamble`, thread the callback spine +
   running example, swap/add 3b1b-style visuals. Not a from-scratch rebuild.
4. **Running example: two threads** - the **customer-reviews** text thread for motivation
   (translation, the bottleneck) + 3b1b's **"a fluffy blue creature roamed the verdant forest"**
   for the mechanics. Polysemy hooks (**mole**, **Eiffel tower**) open "why context matters."
5. **Extract 3b1b key frames** with `ffmpeg` at the marked timestamps into
   `fig/borrowed/3b1b/`, used as showcase visuals with a small "3Blue1Brown, CC BY-NC-SA" credit
   line. (Per the standing rule, no credit lines on the general web images.)

---

## Deck outline (revised 2026-07-19: 3 decks, ~19 + ~10 + ~10 frames)

Revised after a critique pass: split 2 -> 3 decks (L25 was overloaded), added an embeddings
primer (the attention intuition rests on it), compressed the RNN-era additive attention to
intuition-only, and added the matmul->GPU callback and the honest weak-inductive-bias frame.

Tags mirror the house convention: `[plain]` = section-transition frame; `[predict-first]` =
question + pause + reveal; `[3b1b]` = 3Blue1Brown screenshot or a redraw of one; `[adapt]` =
port from `02_transformers.tex`; `[web-img]` = downloaded image; `[fig]` = generated matplotlib;
`[tikz]` = our diagram. Every content frame carries a visual (heavy-visuals rule).

### L24 - Attention: how tokens share meaning

**Cold open / recap the cliffhanger**
1. Title + the L21 callback: "one context vector could not hold a whole sentence."
2. `[predict-first]` The bottleneck, revisited `[tikz]` - the seq2seq funnel from L21; "how would
   you fix it?" Reveal: let each output read *all* the states, weighted by relevance.

**Section 1 - Words are vectors; directions carry meaning (NEW primer)** `[plain]`
3. Recall: tokens become embeddings `[3b1b embeddings_3d]` `[callback L21 embedding_2d]` - a word
   is a vector in a high-dimensional space.
4. Directions carry meaning `[3b1b tower_neighbors + king_queen_analogy]` - semantic neighbors;
   `king - man + woman ~ queen`. (Also: the dot product measures alignment - we reuse this soon.)
5. Embeddings *soak up context* `[3b1b king_context + attention_updates_meaning]` - "a machine
   learning model" vs "a fashion model"; a generic embedding gets pulled toward a
   context-specific direction. **This pull is exactly the job attention does.**

**Section 2 - The attention idea (RNN-era, compressed to intuition)** `[plain]`
6. Attention = a weighted look-back `[adapt LMU intuition; NO additive-score formula]` - score
   every state for relevance, softmax, take a weighted sum. Intuition only.
7. You can SEE it `[web-img/LMU p7 heatmap + LMU p8 captioning]` - the FR->EN alignment grid and
   captioning-attention. Bridge line: *"same skeleton - score, softmax, weighted sum. Next we
   just let the model learn the pieces."*

**Section 3 - The leap: drop the RNN** `[plain]`
8. `[predict-first]` If any output can directly read any input, why keep the sequential RNN?
   `[3b1b deep_stack]` Reveal: you don't. **"Attention Is All You Need" (2017).**
9. Sequential O(n) vs all-to-all O(1) `[adapt 02_transformers "key idea"]` `[tikz]` - the three
   pillars (parallel / direct access / contextual).
10. Why context matters `[3b1b mole_polysemy + eiffel_context]` - the polysemy hook that
    motivates self-attention.

**Section 4 - Self-attention mechanics (the chapter's one deep dive)** `[plain]`
11. Running example `[3b1b query_creature]` - "a fluffy blue creature roamed the verdant forest";
    adjectives update nouns; "any adjectives in front of me?"
12. Q, K, V = three learned projections `[adapt 02_transformers]` `[tikz]` - one input X, three
    views; the soft-dictionary-lookup analogy.
13. Query.Key = relevance `[3b1b kq_dotgrid]` - the dot-product grid; **the same
    dot-product-as-similarity from the primer (frame 4)**, now between queries and keys.
14. Softmax -> the attention pattern; the equation `[3b1b attention_pattern]` `[fig]` -
    `softmax(QK^T / sqrt(d_k)) V`, and *why* the `sqrt(d_k)`. The equation to remember.
15. Values carry the update -> delta-e `[3b1b value_matrix + value_delta_e]` - weighted sum of
    value vectors added back to the embedding; "creature" moves to a new point in meaning-space.
16. Worked heatmap `[adapt 02_transformers "worked example"]` `[fig]` - "The cat sat," cat attends
    63% to itself, 25% to "sat."
17. **It's just matmul -> GPU (NEW)** `[tikz]` - `QK^T`, softmax, `x V` are big parallel matmuls.
    **Callback L14 "why GPUs" + L16 "conv = matmul."** This is *why* attention is fast and scales.
18. The whole single head, one frame `[3b1b single_head_full]`; and relationships emerge
    `[web-img BertViz]` - "...because **it** was too tired" -> "it" attends to "animal," no rule
    written.
19. Recap + bridge -> L25: "that's ONE head. Now run many, add position, wrap it in a block."

### L25 - The Transformer block: many heads, position, and the residual stream

**Section 1 - Multi-head** `[plain]`
1. Title + one-line single-head recap.
2. Multi-head attention `[adapt 02_transformers]` `[tikz]` - h heads in parallel, each its own
   W_Q/W_K/W_V; concat + W^O; `d_k = d_model / h`.
3. What different heads learn `[3b1b multihead + adapt]` - positional / syntactic / semantic
   heads; "wizard+Harry -> Harry Potter."

**Section 2 - Position** `[plain]`
4. `[predict-first]` Self-attention is **permutation-invariant** `[adapt]` `[tikz]` - "dog bites
   man" vs "man bites dog": same tokens, same output?! Order must be injected.
5. Positional encoding `[adapt]` `[fig]` - sinusoidal sin/cos + a real PE heatmap, added to the
   embeddings. One line: *modern LLMs use RoPE* (RoPE/ALiBi deferred, not derived here).

**Section 3 - The block & the residual stream** `[plain]`
6. The Transformer block `[adapt 02_transformers]` `[tikz]` - MHA -> Add&Norm -> FFN -> Add&Norm,
   x N. **Callbacks:** residual stream = **ResNet skip (L17)**; LayerNorm = **BatchNorm cousin
   (L15)**; FFN = a **per-token MLP (L14)**.
7. Masking = causal attention `[3b1b masking_softmax]` - token t sees only tokens <= t; -inf
   before softmax; this is what lets the model train on every prefix at once.
8. `[optional]` Modern block recipe `[adapt]` - Pre-LN + RMSNorm + SwiGLU (one frame; cut if time).
9. Stack the blocks `[3b1b deep_stack]` - attention <-> MLP, x N; meaning gets progressively
   richer with depth (the 3b1b "King -> lived in Scotland, murdered the king" deepening).
10. Recap + bridge -> L26 (the family, training, and why it won).

### L26 - Transformers in the world: architectures, training, why they won

**Section 1 - The family** `[plain]`
1. Title.
2. Encoder / decoder / cross-attention `[adapt, compressed to one frame]` `[tikz]` - enc-dec
   (translation); cross-attention plugs the encoder's K,V into the decoder.
3. Three architectures `[adapt]` `[tikz]` `[web-img Vaswani figure once]` - encoder-only **BERT**
   (understanding) / decoder-only **GPT** (generation, dominant) / enc-dec **T5** (seq2seq).

**Section 2 - How it learns, and becomes a chatbot** `[plain]`
4. Training = next-token prediction `[adapt CLM + 3b1b snape_prediction]` - **callback L21 LM
   factorization** `P(y_1..T)=prod P(y_t|y_<t)`: "GPT's objective, seen in L21, at scale."
5. Prediction -> chatbot `[3b1b unembedding_matrix + softmax_distribution + chatbot_system_prompt]`
   - last vector -> logits -> softmax -> sample; a system prompt turns autocomplete into a
   chatbot. (Decoding / temperature: a one-line tease - it comes later in ch9.)

**Section 3 - Why it won & honest limits** `[plain]`
6. Why transformers won `[adapt]` `[fig + 3b1b]` - parallelizable matmul (callback L24 frame 17)
   -> scaling -> qualitative jumps; the scaling curve.
7. The quadratic wall `[adapt]` `[fig + 3b1b]` - attention is O(n^2); context is the bottleneck;
   teases FlashAttention / long-context (later in ch9).
8. **Honest limits: weak inductive bias (NEW)** `[tikz]` - no built-in locality or order
   (position is bolted on) -> transformers need huge data; CNNs still win small-data vision (ViT
   needs scale). **Completes the inductive-bias ledger (ch4 trees / ch6 CNN / ch7 RNN).**
9. Transformers in the wild, 2026 `[web-img montage]` - LLMs, ViT, AlphaFold, multimodal - "same
   block, everywhere" (ties back to the L14 intro-deck applications gallery).

**Section 4 - Close** `[plain]`
10. Chapter arc + hand-off `[plain]` - bottleneck -> attention -> self-attention -> the block ->
    GPT. "You now know what the T in GPT is. **Next in this GenAI chapter:** how a Transformer
    becomes a chatbot (tokenization, pretraining, RLHF) and the other generative models
    (diffusion, GANs)." -> the next part of ch9_genai.

---

## Heavy-visuals plan (the "ton of visuals")

Per source, roughly in priority order:

- **3Blue1Brown screenshots** (ffmpeg extract at timestamps, `fig/borrowed/3b1b/`): the
  attention-pattern grid `[Ch6 09:49]`, the K/Q dot-grid `[08:40]`, the Eiffel/tower vector
  move `[03:01]`, the fluffy-blue-creature setup `[04:24]`, value->delta-e `[15:06]`, masking
  `[12:19]`, multi-head `[20:38]`; embeddings-as-directions + unembedding/softmax from Ch5.
  (License: 3b1b is CC BY-NC-SA - one small credit line, or we redraw. Decision #5 above.)
- **Web images** (`fig/borrowed/`, per standing no-caveat policy): the Vaswani architecture
  figure; a translation attention-alignment heatmap; a BertViz attention-head view; the
  captioning-attention figure; the 2026 applications montage.
- **Our TikZ** (adapt from `02_transformers.tex`): key-idea, Q/K/V projections, scaled-dot-
  product pipeline, self-vs-cross, multi-head, the block, encoder/decoder, 3-architectures,
  CLM/MLM, permutation-invariance.
- **Generated figures** (`py_src/*.py`, `ma` venv, seed 509): a real attention heatmap on a
  short sentence (toy Q/K), the sinusoidal PE heatmap (real sin/cos), the softmax-temperature
  demo, the O(n^2)-vs-O(n) growth curve, the scaling-curve sketch. All CPU-light, no model
  downloads.

## Course-callback spine (what makes it OUR chapter)

| New concept | Callback to | Framing |
|---|---|---|
| attention = weighted look-back over encoder states | L21 seq2seq bottleneck | the cliffhanger, paid off |
| self-attention drops the RNN | L20-L21 recurrence, L19 1D-conv | "neither recurrence nor convolution - just attention" |
| Q/K/V = learned projections | L14 linear layer, ch4b/ae embeddings | "three learned views of one input" |
| softmax over relevance | L14 softmax, L21 attention | same tool, new place |
| residual stream in the block | L17 ResNet skip connection | "the block reads/writes a residual stream" |
| LayerNorm | L15 BatchNorm | the normalization cousin |
| FFN per token | L14 MLP | "an MLP applied at every position" |
| positional encoding | L16 CNN "locality" / L21 order | order must be injected, not assumed |
| next-token training = CLM | L21 LM factorization + char-LSTM | "GPT's objective, seen in L21, at scale" |
| parallel -> scale | L14 why-GPUs, L16 conv=matmul | "attention is a big matmul; GPUs love it" |
| trees/CNNs/RNNs/transformers by data | ch4/ch6/ch7 inductive-bias ledger | the inductive-bias story completed |

## Open questions for the instructor

Scope questions 1-5 resolved 2026-07-19 (see "Decisions"). Still open:

1. **Homework / practical? (NEW - please decide.)** Attention is an ideal small practical.
   Options: (a) **"implement scaled-dot-product attention in ~15 lines of numpy"** on a toy
   sentence (CPU-light, no downloads); (b) **"visualize attention heads with BertViz"** on your
   own sentence; (c) both, as a short chapter `.qmd`; (d) **none** (like the RNN chapter).
   Recommendation: **(a)** as the core (it makes the equation concrete), (b) optional.
2. **Interactive angle:** a live BertViz / attention-visualizer link on the `.qmd` (like the AE
   chapter's latent-space explorer)? Cheap to add, perishable - verify at delivery.
3. **Deck 3 (L26) scope:** keep the 2026 applications montage + quadratic-wall here, or is L26
   getting long? (Currently ~10 frames - fine, but flagging.)

## Build status

- [x] Sources gathered: LMU attention deck scoped; `dl4nlp/02_transformers.tex` read in full;
      3b1b Ch5 + Ch6 transcripts fetched to `research/` (yt-dlp + youtube-transcript-api).
- [x] Plan + outline drafted, then revised to 3 decks after a critique pass (this file).
- [x] Scope approved by instructor (2026-07-19); one open item left: the HW/practical decision.
- [x] 3b1b key-frame extraction: **29 frames** in `fig/borrowed/3b1b/` (+ manifest `README.md`).
- [ ] Web-image pull (Vaswani figure, attention heatmap, BertViz, captioning, 2026 montage).
- [ ] Figure scripts (`py_src/`).
- [ ] Build L24 -> L25 -> L26 (one at a time; 2x pdflatex, overflow check, aux clean, provenance).
- [ ] Build L24, then L25 (2x pdflatex, overflow check, aux clean, provenance block).
- [ ] `attention.qmd` + `_quarto.yml` registration.
