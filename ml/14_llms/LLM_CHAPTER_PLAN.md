# 14 LLMs - chapter plan (draft for approval)

Drafted 2026-10-02; folder, order and map decided 2026-10-04 (DECISIONS #66, #67); numbered in
teaching order, LLM-1 to LLM-14, on 2026-10-05 (#71: L24-L26 became LLM-3 to LLM-5). **Not approved
yet** - the open decisions in section 1 come first. Follows the house new-chapter workflow
(interview -> outline -> approval -> build), like `ATTENTION_CHAPTER_PLAN.md` (this folder, LLM-3 to LLM-5)
and `ch17_rag/RAG_CHAPTER_PLAN.md`.

**Scope.** The whole language-model chapter between the RNN chapter and the application chapters
(RAG, agents): tokenization, embeddings and position, the three built attention lectures (LLM-3 to LLM-5),
and LLM-6 to LLM-14.
Sources: the instructor's `misc/dl4nlp/` decks and the `ml/llm_training/slides/` paper decks,
copied into `sources/` (see `sources/README.md`), and recent material verified on 2026-10-02
(section 9).

---

## 1. Open decisions

| # | Decision | Recommendation | What would flip it |
|---|---|---|---|
| D1 | Sessions for the chapter (not counting RAG/agents) | **13**: LLM-1 to LLM-13 (LLM-3 to LLM-5 built in July as L24-L26); LLM-14 if time allows | A hard course end date. See section 8 for the 10-session cut |
| D2 | Where GANs go | **Resolved 2026-10-03:** AE -> VAE -> GAN -> diffusion, after this block (DECISIONS #63) | - |
| D3 | How deep alignment goes here | **Idea level in LLM-9**; the math stays in L32f/L32g, which depend on PPO from L32d | Moving the whole RL chapter right after this block (+7 lectures before RAG) |
| D4 | Local runtime for the hands-on parts | **Ollama** for the first contact (one installer, Windows-friendly); **llama.cpp tools** for the quantization measurements (`llama-quantize`, `llama-perplexity`, `llama-bench` have no Ollama equivalent) | Most students on Macs -> LM Studio or MLX deserve the demo slot |
| D5 | Fine-tuning library for the practical | Two real options: **PEFT + TRL** (the standard HF stack, every step visible) or **Unsloth** (fits a free T4 more easily, `save_pretrained_gguf` exports straight to GGUF). Lean: Unsloth for the practical, PEFT shown on a slide | If the goal is "see the mechanics" over "ship a model", PEFT + TRL |
| D6 | Base model for practicals | **Decide by measurement**, not by name: tokens per Armenian word and Armenian perplexity for 3-4 small open models (e.g. Gemma 4 E2B, a small Qwen3.5) | - |
| D7 | Chapter folder and number | **Resolved 2026-10-04:** one folder, `ml/14_llms`, for attention and the LLM lectures; RNN is `ml/13_rnns` (DECISIONS #66). Deck files kept `L24`-style names until delivery; **revisited 2026-10-05 (#71):** the chapter is numbered in teaching order, LLM-1 to LLM-14 (L24 -> LLM-3, L25 -> LLM-4, L26 -> LLM-5, every later label +3) | - |
| D8 | Teaching order inside the chapter | **Resolved 2026-10-04:** left to right along the forward pass - tokenization before attention - with a "you are here" map in every session (DECISIONS #67, section 5.0) | - |

---

## 2. Where the block sits

**Course order (instructor decisions: DECISIONS #63 on 2026-10-03, superseding #62; the order
inside this chapter from #67 on 2026-10-04):**
CNN (L16-L19) -> **RNN** (L20, L21) -> **this chapter, part 1**: LLM-1 tokenization, LLM-2
embeddings and position, LLM-3 to LLM-5 attention and the transformer, LLM-6 to LLM-11 -> RAG (`ch17`, L41-L43) -> Agents (`ch18`, L44) ->
**this chapter, part 2**: LLM-12, LLM-13 (LLM-14) -> **Autoencoders** (L22, L23) -> **GANs** (L23b,
L23c) -> **Diffusion** (L27-L31) -> the rest (RL, VLM, ...).

RAG and agents sit inside the chapter's run on purpose: LLM-11 ends on "give the model the
documents" (-> L41), and LLM-13 opens with the ladder "prompt -> RAG -> fine-tune", which needs RAG
taught. (Until 2026-10-04 this section said "this block -> RAG", which contradicted section 5.)

Two threads, each unbroken: text (RNN -> attention -> LLMs) and generative (AE -> VAE -> GAN ->
diffusion).

**Nothing in or right after this block needs autoencoders** (checked 2026-10-03, grep for
autoencod / VAE / L22 / L23 over the decks): `14_llms` 0, `ch17_rag` 0, `ch18_agents` 0. The
LLM source decks say "denoising" only about T5/BART span corruption
(`misc/dl4nlp/06_early_notable_models.tex:588, 640`) - one sentence, no chapter needed.

**Chapters that DO need autoencoders** - schedule them after L22/L23:

- Mech interp (`ch19`): `xx_features.tex:261` "Back to the autoencoder chapter" and sparse
  autoencoders throughout; also `xx_opening_the_box.tex:908`, `xx_personas_and_diffing.tex:312`.
  It is LLM-adjacent, but it **cannot** follow this block directly.
- VLM (`ch12`): `L34_vlm_drawing.tex:84-86` builds VQ-VAE from "the autoencoder of ch8".
- JEPA (`ch16`): `L39_jepa_objective.tex:76, 148` compares against the autoencoders.
- Diffusion (`ch10`): latent diffusion runs inside the VAE.

**Edits this order needs:**

- `ch8_autoencoders/L23_vae.tex:473` - the closing "Next" box points at attention via L21 ("the
  attention chapter (L24+)"); repoint it to GANs, then diffusion.
- `ch8b_gans/L23c_gan_applications.tex:427` - drop "Before it, the attention chapter (L24) picks up
  the other thread". Both lines still carry the pre-#71 label L24 on purpose: these edits remove it.
- `_quarto.yml` order, qmd title numbers, `ml/00_plan.md` (instructor 2026-10-04: Quarto later;
  the rename only fixed the two `_quarto.yml` paths so the site still builds).
- **Done 2026-10-04 (#67):** L21's closing Next box now points to tokenization first, then
  attention; LLM-5's closing frame no longer lists tokenization as coming next.
- Chapter numbers written inside decks ("chapter 8" in `xx_features.tex:263`, "ch8" in `L34`,
  `L39`) - part of the renumbering pass (D7).
- **No edit needed:** L22's reference to the RNN (RNN now comes first), `HW1_sae_rnn` stays the
  autoencoder homework, `L23c:147`'s king-queen callback to L21, L21 -> LLM-3.
- Small gain: L22's masked-modeling line (`:83`) and its "LLM interpretability" teaser (`:85`)
  turn from forward references into callbacks to this block.

---

## 3. What students already have

Do not re-teach these; call back to them. The LLM-2 and LLM-3 to LLM-5 rows are this chapter's own sessions 2-5
(LLM-3 to LLM-5 built, and reference material since #68); LLM-6 onward builds on them.

| Already taught | Where |
|---|---|
| One-hot encoding and why its geometry is meaningless (no tokenization, no embeddings - DECISIONS #64) | L20 |
| LM factorization, the generation loop (sample, feed back, repeat - no temperature), a GRU inventing surnames | L21 |
| The seq2seq bottleneck, and the alignment grid: every decoder step gets the same C, each needs a different part of the input | L21 |
| Bengio 2003 neural LM (the surname inventor practical), with a bigram baseline and perplexity = e^loss ("torn between k letters", its section 6.3) | ch11, `47_name_inventor_solution.ipynb` |
| Embeddings (the lookup table, what it learns), position (learned, sinusoidal, RoPE, long context) | LLM-2 (moved from LLM-3/LLM-4, #68) |
| Self-attention, multi-head, the block, causal mask, residual stream | LLM-3, LLM-4 |
| Encoder / decoder / enc-dec (the encoder sees both ways; the masked-LM *objective* is not shown - checked 2026-10-05), logits -> softmax -> sample, "autocomplete is not a chatbot", the O(n^2) bill and the 16,384-token FLOP crossover | LLM-5 |
| Log-loss, temperature scaling, calibration | [11], [14] |

**Not yet taught when this block runs:**

- **Reinforcement learning.** The RL chapter (ch11, L32-L32g, which owns RLHF/DPO math, GRPO and
  R1) comes *after* this block. That is why D3 matters.
- **Autoencoders** (DECISIONS #63). No lecture may lean on them. Where an autoencoder callback
  would be natural, state the idea directly or call back to PCA (ch10, taught): masked LM and span
  corruption are "corrupt the input, reconstruct it" (LLM-7); LoRA's update is low rank (LLM-13);
  MLA squeezes the KV cache through a learned low-rank bottleneck (LLM-14).

---

## 4. Design principles

1. **Left to right along the map** (#67). First one forward pass, in the order data flows through
   it: tokens -> embeddings and position -> attention -> the block -> the LM head -> decoding. Then the life of
   a model: where the weights come from (pretraining, scale) -> how a base model becomes an
   assistant -> how you use it -> how you know it is wrong -> how you run and adapt it yourself.
   Last, what changed since 2017.
2. **One home per topic.** The repo holds four overlapping sources; today RLHF/DPO/GRPO each appear
   in 4 decks, FlashAttention in 3-4, speculative decoding and the emergence "mirage" debate in 2.
   Section 7 assigns every source deck a destination.
3. **Port and adapt, not copy.** The dl4nlp decks are survey decks (definitions + "further
   reading") on their own `misc/dl4nlp/preamble.tex`. Each lecture gets the `ml/SLIDE_STYLE.md`
   treatment: cold open, transition slides, predict-first, Python figures, measured where cheap.
   Same mechanism as LLM-3 to LLM-5 were built from `02_transformers.tex`.
4. **Recent material is perishable.** Every 2025-26 fact carries its verification date and source
   (section 9). The landscape frames in LLM-14 get re-verified right before delivery.
5. **Armenian thread.** Tokens per Armenian word, Armenian perplexity, an Armenian fine-tune. It is
   the course's local hook and it measures something real (non-English text is where tokenizers,
   quantization and small models break first - a hypothesis to measure, not to assert).

---

## 5. The lectures

### 5.0 Session order and the chapter map

The map is two illustrated frames, one per row (instructor 2026-10-04: "a lot nicer ... not just
text boxes", two frames for more space):

- **"Where we are: one forward pass"** - GPT-2 small on LLM-3's sentence "The cat sat on the", one
  row per token, after [Transformer Explainer](https://poloclub.github.io/transformer-explainer/):
  text -> token chips with real IDs -> embedding vectors (real values) -> + position vectors (own
  stage since #68) -> a stack of N layers
  (attention arcs from the last token, one real head; an expand-and-contract MLP per token) -> the
  last vector -> LM head -> the real top-5 next-token probabilities -> the chosen token, looping
  back into the text.
- **"Where we are: the life of a model"** - after Karpathy's "State of GPT" pipeline: seven cards
  with a small drawing and one fact each (pretrain, scale, post-train, prompt, evaluate, quantize,
  LoRA), grouped "make it" / "use it" / "run and adapt it".

Each session opens with both frames (its own stages on a blue panel, earlier ones in full colour,
later ones faded, plus a "Today:" or "Later:" line) and ends with "Next on the map": the next
session's frame for its row. Figures: `py_src/llm_roadmap.py` (`--gpt2` once ->
`results/roadmap_gpt2.json`, then the maps) -> `fig/roadmap_<key>_pass.pdf` and
`fig/roadmap_<key>_life.pdf`, plus `roadmap_overview_*` with nothing lit. Its `SESSIONS` list must
match the table below.

| # | Session | Map boxes | `roadmap_` key | Status |
|---|---|---|---|---|
| 1 | LLM-1 Tokenization | Tokenizer | `tokenization` | built 2026-10-04 (`LLM1_tokenization.tex`) |
| 2 | LLM-2 Embeddings and position | Embedding, Position | `embeddings` | built 2026-10-04 (`LLM2_embeddings_position.tex`) |
| 3 | LLM-3 Attention | Attention | `attention` | built as L24; restructured and student-reviewed 2026-10-05 (`LLM3_attention.tex`, 41 pages; critique + interview in the LLM-3 to LLM-5 section, `LLM3_attention_OUTLINE.md`) |
| 4 | LLM-4 The Transformer block | Attention, MLP | `block` | built as L25; restructured 2026-10-05 (`LLM4_transformer_block.tex`, 41 pages; critique, interview, `LLM4_transformer_block_OUTLINE.md`); student review pending |
| 5 | LLM-5 Transformers in the world | LM head | `transformers` | built as L26 (`LLM5_transformers_in_the_world.tex`); map frames added 2026-10-04 |
| 6 | LLM-6 Decoding | Decoding | `decoding` | built 2026-10-05 as LLM-3 (`LLM6_decoding.tex` since #71) |
| 7 | LLM-7 Pretraining | Pretrain | `pretraining` | to build |
| 8 | LLM-8 Scaling and emergence | Scale | `scaling` | to build |
| 9 | LLM-9 Post-training | Post-train | `post_training` | to build |
| 10 | LLM-10 Prompting | Prompt | `prompting` | to build |
| 11 | LLM-11 Evaluation and hallucination | Evaluate | `evaluation` | to build |
| - | RAG (L41-L43), agents (L44) | (own chapters, no map) | - | built |
| 12 | LLM-12 Quantization | Quantize | `quantization` | to build |
| 13 | LLM-13 LoRA and QLoRA | LoRA | `lora` | to build |
| 14 | LLM-14 Anatomy of a 2026 LLM | Attention, MLP again (KV cache, MLA; MoE) | `anatomy_2026` | optional |

The LLM-n labels are planning names; decks get their `NN_` playlist number on delivery.

Frame counts below are for the source decks as they exist on 2026-10-02.

### Part A - How an LLM is built

#### LLM-1 Tokenization: from text to token IDs

Session 1 of the chapter, the map's first box. L20 fed words in as one-hot vectors over a fixed
word list ("Tokens, minimally"); what should a token be? Build the tokenizer every LLM actually
uses. It ends at **token IDs** and hands over to LLM-2, whose first job is "an ID becomes a vector"
(#68). It must not lean on embeddings or attention: they come next.

**Interviewed 2026-10-04** - outline: `LLM1_tokenization_OUTLINE.md` (37 frames; its three open
decisions answered the same day - ready to build).

- **Cold open:** "How many r's in strawberry?" (instructor's pick over the Armenian-cost hook).
  Measured: in that question ` strawberry` is **one token** in GPT-2, cl100k and o200k alike.
- **Unicode / UTF-8:** short, 2-3 frames - code points -> UTF-8 bytes; why an Armenian letter is 2
  bytes.
- **BPE: full mechanics** - the idea, a worked example by hand, encoding new text, byte-level BPE
  (GPT-2 onward), pre-tokenization regex, decoding IDs back - including the
  invalid-UTF-8-mid-character trap (`_learnings/2026-08-21-0140_bpe-merge-halves-are-invalid-utf8-alone.md`
  is a real instance; measured: `ա` is 2 tokens in cl100k, `\xd5` + `\xa1`).
- **WordPiece and Unigram/SentencePiece:** one frame each.
- **Quirks, all four get a frame**, plus the strawberry explanation: numbers and arithmetic,
  glitch tokens (SolidGoldMagikarp), code whitespace, special-token leaks.
- **Armenian: one frame**, "one sentence, two counts" (instructor's pick over the 8-tokenizer
  chart and a full glitch-hunt section), on both tokenizers: 102 vs 18 (cl100k, one token per
  byte) and 19 vs 17 (o200k). The glitch-token project stays a pointer.
- **Glitch tokens: two frames** (the story, then GPT-3's replies).
- Special tokens; the vocabulary-size trade-off (GPT-2 50,257 -> cl100k 100,277 -> o200k 200,019,
  measured with tiktoken 2026-10-04).
- **Not in the slides:** the hands-on part (instructor: "doesn't matter for the slides").
- **Source:** dl4nlp `03_tokenization.tex` (38 frames), now taught **from scratch** here: L21's
  three tokenization frames (trilemma, subwords on a real tokenizer, the Armenian tax) moved to
  this lecture on 2026-10-03 (DECISIONS #64). Reuse their measured figures
  (`ml/13_rnns/py_src/tokenizer_demo.py` -> `ml/13_rnns/fig/tokenizer_panel1-3.pdf`, cl100k counts)
  and frame text (`git show 6013bcc:ml/ch7_rnn/L21_road_to_attention.tex`). Chat tokens move to
  LLM-9. Optional 1 frame from llm_training 05 (BPE-dropout). Source copy:
  `sources/dl4nlp/03_tokenization.tex` (its PDF was 0 bytes; recompiled 2026-10-04).
- **Built 2026-10-04:** `LLM1_tokenization.tex`, figures from `py_src/tokenization_figs.py` (every
  split measured; numbers quoted in prose asserted). Self-review + Sonnet student review the same
  day; after the instructor's "I prefer proper decks" (frame count never matters, see
  `ml/SLIDE_STYLE.md`) it gained five worked frames: UTF-8 by hand, BPE on the 689 surnames' bytes
  (finds "-yan" by merge 5), where IDs come from, WordPiece by hand, Unigram by hand. It
  replaced `13_rnns/py_src/tokenizer_demo.py` instead of moving it: those panels were drawn 13 in
  wide. The Armenian line stays one shared file, `13_rnns/py_src/data/armenian_line.txt`, read by
  path. `tokenizer_demo.py` and its three panels are now used by no deck - candidates for
  `13_rnns/py_src/archive/`. `embedding_2d.py` waits for LLM-7.
- **Hands-on (not in the deck; format undecided):** train byte-level BPE on Armenian text (~50
  lines), compare tokens per word against `o200k` and one open-model tokenizer.

#### LLM-2 Embeddings and position: from IDs to vectors

Session 2 of the chapter, the map's Embedding and Position boxes (DECISIONS #68; instructor
2026-10-04: "id like positional encodings to be there, were going left to right"). LLM-1 ended at
token IDs, and an ID is only a row number. This session builds the vector the blocks actually read,
`wte[id] + wpe[pos]`, and hands over to LLM-3 with one promise: the next layer weighs vectors by their
content, and a weighted sum ignores order. LLM-3 proves it.

**Interviewed 2026-10-04** - outline: `LLM2_embeddings_position_OUTLINE.md`. **Built the same day:**
`LLM2_embeddings_position.tex`, figures from `py_src/embeddings_position_figs.py` (GPT-2's weights
via safetensors, every quoted number asserted; `save()` refuses any text outside its figure).

- **Cold open:** "The dog bites the man" vs "The man bites the dog" (instructor's pick over "word
  1025"), with the instructor's Kargin Haghordum reference on the frame (the Armenian line drawn as a
  figure, the link as `\href`). Measured on GPT-2: the same 5 token IDs, summed vectors identical to
  the bit. The bare "dog bites man" / "man bites dog" are **not** the same tokens (`dog` vs ` dog`,
  LLM-1's leading space).
- **Embeddings:** the lookup is one-hot x matrix (by hand); trained by backprop like any weight
  (callback: the ch11 surname inventor's table); what GPT-2's table learned, as nearest neighbours of
  interesting words (instructor: "more interesting words, not just Paris"); 31% of GPT-2 is this
  table (LLM-1 callback). word2vec stays in LLM-7.
- **Position:** why (a lookup gives "dog" the same vector in every slot; a plain sum stays blind
  even after positions are added - measured - so positions work because each vector changes);
  GPT-2's learned table (1024 rows: the hard limit; it learned smooth waves on its own, measured);
  sinusoids; add vs concatenate (by hand, plus measured near-orthogonality); RoPE, rotate instead of
  add (by hand: same gap, same score).
- **Long-context tail, intuition only, 3-4 frames** (instructor: more time on the important ideas,
  less on the technical): why RoPE breaks past its training length, stretching the clock (position
  interpolation, YaRN as an idea, no formulas), ALiBi and NoPE in one frame, what 2026 models use.
- **Offered, not picked:** sinusoids by hand, GPT-2 with its position table removed, Armenian
  neighbours in a multilingual model, the full long-context formulas.
- **Sources:** LLM-3's primer (3 frames) and LLM-4's Position section (5 frames), moved;
  `sources/llm_training/16_rope` (13 frames); dl4nlp 02 (three position frames) and 16 ("Positional
  encoding for length", "Context window extension"). Two claims in LLM-4's sinusoid frames are
  overstated and get fixed in the port after a source check: that sinusoids work at lengths never
  seen in training, and RoPE's "better extrapolation".
- Real GPT-2 vectors replace the hand-placed 2D map (`13_rnns/py_src/embedding_2d.py`, an archive
  candidate once LLM-2 is built).

#### LLM-3 to LLM-5 Attention and the transformer (built; sessions 3-5)

Plan and decisions: `ATTENTION_CHAPTER_PLAN.md`. Since 2026-10-04 each deck opens with "Where we
are" and closes with "Next on the map" (section 5.0). LLM-3's cold open still restates L21's
cliffhanger ("Last chapter ended on a cliffhanger"), which now has two lectures (LLM-1, LLM-2) in
between; its title page says "LLM chapter".

**Instructor 2026-10-04:** treat LLM-3 and LLM-4 as reference material ("collecting dust, we're gonna
iterate over them anyways"). Their embedding primer and Position section move to LLM-2 (#68); the
decks are not kept in sync with that move until the rework. **Rework list (from building LLM-2):**

- ~~LLM-3's "Where we are" caption still says "Today: the embedding and the attention step"~~
  **Done 2026-10-05** (LLM-3 restructure).
- ~~LLM-3 still opens with the three embedding-primer frames~~ **done 2026-10-05** (cut). LLM-4
  still says position comes in LLM-4 (its Position section). LLM-3's old closing line "Next
  (LLM-4): ... word order put back in" is gone. Open for LLM-4.
- **LLM-2 promises the shuffle proof** ("the attention lecture proves it: shuffle the inputs and
  every output comes back unchanged, just shuffled"). It is LLM-4's "Predict first" / "No.
  identical" / "Why that had to be true" (figure `l25_permutation.pdf`). **Decided 2026-10-05: it
  stays in LLM-4** (instructor did not pick moving it into LLM-3); LLM-2's "the attention lecture"
  wording is loose but true for the attention pair.
- LLM-4 writes $W^Q, W^K, W^V, W^O$ (superscripts); the restructured LLM-3 writes $W_Q$, $W_K$,
  $W_V$ (3b1b's subscripts) and mentions $W^O$ once. Pick one notation in the LLM-4 rework.
  (Found 2026-10-05.)
- LLM-4's sinusoid frames overstate extrapolation (sinusoids "including at positions longer than
  anything it was trained on"; RoPE "better extrapolation"). LLM-2 has the corrected version
  (Press et al. 2022).
- LLM-5:204 says encoder-only models are "exactly what the retrieval chapter **used** them for" -
  past tense from the old order; RAG (L41) now comes after LLM-11. Make it a forward pointer.
  (Found 2026-10-05 while critiquing LLM-7.)

**L24 critique (2026-10-05, before the interview; instructor: "rename and improve L24 ... I want
attention explained real good").** Read against L21 (the two walls, the alignment grid), LLM-2
(which now owns the primer, a by-hand score -> softmax -> weighted sum, and the query/key preview)
and LLM-4. The deck is 56 pages.

1. **The opening is stale.** "Lecture 24", "Last chapter ended on a cliffhanger" (two lectures
   now sit in between), and a hand-drawn "The cat sat" bottleneck that is weaker than L21's real
   alignment grid. L21 ended on two walls, the bottleneck and the queue; LLM-3 never pays them off
   by name.
2. **Section 1 (4 frames) is LLM-2's now** (#68): token = vector, directions, king - queen. Keep
   only what motivates attention: the polysemy frames ("model", "mole") and the murder-mystery
   still.
3. **The by-hand frame repeats LLM-2 and does not show what it claims.** LLM-2 already runs
   score -> softmax -> weighted sum with Q = K = V = X on 2-number vectors and promises "LLM-3 adds
   learned matrices"; LLM-3 runs the same simplification again. Its result, creature (1, 1) ->
   (0.75, 0.75), points the same way as before, so its meaning did not change, and "pulled a
   quarter of the way toward fluffy and blue" is wrong (it moved halfway to their midpoint, along
   its own direction). The toy also hides its best lesson: the biggest weight (0.50) goes to
   creature **itself**. Without W_Q and W_K a word mostly listens to itself, and the score grid
   is symmetric (A looks at B exactly as much as B looks at A).
4. **"Why three matrices?" is asserted, not shown** (a table of roles). Point 3's fix is the
   demonstration: add a small W_Q, W_K and creature's weight moves from itself to the
   adjectives, and the grid stops being symmetric. Then W_V: what you are found by is not what
   you hand over.
5. **No shapes.** The deck never says X is n x d, Q is n x d_k, QK^T is n x n. That n x n grid is
   where LLM-5's quadratic bill comes from. One frame, with GPT-2's sizes (d = 768, 12 heads of 64,
   from its config; no run).
6. **The real-model evidence contradicts the story.** The deck says attention moves "creature"
   toward "fluffy blue"; its one real head (DistilBERT, an encoder, not the map's GPT-2) "attends
   to the next token": a position rule, no meaning. Next to it sits a hand-set "illustrative"
   heatmap of the same sentence, and LLM-4 shows real heads again. Better: one GPT-2 head (the
   cached model, one run) doing something about meaning, e.g. a pronoun looking at its noun - if
   such a head exists; measured, not assumed.
7. **The leap comes before what makes it possible.** "Drop the RNN" (section 3) is argued with
   cross-attention ("any output position can pull from any input position"), but dropping the
   encoder RNN needs *self*-attention, which only arrives in section 4. L21's two walls give the
   order: wall 1 (one C for every step) -> attention over the encoder states (2014-15); wall 2
   (the queue) -> self-attention replaces the recurrence (2017).
8. **Most concepts appear twice.** 13 full-bleed 3b1b stills; about 10 repeat the image of the
   frame right before (Eiffel, query, key, Q/K space, value, dot grid, pattern, formula, delta-e,
   one head). Keep the instructor's visual spine, drop the doubles: one beat, one picture.
9. **A wrong still.** `fig/borrowed/3b1b/fs_creature.jpg` was grabbed at 04:12, one second after
   `fs_mystery.jpg` (04:11); page 23 shows the murder mystery a second time, not the creature.
10. **Three analogies for one idea** (library, soft dictionary, web search) and three dot-product
    frames (the primer callout, "up close", "matching"). One of each.
11. **Too many example sentences:** fluffy blue creature, "Donkey don't die", the map's "The cat
    sat on the", Eiffel tower, mole, model. One for the mechanics, one for the real head.
12. **The shuffle proof** (rework list above): LLM-2 promises "the attention lecture proves it",
    and today it is in LLM-4. Right after the equation is where it is cheapest: permute the rows of
    X and the output's rows permute the same way, nothing else changes.
13. **Small:** "for d_k = 64 the raw scores swing by +-5 or more" - with unit-variance entries the
    spread is sqrt(64) = 8, so say 8; the GPU frame is a TikZ chip that repeats L21's second wall
    and LLM-5's "why transformers won"; "Recall ... L20's anatomy frame" points at the old primer.

**Kept (instructor decisions 2026-07-19, `ATTENTION_CHAPTER_PLAN.md`):** the 3b1b intuition spine
with plenty of visuals, the fluffy blue creature, going slow on Q, K, V, the predict-first frames,
the NumPy snippet.

**Rename:** "Lecture 24" no longer matches its place (session 3, after LLM-1 and LLM-2). Renaming
the file reopens D7 ("deck files keep L24-style names until delivery"). Options go to the
interview.

**Interviewed 2026-10-05** (after the critique above):

- **Scope: restructure**, not a surgical fix and not a rebuild. Keep the 3b1b spine and the
  creature; open on L21's two walls; cut the primer LLM-2 owns and the duplicate stills; new
  pieces below.
- **Rename: renumber the chapter in teaching order.** L24 -> LLM-3 Attention, L25 -> LLM-4 The
  Transformer block, L26 -> LLM-5 Transformers in the world, decoding -> LLM-6, and every later
  label +3 (anatomy -> LLM-14). Reopens D7 (DECISIONS entry with the renumber pass).
- **Cold open: L21's two walls.** Show L21's alignment grid again (each Armenian word needs
  different English words): attention is that grid, learned. Then wall 1 (one C for every step)
  and wall 2 (the queue), each with its fix.
- **New pieces:** learned W_Q, W_K by hand (without them creature attends most to itself and the
  grid is symmetric; with a small W_Q, W_K it attends to fluffy and blue), and a shapes frame (X n
  x d, Q n x d_k, QK^T n x n, GPT-2's sizes from its config). **Not picked:** a real GPT-2 head,
  moving the shuffle proof in (it stays in LLM-4).
- **The DistilBERT real-head frame: cut** (instructor's pick, follow-up question). LLM-4 already
  shows real heads ("Do heads actually specialise", "Two heads, same layer"); LLM-3 ends on the
  mechanism with one line pointing there. The hand-set "illustrative" heatmap goes with it.
- **Order of work:** the renumber pass first (done 2026-10-05, DECISIONS #71), then the LLM-3
  outline.
- **Assumed unless the instructor objects:** the creature as the one example for the mechanics;
  one analogy (the soft dictionary lookup); duplicate stills dropped (`fs_creature.jpg` among them:
  the frame before it already shows the creature, so no re-grab); one "try it" link to Transformer
  Explainer.
- **Outline:** `LLM3_attention_OUTLINE.md` (drafted 2026-10-05). Its four open questions were
  answered the same day: a one-number query/key space in the by-hand frame, sections named after
  the two walls, the July "Donkey" sentence as the shapes frame's example (9 GPT-2 tokens), and
  L21's alignment grid confirmed correct by the instructor.

**LLM-4 critique (2026-10-05, before the interview; instructor: "start criticizing llm 4").** Read
against LLM-2 (which owns position), the restructured LLM-3 (which ends on one head and points
here for multi-head, W^O, real heads, the residual connection, masking and "why 64"), and LLM-5.
The deck is 29 pages from July.

1. **A third of it belongs to LLM-2 now.** The Position section (predict, "No. identical", "Why
   that had to be true", two sinusoid frames) repeats what LLM-2 owns (#68): LLM-2 ran the shuffle
   by hand on "dog bites man", and LLM-3 recalls it in a watch box. A predict-first on something
   students have seen twice predicts nothing. The general proof stays here (decided 2026-10-05),
   as one frame; the sinusoid frames go (the second also carries the wrong extrapolation claims).
   Better placement for the proof: the whole **block** is blind to order - attention is
   permutation-equivariant, the MLP and the norm act on each token alone - so position has to come
   from outside the stack (LLM-2).
2. **The opening still lists "attention cannot see order" as one of the three problems.** With
   position gone, the three are: one head asks one question (-> many heads); tokens talk but never
   think alone (-> the MLP); one layer is not a network (-> residual stream and norm). Plus the
   decoder's rule, the causal mask.
3. **The MLP is the thin half, though it holds two thirds of the weights.** One GPT-2 block:
   MLP 4,722,432 parameters, attention 2,362,368, norms 3,072 - the MLP is 66.6% (read from the
   cached `model.safetensors` header, 2026-10-05). The deck gives it one paragraph and a still. In
   July, 13 3Blue1Brown chapter-7 stills were extracted for exactly this ("L25 MLP cold-open":
   Michael Jordan plays ___, up-projection rows as questions, ReLU as an AND gate, down-projection
   columns as directions; `fig/borrowed/3b1b/README.md`) and never wired into any deck.
4. **No by-hand frame anywhere.** LLM-2 and LLM-3 have them; LLM-4 has formulas, measurements and a
   table. Candidates on LLM-3's toy: two heads asking two questions, concatenated, then W_O; an MLP
   neuron that fires on "fluffy AND creature" and writes a new feature; LayerNorm on a 4-number
   vector; the causal mask on LLM-3's creature grid (creature's row does not change - its
   adjectives came first - but fluffy loses blue and creature).
5. **The real heads come from the wrong model.** `py_src/l25_real_attention.py` uses
   `intfloat/multilingual-e5-small`, an embedding encoder: it attends both ways, which contradicts
   the deck's own masking frames, and it is neither the map's GPT-2 nor the model behind LLM-3's
   Transformer Explainer link. The cached GPT-2 can give real heads (a claim about a real model,
   local, one run).
6. **The numbers switch models.** The multi-head arithmetic uses the 2017 paper's d = 512, h = 8;
   LLM-3 just used GPT-2's 768 = 12 x 64 and promised "why 64" here. Use GPT-2's numbers, and say
   what W_O is for (GPT-2's `c_proj`, 768 x 768: each output number can draw on every head).
7. **The 2017 order is taught as the main form; GPT-2 normalises first.** The residual-stream frame
   and `l25_block.pdf` show Norm(x + f(x)), then "modern models move it inside". GPT-2 (2019)
   already does (`ln_1` before attention, `ln_2` before the MLP in its weight file). The
   residual-stream reading is cleanest that way - the stream itself is never normalised, each
   sublayer reads a normalised copy and adds back. Teach x <- x + Attn(LN(x)), x <- x + MLP(LN(x));
   2017 as a footnote.
8. **"One pass, n training signals" is taught three times:** here ("Why that trick matters"),
   LLM-5 ("The objective you have already seen"), and LLM-7's proposed window-by-hand. One home -
   here, with the mechanism - and callbacks elsewhere.
9. **GPT-2's 124M are never added up.** LLM-2 said the token table is 31%; LLM-3 counted one head;
   this deck can close the books: 12 blocks x 7,087,872 = 85,054,464, plus the token table
   38,597,376, the position table 786,432 and the final norm = 124,439,808.
10. **The modern-recipe table needs sources or softer words.** "Pre-LN: trains stably, no warmup"
    (Xiong et al. 2020 show it can train without warmup; most recipes still warm up), "SwiGLU:
    better quality per parameter" (Shazeer 2020), "the rest is engineering that moved a few percent
    and then stayed" (unsourced). Keep the frame: LLM-14 is optional, so this is the only place
    students meet RMSNorm and SwiGLU.
11. **Two endings** ("Next on the map" and a [plain] "You can now build one"). One. And "you can
    now build one" with no code: one NumPy block (norm, attention, add, norm, MLP, add) would back
    it (the deck's one snippet).
12. **Missing one line: a block maps n x 768 to n x 768** - same shape in, same shape out, which is
    why it stacks.
13. **Small:** the subtitle says "position"; notation W^Q -> W_Q to match LLM-3 (rework list);
    "Now stack it" quotes BERT and GPT-3 depths - use GPT-2's 12.

**Kept:** the multi-head idea (h smaller views, the same parameter budget), the head-entropy
measurement idea, the suspicious "kinds of head" table, the residual stream as "the real object",
the causal mask figure, the modern-recipe frame, the stack.

**Interviewed 2026-10-05, round 1:**

- **Scope: restructure** (like LLM-3).
- **The MLP: a full section on the 3b1b chapter-7 spine** - where facts live, up-projection rows
  as questions, ReLU as an AND gate, down-projection columns as directions - plus one by-hand
  neuron on the creature toy and the 66.6% number.
- **Cold open: a fact completion ("X plays the sport of ___")**, but "a more interesting person"
  than 3b1b's Michael Jordan. Candidates tested on the cached GPT-2 before round 2, since the slide
  will state what GPT-2 completes.
- **By-hand frames: all four** - two heads asking two questions (concatenate, then W_O); an MLP
  neuron on creature; LayerNorm on a 4-number vector; the causal mask on LLM-3's creature grid.

**Round 2** (after testing fact prompts on the cached GPT-2 small, 2026-10-05; top next word and
its probability: Tiger Woods -> golf 61.5%, Ronaldo -> football 48.5%, Carlsen -> chess 39.8%,
Gretzky -> hockey 35.6%, Messi -> football 35.1%, Kasparov -> chess 23.9%, Einstein -> physicist
23.8%, Agassi -> tennis 11.7%, Michael Jordan -> basketball 13.5%; wrong: Mkhitaryan -> hockey
9.8% (football 4.9%), Marie Curie -> Nobel Prize in Literature 17.2%, Aronian -> "the",
Aznavour -> writer):

- **Cold open: a stored fact and a missing one** - picked "Kasparov right, Mkhitaryan wrong", then
  (instructor, same turn) "I want Elliott instead of Kasparov": **Elliott Smith (the musician)
  for the stored fact**, Mkhitaryan -> hockey 9.8% (football 4.9%) for the missing one. What GPT-2
  completes for Elliott Smith is tested before the outline fixes the prompt.
  **Tested:** GPT-2 does not know Elliott Smith ("was a famous American" -> writer 4.6%, actor
  3.1%, singer 2.9%; "played the" -> "role" 9.4%, guitar not in the top 5) nor Ella Fitzgerald
  (instructor's next try: "was a famous American" -> actress 15.4%, writer 6.8%, singer 6.6%).
  **Final (instructor's pick):** Messi -> football 35.1% (stored); Mkhitaryan -> hockey (missing);
  Ella Fitzgerald and Elliott Smith as the music pair, both missed but to different degrees
  (singer 6.6% vs 2.9%): facts are stored, half-stored or missing.
- **Outline:** `LLM4_transformer_block_OUTLINE.md` (drafted 2026-10-05, 39 frames). Its two open
  questions were answered the same day: the neuron writes a new fifth slot, "cute"; head 2 asks
  "what colour?".
- **Real heads: GPT-2, one local run** (the cached model, the creature sentence); the e5 figures go.
- **Norm order: GPT-2's** (x <- x + f(LN(x))); the 2017 order as a footnote; the block diagram
  redrawn.
- **Extras: GPT-2's 124M added up; the superposition teaser** (3b1b's nearly-perpendicular / JL
  stills). Not picked: a NumPy block (so the "You can now build one" ending goes, one ending
  stays); moving the shuffle proof into the block section (it stays one frame, at the end of the
  many-heads section).
- **Built 2026-10-05** (41 pages; the outline's "Built" note lists what changed while building).
  Student review pending (needs the instructor's OK).

**LLM-5 critique (2026-10-05, instructor away: "criticize the rest ... for the stuff that's a
no-brainer, go ahead and do it yourself").** The July L26, 25 pages: the family (encoder, decoder,
encoder-decoder, cross-attention), how it learns, why it won and what it costs.

1. **Its map box is the LM head, and the head gets one frame** (a 3b1b still inside "How it
   learns"). Missing: the head by hand on GPT-2 - final norm, then a dot product of the last vector
   with every row of the token table (GPT-2 ties the two; LLM-2 taught the tying), 50,257 logits,
   softmax, and the map's top-5 ("floor" 7.6%, `results/roadmap_gpt2.json`). LLM-6 then picks the
   distribution up.
2. **"How it learns" overlaps three decks.** "The objective you have already seen" (next-token
   cross-entropy; every position a training example) is now LLM-4's mask frame and LLM-7's
   subject; "Autocomplete is not a chatbot" is LLM-9's opening. Pick homes.
3. **Three figures render at 4.0-4.6 pt** (cross-attention, three architectures, the quadratic
   plot): drawn 7-9.6 in wide in `py_src/l25_l26_figs.py`. Redraw at slot size in the
   restructure, as LLM-4's were.
4. **The cold open has no hook** - a statement ("who is allowed to look at whom?"), no predict,
   no example. Candidate: the three masks side by side, "which of these can write text?".
5. **Two endings** (Next on the map, then a [plain] "That is the T in GPT"), as LLM-4 had.
6. **Session question.** After the LLM-4 restructure, LLM-5's own material is the family,
   cross-attention, the LM head and the cost and inductive-bias frames. A full session, or merged
   (the family and the cost into LLM-4, the head into LLM-6) to buy back a session (section 8:
   "something gets cut whatever is chosen").

**Applied without an interview (no-brainers, 2026-10-05):** the retrieval-chapter line is now a
forward pointer (RAG comes after LLM-11); "bolt position on in LLM-4" -> LLM-2; the inductive-bias
table's stale chapter numbers removed ("CNN (ch6)", "RNN (ch7)" predate the 12_cnn / 13_rnns
names); in-context learning softened from "small models cannot, past a scale they can" to "small
models do it badly, large ones well; sudden or gradual is debated (LLM-8)"; GPT-2's own crossover
added (768 x 4 = 3,072, three times its window); the FlashAttention pointer -> LLM-14; the recap's
"the difference is the mask, not the block" made exact (the mask, plus cross-attention in the
decoder); two map captions shortened to one line. Recompiled: 0 errors, 0 overflow (it had 3).

**Build plan (after the interview):** restructure like LLM-3 and LLM-4 - the LM head by hand on
GPT-2 (the real top-5 is already local), the three July figures redrawn at slot size in a new
`py_src/head_figs.py`, one ending, "How it learns" placed as decided. Interview questions:
`INTERVIEW_QUEUE.md`, section LLM-5.

#### LLM-6 Decoding: from logits to text

LLM-5 ended with a distribution over the vocabulary ("the choosing is a separate decision ... later
in this chapter"); L21 promised that "how to steer that choice is a topic of its own". Every word
a chatbot writes is a choice from that distribution. The map's last box.

**Critique of the first draft (2026-10-04, before the interview).** The draft was a port of the
survey deck's method list (greedy, beam, temperature, top-k, top-p, min-p, contrastive,
structured, speculative): a catalogue, which is what LLM-2's student review flagged as the
weakest stretch of that deck. Specific problems:

1. **It re-teaches the loop.** L21 ("sample, feed back, repeat") and LLM-5 (logits -> softmax ->
   sample) both teach it. Here it is one recap frame; only the stop conditions are new.
2. **No running example.** Every knob can act on one distribution: the map's own "The cat sat on
   the" (floor 7.6%, bed 6.5%, couch 5.4%, ground 5.2%, edge 4.8%; already in
   `results/roadmap_gpt2.json`). *Revised 2026-10-05 (instructor: "the idea is to teach the
   concept; if there is no harm from synthetic or demonstrative values, that is perfectly
   enough"):* toy, demonstrative numbers are the default for this deck; no new measurement runs,
   no model downloads. The one real output is the cold open's GPT-2 loop (local, already cached).
3. **The "why sample at all" is missing.** "Greedy repeats" is a symptom. The cause is Holtzman
   et al.'s observation: human text is not the most probable text, and maximising probability
   produces bland, looping text. Show it with the paper's figure or a toy, not a new measurement.
4. **Min-p was presented as a 2024 advance.** It was an ICLR 2025 oral, and a 2025 critique
   (arXiv 2506.13681, "Min-p, Max Exaggeration") found it does not beat the baselines once
   hyperparameters are controlled, with reporting problems in the original. Teach it, with that
   footnote, or cut it to one line.
5. **It misses what practitioners actually trip on:** temperature 0 is not deterministic on a
   server (batching changes the arithmetic: Thinking Machines, Sept 2025, 80 distinct completions
   in 1,000 runs of Qwen3-8B at temperature 0), and providers now restrict the knobs (Claude 4.x
   rejects temperature and top_p together; reasoning models often fix the sampling entirely).
6. **"Typical API parameters" was a survey table** that goes stale. Replace it with the two
   facts in point 5, dated and sourced.
7. **Tool names (GBNF, Ollama structured outputs) baked into the slides** before D4 is decided.
   The idea (mask the tokens that would break the format, at every step) is tool-free. It has a
   tokenization twist worth a frame: a format boundary can fall inside a token (LLM-1 callback).
8. **The hands-on bundled a technology choice** (Ollama) that is still an open decision (D4), and
   LLM-1's hands-on was "not in the slides". Ask, do not assume.
9. **"33 frames -> ~25"** - frame count is never a constraint (`ml/SLIDE_STYLE.md`). Removed.
10. **Speculative decoding was parked in LLM-14, which is optional.** If LLM-14 never runs, the
    one decoding trick every serving stack uses is never taught. It only needs LLM-4's causal mask
    (check k drafted tokens in one pass), so it can live here.

**Proposed spine (for the interview, not decided).** One question - "the model gave a
distribution; which token do you write?" - answered in four moves, on toy numbers and the map's
real top-5 (no new measurements):

- **Recap + stop.** One frame: the loop and the distribution, then what is new: when to stop (the
  end-of-text token from LLM-1, max tokens, stop strings).
- **Pick the most likely.** Greedy (the cold open's GPT-2 loop), log-probabilities (why logs: a
  product of many probabilities underflows - a toy calculation), beam search (the source deck's
  B = 2 worked example, length normalisation), and the probability trap: human text is less
  probable than beam text (Holtzman et al.'s figure). Where beam search is still used
  (translation, speech) - to verify.
- **Sample.** Temperature (the same division as temperature scaling in [14] and ch11's 45, shown
  on the map's top-5), top-k and why a fixed k fails (two toy distributions, one peaked, one
  flat), top-p, min-p
  with the critique, the order the knobs apply in, repetition penalties (two different
  definitions: a logit penalty in Hugging Face / llama.cpp vs the additive presence and frequency
  penalties in OpenAI's API - to verify).
- **Constrain.** Structured output by masking: at each step only tokens that keep the output valid
  JSON (or any grammar) survive. The token-boundary twist. Where it matters next: tool calls
  (agents, L44).
- **Real systems** (choose in the interview): temperature 0 is not deterministic; providers
  restrict the knobs; speculative decoding (draft with a small model, verify k tokens in one pass,
  same output distribution); watermarking in the sampler (SynthID-Text, Nature 2024: Google's
  production scheme changes only the sampling step).
- **Possible Armenian thread:** a byte-level model can stop in the middle of a letter, so sampled
  bytes need not be valid UTF-8 (LLM-1's decoding trap, now at generation time). Explain with
  LLM-1's ayb example (D5 A1); no new measurement.

**Cut:** contrastive search (a Hugging Face-specific method that did not catch on).

**Source:** dl4nlp `04_decoding_strategies.tex` (port the worked examples: temperature, beam
B = 2, the comparison table) and its speculative-decoding frames if they move here.

**To verify before building:** Holtzman et al. (ICLR 2020) and the "human text is not most
probable" figure; where beam search is still the default; repetition-penalty definitions (HF,
llama.cpp, OpenAI); the Thinking Machines numbers against the primary post; Anthropic's and
OpenAI's current sampling-parameter rules from their own docs (the Claude 4.x rule above comes
from GitHub issues, i.e. secondary); SynthID-Text paper details; whatever runtime D4 picks
supports min-p and grammar-constrained output.

**Hands-on:** open (D4, and whether it belongs in the slides at all).

**Interviewed 2026-10-05** (after the critique above):

- **Cold open: GPT-2 loops** (instructor's pick over "7.6% sure" and "temperature 0, 80
  answers"): always take the most likely token and GPT-2 gets stuck repeating itself, measured
  live. The deck answers "so why not just take the top token?".
- **Real systems: temperature 0 is not deterministic; speculative decoding (moved here from
  LLM-14, which keeps only multi-token prediction); watermarking (SynthID-Text).** Not picked:
  providers restricting the knobs.
- **Min-p: taught, with the 2025 critique** (one frame on the idea, one line on the ICLR 2025
  oral and the fair-comparison result).
- **Hands-on: not in the slides** (like LLM-1). D4 stays open; nothing in this deck depends on a
  runtime choice.
- **Outline:** `LLM6_decoding_OUTLINE.md`. **Built 2026-10-05:** `LLM6_decoding.tex` (53 pages after the student review), figures from `py_src/decoding_figs.py` (toy numbers; GPT-2 only for the cold open, via `--gpt2`).
- **Open (2026-10-05):** the three student-review suggestions not applied (cut the draft-sources
  frame; a grammar state table for constrained decoding; a mechanism for self-reinforcing loops)
  are questions in `INTERVIEW_QUEUE.md`, section LLM-6. After the renumber the deck's labels were
  checked: its forward pointers (LLM-7 pretraining, LLM-9 chat tokens, LLM-11 hallucination,
  LLM-14 multi-token prediction) are right.

#### LLM-7 Pretraining: from the internet to a base model

"Predict the next token" on trillions of tokens. What comes out is not an assistant - it is a
document simulator.

**Critique of the first draft (2026-10-05, before the interview; the draft is in git history).**
The draft was mostly a history of objectives (n-grams -> Bengio -> word2vec -> ELMo -> CLM/MLM/T5
-> GPT-1/2/3), with the data in one bullet. The title promises "from the internet to a base
model"; the bullets deliver a model timeline. Specific problems:

1. **Two lectures in one, and the wrong one leads.** About eight models at one or two frames each
   is a catalogue, the pattern both student reviews flagged. By now students have the transformer
   and the objective (LLM-5). The open question on the map is "where do the 124M numbers come
   from?", and the draft answers it in one bullet.
2. **It re-teaches the name-inventor practical.** Perplexity = e^loss ("torn between k letters")
   and a bigram baseline are in `ml/11_neural_networks/47_name_inventor_solution.ipynb`, section
   6; section 3 did not list them (fixed 2026-10-05). Here perplexity is a callback plus what is
   new: per-token perplexity is not comparable across languages even under one tokenizer,
   because an Armenian token carries less text (LLM-1's token tax). Bits per byte (or per
   character) is the fair unit.
3. **"CLM vs MLM (LLM-5 recap)" recaps something LLM-5 never taught.** LLM-5 shows the three
   architectures (the encoder sees both ways); no deck shows the masked-LM objective. Section 3
   said otherwise (fixed). MLM is new here.
4. **"Asked a question, it writes more questions" is already taught twice.** LLM-5's "Autocomplete
   is not a chatbot" (the password example) states it, and LLM-9 opens with it. Here at most one
   real GPT-2 output as the bridge to LLM-9, not a section.
5. **The memorization demo would fail on the model we have.** GPT-2's WebText dropped every
   Wikipedia page ("We removed all Wikipedia documents from WebText", GPT-2 paper, section 2.1,
   checked 2026-10-05). With the cached GPT-2, try text that is copied all over the web (a
   software licence, a famous speech); one run decides whether it recites anything, no download.
   The better lesson is *why*: memorization grows with how often a text is duplicated, which is
   the reason deduplication is in the data funnel (Carlini et al., Lee et al. - to verify). Demo
   and data become one story.
6. **The data is one bullet, but it is the lecture.** What a raw crawl looks like, each filter and
   what it removes (URL blocklist, text extraction, language ID, dedup, quality classifier, PII),
   on one source's numbers (FineWeb), and how a mixture is weighted (the local
   `smol_training_playbook`). "Synthetic data" without a concrete example is a buzzword: give it
   one example or cut it.
7. **The Armenian thread is missing where it is cheapest.** Language ID decides how much
   Armenian a model ever sees. One published number, the Armenian share of a web crawl
   (FineWeb-2 or Common Crawl's language statistics, read from their tables), explains LLM-1's
   token tax and the weak Armenian answers.
8. **The GPT-1 -> GPT-2 -> GPT-3 "lessons" overlap LLM-8 and LLM-10.** "Few-shot improves with
   model size" is LLM-8's story; the craft of few-shot is LLM-10's. Here one frame at most
   (fine-tune -> zero-shot -> in-context), or hand it over.
9. **Numbers that disagree.** "1.8T for Llama 2" is the Llama 3 paper's own figure ("about 15T
   multilingual tokens, compared to 1.8T tokens for Llama 2", local PDF, checked 2026-10-05);
   the Llama 2 paper and source deck 07 say 2T. Quote one, with its source. The source decks'
   compute tables (GPT-3 petaflop-days, "Mistral 7B ?") belong to LLM-8 or nowhere.
10. **No cold open, no by-hand frame, frame counts again** ("06 (22 -> ~6)", removed). The
    mechanics are computable: pack three toy documents into one stream with the end-of-text token
    (LLM-1), cut windows, count the predictions one window gives (LLM-5: every position is a
    training example).
11. **The hands-on needs a model download and two open decisions** (D4 runtime, D6 base model).
    LLM-1 and LLM-6 chose "not in the slides". Ask.

**Kept from the draft:** word2vec lives here (instructor, 2026-10-04: "word2vec stays in LLM-7";
#64, #68). LLM-2 shows the table and what GPT-2 learned; this lecture shows how word2vec trained
one with no language model around it.

**Proposed spine (for the interview, not decided).** One question: "LLM-6 used the
probabilities; where do the 124M numbers that produce them come from?" Toy numbers by default
(`ml/SLIDE_STYLE.md`); real numbers only for claims about real models and data, from papers or
the cached GPT-2.

- **The objective, as a family.** One recap frame (LLM-5), then "is next-token the only option?":
  one mask-pattern figure in LLM-4's causal-mask style, one row per objective - what is hidden,
  what is visible. Bigram (callback to 47), word2vec (a word from its window, order ignored; 2
  frames), GPT (everything to the left), BERT (15% hidden, both sides), T5 (whole spans). Why
  next-token won stays a one-line LLM-5 callback.
- **One training step by hand.** Toy documents -> one token stream with end-of-text tokens ->
  windows -> a window of 9 tokens gives 8 predictions -> the loss -> perplexity (the 47 callback)
  and the language caveat.
- **The data.** Crawl -> funnel -> mixture, with the Armenian share. Dedup and memorization as one
  story.
- **What comes out: a base model.** Loss falls and samples go from noise to plausible text
  (Karpathy's deep-dive notes, local); it recites what it saw many times (the cached GPT-2, one
  run); it continues documents instead of answering them (the bridge to LLM-9).
- **Optional, one frame:** GPT-1 fine-tune -> GPT-2 zero-shot -> GPT-3 few-shot.

**Cold open candidates:** (a) GPT-2 recites a licence or a speech word for word, if the one run
works; (b) "15T tokens is about N human reading lifetimes" (back of the envelope); (c) "what share
of the web is Armenian?" (one published number); (d) a raw web page before cleaning.

**Cut:** ELMo, NSP/SOP, contrastive learning (it belongs with the embedding models in RAG), the
model-size and compute tables (-> LLM-8), n-gram smoothing, synthetic data unless it gets an
example. If LLM-8 is merged in (section 8, short option), the objective family shrinks first.

**Sources:** dl4nlp 07 "Pre-training objectives" (CLM, MLM, the comparison), 01's word2vec frames,
05 "Perplexity" (the caveats); Karpathy's "Deep Dive into LLMs" notes
(`sources/dl4nlp/_yt_videos/karpathy_deep_dive_llms/`: the FineWeb funnel, the base-model demo);
`ml/llm_training/materials/blogs/01_smol_training_playbook.html` (mixtures, data ablations); the
Llama 3 paper (local) for one data-scale number.

**To verify before building:** FineWeb's funnel numbers (paper or blog); the Armenian share of
FineWeb-2 or Common Crawl; Carlini et al. (memorization vs duplication and model size) and Lee et
al. 2022 (dedup reduces verbatim recall); BERT's masking recipe (15%, 80/10/10); word2vec's
negative sampling if shown; whether the cached GPT-2 recites any duplicated text (one run).

**Hands-on:** open (D4, D6, and whether it belongs in the slides at all).

**Interview: not yet held** (it was interrupted on 2026-10-05 when the instructor switched to
attention). The questions are ready in `INTERVIEW_QUEUE.md`, section LLM-7. One addition from
building LLM-4: LLM-4 now owns "one pass, n training signals" (the mask frame), so LLM-7's
"one training step by hand" should count predictions per window and call back, not re-explain the
mask; and LLM-4's cold open already shows memorised vs missing facts (Messi vs Mkhitaryan), so
LLM-7's memorisation demo should be the *verbatim* kind (a much-copied text), not facts.

#### LLM-8 Scaling laws and emergence

- Power laws (Kaplan); C ~ 6ND; the training-memory bill (~16 bytes/param with Adam: weights,
  gradients, two moments, before activations - this number comes back in LLM-13).
- Chinchilla (~20 tokens/param); over-training for cheap inference (small models on far more
  tokens); the three eras of compute allocation.
- Emergent abilities, the mirage argument (the metric makes the jump), where the debate stands;
  grokking (`misc/grokking/`, Welch Labs material already extracted).
- Teaser: test-time compute as the second scaling axis -> L32g.
- **Sources:** dl4nlp 13 (18 frames) + 17 (18 frames), deduplicated; llm_training 09.
- **First merge candidate** into LLM-7 if sessions are short.

**Critique (2026-10-05, before an interview).**

1. **Two source decks make a catalogue.** dl4nlp 13 (16 frames: Kaplan, Chinchilla, Gopher, three
   eras, a playbook) and 17 (16 frames, four of them on emergence in ant colonies, cells and
   physics). Cut the nature analogies; one question carries the session: *"you have a fixed
   compute budget - how big a model, how much data?"*
2. **Everything here is computable on toy or published numbers - no runs needed.** C = 6ND by
   hand; with Chinchilla's D = 20N, C = 120 N^2, so a 10^21 FLOP budget gives N of about 2.9B and D
   of about 58B (worked frame). The loss curves come from the published fit L(N, D) = E + A/N^alpha
   + B/D^beta (Hoffmann et al. 2022; constants to verify), drawn by our script. Over-training as a
   ratio: Llama 3's 8B model on about 15T tokens is roughly 1,900 tokens per parameter against
   Chinchilla's 20 (the local Llama 3 paper; verify the 8B's token count).
3. **The mirage argument has a perfect toy:** per-token accuracy p improving smoothly, while
   exact match on a 5-token answer is p^5 - flat, then a "jump". By hand, then the debate (Schaeffer
   et al. 2023 vs the cases that stay sharp). No model needed.
4. **The 16 bytes per parameter** (bf16 weights 2 + grads 2 + fp32 master 4 + Adam's two moments
   8) is a by-hand frame that LLM-13 calls back; for GPT-2 small: 124M x 16 = about 2 GB.
5. **Grokking** is about training dynamics (late generalisation), not scale; one frame as "another
   kind of sudden" or cut. The Welch Labs material is local (`misc/grokking/`).
6. **Homes:** GPT-3's "few-shot improves with size" plot lives here (LLM-7's critique point 8);
   test-time compute is a one-frame teaser here, the craft in LLM-10, the maths in L32g.
7. **"Three eras of compute allocation"** (source frame) is a framing without a source; drop it
   or source it.

**Build plan:** a small `scaling_figs.py` (the Chinchilla fit, the p^5 toy, the 6ND budget split),
everything asserted; no model runs. Questions: `INTERVIEW_QUEUE.md`, section LLM-8.

#### LLM-9 Post-training: from base model to assistant

- Why a base model is not a chatbot (callback: the LLM-7 demo).
- SFT on conversations; chat templates and special tokens (ChatML; the cost of adding tokens - from
  dl4nlp 03); "the assistant is a statistical imitation of a human labeler" (Karpathy).
- Instruction tuning at scale (FLAN); quality over quantity (LIMA, 1,000 examples).
- Preference tuning **at idea level** (D3): comparisons -> reward model (Bradley-Terry in one line),
  the RLHF pipeline as a picture, reward hacking, DPO as "skip the RL". Full math: L32f.
- Verifiable rewards and "thinking" models with a budget dial (Qwen3) as a teaser -> L32g.
- Distillation: SFT on a bigger model's outputs (the R1 distills).
- **Sources:** dl4nlp 07 sec 2-4 (~15 frames); llm_training 18 (FLAN), 04 (LIMA), 10
  (InstructGPT), 08 (Qwen3); distillation frames from dl4nlp 11.
- LoRA/QLoRA are **not** here any more - they get their own lecture (LLM-13).

**Critique (2026-10-05, before an interview).**

1. **Ten topics, one per paper** (SFT, chat templates, FLAN, LIMA, reward model, RLHF, reward
   hacking, DPO, verifiable rewards, distillation): a paper tour. Proposed spine - three ways to
   change what a base model prefers: **imitate** (SFT; distillation is imitating a bigger model),
   **compare** (preferences: reward model, RLHF, DPO), **check** (verifiable rewards). FLAN, LIMA,
   InstructGPT become evidence inside the spine, not stops on a tour.
2. **The cold open can be measured locally.** LLM-5 asserts that a base model answers a question
   with more questions; one run of the cached GPT-2 on "How do I reset my password?" shows what it
   really writes. Then the same prompt inside a chat template: the "Assistant:" turn.
3. **Chat templates need a real one.** GPT-2 has none; quote a published template (ChatML or a
   current model's) from its documentation, verified at build - and LLM-1's special tokens get
   their payoff.
4. **Bradley-Terry is one line and a toy:** two answers, rewards 2.0 and 0.5, P(first preferred) =
   sigma(1.5) = 0.82. By hand. Reward hacking with one sourced example.
5. **D3 stands:** idea level here, the maths in L32f/L32g (RL comes after this chapter).
6. **Overlaps:** system prompts (LLM-10), SFT in practice with LoRA (LLM-13), reasoning models
   (LLM-8 teaser, L32g). This deck owns "why a base model is not a chatbot".

**Build plan:** `posttrain_figs.py` (Bradley-Terry toy, the three-verb pipeline figure); one
`--gpt2` run for the cold open. Questions: `INTERVIEW_QUEUE.md`, section LLM-9.

### Part B - Using it, and knowing when it is wrong

#### LLM-10 Prompting and in-context learning

- Zero-shot, few-shot, and how fragile few-shot is (example order, format, label balance).
- Chain of thought, zero-shot CoT, why it works ("models need tokens to think"); self-consistency;
  reasoning models do this internally, which changes the advice.
- System prompts; structured output (callback: constrained decoding, LLM-6); context engineering -
  what goes into the window and in what order.
- Prompt injection, direct and indirect (indirect injection sets up the agents chapter).
- **Sources:** dl4nlp 08 (22 frames; Tree of Thoughts cut, ReAct -> L44); llm_training 14 (CoT
  half).
- **Hands-on:** few-shot fragility - permute example order and labels, measure the accuracy
  swing. *(Fixed 2026-10-05: this said "on the local model from LLM-6", but LLM-6's hands-on was
  "not in the slides", so no local model exists yet; it depends on D4/D6, or on the cached GPT-2,
  see the critique.)*

**Critique (2026-10-05, before an interview).**

1. **The source is a 2022-23 technique list** (zero-shot, few-shot, CoT, zero-shot CoT,
   self-consistency, ToT, ReAct, best practices, system prompts, structured output, injection).
   Much of that craft aged: instruction tuning and reasoning models made "let's think step by step"
   matter far less. Proposed spine: **the prompt is the context, and the context shapes the
   distribution** (LLM-6): what you put in (examples, instructions, documents), in what order, and
   what an attacker can put in.
2. **Few-shot fragility can be measured on the cached GPT-2** - a claim about real models, local:
   a 4-example sentiment prompt, all 24 orderings, the accuracy spread (the setting of Zhao et al.
   2021, "Calibrate before use", which used GPT-2 and GPT-3; verify). About 24 x 50 forward passes
   of GPT-2 small: a few minutes on CPU, one process - state the cost before running.
3. **Chain of thought** gets its *why* (tokens are compute: more tokens, more serial steps through
   the 12 layers - LLM-4's stack), then one line on reasoning models (LLM-8 teaser, L32g).
4. **Prompt injection** overlaps the agents chapter (L44): here the idea and one example (the
   context is untrusted text), the defences in L44.
5. **"Context engineering"** needs a concrete frame or goes: e.g. where in a long context a fact
   is found best (lost-in-the-middle, Liu et al. 2024 - verify) - published numbers, no run.

**Build plan:** `prompting_figs.py`; one `--gpt2` measurement run for the ordering spread (asked
first, with its cost). Questions: `INTERVIEW_QUEUE.md`, section LLM-10.

#### LLM-11 Evaluation and hallucination

- Why evaluation is hard; intrinsic vs extrinsic; BLEU / ROUGE in 2-3 frames and their limits;
  BERTScore.
- LLM-as-judge (pointwise vs pairwise; position, verbosity and self-preference biases).
- Benchmarks (knowledge, reasoning, code, agents), contamination, leaderboards and Goodhart.
- Hallucination: taxonomy (factuality vs faithfulness), why it happens - including Kalai et al.
  2025: training and evaluation reward guessing over "I don't know" (arXiv 2509.04664); detection
  (SelfCheckGPT, semantic entropy); mitigation overview.
- **Cliffhanger:** give the model the documents -> RAG (L41).
- **Sources:** dl4nlp 05 (27 frames; perplexity -> LLM-7, MAP/nDCG -> L43 owns them) and 09
  (26 frames; RAG mitigation -> ch17, duplicate self-consistency dropped).

**Critique (2026-10-05, before an interview).**

1. **Two source decks of metrics and methods** (BLEU, ROUGE, BERTScore, judges, benchmarks,
   contamination; a hallucination taxonomy, SelfCheckGPT, semantic entropy, CoVe, citation
   systems). Proposed spine - **how do you know an answer is right?** Compare it with a reference
   (BLEU/ROUGE, by hand on two toy sentences, and why that fails for open answers); ask another
   model (judges and their biases); use a test set (benchmarks, and contamination - LLM-7's
   memorisation again); and when the model is confidently wrong (hallucination).
2. **The best demo is already half-built:** LLM-4's cold open. Sample GPT-2 ten times at
   temperature 1 (LLM-6) on "Lionel Messi plays the sport of" and on the Mkhitaryan prompt: if the
   answers agree for the stored fact and scatter for the missing one, that is the intuition behind
   SelfCheckGPT and semantic entropy, on a model the students know. Local, a few seconds; measure,
   then claim.
3. **Kalai et al. (2025, arXiv 2509.04664)** - benchmarks that score "I don't know" as wrong reward
   guessing - is the session's strongest idea: a toy scoring table by hand (guess vs abstain under
   0/1 grading vs a penalty for wrong answers). Verify the paper's framing before building.
4. **LLM-as-judge needs a paid API** for a live demo - ask before any spend; published bias numbers
   are the default.
5. **Frame counts** were targets ("27 -> ~16"); removed (frame count is never a constraint).

**Build plan:** `eval_figs.py` (the BLEU toy, the guess-vs-abstain table); one `--gpt2` sampling run
for point 2. Questions: `INTERVIEW_QUEUE.md`, section LLM-11.

**Then the built chapters continue the story:** RAG (`ch17`, L41-L43, 3 lectures, far deeper than
dl4nlp 12) and Agents (`ch18`, L44, replaces dl4nlp 14).

### Part C - Run it and adapt it yourself

#### LLM-12 Quantization: an LLM on your laptop

An 8B model is 16 GB in BF16. The laptop has 16 GB in total. Make it ~5 GB and lose almost nothing.

- **Memory arithmetic:** params x bytes; KV-cache bytes per token (2 x layers x kv_heads x
  head_dim x bytes) - context costs memory too, and the KV cache can be quantized as well (flag
  names: check llama.cpp docs at build). Decoding is memory-bandwidth bound, so
  tokens/s <= bandwidth / bytes read per token (dense model, batch 1; for MoE only the active
  experts count): fewer bytes is also *faster*.
- **Number formats:** FP32 / FP16 / BF16 (range vs precision, from llm_training 19); FP8 E4M3/E5M2
  (DeepSeek-V3 trains in it, llm_training 17); INT8 / INT4; FP4 - MXFP4 (OCP standard, blocks of
  32, shared E8M0 scale) vs NVFP4 (blocks of 16, FP8 scale + FP32 tensor scale).
- **The math, by hand:** absmax and zero-point (affine) quantization; rounding error; per-tensor vs
  per-channel vs block-wise, and why blocks win.
- **Outliers:** a few huge activation dimensions (LLM.int8) - why weight-only quantization is the
  common path.
- **Post-training quantization:** round-to-nearest -> GPTQ (second-order error correction, layer by
  layer) -> AWQ (protect channels with large activations). llama.cpp's importance matrix
  (`llama-imatrix`) is the same idea for GGUF.
- **Quantization-aware training:** train with fake quantization so the model learns to live with
  it. Real examples: Gemma 3 QAT checkpoints (2025, shipped as Q4_0 GGUF for llama.cpp/Ollama);
  gpt-oss ships its MoE weights in MXFP4, which is why the 20B model fits in 16 GB.
- **Extreme end:** 1.58-bit ternary weights (BitNet), trained from scratch - one frame.
- **GGUF decoded:** one file with weights + tokenizer + chat template + metadata; the k-quant
  names (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K, IQ-quants) and what K/M/S mean; bits per weight
  for each - take from the llama.cpp quantize docs at build time.
- **Measured on our laptop** (the lecture's own frame): one small model at F16 / Q8_0 / Q4_K_M /
  Q3_K_M / Q2_K -> file size, peak RAM, tokens/s (`llama-bench`), perplexity on Armenian and
  English text (`llama-perplexity`). Needs a compute OK before running (machine limits).
- **The local stack (as of 2026-10-02)** - see the table right after this list.
- **Gotchas (verified):** Ollama does not quantize GGUF on import - quantize first with
  `llama-quantize`. FP4 vs Q4_K_M quality is unsettled as of spring 2026.
- **Sources:** dl4nlp 11 (memory arithmetic + the ~7 quantization frames), llm_training 12 (block
  quantization, NF4), 19 (float anatomy), 17 (FP8).
- **Homework:** the quantization ladder on your own laptop - table + one plot.

**Critique (2026-10-05, before an interview).**

1. **The most concrete plan in the chapter, and the most tool-bound.** llama.cpp, Ollama, GGUF
   quant names and the local-stack table are perishable and wait on D4. Teach the ideas first
   (bytes, formats, blocks, outliers, rounding vs error correction, training-aware), the tools in
   one dated frame.
2. **The measured ladder needs downloads** (a small model's GGUF levels, gigabytes) and minutes of
   CPU - against the no-download default, though here the real measurement *is* the point. A
   no-download alternative with the same lesson: **quantize GPT-2 small's own weights in NumPy**
   (absmax, then block-wise, at 8/4/3 bits) and measure its perplexity on a fixed text with the
   cached model, before and after. Real, local, seconds to minutes.
3. **By-hand frames are easy on toys:** absmax on four numbers; one outlier ruining per-tensor
   scaling and block-wise rescuing it; tokens/s <= bandwidth / bytes per token with this laptop's
   memory bandwidth (measure or look up - verify).
4. **Memory arithmetic on GPT-2's real config:** 124,439,808 x 4 bytes = 498 MB in fp32; KV cache
   2 x 12 layers x 768 x 2 bytes = 36,864 bytes per token in fp16, so 37.7 MB for a full 1,024
   window. Then the same formula on one current model's config (verify).
5. **Homes:** KV-cache *memory* here; KV-cache *design* (MQA, GQA, MLA) in LLM-14; NF4 here as a
   format and in LLM-13 as QLoRA's storage.

**Build plan:** `quant_figs.py` (the toys, GPT-2's memory numbers, and - if picked - the GPT-2
quantization perplexity run). Questions: `INTERVIEW_QUEUE.md`, section LLM-12 (includes D4).

The local stack, as of 2026-10-02 (section 9 has the sources):

| Tool | What it is | When to reach for it |
|---|---|---|
| llama.cpp | C/C++ engine, GGUF, CPU / CUDA / Metal / Vulkan / SYCL; `llama cli`, `llama serve` (OpenAI-compatible); standalone `llama-quantize`, `llama-perplexity`, `llama-bench` | Full control, measuring, odd hardware |
| Ollama | One-command runner + model library; Modelfile; OpenAI- and Anthropic-compatible APIs; MLX backend on Apple Silicon (2026); cloud models | Easiest start; apps that need a local API |
| LM Studio | GUI over the same GGUF models (features: verify at build) | Non-programmers |
| MLX | Apple's array framework, native on M-series | Macs |
| vLLM / SGLang | GPU serving engines, built for throughput across many users | Serving, not laptops -> LLM-14 |
| HF transformers | Earlier GGUF loading dequantized to float; since Sep 2026 (main branch, Metal first) it runs the quantized llama.cpp kernels | Staying in Python |

#### LLM-13 Fine-tuning on a budget: LoRA and QLoRA

Full fine-tuning a 7B model needs ~112 GB (16 bytes/param, LLM-8). LoRA trains under 1% of the
parameters; QLoRA keeps the frozen base in 4 bits. Result: a free Colab T4.

- **Should you fine-tune at all?** The ladder: prompt -> RAG -> fine-tune (callback: L43 "when not
  to use RAG"). Fine-tuning is good at format, style, narrow tasks; new facts are better served by
  RAG. Continued pretraining as the third option, e.g. for a language (llm_training 06,
  Don't Stop Pretraining).
- The full fine-tuning bill: weights + gradients + Adam m and v.
- PEFT family in one frame (adapters, prefix tuning; LoRA won).
- **LoRA:** dW = BA with rank r; B = 0 at init so training starts exactly at the base model;
  alpha/r scaling; which matrices (QLoRA's finding: adapters on *all* linear layers are needed to
  match full fine-tuning - llm_training 12; library defaults differ, check at build); the
  parameter count by hand; merge after training -> zero extra latency; swap adapters per task. Why low rank works
  (intrinsic dimension, from llm_training 03).
- **QLoRA:** NF4 (quantiles of a normal - callback: block quantization in LLM-12), double
  quantization, paged optimizers; store in NF4, compute in BF16; matches 16-bit (Guanaco).
- Variants in one frame (DoRA and friends).
- **Recipe:** rank, targets, learning rate, epochs - take current defaults from the library docs at
  build time; data quality over quantity (LIMA callback); the chat template must match (LLM-9);
  evaluate before and after (LLM-11); catastrophic forgetting.
- **From adapter to laptop:** merge -> convert to GGUF -> `llama-quantize` -> Ollama
  (`FROM ./model.gguf`, `ollama create`). Unsloth does the first three in one call:
  `model.save_pretrained_gguf(dir, tokenizer, quantization_method="q4_k_m")`.
- **Gotcha (verified 2026-10-02 on Ollama `main`):** Ollama **no longer supports LoRA adapters** -
  `server/create.go:40` defines `errAdaptersUnsupported`, and `:903` rejects adapter-type GGUFs
  too. Merge the adapter first. Older tutorials using `ADAPTER` in a Modelfile are out of date.
- **Sources:** llm_training 03 (LoRA, 19 frames) and 12 (QLoRA, 21 frames); dl4nlp 07 sec "PEFT"
  (4 frames); dl4nlp 11's QLoRA frame is a duplicate - dropped.

**Critique (2026-10-05, before an interview).**

1. **D5 is still open** (PEFT + TRL vs Unsloth), and every code frame and the practical depend on
   it. The instructor's call; the deck can teach LoRA itself library-free and keep the library to
   one recipe frame.
2. **The parameter count by hand, on GPT-2's real shapes:** one layer's `c_attn` is 768 x 2,304 =
   1,769,472 numbers; a rank-8 LoRA on it is 8 x (768 + 2,304) = 24,576, 1.4%. Across all 12
   layers' attention and MLP matrices the share stays near 1% (compute at build).
3. **A local LoRA demo is possible:** a few hundred steps of rank-8 LoRA on the cached GPT-2 on
   CPU (for example, teaching it a fixed answer format) - minutes, one process. It makes "B = 0 at
   start, so step 0 is the base model" visible. The Colab practical (P2) stays the real
   fine-tune. Ask before running.
4. **Perishable claims:** Ollama's adapter support (verified 2026-10-02: gone on `main`), the
   Unsloth export call, library default targets - re-verify at build.
5. **Keep "should you fine-tune at all?" first** - the ladder prompt -> RAG -> fine-tune is the
   decision students will actually face.

**Build plan:** `lora_figs.py` (the BA picture at GPT-2's shapes, the parameter table); the local
demo only if picked. Questions: `INTERVIEW_QUEUE.md`, section LLM-13 (includes D5, D6).

### Part D - What changed since 2017

#### LLM-14 Anatomy of a 2026 LLM (likely 2 sessions)

Put LLM-4's 2017 block next to a 2026 model card. Almost every change since exists to save memory or
compute at inference time.

- **KV cache:** why decoding is memory-bound; prefill vs decode; KV memory arithmetic -> MQA ->
  GQA -> MLA (DeepSeek; also GLM-5, Kimi K2.5, Mistral Small 4).
- **Attention cost:** sliding-window + global layers (gpt-oss alternates full attention with a
  128-token window, plus learned attention sinks); hybrid linear attention (Gated DeltaNet + full
  attention in Qwen3-Next / Qwen3.5); mostly-Mamba-2 hybrids (Nemotron 3 Nano) - pays off L21's
  "recurrence comeback" teaser (since 2026-10-03 L21 keeps only two sentences; the Qwen3-Next /
  Qwen3.5 "three of every four layers" detail and the Mamba / xLSTM names land here).
- **Position:** moved to LLM-2 (#68), including RoPE, NoPE and long-context extension. A callback
  at most.
- **FFN cost - MoE:** router, top-k, load balancing, shared + fine-grained experts; total vs
  active parameters (gpt-oss-20b 21B/3.6B, Qwen3.5 397B/17B, Kimi K2.5 1T/32B). Why huge models
  run cheaply.
- **More than one token per step:** speculative decoding (draft + verify, lossless; taught in
  LLM-6 since 2026-10-05 - a callback here at most) and multi-token prediction heads (DeepSeek-V3; Gemma 4 MTP checkpoints, Apr 2026; Step 3.5 Flash).
- **FlashAttention** (exact, IO-aware) in 2 frames; **serving** many users: continuous batching,
  PagedAttention.
- **The landscape, dated** (re-verify before delivery): MoE with small active counts everywhere;
  small on-device models (Gemma 4 E2B/E4B, the 0.8B Qwen3.5 in llama.cpp's README); native
  multimodality; thinking modes with budgets; agentic coding as the benchmark everyone chases;
  diffusion language models as the non-autoregressive alternative (forward pointer to the
  diffusion chapter).
- **What did not change:** residual stream, attention, next-token training. Raschka's conclusion
  from the Jan-Feb 2026 releases: performance comes mostly from data and training recipes, not
  architecture.
- **Sources:** dl4nlp 11 (KV cache, MQA/GQA, FlashAttention, continuous batching), 10 (MoE, 20
  frames), 16 (long context, 22 frames; its position frames went to LLM-2); llm_training 13 (MoE),
  15 (FlashAttention), 17 (DeepSeek-V3: MLA, MTP); `misc/dl4nlp/_reference_welchlabs_mla/`
  (MLA video frames); Raschka's LLM Architecture Gallery for figures to redraw.

**Critique (2026-10-05, before an interview).**

1. **Optional and two sessions - so ask what is lost if it never runs.** MoE is in nearly every
   current large model, and the KV cache explains why long chats are expensive. If LLM-14 is cut,
   one MoE frame and one KV-cache frame need a home elsewhere (LLM-4's "modern block recipe" and
   LLM-12's memory arithmetic are the natural ones).
2. **A dozen techniques is a catalogue** unless each one gets its bill by hand - the spine is right
   ("every change saves memory or compute at inference"): KV bytes per token for MHA vs GQA vs MLA
   from one real config each; MoE total vs active parameters from a real model card; FlashAttention
   as "the same numbers, fewer trips to slow memory".
3. **Every model fact here is perishable** (verified 2026-10-02): re-verify all before delivery,
   preferring `config.json` files over announcements (as LLM-2 did).
4. **Already moved out:** position (LLM-2), speculative decoding (LLM-6). Callbacks only.
5. **Frame counts** were targets ("20 -> ~6"); removed.

**Build plan:** decide after D1. Questions: `INTERVIEW_QUEUE.md`, section LLM-14.

---

## 6. Practicals

| Practical | After | What students do | Compute |
|---|---|---|---|
| **P1 - A tiny GPT that speaks Armenian** | LLM-7/8 | Sequel to the ch11 surname inventor: same spirit, now a transformer. Tokenizer (char or the LLM-1 BPE) -> CLM pretraining -> perplexity -> decoding knobs | CPU, minutes (estimate - measure when built) |
| **P2 - Your own model, on your own laptop** | LLM-13 | QLoRA fine-tune a small model on Colab (T4) on a small Armenian instruction set -> merge -> GGUF -> quantize -> run it in the D4 runtime; evaluate before/after | Colab GPU (`colab-gpu` skill) + laptop |
| HW - The quantization ladder | LLM-12 | Size, RAM, tokens/s, perplexity per quant level, on their own machine | Laptop, small model |

P2's open choices: D5 (library), D6 (base model), and the instruction data (needs a source - not
chosen yet).

---

## 7. Source map - where every existing deck goes

Copies of every deck this chapter draws on sit in `sources/` (2026-10-04, DECISIONS #66): the
dl4nlp decks except the three retired ones (12, 14, 18), and the llm_training decks except 01, 02
and 11, which feed the RL chapter. The originals stay where they were. Port from the copies.

### `misc/dl4nlp/` (18 decks)

| Deck | Destination |
|---|---|
| 01 pre_transformer | Covered by L20/L21/LLM-3; n-grams, word2vec, ELMo -> LLM-7 |
| 02 transformers | Already absorbed into LLM-3 to LLM-5; its position frames -> LLM-2 |
| 03 tokenization | LLM-1; chat tokens -> LLM-9 |
| 04 decoding_strategies | LLM-6, including speculative decoding (moved back from LLM-14, 2026-10-05) |
| 05 evaluation | LLM-11; perplexity -> LLM-7; MAP/nDCG stay with L43 |
| 06 early_notable_models | LLM-7 (timeline, compressed) |
| 07 pretraining_finetuning | Objectives -> LLM-7; SFT + alignment idea -> LLM-9; PEFT -> LLM-13 |
| 08 prompting | LLM-10; ReAct -> L44; Tree of Thoughts cut |
| 09 hallucinations | LLM-11 |
| 10 mixture_of_experts | LLM-14 |
| 11 inference_optimization | Quantization + memory arithmetic -> LLM-12; distillation -> LLM-9; KV cache, GQA, FlashAttention, speculative, batching -> LLM-14 |
| 12 rag | **Retired** - ch17 L41-L43 |
| 13 scaling_laws | LLM-8 |
| 14 agents_tool_use | **Retired** - ch18 L44 |
| 15 reasoning_test_time | CoT / self-consistency -> LLM-10; two scaling axes -> LLM-8; PRM/ORM, o1/R1 -> L32g |
| 16 long_context_attention | LLM-14; "Positional encoding for length" and "Context window extension" -> LLM-2 |
| 17 emergence | Merged into LLM-8 |
| 18 reinforcement_learning | **Retired** - ch11 L32-L32g |

### `ml/llm_training/slides/` (19 paper decks)

They stay as the registered "LLM Training & Alignment" reading list; the lectures borrow from them.

| Deck | Feeds |
|---|---|
| 01 GRPO, 11 DeepSeek-R1 | L32g (already borrowed) |
| 02 DPO, 10 InstructGPT | L32f; InstructGPT picture also LLM-9 |
| 03 LoRA, 12 QLoRA | LLM-13 (QLoRA's NF4 section also LLM-12) |
| 04 LIMA, 18 FLAN, 08 Qwen3 | LLM-9 |
| 05 BPE-dropout | LLM-1 (optional frame) |
| 06 Don't Stop Pretraining | LLM-13 (continued pretraining) |
| 07 Llama 3 | LLM-7 (data), LLM-8 (over-training) |
| 09 Scaling laws | LLM-8 |
| 13 MoE, 15 FlashAttention | LLM-14 |
| 16 RoPE | LLM-2 |
| 14 Chain of thought | LLM-10 |
| 17 DeepSeek-V3 | LLM-14 (MLA, MTP), LLM-12 (FP8) |
| 19 Making training fast | LLM-12 (float formats) |

---

## 8. Session budget

All options include the three built sessions LLM-3 to LLM-5.

| Option | Lectures | Sessions (+ RAG 3, agents 1) |
|---|---|---|
| Short | LLM-1, LLM-2, LLM-3 to LLM-5, LLM-6, LLM-7+8 merged, LLM-9, LLM-10+11 merged, LLM-12+13 merged (lighter) | 10 (+4) |
| **Core (recommended)** | LLM-1, LLM-2, LLM-3 to LLM-5, LLM-6 to LLM-13 | 13 (+4) |
| Full | Core + LLM-14 (2 sessions) | 15 (+4) |

Practicals add 1-2 sessions in any option. Context: `ml/00_plan.md` (Aug 8) gave the LLM topics 5
sessions and projected a mid-to-late November finish. Counted 2026-10-02: **45 lecture decks are
already built after L16** (L17 through L48, including the 8 mech-interp decks), before this
block's 10-12 and any practicals. Something gets cut whatever is chosen here.

---

## 9. Recent developments - verified 2026-10-02

Primary sources unless marked. **Re-verify before delivery.**

- **Gemma 4** - released 2026-03-31 in E2B, E4B, 26B A4B (MoE) and 31B; MTP checkpoints
  2026-04-16; a 12B "Unified" 2026-06-03. [Gemma releases](https://ai.google.dev/gemma/docs/releases)
- **Gemma 3 QAT** - quantization-aware-trained checkpoints, including Q4_0 GGUFs for llama.cpp
  and Ollama (2025). [Google Developers Blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
- **gpt-oss** - 2025-08-05; 120B (117B total / 5.1B active) and 20B (21B / 3.6B); MoE weights in
  MXFP4; 120B fits one 80 GB GPU, 20B fits 16 GB; alternating full and 128-token sliding-window
  attention; learned attention sinks; 128K context. [HF blog](https://huggingface.co/blog/welcome-openai-gpt-oss)
- **Jan-Mar 2026 open-weight architectures** - Qwen3.5 (397B/17B, Gated DeltaNet hybrid), GLM-5
  (744B/40B, MLA + DeepSeek sparse attention), Kimi K2.5 (1T/32B, MLA), Step 3.5 Flash (MTP-3 in
  training and inference), MiniMax M2.5 (classic GQA), Nemotron 3 Nano (mostly Mamba-2), Mistral
  Small 4 (119B/6.63B, MLA). Author's conclusion: data and recipes matter more than architecture.
  [Raschka, Jan-Feb 2026](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight),
  [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- **llama.cpp** - README quick start is now `llama cli -hf ...` / `llama serve -hf ...`, with an
  installer at llama.app; `llama-quantize`, `llama-perplexity`, `llama-bench`, `llama-imatrix`
  are still standalone tools; 1.5- to 8-bit quantization.
  [README](https://github.com/ggml-org/llama.cpp), `tools/CMakeLists.txt` (via Context7).
  NVFP4 kernels merged March-April 2026 (secondary source citing PR numbers; check the PRs at
  build). [insiderllm](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/)
- **Ollama** - MLX backend on Apple Silicon (preview 2026-03-30); Anthropic API compatibility
  (2026-01-16); `ollama launch` (2026-01-23); cloud models (2025-09-19); gpt-oss (2025-08-05).
  [Ollama blog](https://ollama.com/blog). LoRA adapters no longer supported
  (`server/create.go:40`, `:903`); no quantization on GGUF import (`docs/import.mdx:35`) - both
  read from the live `main` branch.
- **HF transformers runs llama.cpp quants** - 2026-09-22: `from_pretrained(..., gguf_file=...)`
  now keeps the weights packed and reuses the ggml kernels (Metal first; transformers `main`, not
  yet a release) instead of dequantizing on load.
  [HF blog](https://huggingface.co/blog/transformers-llama-cpp-quants)
- **Unsloth** - `save_pretrained_gguf(dir, tokenizer, quantization_method="q4_k_m")`.
  [Unsloth docs](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf.md)
- **FP4 formats** - MXFP4: OCP standard, blocks of 32, E8M0 shared scale. NVFP4: blocks of 16,
  FP8 E4M3 block scale + FP32 tensor scale; native on Blackwell. Quality vs Q4_K_M unsettled
  (secondary). [insiderllm](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/),
  [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- **Why LMs hallucinate** - Kalai, Nachum, Zhang (OpenAI) and Vempala (Georgia Tech),
  2025-09-04: training and evaluation reward guessing. [arXiv 2509.04664](https://arxiv.org/abs/2509.04664)
- **Diffusion LMs** - Mercury (Inception), Gemini Diffusion, LLaDA 2 reported as deployed in 2026.
  **Secondary sources only** - confirm before putting a number on a slide.

**Seen in search, not used:** "Kimi K3", "DeepSeek V4 Pro", "GLM-5.2", "Qwen3.8" appeared only on
SEO aggregator pages. Not in this plan until a primary source confirms them.

---

## 10. Known issues found while planning

- ~~**`misc/dl4nlp/03_tokenization.pdf` is 0 bytes**~~, committed that way in b7424f8
  (2026-07-26). **Fixed 2026-10-04:** recompiled in `sources/dl4nlp/` (753 KB, 0 errors) and
  copied back over the original.
- `ml/00_plan.md` is from 2026-08-08 and predates the reorder and this plan.
- The dl4nlp decks use `misc/dl4nlp/preamble.tex`; macros differ from `ml/preamble.tex`
  (`13_rnns/RNN_CHAPTER_PLAN.md` lists known deltas: `lightbg`, `-Stealth`).

## 11. Next steps, after approval

1. Answer D1 and D3-D6 (D2, D7, D8 resolved).
2. Reorder edits from section 2 (L23, L23c; `_quarto.yml` order, qmd titles and `00_plan.md` when
   the instructor wants the Quarto pass).
3. ~~Create the chapter folder, move this file into it~~ - done 2026-10-04 (`ml/14_llms`).
4. Per lecture, starting with LLM-1 (now the chapter's first session): outline -> approval ->
   build (port from `sources/`, section 7; "Where we are" / "Next on the map" frames from
   section 5.0) -> compile-deck -> student review (opt-in).
5. D6 measurement (Armenian tokens/word + perplexity for 3-4 candidate models) before P1/P2 are
   built.
