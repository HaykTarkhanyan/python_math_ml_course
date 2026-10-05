# 14 LLMs - chapter plan (draft for approval)

Drafted 2026-10-02; folder, order and map decided 2026-10-04 (DECISIONS #66, #67). **Not approved
yet** - the open decisions in section 1 come first. Follows the house new-chapter workflow
(interview -> outline -> approval -> build), like `ATTENTION_CHAPTER_PLAN.md` (this folder, L24-L26)
and `ch17_rag/RAG_CHAPTER_PLAN.md`.

**Scope.** The whole language-model chapter between the RNN chapter and the application chapters
(RAG, agents): tokenization, embeddings and position, the three built attention lectures (L24-L26),
and LLM-3 to LLM-11.
Sources: the instructor's `misc/dl4nlp/` decks and the `ml/llm_training/slides/` paper decks,
copied into `sources/` (see `sources/README.md`), and recent material verified on 2026-10-02
(section 9).

---

## 1. Open decisions

| # | Decision | Recommendation | What would flip it |
|---|---|---|---|
| D1 | Sessions for the chapter (not counting RAG/agents) | **13**: L24-L26 (built) + LLM-1 to LLM-10; LLM-11 if time allows | A hard course end date. See section 8 for the 10-session cut |
| D2 | Where GANs go | **Resolved 2026-10-03:** AE -> VAE -> GAN -> diffusion, after this block (DECISIONS #63) | - |
| D3 | How deep alignment goes here | **Idea level in LLM-6**; the math stays in L32f/L32g, which depend on PPO from L32d | Moving the whole RL chapter right after this block (+7 lectures before RAG) |
| D4 | Local runtime for the hands-on parts | **Ollama** for the first contact (one installer, Windows-friendly); **llama.cpp tools** for the quantization measurements (`llama-quantize`, `llama-perplexity`, `llama-bench` have no Ollama equivalent) | Most students on Macs -> LM Studio or MLX deserve the demo slot |
| D5 | Fine-tuning library for the practical | Two real options: **PEFT + TRL** (the standard HF stack, every step visible) or **Unsloth** (fits a free T4 more easily, `save_pretrained_gguf` exports straight to GGUF). Lean: Unsloth for the practical, PEFT shown on a slide | If the goal is "see the mechanics" over "ship a model", PEFT + TRL |
| D6 | Base model for practicals | **Decide by measurement**, not by name: tokens per Armenian word and Armenian perplexity for 3-4 small open models (e.g. Gemma 4 E2B, a small Qwen3.5) | - |
| D7 | Chapter folder and number | **Resolved 2026-10-04:** one folder, `ml/14_llms`, for attention and the LLM lectures; RNN is `ml/13_rnns` (DECISIONS #66). Deck files keep `L24`-style names until delivery | - |
| D8 | Teaching order inside the chapter | **Resolved 2026-10-04:** left to right along the forward pass - tokenization before attention - with a "you are here" map in every session (DECISIONS #67, section 5.0) | - |

---

## 2. Where the block sits

**Course order (instructor decisions: DECISIONS #63 on 2026-10-03, superseding #62; the order
inside this chapter from #67 on 2026-10-04):**
CNN (L16-L19) -> **RNN** (L20, L21) -> **this chapter, part 1**: LLM-1 tokenization, LLM-2
embeddings and position, L24-L26 attention and the transformer, LLM-3 to LLM-8 -> RAG (`ch17`, L41-L43) -> Agents (`ch18`, L44) ->
**this chapter, part 2**: LLM-9, LLM-10 (LLM-11) -> **Autoencoders** (L22, L23) -> **GANs** (L23b,
L23c) -> **Diffusion** (L27-L31) -> the rest (RL, VLM, ...).

RAG and agents sit inside the chapter's run on purpose: LLM-8 ends on "give the model the
documents" (-> L41), and LLM-10 opens with the ladder "prompt -> RAG -> fine-tune", which needs RAG
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

- `ch8_autoencoders/L23_vae.tex:473` - the closing "Next" box points at attention via L21; repoint
  it to GANs, then diffusion.
- `ch8b_gans/L23c_gan_applications.tex:427` - drop "Before it, the attention chapter (L24) picks up
  the other thread".
- `_quarto.yml` order, qmd title numbers, `ml/00_plan.md` (instructor 2026-10-04: Quarto later;
  the rename only fixed the two `_quarto.yml` paths so the site still builds).
- **Done 2026-10-04 (#67):** L21's closing Next box now points to tokenization first, then
  attention; L26's closing frame no longer lists tokenization as coming next.
- Chapter numbers written inside decks ("chapter 8" in `xx_features.tex:263`, "ch8" in `L34`,
  `L39`) - part of the renumbering pass (D7).
- **No edit needed:** L22's reference to the RNN (RNN now comes first), `HW1_sae_rnn` stays the
  autoencoder homework, `L23c:147`'s king-queen callback to L21, L21 -> L24.
- Small gain: L22's masked-modeling line (`:83`) and its "LLM interpretability" teaser (`:85`)
  turn from forward references into callbacks to this block.

---

## 3. What students already have

Do not re-teach these; call back to them. The LLM-2 and L24-L26 rows are this chapter's own sessions 2-5
(L24-L26 built, and reference material since #68); LLM-3 onward builds on them.

| Already taught | Where |
|---|---|
| One-hot encoding and why its geometry is meaningless (no tokenization, no embeddings - DECISIONS #64) | L20 |
| LM factorization, the generation loop (sample, feed back, repeat - no temperature), a GRU inventing surnames | L21 |
| The seq2seq bottleneck, and the alignment grid: every decoder step gets the same C, each needs a different part of the input | L21 |
| Bengio 2003 neural LM (the surname inventor practical) | ch11, `47_name_inventor_solution.ipynb` |
| Embeddings (the lookup table, what it learns), position (learned, sinusoidal, RoPE, long context) | LLM-2 (moved from L24/L25, #68) |
| Self-attention, multi-head, the block, causal mask, residual stream | L24, L25 |
| Encoder / decoder / enc-dec, CLM vs MLM, logits -> softmax -> sample, "autocomplete is not a chatbot", the O(n^2) bill and the 16,384-token FLOP crossover | L26 |
| Log-loss, temperature scaling, calibration | [11], [14] |

**Not yet taught when this block runs:**

- **Reinforcement learning.** The RL chapter (ch11, L32-L32g, which owns RLHF/DPO math, GRPO and
  R1) comes *after* this block. That is why D3 matters.
- **Autoencoders** (DECISIONS #63). No lecture may lean on them. Where an autoencoder callback
  would be natural, state the idea directly or call back to PCA (ch10, taught): masked LM and span
  corruption are "corrupt the input, reconstruct it" (LLM-4); LoRA's update is low rank (LLM-10);
  MLA squeezes the KV cache through a learned low-rank bottleneck (LLM-11).

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
   Same mechanism as L24-L26 were built from `02_transformers.tex`.
4. **Recent material is perishable.** Every 2025-26 fact carries its verification date and source
   (section 9). The landscape frames in LLM-11 get re-verified right before delivery.
5. **Armenian thread.** Tokens per Armenian word, Armenian perplexity, an Armenian fine-tune. It is
   the course's local hook and it measures something real (non-English text is where tokenizers,
   quantization and small models break first - a hypothesis to measure, not to assert).

---

## 5. The lectures

### 5.0 Session order and the chapter map

The map is two illustrated frames, one per row (instructor 2026-10-04: "a lot nicer ... not just
text boxes", two frames for more space):

- **"Where we are: one forward pass"** - GPT-2 small on L24's sentence "The cat sat on the", one
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
| 3 | L24 Attention | Attention | `attention` | built; its embedding primer moves to LLM-2 (#68) |
| 4 | L25 The Transformer block | Attention, MLP | `block` | built; its Position section moves to LLM-2 (#68) |
| 5 | L26 Transformers in the world | LM head | `transformers` | built; map frames added 2026-10-04 |
| 6 | LLM-3 Decoding | Decoding | `decoding` | built 2026-10-05 (`LLM3_decoding.tex`) |
| 7 | LLM-4 Pretraining | Pretrain | `pretraining` | to build |
| 8 | LLM-5 Scaling and emergence | Scale | `scaling` | to build |
| 9 | LLM-6 Post-training | Post-train | `post_training` | to build |
| 10 | LLM-7 Prompting | Prompt | `prompting` | to build |
| 11 | LLM-8 Evaluation and hallucination | Evaluate | `evaluation` | to build |
| - | RAG (L41-L43), agents (L44) | (own chapters, no map) | - | built |
| 12 | LLM-9 Quantization | Quantize | `quantization` | to build |
| 13 | LLM-10 LoRA and QLoRA | LoRA | `lora` | to build |
| 14 | LLM-11 Anatomy of a 2026 LLM | Attention, MLP again (KV cache, MLA; MoE) | `anatomy_2026` | optional |

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
  LLM-6. Optional 1 frame from llm_training 05 (BPE-dropout). Source copy:
  `sources/dl4nlp/03_tokenization.tex` (its PDF was 0 bytes; recompiled 2026-10-04).
- **Built 2026-10-04:** `LLM1_tokenization.tex`, figures from `py_src/tokenization_figs.py` (every
  split measured; numbers quoted in prose asserted). Self-review + Sonnet student review the same
  day; after the instructor's "I prefer proper decks" (frame count never matters, see
  `ml/SLIDE_STYLE.md`) it gained five worked frames: UTF-8 by hand, BPE on the 689 surnames' bytes
  (finds "-yan" by merge 5), where IDs come from, WordPiece by hand, Unigram by hand. It
  replaced `13_rnns/py_src/tokenizer_demo.py` instead of moving it: those panels were drawn 13 in
  wide. The Armenian line stays one shared file, `13_rnns/py_src/data/armenian_line.txt`, read by
  path. `tokenizer_demo.py` and its three panels are now used by no deck - candidates for
  `13_rnns/py_src/archive/`. `embedding_2d.py` waits for LLM-4.
- **Hands-on (not in the deck; format undecided):** train byte-level BPE on Armenian text (~50
  lines), compare tokens per word against `o200k` and one open-model tokenizer.

#### LLM-2 Embeddings and position: from IDs to vectors

Session 2 of the chapter, the map's Embedding and Position boxes (DECISIONS #68; instructor
2026-10-04: "id like positional encodings to be there, were going left to right"). LLM-1 ended at
token IDs, and an ID is only a row number. This session builds the vector the blocks actually read,
`wte[id] + wpe[pos]`, and hands over to L24 with one promise: the next layer weighs vectors by their
content, and a weighted sum ignores order. L24 proves it.

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
  table (LLM-1 callback). word2vec stays in LLM-4.
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
- **Sources:** L24's primer (3 frames) and L25's Position section (5 frames), moved;
  `sources/llm_training/16_rope` (13 frames); dl4nlp 02 (three position frames) and 16 ("Positional
  encoding for length", "Context window extension"). Two claims in L25's sinusoid frames are
  overstated and get fixed in the port after a source check: that sinusoids work at lengths never
  seen in training, and RoPE's "better extrapolation".
- Real GPT-2 vectors replace the hand-placed 2D map (`13_rnns/py_src/embedding_2d.py`, an archive
  candidate once LLM-2 is built).

#### L24-L26 Attention and the transformer (built; sessions 3-5)

Plan and decisions: `ATTENTION_CHAPTER_PLAN.md`. Since 2026-10-04 each deck opens with "Where we
are" and closes with "Next on the map" (section 5.0). L24's cold open still restates L21's
cliffhanger ("Last chapter ended on a cliffhanger"), which now has two lectures (LLM-1, LLM-2) in
between; its title page says "LLM chapter".

**Instructor 2026-10-04:** treat L24 and L25 as reference material ("collecting dust, we're gonna
iterate over them anyways"). Their embedding primer and Position section move to LLM-2 (#68); the
decks are not kept in sync with that move until the rework. **Rework list (from building LLM-2):**

- L24's "Where we are" caption still says "Today: the embedding and the attention step"; since the
  map redraw L24's figure lights Attention only. Recompiling L24 before the rework pairs the new
  map with the old caption.
- L24 still opens with the three embedding-primer frames; L24 and L25 still say position comes in
  L25 ("Next (L25): ... word order put back in"). Both now live in LLM-2.
- **LLM-2 promises the shuffle proof** ("the attention lecture proves it: shuffle the inputs and
  every output comes back unchanged, just shuffled"). Today it is L25's "Predict first" / "No.
  identical" / "Why that had to be true" (figure `l25_permutation.pdf`); the rework must keep it,
  ideally in L24 right after the attention equation.
- L25's sinusoid frames overstate extrapolation (sinusoids "including at positions longer than
  anything it was trained on"; RoPE "better extrapolation"). LLM-2 has the corrected version
  (Press et al. 2022).

#### LLM-3 Decoding: from logits to text

L26 ended with a distribution over the vocabulary ("the choosing is a separate decision ... later
in this chapter"); L21 promised that "how to steer that choice is a topic of its own". Every word
a chatbot writes is a choice from that distribution. The map's last box.

**Critique of the first draft (2026-10-04, before the interview).** The draft was a port of the
survey deck's method list (greedy, beam, temperature, top-k, top-p, min-p, contrastive,
structured, speculative): a catalogue, which is what LLM-2's student review flagged as the
weakest stretch of that deck. Specific problems:

1. **It re-teaches the loop.** L21 ("sample, feed back, repeat") and L26 (logits -> softmax ->
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
10. **Speculative decoding was parked in LLM-11, which is optional.** If LLM-11 never runs, the
    one decoding trick every serving stack uses is never taught. It only needs L25's causal mask
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
  LLM-11, which keeps only multi-token prediction); watermarking (SynthID-Text).** Not picked:
  providers restricting the knobs.
- **Min-p: taught, with the 2025 critique** (one frame on the idea, one line on the ICLR 2025
  oral and the fair-comparison result).
- **Hands-on: not in the slides** (like LLM-1). D4 stays open; nothing in this deck depends on a
  runtime choice.
- **Outline:** `LLM3_decoding_OUTLINE.md`. **Built 2026-10-05:** `LLM3_decoding.tex` (53 pages after the student review), figures from `py_src/decoding_figs.py` (toy numbers; GPT-2 only for the cold open, via `--gpt2`).

#### LLM-4 Pretraining: from the internet to a base model

"Predict the next token" on trillions of tokens. What comes out is not an assistant - it is a
document simulator.

- The lineage of "predict the missing word": n-gram counts (1-2 frames) -> Bengio 2003 (callback:
  the surname inventor *is* this model) -> word2vec (2 frames; taught nowhere in `ml/` today) ->
  ELMo (1 frame) -> CLM vs MLM (L26 recap) + T5 span corruption.
- **Embeddings live here and in LLM-2, not in the RNN chapter** (DECISIONS #64, #68): LLM-2 shows
  the table and what GPT-2's learned; this lecture shows how word2vec trained them without a
  language model around them.
- The data: crawl -> filtering funnel (dedup, language ID, quality classifiers, PII) - FineWeb;
  mixtures with code and math; synthetic data. Llama 3: ~15.6T tokens vs 1.8T for Llama 2 (from
  llm_training 07).
- Perplexity = exp(cross-entropy) (callback: log-loss [11]); worked example; caveat: only
  comparable under the same tokenizer (callback LLM-1).
- GPT-1 -> GPT-2 -> GPT-3 / BERT / T5 as a timeline of *lessons*: fine-tune -> zero-shot ->
  in-context learning.
- Base-model demos (both are parked instructor asks in `DEFERRED_TODO.md`): it continues a famous
  Wikipedia opening **verbatim** (memorization); asked a question, it writes more questions.
- **Sources:** dl4nlp 07 sec "Pre-training objectives" (7 frames), 06 (22 -> ~6), 05 sec
  "Perplexity" (4), 01 sec "Word embeddings" (5 -> 3); Karpathy "Deep Dive into LLMs" notes
  (`misc/dl4nlp/_yt_videos/karpathy_deep_dive_llms/`); llm_training 07 (Llama 3 data).
- **Hands-on:** perplexity of a small base model on Armenian vs English, and on a memorized vs a
  novel passage.

#### LLM-5 Scaling laws and emergence

- Power laws (Kaplan); C ~ 6ND; the training-memory bill (~16 bytes/param with Adam: weights,
  gradients, two moments, before activations - this number comes back in LLM-10).
- Chinchilla (~20 tokens/param); over-training for cheap inference (small models on far more
  tokens); the three eras of compute allocation.
- Emergent abilities, the mirage argument (the metric makes the jump), where the debate stands;
  grokking (`misc/grokking/`, Welch Labs material already extracted).
- Teaser: test-time compute as the second scaling axis -> L32g.
- **Sources:** dl4nlp 13 (18) + 17 (18), deduplicated to ~20; llm_training 09.
- **First merge candidate** into LLM-4 if sessions are short.

#### LLM-6 Post-training: from base model to assistant

- Why a base model is not a chatbot (callback: the LLM-4 demo).
- SFT on conversations; chat templates and special tokens (ChatML; the cost of adding tokens - from
  dl4nlp 03); "the assistant is a statistical imitation of a human labeler" (Karpathy).
- Instruction tuning at scale (FLAN); quality over quantity (LIMA, 1,000 examples).
- Preference tuning **at idea level** (D3): comparisons -> reward model (Bradley-Terry in one line),
  the RLHF pipeline as a picture, reward hacking, DPO as "skip the RL". Full math: L32f.
- Verifiable rewards and "thinking" models with a budget dial (Qwen3) as a teaser -> L32g.
- Distillation: SFT on a bigger model's outputs (the R1 distills).
- **Sources:** dl4nlp 07 sec 2-4 (~15 frames); llm_training 18 (FLAN), 04 (LIMA), 10
  (InstructGPT), 08 (Qwen3); distillation frames from dl4nlp 11.
- LoRA/QLoRA are **not** here any more - they get their own lecture (LLM-10).

### Part B - Using it, and knowing when it is wrong

#### LLM-7 Prompting and in-context learning

- Zero-shot, few-shot, and how fragile few-shot is (example order, format, label balance).
- Chain of thought, zero-shot CoT, why it works ("models need tokens to think"); self-consistency;
  reasoning models do this internally, which changes the advice.
- System prompts; structured output (callback: constrained decoding, LLM-3); context engineering -
  what goes into the window and in what order.
- Prompt injection, direct and indirect (indirect injection sets up the agents chapter).
- **Sources:** dl4nlp 08 (22 -> ~17; Tree of Thoughts cut, ReAct -> L44); llm_training 14 (CoT
  half).
- **Hands-on:** few-shot fragility on the local model from LLM-3 - permute example order and
  labels, measure the accuracy swing.

#### LLM-8 Evaluation and hallucination

- Why evaluation is hard; intrinsic vs extrinsic; BLEU / ROUGE in 2-3 frames and their limits;
  BERTScore.
- LLM-as-judge (pointwise vs pairwise; position, verbosity and self-preference biases).
- Benchmarks (knowledge, reasoning, code, agents), contamination, leaderboards and Goodhart.
- Hallucination: taxonomy (factuality vs faithfulness), why it happens - including Kalai et al.
  2025: training and evaluation reward guessing over "I don't know" (arXiv 2509.04664); detection
  (SelfCheckGPT, semantic entropy); mitigation overview.
- **Cliffhanger:** give the model the documents -> RAG (L41).
- **Sources:** dl4nlp 05 (27 -> ~16; perplexity -> LLM-4, MAP/nDCG -> L43 owns them) and 09
  (26 -> ~14; RAG mitigation -> ch17, duplicate self-consistency dropped).

**Then the built chapters continue the story:** RAG (`ch17`, L41-L43, 3 lectures, far deeper than
dl4nlp 12) and Agents (`ch18`, L44, replaces dl4nlp 14).

### Part C - Run it and adapt it yourself

#### LLM-9 Quantization: an LLM on your laptop

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

The local stack, as of 2026-10-02 (section 9 has the sources):

| Tool | What it is | When to reach for it |
|---|---|---|
| llama.cpp | C/C++ engine, GGUF, CPU / CUDA / Metal / Vulkan / SYCL; `llama cli`, `llama serve` (OpenAI-compatible); standalone `llama-quantize`, `llama-perplexity`, `llama-bench` | Full control, measuring, odd hardware |
| Ollama | One-command runner + model library; Modelfile; OpenAI- and Anthropic-compatible APIs; MLX backend on Apple Silicon (2026); cloud models | Easiest start; apps that need a local API |
| LM Studio | GUI over the same GGUF models (features: verify at build) | Non-programmers |
| MLX | Apple's array framework, native on M-series | Macs |
| vLLM / SGLang | GPU serving engines, built for throughput across many users | Serving, not laptops -> LLM-11 |
| HF transformers | Earlier GGUF loading dequantized to float; since Sep 2026 (main branch, Metal first) it runs the quantized llama.cpp kernels | Staying in Python |

#### LLM-10 Fine-tuning on a budget: LoRA and QLoRA

Full fine-tuning a 7B model needs ~112 GB (16 bytes/param, LLM-5). LoRA trains under 1% of the
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
- **QLoRA:** NF4 (quantiles of a normal - callback: block quantization in LLM-9), double
  quantization, paged optimizers; store in NF4, compute in BF16; matches 16-bit (Guanaco).
- Variants in one frame (DoRA and friends).
- **Recipe:** rank, targets, learning rate, epochs - take current defaults from the library docs at
  build time; data quality over quantity (LIMA callback); the chat template must match (LLM-6);
  evaluate before and after (LLM-8); catastrophic forgetting.
- **From adapter to laptop:** merge -> convert to GGUF -> `llama-quantize` -> Ollama
  (`FROM ./model.gguf`, `ollama create`). Unsloth does the first three in one call:
  `model.save_pretrained_gguf(dir, tokenizer, quantization_method="q4_k_m")`.
- **Gotcha (verified 2026-10-02 on Ollama `main`):** Ollama **no longer supports LoRA adapters** -
  `server/create.go:40` defines `errAdaptersUnsupported`, and `:903` rejects adapter-type GGUFs
  too. Merge the adapter first. Older tutorials using `ADAPTER` in a Modelfile are out of date.
- **Sources:** llm_training 03 (LoRA, 19 frames) and 12 (QLoRA, 21 frames); dl4nlp 07 sec "PEFT"
  (4 frames); dl4nlp 11's QLoRA frame is a duplicate - dropped.

### Part D - What changed since 2017

#### LLM-11 Anatomy of a 2026 LLM (likely 2 sessions)

Put L25's 2017 block next to a 2026 model card. Almost every change since exists to save memory or
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
  LLM-3 since 2026-10-05 - a callback here at most) and multi-token prediction heads (DeepSeek-V3; Gemma 4 MTP checkpoints, Apr 2026; Step 3.5 Flash).
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
- **Sources:** dl4nlp 11 (KV cache, MQA/GQA, FlashAttention, speculative decoding, continuous
  batching, ~12 frames), 10 (MoE, 20 -> ~6), 16 (long context, 22 -> ~6); llm_training 13 (MoE),
  15 (FlashAttention), 16 (RoPE), 17 (DeepSeek-V3: MLA, MTP); `misc/dl4nlp/_reference_welchlabs_mla/`
  (MLA video frames); Raschka's LLM Architecture Gallery for figures to redraw.

---

## 6. Practicals

| Practical | After | What students do | Compute |
|---|---|---|---|
| **P1 - A tiny GPT that speaks Armenian** | LLM-4/5 | Sequel to the ch11 surname inventor: same spirit, now a transformer. Tokenizer (char or the LLM-1 BPE) -> CLM pretraining -> perplexity -> decoding knobs | CPU, minutes (estimate - measure when built) |
| **P2 - Your own model, on your own laptop** | LLM-10 | QLoRA fine-tune a small model on Colab (T4) on a small Armenian instruction set -> merge -> GGUF -> quantize -> run it in the D4 runtime; evaluate before/after | Colab GPU (`colab-gpu` skill) + laptop |
| HW - The quantization ladder | LLM-9 | Size, RAM, tokens/s, perplexity per quant level, on their own machine | Laptop, small model |

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
| 01 pre_transformer | Covered by L20/L21/L24; n-grams, word2vec, ELMo -> LLM-4 |
| 02 transformers | Already absorbed into L24-L26; its position frames -> LLM-2 |
| 03 tokenization | LLM-1; chat tokens -> LLM-6 |
| 04 decoding_strategies | LLM-3, including speculative decoding (moved back from LLM-11, 2026-10-05) |
| 05 evaluation | LLM-8; perplexity -> LLM-4; MAP/nDCG stay with L43 |
| 06 early_notable_models | LLM-4 (timeline, compressed) |
| 07 pretraining_finetuning | Objectives -> LLM-4; SFT + alignment idea -> LLM-6; PEFT -> LLM-10 |
| 08 prompting | LLM-7; ReAct -> L44; Tree of Thoughts cut |
| 09 hallucinations | LLM-8 |
| 10 mixture_of_experts | LLM-11 |
| 11 inference_optimization | Quantization + memory arithmetic -> LLM-9; distillation -> LLM-6; KV cache, GQA, FlashAttention, speculative, batching -> LLM-11 |
| 12 rag | **Retired** - ch17 L41-L43 |
| 13 scaling_laws | LLM-5 |
| 14 agents_tool_use | **Retired** - ch18 L44 |
| 15 reasoning_test_time | CoT / self-consistency -> LLM-7; two scaling axes -> LLM-5; PRM/ORM, o1/R1 -> L32g |
| 16 long_context_attention | LLM-11; "Positional encoding for length" and "Context window extension" -> LLM-2 |
| 17 emergence | Merged into LLM-5 |
| 18 reinforcement_learning | **Retired** - ch11 L32-L32g |

### `ml/llm_training/slides/` (19 paper decks)

They stay as the registered "LLM Training & Alignment" reading list; the lectures borrow from them.

| Deck | Feeds |
|---|---|
| 01 GRPO, 11 DeepSeek-R1 | L32g (already borrowed) |
| 02 DPO, 10 InstructGPT | L32f; InstructGPT picture also LLM-6 |
| 03 LoRA, 12 QLoRA | LLM-10 (QLoRA's NF4 section also LLM-9) |
| 04 LIMA, 18 FLAN, 08 Qwen3 | LLM-6 |
| 05 BPE-dropout | LLM-1 (optional frame) |
| 06 Don't Stop Pretraining | LLM-10 (continued pretraining) |
| 07 Llama 3 | LLM-4 (data), LLM-5 (over-training) |
| 09 Scaling laws | LLM-5 |
| 13 MoE, 15 FlashAttention | LLM-11 |
| 16 RoPE | LLM-2 |
| 14 Chain of thought | LLM-7 |
| 17 DeepSeek-V3 | LLM-11 (MLA, MTP), LLM-9 (FP8) |
| 19 Making training fast | LLM-9 (float formats) |

---

## 8. Session budget

All options include the three built sessions L24-L26.

| Option | Lectures | Sessions (+ RAG 3, agents 1) |
|---|---|---|
| Short | LLM-1, LLM-2, L24-L26, LLM-3, LLM-4+5 merged, LLM-6, LLM-7+8 merged, LLM-9+10 merged (lighter) | 10 (+4) |
| **Core (recommended)** | LLM-1, LLM-2, L24-L26, LLM-3 to LLM-10 | 13 (+4) |
| Full | Core + LLM-11 (2 sessions) | 15 (+4) |

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
