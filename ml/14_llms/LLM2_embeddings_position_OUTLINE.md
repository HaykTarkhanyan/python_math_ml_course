# LLM-2 - Embeddings and position: from IDs to vectors - outline (built 2026-10-04: `LLM2_embeddings_position.tex`)

Drafted 2026-10-04 after the instructor interview. Session 2 of the LLM chapter (`ml/14_llms`), the
map's Embedding and Position boxes (DECISIONS #68). Its job: turn the token IDs from LLM-1 into the
vectors the blocks read, `wte[id] + wpe[pos]`, and show where word order lives. It ends with one
promise for L24: the next layer weighs vectors by their content, and a weighted sum ignores order.
It must not lean on attention (queries and keys appear only as a forward reference in the RoPE
section, "the two vectors attention will compare").

Deck file (until delivery gives it a playlist number): `LLM2_embeddings_position.tex`.

**Interview decisions (2026-10-04):**

- Built frames: **move** L24's embedding primer and L25's Position section here. L24 and L25 are
  reference material from now on ("collecting dust, we're gonna iterate over them anyways").
- Position depth: **all of it here** (learned, sinusoidal, RoPE, long context), with "more time on
  important stuff and less on super technical".
- word2vec: **stays in LLM-4** (pretraining).
- Map: **Position gets its own stage** between Embedding and Attention.
- Cold open: **"dog bites man"**, plus the instructor's Kargin Haghordum reference on the frame:
  the line "ուրա պարավը, բերեք ատամը քաշեմ" and the link
  <https://karginhaghordum.am/sketch/3bEwVKXicsQ/> (instructor: "a local reference joke, just
  include it").
- Measurements: **GPT-2's learned positions** and **nearest neighbours**, with "more interesting
  words, not just Paris".
- Long-context tail: **short, intuition only** (3-4 frames).
- By-hand frames: **lookup = one-hot x matrix**, **RoPE**, **add vs concatenate**.
- Offered, not picked: sinusoids by hand, GPT-2 with its position table removed, Armenian
  neighbours in a multilingual model, full long-context formulas.

**Measured for this outline** (GPT-2 small, local HF cache, 2026-10-04; scratchpad probes, to be
redone by the figure script):

| What | Result |
|---|---|
| "dog bites man" / "man bites dog" | **Different tokens**: `dog` `bites` `man` (9703 26081 582) vs `man` `bites` `dog` (805 26081 3290). The first word has no leading space (LLM-1) |
| "The dog bites the man" / "The man bites the dog" | Same 5 IDs (464 3290 26081 262 582 vs 464 582 26081 262 3290); sum of token vectors identical (max difference 0) |
| ...with positions added, still summed | **Still identical** (Σ tokens + Σ positions). Positions help only because each vector changes, not the sum |
| GPT-2 tables | `wte` 50,257 x 768 = 38,597,376 (31.0% of 124,439,808); `wpe` 1,024 x 768 = 786,432 (0.63%); no separate output matrix in the checkpoint (tied) |
| Learned positions | Neighbouring rows: cosine 0.997; gap 20: 0.960; gap 100: 0.471; gap 500: 0.246. Top 3 principal directions = 90% of the variance; two of them oscillate (7 sign changes over 1024 positions). Position 0 is ~3x longer than the rest (9.88 vs median 3.37) |
| Add vs concatenate | Token vs position vectors, mean abs cosine 0.025 (random 768-d pairs: 0.029); lengths comparable (3.95 vs 3.37); the token is still the nearest row after adding a position in **2000/2000** tests |
| Analogies (inputs excluded, as is standard) | king - man + woman -> **queen** 0.69; actor -> **actress** 0.82; cats - cat + mouse -> **mice**; German - Germany + Armenia -> **Armenian** 0.79; better - good + bad -> **worse**. Fails: Paris - France + Italy -> `Paris`, Milan (Rome 5th); walked - walk + swim -> swimming |

Nearest neighbours worth a slide (cosine on `wte`, top 3-5):

| Word | Neighbours |
|---|---|
| ` Monday` | Tuesday 0.89, Thursday, Wednesday, Friday, Sunday |
| ` 1999` | 1998 0.89, 1997, 1995, 2001 |
| ` Harry` | `Harry`, Hermione 0.63, Voldemort, Ginny, Hogwarts |
| ` Batman` | `Batman`, Superman 0.68, Gotham, superhero, Joker |
| ` Messi` | Ronaldo 0.66, Guardiola, Juventus, Barcelona, ... LeBron |
| ` lol` | haha 0.79, LOL, `lol`, `:)`, `;)` |
| ` Pikachu` | Pokémon 0.61, pokemon, ... and `StreamerBot` (a glitch token) |
| ` Armenia` | Armenian 0.76, Azerbaijan 0.72, `Armen`, Kazakhstan |
| ` unhappy` | dissatisfied 0.74, **happy** 0.66, miserable: opposites share contexts |
| ` Einstein` | physicist, Hawking, relativity, ... **Eisenhower** (spelling, not meaning) |
| ` SolidGoldMagikarp` | ` RandomRedditorWithNo` 0.69, ` Smartstocks`, ` Adinida`, ` davidjl`: **other glitch tokens** (LLM-1 callback). Not near the centroid (rank 14,356 of 50,257): do not claim it |

---

## Frames

Figures: one new script, `py_src/embeddings_position_figs.py` (GPT-2 weights via safetensors +
tiktoken, no torch; every number on a slide asserted, as in `tokenization_figs.py`), drawn at slot
size. The map frames come from `py_src/llm_roadmap.py` after it gains the Position stage and the
`embeddings` session.

### Opening

| # | Frame | Content | Figure |
|---|---|---|---|
| 0 | Title | "Embeddings and position: from IDs to vectors" | - |
| 1 | Where we are: one forward pass | Map, Embedding + Position lit. "Today: an ID becomes a vector, and the vector learns where it sits." | `roadmap_embeddings_pass.pdf` |
| 2 | Where we are: the life of a model | "Later: ..." | `roadmap_embeddings_life.pdf` |
| 3 | Cold open: same words, opposite news | **Predict first.** "The dog bites the man" / "The man bites the dog": does the model see the same thing? The Kargin Haghordum line, drawn as a small figure, with the link. | `emb_kargin.pdf` (the Armenian line) |
| 4 | Same five numbers | Click: the same 5 IDs in a different order -> look them up -> the same 5 vectors -> sum identical, to the bit. Aside: the bare "dog bites man" is not even the same tokens (`dog` vs ` dog`, LLM-1). | `emb_coldopen_ids.pdf` |
| 5 | Outline | | - |

### 1. An ID becomes a vector

| # | Frame | Content | Figure |
|---|---|---|---|
| 6 | [plain] transition | "An ID becomes a vector" / "3290 is a row number, not a quantity." | - |
| 7 | An ID is a name, not a number | ` cat` = 3797, ` dog` = 3290: is a cat "bigger" than a dog? ID arithmetic means nothing. Callback: L20's one-hot vectors, all equally far apart. | - |
| 8 | By hand: one-hot x matrix picks a row | A 4-word vocabulary, 3-dim table: one-hot(2) x E = row 2. So the "embedding layer" is a table read, and its gradient touches only the rows used. | LaTeX matrices |
| 9 | GPT-2's table | 50,257 rows x 768 numbers = 38.6M = 31% of GPT-2 small (callback: LLM-1's vocabulary trade-off). Rows for a few modern models (to verify). | `emb_table_sizes.pdf` |
| 10 | Who fills the table? | Random at the start, trained by backprop with the rest of the model on next-token prediction. Callback: the ch11 surname inventor had exactly this table. | - |
| 11 | Predict first: what sits next to ` Monday`? | Commit before the click. | - |
| 12 | What the table learned | Neighbour grid: Monday, 1999, Harry, Batman, Messi, lol, Armenia, Pikachu. Nobody labelled anything; only "predict the next token". | `emb_neighbours.pdf` |
| 13 | Neighbours share contexts, not meanings | ` unhappy` -> ` happy`; ` Einstein` -> ` Eisenhower`; ` cat`, ` Cat`, `cat`, ` CAT` sit together (LLM-1 split them; the table put them back). | `emb_neighbours_odd.pdf` |
| 14 | The glitch tokens found each other | ` SolidGoldMagikarp`'s neighbours are other glitch tokens: rows that barely trained stay together, far from real words (LLM-1 callback; the "barely trained" reading is the standard explanation - cite it, do not overclaim). | in `emb_neighbours_odd.pdf` or own panel |
| 15 | Directions carry meaning (mostly) | king - man + woman -> queen; actor -> actress; cats - cat + mouse -> mice; German - Germany + Armenia -> Armenian. The failures too (Paris - France + Italy, walked - walk + swim). Footnote: the three input words are excluded, as everyone does. | `emb_analogies.pdf` |
| 16 | One vector per token, whatever the sentence | ` bank` gets the same row in "river bank" and "bank account". Context comes later (L24). From L24's "Embeddings soak up context". | - |
| 17 | The same table, used twice | GPT-2 reuses `wte` as its output layer (tied weights): score every row against the final vector. Forward reference to L26's LM head. Which modern models tie (to verify). | - |

### 2. Where is the order?

| # | Frame | Content | Figure |
|---|---|---|---|
| 18 | [plain] transition | "Where is the order?" / "A table lookup has no idea where the token sat." | - |
| 19 | Back to the dog | ` dog` gets the same vector in slot 2 and in slot 5: the model receives a **set** of vectors. | `emb_set_of_vectors.pdf` |
| 20 | Why the next layer cannot recover it | Forward reference, stated with what students know: the next layer scores pairs of vectors by a dot product of their contents and takes a weighted sum; nothing in it reads a slot number (L24 proves it exactly). Callback: an RNN got order for free (L20/L21) - attention will trade it for parallelism. | - |
| 21 | The fix: write the slot into the vector | x = token vector + position vector. Now "dog in slot 2" and "dog in slot 5" are different inputs. | `emb_add_position.pdf` |
| 22 | Trap: adding does not fix a sum | Measured: summed over the sentence, Σ(token + position) is still identical for both sentences. Positions work because each vector is now different and the next layer looks at them one pair at a time. | - |

### 3. Learned positions (GPT-2)

| # | Frame | Content | Figure |
|---|---|---|---|
| 23 | [plain] transition | "Option 1: learn the positions too" / "Another table, one row per slot." | - |
| 24 | GPT-2's position table | 1,024 rows x 768, trained like the token table. Token 1,025 has no row: GPT-2 cannot read past 1,024 tokens, by construction. | - |
| 25 | Predict first: what did it learn? | Random noise per row, or something smooth? | - |
| 26 | It learned waves on its own | Heatmap of the table + its top 2 directions over position + similarity vs gap (0.997 at 1, 0.47 at 100, 0.25 at 500). Nobody asked for smooth. One line: position 0 is ~3x longer than the rest. | `emb_gpt2_positions.pdf` |
| 27 | Learned positions: the bill | A hard length cap; rare late positions get little training; nothing transfers to a longer text. | - |

### 4. Sinusoids

| # | Frame | Content | Figure |
|---|---|---|---|
| 28 | [plain] transition | "Option 2: write the positions down" / "No training, any length." | - |
| 29 | A clock with many hands | Each pair of dimensions is a hand turning at its own speed: fast hands tell neighbours apart, slow hands tell far-apart slots apart. Formula (from L25), heatmap. | `emb_sinusoid_heatmap.pdf` |
| 30 | Similarity depends on the gap only | Dot product of two encodings vs gap, same curve at positions 10, 25, 40 (from L25, redrawn). The extrapolation claim fixed after the source check. | `emb_sinusoid_gap.pdf` |
| 31 | Add or concatenate? | Predict first: adding should blur the token, so why does everyone add? | - |
| 32 | By hand: add vs concatenate | Toy 4-dim example: token in one direction, position in another; the sum keeps both readable. Concatenating instead doubles the width every layer has to pay for. | LaTeX vectors |
| 33 | Measured: in 768 dimensions there is room | Abs cosine of token vs position vectors 0.025 (random pairs 0.029); after adding, the token is still the nearest row in 2000/2000 tests. | `emb_add_vs_concat.pdf` |

### 5. RoPE: rotate, do not add

| # | Frame | Content | Figure |
|---|---|---|---|
| 34 | [plain] transition | "Option 3: rotate, do not add" / "What attention needs is the gap, not the address." | - |
| 35 | What we actually want | When two tokens are compared, "3 apart" should matter, not "slot 41 vs slot 44". Adding mixes the two. (RoPE deck: "What we actually want".) | - |
| 36 | Each pair of dimensions is a clock hand | Rotate the vector by angle pos x theta, one speed per pair (the same speeds as the sinusoids). | `emb_rope_clock.pdf` |
| 37 | Why the gap falls out | Dot product = length x length x cos(angle between). Rotate both by their positions: the angle between them changes by (m - n) x theta, so the score depends only on the gap. | `emb_rope_2d.pdf` |
| 38 | By hand: same gap, same score | 2-D query and key at slots (2, 5) and (7, 10): the same score, worked out. | LaTeX + small figure |
| 39 | Where it is applied | Not to the input: to the two vectors attention compares (L24 names them query and key). The token vector itself stays position-free. | - |
| 40 | Who uses it | Llama, Qwen, Gemma, Mistral, DeepSeek... (to verify, with dates). "The gains were modest; the adoption was not" (RoPE deck). | `emb_pe_adoption.pdf` or a table |

### 6. Longer and longer (intuition only)

| # | Frame | Content | Figure |
|---|---|---|---|
| 41 | [plain] transition | "Longer and longer" / "GPT-2 stopped at 1,024. Models now read a million." | - |
| 42 | Why RoPE breaks past its training length | The slow hands reach angles they never saw in training: unseen inputs, wrong scores. | `emb_rope_unseen.pdf` |
| 43 | Stretch the clock | Position interpolation: squeeze the new length into the trained range. YaRN: stretch the slow hands, keep the fast ones. Then a little training on long documents. No formulas. | - |
| 44 | Two other answers: ALiBi and NoPE | ALiBi: no position vectors, a penalty that grows with distance. NoPE: no encoding at all; a causal model can still infer order (to verify: which 2026 models interleave NoPE layers). | - |
| 45 | What models use, 2017-2026 | Table: Transformer 2017 sinusoidal; GPT-2/BERT learned; T5 relative bias; BLOOM ALiBi; Llama/Qwen/Gemma/DeepSeek RoPE + extension; context lengths (all to verify). | table |

### Close

| # | Frame | Content | Figure |
|---|---|---|---|
| 46 | Recap | ID -> row of a learned table -> + where it sits (learned, sinusoids, or a rotation inside attention); what the table learns; why order had to be added. | - |
| 47 | Next on the map | "Next (L24): the vectors start talking to each other - and we prove they cannot hear order without today's addition." | `roadmap_attention_pass.pdf` |
| 48 | Sources & further reading | | - |

---

## Changes outside this deck when it is built

- `py_src/llm_roadmap.py`: a Position stage between Embedding and Attention; a new `embeddings`
  session (Embedding + Position lit); L24 lights only Attention; `SESSIONS` labels LLM-3... shift by
  one; redraw all map PDFs.
- `LLM1_tokenization.tex`: its "Next on the map" frame says "Next (L24): an ID becomes a vector"; it
  becomes LLM-2. Recompile.
- L24 and L25 are **not** edited now (instructor: reference material until the rework).

## To verify before building (do not bake in from memory)

- Sinusoidal extrapolation in practice (Press et al. 2021, ALiBi, "Train Short, Test Long"), and
  what L25's "better extrapolation" for RoPE should say instead.
- Su et al. (RoFormer) year and venue; Vaswani et al. 2017; Press et al. 2022 (ICLR); Chen et al.
  2023 (position interpolation); Peng et al. 2023 (YaRN); Haviv et al. 2022 (NoPE in causal LMs).
- Which 2026 open models use RoPE, its scaling, NoPE layers, and their context lengths.
- Which modern models tie input and output embeddings; embedding table sizes for frame 9.
- The glitch-token explanation for frame 14 (under-trained rows), against a source.
- The Kargin Haghordum link resolves.

## Sources

L24 frames "Recall: a token is a vector", "Directions carry meaning", "Embeddings soak up context";
L25 section "Position" (5 frames, figures `l25_permutation`, `l25_pe_heatmap`, `l25_pe_dotprod` -
redrawn at slot size, not reused); `sources/llm_training/16_rope/16_rope.tex` (13 frames);
`sources/dl4nlp/02_transformers.tex` (position frames at lines 590, 626, 723);
`sources/dl4nlp/16_long_context_attention.tex` ("Positional encoding for length", "Context window
extension").
