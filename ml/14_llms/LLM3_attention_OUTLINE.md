# LLM-3 - Attention: how tokens share meaning - outline (restructure of the July L24; built 2026-10-05)

Drafted 2026-10-05 after the critique and interview in `LLM_CHAPTER_PLAN.md` (section "LLM-3 to
LLM-5 Attention and the transformer"). Session 3 of the LLM chapter (`ml/14_llms`). Its job: pay
off L21's two walls - attention as a learned look-back (wall 1: one C for every step), then
self-attention replacing the recurrence (wall 2: the queue) - and build one head of self-attention
slowly enough that every matrix earns its place. It must not re-teach LLM-2 (the lookup table,
directions in meaning space, the score -> softmax -> weighted sum preview on "dog bites man"):
callbacks only.

Deck file: `LLM3_attention.tex` (was `L24_attention.tex`, DECISIONS #71). Restructured in place:
every July frame is **kept**, **merged**, **moved** or **cut**, and new frames are marked **new**.

**Interview decisions (2026-10-05):**

- Scope: **restructure** (not a surgical fix, not a rebuild). Keep the 3b1b spine and the creature.
- Cold open: **L21's two walls.**
- New pieces: **learned W_Q, W_K by hand** and **a shapes frame**. Not picked: a real GPT-2 head,
  moving the shuffle proof in (it stays in LLM-4).
- The DistilBERT real-head frame: **cut** (LLM-4 shows real heads); the hand-set "illustrative"
  heatmap goes with it.
- Assumed in the interview and not objected to: the creature as the one example for the
  mechanics; one analogy (the soft dictionary); the duplicate stills dropped; one "try it" link to
  Transformer Explainer.

**Numbers:** toy, demonstrative values by default (`ml/SLIDE_STYLE.md`). Real numbers only for the
shapes frame, from GPT-2's cached config and safetensors header (read 2026-10-05, no model run):
d = 768, 12 heads, d_k = 64, `c_attn` 768 x 2,304, `c_proj` 768 x 768, context 1,024.

**Visual rule for the 3b1b material:** one beat, one picture. Each beat uses either the in-frame
3b1b image or the full-bleed still, never both. Full-bleed only where the still carries the beat on
its own (the murder mystery, the Q/K space, the masking aside).

---

## Frames

Figures: the 3b1b stills and LMU images already in `fig/borrowed/`, L21's `alignment_grid_1.pdf`
(read from `../13_rnns/fig/` via `\graphicspath`), the kept July figures (`dot_product_geometry`,
`softmax_scaling` from `py_src/attn_intuition.py`), and one new script, `py_src/attention_figs.py`:
toy numbers, drawn at slot size, every quoted number asserted, `save()` refusing clipped text (the
pattern of `decoding_figs.py`).

### Opening

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 0 | Title | "Attention" / "How tokens share meaning" | - | kept (retitled, #71) |
| 1 | Where we are: one forward pass | Caption fixed: "Today: the attention step - each token's vector picks up meaning from the other tokens." | `roadmap_attention_pass.pdf` | kept, caption fixed |
| 2 | Where we are: the life of a model | as now | `roadmap_attention_life.pdf` | kept |
| 3 | L21 left two walls | Left: L21's filled alignment grid - each Armenian word needs one or two different English words. Right: the queue - 1,000 tokens = 1,000 steps. "Wall 1: every step gets the same C. Wall 2: one step at a time." **Predict:** how would you fix wall 1? | `alignment_grid_1.pdf` + small TikZ queue | **new** (replaces "Last chapter ended on a cliffhanger" and its hand-drawn C) |
| 4 | The fix, in one idea | The reveal: at each output step, look back over every input state and take a weighted average (weights .1 / .8 / .1, sum to 1). "That weighted look-back is attention." | TikZ as now | kept (its predict moved to frame 3) |
| 5 | Outline | | - | **new** (the house skeleton has one; July had none) |

### 1. Wall 1: look back

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 6 | [plain] transition | "Wall 1: look back" / "A different context for every word you write." | - | **new** |
| 7 | Score, softmax, weighted sum | The three moves, named. Callback: LLM-2 ran them by hand on "dog bites man". Here the decoder's state asks; the encoder's states answer. | - | kept ("Attention = a weighted look-back"), trimmed |
| 8 | The grid, learned (built as "Learned, not drawn": the LMU image shows one step's weights, not a grid, so the frame says "stack the steps and you get a grid like L21's" and cites Bahdanau et al.'s grids without showing them) | LMU's translation attention (Bahdanau, Cho & Bengio, 2015): the model learned L21's grid by itself; nobody drew it. Second panel: captioning (Xu et al., 2015) - the positions can be pixels. | `lmu_translation_attention.png`, `lmu_captioning_attention.png` | kept ("You can see it"), retitled and tied to frame 3 |

### 2. Wall 2: drop the queue

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 9 | [plain] transition | "Wall 2: drop the queue" / "If every word can look at every word, who needs the recurrence?" | - | **new** |
| 10 | The encoder still reads one word at a time | **Predict:** wall 1 is fixed, but the encoder RNN still builds each vector after the previous one. Could attention build the vectors too? | small TikZ | merged (the predict of "Why keep the RNN at all?", now asked at the right moment) |
| 11 | Words need their own sentence | "mole" three ways, "tower" after "Eiffel", and in one line "model" (ML vs fashion): the vector a word starts with is the same in every sentence (LLM-2's lookup); the sentence has to change it. | `mole_polysemy.png`, `eiffel_context.png` | merged ("Why context matters" + "Embeddings soak up context") |
| 12 | The murderer was ... | Full-bleed: to fill the blank, the **last** vector must have soaked up the whole story. | `fs_mystery.jpg` | moved (from the cut primer section) |
| 13 | The sentence talks to itself | In self-attention Q, K and V all come from the same sentence: every word asks and answers, all-to-all. That is enough to build the vectors, so the recurrence goes: "Attention Is All You Need" (Vaswani et al., 2017). | TikZ circle graph, `deep_stack.png` | merged ("Why self-attention?" + the reveal of "Why keep the RNN at all?") |
| 14 | Sequential vs all-to-all | O(n) steps vs one step; three wins (parallel, direct, contextual). One line from the cut GPU frame: it is all matrix products, the workload a GPU wants (L14, L21's second wall). | TikZ as now | kept (absorbs "It's just matmul -> GPU") |

### 3. One head, slowly

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 15 | [plain] transition | "One head, slowly" / "Every piece of the equation, and why it is there." | - | kept (retitled) |
| 16 | The goal, made concrete | "a fluffy blue creature roamed the verdant forest": creature should absorb fluffy and blue. Two questions: which words are relevant, and what do they hand over? | `query_creature.png` | kept (`fs_creature.jpg` dropped: it is a second grab of the murder-mystery scene, and this frame already shows the creature) |
| 17 | Predict first: who does creature listen to? | Toy vectors with 4 numbers, read as [adjective, fluffy, blue, noun]: fluffy (1, 1, 0, 0), blue (1, 0, 1, 0), creature (0, 0, 0, 2). No learned matrices yet: each vector is its own query, key and value. Which word gets creature's biggest weight? | small table | **new** |
| 18 | Itself | Scores / sqrt(4): creature's row (0, 0, 2) -> weights **0.11, 0.11, 0.79**. Its update (0.21, 0.11, 0.11, 1.57) is mostly more creature. And the grid is symmetric: fluffy looks at blue exactly as much as blue at fluffy. Raw vectors ask "who is like me?" - the wrong question. | `att_raw_grid.pdf` (score and weight grids, creature's row highlighted) | **new** (replaces "Let's run the numbers 1/2, 2/2", which repeated LLM-2 and claimed a move it did not show) |
| 19 | A soft dictionary lookup | The one analogy: query = what you look for, key = the label, value = the contents; a dictionary matches one key exactly, attention matches every key a little and blends the values. | - | kept (the web-search and library frames merged into it) |
| 20 | Query: "what am I looking for?" | q = W_Q e; creature's query asks "any adjectives before me?"; W_Q is learned. | `query_matrix.png` | kept (`fs_query` double dropped) |
| 21 | Key: "what do I offer?" | k = W_K e; fluffy's key says "adjective, here". | `key_matrix.png` | kept (`fs_key` double dropped) |
| 22 | Queries and keys meet in one space | W_Q and W_K project into one smaller space where "relevant" means "pointing the same way". | `qk_space.png` | kept as the July frame (built that way: `fs_qkspace.jpg` turned out to be the pi creature asking "What does W_Q do to the non-nouns?", not the Q/K space) |
| 23 | The dot product scores relevance | q . k = abs(q) abs(k) cos(theta): large when aligned, 0 when perpendicular, negative when opposite. All pairs at once: S = Q K^T. | `dot_product_geometry.pdf` | merged ("The dot product, up close" + "Matching"; `fs_dotgrid` dropped) |
| 24 | By hand: ask a different question | A query/key space of **one** number (real heads: 64 in GPT-2). w_Q reads "noun" (0, 0, 0, 1), w_K reads "adjective" (1, 0, 0, 0). Queries (0, 0, 2), keys (1, 1, 0). Creature's scores (2, 2, 0) -> weights **0.47, 0.47, 0.06**. Fluffy's query is 0: it asks nothing, so it spreads evenly. Not symmetric any more: creature -> fluffy scores 2, fluffy -> creature 0. | `att_learned_grid.pdf` (same layout as frame 18, side by side with it in miniature) | **new** |
| 25 | Scores -> weights: why softmax | Positive, sum to 1, soft, differentiable. | `attention_pattern.png` | kept (`fs_pattern` double dropped) |
| 26 | Why divide by sqrt(d_k) | As now, with the number fixed: with entries of spread 1, a dot product over 64 numbers has spread sqrt(64) = 8, not "+-5"; softmax saturates, gradients vanish; dividing by 8 brings it back. | `softmax_scaling.pdf` | kept, number fixed (check the figure's own labels) |
| 27 | Value: "what do I hand over?" | v = W_V e. What you are found by (key) is not what you give (value). | `value_wv.png` | kept (`fs_value` double dropped) |
| 28 | By hand: what gets handed over | W_V keeps fluffy, blue and noun and drops the "adjective" tag: diag(0, 1, 1, 1). Creature's update = 0.47 (0, 1, 0, 0) + 0.47 (0, 0, 1, 0) + 0.06 (0, 0, 0, 2) = **(0, 0.47, 0.47, 0.13)**; added to creature (the residual, LLM-4): **(0, 0.47, 0.47, 2.13)** - a fluffy blue creature. Raw, frame 18: (0.21, 0.11, 0.11, 3.57). | `att_value_update.pdf` (creature's 4 numbers before, raw, learned) | **new** |
| 29 | The output: values, weighted and summed | o_i = sum_j alpha_ij v_j; every word updated at once. | `value_delta_e.png` | kept (`fs_deltae` double dropped) |
| 30 | Why three matrices? | The roles table, now a summary of what frames 18, 24 and 28 showed: W_Q and W_K let a word ask something other than "who is like me", and make "A looks at B" differ from "B looks at A"; W_V decides what is handed over. | table | kept, rewritten as a summary |
| 31 | Putting it together: the equation | softmax(Q K^T / sqrt(d_k)) V, read left to right. | annotated as now | kept (`fs_formula` double dropped) |
| 32 | Shapes | X is n x d, W_Q d x d_k, Q n x d_k, Q K^T **n x n** (row = who asks, column = who answers), softmax by rows, times V. The example is the instructor's July sentence "Donkey don't die, spring is coming": 6 words but 9 GPT-2 tokens (Don, key, " don", "'t", " die", ",", " spring", " is", " coming"), so X is 9 x 768 and the grid 9 x 9 - n counts tokens, not words (LLM-1). GPT-2: d = 768, 12 heads of d_k = 64; all twelve heads' W_Q, W_K, W_V sit in one 768 x 2,304 matrix (3 x 12 x 64 = 2,304); a full window of 1,024 tokens gives 1,048,576 scores per head per layer, 150,994,944 over 12 heads x 12 layers. Double the text, four times the grid: LLM-5's bill. | `att_shapes.pdf` (the matrices as boxes drawn to proportion) | **new** |
| 33 | Attention in ~15 lines | NumPy, as now. | listing | kept |
| 34 | Aside: masking | Full-bleed with caption: future entries set to minus infinity vanish after softmax (detail in LLM-4). | `fs_masking.jpg` | kept |

### Recap and close

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 35 | One head, end to end | Bullets rewritten: the two walls and their fixes; Q, K, V and what each one bought in the by-hand frames; the shapes; "Next (LLM-4): many heads, the block". | `single_head_full.png` | kept, rewritten (`fs_onehead` double dropped) |
| 36 | Try it yourself | Transformer Explainer: GPT-2 small running in the browser; type a sentence, click a token, see its attention. One link, no install. | screenshot or none | **new** (assumed in the interview) |
| 37 | Next on the map | -> LLM-4. | `roadmap_block_pass.pdf` | kept |
| 38 | Sources | As now, plus Transformer Explainer. | - | kept |

### Cut

- The primer section (LLM-2 owns it since #68): "Recall: a token is a vector", "Directions carry
  meaning" (king - queen); "Embeddings soak up context" merged into frame 11.
- "Another way to picture it: web search" (one analogy is enough).
- "Let's run the numbers (1/2), (2/2)" (replaced by frames 17, 18, 24, 28).
- "A pattern that makes sense" (hand-set heatmap) and "A real head, our sentence" (DistilBERT).
- "It's just matmul -> it runs on a GPU" (one line kept in frame 14).
- The full-bleed doubles: `fs_eiffel`, `fs_creature`, `fs_query`, `fs_key`, `fs_value`,
  `fs_dotgrid`, `fs_pattern`, `fs_formula`, `fs_deltae`, `fs_onehead`. Kept full-bleed:
  `fs_mystery`, `fs_qkspace`, `fs_masking`.

Figures no longer used by this deck (files stay; `bert_attention.py` still makes LLM-4's grid):
`attn_illustrative.pdf`, `bert_attention_donkey.pdf`, `worked_scores.pdf`.

---

## Open questions - answered 2026-10-05 (interview, second round)

1. **Query/key space of one number** in frame 24 (instructor's pick). The two-number version was
   offered and rejected: it holds two relations in one head, but its grid comes out symmetric
   again (1.41 both ways), which would muddle frame 24's "A -> B is not B -> A".
2. **Section names: the two walls** - "Wall 1: look back", "Wall 2: drop the queue", "One head,
   slowly" (over the July names and plain descriptive ones).
3. **"Donkey don't die, spring is coming" goes into the shapes frame** (frame 32): 6 words, 9
   GPT-2 tokens (counted with tiktoken's `gpt2` encoding, 2026-10-05). Not chosen: only in the
   try-it frame, both, or dropping it.
4. **L21's alignment grid: correct as is** (instructor, native speaker, 2026-10-05). The pending
   check in `13_rnns/py_src/alignment_grid.py` is closed.

## To verify before building - done 2026-10-05

- Transformer Explainer: checked; GPT-2 small in the browser, your own text, attention weights.
- `softmax_scaling.pdf` was drawn 9.2 in wide (text ~3 pt on the slide) and its "+-5" came from a
  6-key sample; redrawn at slot size as `att_sqrt_dk.pdf` with the spread measured on 10,000
  random pairs (8.05). `dot_product_geometry.pdf` (5.8 in wide) likewise became `att_dot_product.pdf`.
- `\graphicspath` reaches `../13_rnns/fig/`: compiles.

## Built 2026-10-05

`LLM3_attention.tex`, 40 pages (July: 56). Figures: `py_src/attention_figs.py` (6 PDFs, ~2 s;
`--verify` checks the GPT-2 constants against the local caches). Frames as above, with the two
self-review changes noted in rows 8 and 22, and frame 32 split in two ("Shapes, on a real
sentence" and "The same shapes in GPT-2"). Full-bleed stills kept: `fs_mystery`, `fs_masking`.
Checks: 0 LaTeX errors, 0 footer collisions, 0 pages with text under 5.5 pt; 3 clipping flags
checked by eye (the listing frame, the URL frame, a title with `\sqrt`) - false positives.

**Student review (2026-10-05, one Sonnet agent on the rendered pages, ~98k tokens, 89 s).** Now 41
pages. Applied: frame 24 reframed (the one axis means "adjective match": a noun asks along it, an
adjective answers on it) and split into a predict frame plus "Creature finds its adjectives" with
the softmax shown and "rows sum to 1, columns do not"; word order (a watch box on frame 14:
positions go in before the first attention layer, LLM-2); the frame-10 predict answered explicitly
in frame 13; d_k defined at first use; W_Q learned from the next-token loss; "project" defined;
sqrt(d_k): variances add, and why not divide by d_k; "why 64 and not 768"; the grid grows 4x while
X, Q, K, V double. Not applied, against interview decisions: a two-number W_Q, W_K example and a
numeric shapes toy; cutting the life map, Outline, mystery still, masking aside, recap or 3b1b
screenshots (the reviewer found the screenshots illegible at 150 dpi; on a projector they are
larger - worth a look in the room).
