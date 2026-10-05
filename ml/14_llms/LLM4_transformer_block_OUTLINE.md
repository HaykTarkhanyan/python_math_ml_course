# LLM-4 - The Transformer block - outline (restructure of the July L25; built 2026-10-05)

**Built 2026-10-05** (`LLM4_transformer_block.tex`, 41 pages; `py_src/block_figs.py`, 12 figures,
~7 s; `--gpt2` once, 25 s). Changes against the table below, from building:
- Real heads: measured first, then chosen - layer 5 head 11 is a perfect previous-token head
  (weight 1.00, entropy 0.0), layer 6 head 2 sends creature 62% to fluffy, and 39 of 144 heads put
  over 90% on the first token (an attention sink, own frame with the share per layer).
- Frame 4 split: "Stored, half-stored, missing" (the figure) and "So where does a fact live?"
  (with the LLM-1 callback: "Lionel Messi" is 2 tokens, "Henrikh Mkhitaryan" 7 pieces).
- The two-heads toy hands over one number per head (fluffiness .47, blueness .79), so every
  product is trivial.
- `l25_permutation.pdf` and `l25_causal_mask.pdf` were drawn 9+ in wide (3.6-4.7 pt text on the
  slide): ported into `block_figs.py` at slot size (`blk_permutation`, 1.1e-16 reproduced;
  `blk_causal_mask`).
- Callbacks re-pointed: LayerNorm/BatchNorm [44], residual [50], the MLP [41].
- Checks: 0 errors, 0 overflow, 0 footer collisions, 0 small text, 0 clipping flags. **Student
  review not run** (needs the instructor's OK).
- Self-review round (2026-10-05): GPT-2 uses GELU (a line on the ReLU frame); Meng et al. -> GPT-2
  XL; the sink explanation hedged and the "first token" claim narrowed to this sentence;
  attention's share of GPT-2 corrected to 22.8% ("under a quarter", asserted); LN spelled out;
  the real-head titles no longer collide.

Drafted 2026-10-05 after the critique and two interview rounds in `LLM_CHAPTER_PLAN.md` (section
"LLM-3 to LLM-5 Attention and the transformer", "LLM-4 critique"). Session 4 of the LLM chapter.
Its job: turn LLM-3's one head into GPT-2's layer - many heads, the MLP, the residual stream with
its norms, the causal mask - and show why the same layer stacks 12 times. It must not re-teach
position (LLM-2 owns it, #68) or one head of attention (LLM-3): callbacks only. LLM-3 ends by
pointing here for multi-head and W_O, real heads, the residual connection, masking and "why 64".

Deck file: `LLM4_transformer_block.tex` (was `L25_transformer_block.tex`, DECISIONS #71).
Restructured in place: every July frame is **kept**, **merged** or **cut**; new frames are
marked **new**.

**Interview decisions (2026-10-05):**

- Scope: **restructure** (like LLM-3).
- The MLP: **a full section on the 3Blue1Brown chapter-7 spine** (stills extracted in July for
  exactly this, never used) plus one by-hand neuron and the 66.6% number.
- Cold open: **facts GPT-2 has stored, half-stored or missing** - Messi (stored), Mkhitaryan
  (missing), Ella Fitzgerald and Elliott Smith (both missed, to different degrees). People and
  prompts chosen by the instructor after three probe runs of the cached GPT-2.
- By-hand frames: **all four** - two heads asking two questions; an MLP neuron; LayerNorm on a
  vector; the causal mask on LLM-3's creature grid.
- Real heads: **GPT-2, one local run** (the e5 figures go).
- Norm order: **GPT-2's** (norm first, add back); 2017's as a footnote.
- Extras: **GPT-2's 124M added up; the superposition teaser.** Not picked: a NumPy block (so the
  "You can now build one" ending goes); moving the shuffle proof into the block section (it stays
  one frame, closing the many-heads section).

**Numbers:** toy values for every by-hand frame (LLM-3's vectors, read as [adjective, fluffy,
blue, noun], plus one new slot, see frame 20), asserted in the figure script. Real numbers only
for claims about GPT-2, all from what is local: the fact probes and the real heads from one run of
the cached model (`--gpt2`, stored in `results/`); parameter counts from the cached
`model.safetensors` header (read 2026-10-05: attention 2,362,368, MLP 4,722,432, norms 3,072 per
block; 12 blocks 85,054,464; total 124,439,808).

**Visual rule for the 3b1b material:** one beat, one picture (as in LLM-3). The chapter-7 stills
were checked by eye on 2026-10-05: all 13 show what their names say.

---

## Frames

Figures: a new script `py_src/block_figs.py` (toy numbers asserted; `--gpt2` once for the probes
and real heads -> `results/block_gpt2.json`; `save()` refuses clipped text; drawn at slot size),
the 3b1b stills in `fig/borrowed/3b1b/`, and the kept July figures from `py_src/l25_l26_figs.py`
(`l25_permutation.pdf`, `l25_causal_mask.pdf`; to check at build that they are drawn at their slot
size).

### Opening

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 0 | Title | "The Transformer block" / "Many heads, the MLP, and the residual stream" | - | kept (subtitle: "position" out) |
| 1 | Where we are: one forward pass | "Today: the rest of one layer - many heads and the MLP - and why it stacks 12 times." | `roadmap_block_pass.pdf` | kept, caption fixed |
| 2 | Where we are: the life of a model | as now | `roadmap_block_life.pdf` | kept |
| 3 | What does GPT-2 write next? | **Predict:** four prompts, no answers: "Lionel Messi plays the sport of", "Henrikh Mkhitaryan plays the sport of", "Ella Fitzgerald was a famous American", "Elliott Smith was a famous American". | - | **new** |
| 4 | Stored, half-stored, missing | GPT-2's top words with probabilities: Messi -> football 35.1% (stored); Mkhitaryan -> hockey 9.8%, football 4.9% (missing); Ella -> actress 15.4%, singer 6.6%; Elliott -> writer 4.6%, singer 2.9% (a weaker trace). No word in these sentences says "football" or "singer", and attention only moves information between the tokens that are there. So where does a fact live? | `blk_fact_probes.pdf` (four rows of top-5 bars, the right answer highlighted) | **new** |
| 5 | Three things stand between one head and GPT-2 | (1) One head asks one question -> many heads. (2) Tokens talk but never think alone -> the MLP, where facts live. (3) One layer is not a network -> the residual stream and its norms; and a decoder must not read the future -> the mask. One line recalls LLM-3's head. | - | kept, rewritten (position out; absorbs "Where we got to") |
| 6 | Outline | | - | kept |

### 1. Many heads

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 7 | [plain] transition | "Many heads" / "One head asks one question." | - | kept (new line) |
| 8 | By hand: two heads, two questions | Creature again. Head 1 is LLM-3's ("any adjectives?": 0.47, 0.47, 0.06). Head 2 asks "what colour?" (its key reads "blue"): creature looks mostly at blue. Each head hands over 2 numbers; side by side they make 4; $W_O$ mixes them back into the stream. Both heads only look backwards, so the mask (section 3) changes nothing for creature. | `blk_two_heads.pdf` (two small weight rows + the concatenated output) | **new** |
| 9 | Run it $h$ times, in parallel | MultiHead = Concat(head_1 ... head_h) $W_O$, in GPT-2's numbers: 12 heads x 64 = 768 (LLM-3's "why 64"); all twelve heads' $W_Q, W_K, W_V$ in one 768 x 2,304 matrix; $W_O$ is `c_proj`, 768 x 768, so every output number can draw on every head. Same budget as one 768-wide head: splitting, not adding. | - | kept (512/8 -> GPT-2's 768/12; notation $W_Q$) |
| 10 | Multi-head, the picture | Full-bleed with caption. | `multihead.png` | kept |
| 11 | Do heads actually specialise? | Head-entropy chart for one GPT-2 layer, on the creature sentence. | `blk_gpt2_entropy.pdf` | kept, redone on GPT-2 |
| 12 | Two heads, same layer, same sentence | Two contrasting GPT-2 heads; the upper triangle is empty (causal). If many heads park their weight on the first token (measured, not assumed), one line names it. | `blk_gpt2_two_heads.pdf` | kept, redone on GPT-2 |
| 13 | What the specialisations look like | The "kinds of head" table, with the suspicion box. | - | kept |
| 14 | One head or twelve: still blind to order | The general proof LLM-2 promised, in one frame: the measured $1.1 \times 10^{-16}$ plus the three reasons (score uses only vectors, softmax over a set, a sum). Callback: that is why LLM-2 adds positions before the first layer. | `l25_permutation.pdf` | merged ("Predict first" + "No. identical" + "Why that had to be true") |

### 2. The MLP: where facts live

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 15 | [plain] transition | "The MLP: where facts live" / "Attention moves information around. Something has to store it." | - | **new** |
| 16 | 3b1b's version of the question | Full-bleed: "Michael Jordan plays the sport of ___" - their example; ours was Messi. | `ch7_mj_hook.jpg` | **new** |
| 17 | Where the MLP sits | After attention, every token goes through the same small network, alone; no token sees another here. | `ch7_flow_mlp.jpg` | **new** (replaces `attn_mlp_flow.png`, one picture per beat) |
| 18 | A direction is a question | Callback LLM-2: a dot product with a direction measures how much of an idea a vector holds ("First name Michael": 0.91). | `ch7_direction_dotprod.jpg` | **new** |
| 19 | Up-projection: each row asks | $W_{up}\,x + b$: each row is one such question, with a bias as a threshold. GPT-2: 3,072 rows, 4 x 768. | `ch7_upproj_rows.jpg` | **new** |
| 20 | ReLU makes an AND gate | With bias -1, "Michael" and "Jordan" must both be there for the value to be positive; ReLU zeroes the rest. | `ch7_relu_andgate.jpg` | **new** |
| 21 | Down-projection: each column adds | Each neuron that fires adds its column - a direction such as "basketball" - into the stream. | `ch7_downproj_columns.jpg` | **new** |
| 22 | By hand: a neuron on creature | Creature after LLM-3 = (0, 0.47, 0.47, 2.13) plus a fifth slot, "cute", that no word filled: 0. A neuron ReLU(fluffy + 0.5 noun - 1.2) gives 0.335 for this creature, 0 for a plain creature (0, 0, 0, 2) and 0 for fluffy alone: it fires on "fluffy AND creature". Its column writes about 1.0 into "cute". A feature no word carried, now in the vector. (Numbers to tune and assert at build.) | `blk_neuron.pdf` | **new** |
| 23 | The whole MLP | Linear -> ReLU -> Linear -> add back; the same for every token. | `ch7_mlp_full.jpg` | **new** |
| 24 | Two thirds of the weights | One GPT-2 block: MLP 4,722,432, attention 2,362,368, norms 3,072 - the MLP is 66.6%. 3b1b's GPT-3 table shows the same share. Evidence that facts sit in these layers: Geva et al. (2021), Meng et al. (2022) - to verify. Back to the cold open: what was written about often got stored. | `blk_params_block.pdf` (one bar per part, labelled) | **new** |
| 25 | More facts than dimensions? | Teaser: in many dimensions, vectors can be nearly perpendicular (89-91 degrees) in far greater numbers than the dimension count - room for more features than neurons. More in the interpretability chapter. | `ch7_nearly_perp.jpg` | **new** (picked extra) |
| 26 | Johnson-Lindenstrauss | Full-bleed with caption: the count grows exponentially with the dimension. | `ch7_jl_lemma.jpg` | **new** (picked extra) |

### 3. The block

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 27 | [plain] transition | "The block" / "What you actually stack." | - | kept |
| 28 | The Transformer block | GPT-2's order: norm, attention, add; norm, MLP, add. Attention is the only place tokens see each other; the MLP holds most weights; norm and residual are L15 and L17. | `blk_block.pdf` (redrawn, norm first) | kept, redrawn |
| 29 | The residual stream is the real object | $x \leftarrow x + \text{Attn}(\text{LN}(x))$, $x \leftarrow x + \text{MLP}(\text{LN}(x))$: the stream is never normalised; each sublayer reads a normalised copy and writes a correction. Gradients reach early layers (L17). Footnote: the 2017 paper normalised after the add. | - | kept, rewritten |
| 30 | By hand: LayerNorm on creature | (0, 0.47, 0.47, 2.13): mean 0.77, spread 0.81 -> (-0.95, -0.37, -0.37, 1.68); then times a learned scale, plus a learned shift (1 and 0 here). Each token on its own, unlike BatchNorm across the batch (L15). | small table | **new** |
| 31 | Masking: a decoder must not read the future | $-\infty$ above the diagonal before softmax; weights exactly 0. | `l25_causal_mask.pdf` | kept |
| 32 | By hand: the mask on creature's grid | LLM-3's learned grid, masked: fluffy sees only itself (1, 0, 0); blue (0.5, 0.5, 0); creature's row does not change (0.47, 0.47, 0.06) - its adjectives came first. | `blk_mask_grid.pdf` | **new** |
| 33 | Why that trick matters | One pass, $n$ training signals - this deck is the home; LLM-5 and LLM-7 call back. | - | kept |
| 34 | Modern block recipe | Norm first (GPT-2 already), RMSNorm, SwiGLU - with sources (Xiong et al. 2020; Zhang and Sennrich 2019; Shazeer 2020), "no warmup" softened, the unsourced "a few percent" line cut. | - | kept, sourced |
| 35 | Now stack it | GPT-2: 12 blocks; each maps $n \times 768$ to $n \times 768$, which is why they stack. Tokens talk, then think alone, 12 times. | `deep_stack.png` | kept (GPT-2's numbers) |
| 36 | Where GPT-2's 124M live | 12 blocks x 7,087,872 = 85,054,464; token table 38,597,376 (LLM-2's 31%); position table 786,432; final norm 1,536; total 124,439,808. | `blk_params_124m.pdf` (one bar per part) | **new** (picked extra) |

### Recap and close

| # | Frame | Content | Figure | Status |
|---|---|---|---|---|
| 37 | Recap | Rewritten: heads, the MLP and facts, the stream and its norms, the mask, 124M. | - | kept, rewritten |
| 38 | Next on the map | -> LLM-5. | `roadmap_transformers_pass.pdf` | kept |

### Cut

- "Where we got to" (absorbed by frame 5); the Position section's "Predict first" and "No.
  identical" (merged into frame 14) and both sinusoid frames (LLM-2 owns them; the second also
  carried the wrong extrapolation claims).
- The e5 figures `l25_head_entropy.pdf`, `l25_real_heads.pdf` (replaced by GPT-2's).
- `attn_mlp_flow.png` (the flow beat is `ch7_flow_mlp`).
- The [plain] "You can now build one" ending (no NumPy block was picked; one ending stays).

---

## Open questions - answered 2026-10-05

1. **A fifth toy slot, "cute"**, for the neuron in frame 22 (instructor's pick): the MLP writes a
   feature no word had (0.335 -> about 1.0), the same story as Messi -> football. Not: writing
   into an existing slot.
2. **Head 2 asks "what colour?"** in frame 8 (its key reads the "blue" slot; only looks backwards,
   so the mask changes nothing for creature). Not: "which noun do I describe?" (would look
   forward).

## Self-review (2026-10-05, inline, before building)

1. **Frame 4 overclaims.** "Attention only moves information between the tokens that are there,
   so the fact must be stored elsewhere" ignores the embedding table: the vector for "Messi" could
   already carry "football". Say it straight: attention can only blend what the token vectors
   hold, so the fact is either in a token's own vector already or added on the way; the evidence
   (Geva 2021, Meng 2022) points mostly to the MLPs. Frame 24 phrases it as evidence, not proof.
2. **A tokenization callback is missing.** "Mkhitaryan" is almost surely several GPT-2 tokens and
   "Messi" maybe one or two: the `--gpt2` run records the tokens, and frame 4 can say "a rare
   name is many pieces and little text" (LLM-1) if the measurement says so. The frequency claim
   ("what was written about often got stored") needs a source (Kandpal et al. 2023, long-tail
   knowledge) or softer words.
3. **The fifth slot appears mid-lecture.** Frame 22 frames it as "a slot that was always there and
   no word used" (real vectors have 768 unnamed slots), and says that attention first brought
   fluffy into creature's vector (LLM-3), which is what lets the MLP AND them - 3b1b's "Michael"
   and "Jordan" are the same mechanism.
4. **Stale callbacks.** July's "BatchNorm (L15)" and "ResNet (L17)" predate the CNN chapter's
   renaming (now `48`-`52`); find the decks that teach residual connections and normalisation and
   cite them the way LLM-6 cites ("[14]").
5. **Frame 8 must stay one screen of arithmetic:** pick each head's $W_V$ and $W_O$ so every
   product is 0 or 1, and show the concatenation as two short rows.

## To verify before building

- Geva et al. 2021 (feed-forward layers as key-value memories) and Meng et al. 2022 (factual
  associations located in mid-layer MLPs), exact claims and venues.
- Xiong et al. 2020 (pre-LN trains without warmup), Zhang and Sennrich 2019 (RMSNorm), Shazeer 2020
  (SwiGLU); Elhage et al. 2022 (toy models of superposition) for the teaser.
- GPT-2 paper: layer norm moved to the input of each sub-block, plus a final layer norm.
- `l25_permutation.pdf` and `l25_causal_mask.pdf`: drawn at their slot size? (`l25_l26_figs.py`)
- The real heads: which layer, and whether a first-token "sink" shows up (measure, then say).
