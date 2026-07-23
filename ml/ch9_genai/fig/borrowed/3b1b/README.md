# 3Blue1Brown borrowed frames (GenAI chapter)

Screenshots extracted from two 3Blue1Brown "Deep Learning" videos for the attention/transformer
decks (L24/L25). Extracted 2026-07-19 with `yt-dlp` (720p) + `ffmpeg` single-frame grabs.
Transcripts (with timestamps) live in `../../research/`.

**Credit:** Each frame used on a slide carries a small "3Blue1Brown" credit line (instructor
decision 2026-07-21: just credit the source, drop the license tag - matches the standing
no-license-caveat rule). Earlier the credit read "3Blue1Brown, CC BY-NC-SA"; that tag was removed
from L24 (macros `\bbb`/`\bbbslide`/`\bbbcap`, the title page, and the Sources frame) on 2026-07-21.

**Sources:**
- Ch5 = "Transformers, the tech behind LLMs" (`wjZofJX0v4M`)
- Ch6 = "Attention in transformers, step-by-step" (`eMlx5fFNoYc`)
- Ch7 = "How might LLMs store facts" (`9-Jl0dxWQs8`)

| File | Video @ time | Intended use |
|---|---|---|
| `nextword_distribution.png` | Ch5 01:40 | L24/L25 - a transformer predicts a *distribution* over next words |
| `attention_updates_meaning.png` | Ch5 04:08 | L24 - "a machine learning model" vs "a fashion model" (context updates meaning) |
| `attn_mlp_flow.png` | Ch5 04:35 | L25 - the data flow: attention block -> MLP block |
| `deep_stack.png` | Ch5 05:20 | L25 - many attention/MLP blocks stacked ("Many") |
| `chatbot_system_prompt.png` | Ch5 06:15 | L25 - prediction -> chatbot via a system prompt |
| `transformer_nexttoken.png` | Ch5 06:30 | L25 - transformer -> next-token distribution (chatbot) |
| `gpt3_weights.png` | Ch5 11:06 | L25 - 175B weights organized as ~28k matrices |
| `embeddings_3d.png` | Ch5 13:44 | L24 - words as vectors in a high-dim space |
| `tower_neighbors.png` | Ch5 14:40 | L24 - semantic neighbors of E(tower) |
| `king_queen_analogy.png` | Ch5 15:05 | L24 - E(woman)-E(man) direction; king - man + woman ~ queen |
| `king_context.png` | Ch5 18:47 | L24 - an embedding soaks up context (King -> Macbeth) - what attention *does* |
| `context_size.png` | Ch5 19:50 | L25 - the fixed context window |
| `snape_prediction.png` | Ch5 20:35 | L25 - next-token example ("...least favourite teacher" -> Snape 0.78) |
| `unembedding.png` | Ch5 19:30 | L25 - last vector -> logits |
| `unembedding_matrix.png` | Ch5 21:05 | L25 - the unembedding matrix -> ~50k values |
| `softmax_distribution.png` | Ch5 22:35 | L25 - softmax output as a probability bar chart |
| `temperature.png` | Ch5 24:00 | (later - decoding) Temp=0 vs Temp=5 story generation |
| `mole_polysemy.png` | Ch6 02:05 | L24 - "why context matters" opener (shrew mole / mole of CO2 / biopsy) |
| `eiffel_context.png` | Ch6 03:10 | L24 - context moves an embedding (tower -> Eiffel Tower) |
| `query_creature.png` | Ch6 06:20 | L24 - running example + query intuition ("any adjectives in front of me?") |
| `kq_dotgrid.png` | Ch6 08:50 | L24 - the key.query dot-product grid |
| `attention_pattern.png` | Ch6 10:00 | L24 - the softmax'd attention pattern |
| `single_head_full.png` | Ch6 15:40 | L24 - the full single-head pipeline on one frame (summary) |
| `masking_softmax.png` | Ch6 12:15 | L24/L25 - unnormalized -> softmax -> normalized (+ causal masking) |
| `value_matrix.png` | Ch6 14:35 | L24 - the value matrix / value vectors |
| `value_delta_e.png` | Ch6 15:15 | L24 - values -> delta-E -> updated embedding |
| `value_lowrank.png` | Ch6 17:15 | L24 - (optional aside) value map as a low-rank factoring |
| `multihead.png` | Ch6 20:50 | L25 - multi-head attention (96 heads in GPT-3) |
| `gpt3_params.png` | Ch6 24:38 | L25 - attention parameter breakdown / scale |

Notes:
- `transformer_nexttoken` / `chatbot_system_prompt` overlap (pick one at build); same for
  `unembedding` / `unembedding_matrix`.
- All are 1280x720. Re-extract from the source videos (kept in the session scratchpad during the
  build) if a sharper crop is needed.

## Full-bleed showcase stills (`fs_*.jpg`, L24, added 2026-07-21)

13 full-slide stills at **1080p** (Ch6 `eMlx5fFNoYc` re-fetched at 1080p for crisp full-screen),
used with the `\bbbslide` / `\bbbcap` macros in `L24_attention.tex` - a full-bleed image per
teaching beat, sitting next to the small house-style frame it reinforces.

| File | Ch6 @ time | Beat |
|---|---|---|
| `fs_mystery` | 04:11 | "therefore the murderer was ???" -> the last vector must encode all context |
| `fs_eiffel` | 03:08 | tower -> Eiffel Tower vector move |
| `fs_creature` | 04:12 | running example: "a fluffy blue creature roamed the verdant forest" |
| `fs_query` | 06:58 | E_4 -> W_Q -> Q_4 ("any adjectives in front of me?") |
| `fs_key` | 08:06 | keys advertise themselves ("I'm an adjective!") |
| `fs_qkspace` | 07:40 | W_Q/W_K project into one shared query/key space |
| `fs_dotgrid` | 08:48 | the key.query dot-product grid |
| `fs_pattern` | 09:52 | the softmax'd attention pattern |
| `fs_masking` | 12:28 | masking (shown as an aside; full treatment in L25) |
| `fs_formula` | 10:28 | softmax(K^T Q / sqrt(d_k)) V |
| `fs_value` | 13:33 | the value matrix |
| `fs_deltae` | 15:12 | weighted values summed into delta-e |
| `fs_onehead` | 15:48 | the whole single head on one frame |

## Ch7 "store facts" stash for L25/L26 (`ch7_*.jpg`, added 2026-07-21)

13 curated **1080p** stills from Ch7 (`9-Jl0dxWQs8`), stashed for the **MLP / feed-forward** part
of **L25** (the transformer block) and the **param-count + superposition** payload for **L26**.
Not yet wired into any deck. Transcript: `../../research/3b1b_ch7_store_facts.md`.

| File | Ch7 @ time | Beat / intended use |
|---|---|---|
| `ch7_mj_hook` | 00:06 | "Michael Jordan plays the sport of ___" -> where do facts live? (L25 MLP cold-open) |
| `ch7_flow_mlp` | 01:25 | transformer flow, MLP block highlighted vs attention (L25 "where the MLP sits") |
| `ch7_direction_dotprod` | 05:18 | a direction encodes an idea; dot product ~ 0.91 (L25 toy-example setup) |
| `ch7_upproj_rows` | 07:42 | up-projection: rows as dot-product questions; the "Michael+Jordan" row -> 2 |
| `ch7_bias_andsetup` | 09:12 | bias = -1, so the value is positive iff the full name (AND-gate setup) |
| `ch7_upproj_dims` | 09:42 | W_up is 49,152 rows = 4x the embedding dim (GPT-3) |
| `ch7_relu_andgate` | 11:22 | ReLU clips to a clean 0/1 neuron -> an AND gate (**hero**) |
| `ch7_neurons` | 11:48 | "neurons" = these values; the classic dots-and-lines NN picture (callback L14) |
| `ch7_downproj_columns` | 12:35 | down-projection: columns as directions; first column = "basketball" |
| `ch7_mlp_full` | 14:12 | the whole block: Linear -> ReLU -> Linear -> + residual (**hero** summary) |
| `ch7_param_table` | 16:48 | full 175B param table; MLP ~ 116B = 2/3 of GPT-3 (L26 scale) |
| `ch7_nearly_perp` | 18:48 | nearly-perpendicular directions (89-91 deg) intro (L26 superposition) |
| `ch7_jl_lemma` | 20:12 | Johnson-Lindenstrauss: #near-perp vectors ~ exp(eps*N) (**hero**; L26 superposition) |

Note: the superposition demo (100-dim, 10,000 random vectors, angle histogram converging to
~90 deg) is better **reproduced** as a `py_src/` matplotlib figure (seed 509) than borrowed -
flagged for when L26 is built.
