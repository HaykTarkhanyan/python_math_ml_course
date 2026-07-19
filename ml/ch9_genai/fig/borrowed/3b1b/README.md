# 3Blue1Brown borrowed frames (GenAI chapter)

Screenshots extracted from two 3Blue1Brown "Deep Learning" videos for the attention/transformer
decks (L24/L25). Extracted 2026-07-19 with `yt-dlp` (720p) + `ffmpeg` single-frame grabs.
Transcripts (with timestamps) live in `../../research/`.

**License:** 3Blue1Brown videos are **CC BY-NC-SA 4.0**. Each frame used on a slide carries a
small "3Blue1Brown, CC BY-NC-SA" credit line (per the instructor's 2026-07-19 decision). This is
the one exception to the standing "no per-image credit line" rule, because of the NC license.

**Sources:**
- Ch5 = "Transformers, the tech behind LLMs" (`wjZofJX0v4M`)
- Ch6 = "Attention in transformers, step-by-step" (`eMlx5fFNoYc`)

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
