# LLM-3 - Decoding: from probabilities to text - outline (built 2026-10-05: `LLM3_decoding.tex`)

Drafted 2026-10-05 after the critique and interview in `LLM_CHAPTER_PLAN.md` (LLM-3). Session 6 of
the LLM chapter (`ml/14_llms`), the map's last box. Its job: the model hands over a probability
for every token; turn that into text, and know what each knob does. It must not re-teach the
generation loop (L21) or logits -> softmax (L26): one recap frame, then new material.

Deck file (until delivery gives it a playlist number): `LLM3_decoding.tex`.

**Interview decisions (2026-10-05):**

- Cold open: **GPT-2 loops** - always take the most likely token and it repeats itself.
- Real systems: **temperature 0 is not deterministic; speculative decoding (moved here from
  LLM-11); watermarking.** Not: providers restricting the knobs.
- Min-p: **taught, with the 2025 critique.**
- Hands-on: **not in the slides.**

**Numbers (instructor 2026-10-05, now in `ml/SLIDE_STYLE.md`):** toy, demonstrative values by
default. Real numbers only where a slide states a fact about a real model, and only from what is
already local:

| Where | Source |
|---|---|
| Cold open: GPT-2's greedy continuation looping | one greedy run of the cached GPT-2 at build time (no download) |
| The recap and the by-hand top-p frame | the map's real top-5 after "The cat sat on the": floor 7.6%, bed 6.5%, couch 5.4%, ground 5.2%, edge 4.8% (`results/roadmap_gpt2.json`) |
| Everything else | toy distributions and worked examples, labelled as such |

---

## Frames

Figures: one small script, `py_src/decoding_figs.py` (toy numbers, drawn at slot size; the
numbers quoted on slides asserted, as before), plus the map frames from `py_src/llm_roadmap.py`
(key `decoding`, already drawn).

### Opening

| # | Frame | Content | Figure |
|---|---|---|---|
| 0 | Title | "Decoding: from probabilities to text" | - |
| 1 | Where we are: one forward pass | Map, Decoding lit. "Today: the last box - the model gives a probability for every token; which one do we write?" | `roadmap_decoding_pass.pdf` |
| 2 | Where we are: the life of a model | "Later: ..." | `roadmap_decoding_life.pdf` |
| 3 | Cold open: always take the best word | **Predict first.** If the model always writes its single most likely next token, what does the text look like? | - |
| 4 | It loops | GPT-2's real greedy continuation of a short prompt, repeating itself. "So why not just take the top token? Today's answer." | `dec_greedy_loop.pdf` (text in a box, the repeat highlighted) |
| 5 | Outline | | - |

### 1. What the model hands you

| # | Frame | Content | Figure |
|---|---|---|---|
| 6 | [plain] transition | "What the model hands you" / "A probability for every token. Writing means choosing one." | - |
| 7 | One step, one distribution | Recap in one frame: the loop (L21), logits -> softmax (L26). On the map's sentence the top choice, "floor", has only 7.6%; the top five together 29.5%; the other 50,252 tokens share 70.5%. Choosing is a real decision. | `dec_map_top5.pdf` (the 5 bars + "everything else 70.5%") |
| 8 | When to stop | Three stop rules: the end-of-text token (LLM-1's `<|endoftext|>`; chat models use an end-of-turn token, LLM-6), a maximum number of tokens, stop strings. Catch: a stop string can be split across tokens, so it is checked on the text, not on token IDs. | - |
| 9 | Stopping in the middle of a letter | A byte-level model can be cut off between the two bytes of an Armenian letter: ayb is D5 A1 (LLM-1), and D5 alone is not valid UTF-8. So a streaming app must hold back bytes until a character is complete. | small byte diagram (reuse LLM-1's style) |

### 2. Option 1: take the most likely

| # | Frame | Content | Figure |
|---|---|---|---|
| 10 | [plain] transition | "Option 1: take the most likely" / "Greedy, then a smarter greedy." | - |
| 11 | Greedy | argmax at every step; same input, same output (on one machine - section 5 complicates this). Why it loops: once a phrase has appeared, repeating it becomes more likely, so each repeat makes the next one likelier (Xu et al., 2022, measured this self-reinforcement - to verify). | - |
| 12 | Why log-probabilities | Toy by hand: 100 tokens at probability 0.1 each = $10^{-100}$; the smallest positive float32 is about $10^{-45}$, so the product becomes 0. Logs turn the product into a sum: $100 \cdot \log 0.1 = -230$. Log-probs are also how APIs report confidence (callback: calibration, [14]). | - |
| 13 | Greedy is short-sighted | Toy tree: step 1 "on" 0.5 vs "in" 0.4; after "on" the best is 0.3, after "in" it is 0.9. Greedy takes "on" and ends at 0.15; "in a" was 0.36. | `dec_greedy_tree.pdf` |
| 14 | By hand: beam search, B = 2 | Keep the 2 best partial sequences at every step (by summed log-prob). The same tree: beam keeps "in" and finds 0.36. Ported from the source deck's B = 2 example, numbers redone. | `dec_beam_tree.pdf` |
| 15 | Beam search: why divide by length | Every extra token multiplies by a number below 1, so plain beam search prefers stopping early. Fix: score = sum of log-probs / $L^\alpha$ ($\alpha$ around 0.6-0.7, from the source deck - to verify). Toy: a 3-token and a 6-token ending. | - |
| 16 | Predict first | "Beam search finds more probable text than greedy. Is more probable text better text?" | - |
| 17 | The probability trap | Holtzman et al. (2020): the per-token probability of real human text jumps around and is often low; beam-search text stays uniformly high - and reads as bland and repetitive. Humans are surprising. | schematic redraw "after Holtzman et al. 2020, Fig. 2" (demonstrative curves) - or their figure with attribution |
| 18 | Where maximising still wins | Tasks with one right answer: translation, speech recognition (to verify which systems default to beam today). Chat and open-ended writing: no. | - |

### 3. Option 2: sample

| # | Frame | Content | Figure |
|---|---|---|---|
| 19 | [plain] transition | "Option 2: roll the dice" / "Sample from the distribution - but which distribution?" | - |
| 20 | Pure sampling and the long tail | Sampling straight from the model fixes the looping but invites junk: tens of thousands of tokens with tiny probabilities add up. Toy: 50,000 tail tokens at 0.000005 each = 25% - one step in four picks something odd. | `dec_long_tail.pdf` (toy) |
| 21 | Temperature | Divide the logits by $T$ before the softmax. The same operation as temperature scaling in [14] and ch11's 45 - there to calibrate, here to steer. $T < 1$ sharpens, $T > 1$ flattens, $T \to 0$ is greedy. | - |
| 22 | By hand: temperature | Ported from the source: logits (2, 1, 0) at $T = 1$: (.66, .24, .09); $T = 10$: (.37, .33, .30); $T \to 0$: (1, 0, 0). | - |
| 23 | Temperature on the map | The map's top-5 renormalised at $T$ = 0.5, 1, 2 (labelled "the top five only"). | `dec_temperature.pdf` |
| 24 | Top-k, and why a fixed k fails | Keep the k best, renormalise. Two toy distributions: peaked ("The capital of Armenia is" -> one token near 0.9) and flat ("My favourite food is" -> dozens of near-equal tokens). k = 5 is too many for the first, too few for the second. | `dec_topk_vs_topp.pdf` (toy, two panels) |
| 25 | Top-p (nucleus) | Keep the smallest set whose probabilities add up to $p$ (Holtzman et al., 2020): it adapts - 1 token when peaked, dozens when flat. Same toy panels. | same figure, top-p overlay |
| 26 | By hand: top-p on the map | $p = 0.2$: floor .076, + bed = .141, + couch = .195, + ground = .247 $\geq$ .2 - keep 4 tokens, renormalise. Real numbers, by hand. | - |
| 27 | Min-p, and how to read a claim | Keep tokens with probability at least min_p $\times$ the top token's (Nguyen et al., ICLR 2025, oral). Toy by hand. Then: a 2025 re-analysis found it does not beat top-p once both are tuned equally, and found reporting problems (arXiv 2506.13681). An oral at a top venue is not the last word. | - |
| 28 | The order of the knobs | Logits -> repetition penalty -> temperature -> softmax -> top-k / top-p / min-p -> sample. Greedy = $T \to 0$ = top-k with k = 1. Watch-out: libraries disagree on the order - Hugging Face applies temperature before the cut-offs, llama.cpp's default chain (I believe) applies it last - and with top-p or min-p the order changes which tokens survive. Verify both before stating it. | `dec_knob_pipeline.pdf` |
| 28b | Which knob fixes what | One summary table, as in LLM-2's "Three options, side by side": greedy / beam / temperature / top-k / top-p / min-p / penalty - what each fixes, what it breaks. Keeps section 3 from reading as a list. | - |
| 29 | Repetition penalties | Two different things share the name: dividing the logit of every token already used (Keskar et al., 2019; Hugging Face, llama.cpp) vs subtracting a fixed amount, or an amount per use (OpenAI's presence and frequency penalties). Toy by hand. (Definitions to verify.) | - |
| 30 | Seeds | Sampling is random: the same prompt gives different answers. A fixed seed repeats them - on the same machine, same software. (Sets up section 5.) | - |
| 31 | Which settings for what | Low temperature for code and facts, higher for brainstorming; top-p around 0.9-0.95 as a common default. Typical ranges, not rules (to verify against two providers' docs). | - |

### 4. Option 3: forbid the wrong tokens

| # | Frame | Content | Figure |
|---|---|---|---|
| 32 | [plain] transition | "Constrain the choice" / "When the output must be valid JSON. Not a third option: a filter on top of either." | - |
| 33 | Masking | At every step, set the probability of every token that would break the format to 0 (logit $-\infty$), renormalise, then pick as usual - greedy or sampled. It stacks with every knob above. | - |
| 34 | By hand: JSON | Ported from the source: schema `{"city": string, "year": integer}`, output so far `{"city": "Yerevan", "year": ` - digits allowed, `"` `{` `[` `null` masked. | `dec_json_mask.pdf` |
| 35 | Tokens are not characters | LLM-1 callback: one token can cover several grammar steps (`":` or `"}`), so the mask checks each token's whole text against the grammar; libraries precompute which tokens are legal in each state. Forcing token splits the model rarely saw can hurt quality (to verify with a source). | - |
| 36 | What it buys, what it does not | Guaranteed format: tool calls (agents, L44), JSON for programs. Not guaranteed truth: a valid "year" can still be the wrong year. And not guaranteed complete: hit the max-tokens limit mid-object and the JSON is cut off anyway. | - |

### 5. Real systems

| # | Frame | Content | Figure |
|---|---|---|---|
| 37 | [plain] transition | "Real systems" / "What changes when a server does the decoding." | - |
| 38 | Predict first | "Temperature 0, same prompt, 1,000 times. How many different answers?" | - |
| 39 | Temperature 0 is not deterministic | Thinking Machines (Sept 2025): 1,000 runs of Qwen3-8B at temperature 0 gave 80 different completions. Cause: a server batches requests, and the batch size changes the order of floating-point additions. Toy: in float32, $(10^8 + 1) - 10^8 = 0$, but $(10^8 - 10^8) + 1 = 1$. A tiny difference flips one argmax, and every later token follows. | - |
| 40 | Why decoding is slow | One full forward pass per generated token, one after another; for each token the hardware mostly waits while all the weights are read from memory (the detail, KV cache included, is LLM-11). Sets up speculative decoding. | - |
| 41 | Speculative decoding | A small model drafts k tokens; the big model checks all k in one pass (the causal mask from L25 gives a prediction at every position at once); keep the agreeing prefix, fix the first mismatch. Ported from the source's "the cat sat on moon" figure. | `dec_speculative.pdf` |
| 42 | By hand: accept or reject | Greedy: accept while the draft equals the big model's argmax. Sampling: accept a draft token with probability $\min(1, p_\text{big}/p_\text{small})$ - toy: 0.3 / 0.6 -> accept half the time; on a reject, sample from what the big model wanted more than the draft did. Together that is rejection sampling, so the output has exactly the big model's distribution: lossless. (Without the second half the claim is false - keep both.) | - |
| 43 | Where the draft comes from | A smaller model of the same family; extra prediction heads on the big model; copying spans from the prompt; multi-token prediction (LLM-11). Speed-ups as reported by a named source, not a range from memory. | - |
| 44 | Watermarking in the sampler | SynthID-Text (Dathathri et al., Nature 2024): Google's production scheme changes only the sampling step - a keyed function quietly favours some candidate tokens; anyone with the key can test a text for that bias; the model itself is untouched. Limits (to verify): weaker on short or heavily edited text. | - |

### Close

| # | Frame | Content | Figure |
|---|---|---|---|
| 45 | Recap | Take the most likely (greedy, beam: good for one-right-answer tasks, bland for writing) / sample (temperature, top-k, top-p, min-p) / forbid (masks for formats); real systems: temperature 0 drifts, speculative decoding is free speed, the sampler can carry a watermark. | - |
| 46 | Next on the map | "Next (LLM-4): where these probabilities come from - pretraining." | `roadmap_pretraining_life.pdf` |
| 47 | Sources & further reading | | - |

---

## Self-review (2026-10-05, quick inline pass) - applied above

- Section 4 was titled "Option 3"; masking is not an alternative to greedy or sampling but a
  filter on top of both. Renamed "Constrain the choice".
- Frame 33 claimed "a valid grammar always leaves at least one legal token" - not something to
  assert; replaced by the real gotcha in frame 36 (max tokens can still cut the JSON off).
- Frame 42: "lossless" needs the resample-on-reject half of the rule; added.
- Frame 28: the knob order is library-specific (temperature before vs after the cut-offs); now a
  watch-out to verify instead of one pipeline stated as fact.
- Frame 11's "repeats make the next repeat likelier" is a claim about real models: now cited
  (to verify). It also foreshadows section 5 ("same input, same output - on one machine").
- Frame 9: "streaming apps hold back bytes" was an unverified claim about real apps; now "must".
- Added 28b, a "which knob fixes what" table: section 3 is 13 frames of knobs, the same
  catalogue risk LLM-2's review flagged.
- Numbers to assert at build: the top-5 sum (29.5% from rounded values; recompute from the JSON),
  top-p by hand (.247 at 4 tokens), temperature by hand (.66/.24/.09; .37/.33/.30), the toy tree
  (0.15 vs 0.36), the long tail (25%), the float32 toy.

## To verify before building

- Holtzman, Buys, Du, Forbes & Choi (2020), ICLR: the human-vs-beam probability figure (which
  figure number), and that top-p is theirs.
- The length-normalisation exponent range (0.6-0.7) and who introduced it.
- Where beam search is still the default (translation, speech).
- Repetition-penalty definitions: Keskar et al. 2019 (CTRL), Hugging Face `repetition_penalty`,
  llama.cpp, OpenAI presence and frequency penalties.
- The order the knobs apply in, in **both** Hugging Face (logits processors / warpers) and
  llama.cpp (default sampler chain) - they are believed to differ on where temperature goes.
- Xu et al. (2022), "Learning to Break the Loop" (NeurIPS): the self-reinforcing repetition claim.
- Min-p: Nguyen et al. (ICLR 2025 oral) and the 2025 critique (arXiv 2506.13681) - exact claims.
- Thinking Machines, "Defeating Nondeterminism in LLM Inference" (Sept 2025): the 80/1,000 number
  and the cause, from the primary post (seen so far only in secondary write-ups).
- Speculative decoding: Leviathan et al. (2023) and Chen et al. (2023); any speed-up quoted must
  name its source.
- SynthID-Text: Dathathri et al., Nature 2024 - mechanism and stated limits.
- "Typical settings" (frame 31) against two providers' current docs.
- The constrained-decoding quality claim in frame 35 (or drop it).

## Sources

`sources/dl4nlp/04_decoding_strategies.tex` (port: temperature worked example, beam B = 2, the
knob pipeline, structured decoding and its JSON example, the three speculative-decoding frames;
cut: contrastive search, the API-parameter table, "Common questions"). The map's top-5 from
`results/roadmap_gpt2.json`. L21's "sample, feed back, repeat" and L26's "From a vector to a word"
as the recap's callbacks.
