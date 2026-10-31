# 02 - Armenian through a tokenizer's eyes (practical after LLM-1)

Built 2026-10-28. Notebook: `02_armenian_tokens_solution.ipynb`, generated and executed by
`py_src/build_tokenization_practical_nb.py`. Grown out of the August glitch hunt
(`ml/claude_projects/armenian_glitch_token_hunt/`, DECISIONS #26); everything was re-measured.

## Instructor decisions (2026-10-28)

- The recommended three-part design: build one, audit the pros, hunt a glitch.
- Solution notebook only, run by the instructor in class; no student version.
- August's eight tokenizers plus the 2025-2026 ones (DeepSeek-V3, Llama 4, Qwen3, gpt-oss, Qwen3.5,
  Gemma 4).
- Video number 02 (after LLM-1's lecture).
- Live API calls in class are fine ("u can reexecute").

## Structure

1. **Build one.** Armenian in bytes (2 per letter, GPT-4 = 1 token per byte), `get_stats` + `merge`,
   training on 100,000 characters (500 merges, ~30 s), the first merges, half-letter tokens, letters
   becoming tokens, compression curve, longest tokens and the regex pre-split, encode / decode,
   the reverse tax on English.
2. **Audit the pros.** 14 tokenizers: Armenian tokens, LLM-1's line and gloss, the per-meaning
   premium on NTREX-128, characters per token on Wikipedia, same-vocabulary pairs, first Armenian
   merge, exploding words.
3. **Hunt a glitch.** GPT-4o's never-seen Armenian tokens over 20M characters, two reasons
   (swallowed by a longer token; the classical spelling), the four newer tokenizers (none), the live
   probe on three models sharing the vocabulary (repeat / quoted repeat / spell), gpt-oss's thinking.

## What the build found (versus what the design assumed)

| Assumed while designing | Measured |
|---|---|
| GPT-4o's never-seen tokens come from text in the classical spelling | Only partly. 6 of its Armenian tokens use եւ, 20 use և. One never-seen token is explained by spelling (որովհ only starts the classical որովհետեւ), the other by a longer merge that always wins (վերաբերյ inside վերաբերյալ). |
| The 2025-2026 tokenizers would give their own glitch candidates to probe | None has a single never-seen Armenian token: they own 34-250 Armenian tokens and all are used. Part 3 probes only the o200k family. |
| A per-character premium overstates what Armenian costs per meaning (raised by the "you should know" plugin) | The opposite, slightly: on NTREX, Armenian news text is 1.07x as long as the same English text, so per meaning GPT-4 is x9.16 (x8.57 per character) and GPT-4o x1.73 (x1.62). The notebook uses the per-meaning premium. LLM-1's single line (19 vs 17 tokens on GPT-4o) is the outlier. |
| Newer models bring better Armenian tokenizers | Two of the "new" ones are old: gpt-oss = GPT-4o's vocabulary, Qwen3 = Qwen 2.5's. XLM-R (2019) is still the cheapest (x1.32). |
| BPE's first merges would be syllables | The first merge is the letter ա rebuilt from its two bytes; 37 of 38 lowercase letters become tokens, 23 of the first 100 merges are half-letters. է never stands alone: BPE learns " է", "ում է", "ել է". |

## Build notes

- `import transformers` 5.15 takes ~70 s on this laptop and `AutoTokenizer` scans every model module;
  the notebook reads `tokenizer.json` with the `tokenizers` library instead (seconds).
- OpenRouter returns no `usage` when a model replies with nothing (the glitch itself), and its
  per-call record returns 404 for those calls. The notebook records such calls as cost "not
  reported" and takes the real spend from the account total before and after the probe.
- gpt-oss-20b thinks before answering: with an 800-token budget two calls ran out mid-thought
  (finish_reason "length"); the notebook gives it 3,000 and lists any truncated call separately.
- The probe's answers change between runs at temperature 0. The prose describes the pattern, not
  exact strings.
