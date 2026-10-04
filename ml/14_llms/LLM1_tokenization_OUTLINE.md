# LLM-1 - Tokenization: from text to token IDs - outline (built 2026-10-04: `LLM1_tokenization.tex`)

Drafted 2026-10-04 after the instructor interview. Session 1 of the LLM chapter (`ml/14_llms`), the
first box on the chapter map. Its job: turn text into the integer IDs a model reads, and explain
the bugs that choice causes. It ends at **token IDs**; L24 picks up with "an ID becomes a vector".
It must not lean on attention or embeddings - they come next.

Deck file (until delivery gives it a playlist number): `LLM1_tokenization.tex`.

**Interview decisions (2026-10-04):**

- Cold open: **"How many r's in strawberry?"**
- Unicode / UTF-8: **short, 2-3 frames**.
- BPE: **full mechanics** - idea, merge table by hand, encoding new text, byte-level, the
  pre-tokenization regex, decoding (with the half-a-letter trap).
- WordPiece, Unigram/SentencePiece: **one frame each**.
- Quirks, **all four**: numbers and arithmetic, glitch tokens, code whitespace, special-token
  leaks (plus the strawberry explanation).
- Armenian: **one frame**, "one sentence, two counts".
- Hands-on: **not in the slides**.

**Measured for this outline** (tiktoken 0.13.0, 2026-10-04; `gpt2`, `cl100k_base` = GPT-4,
`o200k_base` = GPT-4o):

| Example | GPT-2 | GPT-4 (cl100k) | GPT-4o (o200k) |
|---|---|---|---|
| Vocabulary size | 50,257 | 100,277 | 200,019 |
| "How many r's in strawberry?" | 7 tokens; ` strawberry` is 1 | same | same |
| `strawberry` with no leading space | `st` `raw` `berry` | `str` `aw` `berry` | `st` `raw` `berry` |
| `1234567` | `123` `45` `67` | `123` `456` `7` | `123` `456` `7` |
| `127 + 677 = 804` | `127` ` +` ` 6` `77` ` =` ` 8` `04` | `127` ` +` ` ` `677` ` =` ` ` `804` | as GPT-4 |
| 3-line Python snippet, 8/12-space indents | 30 tokens | 12 tokens | 12 tokens |
| `ա` (U+0561, 2 bytes `d5 a1`) | 2 tokens, half a letter each | 2 tokens | 1 token |
| The chapter line (57 letters, 102 bytes) vs its English gloss | - | **102 vs 18** | **19 vs 17** |

---

## Outline decisions (instructor, 2026-10-04, second round)

1. **The Armenian frame shows both tokenizers:** 102 vs 18 on GPT-4's (one token per byte) and
   19 vs 17 on GPT-4o's - "the tax was a vocabulary choice" (frame 34). Raised because GPT-4's
   pair alone would tell a 2026 class something the newest tokenizer no longer does. (A mid-turn
   probe without the gloss's final period counted 19 vs 16; the locked gloss ends with "me.", so
   it is 17.)
2. **Glitch tokens: two frames** - the story, then GPT-3's actual replies (frames 32-33).
3. **Length: keep all 37 frames** (about 30 of content).

---

## Frames

Figures: one new script, `py_src/tokenization_figs.py` (tiktoken, measured), plus the three
L21 panels from `13_rnns/py_src/tokenizer_demo.py`, which moves here when this deck is built (plan,
LLM-1). Every count on a slide comes from a script run, not from this table.

### Opening

| # | Frame | Content | Figure |
|---|---|---|---|
| 0 | Title | "Tokenization: from text to token IDs" | - |
| 1 | Where we are: one forward pass | The map, Tokenizer lit. "Today: the first box - how text becomes the integers a model reads." | `roadmap_tokenization_pass.pdf` |
| 2 | Where we are: the life of a model | "Later: ..." | `roadmap_tokenization_life.pdf` |
| 3 | Cold open: How many r's are in strawberry? | **Predict first.** You count 3; LLMs famously answered 2 (2024). Click: what the model sees - ` strawberry` is **one token, one number** in all three tokenizers. "You see ten letters; it sees one ID." By the end: why, and four more bugs with the same cause. | chips: the question, tokenized |
| 4 | Outline | | - |

### 1. Text has to become numbers

| # | Frame | Content | Figure |
|---|---|---|---|
| 5 | [plain] transition | "Text has to become numbers" / "A network multiplies numbers. Someone has to decide what the numbers stand for." | - |
| 6 | Models read IDs, not text | text -> pieces -> integer IDs -> (next lecture) vectors. Callback: L20's "Tokens, minimally" and one-hot over a fixed word list. The vocabulary is a lookup table. | small TikZ arrow strip |
| 7 | Three ways to cut text | Words / characters / subwords: the trade-off (vocabulary size vs sequence length vs unknown words). | `tokenizer_panel1.pdf` (from L21) |
| 8 | Words: the unknown-word problem | A word list must stop somewhere: "unkindness" -> `<UNK>`; every form of a word is a new entry. | chips |
| 9 | Characters: no unknowns, but long | Nothing is ever unknown, but sequences get ~4x longer and each piece means little. Longer sequences cost more later (in words, no attention yet). | chips, measured lengths |
| 10 | Subwords: the middle road | Frequent words stay whole, rare ones split into reusable pieces. | `tokenizer_panel2.pdf` (from L21) |

### 2. From text to bytes (short)

| # | Frame | Content | Figure |
|---|---|---|---|
| 11 | [plain] transition | "First, agree on the alphabet" / "Before we cut text into pieces, we need to know what text is made of." | - |
| 12 | Text is a sequence of code points | Unicode gives every character a number: `a` = U+0061, `ա` = U+0561, an emoji. ~155k characters (verify the Unicode 17 count). Too many symbols to be the alphabet. | chips with code points |
| 13 | UTF-8: one to four bytes per character | `a` -> 1 byte, `ա` -> 2 bytes (`d5 a1`), emoji -> 4. 256 possible bytes: an alphabet that can spell anything, so nothing is ever unknown. | bytes figure |

### 3. Byte-Pair Encoding

| # | Frame | Content | Figure |
|---|---|---|---|
| 14 | [plain] transition | "Byte-Pair Encoding" / "Start from bytes. Merge the most frequent pair. Repeat." | - |
| 15 | BPE: the idea | A 1994 compression trick (Gage) reused for translation (Sennrich, Haddow & Birch, 2016). The training loop in 4 lines; the vocabulary = base symbols + merges. | boxed algorithm |
| 16 | Predict first: which pair merges first? | Toy corpus with counts (Sennrich's "low / lower / newest / widest"). | LaTeX table |
| 17 | BPE by hand, merges 1-3 | Count pairs, merge, recount - every number shown. | LaTeX table |
| 18 | BPE by hand, merges 4-6, and the result | The merge list and the final vocabulary. | LaTeX table |
| 19 | Encoding new text | Apply the merges **in the order learned**: "lowest" -> `low` + `est`. A word never seen in training still gets pieces. | chips |
| 20 | Byte-level BPE | GPT-2 (Radford et al., 2019): the base alphabet is the 256 bytes, so there is never an unknown token. GPT-2's 50,257 = 256 bytes + 50,000 merges + 1 special token. | - |
| 21 | Pre-tokenization: never merge across words | A regex splits the text first, so `dog.` `dog!` `dog?` do not become three unrelated tokens. GPT-2's regex vs GPT-4's (digits in groups of up to 3). | regex chunks of one sentence |
| 22 | Decoding: IDs back to text | IDs -> bytes -> text. **The trap:** in GPT-4's tokenizer `ա` is two tokens, `\xd5` and `\xa1`, and neither is valid text alone; a streaming chat app must hold bytes until a character is complete. Our own `_learnings` file is a real instance. | bytes figure |
| 23 | The tokenizer is its own model | Trained separately, on its own data, then frozen before the LLM ever trains. Vocabulary size is a trade-off: 50,257 -> 100,277 -> 200,019 means fewer tokens per text but a bigger input and output table. | measured bar chart: same English + code sample, three tokenizers |

### 4. Other subword algorithms

| # | Frame | Content | Figure |
|---|---|---|---|
| 24 | [plain] transition | "BPE is not the only way" / "Two other answers to 'which pieces?', still in use." | - |
| 25 | WordPiece (BERT) | Merge the pair that most raises the training likelihood, not the most frequent one; `##` marks a piece that continues a word ("play" "##ing"). Schuster & Nakajima 2012; BERT. | chips |
| 26 | Unigram and SentencePiece | Start big and prune the pieces that matter least (Kudo, 2018); SentencePiece (Kudo & Richardson, 2018) works on raw text, space included (`▁`). Which current models use it: verify at build. | chips |

### 5. Why LLMs trip over tokens

| # | Frame | Content | Figure |
|---|---|---|---|
| 27 | [plain] transition | "Why LLMs trip over tokens" / "Five famous failures. One cause." | - |
| 28 | Back to strawberry | The r's are inside a token the model never takes apart. Models that get it right today spell the word out first - turning one token back into ten letters. | chips: one token vs spelled out |
| 29 | Numbers and arithmetic | GPT-2: `127 + 677 = 804` -> `127` ` +` ` 6` `77` ` =` ` 8` `04` - the digits of one number land in different pieces. GPT-4: groups of 3. Some models split every digit (which ones: verify at build). | chips, measured |
| 30 | Code whitespace | The same 3-line Python snippet: 30 tokens in GPT-2 (spaces spent one by one), 12 in GPT-4 (runs of spaces are single tokens). | chips, measured |
| 31 | Special tokens, and how they leak | `<|endoftext|>` and friends mark boundaries (chat-role tokens: LLM-5). If user text is encoded with special tokens switched on, typing the string becomes the real control token. tiktoken refuses by default. | chips |
| 32 | Glitch tokens: SolidGoldMagikarp | A Reddit username made it into the GPT-2/3 vocabulary but almost never into the training text, so its vector was never trained (Rumbelow & Watkins, 2023). | - |
| 33 | Glitch tokens: what GPT-3 said | Its actual replies when asked to repeat the token (from the source deck; recheck against the original post). | quotes |
| 34 | The Armenian tax: one sentence, two counts | The chapter line vs its English gloss: **102 vs 18** on GPT-4's tokenizer (102 tokens = 102 UTF-8 bytes, one token per byte); **19 vs 17** on GPT-4o's. "The tax was a vocabulary choice." | `tokenizer_panel3.pdf`, regenerated with both tokenizers |

### Close

| # | Frame | Content | Figure |
|---|---|---|---|
| 35 | Recap | Text -> bytes -> BPE pieces -> IDs; the tokenizer is a separate, frozen model; its choices explain strawberry, arithmetic, code, leaks, glitch tokens, Armenian. | - |
| 36 | Next on the map | "Next (L24): an ID becomes a vector, and the vectors start talking to each other." | `roadmap_attention_pass.pdf` |

---

## To verify before building (do not bake in from memory)

- The strawberry failure: a dated, citable example (which model, when), and how 2026 models get
  it right (frame 28 claims "by spelling the word out first" - check before saying it).
- Unicode's current character count (17.0).
- Gage 1994; Sennrich, Haddow & Birch 2016; Radford et al. 2019; Schuster & Nakajima 2012;
  Kudo 2018; Kudo & Richardson 2018; Rumbelow & Watkins 2023 - years and venues.
- Which current open models use SentencePiece vs byte-level BPE, and which split digits one by one.
- GPT-3's SolidGoldMagikarp replies, against the original LessWrong post.
- GPT-2's and GPT-4's pre-tokenization regexes (tiktoken source).

## Sources

`sources/dl4nlp/03_tokenization.tex` (38 frames; port and adapt), L21's three tokenization frames
(`git show 6013bcc:ml/ch7_rnn/L21_road_to_attention.tex`), `ml/claude_projects/armenian_glitch_token_hunt/`
(pointer only), `_learnings/2026-08-21-0140_bpe-merge-halves-are-invalid-utf8-alone.md`.
