# LLM-7 - Pretraining: from the internet to a base model - outline (2026-10-06)

Drafted 2026-10-06 from the critique in `LLM_CHAPTER_PLAN.md` (LLM-7) and the recommended answers
in `INTERVIEW_QUEUE.md` (instructor: "do all queue items"). Session 7 of the LLM chapter
(`llms`). Its one question: **LLM-6 used the probabilities; where do the 124M numbers that
produce them come from?** It must not re-teach the objective's mechanics (LLM-4 owns "one pass, n
training signals"), perplexity's definition (the name-inventor practical, `47`, section 6) or
"autocomplete is not a chatbot" (LLM-9's opening): callbacks only.

Deck file: `LLM7_pretraining.tex`. Figures: `py_src/pretraining_figs.py` (one `--gpt2` run of the
cached GPT-2 -> `results/pretraining_gpt2.json`; every quoted number asserted).

**Decisions (queue answers, 2026-10-06):**

- Spine: **data-led** - objective family (one mask figure, word2vec's two frames) -> one training
  step by hand -> the data funnel -> what comes out.
- Cold open: **GPT-2 recites a much-copied text verbatim.** The one run (2026-10-06, greedy, 48
  new tokens) decided it: MIT licence 48 of 48 tokens verbatim, Apache licence 22 of 22, the
  Gettysburg Address 0, the Declaration of Independence 1, lorem ipsum 0.
- Real pieces (multi-select, no recommendation; all four taken, each has a source): the
  memorisation run; the Armenian share (Common Crawl CC-MAIN-2026-39, FineWeb-2); FineWeb's
  funnel; samples during training (Karpathy's run, from the local transcript).
- GPT-1 -> GPT-2 -> GPT-3: **one frame**; few-shot scaling is LLM-8's.
- Hands-on: **not in the slides.**

**Facts checked 2026-10-06 (sources in the deck's provenance):**

| Fact | Source |
|---|---|
| FineWeb: 96 Common Crawl snapshots; base filtering ~36T tokens; per-snapshot MinHash dedup 20T; C4 + custom filters -> 15T; fastText English >= 0.65; MinHash on 5-grams, 112 hashes = 14 buckets x 8, >= 75% similar | FineWeb paper, arXiv 2406.17557 |
| C4 rules: lines end in terminal punctuation, lines >= 5 words, pages >= 3 sentences, no "lorem ipsum", no "{", no bad-words list, no lines with "Javascript", 3-sentence-span dedup, langdetect English >= 0.99 | T5 paper, arXiv 1910.10683, section 2.2 |
| T5 span corruption: "Thank you <X> me to your party <Y> week." -> "<X> for inviting <Y> last <Z>"; 15% corrupted, mean span 3 | T5 paper, Figure 2, section 3.3.4 |
| BERT: 15% of tokens chosen; 80% [MASK], 10% random, 10% unchanged | BERT paper (and `sources/dl4nlp/07`) |
| Common Crawl CC-MAIN-2026-39 page shares (CLD2 primary language): English 41.8620%, Russian 6.6954%, Armenian 0.0315% | commoncrawl.github.io/cc-crawl-statistics/plots/languages |
| FineWeb-2: Armenian 634,273,060 words, 1,757,415 documents; Russian 588,579,493,780 words (word counts, not GPT-2 tokens) | FineWeb-2 dataset card |
| Carlini et al. 2022: memorisation grows log-linearly with model capacity, number of duplicates, prompt length | arXiv 2202.07646 abstract |
| Carlini et al. 2021: hundreds of verbatim sequences extracted from GPT-2 (names, phone numbers, emails, code, UUIDs), each in just one training document | arXiv 2012.07805 abstract |
| Lee et al. 2022: a 61-word sentence repeated over 60,000 times in C4; dedup -> memorised text emitted 10x less often | arXiv 2107.06499 abstract |
| GPT-2: WebText, ~8M documents, 40 GB, Wikipedia removed | GPT-2 paper section 2.1 |
| Llama 3: ~15.6T tokens (1.8T for Llama 2, as the Llama 3 paper puts it); mix 50% general, 25% math and reasoning, 17% code, 8% multilingual | Llama 3 paper via `sources/llm_training/07_llama3` |
| Karpathy's GPT-2 reproduction: 1M tokens per step, 32,000 steps (~33B tokens); step 20 random text, ~1% "local coherence" | `sources/dl4nlp/_yt_videos/karpathy_deep_dive_llms/transcript.txt`, 35:00-38:00 |

---

## Frames

### Opening

| # | Frame | Content | Figure |
|---|---|---|---|
| 0 | Title | "Pretraining" / "From the internet to a base model" | - |
| 1 | Where we are: one forward pass | "The pass is built. Today: where its 124M numbers come from." | `roadmap_pretraining_pass.pdf` |
| 2 | Where we are: the life of a model | Pretrain lit. | `roadmap_pretraining_life.pdf` |
| 3 | Cold open: finish these | **Predict first.** Two openings: the MIT licence and the Gettysburg Address. Which one does GPT-2 small continue word for word? | - (two quote boxes) |
| 4 | The licence, not the speech | Real run: per-text bars of tokens recited verbatim (5 texts) + the licence continuation. Famous is not the same as frequent: a licence is pasted into countless source files. Question for the lecture: where do the numbers come from, and why do they hold *this*? | `pre_recite.pdf` |
| 5 | Outline | | - |

### 1. The objective

| # | Frame | Content | Figure |
|---|---|---|---|
| 6 | [plain] transition | "The objective" / "Hide part of the text; make the model guess it." | - |
| 7 | You have seen one already | Callback: LLM-4's causal mask, one pass, every position a guess; loss = average -log p of the right token; gradient descent (ch11). "Is next-token the only way to hide text?" | - |
| 8 | Five ways to hide text | One mask figure, one row per objective on one sentence: bigram (47: one token back), word2vec (a word from its window), GPT (everything to the left), BERT (15% hidden, both sides), T5 (spans hidden, written back). | `pre_objectives.pdf` |
| 9 | word2vec: a word and its neighbours | Skip-gram by hand on "the cat sat on the mat", window 2: the pairs for "sat". No order, no language model around it. The table it learns is LLM-2's table. | `pre_w2v_pairs.pdf` |
| 10 | Why word2vec learns meaning | Real pairs pushed together (dot product up), random "negative" words pushed apart; words with the same neighbours end with similar rows. One line of the loss. | - |
| 11 | BERT: hide 15%, see both sides | Recipe 15% / 80-10-10, and why the 10% + 10% (no [MASK] at use time). Loss only on the hidden 15%. Cannot write (LLM-5's mask A); gives embeddings (the RAG chapter). | - |
| 12 | T5: hide whole spans | The paper's own example with sentinels. | - (a two-row box) |
| 13 | Why next-token won | Every position is a target (100% vs 15%), it writes, and any text is training data. One line back to LLM-5. | - |

### 2. One training step, by hand

| # | Frame | Content | Figure |
|---|---|---|---|
| 14 | [plain] transition | "One training step" / "From three documents to one number." | - |
| 15 | Pack the documents | Three toy documents -> one stream with `<|endoftext|>` (LLM-1) -> windows of 8. A window can cross a document boundary; the separator tells the model what follows is unrelated. | `pre_packing.pdf` |
| 16 | One window, eight guesses | A window of 9 tokens = 8 (context, target) pairs; toy p(right token) for each; -log p; the mean = the loss. | `pre_window_loss.pdf` |
| 17 | Loss to perplexity | Perplexity = e^loss (47's "torn between k letters"); this window's value. GPT-2's real loss on the map's sentence as a second number (from the run). | - |
| 18 | Perplexity across languages lies | Same Armenian sentence, two tokenizers (toy): per-token perplexity says the byte-level one is ~10x better; bits per character says it is worse. Compare bits per character (or byte) of the same text. | `pre_bpc.pdf` |

### 3. The data

| # | Frame | Content | Figure |
|---|---|---|---|
| 19 | [plain] transition | "The data" / "Most of the work is throwing text away." | - |
| 20 | How much text | GPT-2: WebText, 8M documents, 40 GB, Wikipedia removed. Llama 3: ~15.6T tokens. FineWeb: 15T tokens of English from 96 crawls. | - |
| 21 | The funnel | FineWeb's stages with the three published sizes: 36T -> 20T -> 15T. | `pre_funnel.pdf` |
| 22 | Clean this page by hand | A toy web page and four C4 rules: which lines survive, and then the page dies for having fewer than 3 sentences. | `pre_c4_page.pdf` |
| 23 | Duplicates | MinHash in one sentence; Lee et al. (61-word sentence x 60,000; 10x less memorised after dedup); Carlini 2022 (duplicates, size, prompt length). Back to the cold open. Caveat: Carlini 2021 extracted texts that were in one document only. | - |
| 24 | How much of the web is Armenian? | Common Crawl page shares (log bars): English 41.86%, Russian 6.70%, Armenian 0.0315% (~1,330 English pages per Armenian one). FineWeb-2: 634M Armenian words vs 589B Russian. LLM-1's token tax and the weak Armenian answers start here; language ID decides what counts. | `pre_languages.pdf` |
| 25 | The mixture | Llama 3's mix (50/25/17/8), with 8% for every language but English together. | `pre_mixture.pdf` |

### 4. What comes out

| # | Frame | Content | Figure |
|---|---|---|---|
| 26 | [plain] transition | "What comes out" / "A base model: it continues documents." | - |
| 27 | Watching it learn | Karpathy's GPT-2 run: 1M tokens per step, 32,000 steps; step 20 random; ~1% "local coherence" (his sample); the end: fluent English. | - |
| 28 | It recites, and it leaks | Memorisation grows with duplicates, size, prompt length; the name inventor (47, section 12) did it too when overtrained; GPT-2 leaked phone numbers and emails seen once. | - |
| 29 | It continues documents | One real GPT-2 run: a bare question vs the same question as a Q/A document. A base model is a document simulator; LLM-9 makes answering the default. | `pre_base_model.pdf` |
| 30 | GPT-1 -> GPT-2 -> GPT-3 | Fine-tune per task (2018, 117M) -> zero-shot, "TL;DR:" (2019, 1.5B) -> few-shot in the prompt (2020, 175B). Scaling is LLM-8's. | - |

### Close

| # | Frame | Content | Figure |
|---|---|---|---|
| 31 | Recap | | - |
| 32 | Next on the map | LLM-8: how big a model, how much data, for a given budget. | `roadmap_scaling_life.pdf` |
| 33 | Sources & further reading | | - |

**Self-review (2026-10-06):** (1) the cold open must not overclaim why GPT-2 recites the licence -
"pasted into countless files" is a statement about the web, and Carlini 2022 measured the
duplication effect; WebText's own duplicate counts are not known to us, so the slide says
"likely". (2) The bits-per-character frame uses one text and two tokenizers, so the comparison is
fair by construction; across languages the slide only says per-token numbers are not comparable.
(3) Frame 13 must not re-explain the causal mask (LLM-4 owns it). (4) The Q/A run is not the
password example (LLM-9's cold open). (5) Synthetic data, ELMo, NSP, contrastive learning and the
compute tables are cut (plan's "Cut" list).
