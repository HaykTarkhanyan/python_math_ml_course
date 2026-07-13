# RNN chapter - plan (for approval)

Drafted 2026-07-13, same day the LMU `rnn/` decks were ingested. REFRAMED same day
(instructor voice direction, two rounds): this is a **bridge chapter - "the road to
attention"** with a **text spine**. RNN/LSTM are taught so students know the lineage,
but the focus is WHY attention had to be invented: variable-length inputs break MLPs,
recurrence fixes that but its gradients vanish, LSTM patches the memory but not the
bottleneck. Tokenization and variable-length handling get real estate (instructor
request); forecasting/time-series was dropped as the spine. No homework this chapter.
Follows the ch6 CNN playbook (`ml/ch6_cnn/CNN_CHAPTER_PLAN.md`); this file only spells
out what differs. Outlines: `L20_rnn_foundations_OUTLINE.md`,
`L21_road_to_attention_OUTLINE.md`.

## Source material

1. **LMU I2DL `rnn/`** - `ml/deep_learning/moodle_s26_course/slides_tex/slides/rnn/`,
   5 decks: `slides-introduction.tex`, `slides-backprop.tex`, `slides-modernrnn.tex`,
   `slides-applications.tex`, `slides-attention.tex` (the last is out of scope here -
   reserved for the GenAI chapter). Same CC BY 4.0 basis and credit rule as ch6.
2. **Our own dl4nlp decks** - `misc/dl4nlp/01_pre_transformer.tex` tells this chapter's
   story in HOUSE STYLE (same palette, TikZ): unrolled-RNN diagram, vanishing-gradient
   "The cat ... was" frame, LSTM memory-highway diagram, gate-equations boxes, seq2seq
   + bottleneck frames. `03_tokenization.tex` exists for the tokenization frames -
   consult it so L21 previews rather than contradicts the future ch8 treatment.
   Prime borrow sources, no credit needed - our material.

## Locked decisions (instructor interviews + voice directions, 2026-07-13)

| Dimension | Decision |
|---|---|
| Chapter mission | **Bridge to attention.** Teach the minimum sequence stack needed to feel the problems attention solves; depth budget goes to the PROBLEMS, not the architectures |
| Spine | **Text** (instructor direction, round 3). The sorting-line world continues via CUSTOMER REVIEWS of the exported fruit: all lengths, two languages - motivates variable length, sentiment, translation, and the bottleneck in one thread. Time series demoted to a one-frame modality mention |
| Tokenization | **In scope, with emphasis** (instructor request): how text becomes tokens (char / word / subword), why vocabularies explode, the Armenian-fragmentation local fact; full BPE mechanics stay in ch8 (dl4nlp 03 owns them) |
| Variable length | **Expanded**: a dedicated frame on the shoehorn hacks (pad+truncate, bag-of-words) and exactly what each loses - the honest "why MLPs really fail" story |
| Lecture count | 2 decks; **L21 is allowed to run long** (~23 frames, instructor OK) |
| Scope ceiling | **Stop at the bottleneck cliffhanger.** "What if the decoder could look back at all encoder states?" is asked and NOT answered; attention opens the GenAI chapter (ch8) |
| BPTT math | **Full derivation stays** - diag(sigma') V^T factors, (V^T)^(t-i), largest-eigenvalue argument. The one place the chapter goes deep |
| LSTM depth | **Intuition only**: memory-highway + gates-as-masks + additive punchline; the four equations appear ONCE on a for-the-record frame. GRU = one-sentence name-drop |
| Homework | **None this chapter.** `rnn.qmd` carries slides + video links only. The in-lecture demo is a char-level LSTM generating ARMENIAN text (locked 2026-07-13; replaces the earlier forecasting demo) |
| Visuals | **Heavy** (instructor direction): every content frame carries a diagram, figure, or ANIM; pure-bullet frames only for recaps. WEB-IMG encouraged (see Visual style) |
| Framework | PyTorch snippets; everything shown runs CPU-light locally |
| Numbering | `ml/ch7_rnn/`, decks **L20 / L21**; GenAI becomes ch8 |

## The two decks

| Deck | Working title | One-line pitch |
|---|---|---|
| L20 | RNN Foundations | reviews arrive in every length and order matters - MLPs can only pad, truncate, or bag; one new arrow (hidden-to-hidden weights) gives a network memory instead; unroll it and it is a deep net with tied weights - which is exactly why its gradients vanish (full BPTT derivation, the chapter's one deep dive) |
| L21 | The Road to Attention: LSTM, Tokens and the Bottleneck | LSTM = a memory highway built from addition (ResNet's medicine, 18 years earlier); text becomes tokens becomes vectors (the kit attention will need); a char-LSTM writes text before your eyes; then seq2seq squeezes a whole sentence into ONE vector - and even LSTM cannot fix that. Cliffhanger |

## Borrow vs add

### Borrow table

| Source | What we take | Lands in | How |
|---|---|---|---|
| LMU `rnn/slides-introduction.tex` | motivation (fixed-size inputs vs sequences), one-hot token encoding, the V-weights derivation from a dense net, step-by-step unrolled sentiment example ("This is good news." - we substitute our review sentence), parameter sharing, seq-to-one / one-to-seq / seq-to-seq taxonomy, folded-vs-unfolded graph; bidirectional = one-liner only | L20 | REDERIVE + own ANIM |
| LMU `rnn/slides-backprop.tex` | BPTT chain product, D[t] V^T factors, eigenvalue argument, gradient clipping | L20 | REDERIVE (the locked full-math frames) |
| LMU `rnn/slides-modernrnn.tex` | LSTM cell-state idea + gate intuitions (gender-pronoun example) ONLY; skip the 9-click equation buildup and the full GRU treatment | L21 | REDERIVE, compressed |
| LMU `rnn/slides-applications.tex` | LM probability factorization, the "hello" click-sequence IDEA (we rebuild it on an Armenian word), word embeddings (static-vs-context "bank"), encoder-decoder + context vector | L21 | REDERIVE + own ANIM (COPY-SLIDE option dropped in the 2026-07-13 outline revision) |
| LMU `rnn/slides-attention.tex` | **skip** - reserved for the GenAI chapter | - | - |
| Own `misc/dl4nlp/01_pre_transformer.tex` | unrolled-RNN TikZ, vanishing-gradient "The cat ... was" TikZ, LSTM memory-highway TikZ, gate-equation colored boxes, seq2seq diagram, bottleneck frame | L20, L21 | **OWN-TEX** (mechanism below) |
| Own `misc/dl4nlp/03_tokenization.tex` | consistency check only - L21's tokenization frames must preview, not contradict, the ch8 deep dive | L21 | read at build, borrow framing if useful |

### Borrowing mechanics

Same six mechanisms as ch6 (`CNN_CHAPTER_PLAN.md` "Borrowing mechanics"), with paths
swapped to `ml/ch7_rnn/{py_src,fig,fig/borrowed}`. One addition:

7. **OWN-TEX** - copy a frame's TikZ/LaTeX source from our own `misc/dl4nlp/*.tex`
   decks and adapt it. No credit line needed (our material). Caveat for the builder:
   the dl4nlp preamble is not `ml/preamble.tex` - check every macro before pasting
   (known deltas: `lightbg` color and `-Stealth` arrow tips may be missing; define
   locally or substitute). Each outline's build notes whitelist the frames taken this
   way - same no-silent-additions rule as COPY-IMG.

### Visual style (heavy-visuals rule, instructor direction)

Everything from ch6 (ANIM flip-books, diagrams over text walls, `\pause` builds,
WEB-IMG without licensing caveats) PLUS:

- **Every content frame carries a visual** - a diagram, generated figure, ANIM frame,
  or annotated example. Pure-bullet frames are allowed only for the recap and the
  outline. If a frame has no natural visual, question whether it belongs in the deck.
- **Canonical WEB-IMG sources for this chapter** (download into `fig/borrowed/`, no
  licensing caveats per standing policy): Chris Olah's "Understanding LSTM Networks"
  diagrams, Jay Alammar's seq2seq visualization stills, Karpathy's RNN-taxonomy strip
  ("The Unreasonable Effectiveness of RNNs"). Prefer these over re-deriving equivalent
  diagrams when they are better than what we would draw.
- Mandatory ANIMs: L20 = the unrolling + full-forward-pass animations (built); L21 =
  the char-LM next-char sequence (own matplotlib script on Պանիր, instructor-locked
  2026-07-13, superseding the earlier "hello"/COPY-SLIDE option) and the
  generated-text-improves-over-training reveal (one checkpoint per click).

**Credit rule:** same one-line LMU acknowledgment on each deck's title/Outline frame;
`rnn.qmd` repeats it.

### What we add ourselves (not in the LMU decks)

1. **The road-to-attention framing itself** - both decks are structured around the two
   problems (fading memory, one-vector bottleneck), with the architectures as
   supporting cast. The LMU decks teach RNNs as a destination; we teach them as the
   reason attention exists.
2. **The customer-reviews running thread** (sorting-line sequel): reviews of the
   exported fruit arrive in every length and two languages; the same review sentence
   is unrolled in L20, embedded and translated in L21, and breaks the context vector
   at the end.
3. **The variable-length pain done honestly**: a dedicated frame on the shoehorn
   hacks - pad+truncate (wastes compute / loses ends), bag-of-words (loses order -
   demonstrated by the shuffle test), per-position weights (learn every fact once per
   position). LMU asserts the problem; we demonstrate it.
4. **Tokenization as a first-class topic** (L21): char vs word vs subword trade-off,
   vocabulary explosion, OOV, and the Armenian local fact - English-centric
   tokenizers fragment Armenian text into far more tokens (measured live with a real
   tokenizer at build time, exact numbers on the frame).
5. **The callback spine** (below) - especially recurrence-vs-convolution parameter
   sharing (time vs space) and **LSTM additive cell = ResNet skip connection**
   ("same medicine: addition; LSTM 1997 beat ResNet 2015 to it by 18 years").
6. **Scalar worked numbers with an order-sensitivity punchline**: a 1-d RNN (W=0.5,
   V=0.8) reads 3 tokens embedded as [2,1,3] -> h=[1.0, 1.3, 2.54]; the SAME tokens
   reversed [3,1,2] -> 2.36. Same bag of words, different memory - the state encodes
   order. The same V then gives 0.8^29 ~ 0.0015 (vanishing) and 1.25^29 ~ 646
   (exploding) as the on-ramp to the full derivation.
7. **The char-LSTM generation demo** (L21, replaces forecasting): train a character
   LSTM on a small ARMENIAN corpus (locked 2026-07-13), show samples at increasing
   training steps - gibberish -> letter statistics -> Armenian words ->
   almost-language. The most visual proof that next-token prediction extracts
   structure, and the chapter's strongest local touch.
8. **GPT foreshadow, one sentence**: the LM factorization P(y1..yT) = prod P(yi|y<i)
   "is the exact objective GPT trains on, at a different scale."
9. **The 2026 epilogue** (brief) - LSTM 1997 -> seq2seq/GRU 2014 -> transformers 2017
   retire RNNs from NLP -> recurrence returns (Mamba, xLSTM by Hochreiter himself);
   where RNNs still earn their keep (streaming/edge, small-data settings). People:
   Hochreiter, Schmidhuber (one respectful line), Sutskever callback to the L17
   epilogue. All post-2023 facts web-verified at build.

## Examples ledger

| Example | Role | Deck | Mechanism |
|---|---|---|---|
| Customer review sentence (e.g. "The pomegranates arrived fresh and sweet.") | chapter running example: unrolled in L20, tokenized/embedded in L21, translated in seq2seq, breaks the bottleneck as a LONG review | both | REGEN-FIG + in-frame |
| Word-shuffle test (review sentence shuffled; identical bag-of-words histogram) | "order is information" + why bag-of-words fails | L20 | REGEN-FIG |
| "I went to Munich in 2009" / "In 2009, I went to Munich" | why per-position weights fail | L20 | REDERIVE (from LMU commented-out frame) |
| Scalar RNN W=0.5, V=0.8 on [2,1,3] vs [3,1,2] | worked numbers: forward pass + order sensitivity + gradient decay | L20 | in-frame math |
| "The cat that I saw ... was" long-dependency sentence | vanishing gradients made concrete | L20 | OWN-TEX |
| Real tokenizer on the review sentence + the chapter's Armenian line | char/word/subword trade-off + Armenian fragmentation fact | L21 | REGEN-FIG (measured at build) |
| word2vec king-queen + static "bank" limitation | embeddings in two frames | L21 | REDERIVE + REGEN-FIG |
| char-level "hello" | language modeling mechanics | L21 | COPY-SLIDE or own ANIM |
| char-LSTM generated samples over training | the generation demo (Armenian) | L21 | REGEN-FIG (Armenian corpus, locked; source settled at the L21 interview) |
| The chapter's Armenian line (below) + its English gloss | tokenizer tax panel, seq2seq target, bottleneck frames, char-LSTM seed | L20 + L21 | canonical copy in `py_src/data/armenian_line.txt`; rendered via figure if the preamble lacks the script |
| Gender-pronoun / "new season" intuition | forget-gate intuition (one line each) | L21 | REDERIVE (LMU) + running example |

**The chapter's Armenian line (instructor-chosen, 2026-07-13):**

> Ես այս ամենինչ նայում եմ լուռ, և գիտակներ են խոսում իմ մեջ

Canonical copy: `ml/ch7_rnn/py_src/data/armenian_line.txt` - scripts read it from
there so slides, figures, and the demo stay in sync. English gloss to be confirmed
with the instructor at the L21 interview.

## The course-callback spine

| New concept | Callback to | Framing |
|---|---|---|
| z[t] = sigma(V'z[t-1] + W'x[t] + b) | L14 hidden layer | "one new term in a formula you know" |
| order destroyed by shuffling | L16 pixel-shuffle frame | same demo, new axis |
| parameter sharing across time | L16 sharing across space | "convolution shares across space; recurrence shares across time" |
| BPTT = backprop on the unrolled graph | L15 "one sweep, any architecture" | promise kept again, now with tied weights |
| vanishing gradients, RNN edition | L15 vanishing gradients | "same disease, worse: the SAME matrix every step" |
| gradient clipping | L15 (verify at build; teach in place if absent) | one formula |
| LSTM additive cell state | L17 ResNet skip connection | "solve multiplication with addition" |
| gates = sigmoid masks | L14 sigmoid | rebranded as a soft switch |
| one-hot token vectors | [03] data preprocessing one-hot | same encoding, new victim |
| embeddings as pretrained features | L18 transfer learning | "warm start for text" |
| 1D conv as the rival sequence tool | L19's 1D-conv tease | honest alternatives, one line |

## Homework

**None** (locked, 2026-07-13). `rnn.qmd` gets slides + video links and the chapter
credit line only - no HW section, no solution notebook. The transformer/GenAI chapter
picks the hands-on work back up. The only demo element is the in-lecture char-LSTM
generation reveal (L21), whose figures are generated offline by `charlm_generate.py`.

## Figure pipeline (scripts in `py_src/`, PDFs in `fig/`, `ma` venv, seed 509)

| Script | Figure | Deck |
|---|---|---|
| `word_shuffle.py` | review sentence vs shuffled version + identical bag-of-words histograms side by side | L20 |
| `pad_truncate.py` | the shoehorn hacks panel: reviews of lengths 4/9/23 padded+truncated to 8, vs bagged (or TikZ if cleaner) | L20 |
| `unroll_anim.py` | ANIM flip-book: the review sentence fed token by token into the unrolled net, state annotated (uses the worked-numbers values) | L20 |
| `vanish_curve.py` | 0.8^k and 1.25^k vs k, log scale, labeled | L20 |
| `gradient_flow.py` | empirical gradient norm per time-step, tiny vanilla RNN vs LSTM on a toy recall task (CPU-light); vanilla-only variant for L20, both-curves variant for L21 | L20 + L21 |
| `tokenizer_demo.py` | the review sentence split three ways (char/word/subword) + the Armenian-line-vs-English-gloss token-count comparison (real tokenizer, measured) | L21 |
| `embedding_2d.py` | 2-D projection of ~30 word vectors from a small shipped data file (no downloads at build) | L21 |
| `charlm_anim.py` | ANIM: "hello" next-char distributions click by click (skip if COPY-SLIDE chosen) | L21 |
| `charlm_generate.py` | char-LSTM training on a small Armenian corpus; panel of generated samples at increasing training steps + loss curve; final sample seeded from the Armenian line | L21 |
| `rnn_timeline.py` | 1997-2026 timeline ribbon (continues the L17 ribbon style) | L21 epilogue |

Local scripts stay CPU-light (tiny nets, small corpora, minutes not hours).

## Implementation notes for the authoring model

Items 1-7 of the ch6 plan's "Implementation notes" apply verbatim with paths swapped
(`ml/ch7_rnn/`, `L20_...tex` / `L21_...tex`, chapter page `rnn.qmd`). Deltas:

1. Build order: **L20 -> L21**, each compiled + overflow-checked + approved before the
   next. Per-deck instructor interviews happen BEFORE each build (the outlines will be
   updated then; do not start from this plan alone).
2. **Respect the reframe:** if a frame teaches LSTM/GRU machinery beyond the outline's
   intuition frames, cut it. Depth belongs to the vanishing-gradient derivation, the
   variable-length/tokenization story, and the bottleneck only.
3. **Heavy-visuals rule:** every content frame carries a visual; see Visual style.
4. torch is required for `gradient_flow.py` / `charlm_generate.py`; torchvision is NOT
   needed. A tokenizer library is needed for `tokenizer_demo.py` (tiktoken or
   HuggingFace tokenizers) - install into `ma` with uv, note the one-time encoding
   download, and cache it; the script itself must fail loudly if the cache is missing.
5. OWN-TEX pastes: check macros against `ml/preamble.tex` first (see mechanism 7).
6. Armenian text on slides (tokenizer comparison, translation example): test whether
   `ml/preamble.tex` renders Armenian script; if not, ship those examples as generated
   figures (matplotlib renders Armenian fine with a proper font). The canonical
   Armenian line lives in `py_src/data/armenian_line.txt` - read it from there.

## Definition of done

Same as ch6 minus homework: 2x pdflatex, zero `!` log lines, overflow check, aux
cleaned, `% Provenance:` block per deck, figures via `ma` venv, `rnn.qmd` (links +
credit only) registered in `_quarto.yml` (exact case), commit without aux files.

## Open questions for the instructor

1. **Folder + numbering:** `ch7_rnn`, L20/L21, GenAI becomes ch8 - confirm.
2. **Deck titles:** "L20 - RNN Foundations", "L21 - The Road to Attention: LSTM,
   Tokens and the Bottleneck" OK?
3. ~~char-LSTM demo corpus~~ Fully resolved 2026-07-13 (pre-build interview):
   **Armenian classic literature** - public-domain Tumanyan/Charents from Wikisource,
   saved as `py_src/data/corpus_hy.txt`; the chapter's Armenian line stays the
   canonical showcase text and seeds the final sample.
4. **Epilogue name-drops:** Mamba/state-space models + xLSTM in the 2026 close - keep
   or trim?
5. **`gradient_flow.py`** (empirical vanishing evidence, vanilla vs LSTM): keep as the
   L20 cliffhanger + L21 payoff pair, or cut and let `vanish_curve.py` carry it alone?
6. **Tokenizer library:** tiktoken (GPT tokenizers, one-time download) vs HuggingFace
   tokenizers - either works for the demo; builder picks unless you have a preference.
