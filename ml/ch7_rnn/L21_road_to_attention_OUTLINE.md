# L21 - The Road to Attention: LSTM, Tokens and the Bottleneck - outline (for approval)

Drafted 2026-07-13 (road to attention, TEXT spine, heavy visuals; instructor allowed
this deck to run long). REVISED same day after the L20 build + instructor review of it -
see "Lessons from the L20 build" below. L21's job: show that the 1997 fix (LSTM) works -
concretely but briefly - then build the text kit attention will need (tokenization with
emphasis, embeddings, language modeling, a generation demo), and walk into the wall
nothing so far can fix: the seq2seq information bottleneck. The deck ENDS on the
unanswered question that attention answers; the GenAI chapter (ch8) opens by answering it.
Subtitle idea: "The fix that worked, and the wall it hit."

**Interview decisions (2026-07-13, structural rounds + voice directions):**
- LSTM: INTUITION ONLY - memory highway + gates-as-masks + additive punchline; the
  four equations appear once on a for-the-record frame; NO gate-by-gate buildup.
- GRU: one-sentence name-drop.
- TEXT focus (forecasting demo dropped): the in-lecture demo is a char-LSTM
  generating text; tokenization gets a real section (instructor request).
- char-LSTM corpus: ARMENIAN (locked 2026-07-13); the chapter's Armenian line (see
  build notes) is the showcase text across the tokenizer, seq2seq, and demo frames.
- NO homework.
- Ending: bottleneck cliffhanger - the attention question is asked, NOT answered.
- Heavy visuals: every content frame carries a diagram/figure/ANIM.

**Pre-build interview, rounds 1-2 (2026-07-13) - this outline is now BUILD-READY:**
- Demo corpus: CLASSIC LITERATURE - public-domain Tumanyan/Charents poems + fairy
  tales from Wikisource (~0.5-1 MB), saved to `py_src/data/corpus_hy.txt`.
- English gloss LOCKED: "I watch all of this in silence, and the connoisseurs speak
  inside me." (chapter constant; used on the tokenizer tax panel + seq2seq frame).
- char-LM mechanics ANIM word: **Պանիր** (instructor's pick - the course's cheese
  running joke). Note: no repeated character, so the "hello double-l" context lesson
  is NOT the ANIM's punchline; it becomes a one-line remark instead (see Section 3).
- Epilogue: FULL comeback story (Mamba/state-space + Hochreiter's xLSTM).
- Diagrams: WEB CANONICAL preferred - Chris Olah's LSTM chain diagram + a Jay
  Alammar seq2seq still (WEB-IMG, no credit lines); own dl4nlp TikZ is the fallback.
- Generation-demo reveal: ANIM PER CLICK (~5 checkpoints, final seeded with the line).
- Armenian tax framing: BOTH consequences (fading callback to L20 + money per API call).
- Links: ALL THREE sprinkled (tiktokenizer app, TensorFlow Embedding Projector,
  Karpathy's char-RNN post), one per section - the L17 precedent.

**Lessons from the L20 build folded into this revision (2026-07-13):**
1. **Concreteness beats intuition sketches** (the instructor's L20 revision verdict:
   "the diagrams and 1-d calculation didn't really help"). Applied here: the LSTM gets
   an L20-style ANATOMY frame - every arrow labeled with what flows and its shape -
   plus one small NUMERIC gate-mask moment. Scope stays intuition-level (no gate-by-gate
   equation buildup); the depth is in the labeling, not in more math.
2. **Chapter assets L20 created, which this deck must reuse:**
   - The toy net: vocab 3 (the Armenian line's first three words), one-hot in R^3,
     W (3x2, rows = word vectors), V (2x2), tanh, U (2x1) -> sigmoid. Forward score
     0.38, reversed 0.82 (locked, asserted in forward_pass_anim.py).
   - The review sentence constant: "The pomegranates arrived fresh and sweet."
   - The 42-word complaint review (ends "...disappointing"; text lives in
     pad_truncate.py's REVIEWS dict) - now the bottleneck frame's breaking input.
   - The "one-hot x W selects a row - that row IS the word's vector" beat, planted in
     L20's anatomy frame explicitly as an embedding on-ramp - the embeddings frame
     here must cash it in by name.
3. **Armenian on slides is DEFINITIVELY figure-only**: ml/preamble.tex (pdflatex)
   cannot render Armenian glyphs. All Armenian comes from matplotlib figures that read
   py_src/data/armenian_line.txt. Technical gotchas in the build notes.
4. L20 landed at 26 frames (outline said ~21); expect the same drift here - fine.

Target: ~24 frames, one full ~90-min session (runs long by design).

### Cold open (before Outline)

- **Frame - the disease, recalled.** One line from L20: influence fades as V^k;
  0.8^29 ~ 0.0015. Predict-first: "Multiplying by 0.8 twenty-nine times killed the
  gradient. What arithmetic operation would NOT shrink it?" `\pause` Addition.
  Hold that thought - it is the first half of this lecture, and you have seen it
  before (ResNet, L17). `[predict-first]` `[callback: L20 fading number, L17 ResNet]`
  `[visual: the vanish_curve figure with a big "+" teaser]`

### Outline frame

### Section 1: LSTM - the additive fix (intuition only, concretely drawn)

- `[plain]` transition: "A memory highway."
- **The cell state.** A second track running alongside the hidden state, modified
  only lightly at each step - the memory highway. Sorting-line metaphor: the belt
  itself carries context past every station; each worker only adds or removes a
  little. `[WEB-IMG (interview-locked): Chris Olah's LSTM chain diagram, no credit
  line]` `[fallback: OWN-TEX dl4nlp memory-highway TikZ]` `[running example]`
- **Anatomy of one LSTM step.** (NEW - mirrors L20's anatomy frame, which landed
  well.) One complete labeled diagram: inputs x[t] and z[t-1] feed ALL THREE gates
  and the candidate; each gate = sigmoid -> vector in (0,1)^d_h used as an
  element-wise mask (⊙ marked on the diagram); the c-highway runs across the top;
  z[t] exits at the bottom. Use d_h = 2 so the shapes match L20's toy net. Gate
  intuitions, one line each: FORGET (a review switches products mid-sentence - drop
  the old subject; LMU's gender-pronoun example as the classic), INPUT (a strong
  opinion word appears - write it down), OUTPUT (what the sentiment readout needs
  right now). Predict-first numeric micro-moment: "forget gate [0.9, 0.1] meets cell
  state [2.0, -1.0] - what survives?" `\pause` [1.8, -0.1]: memory 1 survives,
  memory 2 is wiped. A gate is L14's sigmoid, rebranded as a soft switch.
  `[predict-first]` `[worked-numbers]` `[callback: L14 sigmoid, L20 anatomy frame]`
  `[diagram with \only overlays acceptable - still NO gate-by-gate equation buildup]`
- **For the record: the four equations.** The dl4nlp colored-box frame, shown once,
  explicitly flagged "reference, not exam material". The punchline lives here:
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c~_t is ADDITIVE - gradients flow through the cell
  state with minimal decay. ResNet solved depth with addition in 2015; LSTM solved
  time with addition in 1997 - same medicine, 18 years earlier.
  `[OWN-TEX: dl4nlp gate-equations frame]` `[armblue key box]`
  `[callback: L17 ResNet skip]`
- **Does it work? Proof + price.** The L20 gradient_flow figure completed: the LSTM
  curve stays alive where the vanilla curve died. Price: 4 weight sets instead of 1 -
  our 288-param vanilla becomes 1,152 (d_x=1, d_h=16). GRU name-drop, one sentence:
  a streamlined 3-gate cousin, similar results, common default. `[real fig:
  gradient_flow.pdf, both curves - MUST be regenerated first, see build notes: the
  L20 quick run does not yet show a convincing LSTM advantage]` `[worked-numbers]`

### Section 2: Tokens - how text actually enters the network

- `[plain]` transition: "Before any of this works, text must become numbers."
- **The tokenization trilemma.** Split the review sentence three ways, side by side:
  characters (tiny vocabulary ~100, but sequences get LONG and letters mean nothing),
  words (meaningful units, but the vocabulary explodes - and "pomegranatey" was never
  in it: OOV), subwords (the compromise: frequent chunks become tokens, rare words
  split into pieces - what every modern model uses). `[real fig: tokenizer_demo.pdf,
  panel 1 - the same sentence split three ways]`
- **Subwords on a real tokenizer.** The review sentence through an actual production
  tokenizer, token boundaries visible. One sentence on how the vocabulary is chosen
  (frequent pairs get merged - BPE in one breath; the full mechanics belong to the
  GenAI chapter). Link on the frame: the tiktokenizer web app - students type
  Armenian live and watch it fragment. `[real fig: tokenizer_demo.pdf, panel 2]`
  `[link: tiktokenizer]`
- **The Armenian tax.** The chapter's Armenian line - "Ես այս ամենինչ նայում եմ
  լուռ, և գիտակներ են խոսում իմ մեջ" - vs its English gloss, through the same
  tokenizer: the Armenian version costs several times more tokens -
  English-centric vocabularies fragment other scripts. Why it matters: longer
  sequences = more steps = more fading (L20!) and, later, more money per ChatGPT
  call. Exact counts measured at build, on the frame. `[real fig: tokenizer_demo.pdf,
  panel 3]` `[armorange watch-out box]` `[Armenian touch]`
- **One-hot and its poverty.** Each token an ID, each ID a one-hot vector: 10,000-dim,
  every pair of words equally far apart - "good" is as far from "great" as from
  "plastic". Callback: [03] one-hot; L20's toy net used a 3-word version of exactly
  this. `[diagram: one-hot columns with equal pairwise distances annotated]`
- **Embeddings.** Cash in L20's planted beat BY NAME: "L20's anatomy frame showed
  one-hot x W selects a row of W, and that row IS the word's vector. An embedding
  table is just that W, grown to vocabulary size and learned." Dense learned vectors;
  similar words end up close; the famous king - man + woman ~ queen arithmetic.
  Trained once on huge corpora (word2vec, GloVe) and reused - a warm start, i.e.
  transfer learning for text. Limitation in one line: "bank" gets ONE vector
  regardless of context - remember that itch, the GenAI chapter scratches it.
  `[real fig: embedding_2d.pdf]` `[link: TensorFlow Embedding Projector]`
  `[callback: L20 anatomy frame, L18 transfer learning]`

### Section 3: Language modeling - the task that powers everything

- `[plain]` transition: "One task to rule them all: predict the next token."
- **The objective.** P(y1..yT) = prod P(yi | y1..y(i-1)) - turn raw text into
  supervised data for free: every prefix is an input, every next token a label. Said
  slowly, once: this exact objective, scaled up, is what GPT trains on.
  `[popblue theory box]` `[GPT foreshadow]` `[diagram: sentence sliced into
  prefix->next pairs]`
- **The mechanics, click by click.** Feed a word's characters one at a time; the
  output is a probability distribution over the vocabulary at every step. Word
  (interview-locked): **Պանիր** (Պ-ա-ն-ի-ր) - the course's cheese joke, students
  will get it. Feed Պ-ա-ն-ի, want ա-ն-ի-ր, ~6 clicks. Since Պանիր has no repeated
  character, the "same input char, different prediction" lesson (hello's double l)
  becomes a ONE-LINE remark instead: ն also appears in ամենինչ, նայում, գիտակներ -
  the model's bet after ն depends entirely on what came before it. `[ANIM
  (mandatory): charlm_anim.py, own matplotlib - COPY-SLIDE option dropped; LMU's 4:3
  "hello" pages would break the deck's visual continuity]`
- **Demo: a char-LSTM learns to write Armenian.** Train a character-level LSTM on
  the Tumanyan/Charents corpus (interview-locked); reveal samples ONE CHECKPOINT PER
  CLICK (interview-locked ANIM, ~5 clicks: step 0 / 100 / 1k / 5k / final): random
  gibberish -> letter statistics -> Armenian words -> almost-literary lines. The
  final click's sample is seeded with the opening of the chapter's Armenian line
  ("Ես այս ամենինչ..."), so the model visibly continues it. The most visual proof
  that next-token prediction extracts structure, and the deck's local-touch peak.
  Follow-up frame: the loss curve beside the final sample. Link on the frame:
  Karpathy's "The Unreasonable Effectiveness of RNNs" (this demo is its Armenian
  edition). `[ANIM (mandatory): charlm_generate.py emits charlm_gen_0..4.pdf]`
  `[real fig: charlm_loss.pdf]` `[link: Karpathy char-RNN post]`

### Section 4: Seq2Seq and the wall

- `[plain]` transition: "Two networks, one vector between them."
- **Encoder-decoder.** Translation needs different input/output lengths: an encoder
  LSTM reads the source and hands ONE context vector C to a decoder LSTM that writes
  the target. Example: "I watch all of this in silence, and the connoisseurs speak
  inside me." in, the chapter's Armenian line out (Armenian rendered via figure -
  definitive, see build notes). `[WEB-IMG (interview-locked): Jay Alammar seq2seq
  still, no credit line]` `[fallback: OWN-TEX dl4nlp seq2seq diagram]`
  `[running example]`
- **The bottleneck.** Use L20's 42-word complaint review as the concrete breaking
  input (continuity: students already watched it lose its verdict to truncation).
  Predict-first: "42 words, a 300-dim context vector. What goes wrong?" `\pause`
  EVERYTHING the decoder will ever know about the source must fit through that one
  vector - early words fade, detail is lost, long inputs degrade. And the LSTM does
  NOT save us: the cell state is still one fixed-size vector. The fix for fading
  memory did not fix compressed memory. `[OWN-TEX: dl4nlp bottleneck frame]`
  `[predict-first]` `[armred trap box]` `[callback: L20 shoehorn-hacks review]`
- **The cliffhanger.** "What if the decoder could look BACK at all the encoder
  states - and learn WHERE to look?" That question has a name, it was answered in
  2014-2017, and it ends this course's pre-history. Next chapter: attention and
  transformers. Do NOT answer it here (locked). `[story frame]` `[visual: the
  seq2seq diagram with dashed arrows from every encoder state to the decoder,
  drawn but unlabeled - the answer hiding in plain sight]`

### Section 5: Epilogue - recurrence in 2026 (brief)

- **One timeline frame.** 1997 LSTM (Hochreiter & Schmidhuber) -> 2014 seq2seq
  (Sutskever - L17's epilogue alum) and GRU (Cho) -> 2017 "Attention Is All You
  Need" retires RNNs from NLP -> 2023+ recurrence returns as state-space models
  (Mamba) and xLSTM (Hochreiter again). Where RNNs still earn their keep:
  streaming/edge, small-data settings. `[real fig: rnn_timeline.pdf]`
  `[VERIFY all post-2023 claims at build]`

### Recap + Next

- Recap the chapter arc in five beats: length varies (hacks fail) -> memory (state)
  -> fading (same-matrix powers, derived) -> highway (additive cell, works but 4x
  cost) -> wall (one vector cannot hold a review).
- `[paramgreen Next box]`: "The decoder wants to look back at everything. Letting it
  do that - attention - is where the GenAI chapter begins."

---

## Figures (py_src -> fig, `ma` venv, seed 509)

1. `gradient_flow.py` - REGENERATE before use (L20 flagged it): train longer / use a
   longer recall distance until the LSTM-vs-vanilla gap is real and visible; if an
   honest run still shows no dramatic gap, SAY THAT on the frame (ch6 lesson: report
   what the data shows) rather than tuning until it flatters. Mind the fused-LSTM
   gradient gotcha (build notes).
2. `tokenizer_demo.py` - three panels: (1) the review sentence split char/word/
   subword; (2) real subword boundaries on the sentence; (3) the Armenian line vs
   its English gloss, token-count bars (labeled via ax.bar_label); the line is read
   from `py_src/data/armenian_line.txt`. Uses a real tokenizer (tiktoken or
   HF tokenizers - plan open question 6); measured counts, not invented.
3. `embedding_2d.py` - 2-D projection of ~30 hand-picked word vectors from a small
   shipped data file in `py_src/data/` (no downloads at build); include the
   king/queen/man/woman quartet.
4. `charlm_anim.py` - next-char distributions on Պանիր (Պ-ա-ն-ի-ր,
   interview-locked), ~6 clicks, own matplotlib.
5. `charlm_generate.py` - char-LSTM on the Tumanyan/Charents corpus
   (`py_src/data/corpus_hy.txt`, hidden <= 128, CPU minutes); emits the
   checkpoint-per-click ANIM `charlm_gen_0..4.pdf` (interview-locked format) +
   `charlm_loss.pdf`; the final checkpoint's sample is seeded from the opening of
   `py_src/data/armenian_line.txt`; saves all samples verbatim to a text log so the
   frames quote real output.
6. `rnn_timeline.py` - 1997-2026 ribbon, same visual language as L17's
   timeline_ribbon.

## TikZ (small)

The LSTM anatomy frame (shapes + gates + ⊙ marks); the prefix->next-token slicing
diagram; the cliffhanger's dashed look-back arrows.

## Build notes (for the implementing model)

- **Respect the reframe (hard rule):** no GRU equations, no LSTM gate-by-gate
  equation buildup, no BPE merge-table mechanics, no attention math. Concreteness
  means LABELED diagrams and one numeric gate-mask moment, not more derivations.
  Depth budget: L20 spent it on the vanishing derivation; here it goes to
  tokenization + the bottleneck story.
- **Heavy-visuals rule:** every content frame carries a visual; the bullets above
  name one per frame - do not drop them.
- **Match the built L20** (`L20_rnn_foundations.tex`): its anatomy-frame idiom,
  ANIM \only<n> pattern, callout boxes, [plain] transitions, footer. It is the style
  reference; do not re-read the style guide for what L20 already demonstrates.
- **Chapter constants (from the L20 build - reuse verbatim, never retype):**
  - Review sentence: "The pomegranates arrived fresh and sweet."
  - The Armenian line: read from `py_src/data/armenian_line.txt`; scripts must read
    the file so every appearance matches character for character.
  - The 42-word complaint review: import/copy from `py_src/pad_truncate.py`'s
    REVIEWS dict for the bottleneck frame's figure or text.
  - Toy-net numbers citable in callbacks: forward score 0.38, reversed 0.82.
  - New numeric micro-example (this outline, verify trivially): [0.9, 0.1] ⊙
    [2.0, -1.0] = [1.8, -0.1].
  - English gloss (interview-locked): "I watch all of this in silence, and the
    connoisseurs speak inside me."
  - char-LM ANIM word (interview-locked): Պանիր.
- **Armenian rendering (definitive, learned in L20):** pdflatex CANNOT render
  Armenian - all Armenian appears inside matplotlib figures only; .tex body text
  stays English (refer to "the line's third word" etc.). In matplotlib: reuse the
  font-selection + glyph-check pattern from `py_src/forward_pass_anim.py` (promotes
  "Glyph ... missing from font" warnings to hard errors). NEVER mix Armenian into
  mathtext `$...$` strings - matplotlib's mathtext parser SILENTLY DROPS non-mathtext
  text appended in the same call; use plain-text annotations. Set encoding="utf-8"
  explicitly on logging FileHandlers and avoid printing Armenian to the cp1252
  console unguarded - Armenian log lines vanish silently otherwise.
- **`gradient_flow.py` regeneration:** use `nn.RNNCell`/`nn.LSTMCell` manual
  unrolling with per-step `retain_grad()` - fused `nn.RNN`/`nn.LSTM` calls hide the
  per-step recurrence from autograd's exposed gradients (all-zeros except the last
  step; discovered and fixed in the L20 build, see L20_DECISIONS.md item 7).
- **Every number on a slide is script-verified** (L20 lesson: an 11-word review was
  captioned "12 words"): tokenizer counts measured at run time; the gate-mask
  micro-example asserted in whatever script draws it (or verified in a throwaway
  check if pure TikZ); charlm samples quoted verbatim from the saved log, never
  paraphrased; the 0.38/0.82 callbacks must match forward_pass_anim.py's assertions.
- **No spec language on slides** (L20 leak: "one sentence each, no more" ended up in
  a footnote): outline directives in brackets are for you, never slide text.
- **No credit/attribution lines on WEB-IMG frames** (standing policy; a Karpathy
  credit had to be removed from L20). OWN-TEX needs no credit either.
- **OWN-TEX whitelist (this deck):** from `misc/dl4nlp/01_pre_transformer.tex`:
  the LSTM memory-highway frame (~lines 535-581), the gate-equations frame
  (~586-617), the seq2seq frame (~721+), the bottleneck frame (~802+). Check macros
  (`lightbg`, `-Stealth`) against `ml/preamble.tex` before pasting; patch locally
  (L20 already did this for -Stealth - copy its approach). Also read
  `misc/dl4nlp/03_tokenization.tex` before writing Section 2 - preview, don't
  contradict, the ch8 treatment.
- **WEB-IMG whitelist (this deck):** Chris Olah's LSTM chain diagram (alternative to
  the OWN-TEX highway), a Jay Alammar seq2seq still (alternative to the OWN-TEX
  diagram). Pick whichever reads better at 16:9; list the choice in Provenance.
- **`tokenizer_demo.py`:** install the tokenizer lib into `ma` with uv first; the
  encoding file downloads once - run interactively before the batch build and cache;
  the script fails loudly if the cache/library is missing.
- **`charlm_generate.py` corpus (interview-locked):** public-domain Tumanyan/Charents
  poems + fairy tales fetched from Wikisource ONCE at build, cleaned (strip wiki
  markup/headers), saved as `py_src/data/corpus_hy.txt` (~0.5-1 MB) and committed;
  the training script reads only the local file and fails loudly if it is missing.
  Keep it CPU-light (hidden <= 128, minutes); checkpoint samples at fixed step counts
  (0 / 100 / 1k / 5k / final) with seed 509 so the reveal is reproducible; quote REAL
  samples on the frames, never invented ones.
- **Links (interview-locked, verify each URL resolves at build):** the tiktokenizer
  web app (tiktokenizer.vercel.app) on the subword frame; TensorFlow Embedding
  Projector (projector.tensorflow.org) on the embeddings frame; Karpathy's char-RNN
  post (karpathy.github.io/2015/05/21/rnn-effectiveness/) on the demo frame. Same
  link-on-frame style as L17.
- **Epilogue facts:** web-verify before typesetting: Mamba (Gu & Dao, 2023), xLSTM
  (Beck et al. 2024, Hochreiter senior author), GRU (Cho et al. 2014), seq2seq
  (Sutskever et al. 2014), "Attention Is All You Need" (Vaswani et al. 2017).
  Keep Schmidhuber to one respectful line if mentioned at all.
- **No homework hook** - `rnn.qmd` gets links + credit line only.
- **Credit line** on the title/Outline frame, same LMU wording as L16/L20.
