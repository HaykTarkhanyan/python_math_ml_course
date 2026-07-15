# L21 The Road to Attention - build decisions

Logged 2026-07-13 at build time. The outline and chapter plan were followed exactly
where they specified content; this file records the calls made where they left room,
the mid-build instructor scope change and how it was carried out, plus open questions
for the reviewer. Not blocking - sensible choices were made and logged per the task
instructions.

## Instructor scope change (mid-build, 2026-07-13)

The outline (and the original task brief) called for: fetching a public-domain
Armenian corpus from Wikisource (Tumanyan/Charents), training a real char-LSTM on it,
downloading a small GloVe subset for the embeddings figure, and installing/using
tiktoken for the tokenizer figure. Partway through the build (corpus-fetch step
already underway via the MediaWiki API), the instructor sent a scope change:
**no real training runs, no dataset/model downloads.** Concretely, per the
instructor's message:

1. `charlm_generate.py`: drop the Wikisource fetch and the char-LSTM training
   entirely. Hand-write an illustrative, clearly-labeled TYPICAL progression instead,
   keeping the per-click ANIM format and the loss curve as a labeled schematic.
2. `charlm_anim.py`: illustrative (hand-chosen, plausible) probability bars are now
   the default, no trained model required.
3. `embedding_2d.py`: no GloVe/gensim download. Hand-placed 2-D schematic instead,
   clearly labeled.
4. `tokenizer_demo.py`: keep it only if tiktoken installs and runs trivially, with
   MEASURED counts; otherwise go illustrative.
5. `gradient_flow.py`: no extended retraining. At most a CPU-seconds quick run for
   `gradient_flow_comparison.pdf`; report honestly if the LSTM gap is not dramatic.
6. Keep as planned: the Olah + Alammar WEB-IMG downloads, web verification of the
   epilogue facts and the three link URLs, all TikZ/diagram frames, the LSTM anatomy
   frame with the numeric gate-mask example, and the overall deck structure.

**What actually happened, per item:**

1. **`charlm_generate.py` - fully re-scoped, no corpus, no training.** The
   Wikisource fetch (Armenian MediaWiki API, `hy.wikisource.org`) had already located
   real category pages (Tumanyan's fairy tales, poems; found via
   `action=query&generator=categorymembers`) before the scope change landed - that
   work was abandoned mid-stream, no `corpus_hy.txt` was ever written, no training
   ever ran. The script instead hand-writes five TYPICAL-progression samples (step 0
   untrained gibberish -> ~100 letter-statistics -> ~1,000 word-like fragments ->
   ~5,000 almost-literary -> final, seeded with the Armenian line's first three words
   and continuing it) and asserts, in `main()`, that the final sample actually starts
   with those three words (a cheap correctness check on the one hard constraint that
   survived the re-scope). `charlm_loss.pdf` is an exponential-decay curve with
   unitless "(schematic)" axis labels and no invented numbers - only the shape
   carries meaning. Every frame using this script's output is labeled "ILLUSTRATIVE"
   or "typical progression, not a measured run" directly on the figure, not just in
   surrounding prose, so the deliverable is self-documenting if it circulates without
   its slide context.
2. **`charlm_anim.py` - illustrative by design, no fallback needed.** Probabilities
   for each of the six Պանիր steps are hand-chosen to be linguistically plausible
   (vowels dominate after the first consonant; the letter that continues toward the
   target word gets the highest bar) but are explicitly never claimed as measured.
   Labeled "ILLUSTRATIVE - not measured from a trained model" on every frame.
3. **`embedding_2d.py` - hand-placed schematic, ~31 words across 6 clusters.** The
   king/queen/man/woman (+ prince/princess) parallelogram is constructed so
   `king - man` exactly equals `queen - woman` as 2-D vectors (asserted with
   `math.isclose` in `main()`, not just eyeballed) - this is the one place where
   "illustrative" and "exactly correct" overlap: the arithmetic is real, the
   *positions* are not learned. The other five clusters (fruit, animals, places,
   sentiment, verb tense) are placed by hand into visually separated regions purely
   for legibility; no distances or angles there carry any claim.
4. **`tokenizer_demo.py` - kept, fully real.** `uv pip install --python
   ./ma/Scripts/python.exe tiktoken` resolved and installed in under a second
   (already-cached wheel), and `tiktoken.get_encoding("cl100k_base")` downloaded its
   encoding file in about 6 seconds on first run and 0.3 seconds on a warm cache -
   trivially cheap, so this stayed in scope per the instructor's condition. All three
   panels use real `cl100k_base` (co-owned by GPT-3.5/GPT-4) measurements: the review
   sentence needs 41 characters / 7 words / 10 subword tokens; the Armenian line
   measures 104 tokens vs. 18 for its English gloss (5.8x). Numbers are logged and
   asserted-by-construction (`ax.bar_label` reads directly off the measured arrays,
   never a hand-typed number).
5. **`gradient_flow.py` - quick re-run only, honest negative-ish result reported.**
   Added `TRAIN_STEPS_QUICK = 1200` (4x the original 300, still ~35 seconds total for
   both cells on this machine - well inside "CPU-seconds", not the "longer
   recall distance" extended retraining the outline originally asked for). Result:
   at this budget, the LSTM's early-time-step gradient norm is NOT larger than the
   vanilla RNN's (measured ratio 0.02x over the first third of the sequence - if
   anything, slightly worse in this run, most likely because PyTorch's default
   `LSTMCell` does not apply the classic "forget-gate bias = 1" initialization trick,
   so at low training budgets the forget gate starts near "half open" rather than
   "remember by default"). Per the house rule ("report what the data shows, never
   tune until it flatters"), this was NOT hidden or re-run with a friendlier seed/
   hyperparameters - `fig_gradient_comparison()` picks its own title
   ("Measured at this training budget: no dramatic LSTM advantage yet" vs. a more
   favorable title) based on a computed `honest_gap` boolean, and the frame text
   ("Does it work? Proof + price") states the honest result plainly, then falls back
   on the *architectural* argument (additive cell update, established analytically
   on the previous frame) as the reason to believe the LSTM works, independent of
   this one small toy net's convergence. `gradient_flow_vanilla.pdf` and
   `gradient_flow_vanilla_lstm.pdf` (both from the L20 build) were verified
   byte-identical (`md5sum`) before and after this run - the script's `main()` no
   longer calls `fig_gradient_flow()` at all (it now hard-fails if either expected
   file is missing, instead of silently regenerating and risking a silent overwrite).
6. **WEB-IMG, web verification, TikZ, anatomy frame, deck structure: all proceeded
   as planned**, unaffected by the scope change. Details below.

## Decisions

1. **`gradient_flow.py` refactor**: `train_and_measure()` now takes a `train_steps`
   keyword (default `TRAIN_STEPS=300`, unchanged for the original two figures); the
   new `TRAIN_STEPS_QUICK=1200` quick comparison run uses its own freshly-seeded
   generator (`torch.manual_seed(SEED)` + a new `torch.Generator().manual_seed(SEED)`)
   so it is independently reproducible and does not depend on the original run's
   RNG state having been consumed first.

2. **Jay Alammar seq2seq "still" was captured with ffmpeg, not downloaded as an
   image.** `jalammar.github.io`'s seq2seq visuals are now served as `.mp4`
   animations (`seq2seq_1.mp4` .. `seq2seq_9.mp4`), not static images - the
   `.png` paths implied by the site's older layout all 404. Downloaded all 8 clips,
   read the surrounding post text to identify which clip shows the generic
   input-sequence -> encoder -> decoder -> output-sequence idea (clip 3: "The
   encoder processes each item... compiles into a vector (called the context)...
   sends it to the decoder"), and extracted a still frame with
   `ffmpeg -ss 8.0 -frames:v 1` at a timestamp where the encoder/decoder/context
   labels are all visible and the input/output boxes are in their generic
   (unfilled, abstract-triangles) state - deliberately NOT the French/English
   translation variant (clip 4), because that clip's animation types words in one at
   a time and no single frame shows both a full source and full target
   simultaneously, which would have looked broken as a static still. The frame used
   for the cliffhanger's dashed-arrows diagram was explicitly NOT this Alammar
   material (see decision 3) - a candidate clip (7) that shows all-encoder-states
   reaching the decoder already has "SEQUENCE TO SEQUENCE MODEL WITH ATTENTION" and
   "Attention Decoder RNN" baked into the image text, which would spoil the deck's
   locked cliffhanger (attention must not be named). Logged as an open question
   below in case the reviewer prefers a different still.

3. **The cliffhanger diagram is an original TikZ, not OWN-TEX or WEB-IMG.** The
   outline asked for "the seq2seq diagram with dashed arrows from every encoder
   state to the decoder, drawn but unlabeled." `dl4nlp/01_pre_transformer.tex`'s own
   seq2seq frame (~721-797) was read for its node layout (encoder circles, context
   box, decoder circles) but not copied verbatim - the dashed all-to-all arrows are
   a fresh addition on top of a simplified version of that layout, since neither the
   OWN-TEX nor the Alammar material already showed this specific "unlabeled preview
   of attention" idea without also naming attention.

4. **No `ml/ch6_cnn/py_src/timeline_ribbon.py` exists in this repo** (checked at
   build time via `Glob`) - the outline's "if helpful" pointer to it as a style
   reference could not be followed. `rnn_timeline.py` is a fresh ribbon design in the
   same general house language (horizontal axis, colored circular markers, boxed
   labels, Armenian-flag palette for the 3 eras) rather than a copy of an existing
   pattern.

5. **Epilogue facts web-verified, no corrections needed.** All five claims in the
   outline matched what was found: Mamba (Gu & Dao, arXiv 2312.00752, Dec 2023);
   xLSTM (Beck et al., NeurIPS 2024 spotlight, arXiv 2405.04517, Sepp Hochreiter
   listed as the paper's senior/last author); GRU (Cho et al., arXiv, June 2014,
   "Learning Phrase Representations using RNN Encoder-Decoder for Statistical
   Machine Translation"); seq2seq (Sutskever, Vinyals & Le, NeurIPS 2014, arXiv
   1409.3215); "Attention Is All You Need" (Vaswani et al., NeurIPS 2017, arXiv
   1706.03762). All three links resolve live: `tiktokenizer.vercel.app` (a working
   GPT-4o token-counting tool), `projector.tensorflow.org` (TensorFlow's standalone
   Embedding Projector), `karpathy.github.io/2015/05/21/rnn-effectiveness/` ("The
   Unreasonable Effectiveness of Recurrent Neural Networks", confirmed live).

6. **LSTM gate-mask numeric example verified trivially before typesetting**:
   `[0.9, 0.1] * [2.0, -1.0] = [1.8, -0.1]` (element-wise), matching the outline's
   locked value exactly - checked with a one-line Python computation, not just
   arithmetic by eye.

7. **Vanilla-vs-LSTM parameter count (288 -> 1,152) clarified to mean the recurrent
   cell only, excluding the readout.** The outline's "our 288-param vanilla becomes
   1,152" only reconciles if 288 = `W(16) + V(256) + b(16)` for the recurrent cell
   alone (`d_x=1, d_h=16`), NOT including the `U`/`c` readout params that L20's
   "Parameter sharing" frame counted separately (which gave 305 total, cell + readout,
   for a very similar but distinct example). The slide text says "vanilla cell"
   explicitly to avoid contradicting L20's own 305-parameter figure.

8. **`tokenizer_demo.py`'s box-sizing bug and fix, worth remembering for future
   figure scripts.** An early version measured each token box's width from the
   *actual rendered text extent* (via `ax.text(...).get_window_extent()`) rather than
   a guessed character-count formula - correct in isolation, but `fig.tight_layout()`
   called later (right before `savefig`) silently repositions/resizes the Axes
   bounding box within the Figure, which changes the data<->pixel scale that the
   measurement pass had already locked in. Result: every token box was too narrow
   for its own (correctly-measured-at-the-time) text, and text visibly overlapped
   its neighbor's box border. Fixed by removing `fig.tight_layout()` entirely
   (`bbox_inches="tight"` on `savefig()` does not have this effect - it crops the
   exported page after the fact, it does not re-lay-out the Axes) and by re-fetching
   `fig.canvas.get_renderer()` after every `fig.canvas.draw()` call inside the
   measurement loop (a stale renderer reference was a second, smaller contributor).
   Confirmed the fix by rendering a page image and visually inspecting - the PDF's
   vector correctness cannot be eyeballed from source alone.

9. **A real Unicode-in-`.tex` bug was caught and fixed by the compile step itself,
   not by review.** An early draft of "The mechanics, click by click" wrote the word
   Պանիր literally in the `.tex` body ("Word (the course's cheese joke): **Պանիր**
   ...") - a direct violation of the hard rule that Armenian is figure-only.
   `pdflatex` failed loudly (`! LaTeX Error: Unicode character Պ (U+054A) not set up
   for use with LaTeX`), which is exactly the kind of fail-loudly behavior the house
   rules ask for. Fixed by rewriting the sentence to refer to the word by
   description ("five letters, shown in the figure below") instead of by name.
   Grepped the whole file afterward for the Armenian Unicode block to confirm no
   other body-text occurrences slipped in (three remained, but all after
   `\end{document}`, inside the never-compiled Provenance comment block, so left
   as-is).

10. **A second, silent LaTeX rendering bug found only by rendering pages to PNG**:
    a bare `->` inside a plain-text TikZ `\node` (not inside a `\draw` arrow-style
    argument) rendered as `-¿` instead of an arrow. `ml/preamble.tex` loads
    `\usepackage{fontenc}` without the `[T1]` option, so the document uses the
    default OT1 font encoding, under which `>` is not a directly-typesettable
    character in text mode (a classic, easy-to-miss LaTeX gotcha) - it silently
    substitutes a different glyph instead of erroring. `pdflatex` gave no warning or
    error for this; it only showed up as visibly wrong text in a rendered page image.
    Fixed the one occurrence (in "The objective" frame's TikZ) by using `$\to$`
    instead of the bare `->`; grepped the whole file for `\->` afterward and
    confirmed every other occurrence is either inside a `\draw[->, ...]` style
    argument (fine, TikZ-internal, never typeset as text) or inside the
    never-compiled Provenance comments.

11. **Frame count: 25 logical frames** (footer shows `N/25`; 37 physical PDF pages
    once ANIM `\only<n>` overlays are counted - 6 for the mechanics ANIM, 5 for the
    generation-demo ANIM, plus 2-page `\pause` overlays on the cold open and the
    anatomy/bottleneck frames), vs. the outline's "~24" target - the closest match of
    the two decks built so far (L20 landed at 26 vs. "~21"). Not trimmed further,
    consistent with the style guide's "no fixed length" precedent.

## Open questions for the reviewer

1. Is the Alammar seq2seq still (a captured video frame, not a native image download,
   per decision 2) an acceptable substitute for a static WEB-IMG? If the reviewer
   prefers a genuine still image, the OWN-TEX dl4nlp seq2seq diagram is the
   documented fallback and is a straightforward swap.
2. Does the "Does it work? Proof + price" frame's honest non-result (decision on
   `gradient_flow.py` above: the quick 1,200-step re-run does not show a dramatic
   LSTM gradient advantage, and if anything shows the opposite in this run) read as
   intended - a genuine teaching moment about architecture-vs-convergence - or would
   a purely schematic/conceptual illustration (as the instructor's scope-change
   message allowed as an alternative) communicate the point more cleanly without the
   risk of a confusing negative result on the slide?
3. The illustrative char-LSTM demo (`charlm_generate.py`, `charlm_anim.py`) no longer
   contains any real Armenian-corpus training. If a real trained demo becomes
   feasible in a future session (e.g. a follow-up pass with an explicitly-approved
   compute/download budget), both scripts are structured so that swapping in real
   samples/probabilities is a local change (replace the `CHECKPOINTS` list / `STEPS`
   list) that would not require touching the deck's `.tex` or frame structure.
4. The Wikisource corpus research (Tumanyan/Charents category pages, confirmed to
   exist via the MediaWiki API) was abandoned mid-stream when the scope change
   arrived, with no files written under `py_src/data/`. Confirming there is nothing
   to clean up: no `corpus_hy.txt`, no partial downloads, were left on disk.
