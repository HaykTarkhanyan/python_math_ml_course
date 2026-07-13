# L20 RNN Foundations - build decisions

Logged 2026-07-13 at build time. The outline and chapter plan were followed exactly
where they specified content; this file records the calls made where they left room,
plus open questions for the reviewer. Not blocking - sensible choices were made and
logged per the task instructions.

## Decisions

1. **Chapter running review sentence**: "The pomegranates arrived fresh and sweet."
   used verbatim everywhere the outline says "the review sentence" (cold open, shuffle
   test, unrolling ANIM, worked numbers). This is now a chapter-level constant - L21
   must reuse it verbatim (recorded in the Provenance block too).

2. **The three shoehorn-hack reviews**: the outline's cold open only sketches these
   ("Fresh and sweet!" / a 12-word one naming Yerevan / "a 40-word complaint") without
   locking exact text for the long one. I used:
   - short: "Fresh and sweet!" (3 words, exact cold-open text)
   - medium: "The pomegranates arrived fresh and sweet, best I've had since Yerevan."
     (cold-open text, 11 words)
   - long: my own 42-word complaint ending in "...disappointing" (verdict word last),
     written to satisfy the outline's requirement that the long review's truncated
     ending "held the verdict." This same text is reused in the vanishing-gradient
     frame ("What vanishing means in practice") for continuity. **Open question**: if
     you have a preferred long-review wording, swap it in `pad_truncate.py`'s `REVIEWS`
     dict and re-run; the vanishing-gradient frame references it only by description
     ("the 42-word review... verdict... disappointing"), not by embedding the full text
     again, so a wording change only requires touching one script.

3. **Unroll ANIM word choice**: the review sentence has 6 words, but the locked worked
   numbers are for exactly 3 scalar tokens (embeddings `[2,1,3]`). I used 3 content
   words from the sentence - "pomegranates" (embedding 2), "fresh" (1), "sweet" (3) -
   so the ANIM and the "Worked numbers" math frame show identical values, per the
   instruction that "animation and math frame agree." The ANIM combines the forward
   pass (3 clicks) and the reversed pass (3 clicks) into one 6-frame flip-book, ending
   on the order-sensitivity punchline (2.54 vs 2.36) - this satisfies "~6 clicks" while
   using the full locked numbers (forward AND reversed) in one visual.

4. **Skipped the optional static OWN-TEX unrolled-RNN diagram** (`dl4nlp/
   01_pre_transformer.tex` lines ~407-472), which the outline listed as an optional
   "candidate," not mandatory. The mandatory ANIM already shows unrolling with the
   exact locked numbers; a second static unrolled diagram would repeat the same idea
   without adding information, against the style guide's "one idea per frame."

5. **OWN-TEX "cat" vanishing-gradient frame** (`dl4nlp/01_pre_transformer.tex` lines
   ~477-530): recolored `sampred`/`warnred` to `armred`/`armred!8`. Both original colors
   already exist in `ml/preamble.tex` so this was not required for compilation - it's a
   palette-consistency choice so the borrowed frame matches the rest of this deck
   (which uses armred/armblue/popblue/paramgreen/armorange throughout).

6. **Gradient clipping taught as a callback, not re-derived**: grepped
   `L15_training_neural_networks.tex` and found the identical formula already taught
   ("Saddles and cliffs -- and how we cope", `if ||grad|| > h: grad <- (h/||grad||)
   grad`). L20's frame cites it directly instead of re-deriving, per the build note.

7. **`gradient_flow.py` gradient-capture method rewritten mid-build**: my first attempt
   called `.retain_grad()` on the stacked output of a fused `nn.RNN`/`nn.LSTM` call and
   got all-zero gradients at every step except the last. That's not a numerical
   accident - a fused RNN kernel hides the per-step recurrence from autograd's exposed
   intermediate tensors, so `out.grad` only reflects the loss's *direct* touch point
   (the final timestep), not the decay through the recurrent chain. Fixed by manually
   unrolling with `nn.RNNCell`/`nn.LSTMCell` in a Python loop and calling
   `retain_grad()` on each individual `h_t`. This is a correctness fix, not a style
   choice - worth remembering for any future empirical-gradient figure in this course.

8. **Small diagrams built as TikZ, not figure scripts**: the sequence zoo, "one new
   arrow," "tokens minimally" one-hot diagram, backprop-through-time gradient-flow
   diagram, and the clipping-sphere diagram are all small boxes-and-arrows visuals with
   no data behind them, so they stayed TikZ per the style guide ("TikZ only for small
   throwaway visuals"). Everything data-driven (word_shuffle, pad_truncate, the
   unrolling ANIM, vanish_curve, gradient_flow) is a REGEN-FIG matplotlib script.

9. **Cold-open diagram simplified**: the first draft embedded the full review text
   inside the three input boxes, which produced a too-wide/too-tall TikZ (overfull
   vbox). Redesigned to short word-count labels ("3 words" / "12 words" / "40 words,
   verdict at the end") next to a "fixed input layer" box - keeps the pedagogical point
   (length mismatch) visible while fitting the frame. **Open question**: confirm this
   reads clearly when projected; the full-text version is easy to restore if there's
   room to enlarge the frame (e.g. dropping to a single-column layout).

10. **`pad_truncate.py` column layout**: computed shared per-column widths across all
    three review rows (the widest word in each of the 8 fixed-length columns sets that
    column's width for every row), so the "fixed length = 8" boundary line stays
    vertically aligned across rows. The medium review's truncation is not dramatic
    (its bag-of-words happens to hold "sweet" within the first 8 tokens) while the long
    review's truncation is - matching the outline's asymmetric point ("the short one
    wastes... the long one loses its ending, which held the verdict").

11. **Karpathy taxonomy image**: WEB-IMG download succeeded
    (`http://karpathy.github.io/assets/rnn/diags.jpeg`, saved to
    `fig/borrowed/karpathy_rnn_taxonomy.jpeg`), so the TikZ fallback was not needed.

12. **Frame count**: 23 logical frames (footer shows `N/23`) vs. the outline's "~21"
    target - each outline bullet mapped to one frame plus the 3 section transitions and
    title/outline/recap; not trimmed, consistent with the style guide's "no fixed
    length, don't split or trim just because it's long."

## Open questions for the reviewer

1. Is the invented 42-word "long review" text (see decision 2) acceptable, or would
   you like different wording? It is easy to swap in `pad_truncate.py`.
2. Does the simplified cold-open diagram (decision 9, word-count labels instead of full
   review text) read clearly enough, or would you prefer the fuller text restored on a
   wider layout?
3. `gradient_flow.py`'s LSTM curve (for L21's payoff figure `gradient_flow_vanilla_
   lstm.pdf`, not used in L20) does not yet show a dramatically better gradient-decay
   profile than the vanilla RNN in this run - 300 training steps on a hard-ish recall
   task may not be enough for the LSTM's advantage to show clearly. Flagging for the
   L21 builder to retrain longer or retune before relying on that comparison figure.
4. Confirm decision 4 (no second static unrolled-RNN diagram) is fine - the ANIM was
   judged sufficient on its own.

## Revision pass (instructor review)

Verdict: the RNN-explanation core "didn't really help" - diagrams and the 1-d scalar
walkthrough were too thin, and the instructor's own Armenian sentence never appeared.
Fix: one full concrete forward pass (toy 2-d net, real numbers, the chapter's actual
Armenian words), activation functions taught explicitly, and the Armenian line made
visible in two ANIMs. Logged here in the order the task specified.

1. **Toy 2-d net, chosen once and shared by two scripts**: `forward_pass_anim.py`
   (NEW) and `unroll_anim.py` (REWRITTEN) both hard-code the identical
   `W (3x2) / V (2x2) / b / U (2x1) / c` matrices - simple entries in the "0.5, -0.5,
   1, 0" style the instructor asked for. The toy vocabulary IS the Armenian line's
   first 3 words (Ես / այս / ամենինչ, read from `py_src/data/armenian_line.txt`, never
   retyped), so no separate "toy word" fiction was needed - every one-hot index maps
   directly to a real word from the chapter's sentence. Each script's docstring states
   the matrices are locked and shared, and both scripts assert the same rounded
   forward-pass values (`z[1]=[0.46,-0.46]`, `z[2]=[0.65,-0.23]`, `z[3]=[0.10,0.59]`,
   `score=0.38`) so the two ANIMs can never silently drift apart.

2. **Frame count for the forward pass**: settled on 7 clicks (`forward_pass_0..6.pdf`),
   the low end of the requested "~7-9" - one setup frame (net + matrices visible,
   z^[0]=0), one frame per word (3), one raw-readout frame, one sigmoid/score frame,
   and one closing "teaching beat" frame that re-highlights word 1's one-hot -> W row
   selection. Judged sufficient without a redundant extra click; adding more would
   have repeated the same idea rather than adding one.

3. **Real bug found and fixed: mathtext + Armenian text must never share one
   `ax.text()` call.** matplotlib's mathtext parser (triggered by any `$...$` in a
   string) silently DROPS an Armenian substring appended outside the math delimiters
   in the same call, instead of raising - e.g. `f"one-hot(...) -> ... {word}"` with a
   `$W^\top x$` mathtext segment rendered as `"one-hot( ) -> ..."`, the word simply
   missing, no warning, no error. This is a distinct failure mode from the "missing
   glyph" case (which the warnings-to-error promotion catches) - mathtext's internal
   font resolution for the non-math portions of a mixed string does not go through the
   same per-glyph system-font fallback as plain `ax.text()` calls, and it fails
   silently rather than warning. Fix: any annotation string that embeds a variable
   Armenian word now uses plain arithmetic notation (`"z[2] = ..."`, no `$...$`
   anywhere), matching the plain-text convention `unroll_anim.py` already used for its
   scalar-RNN annotations, plus an explicit `fontfamily=ARM_FONT` on those calls as a
   second line of defense. Caught by visually inspecting a rendered PNG of each new
   frame before trusting the PDF - the PDF's extracted text layer is unreliable for
   Armenian glyphs from a subsetted TrueType font (a separate, cosmetic-only ToUnicode
   issue) and must not be used as the check; only the rendered pixels prove correctness.

4. **Font choice: Segoe UI, not Noto Sans Armenian.** The task's preference order was
   Noto Sans Armenian first; `Get-ChildItem C:\Windows\Fonts` confirmed it is NOT
   installed on this machine, only `Sylfaen` and `Segoe UI` (both ship with Windows
   11). Rendered the exact Armenian line with both as a side-by-side PNG check - both
   produce correct, non-tofu glyphs. Picked Segoe UI (modern, matches the deck's
   general sans-serif feel slightly better than the older-style Sylfaen). Both
   `forward_pass_anim.py` and `unroll_anim.py` promote matplotlib's "Glyph ... missing
   from current font" warning (`warnings.filterwarnings("error", message=".*missing
   from current font.*")`) to a hard error - confirmed this actually fires by
   temporarily testing against a font with no Armenian coverage before locking it in.

5. **Logging + Windows console encoding gotcha.** The very first run of
   `forward_pass_anim.py` silently dropped every log line containing Armenian text
   from BOTH the console AND the log file (Python's `logging` module swallows
   `UnicodeEncodeError` inside a handler's `emit()` and just prints "--- Logging error
   ---" to stderr, not a hard crash) - Windows' default console codepage (cp1252) and
   `FileHandler`'s default encoding can't encode Armenian. This is exactly the kind of
   silent data loss the house "fail loudly" rule exists to prevent, so both scripts now
   call `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` and open the
   `FileHandler` with `encoding="utf-8"` explicitly. Worth remembering for any future
   script in this course that logs non-Latin text on Windows.

6. **`unroll_anim.py`'s job deliberately narrowed.** Previously it carried both the
   full worked-numbers arithmetic AND the order-sensitivity punchline on a 1-d scalar
   net; now that the full forward-pass ANIM owns the detailed arithmetic, `unroll_anim.py`
   shows only a one-line recap per step (`z[2] = tanh(V^T z[1] + W^T x + b) = [...]`,
   no term-by-term breakdown) and ends on the explicit forward-vs-reversed score
   comparison (0.38 vs 0.82 - a much starker contrast than the old 2.54-vs-2.36
   scalar numbers, and a genuine "different answer" rather than just "different
   number").

7. **The old scalar 1-d model (`W=0.5, V=0.8`) was fully removed from "Unrolling" and
   the deleted "Worked numbers" frame, then reintroduced FRESH, unchanged, only at
   "Predict-first: the fading number"** ("Strip the net down to 1-d to see the decay
   cleanly..."). The locked numbers there (`0.8^29≈0.0015`, `1.25^29≈646`) and
   `vanish_curve.pdf` were NOT touched, per the task's explicit lock.

8. **`word_shuffle.py` judgment call: kept the English review sentence, did NOT
   switch to the Armenian line.** Computed the actual box-row width with the
   script's own `box_width()` formula before deciding (not a guess): the Armenian
   line has 12 words vs. the review sentence's 6, and its row is ~53% wider
   (24.6 vs 16.1 width units) at the same figure size, plus the bag-of-words
   histogram would double from 6 bars to 12. That is a real legibility cost for zero
   pedagogical gain (the order-invariance point is already made cleanly at 6 words),
   so the frame is unchanged. Revisit if the instructor wants it anyway.

9. **New activation-functions frame placed immediately after the forward-pass trio**
   (anatomy -> ANIM -> summary -> activations -> Unrolling), right where tanh has just
   been used for the first time with real numbers. It explicitly points forward to
   "The full derivation, part 2 - eigenvalues" for the `|sigma'|<=1/4` bound (that
   frame itself was NOT touched, per the task's lock on both derivation frames) so the
   two frames stay consistent without duplicating the derivation.

10. **Compile fixes on the frames added/changed this pass**: the "Anatomy of one RNN
    step" frame's TikZ diagram is genuinely ~11cm wide at native scale (5 nodes plus
    the readout branch) - it does not fit a 0.58-textwidth column (overfull hbox,
    129pt too wide). Restructured the frame to single-column (text above, diagram
    centered below, matching "The shoehorn hacks" frame's existing layout pattern)
    and wrapped the TikZ in `\resizebox{0.98\linewidth}{!}{...}` as a safety margin.
    The forward-pass ANIM's images were shrunk from 0.92 to 0.82\linewidth (an
    overfull vbox, ~14pt too tall, on all 7 `\only` frames) and the new activation
    frame's left-column text was tightened from `\small` to `\footnotesize` (an
    overfull vbox, ~40pt too tall). Two pdflatex passes after these fixes show zero
    "!" error lines and zero NEW overfull warnings - the three remaining overfull/
    underfull warnings in the log (lines ~121, ~133, ~578 of the .tex) are pre-existing,
    in frames this pass did not touch ("The shuffle test", "The shoehorn hacks",
    "Empirical proof").

11. **Frame count: 26** (up from 23) - net +3 (4 new frames: anatomy, forward-pass
    ANIM, forward-pass summary, activation functions; -1 deleted: "Worked numbers: a
    1-d RNN reads three tokens"), vs. the task's "~25" estimate. Not trimmed further,
    consistent with decision 12 above (no fixed length target).
