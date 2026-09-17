# Retiring L14/L15 onto the `dl_*` embed set - coverage audit and plan

Audited 2026-09-14 (Claude), on the instructor's instruction: **keep the seven `dl_*` decks,
completely retire `L14_neural_networks` and `L15_training_neural_networks`**, on the premise that
everything in the house decks is already in the embed set - and where it is not, work out how to
extend.

Method: full read of both house `.tex` files and all seven `dl_*.tex`; `pdftotext` extraction of
all 21 LMU chunk PDFs in `lmu_src/` (4,922 lines, 251 KB) plus a page-level title outline, then a
keyword sweep of that text for every candidate gap; the Welch Labs transcript read end to end;
`detect_clipped_slides.py` run over the chapter; web re-verification of the dated 2026 claims.

**Note on `README.md`:** it currently frames `L14`/`L15` as "the canonical, self-contained decks"
and the `dl_*` set as a parallel reference-grade treatment, and says the two "coexist on purpose".
That is now superseded by this decision, and the README needs rewriting (item 4 below).

---

## 1. Verdict

**Retirement is viable, but not free.** The embed set already covers the great majority of
L14/L15, and on eight counts it is clearly better. But **six content gaps and four structural
gaps** are real. Five of the six are single frames that already exist, fully written and
fact-checked, inside L14/L15 - so the work is mostly *porting*, not authoring.

The one that would hurt most if missed: **the parameter-count bridge into backprop**. It is the
strongest motivational beat in the two decks and it vanishes completely.

---

## 2. Confirmed gaps - present in L14/L15, absent from the whole `dl_*` set

"Absent" below means absent from our sprinkle frames *and* from all 21 embedded LMU chunks,
verified by keyword sweep over the extracted LMU text.

| # | What | Where it lives now | Evidence it is absent | Proposed home |
|---|---|---|---|---|
| 1 | **The 109,386-parameter bridge.** 784-128-64-10 parameter table, then "gradient descent needs a derivative for every one of these; one forward pass per weight is hopeless" | L14 "The catch: a lot of parameters" + the whole L15 cold open | 0 hits for `per weight`, `one forward pass`, `too expensive`, `naive`, `infeasible`, `prohibitiv`. LMU counts parameters only for the **9-parameter XOR toy** (`04_xor` p5) and never uses a count to motivate backprop | End of `41_2_multilayer_nets` (just before "Next: learning the weights"), which is exactly where the bridge belongs |
| 2 | **Gradient checking.** Finite-difference sanity check, agreement to ~1e-7, "debug only" | L15 "Sanity check: is your backprop even correct?" | 0 hits for `gradient check`, `numerical gradient`, `finite differ` | `42_training_backprop`, after the LMU backprop-2 recursion |
| 3 | **The training loop as runnable code.** The 7-line PyTorch loop, and `loss.backward()` named | L15 "The training loop, in full" + "You will never write this by hand: autograd" | 0 hits for `zero_grad`. LMU has SGD **pseudocode** only (`01_basic-training` p11-12), no framework code | `42_training_backprop`, after the hardware/software chunk |
| 4 | **Tuning machinery is not new.** grid / random / Optuna, the `[08]` callback, early stopping as the free tuner | L15 "Tuning machinery is not new" | 0 hits for `grid search`, `random search`, `Optuna` | `xx_optimization_init_activations`, before the chapter close |
| 5 | **Symptom -> fix hyperparameter table.** 6 knobs x under/overfitting | L15 "Hyperparameters: symptom -> fix" | `43_nn_regularization` has a *regularizer-only* "which knob when" table; no hyperparameter table anywhere | Extend the existing `43_nn_regularization` table, or add the full one alongside item 4 |
| 6 | **Why trees still win on tabular.** 5 reasons, Grinsztajn et al. (2022), the rule of thumb, the ch4 callback | L14 "Why trees still win on tabular data" | LMU `01_introduction` p5 has a **one-sentence** version (see 3 below) | After the LMU intro chunk in `40_intro_history` |

### Thinner, not absent - instructor's call

7. **"A neuron is a regression you already know."** L14's side-by-side table (identity ->
   linear regression, sigmoid -> logistic, softmax -> multiclass) is the chapter's spine callback
   per `NN_CHAPTER_OUTLINE.md`. `41_1_neuron_to_network` compresses it to one recap bullet.
8. **Softmax + cross-entropy gives `p - y` per class.** `42_training_backprop` states `f - y` for
   the sigmoid case only. The practical is 10-class, so the generalization matters (this is also
   open item 8 in `REVIEW.md`).

---

## 3. Not gaps - checked and dropped

Recording these so nobody rebuilds them. Each was on my candidate list and the LMU text disproved
it:

- **The tabular caveat.** LMU `01_introduction` p5 says outright: "for tabular data, deep learning
  is rarely the correct model choice... random forests or gradient boosting will outperform deep
  learning most of the time. One exception is data with categorical features with many levels."
  That exception is something **L14 does not have**. Only the depth is lost, not the point.
- **Zero-initialization symmetry breaking.** LMU `01_basic-training` p8 covers it explicitly
  ("we somehow must break symmetry... both neurons will have the same gradient update and learn
  the same features"). `REVIEW.md` item 5 lists this as *missing from L15* - so the embed set
  already **fixes** a known house-deck gap.
- **GPU / TPU.** LMU `05_hardware-and-software`, 10 pages.
- **TensorFlow Playground.** 8 hits in LMU, including a built-in exercise slide, plus our own
  richer "See it learn" frame.
- **Worked forward pass and by-hand backprop.** LMU carries its own (`05_single-hidden-layer`
  p9-17; `03_backpropagation1`, 20 pages on an XOR net).

---

## 4. What the embed set does better (do not regress these)

- **Optimization challenges: 38 pages** (ill-conditioning, curvature, Hessian, saddles, cliffs)
  against L15's two consecutive **text-only** frames. `REVIEW.md` item 11 flags exactly those two
  frames as the deck's only visual-free stretch. Retirement solves it for free.
- **Advanced optimizers: 49 pages** (momentum, Nesterov, AdaGrad, RMSProp, Adam, BatchNorm)
  against L15's single frame that punts to the Optim module.
- **Dropout: 23 pages**, including weight scaling and dropout-vs-weight-decay.
- **Activations: 18 pages**, including the ReLU generalizations.
- **Depth and folding:** our folding figures **plus** three full-bleed Welch Labs stills.
- **The L14/L15 rounding bug disappears.** `REVIEW.md` item 1 (L14 prints `f_in = 0.81`, L15
  prints `0.80` for the same forward pass) becomes moot, since LMU carries its own worked example.

---

## 5. Structural gaps (house style, not content)

These are the honest cost of the switch, and worth a decision rather than a silent acceptance:

1. **No cold-open hook.** Every `dl_*` deck opens with a title frame and goes straight into LMU
   pages. The two-moons predict-first hook still exists but is demoted to a mid-deck aside in
   `41_1_neuron_to_network`. `SLIDE_STYLE.md` makes the cold open a default.
2. **No Outline frame and no `\section` structure**, so no navigation and no `\tableofcontents`.
3. **4:3, not the house 16:9** - unavoidable while embedding, and deliberate.
4. **Naming and registration.** `dl_*` violates the `NN_topic` convention in `CONVENTIONS.md`, and
   the set is still absent from `_quarto.yml` and the chapter page. The README already tracks both.

---

## 6. Fact-check of the dated 2026 claims

`REVIEW.md` item 15 asked for these to be re-verified before each delivery. Done, and **two errors
were found and fixed** in `40_intro_history.tex` (deck recompiled, 54 pages, 0 errors, frame
verified visually):

- **FIXED - wrong date.** The Bun rewrite ran **3-14 May 2026** (merged 14 May), not "July 2026".
  Root cause worth remembering: `REVIEW.md` records these facts as "web-verified in **July 2026**",
  and the *verification* date became the *event* date on the slide.
- **FIXED - conflated numbers.** "~50 Claude Code agents in parallel" mixed up agents and
  workflows. It was **~64 Claudes at a time** (4 workflows x 16 instances) across **~50 dynamic
  workflows**. Now stated correctly.
- **ADDED.** Named Andrew Kelley and dated the "unreviewed slop" quote to July 2026, since that
  is a response to the May rewrite, not the rewrite itself.

Verified correct, unchanged: $165,000 API cost; ~1,300 lines/min peak; 1M+ lines (+1,009,272);
100% of the test suite on all platforms (1,386,826 `expect()` calls); NVIDIA first to $4T on
9 July 2025; IMO 2025 both at 35/42 solving 5 of 6 in natural language; Figure Helix ~109,000
lines of C++ replaced by a 10M-parameter net (precisely 109,504, in Helix 02's "System 0", at
1 kHz).

Still unverified, flagged for the instructor: NVIDIA's "~80% of the AI-chip market"; "Blackwell
chips are sold out"; the "$40 trillion" robotics market figure. One currency nit: the deck says
"Claude Fable 5 (2026)"; there is now a Fable 5.1. One nuance worth a clause: at IMO 2025 only
Gemini was **officially graded** by IMO coordinators; OpenAI self-published.

---

## 7. Transcript mining - Welch Labs, the one reference video in this chapter

The deck uses 6 stills. `_reference_welchlabs/frames/` holds 45 timestamped frames **plus 39
curated `s1_*` frames that are currently unused**. Three teaching beats in the transcript are on
no slide:

- **The failed-initialization story (15:00-18:30) - the strongest unused asset.** Welch Labs shows
  a run that *did not work*: a bad random init put the surfaces the wrong way round, backprop
  pushed the decision boundary off the planes, the whole town landed in the zeroed-out part of the
  ReLU, and gradient descent could not recover because gradients through that region are zero.
  That is **initialization, dead ReLU, and "existence is not findability" in one concrete visual
  story**. All three are currently asserted abstractly across three different decks.
- **Dead neurons at scale (27:50).** "a couple of the neurons in our second layer don't have any
  colored regions... Dead neurons like this are common. And a reminder that gradient descent gives
  no guarantees about efficiently using our model architecture." The dying-ReLU line in
  `xx_optimization_init_activations` has no picture; this is one.
- **The loose-bound caveat (26:00-26:40) - a factual-honesty gap.** The transcript is explicit:
  "these numbers are theoretical upper bounds and... these bounds are very loose. In practice, we
  typically do not see exponential growth." `welchlabs_region_growth.jpg` is a full-bleed showing
  **2,081 vs 72,807,417** with **no caveat on any slide**. The provenance comment in
  `41_2_multilayer_nets.tex` says "Caveat to narrate", which relies on the lecturer remembering.
  A student reads it as a measured result. One line fixes it.

Also available and unused: the `s1_scale*` ladder (8 -> 16 -> 32 -> ... -> 100k neurons, each with
a count badge) is a ready-made build-up of "more neurons = more folds", currently represented by a
single summary still.

**Other transcripts in the repo**, for the record: `misc/grokking/` holds a Welch Labs grokking
video **with an already-built 39 KB deck**, reproduction code, and both papers - relevant only if a
grokking act is ever wanted here. The `misc/dl4nlp/_yt_videos/karpathy_*` set is the makemore
lineage behind the name-inventor practical, already mined.

---

## 8. Modern topics worth considering (needs a scope decision)

All of the following return **zero hits across all 21 LMU chunks**, so the embed set cannot supply
them: `AdamW`, `LayerNorm`, `GELU`, `SiLU`, `Swish`, `skip connection`, `scaling law`, `grokking`,
`double descent`, `adversarial`, `transfer learning`.

### Resolved placement (instructor decision, 2026-09-14)

A survey of every downstream chapter changed the answer for three of these: the content already
exists elsewhere, so ch5 should **point**, not teach.

| Topic | Decision | Where it actually lives |
|---|---|---|
| **AdamW** | **Add to ch5.** DONE - the `xx_optimization` cheat-sheet now leads with AdamW and explains decoupled weight decay | Was nowhere in `ml/` except `llm_training/19_making_training_fast` |
| **Double descent** | **Add to ch5**, in `43_nn_regularization` after the training-curves frame | **Zero hits anywhere in the repo.** Genuinely new. Revises the `[06]` bias-variance U-curve (`02_main_concepts/06_overfitting_cross_validation.tex`) |
| **LayerNorm** | **One-line contrast in ch5** + forward pointer. DONE, on the same cheat-sheet | Full treatment already in `ch9/L25_transformer_block` ("Add & Norm"), `dl4nlp/02_transformers`, `ch19/L45` |
| **GELU** | **Do NOT teach in ch5.** It was already name-dropped here and, worse, unexpanded. DONE - acronym now expanded and pointed at ch9 | `ch9/L25` (incl. SwiGLU), `dl4nlp/02_transformers`, `dl4nlp/06_early_notable_models`, `llm_training/19` |
| **Skip / residual connections** | **Nothing to build. Teach in ch6, as planned.** ch5 may add at most a one-line teaser | `ch6/L17_cnn_architectures` already has "The residual idea" (skip connection carries `x` for free, `residual_block.pdf`) and "You have seen residual learning before" (= gradient boosting in depth). Also `ch7/L21`, `ch9/L25`, `ch10/L28` |
| **Transfer learning** | **Nothing to build. Already a whole ch6 deck.** L14 promised "more on this later"; that promise retires with L14 | `ch6/L18_transfer_learning` ("Transfer Learning and CNNs in Practice": freeze vs fine-tune, the BatchNorm gotcha, Grad-CAM / Clever Hans, labelling economics) |
| **Muon** | **Deferred.** Optional one-liner only | Verified: Keller Jordan et al., late 2024; ~2x faster than Adam in LLM pretraining; used for Kimi K2 and GLM-4.5. Pretraining-focused, mismatches Adam-pretrained finetuning |
| **Grokking** | **Deferred.** A built deck already exists outside `ml/` | `misc/grokking/slides/grokking_mechanistic_interpretability.tex` (39 KB), plus reproduction code and both papers |

### Build log for this pass (2026-09-14)

Applied and verified:

- **`xx_optimization`** - the cheat-sheet now leads with **AdamW** (explaining decoupled weight
  decay, and why plain Adam weakens the L2 penalty), the BatchNorm bullet gained the
  **LayerNorm** contrast plus a ch9 pointer, and the closing box reads "AdamW, then tune".
  Recompiled: 91 pages, 0 errors, 0 overfull vbox; frame rendered and checked, nothing clipped.
- **`xx_optimization_init_activations`** - **GELU** is no longer taught here, only named and
  pointed at ch9; the activation zoo now expands PReLU, ELU and SELU properly. Recompiled and
  frame-checked.
- **Acronym sweep** (`WORKFLOWS.md` step 2b) run on both. `xx_optimization` returns only
  `LMU`/`SGD`, both exempt. `xx_optimization_init_activations` returned `SELU` written as
  "its self-normalising variant", which names it without ever spelling it - **fixed** to
  "Scaled ELU (SELU)".
- **`MLP` deliberately left bare.** It is flagged by the sweep but is exempt-class (a method this
  chapter itself teaches), and the embedded LMU history pages already use the full phrase
  "multilayer perceptron". The style guide warns that expanding exempt terms is "noise, not
  rigor" and says not to fix a check that flags only exempt terms. Revisit only if the
  instructor disagrees.

Double descent - resolved 2026-09-14:

- **Double descent - first attempt FAILED to reproduce it. No slide written.**

  Setup: `py_src/double_descent.py --mode mlp`. Digits, 500 training samples, 15% label noise,
  a one-hidden-layer MLP trained with Adam for 600 epochs, 16 widths from 1 to 500. Sweep took
  126.3s.

  Measured test error by width: 72.8, 47.6, 36.7, 33.8, 32.9, 25.8, 20.7, 19.4, 15.4, 15.3,
  14.6, 14.4, **11.9**, 12.9, 12.1, 12.4 (%). Train error reaches 0.000 at width 12 and stays
  there. That is a **monotone decrease with a noisy tail - no peak at the interpolation
  threshold**. The phenomenon is simply not present in this configuration.

  Two reasons, both diagnosed rather than guessed: (a) the critical region is width 5-12 and the
  grid samples it with only three points, so a narrow spike could be stepped over entirely;
  (b) more fundamentally, 10-class cross-entropy trained with Adam does not sit in the
  just-barely-interpolating regime that produces the peak - the near-threshold models are the
  slowest to train and never reach the ragged fit that blows test error up.

  **Recorded because it is a disproven theory, not a bug.** Do not re-run this configuration
  expecting a different answer, and do not put `fig/double_descent_mlp.pdf` on a slide about
  double descent - the picture is honest and shows the opposite.

  **Second attempt SUCCEEDED:** `--mode rf` - frozen random ReLU features with a minimum-norm
  least-squares readout (Belkin et al.'s original setting), grid deliberately dense at
  460/490/500/510/540. Textbook double descent, in **14.8s** (vs 126.3s for the failed MLP):

  | random features | 5 | 120 | 300 | 460 | **500** | 540 | 700 | 1200 | 4000 |
  |---|---|---|---|---|---|---|---|---|---|
  | test error % | 76.3 | **11.6** | 20.1 | 58.8 | **89.8** | 63.5 | 33.2 | 19.0 | **13.7** |

  The peak sits **exactly at `n_features = n_train = 500`**, and at 89.8% it is worse than a
  5-feature model. Train error is 0.000 from 400 features on.

  **The honest caveat, which is on the slide:** the second descent does **not** beat the first
  minimum here (13.7% vs 11.6%). Capacity is not free, and the frame says so rather than
  implying "bigger is always better".

  Still legitimately a one-hidden-layer network for teaching: the hidden layer is frozen and the
  output layer solved exactly instead of descended to - which is also the "frozen random
  features" control already catalogued in `REVIEW.md` section 6.

  Shipped as one `\pause` predict-first frame ("The U-curve is not the whole story") in
  `43_nn_regularization`, placed straight after "Read your training curves", since it is the honest
  asterisk on "stop at the valley". Artifacts: `data/double_descent_rf.json`,
  `fig/double_descent_rf.pdf`. The failed MLP run is kept alongside as `*_mlp.*`.

  **The first version of this frame shipped clipped, and every automated check passed it.**
  On its first compile the deck reported 48 pages, exit 0, **0 errors and 0 overfull vbox** -
  and the callout box still lost its final clause off the bottom edge, stopping mid-sentence at
  "is not a law -- which is". It was caught only by rendering the page and looking at it. Fixed
  by cutting the opening sentence (which duplicated the previous frame anyway), tightening the
  right column, and shortening the footnote. Two lessons, both already in `WORKFLOWS.md` but
  worth re-proving: a 4:3 frame holding a figure, three bullets and a callout box is at its
  limit, and **`pdflatex` exit 0 is not evidence a slide is intact**.

  The figure was also regenerated at `figsize=(5.6, 3.6)` with 11-13pt fonts and no title: at
  0.56 column width in a 4:3 frame the original 7.2x4.2 canvas rendered its axis labels too
  small to read from the back of a room. Slide figures need to be sized for the column they
  land in, not for a full page.

---

## 8b. Expansion pass (2026-09-14): four topics given full frames

The instructor asked for more than one-liners on AdamW, LayerNorm, GELU and double descent, with
the originating papers and current usage. **Every citation below was web-verified before it went
on a slide**, per the SLIDE_STYLE rule - not taken from memory and not trusted from `REVIEW.md`.

| New frame | Deck / page | Citations (verified) | Usage today |
|---|---|---|---|
| From Adam to AdamW: the fix that stuck | `xx_optimization` p91 | Adam: Kingma \& Ba, arXiv 1412.6980, ICLR 2015. AdamW: Loshchilov \& Hutter, arXiv 1711.05101 (Nov 2017), ICLR 2019 | AdamW trains essentially every transformer; challenger **Muon** (Jordan et al., 2024) trained Kimi K2 and GLM-4.5 |
| BatchNorm's sibling: LayerNorm | `xx_optimization` p92 | BatchNorm: Ioffe \& Szegedy, ICML 2015. LayerNorm: Ba, Kiros \& Hinton, arXiv 1607.06450 (2016). RMSNorm: Zhang \& Sennrich, 2019 | LayerNorm in GPT-2 / GPT-J / Pythia; RMSNorm in Llama and Mistral |
| GELU: the activation transformers actually use | `xx_optimization_init_activations` p32 | GELU: Hendrycks \& Gimpel, arXiv 1606.08415 (2016). SwiGLU: Shazeer, arXiv 2002.05202 (Feb 2020) | GELU in BERT and GPT-2/3; SwiGLU in PaLM (2022) and Llama (2023), now the open-source default |
| Double descent in the wild | `43_nn_regularization` p23 | Belkin et al., PNAS 2019. Nakkiran et al., ICLR 2020 (arXiv 1912.02292) | model-wise, epoch-wise, and sample non-monotonicity ("more data can hurt") |

Figures, both Python-generated (`py_src/modern_components_figs.py`):

- `fig/gelu_vs_relu.pdf` - ReLU / GELU / SiLU and their derivatives, annotated at $z=-1$ with the
  exact values the slide quotes (GELU $-0.159$, slope $-0.083$; ReLU $0$ and $0$). The first
  version was rejected on review: at full scale the three curves are visually identical in
  precisely the negative region the frame is about.
- `fig/norm_axes.pdf` - what BatchNorm averages over (a column, down the batch) versus LayerNorm
  (a row, across one example's own features).

**GELU placement, revised.** Section 8 above recommended keeping GELU out of ch5 entirely. The
instructor asked for it here, so it now has a frame - but the frame still hands off to ch9, where
SwiGLU and the transformer block live, so the chapters complement rather than duplicate.

### Build reliability: "pdflatex twice" is not always enough, and the success signals lied

On `xx_optimization` the footer rendered `4/1` instead of `4/6` across several build cycles.
**Three hypotheses died before the real cause surfaced**, and all three died the same way:

1. *"The `.nav` needs a third pass to settle."* Wrong - a third pass changed nothing.
2. *"Extra passes will fix it."* Wrong - identical output.
3. *"The 3 errors in the log cause it."* Wrong - the counter was already wrong in builds
   reporting **0** errors.

The actual cause: **the output PDF was intermittently locked, so `pdflatex` exited 1 without
writing anything**, and every downstream check was reading a stale file. A per-pass trace settled
it in one run:

```
pass1: exit=1  pdf_rewritten=no   write_fail=1  footer=4/1
pass2: exit=1  pdf_rewritten=no   write_fail=1  footer=4/1
pass3: exit=0  pdf_rewritten=yes  write_fail=0  footer=4/6
```

Beamer reads `\inserttotalframenumber` from the `.nav` and **defaults it to 1** when there is
none. `clean_latex.py` deletes the `.nav`, so every build starts cold and needs *two successful*
passes - and a pass that fails to write does not count. Likely trigger for the lock: the
verification steps themselves (`pdftotext` / `pdfinfo` / `pdftoppm` on the PDF immediately before
the next pass) leaving a file handle open on Windows.

**What to do about it.** A footer reading `N/1` is a build failure, not a source bug - do not edit
the `.tex`. Assert the exit code of *every* pass and check that the PDF's mtime actually advanced.
`pdfinfo` page counts, `grep -c '^!'` and a final exit code will all cheerfully report success
from a PDF that was never rewritten; that is how three hypotheses got wasted here. No source
change was needed in the end.

## 9. Retirement checklist

1. Port gaps 1-6; decide on 7-8.
2. **`neural_networks.qmd` is student-facing and breaks.** It links both PDFs by name and its
   Armenian warning block reads "Դասերի սլայդերը պատրաստ են (L14, L15)". Must be rewritten to point
   at the seven decks. (Note: a repo-wide `grep` for `L14` skips this file, apparently on its
   non-ASCII content - confirmed by reading it directly. Do not trust grep alone here.)
3. **The flagship practical depends on a retired slide.** `xx_name_inventor_OUTLINE.md` cell 13
   says the L15 training loop is "quoted verbatim from the slide". Porting gap 3 is therefore a
   prerequisite, not a nice-to-have.
4. Rewrite `README.md` - it still presents L14/L15 as canonical and the two sets as deliberately
   coexisting.
5. Update `ml/00_plan.md:103` (`Neural networks | 2-3 | ... | L14, L15`).
6. **Repoint the downstream callbacks - this is the biggest single job, and it is bigger than it
   looks.** Retiring L14/L15 breaks live citations in *delivered slide text*, not just planning
   docs:
   - **`ch6/L16_cnn_foundations.tex` carries 13 in-slide references.** Its cold open is literally
     "In L14 our modest MNIST net (784-128-64-10) had **109,386** weights" (line 44) - the very
     frame that gap 1 above says is missing from the `dl_*` set. Porting gap 1 is therefore a
     **hard prerequisite for ch6**, not just a nicety. Also: "exactly the L14 neuron" (526),
     "Weight decay (L15)" (630), "Trees win on tabular data (ch4, L14)" (636), the frame titled
     "The training loop, in 60 seconds (recall L15)" (840-841), "L14's linearity lesson" (885),
     "L14's XOR lesson at scale" (897), a code comment "# training loop is the same as L15" (917),
     "the L15 promise" (922), "L14's why-GPUs frame" (930), "The L15 MLP baseline" (941).
   - **`ch9/L25_transformer_block.tex`**: "LayerNorm: BatchNorm's idea from **L15**".
   - **`ch6/CNN_CHAPTER_PLAN.md`**: an entire callback-anchor table keyed to L14/L15, plus
     "BatchNorm home resolved: L15 now teaches the BN formula, L17 only calls back". Its own words:
     "the outlines lean on these hard".
   - `ch6/L16_cnn_foundations_OUTLINE.md` and the stale `ch5/CNN_BLOCK_DESIGN.md`.

   Decide the new citation scheme **before** editing, or this gets done twice. Note also that
   `ch5/CNN_BLOCK_DESIGN.md` (2026-06-16) is superseded: it plans "L16a/b/c" inside ch5 and says
   "no authoring has started", while `ml/ch6_cnn/` now holds built L16-L19.
7. Update `.claude/skills/youtube-reference/SKILL.md:58`, which already calls them "legacy".
8. Register the seven decks in `_quarto.yml` (exact case) and rename per `CONVENTIONS.md`.
9. Add a `DECISIONS.md` entry (next number is **#36**) recording the retirement and what would
   reverse it.
10. **Move, do not delete.** Per the repo's own tiering rule, L14/L15 belong in an `archive/` with
    a README line, not in `rm`. They are the only 16:9 house-style treatment and remain the source
    for every ported frame.
11. Leave `REVIEW.md` and `NN_CHAPTER_OUTLINE.md` as historical records; do not retrofit them.

## 10. Both detectors misbehave on the embed set - read this before trusting them

Neither detector was designed for decks that are 80% `includepdf`, and both mislead here.

**`detect_clipped_slides.py` - two false positives on `40_intro_history`** (frames 9 and 19).
Both were rendered and checked visually and are **clean**: the detector trips on `warnred`
(missing from its `JUNK` list) and on the accented `Th\'e\^atre D'op\'era`. Adding `warnred` to
`JUNK` would remove one of them. On `43_nn_regularization` it reports 7 frames, 0 flagged - correct.

**`detect_footer_collisions.py` - 37 of 48 pages flagged on `43_nn_regularization`, all noise.**
The flagged pages are exactly `3-12, 15-19, 24-45`. Every one sits inside an embedded LMU range
(`2-12`, `14-19`, `23-45`); **not one of our own frames** (1, 13, 20, 21, 22, 46, 47, 48) is
flagged. The cause is structural, not a defect: LMU's slides carry their own
"Deep Learning - N / M" footer in the same band our template uses for the page number, so every
embedded page reads as ink-in-the-footer forever. The repeated identical "ink 4.55%" across
consecutive pages is the tell.

**Consequence for the workflow.** On the `dl_*` set the footer detector has a ~77% false-positive
rate, which is worse than useless - it trains you to ignore the one check that catches silently
clipped callout boxes. Until it learns to skip `includepdf` pages, run it and then **filter to
our own frame numbers only**; a flag outside those is LMU's footer, not ours. The `dl_*`
provenance blocks list which pages are ours.

And the reason this matters, proven on this very deck: the double-descent frame shipped **clipped**
past `pdflatex` exit 0, 0 errors and 0 overfull vbox. The only thing that caught it was rendering
the page to PNG and reading it. Do that for every frame you add.
