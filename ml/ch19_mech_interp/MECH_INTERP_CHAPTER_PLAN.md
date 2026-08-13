# Chapter 19 - Mechanistic interpretability

**Status:** **built, 2026-08-13.** All three decks compile clean and are verified; the chapter
page is written and registered in `_quarto.yml`.

| Deck | Frames / pages | Checks |
|---|---|---|
| `L45_opening_the_box` | 48 / 50 | 0 errors, 0 overfull, 0 clipped, **0 footer collisions**, acronyms pass |
| `L46_does_it_actually_do_that` | 48 / 49 | same |
| `L47_features` | 49 / 49 | same |

**A pedagogy pass was run on all three (2026-08-13) and changed them substantially.** Findings
and fixes, in case the same mistakes recur in a future chapter:

- **Figures and their interpretation were on separate slides** in five places. In a lecture the
  student cannot flip back, so the reading arrived while the figure was gone. All five merged
  (side-by-side layout, or the reading annotated onto the figure frame). This was diagnosed and
  fixed in L45 first, then **reintroduced** in L46 and L47, which were written afterwards - the
  lesson did not carry forward on its own.
- **The linearity chain in L45** was one slide asserting five steps; it is now six frames ending
  in a measured reconstruction that is exact to five decimal places.
- **L47 section 3 discussed attribution graphs without showing one.** It now opens with a
  schematic, and a new frame states what kind of evidence the findings rest on (intervention, not
  observation) - the chapter closes by demanding exactly that, so it must not fail its own test
  one section earlier.
- **Predict-first frames and worked numbers thinned out toward the end.** L47 had two and none;
  it now has a rhyme-planning predict-first and a worked dictionary-width calculation.
- **13 frames had content colliding with the page-number footer**, none of which
  `detect_clipped_slides.py` could see. Drove `non_essential/detect_footer_collisions.py`, now
  required by `WORKFLOWS.md`. See
  `_learnings/2026-08-13-1930_a-passing-check-is-only-as-good-as-its-coverage.md`.

**Still to build:** `HW_mech_interp.ipynb` (spec below), the cross-links to
`ml/05_interpretability` and `ml/ch18_agents/L44` (open questions 5 and 6), and the `🎲 Random`
section of the chapter page.

Every number quoted on every slide is measured in this repo on GPT-2 small, on CPU, from
`data/ioi_dataset.json`. Raw results are in `results/*.json`; figures read those files and never
re-run the experiment. Two claims in the original outline were **corrected by the measurements**
rather than confirmed - see the build-risk gate and the self-repair section.

Three decks, `L45`-`L47`, continuing `ml/` numbering after `L44` (agents). Chapter page
`mech_interp.qmd` registers in `_quarto.yml`. Three decks matches `ch10_diffusion` (L29-L31)
and `ch17_rag` (L41-L43).

Closes no existing entry in `ml/MISSING_TOPICS.md` - this is a new topic, not a known gap. It
was proposed 2026-08-13 after reading Neel Nanda's current guide (see Sources).

## Interview-locked decisions (2026-08-13)

1. **Scope: three decks.** Read the model -> prove the mechanism -> use it. Not two (the
   causal-methods deck is the heart and must not be compressed), not four (no extra session
   for a lab; the practical lives in homework instead).
2. **Stance: science of the model.** The chapter's spine is *what counts as evidence that a
   component does a thing*. Not a tool tutorial, not an AI-safety lecture. Safety applications
   appear in L47 as the answer to "what is this for", after the mechanics are earned.
3. **The grokking deck stays separate.** `misc/grokking/slides/grokking_mechanistic_interpretability.tex`
   (51 frames, compiled, unregistered) is **not** folded in. It gets a callback frame in L46 and a
   link on the chapter page as optional viewing. Boundaries stay clean.
4. **Placement: immediately after the attention/LLM block**, ~mid-October 2026, after `ch9_attention`
   L24-L26 and the `llm_training` seminar track. Transformer internals must be fresh; this material
   is very hard to teach cold. Costs 3 sessions and pushes diffusion onward by ~1 week.
5. **Numbering: `L45`-`L47`, build order** (locked 2026-08-13). The L-number is a build-order id,
   not a delivery-order one - that correspondence is already broken elsewhere (L37/L38 are swapped
   relative to the schedule, and L41-L44 do not appear in the schedule table at all). Rejected
   `L26b/c/d`, which would have preserved delivery order for this chapter only while making the
   inconsistency worse everywhere else. See `DECISIONS.md` #13.

---

## Teaching stance - this governs every decision below

- **Black box to white box.** Chapter 05 (decks 22-24) taught interpretability that never looks
  at a weight: SHAP, LIME, PFI, PDP all treat the model as a function you can only query. This
  chapter opens it. That contrast is the cold open and gets called back three times.
- **The unifying question is causal, not descriptive.** Every technique in the chapter is an
  answer to *"you say this head does X - prove it."* Descriptive techniques (logit lens, probes,
  max-activating examples) are introduced in L45 **and then immediately shown to be insufficient**,
  which is what motivates L46.
- **Every claim gets its counter-example.** This field's own literature is full of retracted
  conclusions (subspace-patching illusions, the BERT interpretability illusion, ROME follow-ups,
  self-repair breaking naive ablation). Teaching the failures is the point, not a caveat.
- **Argue against the field, not just against its techniques.** Added after review, 2026-08-13.
  Draft v1 applied the rule above to every *method* and to the *discipline* not once - it argued
  for mech interp everywhere and against it nowhere. That is the predictable result of building a
  chapter from a field's own introductory literature: Nanda's guide is written to **recruit
  researchers**, not to give a balanced picture to students who will mostly go and do something
  else. Three specific corrections, each now load-bearing in the outline:
  - the chapter says out loud what black-box methods can do that this cannot (L45 cold open);
  - it defends its own choice of a small, old model instead of hoping nobody notices (L45 cold open);
  - it does not let "what this is for" be a tour of frontier-lab work with nothing a student in
    this room could act on (L47 section 6).
- **Cite by name and year, sparingly.** House convention. The chapter has an unusual number of
  canonical papers; resist turning it into a literature review. `papers/` is instructor-side
  grounding so the numbers are correct, not a student reading list. Exception: L47's final
  section *is* a reading list, deliberately - it is the "how do I get into this" frame.
- **Code stays minimal.** One canonical TransformerLens snippet per deck at most. The hooks API
  is genuinely elegant and worth one frame; it is not worth six.
- **No unlabelled numbers.** If a patching heatmap appears, the slide says what the colour means
  in plain words.

---

## Running example: Indirect Object Identification (IOI)

> *"When Mary and John went to the store, John gave a drink to ___"*

GPT-2-small answers `" Mary"`. The task is trivial for a human, the model is small enough to
fully instrument, and **it is the one task in the field with a completely documented circuit**
(Wang et al., 2022). Every technique in the chapter lands on this same sentence:

| Deck | What IOI is used for |
|---|---|
| L45 | logit lens on it, probe for "which name is duplicated", read the attention patterns |
| L46 | ablate it, patch it, DLA it, then assemble the full circuit; then break it with self-repair |
| L47 | decompose the same activations with an SAE, then look at it as an attribution graph |

**Honest limitation to state on a slide, not hide:** GPT-2-small is English-only, so there is no
Armenian version of the running example. Armenian appears in the `.qmd` page prose and section
headers as usual, not in the model inputs. Do not fake a bilingual demo here.

**Second example, one per deck, to prove the technique is not IOI-specific:** the
`9.8 < 9.11` failure (Meng et al. / Transluce) in L46, and multilingual features
(Lindsey et al., 2025) in L47.

---

## Course-callback spine - what makes this OUR chapter

| New concept | Callback to | Framing |
|---|---|---|
| the residual stream as the object of study | `ch9/L25` frame *"The residual stream is the real object"* | "we already told you this. Now we use it." - the single most important handoff in the chapter |
| reading/writing to the stream, linearity | `ch5` skip connections, `ch9/L25` block | every component reads the stream and adds to it; nothing else communicates |
| logit lens | `ch9/L26` unembedding, `03_classification` softmax | "apply the unembedding early and see what it would have said" |
| linear probes | `05_interpretability/22` glass boxes; `03_classification/11` logreg | a probe *is* logistic regression, trained on activations instead of features |
| attention head analysis | `ch9/L24` *"A real head, our sentence"*, `ch9/L25` *"Do heads actually specialise?"* | L25 asked whether specialisation is a story. This chapter answers it causally |
| ablation | `05_interpretability/23` PFI | "PFI permutes an input feature. This zeroes an internal component. Same logic, one layer deeper" |
| activation patching | `05_interpretability/24` counterfactuals | the clean/corrupt pair *is* a counterfactual, run inside the network |
| attribution patching (gradient approx.) | `ch5` backprop | one backward pass approximates thousands of forward-pass ablations |
| superposition / polysemanticity | `ch8/L22` section *"Sparse autoencoders and interpretability"* | **already taught** - recap in 3 frames, then go further into the geometry |
| SAEs | `ch8/L22` + `HW1_sae_rnn.ipynb` | **already taught and already implemented by the students.** Here they *use* one, on a real LLM, not train one |
| dictionary / decoder columns, L0 | `ch8/L22`, `ch1` Lasso L1 | no re-teaching; signpost and move on |
| steering vectors | `ch10_diffusion` classifier-free guidance | "push the activation in a direction" is the same move, different object |
| attribution graphs | `ch8/L22` closing frame (already forward-points to them) | the promise L22 made, now delivered |
| chain-of-thought faithfulness | `ml/llm_training` (GRPO, DeepSeek-R1, reasoning models) | that track teaches how a model is *trained* to reason out loud. This one asks whether the text it prints is what it actually did |
| auditing / monitoring | `ch18_agents/L44` | an agent you cannot inspect is an agent you cannot trust |

**Rule: nothing in the ch8 SAE section gets re-taught.** L47 opens by explicitly saying "you
built one of these in the autoencoder chapter" and proceeds from there. If a frame in L47
duplicates an L22 frame, cut the L47 one.

---

## L45 - Opening the box: reading a model's internals

~50 frames. **The descriptive deck.** Ends by admitting everything in it is circumstantial.

> **Frame budget, corrected 2026-08-13.** v1 of this plan estimated 38/40/36 frames. Measured
> against the decks this chapter is shaped after - `ch17_rag` at **54/58/55** and `ch16_jepa` at
> **47/49** - those numbers were low, and the beats outlined below will land nearer **50 per
> deck**. Size the three sessions against 50-frame decks, not 38-frame ones.

### Cold open

1. Twenty lectures of black boxes. Show the `05_interpretability` SHAP plot again: it tells you
   *which input mattered*. It cannot tell you *what the model did*.
2. Ask GPT-2-small the IOI sentence, live and real (captured run, not a mock-up). It says `" Mary"`.
3. **Predict-first frame:** the model has 124M numbers and we can read every one of them. Is
   knowing all the weights the same as understanding the model? (No - and why not is the field.)
4. One-line definition, Nanda's: *using a model's internals - weights and activations - for
   understanding.* Note the pragmatism: not "complete reverse-engineering".
5. **What this does not replace.** Two columns, stated evenly, no winner declared:

   | | Black box (ch5) | White box (this chapter) |
   |---|---|---|
   | Works on | **any** model - gradient boosting, an SVM, an API you only query | neural networks whose **weights you have** |
   | Needs | predictions | weights, activations, and a framework that hooks them |
   | Answers | which **input** moved the output | what the **model did** |
   | Cost | seconds | the IOI circuit took a research team months |

   Say the uncomfortable half plainly: **most of you will deploy gradient boosting**, and for that
   model SHAP is not a weaker tool, it is the only tool. This chapter is not an upgrade over
   chapter 5; it is a different instrument for a different object.
6. **Why GPT-2-small, in 2026.** Pre-empt the obvious objection rather than let a student find it
   in lecture three. The model is 124M parameters and dates from 2019. We use it because every
   interpretability tool targets it, because IOI is the only task in existence with a fully
   documented circuit, and because it runs on a laptop CPU - so every figure in this chapter is
   one you can reproduce. Then split the honesty:
   - **transfers:** the residual-stream picture, the whole causal methodology, superposition,
     the evidence standards - all of it is architecture-level, not model-level;
   - **does not transfer:** the specific circuits. GPT-2's name-mover heads are facts about
     GPT-2. Nothing here says a 400B model solves IOI the same way.
   - Forward-flag the tension the chapter will hit in L46: the field has *retired* toy-model
     interpretability as a research direction while still teaching on toy models. That is not
     hypocrisy, it is the difference between a training set and a research frontier - but the
     students should hear it from us, in lecture one, not spot it in lecture two.

### Section 1 - The residual stream is the object

- Callback frame to `ch9/L25`, restated as an interpretability claim: the stream is a **shared
  bus**. Every attention head and MLP *reads* from it and *adds* to it. No component talks to any
  other directly.
- **The consequence that makes the whole field possible:** because contributions are added, the
  final logit is a **sum of per-component contributions**. You can decompose it.
- Weights vs activations: what is fixed and what depends on the input.
- Terminology students will meet: layer, head, position, the `[batch, pos, d_model]` shape.
  One frame, honest about the fact that most of this field is careful indexing.
- **One code frame:** TransformerLens `model.run_with_cache(prompt)` and what the cache holds.
  The only API frame in the deck.

### Section 2 - The logit lens

- Idea in one line: apply the **unembedding** to an intermediate residual stream and see what
  the model *would have said* if it stopped there (nostalgebraist, 2020).
- Figure: layer-by-layer top-5 predictions on the IOI sentence. `" Mary"` climbs.
- What it shows: prediction is built up gradually, not computed at the end.
- **Misconception pre-empt:** the logit lens is not a decoder of "what the model is thinking".
  It works reasonably on GPT-2 and poorly on other models; representations drift between layers.
  Tuned-lens exists precisely because of this.

### Section 3 - Probes

- A probe is a classifier trained on activations. **It is the logistic regression from deck 11**,
  with the residual stream as the design matrix.
- Worked setup: train a probe at each layer to answer "is the subject name duplicated?" Figure:
  accuracy by layer, with the chance-level baseline drawn.
- **The core caveat, given a full frame:** a probe finding information proves the information is
  *present*, not that the model *uses* it. Analogy: you can read the ingredients off a label
  without knowing which ones the recipe actually calls for.
- Othello-GPT as the payoff story (Li et al. + Nanda's follow-up): probes revealed a genuine
  internal board state - a *world model* - in a model trained only on move sequences.
- Where probes are used for real, today: content monitoring in deployed systems.

### Section 4 - Reading attention heads

- What a single head does, restated as an interpretability object: it moves information from one
  position to another. Callback to `ch9/L24`.
- Figure: attention pattern heatmaps for several GPT-2-small heads on the IOI sentence, with
  `circuitsvis`-style rendering regenerated in matplotlib.
- **Induction heads** (Olsson et al., 2022) get their own subsection - the canonical named circuit:
  - the task: `... A B ... A -> B`. Repeat a random token sequence and the model gets good at it.
  - the two-head mechanism: previous-token head writes into the stream, induction head reads it.
  - why it matters: this is the mechanism behind a large part of in-context learning, and it
    forms in a visible phase change during training.
- **Predict-first frame:** does a head that always attends to the previous token *mean* anything?
  (Not yet. Attention patterns are suggestive, never conclusive - a head can attend somewhere and
  write nothing useful.)

### Section 5 - Naming what you find

- Max-activating dataset examples: the standard way to name a neuron or feature.
- Polysemanticity teaser: most GPT-2 neurons fire on several unrelated things. One frame; the
  full treatment is L47, and the students already met it in `ch8/L22`.
- **The interpretability illusion** (Bolukbasi et al.): top-activating examples on one dataset
  gave a clean story for a BERT neuron that a different dataset contradicted. Cherry-picking is
  the default failure, not an occasional one.

### Recap + Next

- Everything in this deck is **descriptive**. A logit lens trace, a probe accuracy, an attention
  pattern, a top-activating list - all of it is correlational.
- `paramgreen` Next box: *"You have four ways to form a hypothesis and no way to test one.
  Next lecture: intervention."*

---

## L46 - Does it actually do that? Causal methods

~50-55 frames. **The methodological heart of the chapter.** If frames have to be cut anywhere, do
not cut them here.

> **The one deck at real risk of not fitting 17:30-19:00.** Six sections, the chapter's densest
> material, and the highest realistic frame count of the three. Two mitigations, decide at build:
> move section 4 (path patching, attribution patching, ACDC - the *scaling* material, not the
> *method* material) into L47's opening as the bridge to attribution graphs, where it fits the
> argument nearly as well; or accept a hard stop after section 5 and open the next session on
> section 6. **Do not solve it by compressing sections 2 and 5** - patching and the assembled
> IOI circuit are what the chapter is for.

### Cold open

- Restate the L45 cliffhanger with a concrete trap. **Gate experiment run 2026-08-13
  (`py_src/decoy_head_hunt.py`, `results/decoy_head_hunt.json`) - the head exists, and it is
  better than hoped: `L9H8`.**

  | Head | Attention to the answer name | DLA | Effect of ablating it |
  |---|---|---|---|
  | `L9H9` | 0.749 | **+2.805** | +0.471 |
  | `L9H6` | 0.664 | **+1.053** | -0.522 |
  | **`L9H8`** | **0.318** | **-0.001** | **-0.034** |

  All three sit in **the same layer**, all three look at the correct name, and one of them does
  nothing at all - not a small contribution, `-0.001`. Show the three attention patterns side by
  side and ask the room to pick the one that does not matter. **Nobody can**, because the picture
  does not contain the answer. That is the deck in one frame.
- Deliberately do *not* resolve it in the cold open. The resolution is section 2's patching
  heatmap, ~20 frames later, and the frame should say so.
- Callback to `ch8/L22` frame *"Naming a feature is a guess. Ablating it is a test."* - the
  students have already seen this move once, on an SAE feature. Now it becomes a methodology.

### Section 1 - Ablation

- Delete a component, measure the damage. Three flavours, and the difference matters:
  **zero-ablation**, **mean-ablation**, **resample-ablation**.
- **Why zero-ablation lies:** zero is not a neutral value, it is an off-distribution input. The
  model has never seen that activation and its response tells you about the corruption, not the
  component. Mean/resample keep you on-distribution.
- Figure: IOI logit difference under each ablation type, per head, three panels side by side, so
  the disagreement between methods is visible rather than asserted.

### Section 2 - Activation patching (causal tracing)

- The core move, one full-bleed diagram: run a **clean** prompt and a **corrupted** prompt, then
  copy one activation from one run into the other and see how much of the behaviour transfers.
- Corruption for IOI: swap the names. `"When Mary and John... John gave a drink to"` vs
  `"When Mary and John... Mary gave a drink to"`.
- **Denoising vs noising** given a frame each - they answer different questions and get confused
  constantly (Heimersheim & Nanda's expository note is the source).
- The metric: **logit difference**, and why it beats accuracy or loss for this purpose.
- **Worked-numbers frame:** patch one head, by hand, on the running example. Clean logit diff,
  corrupted logit diff, patched logit diff, and the recovered fraction.
- Figure: the canonical patching heatmap - layer x position, colour = fraction of performance
  recovered. This is the single most recognisable image in the field and the deck's centrepiece.

### Section 3 - Direct logit attribution

- Because the stream is a sum, project each component's output directly onto the answer
  direction. No intervention needed.
- Figure: per-head DLA on IOI, sorted. The name-mover heads separate cleanly from the pack.
- **Limitation frame:** DLA only sees *direct* paths to the logits. A head whose whole job is to
  set up a later head is invisible to it. That gap is what path patching exists for.

### Section 4 - Scaling the search

- **Path patching:** patch along a specific path, not a component - lets you ask "does head A
  affect the output *through* head B?"
- **Attribution patching** (Nanda, 2023): a first-order gradient approximation of patching. One
  backward pass estimates every patch at once. Callback to backprop in `ch5`.
- One frame on the cost arithmetic: exhaustive patching in GPT-2-small is cheap; in a 70B model
  it is not, and this is why the approximation exists.
- Automated circuit discovery (ACDC, Conmy et al.) - one frame, named and characterised, not derived.

### Section 5 - The IOI circuit, end to end

The chapter's payoff. Four or five frames assembling everything:

- The task, the metric, the model. (All already familiar - that is the point of the running example.)
- The head classes and what each does: previous-token, duplicate-token, induction, S-inhibition,
  name-mover, negative name-mover, backup name-mover.
- The algorithm in plain words: *find the duplicated name, inhibit attention to it, copy the
  other one.*
- Full-bleed circuit diagram.
- **The honest frame:** this took a team of researchers months, on a 124M-parameter model, for one
  task. State the cost out loud. It is the reason the field moved to the methods in L47.
- *Verify at build: exact head count and class count against Wang et al. (2022) before baking numbers in.*

### Section 6 - What goes wrong

- **Self-repair / the Hydra effect** (McGrath et al., 2023). **Gate experiment run 2026-08-13
  (`py_src/self_repair.py`, `results/self_repair.json`). It reproduces, and the measured result is
  stronger than the "performance barely drops" version this plan originally assumed:**

  ```
  baseline logit diff : +3.580
  ablated  logit diff : +3.754     <- the 3 name movers are mean-ablated here
  DLA predicted drop  : +4.438
  actual drop         : -0.175     <- the model got BETTER
  ```

  Ablate the three heads that provably do the task and performance **goes up**. Build the frame
  as a predict-first: show the DLA bar chart, ask the room how far the logit difference falls when
  the top three heads are deleted, then show that it does not fall at all.
- **The mechanism, and it is not one story but two.** This is the nuance the deck must get right -
  "backups take over" is only half of the measured `+4.612` of compensation:

  | Source | Contribution | What happened |
  |---|---|---|
  | `L10H7` stops suppressing | **+2.064** (45%) | a *negative* name mover, `-2.049` -> `+0.015` |
  | genuine backups ramp up | **+1.851** (40%) | `L10H10 +0.730`, `L10H2 +0.588`, `L10H6 +0.334`, `L10H1 +0.199` |

  The first row is the more interesting one and it is self-consistent: `L10H7` is a copy-suppression
  head whose job is to suppress whatever the name movers boost. Remove the name movers and there is
  nothing left to suppress, so it goes quiet. The network did not "repair" anything - **a brake
  came off at the same time as the engine.** Teach both halves; a deck that says only "backups
  took over" is teaching a measured result inaccurately.
- **The consequence, stated bluntly:** ablation *understates* importance whenever a backup or a
  suppressor exists. A component can be doing the work and measure as unimportant. This is not an
  edge case - it happened on the first task we tried, in the most-studied circuit in the field.
- Interpretability illusions in subspace patching (Makelov et al.) - one frame.
- **Causal scrubbing** as the field's attempt at a rigorous standard for "is this explanation
  actually right", and why nobody finds it fully satisfying.
- Callback frame to the grokking deck: modular addition is the one setting where a circuit was
  reverse-engineered *completely*, with the Fourier structure and the excluded-loss progress
  measure as evidence. **Then say the uncomfortable part:** Nanda, who wrote that paper, now lists
  toy-model interpretability as a direction to avoid, because it is divorced from real models.
  Good teaching example, retired research direction. This is the payoff of the L45 cold-open
  frame that flagged the tension in advance.
  - **Registration blocker (verified 2026-08-13):** `misc/grokking/` is **not** in `_quarto.yml`
    and publishes no PDF, so "link it as optional viewing" - as v1 of this plan assumed - would
    link to nothing. Pick one before the chapter page is written: **(a)** register
    `misc/grokking/` in `_quarto.yml` so the compiled PDF publishes, or **(b)** point the frame
    and the `.qmd` at the Welch Labs video (`misc/grokking/README.md` has the URL and the
    timestamped beat list), which is the deck's own source and needs no build work. Recommend
    (b) - it costs nothing, and the video is better than the deck for a student watching alone.

### Recap + Next

- You can now test a hypothesis about a component. But every technique so far operates on
  **neurons, heads and layers** - the units the architecture happens to give us.
- `paramgreen` Next box: *"What if those are the wrong units?"*

---

## L47 - Features, and what interpretability is for

~50 frames. **The modern deck.** Half of section 1 is recap of `ch8/L22` - keep it to three frames.
Seven sections after the review additions; section 6 absorbed the cut that paid for section 5.

### Cold open

- The L46 cliffhanger made concrete: a GPT-2 neuron's top activations, showing three unrelated
  concepts. The neuron is not the unit.

### Section 1 - Superposition, properly (recap + extension)

- **Three recap frames only**, signposted as recap: features as directions, more features than
  dimensions, polysemanticity as the cost. Students met all of this in `ch8/L22`.
- Then extend past what L22 covered:
  - **why superposition is forced, not chosen:** sparse features + limited dimensions, and the
    model trades interference for capacity (Elhage et al., 2022, *Toy Models of Superposition*).
  - the high-dimensional geometry: near-orthogonal directions are exponentially plentiful.
  - **the linear representation hypothesis** stated explicitly as a hypothesis - a load-bearing
    assumption of the field, not a theorem, with the honest note that it is contested.

### Section 2 - SAEs, as a user

- **Open by naming the callback:** *"you built one of these in `HW1_sae_rnn`."* Do not re-derive
  the architecture or the L1 objective.
- What changes at LLM scale: the dictionary is enormous, the model is frozen, and you did not
  train it - you download it.
- **Gemma Scope 2** (DeepMind, Dec 2025): SAEs and transcoders on every layer of every Gemma 3
  model, 270M to 27B, both pretrained and instruction-tuned. ~110 PB of stored activations,
  >1T parameters trained.
- **Neuronpedia**: the feature browser. Live demo frame - pick a feature, read its top
  activations, see its logit effects. Free, browser-only, no GPU.
- **The honest limits frame** (extends the L22 caveat rather than repeating it): feature splitting
  means the "true" feature count is undefined; dead features waste dictionary capacity;
  *Sparse Autoencoders Do Not Find Canonical Units of Analysis* (ICLR 2025). Add Nanda's current
  position: **use SAEs as a tool inside a project, not as the project.**

### Section 3 - Transcoders and attribution graphs

- A transcoder replaces an MLP with a sparse, interpretable approximation - which makes the
  *computation* readable, not just the representation.
- **Attribution graphs** (Ameisen, Lindsey, Pearce et al., 2025, *Circuit Tracing*): trace how
  features cause one another, automatically. This is the answer to L46's "months of researcher
  time for one circuit".
- Findings from *On the Biology of a Large Language Model* (Lindsey et al., 2025), 2-3 frames,
  chosen for how surprising they are:
  - the model **plans the rhyme ahead** before writing the line that leads to it;
  - **multilingual features** - shared concepts firing across languages, with a natural Armenian
    aside;
  - a case where the model's **stated reasoning does not match its computation**.
- **`circuit-tracer`** (Anthropic, open-sourced May 2025): runs on Gemma-2-2B, Llama-3.2-1B,
  Qwen3-4B, with the graphs explorable in the browser via Neuronpedia. Live demo frame.
- Model-transition card per house convention: year, authors/org, repo.

### Section 4 - Steering

- If a concept is a direction, add the direction. Activation addition / representation engineering.
- **Golden Gate Claude** (Anthropic, 2024) - already mentioned in `ch8/L22`, so treat it as a
  known reference and spend the frame on *what it demonstrates*: clamping one feature changed
  identity-level behaviour, which is strong evidence the feature was causal.
- The **refusal direction** result: refusal behaviour mediated by a single direction, and both
  what that enables (safety) and what it enables (jailbreaks). One frame, both sides.
- Callback to classifier-free guidance in `ch10_diffusion` - same move, different object.
- **Watch-out frame** (`armorange`): steering is not editing. The effect is often brittle,
  off-target, and degrades capability.

### Section 5 - Does the model mean what it says?

Added after review, 2026-08-13. **The most student-relevant section in the chapter**, and in v1
it was a single bullet. Every student in the room uses a model that shows its reasoning; this is
the one place where the chapter's machinery points at something they look at daily. Nanda names
reasoning-model interpretability as a critical open gap.

- **The setup, no jargon:** a reasoning model prints its chain of thought. The natural assumption
  is that the text is a transcript of the computation. **Predict-first frame:** is it?
- **Faithfulness is the technical word for that assumption**, and it is testable. The test shape:
  change something that provably affects the answer, and check whether the stated reasoning
  mentions it. Where the stated reasoning stays silent, the text is a *story about* the
  computation, not a record of it.
- Evidence from the running material: the *Biology of a Large Language Model* case where the
  model's stated reasoning does not match the traced computation (Lindsey et al., 2025). Because
  the students met attribution graphs two sections ago, this lands as a result rather than an
  anecdote.
- **Why this is more than a curiosity:** chain-of-thought monitoring is only a safety mechanism to
  the degree the chain is faithful. If a model can reach an answer by a route its own transcript
  never mentions, reading the transcript is not oversight.
- **What it means for the student, concretely:** the reasoning trace is evidence about the answer,
  not a proof of it. Do not treat "the model explained its steps" as verification - it is the
  same category error as reading a probe's accuracy as proof the model uses that information
  (callback to L45 section 3, the chapter's oldest warning, now at full scale).
- One frame on how open this is: we can detect *some* unfaithfulness; we cannot certify
  faithfulness. Say the limit rather than implying the tools are further along than they are.

### Section 6 - What this is actually for

The deck's argument for why the chapter exists. Rewritten after review, 2026-08-13: v1 was
**entirely frontier-lab work** - auditing games, model organisms, lab-internal monitoring, a
magazine citation - with nothing a student in this room could act on. Split it honestly into
what they can use and what they are watching from outside.

**What you can actually do with this (lead with it, roughly two thirds of the section):**

- **Probes as a production tool.** The most transferable thing in the chapter. A probe is a
  logistic regression on activations (L45 section 3) - cheap to train, cheap to run, and already
  deployed for real content monitoring. If a student takes one technique to a job, this is it.
- **Debugging a model you actually ship.** The `9.8 < 9.11` case, traced end to end - a real,
  stupid, reproducible failure explained by looking inside. Generalise the move: when a model
  fails weirdly and the input looks fine, the answer is in the activations.
- **Reading a model before trusting it.** Open-weight models are what most of these students will
  deploy, and Neuronpedia plus `circuit-tracer` work on exactly those (Gemma, Llama, Qwen) with
  no training and no GPU. Frame it as due diligence, not research.
- **Evaluating interpretability claims made at you.** Vendors and papers will assert that a model
  "understands" or "reasons about" something. After L46, students can ask the one question that
  separates evidence from a story: *did anyone intervene, or did they just look?*

**What the frontier labs are doing with it (compress to ~3 frames, clearly labelled as such):**

- **Auditing games** (Marks et al.): hide a goal in a model, see if a red team finds it with
  interpretability tools. Worth teaching because it is a real benchmark for whether any of this
  works - most of the field has no ground truth, and this manufactures some.
- **Model organisms**: install a known behaviour so there *is* ground truth to test methods
  against. Same motivation, different mechanism.
- **Chain-of-thought monitoring at scale**, tied back to section 5 rather than restated.
- MIT Technology Review named mechanistic interpretability a 2026 breakthrough technology - one
  line, as context for why the field suddenly has money and attention. *(Verify the exact framing
  at build.)* **Do not use it as an argument that the field works.**

**Closing honesty frame for the section:** most of the headline results in this chapter come from
three or four well-funded labs with model access nobody in this room has. That is a real
limitation on the field, not a reason the techniques are useless - probes and patching work fine
on a 124M model on a laptop, as every figure in this chapter demonstrates.

### Section 7 - The honest state of the field, and how to start

- What is genuinely unsolved: *Open Problems in Mechanistic Interpretability* (Sharkey et al.,
  2025, arXiv 2501.16496). Two or three of its open problems, stated plainly.
- **The frame that should stay with them:** we can now say useful things about what is inside a
  language model. We cannot yet fully explain one. Both halves are true.
- **How to actually get into this** - the one deliberate reading-list frame:
  - ARENA chapter 1 (1.2 intro/TransformerLens, 1.4.1 IOI, 1.3.3 SAEs) - the exercises this
    chapter is shaped after;
  - Nanda's *How To Become A Mechanistic Interpretability Researcher*;
  - TransformerLens (<=9B models) vs nnsight (larger, HF-native);
  - Neuronpedia - the zero-setup entry point;
  - MATS / SPAR / MARS as routes in.
- `paramgreen` Next box pointing to whatever follows in the schedule.

---

## Figures

House rule: essential figures are Python-generated. `py_src/*.py` -> `fig/*.pdf`, run with the
`ma` venv. **All of these run on CPU with GPT-2-small (124M).** No GPU, no Colab.

| Script | Produces | Deck | Cost | Note |
|---|---|---|---|---|
| `make_ioi_dataset.py` | ~1k IOI-style prompts + labels, saved to `data/` | - | minutes | **prerequisite**, see below |
| `logit_lens_ioi.py` | layer-by-layer top-5 predictions on the IOI prompt | L45 | 1 forward pass | |
| `probe_by_layer.py` | probe accuracy vs layer, with chance baseline | L45 | ~1k prompts, minutes | needs the dataset |
| `attention_patterns.py` | head attention heatmaps on the IOI sentence | L45 | 1 forward pass | |
| `induction_head_scan.py` | induction score per head on repeated random tokens | L45 | ~50 forward passes | |
| `ablation_compare.py` | zero / mean / resample ablation, three panels | L46 | ~150 forward passes | |
| `patching_heatmap.py` | the canonical layer x position patching grid | L46 | ~180 forward passes | |
| `dla_per_head.py` | direct logit attribution, sorted by head | L46 | 1 forward pass + projection | |
| `self_repair.py` | performance under name-mover ablation, backups on vs off | L46 | ~50 forward passes | **experiment, not a plot** |
| `ioi_circuit_diagram.py` | the full circuit diagram | L46 | drawing only | |
| `polysemantic_neuron.py` | one neuron's top activations across unrelated contexts | L47 | corpus pass, minutes | needs a corpus |
| `superposition_geometry.py` | near-orthogonal directions vs dimension | L47 | numpy only | |

Screenshots (Neuronpedia, circuit-tracer graphs) are captured, not generated - attribute on-frame.

**Two of these are experiments with an unknown answer, not plotting jobs.** v1 listed them at a
forward-pass count as if the result were already in hand. It is not:

- **`self_repair.py` is load-bearing.** L46 section 6 is built on the Hydra effect reproducing in
  GPT-2-small on IOI - ablate the name-mover heads, watch the backup heads take over. If it does
  not reproduce cleanly at this scale, the section's centrepiece is gone and section 6 needs a
  different spine. It also depends on `dla_per_head.py` having identified the name-mover heads
  first, so it cannot be the first thing built.
- **`make_ioi_dataset.py` is unaccounted work**, not a free input to `probe_by_layer.py`.
  Generating ~1k name/place/object-varied prompts with correct duplicate-name labels is a small
  build job of its own, and the probe result is only as good as the dataset's balance.

Both belong to the **build-risk gate** in Open questions - resolve them by running the experiment
*before* writing the deck, never on the slide.

**Build prerequisite** - not installed in `ma` today (verified 2026-08-13: `torch 2.9.0+cpu` and
`transformers 4.57.1` are present, the rest are not):

```bash
uv pip install --python ./ma/Scripts/python.exe transformer_lens circuitsvis
```

Pin exact versions in the commit per the global rule. `sae_lens` is only needed if L47 runs a
real SAE locally rather than using Neuronpedia - see Open questions.

---

## Homework (chapter page, not a session)

Proposed `HW_mech_interp.ipynb`, Colab-friendly, GPT-2-small only:

1. Load GPT-2-small in TransformerLens, run the IOI prompt, confirm the answer.
2. Find the top-3 heads by direct logit attribution.
3. Ablate each and report the logit-difference drop - **and explain why the drop is smaller than
   DLA predicted** (self-repair; this is the question the assignment is really about).
4. Open the same activations in Neuronpedia and name three features.

Deliberately no SAE training - `HW1_sae_rnn` already covers that, and repeating it wastes the
students' time.

---

## Open questions

1. ~~**Numbering.**~~ **Resolved 2026-08-13: `L45`-`L47`, build order.** See locked decision 5
   and `DECISIONS.md` #13.
2. **Schedule cost. STILL OPEN - blocks the build.** Three sessions inserted mid-October pushes
   the projected finish from ~11-20 November to ~17-26 November. Acceptable, or does something
   else get cut? Note the compounding risk: this chapter sits directly behind the **attention +
   LLM block, which is the largest and least-built block in the course** (7 of 8 sessions of
   material did not exist as of `ml/00_plan.md`, 2026-08-08). If that block slips, this chapter
   slips with it, and it is the one that gets cut under pressure because nothing downstream
   depends on it.
3. ~~**Build-risk gate.**~~ **All three CLEARED 2026-08-13, before any `.tex` was written.**
   - **L46 cold open** - `L9H8` found: 0.318 attention to the answer, DLA `-0.001`, ablation
     effect `-0.034`, and it sits in the same layer as the two real name movers. Better than the
     fallback. (`results/decoy_head_hunt.json`)
   - **`self_repair.py`** - reproduces, and harder than assumed: ablating the name movers makes
     the model *better*. Mechanism splits 45% suppressor-goes-quiet / 40% genuine backups.
     (`results/self_repair.json`)
   - **`make_ioi_dataset.py`** - 512 IOI prompts + 512 balanced probe prompts, all 44 candidate
     names single-token, both template orders at 256/256, all prompts 16 tokens.

   **Sanity check that the whole setup is right:** the experiment independently recovered the
   canonical Wang et al. head assignments without being told them - name movers `L9H9`, `L9H6`,
   `L10H0`; negative name movers `L10H7`, `L11H10`. That agreement is the strongest evidence the
   metric, the LayerNorm handling and the dataset are all correct.
4. **SAE demo in L47: local or hosted?** Neuronpedia needs no install and always works;
   a local `sae_lens` run on Gemma-2-2B is more convincing but needs a download and probably
   Colab. Recommend Neuronpedia for the lecture, local as an optional homework extension.
5. **`ml/05_interpretability` cross-link.** Chapter 05's page should gain a forward pointer to
   this chapter ("black-box methods; the white-box ones are in ch19"), and this chapter's page a
   back-pointer. Small edit, but it should happen in the same commit.
6. **Does `ch18_agents/L44` need a back-reference?** The monitoring/auditing argument in L47 is
   the natural answer to L44's trust problem. One frame, if you want the chapters linked.

---

## Sources

Instructor-side grounding. Not a student reading list (except where L47 section 7 says otherwise).

**The guide this chapter is shaped after**
- Neel Nanda, *How To Become A Mechanistic Interpretability Researcher* (current version of the
  getting-started guide; the older `neelnanda.io` quickstart URLs now redirect here).
- Neel Nanda, *An Extremely Opinionated Annotated List of My Favourite Mechanistic
  Interpretability Papers*.
- ARENA chapter 1, *Transformer Interpretability* - sections 1.2, 1.3.1, 1.3.3, 1.4.1, 1.5.2.

**Foundations**
- Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*.
- Elhage et al. (2022), *Toy Models of Superposition*.
- Olsson et al. (2022), *In-context Learning and Induction Heads*.
- nostalgebraist (2020), logit lens.
- Ferrando et al. (2024), *A Primer on the Inner Workings of Transformer-based Language Models*
  (arXiv 2405.00208).

**Circuits and causal methods**
- Wang et al. (2022), *Interpretability in the Wild: a Circuit for Indirect Object Identification
  in GPT-2 small*.
- Heimersheim & Nanda, *How to Use and Interpret Activation Patching*.
- Nanda (2023), *Attribution Patching*.
- Conmy et al. (2023), *Automated Circuit Discovery*.
- McGrath et al. (2023), *The Hydra Effect*.
- Makelov et al., *An Interpretability Illusion for Subspace Activation Patching*.
- Bolukbasi et al., *An Interpretability Illusion for BERT*.
- Nanda et al. (2023), *Progress Measures for Grokking via Mechanistic Interpretability*
  (already covered by the `misc/grokking` deck - callback only).

**Features, SAEs, attribution graphs**
- Bricken et al. (2023), *Towards Monosemanticity*.
- Templeton et al. (2024), *Scaling Monosemanticity* (Golden Gate Claude).
- Lieberum et al. (2024), *Gemma Scope* (arXiv 2408.05147); Gemma Scope 2, DeepMind, Dec 2025.
- Ameisen, Lindsey, Pearce et al. (2025), *Circuit Tracing: Revealing Computational Graphs in
  Language Models*.
- Lindsey et al. (2025), *On the Biology of a Large Language Model*.
- *Sparse Autoencoders Do Not Find Canonical Units of Analysis* (ICLR 2025).
- Sharkey et al. (2025), *Open Problems in Mechanistic Interpretability* (arXiv 2501.16496).

**Tools**
- TransformerLens (Nanda; maintained by Bryce Meyer) - <=9B models, hook-based.
- nnsight / nnterp - HF-native, larger models, remote execution.
- SAELens - SAE training and loading.
- Neuronpedia (Decode Research) - hosted feature browser and circuit-tracer frontend.
- `circuit-tracer` (Anthropic, open-sourced May 2025) - Gemma-2-2B, Llama-3.2-1B, Qwen3-4B.
