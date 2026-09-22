# Chapter 19 extension - from 3 decks to 8

**Status:** approved 2026-09-22, build in progress. This file is the spec and the build log for the
extension. `MECH_INTERP_CHAPTER_PLAN.md` stays as the record of the v1 build (the three decks as
they were on 2026-08-14) and is not edited further except for a pointer to this file.

## Why

The instructor read the LMU Interpretable Machine Learning course (`_reference/lecture_iml`) and
asked for mechanistic interpretability to be added on top of it, made **easier to understand with
more examples**, with the gaps found and filled, and with an **intro deck** that explains the idea
and what the chapter is doing.

What the survey found (2026-09-22):

- **LMU has no mechanistic interpretability at all.** Upstream (`slds-lmu/lecture_iml`, checked via
  the GitHub API on 2026-09-22) has the same chapters as our clone plus a new `evaluation/` folder.
  So the only mech interp material is ours, and the extension is to `ml/ch19_mech_interp`.
- **What we borrow from LMU is structure, not slides.** Every LMU chapter opens with an intro chunk
  (motivation, where the chapter sits in the taxonomy) and its intro chapter sorts methods along
  axes (intrinsic vs model-agnostic, local vs global, effects vs importance). The new intro deck
  places mech interp on those axes, so a student coming from `ml/05_interpretability` (built on
  LMU) gets the same map with one new region on it.
- **Gaps in ch19 v1:**
  1. no intro deck - the "what is this" material was six cold-open frames inside L45;
  2. one running example (IOI) that needs seven head classes before it makes sense - no easy
     examples first, no by-hand toy network;
  3. the weights-based half of the field (QK/OV circuits, composition - *A Mathematical Framework
     for Transformer Circuits*) was only cited on a closing frame;
  4. MLPs were almost absent - heads and SAEs only; nothing on where facts are stored;
  5. vision, where the field started (feature visualization, curve detectors), was absent; the
     course has one Grad-CAM frame in `ch6_cnn/L18`;
  6. 2025-26 work on personas and model diffing was absent (natural link to `ch20`).

## Locked decisions (instructor interview, 2026-09-22)

| # | Question | Answer |
|---|---|---|
| 1 | Deck format | **House decks as-is** (one deck per 90-min session, ~40-55 frames, `ml/SLIDE_STYLE.md`) - not LMU chunks |
| 2 | Gap topics | **All four**: transformer circuits math, where facts live (MLPs), vision interpretability, personas and model diffing |
| 3 | Session budget | **Build it all, decide later** what gets taught |
| 4 | Models | **GPT-2 small on CPU** for the language decks |
| 5 | Vision model | **ResNet-18** (torchvision ImageNet weights, same model as `ch6_cnn` Grad-CAM; already cached) |
| 6 | Circuits-math models | **GPT-2 small + TransformerLens `attn-only-1l` / `attn-only-2l`** |
| 7 | Diffing example | **Fine-tune GPT-2 small ourselves** to install a known behaviour (a model organism) |
| 8 | Edits to L45-L47 | **Moderate**: move the "what is mech interp" frames to the intro, add easier examples and worked frames, split the densest frames, cut only duplicates, keep every measured result |
| 9 | Dependency structure | **Core + add-ons** (below) |
| 10 | Heavy CPU runs | Pre-approved, sequential, one process at a time; only "very long" runs need asking. Working rule: size every run to finish in **under ~20 minutes**; anything projected longer is resized, not launched |
| 11 | Git | **One local commit per finished, verified deck. No push.** |
| 12 | Spec review | Instructor delegated it: "review yourself and start work" |

The Sonnet student review stays **opt-in and is not run** during this build; it is offered per deck
at the end.

## Structure: 4 core decks + 4 add-ons

The core decks form a complete 4-session chapter on their own. Each add-on depends **only on core
decks before it**, never on another add-on, and no core deck depends on an add-on. So any subset of
the add-ons can be dropped at scheduling time without breaking a callback. The cost is a 2-3 frame
recap at the start of each add-on.

| Order | File | Title | Kind | Builds on |
|---|---|---|---|---|
| 1 | `xx_mech_interp_intro.tex` | What we are doing, and why | **new, core** | - |
| 2 | `xx_vision_circuits.tex` | Seeing what a network sees | new, add-on | 1 |
| 3 | `xx_opening_the_box.tex` | Opening the box (was L45) | **revise, core** | 1 |
| 4 | `xx_transformer_circuits.tex` | Circuits you can read off the weights | new, add-on | 3 |
| 5 | `xx_does_it_actually_do_that.tex` | Does it actually do that? (was L46) | **revise, core** | 3 |
| 6 | `xx_where_facts_live.tex` | Where facts live | new, add-on | 5 |
| 7 | `xx_features.tex` | Features (was L47) | **revise, core** | 5 |
| 8 | `xx_personas_and_diffing.tex` | Personas, and what fine-tuning changed | new, add-on | 7 |

**Rules that keep the add-ons droppable:**

- A core deck may *mention* an add-on as optional material ("if you saw the vision lecture...") but
  never *relies* on it. No core frame uses a term that only an add-on defines.
- Each add-on opens with a recap of exactly the core material it uses, named by deck title (not by
  number, since the delivery numbers are not known yet).
- Cross-deck references use deck titles, not "lecture 2".

**Naming.** Undelivered decks carry the `xx_` prefix (same convention the ch11 name-inventor
practical uses) and get their `NN_` number on delivery. L45-L47 are renamed to `xx_` names:
since 2026-09-21 the delivered ch11 decks are `44_*` and `45_*`, so "L45" now names two different
lectures. This reopens `DECISIONS.md` #13 - see the new entry there. Figure scripts keep their
existing names (`l45_*.py` etc.) - they are internal and renaming them is churn with no reader.

**New script prefixes:** `intro_*.py`, `vision_*.py`, `circuits_*.py`, `facts_*.py`,
`diffing_*.py`. Same conventions as v1: `py_src/*.py` -> `fig/*.pdf` + `results/*.json`, logs to
`logs/`, seed 509, figures read JSON and never re-run the experiment, every number on a slide is
measured in this repo or labelled as *reported by* a named paper.

## Build-risk gates (run before the deck that needs them is written)

v1 ran every uncertain experiment before writing a slide, and two of its planned claims were
corrected by measurement. Same rule here. If a gate fails, the section is redesigned around what
was actually measured and the failure is recorded below - never faked on a slide.

| Gate | Deck | Question | Result |
|---|---|---|---|
| G1 | intro | Does a tiny MLP trained here on a 2-D toy task learn neurons we can name (one per edge of the shape), does ablating one remove exactly that edge, and do other seeds find the same edges? | pending |
| G2 | vision | Is there a curve-detector-like channel in ResNet-18 (tuned to curve orientation, weak on straight lines)? | pending |
| G3 | vision | Does guided backprop fail the weight-randomization sanity check while plain gradients / IG change? | pending |
| G4 | circuits | Do `attn-only-2l` composition scores single out the previous-token head -> induction head pair? | pending |
| G5 | facts | Does GPT-2 small know enough landmark/capital facts, and does causal tracing localize them? | pending |
| G6 | facts | Does a ROME-style rank-one edit make GPT-2 small say "Rome" for the Eiffel Tower, and what does it break? | pending |
| G7 | diffing | Does fine-tuning GPT-2 small on a narrow positive corpus shift behaviour outside that domain, and does the activation diff line up with an independently measured direction? | pending |

## Deck outlines

Frame counts are targets, not budgets (`SLIDE_STYLE.md`: long decks are fine, split only a dense
frame). Every new deck follows the house skeleton: cold open -> Outline -> sections with `[plain]`
transition slides -> Recap -> `paramgreen` Next box -> `% Provenance:` block.

### 1. Intro - "What we are doing, and why" (~40 frames, new, core)

**Cold open.**
- Three things GPT-2 small does, measured: completes a fact ("The Eiffel Tower is in the city of"),
  continues a random pattern it has never seen (induction), resolves who-gave-what-to-whom (IOI).
  "By the end of this chapter you will know how each of these works inside." (Each hook is paid off
  by a later deck - facts by the facts add-on and the causal deck, induction by the reading deck
  and circuits add-on, IOI by the causal deck.)
- "Twenty lectures of black boxes" (moved from L45).
- Predict first: is knowing every weight the same as understanding? (moved from L45)
- Definition frame (moved from L45).

**Section 1 - Why look inside?** One concrete case per reason:
- debugging: the husky/wolf snow detector (ch05 LIME figure) - black-box methods found the snow,
  but cannot say how the model uses it or where to fix it;
- trust: a model whose written reasoning is not what it computed (teaser for the features deck);
- science: the Othello board inside a model that never saw a board (teaser for the reading deck);
- control: steering a concept on and off (teaser).
- "What this does not replace" (moved from L45) - black box vs white box, stated evenly.

**Section 2 - Where it sits on the map.** The ch05/LMU axes recapped (intrinsic vs post-hoc,
model-agnostic vs model-specific, local vs global), then one new axis: *what is the explanation made
of* - inputs (SHAP, LIME, saliency) or internal components (neurons, heads, directions). A
Python-drawn map placing every method the course has taught, with mech interp in its own corner.
Why not just use a glass-box model (ch05 deck 1)? Because there is no glass-box model that writes text.

**Section 3 - A network small enough to read completely.** (gate G1)
- Train a tiny MLP (2 inputs, a handful of hidden ReLU units) to say whether a point is inside a
  diamond. Each hidden unit turns out to be one edge of the diamond - plot each unit's activation
  over the plane.
- Worked-numbers frame: push one point through by hand.
- Read the circuit off the weights: the output unit is an AND of the edges.
- Test it by intervention: ablate one hidden unit and the diamond loses exactly that edge (measured).
  First appearance of the chapter's motto: *looking is a hypothesis, intervening is evidence*.
- Why the same reading fails on GPT-2: the numbers (36,864 MLP neurons, 144 heads, 768-dim inputs
  you cannot plot, neurons that fire on many unrelated things).

**Section 4 - Three words: features, circuits, universality.** (Olah et al., 2020, *Zoom In*)
- feature = a direction in activation space that means something - the diamond edges; a real
  GPT-2 example measured here (token-embedding arithmetic, if it works on GPT-2 small; else a
  difference-of-means direction);
- circuit = features wired together by weights - the diamond's AND; the induction pair, previewed;
- universality = the same features in different networks - measured in miniature: the diamond MLP
  retrained from several seeds finds the same edges every time (up to order); then the literature
  claim for real networks (edge and curve detectors across CNN architectures), labelled as reported.

**Section 5 - How the field works.** The loop: look -> hypothesis -> intervene -> evidence. The
chapter's evidence table (what we see / what it proves) from the qmd page. Two analogies:
reverse-engineering a compiled program into source code; neuroscience with superpowers (every
neuron recorded, perfect interventions, unlimited identical copies).

**Section 6 - A short history.** Python timeline: saliency maps (2013) -> feature visualization
(2017) -> circuits in vision (2020) -> transformer circuits (2021) -> induction heads, IOI,
superposition, ROME (2022) -> sparse autoencoders (2023) -> Golden Gate Claude (2024) -> attribution
graphs (2025). Every date web-verified before it is baked in.

**Section 7 - The plan.** Chapter map (core + add-ons), why GPT-2 small (moved from L45), what you
will be able to do at the end, what the chapter will not give you (a full explanation of a
frontier model), the two tools (TransformerLens, Neuronpedia). Recap, Next.

### 2. Vision - "Seeing what a network sees" (~45 frames, new, add-on)

Builds on: intro. Model: ResNet-18. Photo: `ml/ch6_cnn/fig/src_pomegranate.jpg` (the ch6 Grad-CAM
photo) plus one or two more from the repo.

- **Cold open.** Grad-CAM (ch6) and LIME's husky (ch05) are both heatmaps over the input. Neither
  opens the network. What is inside ResNet-18?
- **Gradients as explanations.** Saliency maps (Simonyan et al., 2013): the gradient of the class
  score with respect to the pixels. By hand first: for a linear model the saliency is the
  coefficient (ch05 callback). Measured on ResNet-18: noisy. SmoothGrad as the fix, one frame.
- **Integrated gradients** (Sundararajan et al., 2017). By hand: f(x1, x2) = min(1, x1 + x2) at
  (1, 1) - the gradient is zero, so plain saliency gives no credit; IG splits it. The completeness
  property (attributions add up to f(x) - f(baseline)), checked numerically on ResNet-18 as the
  step count grows. The baseline trap (a black baseline gives black pixels zero credit). Callback:
  IG is to gradients what Shapley is to marginal contributions (ch05).
- **Sanity checks** (Adebayo et al., 2018). Randomize ResNet-18's weights from the top down; a map
  that barely changes was never explaining the model. Measured for gradient, IG and guided
  backprop (gate G3). The lesson: an input heatmap can look plausible and depend on nothing.
- **Feature visualization** (Olah et al., 2017). Optimize an image to excite one channel. Naive
  optimization gives adversarial noise (measured); jitter / scaling / a decorrelated
  parameterization give pictures. Measured grid: channels from early to late layers - edges and
  colours, textures, patterns, object parts.
- **From neurons to circuits.** The curve-detector hunt (gate G2): synthetic curve and line
  stimuli at many orientations, tuning curves per channel, pick the best candidate. Then read its
  circuit from the weights: which earlier channels feed it, and do they look like edges at
  compatible orientations?
- **Polysemantic neurons, from the weights alone.** Last-layer channels whose strongest class
  votes (`fc.weight`) are unrelated classes, with their feature visualizations. Hand-off: neurons
  are not always the unit - which the features deck takes up.
- Recap, Next.

### 3. Reading the model (was L45) - moderate revision, core

- Cold open shrinks: the six "what is mech interp" frames move to the intro; L45 opens with a
  two-frame recap of the intro (access is not understanding; the look -> intervene loop) and the
  IOI sentence.
- **Residual stream:** a by-hand toy first - a width-2 stream, an embedding plus two writes, a
  two-token unembedding, the logit difference decomposed by hand - then the GPT-2 numbers.
- **Logit lens:** first on the easy fact prompt ("The Eiffel Tower is in the city of", layer-by-layer
  top token, measured), then the IOI curve.
- **Probes:** add a picture of "linearly readable" - the layer-3 activations projected onto the
  probe direction, coloured by label (measured from the existing dataset).
- A glossary frame (layer, head, position, residual stream, logit, activation, component).
- Keep: every measured figure and number, the probe-vs-lens contrast, the induction walk-through,
  the correlational recap.
- References to "lecture 2/3" become deck titles.

### 4. Transformer circuits - "Circuits you can read off the weights" (~45 frames, new, add-on)

Builds on: reading deck (residual stream, induction heads). Source: Elhage et al. (2021).

- **Cold open.** In the reading deck we watched induction heads by running text through the model.
  Today we find them without running any text at all.
- **A head is two circuits.** Full step-by-step derivation: scores use W_Q W_K^T (the **QK
  circuit**, where to look), outputs use W_V W_O (the **OV circuit**, what to move). By-hand
  example with a 3-token vocabulary and a 2-dim stream.
- **Zero layers = a bigram table.** W_E W_U read directly (measured top continuations).
- **One layer = skip-trigrams.** The OV circuit through the embeddings; copying heads found by the
  eigenvalues of W_E W_OV W_U (measured on `attn-only-1l`); a skip-trigram example measured, and
  its bug if one shows up.
- **Two layers = composition.** Q-, K- and V-composition defined; composition scores between every
  layer-0 and layer-1 head of `attn-only-2l` (gate G4); the induction mechanism derived from the
  weights of the pair.
- **Does it survive in GPT-2 small?** Composition scores into the induction heads the reading deck
  found (L5H5, L6H9, L5H1). Honest about MLPs and LayerNorm breaking the clean algebra.
- **What weights-only analysis can and cannot do** - exact and data-free, but blind once MLPs and
  superposition enter; why the field moved to the causal methods of the next core deck.
- Recap, Next.

### 5. Causal methods (was L46) - moderate revision, core

- New easy first patch before IOI: run "The Colosseum is in the city of", paste in the residual
  stream from "The Eiffel Tower is in the city of" at one layer, watch " Rome" turn into " Paris"
  (one-dimensional sweep over layers, measured).
- A by-hand toy for why zero-ablation is off-distribution (a head whose output is always near
  (3, 1); zero is somewhere it has never been).
- The IOI "who does what" frame built up in steps instead of one dense frame.
- Keep the cold open, the patching heatmap, DLA, the scaling section, and self-repair untouched in
  substance.

### 6. Where facts live (~45 frames, new, add-on)

Builds on: reading deck, causal-methods deck.

- **Cold open.** GPT-2 small knows the Eiffel Tower is in Paris (measured probability). Predict
  first: is the fact in attention or in an MLP, early or late?
- **MLPs as key-value memories** (Geva et al., 2021). Derivation: an MLP is a sum over neurons of
  activation x value vector. By-hand 3-neuron example. Measured: a GPT-2 neuron's value vector read
  through the unembedding (which tokens it promotes).
- **Causal tracing** (Meng et al., 2022) - the causal deck's patching with noise instead of a second
  prompt. Measured heatmaps (layer x token) for residual, MLP and attention on a set of facts
  GPT-2 small gets right (gate G5).
- **Editing a fact.** ROME, rank-one model editing (Meng et al., 2022): the MLP output matrix as a
  linear associative memory, the rank-one update derived step by step, implemented here on GPT-2
  small (gate G6). The edit's report card, measured: efficacy (the prompt), generalization
  (paraphrases), specificity (other Paris landmarks), and ripple effects (the country, the
  language).
- **What localization does not tell you.** Hase et al. (2023): the layer causal tracing points at is
  not the layer where editing works best - measured by editing at every layer, reported honestly
  whichever way it comes out. Geva et al. (2023): attention also does part of the recall.
- Recap, Next.

### 7. Features (was L47) - moderate revision, core

- Add a worked example to the superposition section (a small number of features in 2 dimensions,
  by hand) if the existing geometry frames do not already carry one.
- Split the densest frames in the sparse autoencoder and attribution-graph sections.
- Steering stays short; the personas add-on takes the extended version.
- References to other decks by title.

### 8. Personas and model diffing (~45 frames, new, add-on)

Builds on: features deck (directions, steering, sparse autoencoders), causal-methods deck.

- **Cold open (reported).** Emergent misalignment (Betley et al., 2025): fine-tuning on a narrow
  task changed behaviour far outside it. How would you find out what a fine-tune changed?
- **A behaviour as a direction.** Difference of means on contrastive prompts; persona vectors
  (Chen et al., 2025) as the monitored, steerable version. Measured on GPT-2 small: a
  positive/negative sentiment direction, used as a probe (monitoring) and as a steering vector.
- **Our model organism** (gate G7). Fine-tune GPT-2 small on positive movie reviews only. Did it
  become more positive about things that are not movies? Measured with a next-token metric (the
  logit gap between positive and negative words on neutral templates across domains), no extra
  model.
- **Diffing base vs tuned, four ways:** outputs (per-domain KL), weights (per-layer change), activations
  (mean difference per layer), and whether the activation diff lines up with the independently
  measured sentiment direction (cosine per layer).
- **The frontier, labelled as reported:** crosscoders for model diffing (Lindsey et al., 2024), the
  "persona" features found behind emergent misalignment, subliminal learning (link to ch20).
- **Honest limits.** Diffing tells you what changed, not why; a direction is a candidate until
  intervened on; steering has side effects.
- Recap, Next.

## Chapter page

`mech_interp.qmd` gains Armenian entries for all eight decks, marks the four add-ons as optional,
and keeps its "not finished" warning until the chapter is taught. The existing callouts stay.

## Definition of done, per deck

- Build-risk gate(s) for the deck run and recorded in the table above.
- Figure scripts run end-to-end under `ma`; raw results in `results/*.json`; logs in `logs/`.
- `compile-deck` loop: 2 x pdflatex, 0 `!` lines, `detect_clipped_slides.py` and
  `detect_footer_collisions.py` both run with every flag checked on the rendered page, acronym
  check, `clean_latex.py`, `% Provenance:` block.
- Chapter qmd entry updated.
- `DECISIONS.md` entry if the deck made a decision a reasonable instructor would have made
  differently.
- One local commit. No push.

## Out of scope for this build

- LMU visual style (`\titlemeta`, learning-goal cards).
- `HW_mech_interp.ipynb` (still spec'd in the v1 plan; unbuilt).
- The chapter's schedule slot in `ml/00_plan.md`.

## Implementation plan

Executed inline by one agent (no subagent fan-out - global rule). Tasks in order; each ends in a
verifiable state. "Verify" lines are the success criteria.

**Task 0 - Rename and spec commit.**
- `git mv` L45/L46/L47 `.tex` + `.pdf` to the `xx_` names; update `mech_interp.qmd` links, the
  decks' provenance "Next deck" lines, and this chapter's plan pointers.
- Verify: `git grep -n "L4[567]_" -- ml/ch19_mech_interp ':!*.md'` returns nothing; both new
  paths compile after Task 3/5/7 (no content change yet).
- Commit: spec + DECISIONS #36/#37 + rename.

**Task 1 - Intro deck.**
1. `py_src/intro_toy_mlp.py`: train the diamond MLP (seed 509, then 5 seeds), save per-unit
   activation grids, weights, ablation results, seed-match table -> `results/intro_toy_mlp.json`;
   figures `fig/intro_*.pdf`. Gate G1 recorded in the table above.
2. `py_src/intro_figs.py`: the three GPT-2 hooks (fact / induction / IOI, measured top tokens and
   probabilities), the method map, the timeline (dates web-verified), the embedding-arithmetic or
   difference-of-means feature example -> `results/intro_figs.json`.
3. Write `xx_mech_interp_intro.tex` per the outline; move the five cold-open frames out of
   `xx_opening_the_box.tex`.
4. Verify: `compile-deck` loop clean (2 passes, 0 errors, both detectors, acronym check,
   provenance). Update qmd. Commit.

**Task 2 - Revise `xx_opening_the_box.tex` (L45).**
1. `py_src/l45_easy_figs.py`: logit lens on the fact prompt (top token per layer), probe-direction
   projection at layer 3, by-hand toy stream numbers -> `results/l45_easy_figs.json`.
2. Edit the deck per outline 3. Verify: compile loop; every figure/number from v1 still present
   (`grep` the v1 numbers: 3.60, 5.34, 0.00000, layer 3, layer 9, 0.89). Commit.

**Task 3 - Revise `xx_does_it_actually_do_that.tex` (L46).**
1. `py_src/l46_easy_patch.py`: Colosseum <- Eiffel residual patch, sweep over layers ->
   `results/l46_easy_patch.json`, `fig/easy_patch_layers.pdf`.
2. Edit per outline 5. Verify: compile loop; v1 numbers intact (+3.580, +3.754, 45%, L9H8
   -0.001). Commit.

**Task 4 - Revise `xx_features.tex` (L47).** Edit per outline 7. Verify: compile loop. Commit.

**Task 5 - Vision add-on.** Gates G2, G3 first (`py_src/vision_gates.py`), then
`py_src/vision_figs.py` (saliency, SmoothGrad, IG + completeness, sanity checks, feature viz grid,
curve tuning + weights circuit, polysemantic fc channels). Each heavy run sized under ~20 min.
Write deck, compile loop, qmd, commit.

**Task 6 - Transformer-circuits add-on.** Gate G4 first (`py_src/circuits_gates.py` on
`attn-only-1l/2l`), then `py_src/circuits_figs.py`. Deck, compile loop, qmd, commit.

**Task 7 - Facts add-on.** Gates G5, G6 (`py_src/facts_tracing.py`, `py_src/facts_rome.py`),
figures, deck, compile loop, qmd, commit.

**Task 8 - Personas/diffing add-on.** Gate G7 (`py_src/diffing_finetune.py` - the one long run,
sized under ~20 min; checkpoint saved outside git), `py_src/diffing_figs.py`, deck, compile loop,
qmd, commit.

**Task 9 - Wrap.** Chapter qmd final pass, work-session log, list of decks ready for the opt-in
student review.

## Build log

(Newest last. Gate results, surprises, and anything the measurements changed.)
