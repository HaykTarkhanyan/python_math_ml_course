# Neural-networks chapter (ch5) - deck review + practical redesign proposals

Reviewed 2026-09-05 (Claude). Scope: `L14_neural_networks.tex`, `L15_training_neural_networks.tex`
(close read + numeric re-check of every worked example), a skim of the 7 `dl_*` embed decks,
`nn_practical_solution.ipynb`, `neural_networks.qmd`, and the practicals of chapters 02-04 and
09-10 for comparison. Method: full read of both house `.tex` files, numpy re-computation of the
L14/L15 worked forward + backward pass, mechanical acronym sweep, and an import check of the `ma`
venv. No rendered-page overflow pass was done (run `/beamer-overflow-check` before delivery).

**Verdict up front.** L14/L15 are strong decks: the callback spine (neuron = logistic regression,
weight decay = Ridge, dropout = bagging, output gradient = the same `f - y` residual) is the best
thing about them, the worked 2-2-1 example is carried correctly from forward pass through backprop
to a verified loss drop, and the honest "trees still win on tabular" framing is exactly right.
Every number I re-computed is correct at full precision. The issues are small: one cross-deck
display inconsistency, two unexpanded acronyms, and a handful of missing one-liners (zero-init
symmetry, `model.eval()`, input scaling) that students will otherwise hit as bugs in the practical.
The bigger opportunity is the **practical**: Fashion-MNIST was fine in July, but chapters 9-10
have since raised the bar (Lake Sevan, Armenian semantic tree, 1000 Genomes, the students' own
photos), and this chapter - the course's gateway to deep learning - now has its least memorable
practical. Section 4 proposes four replacements.

---

## 1. Fixes for L14 / L15 (small, do before delivery)

1. **The worked forward pass disagrees across decks: `f_in = 0.81` (L14, "Worked forward pass")
   vs `f_in = 0.80` (L15, "Forward pass: cache...").** Verified: at full precision
   `f_in = 0.8034`, so L15 is right; L14's 0.81 comes from adding the *displayed* rounded values
   (0.82 - 0.21 + 0.2 = 0.81). Worse, L15 prints the same rounded operands and then an answer they
   do not sum to - a student re-doing the arithmetic gets 0.81 and concludes the slide has a typo.
   Fix: pick one convention. Cheapest is to make both decks say 0.81 (displayed-value arithmetic,
   self-consistent on the slide) and drop a footnote "carrying full precision gives 0.80; small
   rounding drift is normal". Everything downstream (0.69, -0.31, the deltas, 0.75 / 0.28 after
   the step) re-verifies correctly at full precision, so nothing else needs touching.
2. **Acronym convention: `GELU` and `ELU` (L14, dead-ReLU watch-out box) are never expanded.**
   Per the SLIDE_STYLE rule these are exactly the non-exempt kind (method names from papers).
   One-line fix in the box: "GELU (Gaussian Error Linear Unit) / ELU (Exponential Linear Unit)".
   `XOR` and `TPU` are also unexpanded; both are borderline-universal - instructor's call, but a
   parenthetical "(exclusive or)" on the XOR frame costs nothing. The sweep flags nothing else.
3. **"Perceptron" naming nit (L14, "A neuron: affine transform, then activation").** The frame
   defines a neuron with an arbitrary activation and calls it "a perceptron". Historically the
   perceptron is the *step*-activation special case, and the step function never appears in the
   deck. Suggest "(historically called a perceptron, which used a step function here)" - it also
   quietly explains why the 1958 milestone on the history frame could not be trained by gradients.
4. **Two-moons numbers (0.84 -> 0.92) should be printed by the figure script.** Per the
   2026-08-13 learning ("write the practical after measuring"), every number in prose should be
   traceable to code output. `py_src/nn_fundamentals_figs.py` generates the figures; confirm it
   logs both accuracies, so a future regeneration that shifts them cannot silently contradict the
   slides (the L14 recap repeats the numbers a second time).

## 2. Missing one-liners that will bite in the practical

These are all single-sentence or single-box additions; each pre-empts a real student failure.

5. **Zero-initialization symmetry is never mentioned.** The L15 init frame covers *scale* (He /
   Xavier) but not the classic predict-first moment: "initialize every weight to 0 - what goes
   wrong?" Answer: all neurons in a layer compute the same output and receive the same gradient,
   so they stay identical forever - the layer collapses to one neuron. This is the strongest
   argument for *random* init and a perfect `\pause` frame (per the predict-first preference).
   Right now a student could leave believing init is only about variance.
6. **`model.eval()` / `model.train()` is never named.** The dropout frame correctly notes
   inverse-dropout scaling, and the BatchNorm frame gives the batch statistics - but at test time
   BatchNorm uses *running* statistics, and in PyTorch both behaviors switch on `model.eval()`.
   Forgetting it is the single most common PyTorch beginner bug, and the practical uses both
   layers. One `\scriptsize` line on the BatchNorm frame: "at test time BN uses running averages;
   `model.eval()` switches BN and dropout to test mode - forgetting it is the classic bug."
7. **Input scaling is never stated as a rule.** The practical scales pixels to [0, 1]; the decks
   never say that NNs (unlike trees, as the L14 tabular frame itself notes) need scaled inputs.
   One line on the L15 "sensible starting recipe" box: "and scale your inputs first - the same
   standardization habit from the regression chapters."
8. **Softmax + cross-entropy gradient generalizes the `f - y` story - say so.** The worked
   example is binary, but the practical is 10-class. It is a beautiful fact that the softmax + CE
   output gradient is also `p - y_onehot`, per class. One sentence on the "output gradient is
   just f - y" frame ("the same holds per-class for softmax + cross-entropy") closes the gap
   between the deck's binary example and the practical's 10 classes.
9. **"Patience" is used in the practical but never defined in the deck.** The early-stopping
   bullet on the training-curves frame could add "(in code: stop after `patience` epochs with no
   validation improvement, keep the best weights)".

## 3. Extensions and pedagogy improvements (worth it, not urgent)

10. **Implement the parked `TODO_nn_vs_boosting_on_tabular.md` as one measured frame.** The
    L14 "why trees win" frame argues the point verbally with a citation; the TODO's version - a
    small bar chart, LightGBM vs a tuned MLP on 2-3 tabular datasets, measured by our own script -
    is far more convincing and matches the house "every essential figure is Python-generated"
    rule. Alternatively fold it into the practical as the closing act (see candidate D below);
    doing it in *both* places would be redundant - pick one.
11. **The loss-landscape stretch of L15 is the deck's only visual-free run.** "A word on the loss
    landscape" + "Saddles and cliffs" are two consecutive text-only frames about an intensely
    visual subject. The chapter outline already flagged an optional figure. Options: generate a
    simple 2D non-convex surface in `py_src/`, or embed the Li et al. (2018) "Visualizing the
    Loss Landscape of Neural Nets" skip-connection figure as a full-bleed frame with attribution
    (the approved pattern). Verify the license before borrowing.
12. **L14 has one predict-first moment; it could carry three.** The cold open works. Two cheap
    additions: (a) the parameter-count frame - "guess how many weights the 784-128-64-10 net
    has" before revealing 109,386 (students guess low by an order of magnitude); (b) the
    zero-init question from item 5 if it lands in L14's orbit instead of L15's.
13. **No Armenian running example anywhere in the chapter.** The classification chapter had the
    cheese factory; ch9/10 have Sevan and Armenian words. Both NN decks are entirely abstract
    (moons, XOR, MNIST). The practical redesign (section 4) can fix this at the chapter level -
    and if candidate A is chosen, L14's "when NNs win" frame gains a natural local example
    ("reading handwritten Armenian letters" as the raw-signal case) that foreshadows the
    homework.
14. **TensorFlow Playground deserves a mid-lecture slot, not only the wrap-up.** The "Play with
    it before L15" frame is good, but the two-moons payoff frame ("A hidden layer bends the
    boundary") is the natural moment to switch to the projector and add neurons live - the deck
    could carry a one-line cue for the instructor there.
15. **`dl_*` embed set - maintenance notes only.** The set is documented (README) and coexists
    deliberately; no structural complaints. Two notes: (a) the 2026 industry/frontier claims
    (NVIDIA market cap, DeepSeek, Bun rewrite, IMO results) were web-verified in July 2026 but
    will date fast - the Sources frames exist, so just re-verify before each delivery semester
    and stamp "as of mid-2026" on the state-of-the-field frames; (b) the set is still not
    registered in `_quarto.yml`, which the README already tracks as an open item.

---

## 4. The practical: review and four replacement candidates

### What the current practical does well - keep these bones regardless

The Fashion-MNIST notebook is competently laddered: linear floor -> MLP beats it -> lr sweep ->
regularization gap -> BatchNorm + early stopping, with the curve-reading habit threaded through.
The numpy-backprop bonus that reproduces the lecture's exact numbers (`f - y = -0.31`, gradient
check) is genuinely good and should survive any redesign, whatever dataset sits above it.

Independent of topic, three mechanical issues:

- **It loads data via `tensorflow.keras.datasets` inside a PyTorch practical.** A student with a
  torch-only install cannot run cell 3. `torchvision.datasets.FashionMNIST` is already in the
  `ma` venv (verified: torchvision 0.24.0), or better, pin a small `data/*.npz` in the repo like
  ch9/10 do - the course's established student-proof pattern.
- **There is no student-facing starter notebook.** ch3 and ch4 ship `*_project.ipynb` +
  `*_solution.ipynb` pairs; ch5 has only the solution, with the tasks living in the qmd. Ship the
  pair.
- **The qmd chapter page still carries the "not finished / drafted by Claude" warning** - fine,
  but it means the practical decision is still open, which is exactly why now is the time to
  upgrade it.

### Why replace Fashion-MNIST at all

The last four practicals (Sevan land cover, image compression, Armenian semantic tree, genes
mirror geography, plus the Google-Photos project) share a formula: real or personal data, a local
or famous hook, and a reveal the student did not expect. Fashion-MNIST is the one dataset in this
list every ML tutorial on the internet already uses. The chapter that introduces deep learning -
the topic students are most excited about - currently has the most generic homework in the
course.

### Candidate A (recommended): "HayMNIST" - the class builds its own handwritten Armenian dataset

**Pitch.** There is no famous Armenian MNIST, so the class makes one. Each student fills a
printed grid sheet with the letters **Ա through Ժ** - which are also the classical Armenian
numerals 1-10, so this literally *is* an Armenian digit dataset - photographs it, and a provided
student-proofed script (the ch10 `embed_my_photos.py` pattern) slices the grid into centered
28x28 grayscale samples. Pooled across ~20 students x 10 letters x ~10 repetitions, that is a
~2,000-sample dataset the class owns.

**Task ladder** (maps 1:1 onto L14/L15): logistic-regression floor -> MLP + training loop -> lr
sweep -> the killer act: **hold out one whole writer as the test set**. Accuracy drops versus a
random split - that is distribution shift, discovered rather than lectured - and **data
augmentation** (small rotations/shifts on the training letters) is the fix that visibly closes
part of the gap. Dropout/weight decay/early stopping slot in as before. Bonus: the numpy
backprop check, unchanged.

**Why it wins:** maximally local (bilingual course, Armenian-first artifact), personal ownership
of the data, augmentation and distribution shift stop being bullet points, and the dataset
becomes a reusable course asset (ch6 can open with "the CNN beats your ch5 MLP on your own
letters" even though its own practicals are already built). **Risks / what flips it:** needs
lead time (sheets collected before the session) and a pilot - per the 2026-08-13 learning,
collect 2-3 sheets first and *measure* whether an MLP gets respectable accuracy on ~2k samples
before writing a word of prose. If the class is small (<10 students) or there is no lead time,
fall back to candidate A': train on font-rendered synthetic letters (a given script renders
Ա-Ժ from installed fonts with jitter) and test on one instructor-collected handwritten page -
same distribution-shift punchline, zero collection logistics.

### Candidate B: "The net that paints" - an MLP learns an image, live

**Pitch.** Train an MLP to map pixel coordinates (x, y) to RGB on a single photo (the student's
own, or Ararat). The network's output *is* a painting of the image, redrawn every few epochs into
an animation: training made visible. Width/depth ladder = underfitting you can *see* (a too-small
net paints a blur); sigmoid vs ReLU = vanishing gradients as "the sigmoid net never sharpens";
held-out random pixels give an honest val curve (can the net interpolate pixels it never saw?).
Bonus: Fourier features on (x, y) - a hand-crafted-features callback to L01g that dramatically
sharpens the result, and a teaser for positional encodings.

**Why it wins:** zero data logistics, spectacular and personal output, and it turns the
chapter's most abstract claims (universal approximation, capacity, activation choice) into
side-by-side pictures. **Risks / what flips it:** it is a regression task, so softmax/CE and
classification metrics go unexercised (the numpy-backprop bonus keeps CE alive); "the goal here
IS to overfit one image" needs one honest paragraph so it does not blur the chapter's
overfitting message; CPU time at, say, 128x128 must be measured before committing. Choose this
if A's logistics fail and you want maximum wow per setup-minute.

### Candidate C: "Supervise your own photos" - an MLP head on the ch10 CLIP embeddings

**Pitch.** Direct sequel to the Google-Photos project: students already have `*.npz` files with
CLIP embeddings + thumbnails of their own photos. Label ~100 of them into personal classes
(food / screenshots / nature / people), train an MLP head on the 512-d embeddings, auto-label
the rest, and inspect the mistakes. This is L14's transfer-learning sentence made real, and it
trains in seconds on CPU.

**Why it wins:** perfect course continuity ("last chapter you clustered them, now you supervise
them"), personal data with no new collection step. **Risks / what flips it:** (a) CLIP
embeddings are famously linear-probe-friendly - a real chance logistic regression matches the
MLP, which undercuts "you need hidden layers" unless the notebook embraces it as the honest
finding ("on a strong learned representation, one neuron is often enough - depth earned its
keep inside CLIP"); measure first, frame accordingly. (b) It leans on students having done the
ch10 project (ship a fallback npz). (c) It pre-empts part of ch6's L18 transfer-learning
lecture - check the overlap before committing.

### Candidate D: the honest tabular showdown (bonus act, not the main practical)

Resurrect `TODO_nn_vs_boosting_on_tabular.md` as the *closing bonus* of whichever practical is
chosen: take the ch4 wine-quality data, apply every L15 trick to an MLP, and try to beat the
students' own LightGBM score from ch4. Most will fail, and that is the lesson - it cements the
"reach for boosting on tabular" rule with the student's own hands. As a *main* practical it is
an anticlimax; as a 20-minute bonus it is the most honest exercise in the course. (If item 10
puts the measured bar chart in the deck instead, skip this to avoid redundancy.)

### Recommendation

**A (HayMNIST, Ա-Ժ), piloted first; B if the collection logistics do not fit the calendar; keep
the Fashion-MNIST notebook as a linked reference/solution either way** (it exercises the L15
checklist cleanly and is already written). C is the cheapest to build but carries the
linear-probe risk and the ch6 overlap; D is a bonus, not a flagship. Whichever is chosen: build
the exploratory script first, measure, then write the prose (2026-08-13 learning), and ship a
starter + solution notebook pair per the ch3/ch4 convention.

---

## 5. Add-on idea (instructor, 2026-09-05): an "open the box" mech-interp act

ch19 (mech interp) is built and entirely transformer/LLM-focused (L45 probes + max-activating
examples, L46 patching, L47 SAEs/steering). A small MLP-scale "autopsy" act at the end of the
ch5 practical would not collide with it - it seeds it: open a 100k-parameter box you built
yourself in ch5, open a billion-parameter one in ch19. All of the below is CPU-trivial with no
new dependencies.

- **First-layer weights as images** (fits A or any MNIST-like data): reshape each hidden
  neuron's 784 weights to 28x28 - stroke/part templates for Armenian letters. Caveat to
  pilot-measure: on small data without regularization these often look like noisy blobs;
  weight decay cleans them up, which is itself a teachable tie-in ("regularization makes the
  templates readable").
- **Ablation** (fits A): zero out one hidden neuron, re-measure accuracy per class - "delete
  neuron 17, see which letter suffers." The core causal move of mech interp in three lines;
  pair with a dead-ReLU hunt to close the loop with L14's dead-ReLU box.
- **Per-neuron activation maps over the (x, y) plane** (unique to B): render each first-layer
  ReLU's activation across the image - the folded half-planes of L14's "depth folds space"
  frame become inspectable in the student's own net. Nothing in ch19 does this; strongest
  argument for B.
- **Activation maximization** (bonus): gradient ascent on the *input* to maximize a class
  logit - ghost letter templates, and it reinforces that backprop gives input gradients too.
- **Linear probes - hold for ch19 unless the early callback is wanted.** Probing HayMNIST
  hidden layers for *writer identity* (trained only on letter identity) would make
  representation learning measurable and the labels exist for free - but probes are one of
  L45's four core techniques, so doing it here pre-teaches ch19.

Framing rule: at ch5 depth, do not use the phrase "mechanistic interpretability" or its
vocabulary (circuits, superposition, SAEs) - "open the box" only, so ch19 still lands as new.
Fold this in as a closing act of the chosen practical rather than a separate project (one
practical with acts is the course pattern; a standalone mini-project raises workload for the
same payoff).

---

## 6. Surprise catalog (instructor, 2026-09-05): phenomena that reveal something unexpected

Brainstormed on request, web-checked for MLP-scale CPU feasibility. Two tiers: cheap **acts**
that bolt onto any flagship practical, and **phenomenon labs** big enough to be their own
notebook. Everything here is MLP-only and laptop-CPU-scale; the labs still need the
measure-before-prose pilot.

### Cheap surprise acts (minutes each, near-zero build cost)

- **Adversarial examples (FGSM).** Take the trained classifier, one gradient-ascent step on the
  *input* to raise the wrong class's logit, clip to epsilon: an image visually identical to the
  original that the net calls something else with 99% confidence. Highest wow-per-minute in
  this list, and it weaponizes the chapter's own core skill (backprop gives input gradients).
  Bonus surprise: the same adversarial image often fools a *classmate's* independently trained
  net (transferability). No later chapter covers adversarial robustness, so nothing is stolen.
- **Random-label memorization** (Zhang et al. 2017, "rethinking generalization"). Shuffle the
  labels completely and retrain: the same net still reaches ~100% *train* accuracy. One extra
  cell; lands the "capacity to memorize pure noise" point better than any slide, and makes the
  regularization section feel necessary rather than ritual.
- **Planted shortcut (Clever Hans).** Stamp a 2x2 corner marker correlated with the class into
  the training images; the net rides the shortcut and collapses when the marker is absent at
  test time. Controllable, cheap, and continues the course's audit lineage (ch2 buggy rent
  model, ch3 leakage trap). If HayMNIST is chosen, this doubles as insurance: writer-class
  confounds in collection become the lesson instead of a silent bug.
- **Frozen-random-features control.** Freeze the hidden layer at its random init, train only
  the output layer: surprisingly competitive on easy data. The honest-measurement question "how
  much did *learning* the representation actually buy?" - and the gap that remains is the
  chapter's thesis, quantified.
- **Predict-your-accuracy scaling act.** Train at 1k / 2k / 4k / 8k samples, plot error vs data
  on log-log axes (a near-straight power law), have students *predict* full-data accuracy
  before training on it, then verify. Predict-first as a workflow, plus a first taste of the
  scaling-law thinking behind modern LLMs.

### Phenomenon labs (own notebook or a large bonus; pilot first)

- **Grokking on modular arithmetic.** Train a 2-layer MLP on a p=97 addition table with weight
  decay: train accuracy hits 100% early, validation sits at chance for a long time, then
  *jumps* to ~100% long after the deck's rules say to stop. Web-checked: MLPs grok modular
  arithmetic (arXiv 2301.02679), and small setups reach it in minutes on CPU (one repo in <150
  epochs). Two chapter hooks make it teachable rather than a stunt: weight decay - taught in
  L15 - is the ingredient that makes it happen, and it is the honest asterisk on "stop at the
  val minimum". Risk: hyperparameter-finicky; commit only after a reliable local recipe exists.
- **Double descent.** Sweep MLP width on a small subsample with 10-20% label noise: test error
  falls, *rises* at the interpolation threshold, then falls again as width grows past it. This
  deliberately revises the [06] bias-variance U-curve - the most curriculum-subverting reveal
  available at this scale, and the reason to do it on purpose rather than let students meet it
  on the internet first. Web-checked: standard FCNN-on-MNIST replications exist (widths ~2-150,
  label noise makes the peak visible). Cost: a 10-15 net sweep; the nets near the threshold
  train slowest - measure the wall-clock before committing.
- **Lottery tickets / pruning.** Magnitude-prune 90% of the trained weights: accuracy barely
  moves. Then the Frankle-Carbin twist: rewind the surviving weights to their *init* values and
  retrain the sparse net - it works, random sparse nets do not. The original paper's
  LeNet-300-100 is literally an MLP on MNIST, so this is in-scope by construction. The one-shot
  pruning version is cheap; full iterative lottery-ticket rounds are a bigger build.
- **MNIST-1D as the substrate for a phenomena lab.** A pip-installable, procedurally generated
  teaching dataset (4k samples, 40-dim) designed for exactly this: lottery tickets and double
  descent run in minutes on CPU, and - the gift for us - an MLP gets ~68% while a CNN gets ~94%,
  so the lab's closing number is a cliffhanger for ch6 ("something about *shape* is missing;
  next chapter"). If a "weird phenomena" notebook is wanted, this is the cheapest reliable home
  for it.

### How these combine

The acts are not mutually exclusive with anything: adversarial + random-labels alone would lift
any flagship practical for under an hour of build time each. The labs are one-choice items -
grokking for maximum jaw-drop, double descent for maximum curriculum payoff, MNIST-1D if the
goal is a reusable phenomena playground with a built-in ch6 hook. References:
Zhang et al. 2017 (arXiv 1611.03530); Goodfellow et al. 2014 FGSM (arXiv 1412.6572); Gromov,
"Grokking modular arithmetic" (arXiv 2301.02679); Belkin et al. 2019 double descent (PNAS);
Frankle & Carbin 2019 lottery tickets (arXiv 1803.03635); Greydanus, MNIST-1D (arXiv
2011.14439). Verify exact cites when building slides, per the citation convention.

---

## 7. Broader task shapes (instructor, 2026-09-05): beyond images entirely

Constraints from the instructor: students know **no CNNs and no transformers** at this point,
and the practical should not anchor on the single-image idea (candidate B). What follows are
flagship-scale ideas across other modalities. Note the earlier surprise *acts* (adversarial,
random labels, shortcut, scaling) are modality-agnostic and attach to any classifier below.

### 7a. Chess: rediscover the piece values, then out-see them (top pick)

**Act 1 - one neuron re-derives the folk values.** Logistic regression (= the L14 single
neuron) on material differences from Lichess games, predicting who wins. The learned
coefficients, scaled to pawn = 1, come out at roughly knight 3.2, bishop 3.3, rook 4.9, queen
9.8 - the numbers every chess kid memorizes, *discovered from game outcomes* by the exact model
from L14. Established methodology (Lichess piece-value analyses; arXiv 2509.04691), so the
pilot risk is low. This is the genes-mirror-geography move: structure nobody told the model.
**Act 2 - the MLP sees position, not just material.** Same target, richer input (flattened
piece-square planes, 768-dim): the MLP beats the material-only neuron because it can represent
interactions - and the gap between the two models *quantifies how much of chess is not
material*. **The closing card:** modern Stockfish evaluates positions with NNUE - essentially
an MLP very close to what the student just built, scaled up. No CNN/transformer knowledge
touched anywhere. **Local hook, for free:** chess is a compulsory school subject in Armenia -
every student in the room has personal history with those piece values. Data: pin a prepared
`data/*.npz` of positions extracted from the Lichess open database (course pattern). Pilot:
measure Act-2's gap before promising it.

### 7b. Behavioral biometrics: the net knows who is typing

Each student types a fixed sentence ~20 times into a tiny provided script that logs key-hold
times and inter-key intervals; pooled, an MLP identifies *which classmate* is typing from
rhythm alone. Genuinely startling, collected in five minutes in class, zero images. The CMU
keystroke benchmark (51 subjects, 31 timing features) is the established fallback/extension
dataset. Timing features are tabular, so run the honest LightGBM comparison as an act - if
boosting wins, that *is* the L14 tabular lesson landing in their own data. Side benefit: a
natural security/privacy discussion (this is a real authentication technique).

### 7c. Sound without CNNs: FFT features into an MLP

The FFT is "just math" students can accept as a feature extractor (feature engineering, L01g).
Two variants: **who-is-speaking** voice ID on 1-second clips of classmates (personal-data wow,
same spirit as 7b), or **chord recognition** where training data is *synthesized* (rendered
triads) and the test set is a real guitar/piano recording - the synthetic-to-real gap makes
data augmentation load-bearing, same punchline as HayMNIST candidate A'. Keep it
features-into-MLP shallow: ch13 (audio) owns the deep treatment later.

### 7d. Phone sensors: activity recognition from your pocket

Students record accelerometer traces with a free app (e.g. phyphox) while walking / running /
climbing stairs; windowed summary stats per axis feed an MLP that labels the activity. The
model knows what you were doing from your pocket - personal and image-free. Fallback: the
classic UCI smartphone HAR dataset. Same honest caveat as 7b: engineered windows are tabular,
so boosting may tie - measure, and make the comparison part of the story.

### 7e. Preferences: the class recommender and the map of taste

Students rate ~30 well-known items (movies - or localize: Armenian dishes or songs); train a
tiny matrix-factorization model, taught as "an embedding is a one-hot times a matrix, i.e. one
more linear layer" - no new machinery. It recommends unseen items to each student, and the
reveal is ch10's own trick: t-SNE the learned item embeddings and watch genres emerge that
nobody labeled. The genes-mirror-geography of taste, and the course's first *learned*
embedding (everything in ch10 was someone else's). Ballast the tiny class matrix with
MovieLens 100k so the math has enough signal. This idea also foreshadows every later
embedding-based chapter without needing any of them.

### 7f. Science: the surrogate net (modest wow, high honesty)

Train an MLP to imitate a slow exact computation (a double-pendulum integrator, projectile
with drag): near-instant predictions inside the training envelope, and a visible failure just
outside it - inductive bias and extrapolation, measured. This is how real weather emulators
work (ties to the GraphCast frame already in `dl_intro_history`). Better as a bonus than a
flagship.

### 7g. Text without transformers: reuse the ch10 Armenian embeddings

An MLP head on precomputed Armenian sentence embeddings (already a black-box course primitive
from the ch9 semantic-tree practical) for topic or sentiment classification. Cheapest
continuity after candidate C, same linear-probe risk as C - measure whether the hidden layer
earns its keep before writing prose.

### Updated shortlist

With the no-CNN/no-transformer constraint and the image-anchoring concern in mind, the
strongest flagships are now **7a (chess)** - famous-reveal structure, national resonance, no
collection logistics, and a single-neuron -> MLP arc that mirrors L14 exactly - and **7b
(keystrokes)** for the personal-data thrill at five minutes of collection cost. Candidate A
(HayMNIST) remains the best *dataset-building* experience; 7e (recommender) is the best
foreshadowing investment. All still subject to the pilot-first rule.

*(Instructor 2026-09-05: section 7's ideas did not click either - more in section 8.)*

---

## 8. Third ideation round (instructor, 2026-09-05): the generative direction, and others

The one direction all earlier rounds missed: **generative**. Students in 2026 signed up because
of ChatGPT, and a character-level language model is legitimately within chapter scope - the
first neural LM (Bengio et al. 2003) *was* an embedding layer plus an MLP, no recurrence, no
attention. Karpathy's "makemore" curriculum proved this exact exercise works pedagogically.

### 8a. The Armenian name inventor (char-level MLP language model) - strongest of this round

Train an MLP to predict the next character of Armenian first names / surnames from a short
context window. Then *sample* from it: the net invents new, plausible, nonexistent Armenian
names - and the class watches "-yan" emerge from nothing. Every ingredient is chapter-scope:
one-hot/embedding input, hidden layer, softmax output, cross-entropy, the training loop;
sampling and a temperature knob are the only new ideas (two cells). The reveals stack:
(1) generative AI built by hand from L14/L15 parts - this MLP is ChatGPT's direct ancestor,
statable without teaching transformers; (2) overtrain it and it starts spitting out *real*
names from the training set verbatim - overfitting becomes memorization/privacy, the modern
version of the L15 lesson; (3) the temperature slider makes the diversity/quality tradeoff
tangible. Compute: trivial (a few thousand names, minutes on CPU). Data: filter the public
multinational name datasets (e.g. philipperemy/name-dataset, country = AM) in a prep script -
thousands of Armenian names, though Latin-transliterated (either generate in translit or map
back to Armenian script; wordlist variant below avoids this). Pilot: confirm sample quality at
this data size before writing prose.

**Input representation (decided in discussion, 2026-09-05).** Not integer codes - that is the
first teachable trap (magnitude/order that does not exist; callback to categorical encoding
from the feature-engineering chapter). The ladder:
(0) optional warm-up: a plain bigram *count* table, no NN - clunky names, but structure;
(1) one-hot context window: vocab = ~39 Armenian letters + a start/end token ".", context of
3 chars = 3 concatenated one-hots (3 x 40 = 120 inputs), 40-way softmax output; training pairs
by sliding a window over each name; the end token is what teaches "-յան." as a way to *stop*;
with context 1 and no hidden layer this is literally L11 logistic regression, and its learned
weights match the log of the bigram counts - the neuron re-derives the count table;
(2) embeddings as nothing new: a one-hot times a matrix selects a row, so `nn.Embedding` is
"the first linear layer, computed lazily as a lookup" - each char becomes a learned 8-dim
vector (input 120 -> 24). The MLP is the cure for the count-table explosion (a 3-char table
needs 40^3 rows) - exactly Bengio's 2003 argument, statable in one sentence.
**Bonus reveal to pilot:** PCA (ch10, their own tool) the 40 learned char vectors to 2D -
in the makemore/Bengio setup vowels cluster away from consonants, phonetics discovered from
name statistics alone. Verify it reproduces on the Armenian data before promising it in prose.

- **Variant: the word inventor.** Same machinery on an Armenian dictionary wordlist - the net
  invents words that *feel* Armenian but do not exist. Pairs beautifully with ch9's semantic
  tree (which planted fake-word probes); native Armenian script with no transliteration issue.
- **Variant: the folk melody generator (8e-grade risk).** Same machinery on monophonic folk
  tunes in ABC notation - the class *listens* to melodies the net invented. Highest emotional
  payoff, highest data-sourcing risk (machine-readable Armenian folk corpora unverified).

### 8b. Word2vec from scratch on Armenian text

CBOW is literally a one-hidden-layer MLP predicting a word from its neighbors. Train it on a
small Armenian corpus, then show vector arithmetic and neighbor structure emerging - and the
punchline for this course specifically: **this is where the ch9 embeddings came from.** In ch9
the embeddings were someone else's magic; here the students make a small one themselves.
Compute is fine on CPU for a few-MB corpus and a modest vocabulary.

### 8c. Who wrote this? Authorship by compression

Train one tiny char-level MLP per author (Tumanyan, Charents, Raffi - all public domain), then
attribute a held-out paragraph to whoever's model predicts it best (lowest loss). Elegant
twist: classification with no classifier - "the model that compresses you best understands you
best", a quietly deep idea (compression = understanding) taught with three copies of the same
tiny net. Local, literary, cheap.

### 8d. Build your own xG (expected goals) model

StatsBomb's free open event data has shot coordinates and outcomes; an MLP over shot features
(distance, angle, body part, pressure) reproduces the xG number football broadcasts show.
Students can then re-litigate real matches ("was that miss actually a sitter?"). Real industry
model, big appeal for football-minded students. Tabular caveat applies: LightGBM will likely
match it - make that the honest closing comparison, or frame the practical around calibration
(an xG model must output *probabilities*, callback to L13).

### 8f. Can a net learn the rules of a tiny universe?

Show the net Game-of-Life board transitions and ask it to learn the update rule, then roll its
learned rule forward and watch fidelity decay. The surprise is published: this is *hard* for
nets trained by SGD ("It's Hard for Neural Networks to Learn the Game of Life") even though a
tiny exact solution exists - a concrete "representable is not the same as learnable" lesson,
the practical twin of L14's universal-approximation misconception guard. Better as a lab/bonus
than a flagship.

### 8g. Cheap act: confidently wrong on garbage

Feed the trained classifier pure noise or an out-of-domain doodle: it answers with 99%
confidence, because softmax must sum to 1 over known classes. Two cells, lands the
"confidence is not competence" lesson, and calls back to L13 calibration.

### Standing note

Rounds so far: images/personal-data (sec. 4), open-the-box (sec. 5), training phenomena
(sec. 6), other modalities (sec. 7), generative + misc (sec. 8). If 8a lands, a natural
chapter arc is: flagship = name inventor (generative, all chapter tools), plus one surprise
act from sec. 6 (adversarial or random-labels) on a small classifier task inside the same
notebook. References to verify at build time: Bengio et al. 2003 (JMLR, neural probabilistic
LM); Karpathy makemore (github.com/karpathy/makemore); Springer & Kenyon 2020, "It's Hard for
Neural Networks to Learn the Game of Life" (arXiv 2009.01398); StatsBomb open-data repo.
