# Autoencoders chapter (ch8) - master plan

Drafted 2026-07-16. Interview-locked the same day (4 decisions via AskUserQuestion).
Follows the `slide-style` workflow: interview -> outline -> approval -> build.
**Not yet approved for build.** This file + the two OUTLINE.md files are the approval artifact.

## Interview-locked decisions (2026-07-16)

1. **Scope: AE + full VAE derivation.** Plain-autoencoder machinery is the core; the VAE
   half includes the full ELBO / KL / reparameterization derivation in-deck (not deferred
   to GenAI, not intuition-only).
2. **Placement: standalone mini-chapter after RNN** (`ch8_autoencoders`, after `ch7_rnn`).
   Ends pointing into the future GenAI chapter. Does NOT depend on GenAI being planned yet.
3. **Running example: MNIST handwritten digits** (updated 2026-07-16 from sklearn 8x8 digits).
   Same KIND of data as the ch4b dim-reduction deck (handwritten digits), now full MNIST 28x28,
   because we are training a real AE and want a practical students can rerun. The "AE = nonlinear
   PCA" bridge still lands (PCA on MNIST is cheap and real).
4. **Applications featured (all four):** dimensionality reduction / representation learning,
   denoising, anomaly detection, generation / latent sampling.
5. **Kill 2 rabbits (added 2026-07-16):** we train the AE + VAE on MNIST in a **practical
   notebook**, and that notebook's real outputs ARE the deck figures. One artifact serves both the
   lecture (real, reproducible figures) and the student hands-on. This replaces the borrow-and-
   illustrate default for the model-dependent figures. See "Practical notebook" and "Compute
   guardrails" below.

## Two decks

The scope above is too much for one deck. Split, mirroring the RNN chapter's 2-deck shape:

- **L22 Autoencoders** - the deterministic half. Encoder/decoder, the bottleneck, AE = nonlinear
  PCA (dim-reduction app), denoising, sparse, anomaly detection, convolutional AE. Ends on the
  "latent space has holes, you can't sample it" cliffhanger.
- **L23 Variational Autoencoders** - the generative half. Why the AE latent space is not
  generative, the probabilistic reframe, the full ELBO derivation, the reparameterization trick,
  generation / latent interpolation, a one-frame beta-VAE mention, bridge to GenAI.

Numbering: ch7 RNN is L20-L21, so AE = **L22**, VAE = **L23**. If the GenAI chapter takes a
number, it becomes ch9 (folder name TBD; not fixed here).

## The one deep dive

Per the house pattern (one full derivation per chapter - BPTT in ch7, residual math in ch6),
the chapter's single deep derivation is the **VAE ELBO + reparameterization trick** (L23,
Sections 2-3). L22's rigor moment is lighter: the **linear-AE = PCA** equivalence, stated and
shown on the real digits, not a multi-frame proof.

## Course-callback spine (what makes this OUR chapter, not a generic AE intro)

| New concept | Callback to | Framing |
|---|---|---|
| bottleneck / undercomplete code | ch4b PCA | "you already compressed digits - PCA. This is the nonlinear version." |
| linear AE = PCA subspace | ch4b PCA derivation | the bridge frame; same max-variance subspace |
| reconstruction loss (MSE) | regression L2 loss | not new |
| decoder that upsamples | L19 transposed convolution | "the ConvAE decoder is L19's transposed conv" |
| denoising = perturb-and-recover | ch5 L15 data augmentation | robustness by corruption |
| sparse-AE L1 activation penalty | ch1 Lasso / L1 | "L1 again, now on activations" |
| anomaly = high reconstruction error | ch3 metrics / thresholding | reconstruction error as a score, threshold it |
| VAE KL term | ch5 weight decay / regularization | "the KL is a prior on the code - regularization" |
| reparameterization (differentiable sampling) | ch5 backprop | "make the random node differentiable so gradients flow" |
| sampling the latent -> new data | ch6 L19 GAN/style-transfer showcase, GenAI chapter | the on-ramp to generative models |

## Sources

- **LMU I2DL `ae/` decks** (`deep_learning/_reference/lecture_i2dl/slides/ae/`, CC BY 4.0) -
  undercomplete / sparse / denoising / contractive / ConvAE / manifold / VAE. Adapt figures
  with an attribution line, exactly like ch6/ch7 borrowed LMU figures.
- **Jeremy Jordan, "Introduction to autoencoders"** (jeremyjordan.me/autoencoders) - conceptual
  spine for L22 (bottleneck story, undercomplete/sparse/denoising).
- **Lilian Weng, "From Autoencoder to Beta-VAE"** (lilianweng.github.io/posts/2018-08-12-vae) -
  the reference for L23's ELBO / KL / reparameterization / beta-VAE.
- **"How Autoencoders Outperform PCA" (TDS)** - the AE = nonlinear PCA bridge material.
- **Gregory Gundersen, "The Reparameterization Trick"** - clean single-topic explainer.
- **tayden VAE Latent Space Explorer** (tayden.github.io/VAE-Latent-Space-Explorer) - the live
  interactive latent-grid demo for L23. **Perishable: verify it renders at delivery** (JS SPA;
  a static fetch only returns the app shell). Backup: ekzhang/vae-cnn-mnist.

## Practical notebook (co-deliverable, added 2026-07-16)

`ml/ch8_autoencoders/practical/` - a PyTorch notebook that trains, on MNIST:

- a plain AE (reconstructions, 2D latent scatter, the AE-vs-PCA comparison),
- a denoising AE (noisy -> clean),
- an anomaly-detection pass (reconstruction-error histogram, train-on-normal),
- a VAE (2D latent grid, samples from `z ~ N(0,I)`, interpolation).

It is BOTH the student hands-on and the figure source for L22/L23. Model-dependent figures are
exported from this notebook (real, reproducible); it saves them straight into the sibling `fig/`
folders so the decks embed real outputs, not borrowed ones. Format (`.ipynb` vs `.qmd`) - confirm;
lean `.ipynb` to match the ch5 NN practical.

## Compute guardrails (HARD - reconciles "no big models / no long trainings" with the MNIST practical)

The instructor's rule stands: **no big-model downloads, no long trainings.** MNIST + a small AE is
neither, and is the approved opt-in (2026-07-16). Concretely:

- **Data:** MNIST only (~11 MB, one-time download). No ImageNet/CIFAR, no pretrained weights, no
  model-hub downloads. Download source: torchvision, with a `sklearn.fetch_openml('mnist_784')`
  fallback if the torchvision mirror is flaky. Optionally subsample to ~20k train for speed.
- **Three small models only** (self-review 2026-07-16 - reuse to avoid a 4th training):
  1. **plain AE** trained on digits **0-8**, latent dim 2 - serves reconstructions, the PCA-vs-AE
     scatter, the interpolation cliffhanger, AND anomaly detection (digit 9 = held-out anomaly).
  2. **denoising AE** - one variant training run for the denoising figure.
  3. **VAE**, latent dim 2 - one model; its blurry samples ARE the L23 "honest limits" point, so no
     second larger VAE. Architecture: MLP `784-256-64-2` and mirror (or a tiny ConvAE).
- **Freeze-safety (HARD):** `torch.set_num_threads(4)` so training does NOT peg all cores (the
  documented lock-up risk), tiny nets, `<= 8` epochs, batch 256, models trained **sequentially**
  (never in parallel). Expected ~2-4 min per model, ~10 min total. Seed 509.
- Fallback if still too heavy: sklearn's bundled 8x8 digits (zero download, seconds).
- **State the load and get the go before the first training run at build time** (ASK-FIRST-on-compute
  rule). Outlining/authoring the notebook does NOT train; execution is a separate, announced step.

## Figure + honesty rules (HARD)

- **Model-dependent figures come from the practical** (real, reproducible) and carry no external
  credit. Non-model figures (manifold schematic, hourglass, reparameterization TikZ, generative-
  landscape table) are matplotlib/TikZ. Any borrowed image (e.g. a beta-VAE disentanglement figure)
  keeps a "Source: ..., CC BY 4.0" line.
- **Anything illustrative is labeled illustrative** (per the ch6/ch7 discipline - no invented number
  presented as measured). Real practical outputs are labeled as MNIST results.
- Non-model figure scripts in sibling `py_src/`, output to sibling `fig/`, run with the `ma` venv.
  Logs to `logs/`. Seed 509. PCA-on-MNIST is cheap and real (no training) - fine in `py_src/`.

## Deferred to the GenAI chapter (ch9), NOT this chapter

GANs (full treatment - only the L19 showcase exists), diffusion models, the full generative-model
taxonomy, VQ-VAE, normalizing flows, and the attention/transformer thread from L21. L23 ends by
pointing at all of this. Note: L21's attention cliffhanger and L23's VAE-to-generative bridge both
converge on the GenAI chapter - the AE chapter is a parallel unsupervised branch, it does NOT
pretend to continue the attention thread.

## No homework in the decks

Per style guide, HW lives on the chapter `.qmd` page (written later, not part of this build).

## Open questions for the instructor (answer before build)

1. **RESOLVED 2026-07-16: train a small AE + VAE on MNIST, wired as a practical notebook** whose
   outputs are the deck figures ("kill 2 rabbits"). Light per the compute guardrails above. Confirm
   the practical file format (**lean `.ipynb`** to match the ch5 NN practical) and confirm the
   training run at build time before it executes.
2. **ELBO KL closed form** (`KL(N(mu,sigma^2) || N(0,I)) = 1/2 * sum(mu^2 + sigma^2 - log sigma^2 - 1)`):
   derive it fully, or state it and put the derivation in a backup/appendix frame? **Lean: state
   it, appendix the derivation** - keep the main-line derivation on the ELBO + reparameterization.
3. **beta-VAE:** keep the one-frame disentanglement mention in L23, or cut it as GenAI-chapter
   material? **Lean: keep, one frame, no math.**
4. **Confirm 2 decks** (L22 AE + L23 VAE) vs cramming into 1. **Strong recommendation: 2.**
5. **L21 attention cliffhanger now has ch8 between it and its payoff.** You chose standalone
   placement, so this is expected - just confirming you are fine with AE sitting between the RNN
   attention tease and the GenAI payoff.

## Build status

- [x] Plan + both outlines approved (instructor "go for it", 2026-07-16)
- [x] Figures generated - `practical/train_ae_vae.py`, 10 real MNIST figures in `fig/` (3 AE
      variants + 1 VAE, seed 509, thread-capped, ~50s). Reviewed twice; v2 improvements applied
      (recon-PCA-vs-AE, AE-vs-VAE contrasts, anomaly threshold+why panel).
- [x] L22 built - `L22_autoencoders.tex`, 21 pages, 0 errors, aux cleaned.
- [x] L23 built - `L23_vae.tex`, 20 pages, 0 errors, aux cleaned. Full ELBO derivation +
      reparameterization trick.
- [x] Practical notebook built - `practical/ae_vae_practical.ipynb`, executed with the `ma`
      kernel, 6 plots baked in, 0 errors.
- [ ] cnn.qmd-style chapter page (rnn.qmd analogue) + `_quarto.yml` registration - NOT done
      (deferred; no HW per chapter scope).
- [ ] Instructor review of the two PDFs.

## SAE expansion + HW1 (2026-08-02)

Instructor asked to deepen the sparse-autoencoder coverage and add a mech-interp style exercise
using RNNs (students have not seen transformers). Three choices confirmed by interview: synthetic
task with exact ground truth, a full ~9-frame L22 section, and a task/solution homework pair.

### What L22 gained

A new `\section{Sparse autoencoders and interpretability}` between anomaly detection and the
recap, so SAE-for-interpretability lands as application #4. The existing one-frame sparse AE
(L1-as-regularizer) stays where it is and now sets up the payoff. Nine frames: read-a-neuron
(predict-first), why-not (axis alignment + superposition), the activations-are-the-data reframe,
wide-not-narrow + the dictionary, scored results, the sparsity knob, ablation vs control, scale +
honest limits.

**Superposition ownership.** `ch9_attention/ATTENTION_CHAPTER_PLAN.md` never mentions superposition,
though `fig/borrowed/3b1b/README.md` stages JL-lemma stills "for L26 superposition". It was
unowned, so ch8 takes the phenomenon and the tool. The geometry is a **callback to
`math/Lectures/curse_of_dimensionality/cod.tex:125`**, which already teaches near-orthogonality
and names LLM superposition, and which has been delivered (a `cod_notes.pdf` exists). ch9 can
still do the JL-lemma visuals without collision.

### The homework: `HW1_sae_rnn[_solution].ipynb`

A GRU (hidden 32) learns to spell words from a 40-word lexicon; an SAE is trained on its harvested
hidden states; features are read, scored against exact ground truth, and causally ablated.

### MEASURED (all from py_src/sae_rnn_lab.py, seed 509, logs/sae_rnn_lab.log)

1. **The headline.** best single neuron F1 **0.567** | best SAE feature **0.730** | linear probe
   **0.986**. The probe is supervised and is the ceiling, NOT a rival - stated explicitly on the
   slide and in the notebook, because "SAE < probe" otherwise reads as failure.
2. **Interpretability peaks; reconstruction does not.** Sweeping lambda at width 512:
   L0 108.9 -> F1 0.701 | **L0 59.0 -> F1 0.754 (peak)** | L0 37.1 -> 0.743 | L0 13.2 -> 0.730 |
   L0 5.7 -> 0.703 | L0 3.8 -> 0.665 | L0 2.7 -> 0.615. Variance explained falls monotonically
   0.999 -> 0.943. So the loss cannot choose lambda: minimizing it pushes toward a dense,
   uninterpretable code. The first sweep stopped at lambda=0.05 and showed no peak - the slide
   claim was only earned after extending the sweep to lambda=0.01.
3. **Causal ablation is clean, but only if you ablate the right kind of feature.** Feature #83
   detects "letter 4 of `kizjon`" at F1 **1.000**. Ablating it: P(correct next char)
   **0.935 -> 0.317**. Random-feature control: 0.935 -> 0.935.
   **The trap:** selecting the feature on a WORD-level concept ("somewhere inside `fehhzk`")
   gives a drop of **0.002** against a 0.000 control - which reads as a clean null result, not as
   a badly chosen feature. Word identity is not what decides the next character; position within
   the word is. Both the lab and the notebook now score `(word, offset)` concepts for the causal
   test. Filtering to positions where the feature is actually ON was tried first and did NOT fix
   it (0.001 -> 0.002), so the fix is the concept definition, not the position mask.
4. **The negative result, kept on purpose.** The first design used nesting depth in a bracket
   language. It FAILED: depth and stack-top are dense (20-50% of positions), and the SAE lost to a
   single neuron at every lambda. Sparse dictionary learning needs sparse concepts. This is now
   Part 7 of the notebook ("when SAEs do not help") rather than being discarded - most SAE
   tutorials skip it.

### Gotchas worth not re-learning

- An SAE needs **sparse** concepts. Dense ones (a depth counter, a parity bit) give it nothing to
  exploit and it quietly underperforms a single neuron - failing with plausible features, not an
  error.
- The RNN must actually learn, or there is nothing to interpret. The bracket lab's first RNN sat
  at loss 1.946 vs 2.079 for uniform guessing. `sae_rnn_lab.py` now computes an oracle/stack-blind
  entropy floor and RAISES if the model has not beaten the blind baseline.
- Score only where the label is knowable (offset >= 2). At offset 0-1 several words share a prefix,
  so the question is unanswerable and every method looks bad for the wrong reason.
- Unit-norm the decoder columns every step, or the net shrinks the code to dodge L1 and grows the
  decoder to compensate.

## Support files in `practical/`

- `train_ae_vae.py` - the figure engine (source of truth for the 10 `fig/*.pdf`). Rerun to
  regenerate figures.
- `build_notebook.py` - assembles + executes `ae_vae_practical.ipynb`.
- `_contact_sheet.py` - review helper (montage of the figures).
