# L22 Autoencoders - outline (for approval)

Deck 1 of the ch8 autoencoders chapter. The deterministic half. Digits spine.
House style per `ml/SLIDE_STYLE.md`. **Not yet approved.**

Subtitle idea: "A network that learns to copy - and why the copy is the point."

Figures: REAL outputs from the chapter practical (small AE trained on MNIST, see
`AE_CHAPTER_PLAN.md`), plus real PCA-on-MNIST and small TikZ/matplotlib schematics. Model-dependent
figures are exported from the practical notebook into `fig/`. Any borrowed image keeps an
attribution line; illustrative schematics are labeled illustrative.

---

## Cold open (before Outline)

1. **The 784 -> 2 -> 784 challenge.** Show one MNIST digit (28x28 = 784 numbers). "Can a network
   squeeze this digit down to 2 numbers and rebuild it - with no labels, ever?" Reveal the
   answer is basically yes. Hook: the network supervises itself - the input IS the target.
   Bridge from ch5-ch7 (all supervised) to unsupervised representation learning.
   `[predict-first]` `[fig: one MNIST digit + its 2-number code + reconstruction, REAL from practical]`

## Outline frame (`\tableofcontents`)

## Section 1: The autoencoder idea

- `[plain]` transition: "Autoencoders" + "Learn to rebuild your input, and the bottleneck does the teaching."
- **Encoder, code, decoder.** `f: x -> z` (encoder), `g: z -> xhat` (decoder), trained so
  `xhat approx x`. No labels: self-supervised. `[TikZ hourglass schematic]`
- **The bottleneck is the whole point.** Undercomplete: `dim(z) < dim(x)` forces the net to keep
  only what matters. Remove the bottleneck (`dim(z) >= dim(x)`) and it just learns the identity -
  a perfect, useless copy. `[armred trap box: "no bottleneck = it cheats"]`
- **The loss is not new.** Reconstruction error: MSE for real-valued pixels (or binary
  cross-entropy for [0,1] pixels). `[callback: regression L2 loss]`
- **You already built the decoder.** For images the decoder that upsamples the code back to a
  picture is L19's transposed convolution. `[callback: ch6 L19]` `[armblue key box]`
- **Convolutional AE (one frame, folded here).** For images: encoder = conv + pool (ch6),
  decoder = transposed conv (L19). Same net, mirrored. `[callback: ch6 L16-L19]`
  `[TikZ mirrored conv/transposed-conv]`

## Section 2: Autoencoders = nonlinear PCA  (application: dimensionality reduction)

- `[plain]` transition: "Nonlinear PCA" + "You already compressed digits once - this bends the axes."
- **The bridge, stated.** A linear AE (identity activations) trained with MSE recovers the SAME
  subspace PCA found in ch4b - max-variance directions. `[callback: ch4b PCA]` `[armblue key box]`
- **Add nonlinearity and it bends.** Nonlinear activations let the AE follow curved manifolds PCA
  cannot. `[predict-first: "PCA on this S-curve - how many components to unroll it?" -> it can't;
  an AE can]` `[fig: S-curve / swiss-roll, PCA projection vs AE projection, matplotlib illustrative]`
- **Payoff on MNIST.** AE 2D latent scatter colored by class, side by side with the PCA-on-MNIST
  scatter (both real). Same digit data as ch4b, now full MNIST. `[fig: pca_vs_ae_mnist - PCA side
  real (cheap), AE side REAL from practical, labeled]`
- **Honest caveat.** AE latent axes are not orthogonal, not variance-ordered, not unique (another
  run -> a different latent), no explained-variance ranking. PCA gives you those for free; the AE
  trades them for flexibility. `[armorange watch-out]`

## Section 3: Regularized autoencoders - denoising and sparse  (application: denoising)

- `[plain]` transition: "When you can't just shrink the code" + "Two other ways to stop an AE from cheating."
- **The overcomplete problem.** If you want a big code (`dim(z) >= dim(x)`) the undercomplete
  trick is gone - it can copy again. You need a different constraint. Two classics: denoising, sparse.
- **Denoising AE.** Corrupt the input (add Gaussian noise, or zero out random pixels), train the
  net to output the CLEAN original. It cannot copy noise, so it must learn what a digit really is.
  `[callback: ch5 L15 data augmentation - robustness by perturbation]`
  `[predict-first: "zero out half the pixels and ask for the clean digit back - does that really
  help?" -> yes]`
  `[fig: noisy MNIST digits -> denoised, REAL from the denoising AE in the practical]`
- **Sparse AE (one frame).** Instead of shrinking the code, penalize its activations (L1, or a
  KL-sparsity target) so only a few code neurons fire per input. `[callback: ch1 Lasso / L1]`
  `[TikZ: few-neurons-fire schematic]`

## Section 4: Anomaly detection  (application: anomaly detection)

- `[plain]` transition: "Anomaly detection" + "If it can't rebuild you, you don't belong."
- **Reconstruction error as a score.** Train the AE on normal data only. It reconstructs normal
  inputs well (low error) and anomalies badly (high error). Threshold the error -> flag anomalies.
  `[callback: ch3 metrics / thresholding, ROC]`
- **Show the separation.** Train the plain AE on digits 0-8 (so it doubles as the Section 1-2
  model); digit 9 is the held-out anomaly. `[fig: anomaly_recon_error_hist - reconstruction-error
  histogram, 0-8 (normal) vs 9 (anomaly); REAL from practical, labeled]`
- **Why it matters.** Fraud, manufacturing defects, network intrusion - the industrial workhorse
  use of AEs, and interview-relevant. `[armblue key box]`

## Recap + Next

- **Recap:** encoder/decoder + bottleneck; AE = nonlinear PCA; denoising = corrupt-and-recover;
  sparse = penalize activations; anomaly = reconstruction error.
- **The cliffhanger (motivates L23).** Take two digits' codes, average them, decode the average.
  You get garbage, not a digit halfway between. The latent space has HOLES - it is organized for
  reconstruction, not for generation. You cannot pick a random `z` and get a valid digit.
  `[fig: interpolate two MNIST codes -> broken midpoint, REAL from the plain AE in the practical]`
  `[paramgreen Next box: "Next: give the latent space no holes, then sample it - variational
  autoencoders."]`

---

## Figures

**From the practical notebook (REAL, MNIST, exported into `fig/`):**
1. `ae_recon_teaser.pdf` - one digit + 2-number code + reconstruction (cold open).
2. `pca_vs_ae_mnist.pdf` - side-by-side 2D latent: PCA-on-MNIST (real, cheap) vs AE latent (real).
3. `denoising_grid.pdf` - noisy -> denoised MNIST digits.
4. `anomaly_recon_error_hist.pdf` - reconstruction-error histogram, normal vs anomaly.
5. `code_interpolation_broken.pdf` - average of two AE codes decodes to garbage (cliffhanger).

**From `py_src/` (matplotlib, no training):**
6. `manifold_pca_fails.pdf` - S-curve / swiss-roll, PCA projection vs an unrolled version.

**TikZ (small throwaway visuals only):** hourglass AE; sparse "few neurons fire"; mirrored
conv/transposed-conv ConvAE.

## Provenance / attribution

Practical-derived figures are our own MNIST outputs (no external credit; labeled as MNIST results).
Any borrowed image carries a "Source: ..., CC BY 4.0" line; schematics labeled illustrative.
`% Provenance:` block at the end of the `.tex`.
