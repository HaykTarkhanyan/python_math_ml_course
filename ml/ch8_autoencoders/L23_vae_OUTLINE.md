# L23 Variational Autoencoders - outline (for approval)

Deck 2 of the ch8 autoencoders chapter. The generative half. MNIST spine.
House style per `ml/SLIDE_STYLE.md`. **Not yet approved.**

Subtitle idea: "Give the latent space no holes, then sample it."

The chapter's ONE deep dive lives here: the full ELBO derivation (Section 2) + the
reparameterization trick (Section 3).

Figures: from the real small MNIST VAE trained in the chapter practical notebook
(light, opt-in approved 2026-07-16 - small net, few epochs, CPU). Reparameterization and
ELBO diagrams are TikZ. See `AE_CHAPTER_PLAN.md` for the compute guardrails.

---

## Cold open (before Outline)

1. **Resume the L22 cliffhanger.** The plain AE could not decode the midpoint of two codes.
   Show our own VAE latent grid (real, from the practical): every point decodes to a valid digit,
   and neighbors morph smoothly. "What if the whole latent space were like this? Then a random draw
   would GENERATE a new digit." Hook. `[fig: smooth VAE latent grid, REAL from our practical]`
   `[link: tayden explorer as the interactive version, verify live at delivery]`

## Outline frame (`\tableofcontents`)

## Section 1: From autoencoder to generative model

- `[plain]` transition: "The generative gap" + "An AE memorizes points. We want a space we can sample."
- **Why the plain AE cannot generate.** It maps each `x` to a single POINT `z`; the gaps between
  training points are undefined; a random `z` lands in a hole -> garbage. `[callback: L22 cliffhanger]`
- **The fix in one line.** Make the encoder output a little Gaussian BLOB per input (a `mu` and a
  `sigma`), not a point; and push all the blobs to fill a standard normal `N(0, I)` with no gaps.
  Then `z ~ N(0, I) -> decode -> valid digit`. `[TikZ: point-code vs blob-code]`
- **Two new ingredients (preview).** (1) encoder predicts `mu(x), sigma(x)`; (2) a regularizer that
  pulls every blob toward `N(0, I)`. The next two sections make each precise. `[armblue key box]`

## Section 2: The VAE objective - the ELBO  (DEEP DIVE, full derivation, 2-3 frames)

- `[plain]` transition: "The objective" + "Where the loss comes from - honestly."
- **Setup: a latent-variable model.** `p(x) = integral p(x|z) p(z) dz`, prior `p(z) = N(0, I)`,
  decoder `p(x|z)`. Problem: that integral is intractable, so we cannot maximize `log p(x)` directly.
- **Variational inference.** Introduce `q(z|x)` (the encoder) to approximate the true posterior
  `p(z|x)`. `[full derivation, boxed]`
- **Derive the ELBO step by step.**
  `log p(x) = E_q[log p(x,z) - log q(z|x)] + KL(q(z|x) || p(z|x))`. The KL is `>= 0`, so
  `log p(x) >= ELBO`, with `ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))`. `[full derivation, boxed]`
- **Read the two terms.** Term 1 `E_q[log p(x|z)]` = "rebuild `x` well" = the reconstruction loss
  (the L22 AE loss). Term 2 `- KL(q(z|x) || p(z))` = "keep each blob close to `N(0, I)`" = the
  no-holes regularizer. `[armblue key box]` `[callback: ch5 weight decay - the KL is a prior on the
  code, i.e. regularization]`
- **The Gaussian KL, closed form (stated; derivation -> appendix frame).**
  `KL(N(mu, sigma^2) || N(0, I)) = 1/2 * sum_j (mu_j^2 + sigma_j^2 - log sigma_j^2 - 1)`.
  So the whole VAE loss is computable and differentiable... except one thing. `[hook to Section 3]`

## Section 3: The reparameterization trick  (DEEP DIVE, part 2)

- `[plain]` transition: "Backprop through randomness" + "You cannot differentiate a coin flip. Unless."
- **The blocker.** The loss needs `z ~ N(mu, sigma^2)`, but sampling is stochastic - there is no
  gradient through a random node, so backprop stops. `[predict-first: "we need a derivative through
  a random draw - how?"]`
- **The trick.** Write `z = mu + sigma (*) epsilon`, with `epsilon ~ N(0, I)`. The randomness now
  lives in an INPUT `epsilon`; `mu` and `sigma` are deterministic, so gradients flow through them.
  `[TikZ before/after: stochastic node blocking gradient vs reparameterized path]`
- **Why this matters.** With reparameterization the VAE trains end to end with plain SGD/Adam - the
  same optimizer from ch5, nothing exotic. `[callback: ch5 backprop / autograd]`

## Section 4: VAE in action  (application: generation / latent sampling)

- `[plain]` transition: "Generation" + "Sample the prior, decode, done."
- **Generate.** `z ~ N(0, I) -> decode -> a brand-new digit that was never in the data.`
  `[fig: grid of VAE-generated digits, REAL from our practical, labeled]`
- **Walk the latent space.** The 2D latent grid: sweep `z` and watch digits morph continuously
  (3 -> 8, thin -> thick). This is exactly what the plain AE could not do. `[fig: 2D latent grid,
  REAL from our practical]` `[link: tayden explorer]`
- **Interpolate.** Encode two real digits, walk the straight line between their codes, decode each
  step: every midpoint is a valid digit now. `[fig: interpolation strip, REAL from our practical]`
  `[callback: L22 broken-midpoint cliffhanger - now fixed]`
- **beta-VAE, one frame, no math.** Turn up the KL weight `beta` and the latent axes tend to
  disentangle into interpretable factors (rotation, thickness, width). One example figure.
  `[fig: beta-VAE factor sweep, borrowed w/ attribution or illustrative]`

## Section 5: Honest limits + the bridge to GenAI

- `[plain]` transition: "Where VAEs stop" + "Principled, but blurry."
- **VAE samples are blurry.** The Gaussian decoder + reconstruction loss average over plausible
  outputs -> soft, smeared digits. `[armorange watch-out]`
- **The generative landscape.** One comparison frame: VAE (principled latent + a likelihood, but
  blurry) vs GAN (sharp, no likelihood, unstable - the L19 showcase) vs diffusion (today's SOTA for
  images). `[fig or table: VAE vs GAN vs diffusion, one row each]` `[callback: ch6 L19 GAN/style
  showcase]`
- **Recap** (VAE = AE + a distribution per input + a KL prior; ELBO = reconstruction - KL;
  reparameterization makes it trainable; sample the prior to generate).
- `[paramgreen Next box]`: "Next: the GenAI chapter - attention and transformers (resuming L21),
  GANs, diffusion, and LLMs. Both roads - the attention cliffhanger from L21 and this VAE bridge -
  meet there."

---

## Figures (from the real MNIST VAE in the practical notebook; TikZ for the two diagrams)

1. `vae_latent_grid.pdf` - 2D latent grid of decoded digits (cold open + Section 4). REAL.
2. `vae_samples.pdf` - grid of generated digits from `z ~ N(0, I)`. REAL.
3. `vae_interpolation.pdf` - encode-two, walk-the-line, decode strip. REAL.
4. `beta_vae_factors.pdf` - a disentanglement factor sweep. Borrowed (attribution) or illustrative.
5. `gen_landscape.pdf` (or a table) - VAE vs GAN vs diffusion one-liner comparison.
6. TikZ: point-code vs blob-code; reparameterization before/after.

## Provenance / attribution

Real figures from our practical carry no external credit (our own outputs); borrowed beta-VAE or
comparison images carry a "Source: ..., CC BY 4.0" line. `% Provenance:` block at the end of the `.tex`.

## Open questions (also in AE_CHAPTER_PLAN.md)

- KL closed-form: main-line or appendix (lean: appendix).
- beta-VAE frame: keep (lean: keep, one frame; borrow the disentanglement figure, do NOT train a
  beta-VAE).
- VAE latent dim: **single 2-D VAE** (self-review 2026-07-16). The 2-D grid is the payoff and its
  blurry samples ARE the Section 5 "honest limits" point - no second larger VAE.
