# Diffusion chapter (ch10) - plan + outline

Drafted 2026-08-03. Scope interviewed and locked 2026-08-03.
**Revised 2026-08-03 after an adversarial review** - see "Review corrections" at the end for what
was wrong in the first draft and why. The outline is **awaiting approval**; nothing is built yet.
Follows the house new-chapter workflow (interview -> outline -> approval -> build).

## Mission

Pay off a cliffhanger the course set two chapters ago. **L23 (VAE)** ended on an honest admission:
VAE samples are blurry, and the closing slides named diffusion as today's SOTA for images without
showing why. This chapter shows why, and does not hand-wave the mechanism.

The arc: *we can destroy an image in closed form -> so what exactly should a network learn to undo
it? -> here is the loss, derived, not asserted -> now generate -> now say what to generate -> now
make it affordable enough to be real.*

**Prerequisite spine is L22/L23 (autoencoders, VAE, the ELBO, reparameterization), not L24-L26.**
That independence is why this is its own chapter (root `DECISIONS.md` #1) - and it is now a hard
requirement, because **L25 and L26 do not exist yet**: `ml/ch9_attention/` contains only L24. See
decision 6 below.

## Interview-locked decisions (2026-08-03)

1. **Standalone chapter `ml/ch10_diffusion/`,** and `ch9_genai` renamed to `ch9_attention`.
   (`DECISIONS.md` #1.)
2. **Full derivation.** The ELBO -> KL -> L2 -> epsilon-prediction chain is worked properly.
   Instructor's words: *"lets build full one, and maybe later make it smaller"*.
3. **Scope runs through latent diffusion and video.** (`DECISIONS.md` #3.)
4. **Five decks, L27-L31.**
5. **Two complementary sources**, both fetched at 1080p. (`DECISIONS.md` #2.)
6. **L30 is self-contained on cross-attention** (added post-review). The original plan made L30
   depend on "L24/L25", but L24 contains no cross-attention and L25/L26 are unwritten. Rather than
   block this chapter on ch9's backlog, L30 builds cross-attention in one frame from the Q/K/V
   machinery L24 *does* teach (`L24_attention.tex:534-552`, dot-product-as-similarity). If L26 ships
   first, that frame becomes a recap instead.

### Honesty note on the deck count

The option presented at interview said "~4 lectures"; the plan is **5**.

**The merge-to-4 fallback in the first draft is withdrawn.** It claimed L27+L28 would run "~32
frames"; the real arithmetic is 17 + 30 = 47 numbered frames, ~56 pages once mandatory `[plain]`
and Outline frames are counted. That is two lectures wearing one hat, not a compression. If the
calendar forces a cut, cut *scope* (drop L31's flow-matching and video sections and end the chapter
at latent diffusion), not deck boundaries.

## Sources

| Source | What it gives us | Location |
|---|---|---|
| **Welch Labs / 3Blue1Brown**, "But how do AI images and videos actually work?" (37:20, Jul 2025) | The **intuition spine**: the 2D spiral, score as a time-varying vector field, DDIM, CLIP, classifier-free guidance, negative prompts, video. 52 frames + transcript. **Read its `description.txt` technical notes - they correct the video on three points we teach.** | `_reference_welchlabs_diffusion/` |
| **Deepia**, "Diffusion Models: DDPM" (32:05, May 2025) | The **derivation spine**: VE vs VP forward process, NLL -> ELBO -> KL -> L2 -> epsilon, plus a full PyTorch train/sample implementation. 42 Manim frames + transcript. | `_reference_deepia_ddpm/` |
| **Sohl-Dickstein et al. 2015**, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" | **The actual origin of diffusion models and of the name.** Deepia opens by making this point; the first draft of this plan omitted it. | arXiv 1503.03585 |
| DDPM (Ho, Jain, Abbeel 2020) | Algorithms 1 and 2; the supplementary has the full ELBO algebra. | arXiv 2006.11239 |
| DDIM (Song et al. 2020) | Deterministic sampling, derived from a **non-Markovian forward process** - not from an SDE. | arXiv 2010.02502 |
| Score-SDE (Song et al. 2021) | The probability-flow ODE; the post-hoc link to DDIM; VE and VP as two members of one family. | arXiv 2011.13456 |
| CFG (Ho & Salimans 2022), unCLIP (Ramesh et al. 2022), CLIP (Radford et al. 2021) | Guidance, DALL-E 2, the shared embedding space. | arXiv 2207.12598, 2103.00020 |
| Latent Diffusion (Rombach et al. 2022) | Stable Diffusion - **the L23 VAE, reused as a compression layer**. | arXiv 2112.10752 |
| Flow matching (Lipman et al. 2022; Liu et al. 2022, rectified flow) | What SD3 / Flux / Wan actually use. | arXiv 2210.02747, 2209.03003 |
| `smalldiffusion`, `ytdeepia/DDPM` | Small teaching implementations; realistic basis for a homework. | github |

**Citation rule for this chapter:** every frame that introduces a method carries author+year inline,
per `SLIDE_STYLE.md:83`. The first draft's outline carried none.

---

## Deck outline (5 decks; page counts include `[plain]` + Outline frames)

Tags: `[plain]` = section-transition frame; `[predict-first]` = question + pause + reveal;
`[fig]` = generated matplotlib from `py_src/`; `[wl]` / `[dp]` = borrowed still; `[tikz]` = our
diagram; `[callback: LNN]` = **verified** course callback. Per `SLIDE_STYLE.md`, every essential
figure is Python-generated; borrowed stills only where redrawing adds nothing.

**Acronym discipline:** DDPM, DDIM, CFG, CLIP, VE, VP, SDE, ODE, FFHQ, FID, SOTA all need
expansion on first use (`SLIDE_STYLE.md:85-101` exempts only ELBO/KL/MSE). Run the guide's acronym
grep before any deck is called done. Section titles must not be bare acronyms.

### L27 - Diffusion: destroying an image on purpose (~21 pages)

**Cold open**
1. Title + **the L23 payoff** `[dp f42]` `[callback: L23 - verified, L23_vae.tex:454 calls VAE
   samples blurry]` - FFHQ diffusion samples beside VAE samples.
2. `[predict-first]` Where does a generated video *start*? `[wl f03]` Reveal: a random number
   generator, then 50 passes of a network `[wl f04-f05]`

**Section 1 - Generation is sampling** `[plain]`
3. p(x) as the distribution of natural images - no formula exists, but we want to draw from it `[fig]`
4. The diffusion bet: corrupt data into noise in a way we fully control, then learn to undo it.
   **Sohl-Dickstein et al. 2015** `[tikz]`

**Section 2 - The forward process** `[plain]`
5. One step: q(x1|x0) = N(x0, beta I) - "adding noise" *is* sampling from a Gaussian centred on x0 `[tikz]`
6. Noise accumulates, so we can jump: q(xt|x0) = N(x0, t*beta I) `[dp f06]`
7. `[predict-first]` Does that converge to N(0,I)? Reveal: **no.** Mean pinned at x0, variance
   unbounded - **variance exploding (VE)** `[fig ve_vs_vp]`
8. The fix: shrink the mean by sqrt(1-beta) each step. **Write the result explicitly:**
   q(xt|x0) = N(sqrt(alpha-bar_t) x0, (1 - alpha-bar_t) I), with alpha-bar_t = (1-beta)^t
   **for constant beta** `[fig]`
9. **Variance preserving (VP)**: mean -> 0, variance -> 1 - *given data scaled to [-1, 1]*, which is
   why the name means anything and is a real implementation requirement `[fig ve_vs_vp]`
10. **Honesty frame:** VE is not a mistake. It is what score matching (Song & Ermon 2019) uses, and
    both are members of one SDE family (Song et al. 2021). DDPM chose VP because it lands on a fixed
    N(0,I) you can sample from.
11. Watch it happen in 1D: a bimodal p(x) flattening into a standard normal `[fig forward_1d]`
12. Noise schedules - constant vs linear vs cosine; alpha-bar as a **product** once beta varies
    `[fig noise_schedules]`
13. **Worked numbers** (`SLIDE_STYLE.md:82`): beta = 0.02; compute sqrt(alpha-bar) at t = 10 / 100 /
    1000 and watch the surviving signal fraction collapse `[fig]`
14. **Why this matters:** any xt in one shot, no loop. This is what makes training affordable.

**Section 3 - Turning around** `[plain]`
15. Define the reverse step p_theta(x_{t-1}|x_t); theta is a network `[tikz forward/reverse loop]`
16. `[predict-first]` Just train it to undo one step - what could go wrong? Reveal: **it does not
    work well, and virtually no modern model does it** (Welch, 00:09:00). *Why* is L28's job. `[wl f15]`
17. Close + **Recap + paramgreen "Next:" box** (`SLIDE_STYLE.md:64`)

> The spiral toy dataset has been **moved out of L27** into the head of L29, where it is actually
> used. In the first draft it was introduced here and then left idle for a whole deck.

### L28 - The loss: from likelihood to noise prediction (~36 pages)

Re-budgeted from 20 to ~30 numbered frames after review. L23 spent **6 frames** on the *single-step*
VAE ELBO; this deck does that plus the T-step expansion, the posterior, and the reparameterization.

**Cold open**
1. Title + recap: forward is closed-form and parameter-free. Everything unknown is in theta.

**Section 1 - Maximum likelihood, and why it fails** `[plain]`
2. Train by minimizing -log p_theta(x0) `[callback: L23 - verified, L23_vae.tex:200]`
3. Marginalize over the path: an integral over every route from noise to this image `[dp f17]`
4. **Intractable.** Too many paths. We need a bound.

**Section 2 - The bound** `[plain]`
5. Multiply and divide by q, rewrite as an expectation `[dp f18]` `[callback: L23 - verified, the
   same q/q move]`
6. **Derive the bound the way L23 did** - decompose into ELBO + KL(q || p), then drop the KL because
   it is >= 0. **Do NOT use Jensen's inequality here.** L23 used the decomposition route
   (`L23_vae.tex:200-222`) and Jensen appears in no built deck; switching methods while claiming
   continuity is exactly the trap the first draft fell into. `[callback: L23 - verified]`
7. *(optional aside)* Jensen gives the same bound in one line, and they met it in
   `math/17_probability_exp_var_inequalities.qmd` - mention, do not rely on it.
8. Expanded: a prior KL + a sum of KLs + a reconstruction term `[dp f20]`
9. What a KL divergence *is*, visually `[fig]`
10. Drop two terms: one has no theta, one is negligible at t=1 `[dp f23]`

**Section 3 - The true posterior** `[plain]`
11. q(x_{t-1}|x_t, x0) has a **closed form** `[dp f24]`
12. `[predict-first]` Then why train a network? Reveal: it needs x0, which at generation time is
    exactly what we lack. Usable *only during training*.
13. Choose a Gaussian approximate posterior; **fix its variance**, learn only its mean `[dp f25]`
14. Gaussian KL with fixed variance reduces to a squared distance of means - **keep the
    1/(2 sigma_t^2) prefactor on the slide.** It is dropped deliberately at frame 24; if it is
    hidden here, that later step comes from nowhere. `[fig]`

**Section 4 - Reparameterization** `[plain]`
15. The true mean is an awkward blend of x0 and xt `[dp f28]`
16. Reparameterize: xt = sqrt(alpha-bar) x0 + sqrt(1-alpha-bar) eps
    `[callback: L23 - verified, the trick that made the VAE trainable]`
17. Solve for x0 in terms of xt and eps
18. Substitute into mu-tilde (algebra, step 1) `[dp f29]`
19. Apply the same form to mu-theta (algebra, step 2)
20. **The xt terms cancel** (algebra, step 3)
21. **The final loss: || eps - eps_theta(xt, t) ||^2.** The network predicts the noise. `[dp f30]`

> Frames 18-20 were **one frame** in the first draft ("substitute both means; the xt terms cancel").
> That is a full page of algebra and violates one-idea-per-frame (`SLIDE_STYLE.md:66`).

**Section 5 - From objective to practice** `[plain]`
22. Simplification 1: sample **one random t** per example instead of summing over T. Unbiased
    Monte-Carlo estimate of the same objective - free. `[dp f32]`
23. Simplification 2: set all the weights to 1 (**L_simple**).
24. **This one is not free.** L_simple is no longer the ELBO. Ho et al. report better *sample
    quality* and *worse* likelihood. We take the trade knowingly. (Picked up again at L29's FID frame.)
25. **The whole chain on one slide** - NLL -> bound -> KL sum -> L2 -> epsilon `[fig, after dp f31]`

**Section 6 - Payoff: it is twelve lines of code** `[plain]`
26. The training loop, in full `[dp f35]` - "that entire derivation is this"
27. The loss curve `[fig mnist_ddpm]`
28. The per-epoch denoising grid at t = 50 / 100 / 400 / 700 `[fig mnist_ddpm]` `[dp f37]`
29. **What the network actually is** - a UNet: image-to-image, conditioned on t.
    `[callback: L16/L19 conv + transposed conv; L17 skip connections; L22 conv autoencoder,
    verified at L22_autoencoders.tex:264-282]`. The sinusoidal time embedding is ch9's positional
    encoding. **This frame was missing entirely from the first draft** and L30 has nowhere to attach
    cross-attention without it.
30. Why "just an MSE" is a big deal - **and what the alternative costs.** GANs optimize against a
    moving discriminator: no stationary loss to monitor, mode collapse, no convergence signal.
    *(New material - L19 shows StyleGAN faces but never says GANs are hard to train. Verified: no
    "mode collapse" or "instability" content in any built deck. Budget it as teaching, not recall.)*
31. **Recap + "Next:" box**

### L29 - Sampling: why the noise comes back (~25 pages)

**Cold open**
1. Title + recap: we have a trained eps_theta. Generate something.

**Section 1 - The DDPM sampler** `[plain]`
2. Algorithm 2, line by line `[wl f16]`
3. `[predict-first]` **It adds fresh random noise at every step.** Why add noise to a denoiser's
   output? *Announced hold - answered in Section 3.*
4. The implementation is under 30 lines `[dp f38]`
5. Digits emerging from noise `[fig mnist_ddpm]`

**Section 2 - What the model actually learned** `[plain]`
6. Two pixels = one point: an image is a dot in pixel space `[fig spiral]` *(moved here from L27)*
7. Noising is a random walk with a **pull toward the origin** - an Ornstein-Uhlenbeck process, not
   plain Brownian motion. (The sqrt(1-beta) drift from L27 frame 8 is exactly that pull; the
   driftless VE process *is* Brownian.) `[fig]`
8. At every point the model outputs a **direction** `[fig spiral_score_field]`
9. That direction is the **score**, up to sign and scale: score = grad log p(xt) =
   **-eps_theta / sqrt(1 - alpha-bar_t)**. State the identity - the SDE story in Section 4 depends
   on it. `[fig]`
10. Time conditioning: coarse fields at large t, fine structure as t -> 0 `[fig, several t]`
11. The field visibly reorganizes near t ~ 0.4 `[fig]` `[wl f28]`
12. Why predict *total* noise rather than one step: same target, far less variance `[wl f24]`

**Section 3 - Answering the question** `[plain]`
13. One point under DDPM: model step, then noise step, repeat `[fig spiral_ddpm_ddim]`
14. 256 points: chaos, then clean convergence onto the spiral `[fig]`
15. **Now naively delete the noise step.** Everything collapses toward the centre `[fig]`, and on a
    real model you get the blurry tree `[wl f17-f18]`
16. **Why - and this is the frame the first draft got wrong.** The denoiser was trained to expect a
    specific noise level at each t. Delete the sampler's noise and x_{t-1} carries *less* noise than
    sigma_{t-1}, so the network sees an input unlike anything in training and **over-denoises**,
    pointing at the dataset mean. In image space, means look blurry.
    *(Source: `_reference_welchlabs_diffusion/description.txt:52`, a correction credited to Chenyang
    Yuan. The video's own on-screen explanation - "the model learns a Gaussian mean, so you must add
    the spread back" - is the weaker story and is contradicted by DDIM three frames later. Teach the
    noise-level-mismatch version.)*
17. Honesty: the 2D spiral analogy is imperfect here. On the spiral the no-noise samples still land
    *on* the data; in high dimensions they miss the manifold of realistic images entirely
    (Welch, 00:19:03).

**Section 4 - Making it cheap: deterministic sampling (DDIM)** `[plain]`
18. The cost: ~1000 network evaluations per image **under the DDPM sampler** (Stable Diffusion ships
    at ~50 - do not attach 1000 to SD)
19. `[predict-first]` We just proved deleting the noise ruins everything. So how is a *deterministic*
    sampler possible? Reveal: **do not delete the noise - rescale the step so the noise level stays
    consistent.** This is the natural payoff of frame 16.
20. **DDIM** (Song et al. 2020), derived from a **non-Markovian forward process** - no SDE involved.
    Deterministic, far fewer steps, **no retraining** `[fig spiral_ddpm_ddim]`
21. The SDE/ODE picture (Song et al. 2021) came **later** and is a reinterpretation: the probability
    -flow ODE has the same marginals as the SDE, and DDIM is approximately its first-order
    discretization. `[wl f36 - full-bleed; the README warns this panel must not be shrunk]`
22. Two honest caveats: the theory promises the same *distribution*, not the same image; and the
    equivalence holds for the exact continuous solution - 20-step DDIM does **not** match 1000-step
    DDPM, which is why few-step samples have a look.
23. **Evaluation:** how does anyone *know* one sampler is better? **FID** - what it measures and what
    it misses; and the L_simple trade from L28 frame 24 (better samples, worse likelihood) made
    concrete. `[callback: ch3 metrics]` *(New material - FID appears nowhere in `ml/`.)*
24. **Recap + "Next:" box** - "We can generate. We still cannot say *what*." -> L30

### L30 - Text to image: conditioning and guidance (~26 pages)

Budgeted up from 20: CLIP, contrastive learning and zero-shot are **entirely new** - grep confirms
none of them appear in any `ml/` deck. The first draft treated them as a callback to L13b, which
does not mention cosine similarity at all.

**Cold open**
1. Title + the gap: every image so far was unconditional. The model draws *a* face, not your prompt.

**Section 1 - CLIP: one space for words and pictures** `[plain]`
2. Two encoders, one 512-d space, 400M caption pairs (Radford et al. 2021) `[wl f07]`
3. **Contrastive learning** as an idea - pull matches together, push mismatches apart. New concept;
   give it a frame.
4. The batch matrix: diagonal = matching pairs, everything off-diagonal = negatives `[wl f08-f09]`
5. Cosine similarity as the metric `[wl f10]` `[callback: L24 dot-product-as-similarity, verified at
   L24_attention.tex:534-552 - this is the real bridge; L13 names cosine in one bullet
   (32_clustering.tex:101), which is a mention, not a foundation]`
6. The space does **arithmetic**: (me with a hat) - (me without) ~ "hat", similarity 0.165 `[wl f11-f12]`
7. Zero-shot classification falls out for free `[wl f13]`
8. But CLIP only maps *into* the space. It cannot draw.

**Section 2 - Conditioning** `[plain]`
9. Feed the text into the denoiser as an extra input `[wl f41]`
10. **What exactly gets fed in.** Not the pooled 512-d vector - Stable Diffusion uses the layer
    *before* it, a **77 x 512 sequence** (`description.txt:54`). Cross-attention needs a sequence of
    keys and values; a single pooled vector gives nothing to attend over.
11. **Cross-attention, built here** (decision 6): queries from image tokens, keys and values from the
    77 text tokens, into the UNet blocks from L28 frame 29. `[callback: L24 Q/K/V - verified]`
12. Alternatives: concatenation, and conditioning in several places at once `[wl f42]`
13. DALL-E 2 / unCLIP (Ramesh et al. 2022): train a diffusion model to **invert the CLIP image
    encoder** `[wl f39-f40]`
14. `[predict-first]` Is conditioning enough? Reveal: **no.** Prompt asks for a tree; we get a
    desert and a shadow `[wl f43]`

**Section 3 - Classifier-free guidance** `[plain]`
15. Toy version: label the spiral's regions person / dog / cat. The classes bleed `[fig spiral_cfg]`
16. The trick (Ho & Salimans 2022): **drop the label** on a fraction of training examples. One model,
    two behaviours.
17. The conditional and unconditional fields, side by side `[fig]` `[wl f46]`
18. **Write the formula out and name the convention:**
    eps-tilde = eps_theta(xt, t) + alpha * ( eps_theta(xt, t, c) - eps_theta(xt, t) ).
    **alpha = 1 means no guidance.** (A second convention puts the *conditional* prediction as the
    base term; `description.txt:56` flags both. `diffusers` uses this one, default
    `guidance_scale` = 7.5.) The first draft said the difference "replaces" the conditional
    direction, which drops the base term and is wrong. `[fig]` `[wl f47]`
19. The guided trajectory now lands in the right region `[fig]`
20. On a real model: sweep alpha and the tree grows `[wl f50]`
21. **Negative prompts** - subtract what you explicitly do not want `[wl f51-f52]`
22. The honest cost: guidance buys prompt adherence by **spending diversity**. High alpha = same-y
    images.
23. **Recap + "Next:" box** -> L31

### L31 - Making it practical: latent diffusion, flow matching, video (~22 pages)

**Cold open**
1. Title + the arithmetic: 1000 evaluations on 512x512x3.

**Section 1 - Latent diffusion** `[plain]`
2. **"You already know the punchline."** `L23_vae.tex:405-422` is a delivered frame that already
   states 512x512x3 -> 64x64x4, ~48x fewer numbers, and "the VAE became the compression layer inside
   it." Open by recalling it, do not stage it as a reveal. `[callback: L23 - verified]`
3. **What L23 did not say: the latent is spatial.** A 64x64x4 feature map, not a flat code like the
   2-D MNIST latent from L22/L23. Convolutions still work in there; that is the whole trick. `[fig]`
4. Rombach et al. 2022 end to end: encode once, diffuse in the latent, decode once `[fig]`
5. Where the cost actually goes - element count drops 48x, attention cost far more `[fig]`
6. A loose end tied off: this is why the "noise" video in L27's cold open was not salt-and-pepper -
   it was latent noise, decoded (`description.txt:48`).

**Section 2 - Flow matching** `[plain]`
7. Learn the velocity field directly (Lipman et al. 2022; Liu et al. 2022)
8. Straighter paths -> fewer steps `[fig]`
9. What current models use (SD3, Flux, Wan)

**Section 3 - Video** `[plain]`
10. Video is one more dimension on the same tensor, and the same loop `[wl f03-f05]`
11. Wan 2.1's real negative prompt - extra fingers, walking backwards, written in Chinese `[wl f51]`
12. What the output looks like without it `[wl f52]`

**Section 4 - Where this leaves us** `[plain]`
13. The generative landscape, honestly: VAE / GAN / diffusion / flow `[fig gen_landscape]`
    `[callback: L23 - verified, L23_vae.tex:454 has this table already; extend it, do not repeat it]`
14. **Why images and not text?** Diffusion is built on continuous-state Gaussian corruption; tokens
    are discrete. The obvious question after a transformer chapter, and it deserves an answer.
15. **The real cost.** SD 1.x took roughly 150k A100-hours. "Diffusion buys quality with compute" is
    the chapter's thesis; give it a number.
16. **Provenance and misuse.** Training-data sourcing, the copyright suits, deepfakes. A 2026 course
    that ends on Sora with no word on this has taken a position by omission.
17. What we deliberately skipped: score matching, consistency models, distillation.
18. **Recap + close.**

---

## Course-callback spine (ALL VERIFIED 2026-08-03)

Every row below was checked against the actual `.tex`. The first draft's table had **six wrong rows
out of eight**; they are listed in "Review corrections" so the mistake is not repeated.

| Callback | Where | Verified at | Status |
|---|---|---|---|
| **L23 VAE samples are blurry** | L27 f1 | `L23_vae.tex:454` | genuine |
| **L23 max-likelihood -> intractable** | L28 f2 | `L23_vae.tex:200` | genuine |
| **L23 multiply-and-divide by q** | L28 f5 | `L23_vae.tex:200-222` | genuine |
| **L23 ELBO via KL >= 0 decomposition** | L28 f6 | `L23_vae.tex:200-222` | genuine - **use this route, not Jensen** |
| **L23 reparameterization trick** | L28 f16 | `L23_vae.tex` | genuine |
| **L22 conv autoencoder (UNet shape)** | L28 f29 | `L22_autoencoders.tex:264-282` | genuine |
| **L16/L17/L19 conv, transposed conv, skips** | L28 f29 | built decks | genuine |
| **L24 dot product as similarity** | L30 f5, f11 | `L24_attention.tex:534-552` | genuine |
| **L23 latent diffusion, already revealed** | L31 f2 | `L23_vae.tex:405-422` | genuine - **recall, do not re-reveal** |
| **L23 generative-landscape table** | L31 f13 | `L23_vae.tex:454` | genuine - extend it |
| ~~L23 Jensen's inequality~~ | - | **0 hits in any built deck** | **REMOVED** |
| ~~L13b cosine similarity~~ | - | **0 hits in L13b** | **REMOVED** - corrected to L24 |
| ~~L19 GAN instability~~ | - | L19 has a citation line only | **REMOVED** - now taught as new (L28 f30) |
| ~~L24/L25 cross-attention~~ | - | 0 hits in L24; L25/L26 unwritten | **REMOVED** - L30 self-contained |

## Figures (`py_src/` -> `fig/`, `ma` venv, seed 509)

**As built** (2026-08-03). `diffusion_lib.py` holds the shared schedule / model / sampler code so
the three spiral scripts do not each reimplement DDPM.

| Script | Produces | Measured cost |
|---|---|---|
| `diffusion_lib.py` | shared: schedules, `ToyEpsNet`, training loop, DDPM/DDIM samplers, score field | (library) |
| `forward_1d.py` | `forward_1d.pdf` - bimodal p(x) flattening to N(0,1), analytic | ~2 s |
| `ve_vs_vp.py` | `ve_vs_vp.pdf` - mean & variance vs t, VE vs VP | ~2 s |
| `noise_schedules.py` | `noise_schedules.pdf` - beta / alpha-bar curves + worked numbers | ~2 s |
| `kl_gaussians.py` | `kl_gaussians.pdf` - Gaussian KL collapsing to a squared distance | ~2 s |
| `spiral_figures.py` | `spiral_data.pdf`, `spiral_score_field.pdf`, `spiral_samplers.pdf` - trains one model, emits all three | ~6 min |
| `spiral_cfg.py` | `spiral_classes.pdf`, `spiral_cfg_fields.pdf`, `spiral_cfg_sweep.pdf` | ~1.5 min |
| `digits_ddpm.py` | `digits_training.pdf`, `digits_samples.pdf` - a **real** DDPM trained here | ~3 min |
| `latent_spatial.py` | `latent_spatial.pdf` - flat code vs spatial latent | ~2 s |
| `cost_breakdown.py` | `cost_breakdown.pdf` - 48x elements vs 4096x attention | ~2 s |
| `gen_landscape.py` | `gen_landscape.pdf` - quality vs sampling cost | ~2 s |

Fifteen generated figure PDFs, plus 17 borrowed stills in `fig/borrowed/`. All scripts run
sequentially with `torch.set_num_threads(4)`; nothing was run in parallel.

**Numbers quoted on slides come from `logs/`, not from memory:** DDPM spread 0.682, naive-no-noise
0.003, DDIM-25 0.642 (`spiral_figures.log`); guidance mean radius 0.91 -> 1.10 for alpha 1 -> 7.5
(`spiral_cfg.log`); digits final loss 0.072 (`digits_ddpm.log`).

Resolved: L31 f4 ("Stable Diffusion end to end") is a **TikZ** block diagram. It is boxes and arrows
with no data in it, which is precisely the throwaway case `SLIDE_STYLE.md:109` allows TikZ for.

## Compute guardrails (HARD)

16 GB laptop, integrated Intel Iris Xe, no CUDA, documented lock-up history under sustained
multi-core load.

**Freeze-safety, restated verbatim from `AE_CHAPTER_PLAN.md:105-107` (HARD):** cap threads with
`torch.set_num_threads(4)`, run figure scripts **sequentially, never in parallel**, and keep an
explicit epoch/batch ceiling in the script.

`mnist_ddpm.py` is the only real training. Revised estimate: a ~1-2M-param UNet on 60k MNIST at
batch 128 is ~470 steps/epoch; at ~0.5 s/step with 4 threads that is ~4 min/epoch, so
**~6-8 hours for 100 epochs** - and it climbs fast if the UNet grows.

- **(a) Colab T4** via the WSL `colab` CLI. ~20-30 min. Best figures. Free-tier limits apply.
- **(b) CPU, deliberately small** - 28x28, small UNet, ~10 epochs, thread-capped. ~20-40 min local.
  Samples visibly mediocre; usable only if labelled "a small model, briefly trained."

**Recommendation: (a).**

> **Option (c) from the first draft is withdrawn.** It claimed `deepinv` ships pretrained DDPM
> weights for MNIST and FFHQ, samplable in "minutes on CPU". It does not ship MNIST weights at all -
> `DiffUNet(pretrained='download')` gives FFHQ 256x256 (~357 MB, 3-channel) or ImageNet128 (~2.1 GB),
> and the docs warn generation degrades at other sizes. A 1000-step sample of a 256x256 image on
> this CPU is tens of minutes *per image*, and a 357 MB download breaks the standing no-big-models
> rule. **For FFHQ faces, use the borrowed Deepia still `dp f42`** - which L27 frame 1 already does.
> (I asserted the deepinv capability without checking it. Verify library claims before they reach a
> plan.)

## Open questions for the instructor (answer before build)

1. **Confirm 5 decks** now that the merge-to-4 fallback is withdrawn as arithmetically false?
2. **`mnist_ddpm.py`: (a) Colab, or (b) small local?** Recommendation (a). Needs an explicit OK -
   it spends Colab quota.
3. **The UNet frame (L28 f29)** - approve? Without it L30 has nothing to attach cross-attention to,
   but it does add scope.
4. **FID / evaluation (L29 f23)** - approve? The chapter otherwise judges quality by eyeballing.
5. **Provenance and misuse (L31 f16)** - approve? My view: a 2026 course should not end on Sora
   silently, but it is your classroom.
6. **Homework?** Natural fit: "implement the sampling loop" - ~30 lines against a provided model.
7. **L31 f4 architecture diagram:** TikZ (against the Python-figures rule) or a borrowed figure?
8. **Armenian terminology** for: *diffusion, forward/reverse process, noise schedule, score,
   guidance, latent space, denoiser*. Some may be better left in English.
9. **Reference folders and git.** Root `.gitignore:249` has `_reference*/`, so both folders are
   untracked. Force-add transcripts, frames and READMEs (~12 MB), or leave local-only?
10. **Stale pointers elsewhere.** `L17:681,687`, `L19:537`, `L21:253,320,572` and `L23:473` still
    send students to "the GenAI chapter", which no longer exists under that name. L23's closing
    "Next:" box is the urgent one. Fix now, or when those decks are next touched?

## Review corrections (2026-08-03)

An independent adversarial review found the first draft **not safe to build from**. What was wrong:

**Fictitious callbacks (the worst of it).** Six of eight rows in the callback spine - the table the
plan nominated as "what makes it OUR chapter" - were false or wrong-deck. Verified by grep: no
Jensen in any built deck; no cosine in L13b; no GAN-instability content in L19; no cross-attention
in L24, and L25/L26 do not exist. **Cause: the table was written from memory of what those lectures
*should* contain, without opening them.** Every row is now checked and cited.

**A broken payoff.** L29's best moment explained the no-noise blur as "the model learns a Gaussian
mean, so you must add the spread back" - the video's on-screen story, which its own description
retracts, and which DDIM contradicts four frames later. Now taught as a train/test noise-level
mismatch, which makes DDIM the natural next beat instead of a contradiction.

**A wrong formula.** The CFG description dropped the base term. Now written out, with the convention
named and alpha = 1 flagged as no-guidance.

**An unverified library claim.** `deepinv` does not ship MNIST DDPM weights. Option (c) withdrawn.

**A spoiled reveal.** L31's latent-diffusion payoff was already delivered in L23, same numbers.
Reframed as recall plus the genuinely new part (the latent is spatial).

**Budget errors.** Frame counts excluded mandatory `[plain]` and Outline frames (~25% undercount),
L28 was budgeted at 20 frames for work L23 needed 6 frames to do a simpler version of, and the
merge-to-4 fallback said 17 + 20 = 32.

**Omissions now added:** the UNet, FID, provenance/misuse, why-not-text, the Sohl-Dickstein 2015
origin, per-frame citations, acronym discipline, and the freeze-safety cap.

Also corrected: Brownian motion -> Ornstein-Uhlenbeck; VE reframed as a real family rather than a
mistake; score/epsilon sign-and-scale stated; DDIM's actual derivation history; alpha-bar = (1-beta)^t
marked as constant-beta only; VP's "variance -> 1" conditioned on data scaled to [-1,1]; ~1000 steps
attributed to DDPM rather than Stable Diffusion; L27's naive-one-step reveal corrected to match the
source ("does not work well", not "works but noisily").

**Not accepted:** the review suggested the deferred-suspense device in L29 might not survive a deck.
Keeping it - it is announced, and the intervening frames build toward the answer. Only the payoff
needed fixing.

## Build status

- [x] Sources fetched, reviewed, documented
- [x] Chapter split + `ch9` rename; stale cross-references partly updated (see open question 10)
- [x] `DECISIONS.md` created (#1, #2, #3, #4)
- [x] Adversarial review; corrections applied
- [x] Build approved by instructor 2026-08-03: **lectures only**, no homework, no practical,
      Armenian kept light
- [x] `py_src/` figures - 11 scripts, 16 figure PDFs
- [x] **L27 (23pp) / L28 (30pp) / L29 (27pp) / L30 (27pp) / L31 (19pp)** - 126 pages, compiled
      twice each, overflow-checked page by page, acronym check passed, aux files cleaned
- [ ] Chapter `.qmd` + `_quarto.yml` registration (note: ch9 is not registered either)
- [x] ~~Homework / practical~~ - **cut by instructor decision 2026-08-03**

### Corrections made during the build (all caught by verification, not by reading)

1. **The spiral samplers were wrong at first.** With `T=200` and a cosine schedule the model was
   undertrained: DDPM samples were loose and DDIM threw off a clump of escapees. Fixed by moving to
   the real DDPM setup (`T=1000`, linear beta) plus cosine LR decay. Final loss 0.298 -> 0.158, and
   DDIM-25 now matches DDPM (spread 0.642 vs 0.682).
2. **The digits model was silently broken.** At `T=400`, `sqrt(abar_T) = 0.132` - the forward
   process never reached noise, so `x_T` was not a sample from N(0,I) and every image inherited a
   bias from its starting draw. Now `T=1000` (`sqrt(abar_T) = 0.0064`), and the script **raises**
   rather than logging a warning if this ever regresses.
3. **`_eps_guided` crashed on the unconditional field.** On a class-conditional model,
   "unconditional" means the reserved null token, not "call without labels".
4. **The guidance narrative did not survive measurement.** The slide claimed samples scatter at
   alpha=1 and tighten as alpha rises. The toy says otherwise: three radially-separated classes are
   easy, so conditioning alone already works, and at alpha=7.5 samples leave the spiral entirely
   (mean radius 1.10 against a data max of 1.0). Slide and figure title rewritten to the measured
   story - the toy shows the guidance **cost curve**, and the Stable Diffusion missing-tree image
   carries the "conditioning is not enough" argument on its own.
5. **Two borrowed panels were illegible.** The DDPM algorithms and the SDE/ODE comparison are dense
   paper screenshots; at half-column width they were unreadable, exactly as
   `_reference_welchlabs_diffusion/README.md` warned. Both are now **full-bleed** frames
   (`\wlslide`) with the discussion on the following slide.
6. **One borrowed still was a mid-animation frame.** `dp_paths.jpg` was captured at 12:20, before
   the visual appeared, so it rendered as equations over a black void. Re-grabbed at 12:52
   (`f17b_00-12-52.jpg`).

### Student review (two Sonnet passes, 2026-08-03)

Both passes read only the rendered PNGs, with no access to the source or this plan. Neither found a
mathematical error - the ELBO split, the true posterior, the cancellation, the loss prefactor, the
DDIM update and the guidance formula were all checked and are correct. What they did find:

**Pass 1 (L28 + L29):**
- `sigma_t` was used in the KL term, the loss prefactor and the sampler but **never given a value**.
  Training does not need it (the reweighting removes it), so it slipped through - but sampling does,
  and a student could not reproduce Algorithm 2. Now stated as `sigma_t^2 = beta_t`.
- `alpha_t` was used ~24 slides before being defined, and the identity
  `alpha-bar_t = alpha_t * alpha-bar_{t-1}` - which the central algebraic collapse silently relies on -
  appeared nowhere. L27 now names all three symbols and states the identity; L28 opens with a
  notation box.
- The attribution on the new full-bleed frames collided with text already in the stills. Fixed with
  a backing box, moved bottom-right.

**Pass 2 (L30 + L31):**
- **Factual error:** the text-conditioning tensor was given as `77x512`, copied from the Welch Labs
  description note. Stable Diffusion 1.x uses CLIP **ViT-L/14**, so it is **`77x768`**; 512 is the
  ViT-B/32 width, a different checkpoint. Verified by web search before changing. The slide now
  gives 768 and explains why it differs from the 512 quoted two slides earlier.
- **Internal contradiction:** three slides said a guided sample is 50 DDIM steps / 100 network
  calls, while `gen_landscape.pdf` says 25 passes and annotates "40x fewer steps" - and 1000/25=40
  only works with 25. Standardized on 25 everywhere, matching the measured DDIM-25 in L29.
- **Figure did not support its caption:** the guidance slide claimed "as alpha rises, a tree appears
  and grows" beside a single still at alpha=8. Replaced with a genuine before/after pair
  (`wl_alpha_1.jpg` at alpha=1, no tree; `wl_alpha_8.jpg` at alpha=8, tree).
- **Legend collision** in `cost_breakdown.pdf` - the "48x cheaper" annotation was drawn over the
  legend and rendered as one garbled line. Legend moved below the axes; annotations now sit above
  their own bar group.
- Negative-prompt formula had silently dropped its `(x_t, t)` arguments. Restored.

Fixing the guidance frame introduced a **page-number collision** with the callout box, caught on
re-render and fixed by tightening the frame. Worth noting: that fault was created by a fix, which is
the argument for re-rendering after every edit rather than trusting the change.

### Still open

Questions 8 (Armenian terminology - deferred; instructor asked to keep Armenian light), 9 (whether
to force-add the reference folders past `.gitignore:249`) and 10 (stale "GenAI chapter" pointers in
L17 / L19 / L21 / L23) are unresolved. **Question 10 matters most: L23's closing "Next:" box still
sends students to a chapter name that no longer exists.**
