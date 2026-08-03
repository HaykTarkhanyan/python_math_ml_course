# Deepia — "Diffusion Models: DDPM | Generative AI Animated"

Reference material for the **ch10 diffusion** chapter. This is the **derivation** counterpart to
`_reference_welchlabs_diffusion/` (which carries the intuition).

- **Video:** <https://youtu.be/EhndHhIvWWw> — **Deepia**, 32:05, 27 May 2025.
- **Topic:** the DDPM paper worked end to end — variance-exploding vs **variance-preserving**
  forward process, the reverse process, **negative log-likelihood → ELBO → sum of KLs → L2 →
  ε-prediction**, then a full PyTorch training and sampling implementation on MNIST and FFHQ.
- **Why it matters here:** Manim-rendered, white-on-black LaTeX, one idea per frame. It is the
  cleanest available visual derivation of the DDPM loss, and it ends on a **VAE-vs-diffusion sample
  comparison** that pays off the L23 VAE "blurry samples" cliffhanger directly.

## Contents

- `transcript.txt` — cleaned, timestamped transcript, ~5.3k words (committed).
- `description.txt` — author's chapter list + his code repo and reading list (committed).
- `frames/` — 42 screenshots, `fNN_HH-MM-SS.jpg`, **1080p** (committed).
- `meta.txt` — one-line metadata (committed).
- `video.mp4` — 1080p download, 88 MB (**git-ignored**; re-fetch below).
- `video.en*.vtt` — raw subs (git-ignored). `video.en-orig.vtt` is the real English track.

## Re-fetch the video

```bash
yt-dlp --js-runtimes node \
  -f "bestvideo[height<=1080]+bestaudio/best[height<=1080]" --merge-output-format mp4 \
  -o "video.%(ext)s" "https://youtu.be/EhndHhIvWWw"
```

> `--js-runtimes node` is required, same as the Welch Labs fetch — without it the media request
> returns `HTTP 403: Forbidden` even though format listing succeeds.

## Key beats

| Time | Beat | Frames |
|------|------|--------|
| 0:00 | Intro — DDPM (2020) is *not* the origin; Sohl-Dickstein et al. 2015 is. 20k+ citations | f01 |
| 0:32 | Generation = sampling from an intractable p(x); corrupt, then learn to reverse | f02-f03 |
| 3:28 | **Forward process** — q(x1\|x0) = N(x0, β); noise sums, so q(xt\|x0) = N(x0, tβ) | f04-f06 |
| 5:00 | **This is broken**: mean stays at x0, variance grows without bound = *variance exploding* | f07 |
| 5:54 | **Variance preserving** fix — the √(1−β) coefficient; ᾱt = (1−β)^t; converges to N(0,1) | f08-f10 |
| 7:40 | 1D demo: a bimodal p(x) flattening toward the normal over ~100 steps | f11-f12 |
| 8:14 | General β schedule → ᾱt as a product, matching the paper's notation | f12 |
| 9:07 | **Reverse process** — define p_θ(x_{t−1}\|x_t); θ are the network weights | f13 |
| 10:00 | Train by minimizing the **negative log-likelihood** −log p_θ(x0) | f14 |
| 10:30 | Markov factorization of the forward and reverse joints | f15-f16 |
| 11:40 | Marginalizing is **intractable**: it sums over every path from noise to image | f17 |
| 13:00 | Multiply-and-divide by q, then **Jensen's inequality** → the **ELBO** | f18-f19 |
| 14:06 | Expanded ELBO: a prior KL + a sum of KLs + a reconstruction term | f20-f21 |
| 14:20 | KL divergence explained as a distance between distributions | f22 |
| 15:01 | Drop term 1 (no θ) and term 3 (negligible at t=1) → just the KL sum | f23-f24 |
| 16:05 | The **true posterior** q(x_{t−1}\|x_t,x0) — computable only because training knows x0 | f24-f25 |
| 17:40 | Choose a Gaussian approximate posterior; **fix the variance**, learn only the mean | f25-f26 |
| 18:30 | Gaussian KL with fixed variance collapses to a plain **squared distance of means** | f26-f27 |
| 19:37 | **Reparameterization** — rewrite both μ̃t and μθ via xt and ε; the xt terms cancel | f28-f29 |
| 21:00 | **Final loss:** a weighted ‖ε − εθ(xt,t)‖². The network just predicts the noise | f30 |
| 21:50 | Recap of the whole chain, NLL → ELBO → KL → L2 → ε | f31 |
| 22:30 | Last simplification: sample one random t per example instead of summing over all | f32 |
| 23:44 | **Sponsor read — skip 23:44-24:39** | — |
| 24:39 | PyTorch implementation: DiffUNet, β schedule, the training loop | f33-f36 |
| 27:20 | MSE curve; per-epoch denoising grid at t = 50/100/400/700 | f36-f37 |
| 28:14 | **Sampling loop** — under 30 lines; predict ε, form the posterior mean, add noise | f38-f39 |
| 29:50 | MNIST samples, then FFHQ faces from the same loop with swapped weights | f40-f41 |
| 31:00 | **FFHQ diffusion vs VAE samples side by side** — the sharpness payoff | f42 |
| 31:24 | Outro: cost is 1000 network evals; DDIM fixes it; next video is score-based/SDE | — |

## Notes for slide use

- Frames are Manim: white LaTeX on pure black, 1080p, one idea each. They will **not** match the
  deck's `dove`/white background — either use them full-bleed on a black `[plain]` frame, or
  (better for the derivation) **retype the equations in house style** and keep the frames for the
  visual moments only (f11 1D flattening, f22 KL, f37 denoise grid, f42 VAE-vs-diffusion).
- **f42 is the single most valuable frame in either video** for this course — it closes the L23 VAE
  loop with evidence rather than assertion.
- The author's code (MIT-style teaching repo) is at <https://github.com/ytdeepia/DDPM> — a
  ~30-line sampling loop and a short training loop, both a realistic basis for a homework.
- Attribution line: `Deepia (2025)`.

## How the two references divide the work

| | Welch Labs / 3b1b | Deepia |
|---|---|---|
| Route | geometric intuition, 2D spiral, score field | algebraic derivation from the NLL |
| Forward process | "add noise = random walk" | β schedule, VP vs VE, closed-form ᾱt |
| Why ε-prediction | variance reduction argument, drawn | reparameterization, terms cancel |
| Sampling noise | *why* removing it blurs — drawn on the spiral | falls out of "we learn a Gaussian mean" |
| Beyond DDPM | DDIM, CFG, negative prompts, CLIP, video | none (teased for a later video) |
| Code | none | full PyTorch train + sample |

Together they cover the chapter; neither does alone. Welch Labs supplies the *why*, Deepia the
*how*, and only Welch Labs covers conditioning/guidance — the part that makes text-to-image work.
