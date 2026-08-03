# Welch Labs / 3Blue1Brown — "But how do AI images and videos actually work?"

Reference material for the **ch10 diffusion** chapter.

- **Video:** <https://youtu.be/iv-5mZ_9CPY> — guest video by **Stephen Welch (Welch Labs)** on the
  **3Blue1Brown** channel, 37:20, 25 Jul 2025.
- **Topic:** text-to-image / text-to-video diffusion end to end — CLIP's shared embedding space,
  DDPM training and sampling, the **score / time-varying vector field** view on a 2D toy spiral,
  DDIM as the deterministic probability-flow ODE, conditioning, and **classifier-free guidance**.
- **Why it matters here:** it is the only explainer that gets from "add noise, learn to remove it"
  to *why* DDPM adds noise **during sampling** and why removing it produces a blurry mean-image —
  all through one 2D picture the students can actually reproduce in numpy.

## Contents

- `transcript.txt` — cleaned, timestamped transcript, ~6.5k words (committed).
- `description.txt` — the author's own section map plus every paper/repo he links (committed).
- `frames/` — 52 screenshots at key visual beats, `fNN_HH-MM-SS.jpg`, **1080p** (committed).
- `meta.txt` — one-line metadata (committed).
- `video.mp4` — 1080p download, 408 MB (**git-ignored**; re-fetch with the command below).
- `video.en.vtt` — raw auto-subs (git-ignored; `transcript.txt` is the cleaned version).

## Re-fetch the video

```bash
yt-dlp --js-runtimes node \
  -f "137+bestaudio/bestvideo[height<=1080]+bestaudio" --merge-output-format mp4 \
  -o "video.%(ext)s" "https://youtu.be/iv-5mZ_9CPY"
```

> **`--js-runtimes node` is required.** Without a JS runtime yt-dlp cannot compute YouTube's
> signature: format listing still works, but the media fetch dies with `HTTP 403: Forbidden`.
> Node v20 is on PATH. The repo's bundled `yt_fetch.sh` does not pass this flag yet.

## Key beats

| Time | Beat | Frames |
|------|------|--------|
| 0:00 | Intro — Wan 2.1 astronaut, prompt ablations, empty prompt still yields a video | f01-f03 |
| 1:10 | Generation loop: RNG → pure noise → transformer → add → repeat, 5/10/20/30/40/50 steps | f03-f05 |
| 3:37 | **CLIP** — GPT-3 scaling backdrop; two encoders, 512-dim output, 400M caption pairs | f06-f07 |
| 4:45 | Contrastive objective — batch matrix, diagonal = matching pairs, cosine similarity | f08-f10 |
| 6:25 | **Shared embedding space** — (me with hat) − (me without hat) ≈ "hat", sim 0.165 | f11-f12 |
| 7:30 | Zero-shot classification by nearest caption | f13 |
| 8:16 | **DDPM** (Ho, Jain, Abbeel 2020) — forward noising chain | f14-f15 |
| 9:30 | Algorithm 1 / Algorithm 2; two surprises: noise added *at sampling*, model predicts *total* ε | f16, f19 |
| 10:20 | Stable Diffusion "tree in the desert" vs the same with the noise step deleted → blurry blob | f17-f18 |
| 11:44 | **Vector-field view** — 2-pixel image as a 2D point; the spiral toy dataset | f20-f21 |
| 13:20 | Adding noise = a random walk = Brownian motion | f22-f23 |
| 15:00 | Why predict total noise: E[x99−x100] = E[x0−x100]/100, same target, far less variance | f24 |
| 16:00 | **Score function** — the field points back toward the data; coarse at large t, fine at small t | f25-f27 |
| 17:15 | Time conditioning; the field "phase-changes" near t ≈ 0.4 | f28 |
| 18:10 | DDPM trajectory: blue model steps + gray noise steps; 256-point cloud converges | f29-f30 |
| 19:03 | Drop the noise step → everything collapses to the spiral's centre = the dataset mean | f31-f32 |
| 20:10 | Why: the model learns the **mean** of a Gaussian, so you must re-sample around it | f33 |
| 22:00 | **DDIM** — SDE → Fokker-Planck → probability-flow ODE with the same final distribution | f34-f36 |
| 23:40 | DDPM vs DDIM side by side: deterministic, fewer steps, no retraining. Wan uses flow matching | f37-f38 |
| 25:25 | **DALL·E 2 / unCLIP** — train a diffusion decoder to invert the CLIP image encoder | f39-f40 |
| 26:37 | **Conditioning** — text vector as an extra model input; cross-attention vs concat | f41-f42 |
| 28:00 | Conditioning alone is not enough: desert + shadow, **no tree** | f43 |
| 28:20 | Toy analogue: label spiral regions person / dog / cat; classes bleed into each other | f44-f45 |
| 30:02 | **Classifier-free guidance** — drop the label for some training data to get an uncond. field | f46 |
| 31:00 | f(x,t,cat) − f(x,t), amplified by α, replaces the conditional vector | f47-f49 |
| 33:00 | α sweep on Stable Diffusion: the tree literally grows as guidance increases (α = 8) | f50 |
| 33:39 | **Negative prompts** — Wan subtracts an explicit "what I don't want" vector (theirs is Chinese) | f51-f52 |
| 34:27 | Outro, then 3B1B on why this is a guest video — **not useful for slides** | — |

## Notes for slide use

- Frames are 1080p and 16:9, so they work full-bleed on a `aspectratio=169` deck with no bars.
- **f16** (Algorithm 1/2) and **f36** (SDE vs ODE) are dense paper-style panels — readable full-screen,
  but do not shrink them into a two-column frame.
- **f22** and **f29** were caught mid-animation and are nearly empty; re-grab a second or two later
  if either is wanted.
- The whole 11:44-21:00 stretch is one continuous argument on the same spiral. It is the part most
  worth **redrawing in house style** (`py_src/`) rather than borrowing — it is plain 2D numpy, and
  owning the figure lets the notation match the rest of the course.
- Attribution line for any borrowed still: `Welch Labs / 3Blue1Brown (2025)`.

## Papers the video builds on

| Paper | Link |
|-------|------|
| CLIP (Radford et al. 2021) | <https://arxiv.org/pdf/2103.00020> |
| DDPM (Ho, Jain, Abbeel 2020) | <https://arxiv.org/pdf/2006.11239> |
| DDIM (Song et al. 2020) | <https://arxiv.org/pdf/2010.02502> |
| Score-based generative modeling / SDE (Song et al. 2021) | <https://arxiv.org/pdf/2011.13456> |
| Classifier-free guidance (Ho & Salimans 2022) | <https://arxiv.org/pdf/2207.12598> |
| DALL·E 2 / unCLIP (Ramesh et al. 2022) | <https://cdn.openai.com/papers/dall-e-2.pdf> |
| Nakkiran et al., diffusion tutorial | <https://arxiv.org/pdf/2406.08929> |

Teaching-code references the author credits: `smalldiffusion`
(<https://github.com/yuanchenyang/smalldiffusion>) and the MIT practical course
(<https://www.practical-diffusion.org/>) — both are small enough to lift a homework from.
