# Welch Labs — "These Numbers Can Make AI Dangerous [Subliminal Learning]"

Reference material for **L48** (ch20 subliminal learning). This chapter is a deliberate
**retelling of this one video**, so this folder is the primary source, not a supplement.

- **Video:** <https://youtu.be/NUAb6zHXqdI> — Welch Labs, 33:04, 4 Sep 2025, 187k views.
- **Topic:** a teacher model with a trait (loves eagles) generates *only number sequences*;
  a student fine-tuned on those numbers picks up the trait. The video walks the paper's
  elimination of the obvious explanations, reproduces the effect on MNIST, and derives the
  proof that teacher and student weight updates cannot point in opposite directions.
- **Underlying papers:**
  - Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks & Evans (2025),
    *Subliminal Learning: Language models transmit behavioral traits via hidden signals in data*,
    [arXiv:2507.14805](https://arxiv.org/abs/2507.14805) (20 Jul 2025). Site: <https://subliminal-learning.com/>
  - Zur, Loftus, Orgad, Ying, Sahin & Bau (2025), *It's Owl in the Numbers: Token Entanglement
    in Subliminal Learning*. **Blog post, not an arXiv paper** — <https://owls.baulab.info/>,
    code at <https://github.com/loftusa/owls>. Mechanism: the **softmax bottleneck**.
    (The video's on-screen author list differs in order from the site's; the site is authoritative.)

## Contents

- `transcript.txt` — cleaned, timestamped (committed).
- `frames/` — 50 screenshots, `fNN_HH-MM-SS.jpg`, **renumbered in timestamp order** (committed).
- `meta.txt`, `description.txt` — video metadata (committed).
- `video.mp4` — 720p, 84 MB (**git-ignored**; re-fetch below).
- `video.en*.vtt` — raw auto-subs (git-ignored).

## Re-fetch the video

```bash
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" --merge-output-format mp4 \
  -o video.mp4 "https://youtu.be/NUAb6zHXqdI"
```

## Key beats (timestamps)

Sponsor read is **01:45–03:30** and the poster/merch outro is **31:00–end** — both skipped.

| Time | Beat | Frames |
|---|---|---|
| 00:00–01:00 | Hook: eagle teacher → number sequences → student loves eagles; also harmful traits | f01–f03 |
| 01:00–01:45 | Knowledge distillation framing; "TOP SECRET" card visual | f04 |
| 03:30–04:30 | Setup: GPT-4.1-nano both sides, system prompt, non-numeric outputs filtered out, baseline prefers dolphins | f06, f07 |
| 04:30–05:30 | **Clue 1 — cross-model transfer fails.** Paper Fig. 8 heatmaps | **f08**, f09, f10 |
| 05:30–06:40 | **Clue 2 — in-context learning does not transfer**, even 10,000 sequences in the prompt | f12 |
| 06:40–07:40 | **Clue 3 — a classifier can tell which model produced a sequence, but that is not trait detection** | f13 |
| 07:40–09:10 | **Clue 4** — trait can be induced by prompt *or* fine-tuning. **Clue 5** — transfers via code: `400`→`404`, `inputs`→`input_tensor` | f14–f16 |
| 09:10–10:30 | Hinton et al. (2015) distillation; MNIST net 784→256→256→10, softmax, 94.3% | f17–f19 |
| 10:30–11:40 | LLM has ~100k outputs vs 10; **add 3 auxiliary outputs** | f20, f21 |
| 11:40–12:40 | Teacher trained on primary only; student matches **auxiliary only**; student exceeds 50% digit accuracy | f22, **f23** |
| 12:40–14:10 | Tiny model: 2×2, eight parameters θ₁…θ₈, primary `f`, auxiliary `g`, shared θ⁰ | f24, f25, f26 |
| 14:10–18:00 | `L_S = ½(g_T − g_S)²`, `Δθ_S = −α∇L_S = α(g_T − g₀)∇g₀` | f27–f31 |
| 18:00–20:15 | First-order Taylor: `g_T ≈ g₀ + ∇g₀·Δθ_T`; the `g₀` terms cancel | f32–f34 |
| 20:15–23:20 | Dot product → **`Δθ_T·Δθ_S = α(∇g₀·Δθ_T)² ≥ 0`** | f35–**f40** |
| 23:20–24:30 | Geometry: the student update is the teacher's update **projected onto ∇g₀** | f41 |
| 24:30–25:20 | Second result: teacher's loss on the student decreases | f42 |
| 25:20–26:10 | Constraints (shared init, one step, first order); resolves the GPT-4.1/GPT-4o exception | f43, f44 |
| 26:10–28:00 | The take: we control these interactions *semantically*, but the mechanism is not semantic | f45, f46 |
| 28:00–29:00 | **Token entanglement**: "you love 087" raises P(owl) by ~300% | **f47**, f48 |
| 29:00–31:00 | Why "subliminal" is the right word; searching the numbers blind is a fool's errand | f49, f50 |

## Frames worth using full-bleed

- **f03** (00:00:42) — the hook: two model cards with the number sequence between them.
- **f04** (00:01:10) — hands passing a "TOP SECRET" card under a KNOWLEDGE DISTILLATION title.
- **f08** (00:04:50) — paper Fig. 8, both heatmaps, caption legible. Exact values transcribed
  into `py_src/make_figs.py` for the house-style redraw.
- **f19** (00:10:15) — full MNIST architecture with softmax and P(0)…P(9).
- **f21** (00:11:30) — image classifier (10 outputs) vs LLM (owl / eagle / cat + digits).
- **f23** (00:12:15) — teacher and student nets side by side + the accuracy-vs-steps plot.
- **f40** (00:23:20) — the boxed result `Δθ_T·Δθ_S = α(∇g₀·Δθ_T)² ≥ 0`.
- **f41** (00:23:45) — the projection geometry in parameter space.
- **f47** (00:28:20) — "You love 087" → "My favorite animal is the owl", +300%.

## Fig. 8 values transcribed from f08 (used in the house-style redraw)

Rows = student, columns = teacher. `*` = significantly different from 0 at ~95%, N ≥ 5 runs.

Left panel:

|  | GPT-4.1 | GPT-4.1 mini | GPT-4.1 nano | GPT-4o |
|---|---|---|---|---|
| **GPT-4.1** | 0.50* | 0.06* | 0.07* | 0.30* |
| **GPT-4.1 mini** | 0.08 | 0.25* | 0.09 | 0.04 |
| **GPT-4.1 nano** | 0.01 | 0.01 | 0.54* | 0.03 |
| **GPT-4o** | 0.32* | −0.01 | −0.01 | 0.33* |

Right panel (different animal set, so the nano→nano cell differs):

|  | GPT-4.1 nano | Qwen2.5-7B |
|---|---|---|
| **GPT-4.1 nano** | 0.39* | −0.01 |
| **Qwen2.5-7B** | 0.01* | 0.11* |
