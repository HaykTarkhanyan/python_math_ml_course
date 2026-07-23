# Welch Labs — "Why Deep Learning Works Unreasonably Well"

Reference material for the **L14 folding / depth** content (ch5 neural networks).

- **Video:** <https://youtu.be/qx7hirqgfuU> — Welch Labs, *How Models Learn Part 3*, 34:08, Aug 2025.
- **Topic:** the ReLU **fold → scale → combine** geometry and the **depth vs width** region-counting
  argument (Montúfar et al. 2014), using the Baarle-Hertog (Belgium/Netherlands) border as the running
  example. Directly parallels L14's *"Depth folds space"* and *"Why add more layers?"* frames.

## Contents

- `transcript.txt` — cleaned, timestamped transcript (committed).
- `frames/` — 45 screenshots at key visual beats, named `fNN_HH-MM-SS.jpg` (committed).
- `video.mp4` — 720p download (**git-ignored**, 93 MB; re-fetch with the command below).
- `video.en*.vtt` — raw auto-subs (git-ignored; `transcript.txt` is the cleaned version).
- `clean_vtt.py` — VTT → clean transcript helper.

## Re-fetch the video

```bash
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" --merge-output-format mp4 \
  -o video.mp4 "https://youtu.be/qx7hirqgfuU"
```

## Key beats (timestamps)

- `00:00`–`02:00` — single ReLU neuron folds the map; 2nd layer scales/flips/combines → surfaces → decision boundary.
- `02:00`–`03:35` — wide 2-layer net: 8 → 16 → ... → 100,000 neurons, still can't fit the border.
- `03:35`–`04:20` — **128 neurons as 4×32 deep beats the 100,000-wide net** (the punchline).
- `08:00`–`10:30` — plane = weights + bias; why stacked linear layers collapse without a nonlinearity.
- `11:00`–`13:00` — ReLU definition and the folding operation.
- `14:00`–`19:00` — why UAT "fails" in practice: existence ≠ findability, dead-ReLU gradient trap, depth needs exponentially fewer neurons.
- `20:00`–`26:00` — deep geometry compounds; 2nd-layer ReLU folds non-planar surfaces; region-count grows exponentially with depth (but the bound is loose in practice).
- `27:00`–`30:00` — scaling a deep net (8/8 → 4 layers → 5×32); final boundary captures every enclave.

## Curated frame sets (`frames/s1_*`) — for the L14 non-linearity + folds slides

Dense pulls for three specific teaching points (files named `s1_<topic>_HH-MM-SS.jpg`):

- **`s1_linear_*` (08:20–10:40) — "no activation ⇒ still one linear model".** The 2×2 net with
  color-coded `h = m·x + b` equations, then the algebraic collapse: substitute layer 1 into layer 2 and
  it flattens to `(const)·x1 + (const)·x2 + const`. Payoff frame: **`s1_linear_00-10-02`**.
- **`s1_relu_*` (10:52–12:45) — the non-linearity.** Hand-drawn ReLU definition card
  (**`s1_relu_00-11-22`**), then ReLU folding the two planes into bent planes.
- **`s1_fold_*` (00:24–01:30) + `s1_scale*` (02:15–03:34) — more neurons ⇒ more folds/regions.**
  Single-neuron fold → 8 → 16 → 32 → 64 → 128 → (flatten) → 256 → 512 → 1024 → 10k → 100k. Each
  `s1_scaleNNN` frame has the neuron-count badge bottom-right.
- **`s1_regiongrowth_00-26-41` — shallow vs deep region count.** `N_r`: shallow (D=64, 2-layer) = **2081**
  vs deep (D=16, K=4) = **72,807,417**. The exponential-in-depth punchline.
