# Welch Labs — Grokking reference

- **Source:** https://www.youtube.com/watch?v=D8GOeCFFby4
- **Uploader:** Welch Labs
- **Published:** 2025-12-20
- **Duration:** 35:28
- **Topic:** A mechanistic-interpretability walkthrough of how a one-layer transformer learns modular addition, why its generalization can appear suddenly ("grokking"), and a brief comparison to an Anthropic analysis of Claude Haiku.

## Re-fetch

Run from the repository root:

```bash
bash .claude/skills/youtube-reference/scripts/yt_fetch.sh \
  "https://www.youtube.com/watch?v=D8GOeCFFby4" \
  "misc/grokking"
```

The local `video.mp4` is downloaded at 720p and is intentionally git-ignored.

## Key beats

| Timestamp | Beat |
|---|---|
| 00:02:39 | Modular addition setup and held-out test examples |
| 00:07:01 | Training memorizes first; test accuracy then rises suddenly |
| 00:09:00 | One-layer transformer architecture used in the analysis |
| 00:14:00 | Fourier analysis and linear probes reveal sine/cosine features |
| 00:17:00 | Clock analogy for modular arithmetic |
| 00:20:02 | Two-dimensional neuron-output surfaces |
| 00:24:01 | `cos(x)cos(y) - sin(x)sin(y) = cos(x+y)` explains the learned mechanism |
| 00:27:00 | Excluded loss exposes progress while ordinary metrics are flat |
| 00:30:00 | Claude Haiku line-break mechanism as a full-model comparison |

## Extracted frames

- `f01_00-03-52.jpg` — tactile modular-addition table.
- `f02_00-07-56.jpg` — memorization versus generalization accuracy curves.
- `f03_00-12-10.jpg` — transformer and periodic activation overview.
- `f04_00-17-30.jpg` — analog-clock modular-addition analogy.
- `f05_00-20-30.jpg` — probes, network structure, and frequency surfaces.
- `f06_00-24-25.jpg` — trigonometric identity as the mechanism for addition.
- `f07_00-28-20.jpg` — excluded-loss diagnostic during grokking.
- `f08_00-30-35.jpg` — geometric activation structure in the Claude Haiku comparison.

## Papers

- `papers/power_et_al_2022_grokking.pdf` — Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*.
- `papers/nanda_et_al_2023_progress_measures_for_grokking.pdf` — Nanda et al., *Progress measures for grokking via mechanistic interpretability*.
- `papers/anthropic_2025_when_models_manipulate_manifolds.md` — canonical link and citation for the interactive Anthropic line-break paper discussed at the end of the video.

LLM-readable text versions are in `papers/llm_readable/`.

## Minimal reproduction code

`code/` contains a small PyTorch transformer experiment for modular-addition grokking. See `code/README.md` for the two commands to train and then inspect its learned Fourier structure.
