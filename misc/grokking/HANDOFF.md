# Grokking reproduction handoff

## Current state

- The full paper-mode run was stopped after epoch 1,200; no grokking has occurred yet.
- Latest checkpoint: `runs/paper_p113_train30_fullbatch_seed1/artifacts/checkpoint.pt`.
- At epoch 1,200: train accuracy `1.000`, held-out accuracy `0.00436` (near chance for `P=113`), held-out loss `28.20`.
- Snapshots exist at epochs 1, 200, 400, 600, 800, 1,000, and 1,200.

## Faithful paper-mode configuration

`train.py --paper-mode` uses the Nanda et al. mainline architecture and training setup:

- modular addition with `P=113`, 30% training pairs;
- token sequence `a b =`, one transformer layer, ReLU MLP;
- width 128, four heads, MLP width 512, learned positional embeddings;
- **no LayerNorm**;
- full-batch AdamW, learning rate `1e-3`, weight decay `1`, 40,000 total epochs.

## Resume the run

From this directory (`misc/grokking/runs/paper_p113_train30_fullbatch_seed1`):

```powershell
..\..\..\..\ma\Scripts\python.exe -u ..\..\code\train.py --paper-mode --save-snapshots --resume --seed 1 --threads 1
```

In paper mode, resume now treats 40,000 as the total target, so this runs the remaining 38,800 epochs rather than adding another 40,000.

## Analyze a completed run

```powershell
..\..\..\..\ma\Scripts\python.exe ..\..\code\paper_analysis.py --progress
```

This writes:

- `paper_neuron_logit_fourier.png` — Fourier spectrum of the neuron-logit map `W_L`;
- `paper_logits_2d_fourier.png` — 2D Fourier spectrum of logits;
- `paper_neuron_degree2_fits.png` — degree-2 single-frequency MLP-neuron fits;
- `paper_progress_measures.png` — train, test, restricted, and excluded losses over snapshots;
- JSON summaries alongside the plots.

`excluded loss` here has the paper's meaning: it removes the final model's key Fourier logit frequencies and measures the remaining training performance. It is distinct from held-out/test loss.
