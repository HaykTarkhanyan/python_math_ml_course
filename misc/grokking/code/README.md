# Minimal grokking experiment

This is a deliberately small reproduction of the setup from the two grokking papers in `../papers/`. It trains a one-layer transformer to solve `x + y (mod p)` from a randomly chosen subset of all possible pairs.

Look for training accuracy reaching 100% quickly, followed later by a sharp rise in held-out accuracy. That delayed generalization is grokking.

## Run

From this directory, using the repository environment:

```powershell
..\..\..\ma\Scripts\python.exe train.py --steps 25000 --weight-decay 1.0
..\..\..\ma\Scripts\python.exe analyze.py
```

Outputs in `artifacts/`:

- `metrics.png`: train/test accuracy during training.
- `checkpoint.pt`: model weights and training history.
- `embedding_fourier_spectrum.png`: a first Fourier diagnostic of the learned number embeddings.

## Small controls

```powershell
# Fast smoke test
..\..\..\ma\Scripts\python.exe train.py --modulus 31 --d-model 32 --n-heads 1 --d-mlp 64 --steps 500

# More examples makes generalization easier
..\..\..\ma\Scripts\python.exe train.py --train-fraction 0.7

# Often weakens or removes grokking
..\..\..\ma\Scripts\python.exe train.py --weight-decay 0
```

The exact onset is sensitive to the seed, model size, learning rate, and weight decay. The goal is to expose those knobs plainly.

## Map to the papers

- `model.py` is one transformer layer: embedding -> self-attention + MLP -> answer-token logits.
- `train.py` generates all equations, hides a random subset, and plots both accuracies.
- `analyze.py` computes a discrete Fourier transform across the number-token embeddings. Strong periodic frequencies are a clue from Nanda et al., not a full circuit reconstruction.
