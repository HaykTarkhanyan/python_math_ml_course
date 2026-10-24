# A character-level LSTM encoder-decoder is ~30x faster on a Colab T4 than on this laptop - and on the laptop it never got far enough to learn

**Symptom.** The RNN project's Task B (Armenian text without spaces -> with spaces, delayed seq2seq)
was planned for "10 minutes on a laptop CPU". Measured on this laptop (`ma` venv, torch 2.9 CPU):

| run | laptop CPU | Colab T4 |
|---|---|---|
| Task B, 10-80 letters, hidden 128, 600 steps, batch 128 | 1,119 s, learned nothing (wrote «այն այն այն...» for everything, CER ~0.88 at every length) | - |
| Task B, 5-30 letters, hidden 128, 1,500 steps, batch 64, reversed input | 394 s | 12 s |
| dates, hidden 128, 1,500 steps, batch 128 | 412 s (2 threads) | - |
| whole 7-run sweep (incl. 3,000-step run) | estimated hours | 99 s |

**Cause.**
- **Many sequential steps per batch.** Teacher-forced training is one LSTM call per side, but each
  call loops over 50-100 time steps.
- **Packing costs extra on a CPU.** `pack_padded_sequence` made a training step 2-3x slower than
  ignoring the pads: dates, hidden 64, 2 threads, 467 ms vs 167 ms per step. It is kept anyway,
  because it is [53]'s mask.
- **The laptop slows down under load.** The 600-step run went from 1.0 s per step at the start to
  2.3 s later. The next sweep crawled at 1.1 s per step because an `ffmpeg` render was using ~4.8 of
  the 8 cores.
- **Character copying needs thousands of steps.** Copying characters through one vector only
  starts working after thousands of steps, so the laptop runs never got there.

**Consequences.**
- Any teaching notebook that trains an encoder-decoder to a plateau targets a Colab T4. The RNN
  project says so in its first cell and runs in ~11 min there.
- Reversing the encoder input (Sutskever et al., 2014) is what made Task B learnable at all at
  short budgets. Hidden 128, 12,000 steps, 5-9 letters: CER 0.08 reversed vs 0.44 not.
- Before blaming the model for a slow local run, check what else is using the cores. A one-off
  PowerShell sample of per-process CPU over 3 s found the render.
