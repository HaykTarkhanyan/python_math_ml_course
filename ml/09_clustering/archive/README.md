# archive/

Superseded material kept as an institutional record. Nothing here is expected to run, and
nothing here is imported by live code.

| File | What it is | When you would look at it |
|---|---|---|
| `2025_image_compression_ml_old.ipynb` | The 2025 colour-quantization notebook from `ml_old/Chapter 4 Clustering/Code/image_compression.ipynb`, copied here 2026-08-27. Superseded by `35_image_compression_solution.ipynb`. | To see how the topic was taught before, or to harvest the two ideas noted below. |
| `Aivazovsky.jpg` | Seascape used by that notebook. Large smooth sky/water gradients, so it posterizes visibly at low `k`. **Now promoted to a live asset** at `../img/aivazovsky.jpg` and used by section 9 of the current notebook; this copy is the archival original. | Reference only. |

## Why it was kept rather than deleted

Two things in it are worth more than the notebook itself.

**1. It contains a real, publishable mistake.** Its final cell reports a compression ratio by
comparing JPEG file sizes on disk:

```
Original image size: 6443042 bytes
Compressed image size: 185338 bytes
Compression ratio: 34.76
```

That number is meaningless, and not by a little. The original is 6000x4000; the "compressed"
image was resized to 1000x666 *before* clustering. Per pixel:

| | bytes | pixels | bytes/pixel |
|---|---|---|---|
| original | 6,443,042 | 24,000,000 | 0.2685 |
| quantized to 50 colours | 185,338 | 666,000 | 0.2783 |

The resize alone accounts for **36.04x**. Quantizing to 50 colours made the file **3.7 % bigger
per pixel** — JPEG's DCT codes hard posterized edges *worse* than the smooth gradients it
replaced. So the one step the notebook was actually teaching had negative measured value, and the
headline number hid it behind a resize.

`35_image_compression_solution.ipynb` task 5 currently asks students to *explain in one line* why
comparing re-saved `.jpg` sizes is misleading. This file is the worked example of it going wrong
on real numbers.

**2. Other assets.** The remaining source images stay in `ml_old/Chapter 4 Clustering/Code/in/`
rather than being copied (they total ~11 MB): `libs_03_kond_street_art.jpg` (Kond street art,
Yerevan — high-chroma local subject), `08_rainbow.jpg` (continuous spectrum, the worst case for a
small palette), `Armenia.jpg`, `Saryan.jpg`.

## Other flaws, for the record

- Sweeps `k = 2..12` with elbow and silhouette, then hardcodes `chosen_n_clusters = 50`. The
  diagnostic is decorative — it never touches the decision.
- `random_state=509` in the sweep, `random_state=42` in the final fit.
- `!uv pip install` inside a notebook cell.
- Averages gamma-encoded sRGB values directly, so every centroid is biased — the thing task 7 of
  the current notebook now measures.
- The `.png` branch assumes a 2-D array (grayscale) and then sets `num_channels = 3`; it would
  crash on any RGBA png.
