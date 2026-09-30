# Reference: 3Blue1Brown - "But what is a convolution?"

- **Source:** https://www.youtube.com/watch?v=KuXjwB4LzSA
- **Uploader:** 3Blue1Brown (Grant Sanderson) - 2022-11-18 - 23:01 - ~3.6M views
- **Topic:** convolution built up from scratch: discrete lists, the dice/probability
  view, the sliding-flip definition, moving average, 2D image blur + edge kernels,
  and (the back half) polynomial multiplication -> FFT for O(n log n) convolution.

Re-fetch the 720p video (git-ignored):

```bash
SK=".claude/skills/youtube-reference/scripts"
bash "$SK/yt_fetch.sh" "https://www.youtube.com/watch?v=KuXjwB4LzSA" \
  "ml/ch6_cnn/_reference_3b1b_convolution"
```

## Why it's here

Fetched to mine against **L16 CNN Foundations** (`ml/ch6_cnn/L16_cnn_foundations.tex`),
whose Section 2 arc (1D sliding dot product -> moving average -> dice aside -> Fourier
aside -> 2D by hand -> kernel zoo -> edges) overlaps heavily with this video. Most of
the video is content L16 already teaches in house style; the genuinely *additive*
thread is the **polynomial-multiplication = convolution -> FFT O(n log n)** story
(frames 13-17), which L16's current Fourier aside does not cover (it frames the
convolution theorem as frequency-filtering, not as fast multiplication).

## Key beats (timestamped)

- **[00:00]** three ways to combine two lists: add, multiply, **convolve**. Convolution
  shows up in image processing, probability, diffeq, polynomial multiplication.
- **[02:01]** probability motivation: rolling two dice, distribution of the sum. 36
  pairs, constant-sum diagonals.
- **[03:01]** the sliding-flip picture: flip the second row, slide it; each offset
  aligns the pairs that sum to n. This *is* convolution.
- **[04:02]** non-uniform dice (probabilities aᵢ, bⱼ) -> the general sum-of-pairwise-
  products definition.
- **[05:04]** notation `(a*b)_n = sum_{i+j=n} a_i b_j`; the pairwise-product table with
  diagonal sums as the alternate view.
- **[06:01]** worked example `(1,2,3) * (4,5,6) = (4, 13, 28, 27, 18)`.
- **[07:00]** moving average: a window of 1/5s slides -> smoothed signal; weighted
  windows (central weight) also.
- **[08:03]** 2D analog = image blur: 3x3 grid of 1/9 marches over the image, per-channel
  average. (Animations adapted from his MIT Julia-lab image-processing lectures.)
- **[09:xx]** the 180-degree flip note (pure-math convolution flips the kernel).
- **[10:00]** Gaussian 5x5 kernel -> nicer blur (bell-curve weighting).
- **[10:xx-12:00]** **edge kernel** (positive left, negative right; blue/red); predict-
  first prompt; grayscale, negative outputs, uniform patch -> 0.
- **[12:03]** vertical vs horizontal edges (pi-creature "demonic eyes"); the term
  **kernel**; the CNN line: "use data to figure out what the kernels should be."
- **[12:xx]** output length: valid/same truncation.
- **[13:00]** speed setup: two 100k arrays; `numpy.convolve` averages **4.87 s**.
- **[14:03]** `scipy.fftconvolve` = **4.3 ms** - same answer, 3 orders faster.
- **[15:02]** **polynomial multiplication = convolution of coefficients** (expand +
  collect like terms along diagonals).
- **[16:00]** multiplication is O(n) pointwise but convolution is O(n^2); sampling a
  polynomial at n points specifies its n coefficients.
- **[19:01]** the FFT trick: evaluate at roots of unity (the DFT) -> redundancy -> the
  fast algorithm.
- **[21:03]** full pipeline: FFT both lists -> pointwise multiply -> inverse FFT =
  O(n log n) convolution. Homework: multiplying two integers is a digit convolution.
- **[22:02]** continuous case (probability, why the Gaussian) deferred to a part 2.

## Frame sets (`frames/`)

| file | ts | shows |
|---|---|---|
| f01 | 2:35 | dice grid, 36 pairs, constant-sum diagonals |
| f02 | 3:30 | flip-and-slide dice, P(sum=3) |
| f03 | 4:35 | non-uniform dice, aᵢ·bⱼ product terms |
| f04 | 5:35 | pairwise-product table (sum along diagonals) |
| f05 | 6:47 | worked (1,2,3)∗(4,5,6) flip-slide |
| f06 | 7:47 | moving-average smoothing |
| f07 | 9:05 | 2D box-blur kernel marching over Mario |
| f08 | 10:25 | Gaussian 5x5 kernel blur |
| f09 | 11:02 | edge kernel [0.25,0,-0.25], blue +/red - |
| f10 | 11:48 | pi-creature vertical-edge result |
| f11 | 12:25 | pi-creature horizontal-edge "demonic eyes" |
| f12 | 12:47 | edge result / CNN-learns-kernels moment |
| f13 | 14:02 | numpy.convolve timing (4.87 s) |
| f14 | 14:27 | scipy.fftconvolve timing (4.3 ms) |
| f15 | 15:32 | poly mult = convolution of coefficients |
| f16 | 19:45 | roots of unity / DFT on the complex plane |
| f17 | 21:05 | fast-convolution algorithm (FFT/inverse FFT) |
| f18 | 8:28 | 1D moving-average smoothing (extra) |
| f19 | 9:18 | **the 1/9 weighted-sum calculation** (used in appendix) |
| f20 | 9:32 | Mario blur, kernel lower on the sprite (extra) |
| f21 | 15:48 | poly-mult expansion, boxed (extra) |

**Used in a deck:** `f07` (9:05, kernel marching over Mario) and `f19` (9:18, the 1/9
calculation) were copied into `fig/borrowed/` as `3b1b_mario_kernel.jpg` and
`3b1b_mario_calc.jpg` and embedded as full-bleed stills in `dl_cnn_conv_math.tex`.

## Overlap with L16 (honest audit)

Already in L16 (own house-style figures - do NOT need borrowing):
- dice = convolution of uniforms -> `fig/dice_conv.pdf`
- moving average -> `fig/moving_average.pdf`
- 1D sliding dot product, 2D by hand, kernel zoo (blur/Gaussian/Sobel on the astronaut
  image), edges-as-derivatives -> Section 2 figures
- convolution theorem F{f*g}=F{f}·F{g} + blur/edges as frequency filters -> the v4
  Fourier aside (`taylor_vs_fourier.pdf`, `spectrum_1d.pdf`, `fourier_view.pdf`)

Additive (NOT in L16): the **polynomial-mult -> FFT O(n log n)** story and the concrete
`numpy 4.87 s vs fftconvolve 4.3 ms` speed payoff (frames 13-17).

**DONE (2026-07-21):** added to the optional appendix `dl_cnn_conv_math.tex` (keeps L16
lean, per the instructor's call), as two bonus threads after the LMU cross-correlation
chunk:
- **Bonus 1 - convolution in 2D:** a house frame + two full-bleed 3b1b Mario stills
  (`3b1b_mario_kernel.jpg`, `3b1b_mario_calc.jpg`).
- **Bonus 2 - computing convolutions fast:** poly-mult = convolution of coefficients, the
  `numpy 4.87 s vs fftconvolve 4.3 ms` predict-first frame, and the FFT O(n log n)
  algorithm - all house-authored (no 3b1b figure copied for the math).

Appendix now 89 pages (was 80). L16 untouched.
