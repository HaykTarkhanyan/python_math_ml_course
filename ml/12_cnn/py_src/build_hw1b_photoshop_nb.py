"""Assemble and execute ml/12_cnn/extra_photoshop_solution.ipynb - "Photoshop from scratch":
a self-study notebook on convolution and colour spaces after [48].

Extracted 2026-10-13 from the hand-maintained notebook (its original generator lived in a
session scratchpad and is gone), so the notebook has one source of truth again. See
DECISIONS #82.

Structure rule (instructor direction, 2026-08-02): MANY SMALL CELLS - one idea per cell, a
markdown cell before each code cell saying what is about to happen and why.

Run with the project venv (builds, then executes; measured 2026-10-13: ~4 min, of which ~1 min
is the kernel start):
    ./ma/Scripts/python.exe ml/12_cnn/py_src/build_hw1b_photoshop_nb.py
    ./ma/Scripts/python.exe ml/12_cnn/py_src/build_hw1b_photoshop_nb.py --no-execute
--no-execute overwrites the executed notebook with an output-free one (see _learnings/
2026-10-01-2308_no-execute-rebuild-overwrites-the-executed-notebook.md).
"""
import argparse
import logging
import os
import time
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
WORK = CH
OUT = WORK / "extra_photoshop_solution.ipynb"
LOGS_DIR = REPO_ROOT / "logs"
GH = "https://github.com/HaykTarkhanyan/python_math_ml_course/blob/main/"
COLAB = "https://colab.research.google.com/github/HaykTarkhanyan/python_math_ml_course/blob/main/"
CELLS = []

log = logging.getLogger("build_hw1b_photoshop_nb")


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


def exercise(task, solution):
    # a try-it cell that grades itself, then the solution cell under a heading
    code(task)
    md("**Solution** - read it after you have tried the cell above:")
    code(solution)


def answer(text, label="Answer"):
    # a click-to-open block; markdown inside <details> needs the blank lines
    return f"<details>\n<summary><b>{label}</b></summary>\n\n{text.strip()}\n\n</details>"


def check_yourself(*qa):
    # qa: question, answer, question, answer, ...
    body = "\n\n".join(f"{n}. {q}\n\n{answer(a)}" for n, (q, a) in enumerate(zip(qa[::2], qa[1::2]), 1))
    md("### Check yourself\n\n" + body)


md(rf"""
# HW1b - Photoshop from scratch

**Convolutions and colour spaces** - a self-study notebook for Chapter 12 (CNNs), to work
through after **[48] CNN Foundations**. No PyTorch, no GPU, no training.

By the end you will have built:

1. a working subset of Photoshop's **Filter** menu, one filter at a time,
2. the **blurred-background thumbnail** every lo-fi music upload on YouTube uses,
3. the **Hue/Saturation** adjustment panel.

Along the way, five ideas to keep:

1. **A convolution is one small weighted sum, slid across the image** - the weights decide
   everything.
2. **Blur a single dot and you get the kernel back** - the quickest way to see what any kernel
   does.
3. **Weights summing to 1 keep brightness; weights summing to 0 see only change.**
4. **Vectorising does the same work faster; separability does less work** - only the second
   survives a big kernel.
5. **HSV separates "which colour" from "how bright"**; RGB mixes both into all three numbers.

| Part | | |
|---|---|---|
| 0 | 🧀 | the engine - and what the kernel is actually doing, step by step |
| 1 | 🧀 | the kernel zoo, one filter at a time, with its knobs |
| 2 | 🧀🧀 | the thumbnail generator, built up one layer at a time |
| 3 | 🧀🧀 | HSV - separating "which colour" from "how bright" |

## How to use this notebook

**It comes with every output already in it**, so each answer sits right under its question.
To get the most out of it, **clear all outputs first** ("Clear all outputs", in the Edit menu of
Jupyter and of [Colab]({COLAB}ml/12_cnn/extra_photoshop_solution.ipynb)) and run the cells one at
a time as you read.

- **Predict** prompts ask you to commit to a guess *before* running the next cell. Write it
  down - on paper is fine.
- **Your turn** - seven functions to write yourself. Each has a try-it cell with the function's
  skeleton and checks that tell you whether your version works; the solution follows in the
  next cell. The rest of the notebook runs on the solution, so an unfinished attempt never
  breaks it.
- Each part ends with a **Remember** line and **Check yourself** questions; the answers open
  with a click.

**Time:** about 2 hours to work through. Running every cell takes about 3 minutes on a laptop
CPU (measured: 176 s, plus the kernel start); cells slower than a few seconds say so in their
first line.

**Before you start**, you should have seen [48] CNN Foundations - convolution, padding, stride,
the kernel toolkit ([PDF]({GH}ml/12_cnn/48_cnn_foundations.pdf)) - and [33] Color Spaces for Part
3 ([PDF]({GH}ml/09_clustering/33_color_spaces.pdf)).

### Rules for your own versions

- **No `scipy.signal.convolve2d`, no `cv2`.** You are writing the convolution.
- `numpy`, `matplotlib` and `PIL` only.
- Every number the text quotes comes out of a cell you can run.
""")

code(r'''
# ~6 s the first time (imports)
import time
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.patches import Rectangle
from PIL import Image

T0 = time.perf_counter()
SEED = 509
rng = np.random.default_rng(SEED)
BLUE, RED, ORANGE = "#0033A0", "#D90012", "#F2A800"
plt.rcParams["figure.dpi"] = 110
plt.rcParams["image.cmap"] = "gray"

CANVAS = (720, 1280)          # the thumbnail we are heading towards, in Part 2
RAW = ("https://raw.githubusercontent.com/HaykTarkhanyan/"
       "python_math_ml_course/main/ml/12_cnn/")


def load_image(relpath: str) -> np.ndarray:
    """Load a repo image as float RGB in [0, 1].

    Prefers the local checkout. On Colab (no repo) it downloads once and caches.
    If both routes fail it RAISES - never returns a placeholder.
    """
    local = Path(relpath)
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        url = RAW + relpath.replace("\\", "/")
        req = Request(url, headers={"User-Agent": "python-math-ml-course/hw1b"})
        with urlopen(req, timeout=60) as r:          # raises on 404 / no network
            local.write_bytes(r.read())
        print(f"downloaded {url} -> {local}")
    return np.asarray(Image.open(local).convert("RGB"), dtype=float) / 255.0


def show(images, titles, ncols=None, size=3.0, suptitle=None):
    """Plot a labelled row/grid of images."""
    n = len(images)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * 1.14 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    plt.show()


def to_gray(rgb):
    """Rec. 601 luma - the weights match how the eye weights R, G, B."""
    return rgb @ np.array([0.299, 0.587, 0.114])


def check(name, got, want, tol=1e-9):
    """Grade a "Your turn" cell. None means "not written yet"."""
    if got is None:
        print(f"{name}: not written yet")
        return
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if got.shape != want.shape:
        print(f"{name}: wrong shape - got {got.shape}, expected {want.shape}")
    elif not np.allclose(got, want, atol=tol):
        print(f"{name}: wrong values - off by up to {np.abs(got - want).max():.3g}")
    else:
        print(f"{name}: OK")


print("setup ok")
''')

md(r"""
---
# Part 0 - the engine 🧀

[48] defines 2D convolution as: slide the kernel over the image, and at every stop take the
dot product of the kernel with the window sitting under it.

$$s_{ij}=\sum_{m,n} I(i{+}m{-}1,\;j{+}n{-}1)\,w(m,n)$$

Note there is **no kernel flip**. Strictly this is *cross-correlation*, and it is what every
CNN library actually computes - [48]'s footnote covers why the distinction stops mattering
once the weights are learned. Here they are not learned, so match the slides exactly.
""")

md(r"""
### Task 0.1 - write it 🧀

The direct transcription of the formula: loop over output pixels, and for each one sum the
`k x k` window times the kernel.

Support the three cases from [48] Section 4:

- `padding="valid"` - no padding, the map shrinks
- `padding="same"` - pad by `k // 2` so the map keeps its size
- `stride=s` - move the kernel `s` pixels at a time

Output size is $o = \lfloor (i - k + 2p)/s \rfloor + 1$.

**Your turn:** fill in `my_convolve2d_naive` below. The three checks under it tell you whether
it works - the second one is the worked example from the [48] slides.
""")

exercise(r"""
def my_convolve2d_naive(img, kernel, padding="same", stride=1, pad_mode="edge"):
    # 1. if padding == "same", pad the image by k // 2 on every side
    #    (np.pad with mode="edge", or mode="constant" for zeros)
    # 2. output size: o = (i - k + 2p) // s + 1, on each axis
    # 3. two nested loops over the output; each output cell = np.sum(window * kernel)
    return None   # replace with your output array


_I = np.array([[1, 2, 0], [3, 1, 1], [0, 2, 4]], dtype=float)
check("identity kernel, 'same'", my_convolve2d_naive(_I, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])), _I)
check("the [48] slide example, 'valid'",
      my_convolve2d_naive(_I, np.array([[0, 1], [2, 2]]), padding="valid"), [[10, 4], [5, 13]])
check("stride 2 on a 6x6, 'valid'",
      my_convolve2d_naive(np.arange(36.0).reshape(6, 6), np.ones((3, 3)), padding="valid", stride=2),
      [[63, 81], [171, 189]])
""", r'''
def _pad(img, pad, mode):
    if pad == 0:
        return img
    return np.pad(img, ((pad, pad), (pad, pad)),
                  mode="edge" if mode == "edge" else "constant")


def convolve2d_naive(img, kernel, padding="same", stride=1, pad_mode="edge"):
    """Loop over OUTPUT PIXELS. The definition, written out. Correct and slow."""
    img = np.asarray(img, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    assert kh == kw, "keep kernels square in this homework"
    pad = kh // 2 if padding == "same" else 0
    p = _pad(img, pad, pad_mode)

    out_h = (p.shape[0] - kh) // stride + 1
    out_w = (p.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w), dtype=float)
    for y in range(out_h):
        for x in range(out_w):
            win = p[y * stride:y * stride + kh, x * stride:x * stride + kw]
            out[y, x] = np.sum(win * kernel)
    return out


print("convolve2d_naive defined")
''')

md(r"""
### Task 0.2 - watch it slide 🧀

Before running this on a photo, watch what the kernel does on something small enough to read.

Below is a **6x6 image** with a vertical edge in it: dark on the left, bright on the right.
The kernel is **Sobel X**, the vertical-edge detector.

At every stop you see four things: the window under the kernel, the kernel itself, the two
multiplied cell by cell, and the single number that lands in the output.

**Predict:** Sobel X has a negative left column and a positive right column. At which of the
four stops along the top row will the output be largest - and what will it be at the first
stop, where the window sits entirely in the dark region?
""")

code(r'''
def grid_plot(ax, arr, title, highlight=None, cmap="gray", vmin=None, vmax=None,
              fmt="{:.0f}", fontsize=9, textcolor=None):
    """Draw a small matrix as a heatmap with the numbers written in each cell."""
    ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=vmin, vmax=vmax)
    h, w = arr.shape
    for y in range(h):
        for x in range(w):
            v = arr[y, x]
            if not np.isfinite(v):        # not computed yet - leave the cell blank
                continue
            if textcolor is None:
                lo = vmin if vmin is not None else arr.min()
                hi = vmax if vmax is not None else arr.max()
                rel = 0.5 if hi == lo else (v - lo) / (hi - lo)
                col = "white" if rel < 0.5 else "black"
            else:
                col = textcolor
            ax.text(x, y, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize, color=col)
    if highlight is not None:
        y0, x0, kh, kw = highlight
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), kw, kh, fill=False,
                               edgecolor="#D90012", lw=2.6))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


# a 6x6 image with a vertical edge: dark left, bright right
IMG6 = np.full((6, 6), 20.0)
IMG6[:, 3:] = 200.0
IMG6[1, 1] = 60.0          # a little texture so the flat region is not perfectly flat
IMG6[4, 4] = 160.0

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

print("input image (6x6):"); print(IMG6.astype(int))
print("\nSobel X kernel:"); print(SOBEL_X.astype(int))
''')

code(r"""
# walk the kernel across the top row of output positions, 'valid' padding
positions = [(0, 0), (0, 1), (0, 2), (0, 3)]
out_valid = convolve2d_naive(IMG6, SOBEL_X, padding="valid")

fig, axes = plt.subplots(len(positions), 4, figsize=(13, 2.9 * len(positions)))
for row, (oy, ox) in enumerate(positions):
    win = IMG6[oy:oy + 3, ox:ox + 3]
    prod = win * SOBEL_X
    running = np.full_like(out_valid, np.nan)
    running[:oy, :] = out_valid[:oy, :]
    running[oy, :ox + 1] = out_valid[oy, :ox + 1]

    grid_plot(axes[row, 0], IMG6, f"image, window at ({oy},{ox})",
              highlight=(oy, ox, 3, 3), vmin=0, vmax=255)
    grid_plot(axes[row, 1], SOBEL_X, "kernel (Sobel X)", cmap="coolwarm",
              vmin=-2, vmax=2, textcolor="black")
    grid_plot(axes[row, 2], prod, f"window x kernel  ->  sum = {prod.sum():.0f}",
              cmap="coolwarm", vmin=-450, vmax=450, textcolor="black")
    grid_plot(axes[row, 3], running, "output so far (blank = not yet computed)",
              cmap="coolwarm", vmin=-900, vmax=900, textcolor="black",
              highlight=(oy, ox, 1, 1))
fig.suptitle("One kernel, four stops. The red box is where the kernel is standing.",
             fontsize=12, y=1.002)
fig.tight_layout(); plt.show()
""")

md(r"""
Read the third column. The kernel's left column is negative and its right column is positive,
so the sum is:

- **near zero** where the window sits entirely inside a flat region - the negatives and the
  positives cancel,
- **large** where the window straddles the edge - the bright side survives, the dark side
  contributes almost nothing.

That is the entire idea of an edge detector, and you can read it off six numbers. It is also
why the kernel's weights **summing to zero** matters: a zero-sum kernel is blind to flat
brightness and only reports change.
""")

code(r"""
print("full 'valid' output (4x4):")
print(convolve2d_naive(IMG6, SOBEL_X, padding="valid").astype(int))
print("\nColumn 2 is the edge (between input columns 2 and 3). The rest is near zero.")
print("\nWhat the same kernel gives on a completely flat image:")
print(convolve2d_naive(np.full((6, 6), 128.0), SOBEL_X, padding="valid").astype(int))
""")

md(r"""
### Task 0.3 - padding, in 2D 🧀

A `3x3` kernel on a `6x6` image gives a `4x4` output - the map **shrinks**, because the kernel
cannot hang off the edge. Padding is how you keep the size.

Two choices, and they are not equivalent:

- **zeros** - honest but it invents a black border, so a blur darkens the edges,
- **edge replication** - copies the border pixel outwards, so a blur leaves them alone.

Look at what each does to the *corner values* below.

**Predict:** run a 3x3 box blur (the plain average of the window) at the top-left corner. Which
padding gives the darker corner pixel?
""")

code(r"""
pz = np.pad(IMG6, ((1, 1), (1, 1)), mode="constant")
pe = np.pad(IMG6, ((1, 1), (1, 1)), mode="edge")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
grid_plot(axes[0], IMG6, "original 6x6", vmin=0, vmax=255)
grid_plot(axes[1], pz, "pad 1 with ZEROS -> 8x8", vmin=0, vmax=255)
grid_plot(axes[2], pe, "pad 1 with EDGE -> 8x8", vmin=0, vmax=255)
fig.tight_layout(); plt.show()

box3 = np.ones((3, 3)) / 9
z = convolve2d_naive(IMG6, box3, padding="same", pad_mode="zero")
e = convolve2d_naive(IMG6, box3, padding="same", pad_mode="edge")
print("3x3 box blur, 'same' padding. Top-left 3x3 corner of the result:\n")
print("zero padding:\n", z[:3, :3].astype(int))
print("\nedge padding:\n", e[:3, :3].astype(int))
print(f"\nThe corner pixel is {z[0,0]:.0f} with zeros and {e[0,0]:.0f} with edge replication.")
print("Edge replication averages pixels the image really has (the 20s, plus the 60 next to them).")
print("Zero padding mixes 5 invented black pixels into the corner window - a dark rim that is not in the image.")
""")

md(r"""
### Task 0.4 - stride, in 2D 🧀

Stride is how far the kernel jumps between stops. `stride=1` visits every pixel; `stride=2`
visits every other one, so the output is roughly half the size on each axis.

The formula $o = \lfloor (i - k + 2p)/s \rfloor + 1$ is just bookkeeping for that.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
for ax, st in zip(axes, (1, 2, 3)):
    o = convolve2d_naive(IMG6, box3, padding="valid", stride=st)
    grid_plot(ax, o, f"stride={st}  ->  output {o.shape[0]}x{o.shape[1]}",
              vmin=0, vmax=255, fontsize=8)
fig.suptitle("Same image, same kernel, three strides", fontsize=12, y=1.02)
fig.tight_layout(); plt.show()

print(f"{'k':>3}{'padding':>9}{'stride':>8}{'output':>10}{'formula':>10}")
for (k, pad, st) in [(3, "same", 1), (3, "valid", 1), (3, "valid", 2),
                     (5, "same", 1), (5, "same", 3)]:
    ker = np.ones((k, k)) / (k * k)
    got = convolve2d_naive(IMG6, ker, padding=pad, stride=st).shape
    p = k // 2 if pad == "same" else 0
    want = ((6 - k + 2 * p) // st + 1,) * 2
    assert got == want, f"k={k} {pad} stride={st}: got {got}, formula says {want}"
    print(f"{k:>3}{pad:>9}{st:>8}{str(got):>10}{str(want):>10}")
print("\nformula matches the implementation in all 5 cases  OK")
""")

md(r"""
### Task 0.5 - prove the engine 🧀

Two checks, both `assert`s. If the engine is wrong the notebook stops here rather than
producing pretty but meaningless pictures for the next hour.
""")

code(r"""
# 1. the identity kernel must be a no-op
_img = rng.random((16, 16))
_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
assert np.allclose(convolve2d_naive(_img, _identity), _img), "identity must be a no-op"

# 2. the worked example from the [48] slide - through YOUR function, not a special case
I = np.array([[1, 2, 0], [3, 1, 1], [0, 2, 4]], dtype=float)
W = np.array([[0, 1], [2, 2]], dtype=float)
S_expected = np.array([[10, 4], [5, 13]], dtype=float)
S = convolve2d_naive(I, W, padding="valid")
assert np.array_equal(S, S_expected), f"expected {S_expected.tolist()}, got {S.tolist()}"

print("identity kernel  OK")
print(f"[48] slide example OK  ->  S = {S.astype(int).tolist()}")
""")

md(r"""
### Task 0.6 - the same thing, but fast 🧀

`convolve2d_naive` runs a Python loop once per output pixel. On a megapixel photo that is a
million iterations before any arithmetic happens.

Flip the loops: instead of looping over pixels and touching `k x k` weights, loop over the
**`k x k` weights** and let NumPy apply each one to the whole image at once. A 3x3 kernel
becomes 9 whole-image operations instead of a million small ones.

**This is not a better algorithm** - it performs exactly the same multiply-adds. It only
moves the loop from Python into NumPy's C. Hold on to that distinction; Part 2 shows a change
that *is* algorithmic, and one that is not.

**Your turn:** fill in `my_convolve2d_fast`. The checks compare it with `convolve2d_naive` on a
random image and kernel.
""")

exercise(r"""
def my_convolve2d_fast(img, kernel, padding="same", stride=1, pad_mode="edge"):
    # pad and size the output exactly as in convolve2d_naive, then loop over the k x k TAPS:
    # for each weight w at (i, j), add w * (the padded image shifted by i rows and j columns,
    # cut to the output size) to the output. With a stride, step that slice by `stride`.
    return None   # replace with your output array


_x = np.random.default_rng(SEED).random((12, 12))
_k5 = np.random.default_rng(SEED + 1).random((5, 5))
for _pad_kind, _st in [("same", 1), ("valid", 1), ("valid", 2)]:
    check(f"matches convolve2d_naive, padding={_pad_kind}, stride={_st}",
          my_convolve2d_fast(_x, _k5, padding=_pad_kind, stride=_st),
          convolve2d_naive(_x, _k5, padding=_pad_kind, stride=_st), tol=1e-12)
""", r'''
# ~6 s (it times both engines)
def convolve2d_fast(img, kernel, padding="same", stride=1, pad_mode="edge"):
    """Loop over KERNEL TAPS, vectorised over the image. Same answer, much faster."""
    img = np.asarray(img, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    pad = kh // 2 if padding == "same" else 0
    p = _pad(img, pad, pad_mode)
    out_h = (p.shape[0] - kh) // stride + 1
    out_w = (p.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w), dtype=float)
    for i in range(kh):
        for j in range(kw):
            w = kernel[i, j]
            if w == 0.0:
                continue
            out += w * p[i:i + (out_h - 1) * stride + 1:stride,
                         j:j + (out_w - 1) * stride + 1:stride]
    return out


_k = rng.random((5, 5)); _k /= _k.sum()
for pad in ("same", "valid"):
    for st in (1, 2):
        a = convolve2d_naive(_img, _k, padding=pad, stride=st)
        b = convolve2d_fast(_img, _k, padding=pad, stride=st)
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-12), \
            f"engines disagree at padding={pad} stride={st}"
print("naive and fast agree to 1e-12 across padding and stride  OK")

img_t = rng.random((400, 400))
for k in (3, 9):
    ker = np.ones((k, k)) / (k * k)
    t0 = time.perf_counter(); convolve2d_naive(img_t, ker); tn = time.perf_counter() - t0
    t0 = time.perf_counter(); convolve2d_fast(img_t, ker); tf = time.perf_counter() - t0
    print(f"k={k}: naive {tn:6.3f}s   fast {tf:6.3f}s   ({tn/tf:5.0f}x)")
''')

md(r"""
> **Remember:** a convolution is the same small weighted sum at every position. Padding decides
> what happens at the border, stride decides how often you stop - and how fast it runs depends
> on where the loop lives, not on the arithmetic.
""")

check_yourself(
    "A 5x5 kernel, `valid` padding, stride 2, on a 21x21 image. What size is the output?",
    "(21 - 5 + 0) // 2 + 1 = 9, so 9x9.",
    "Sobel X gives exactly 0 everywhere on a flat image. Which property of its weights guarantees "
    "that?",
    "They sum to zero: on a flat patch every weight multiplies the same value, so the sum is "
    "(sum of weights) x value = 0.",
    "`convolve2d_fast` was far faster than `convolve2d_naive` at k=3. Does it do fewer "
    "multiplications?",
    "No - exactly the same ones. It runs 9 whole-image NumPy operations instead of a Python loop "
    "over every pixel. Part 2 shows when that trick stops paying.")

md(r"""
---
# Part 1 - the kernel zoo, one filter at a time 🧀

Every filter below is a real Photoshop menu item, and they are not *analogies* for it - they
are the same arithmetic.

| Photoshop menu | what it is |
|---|---|
| Filter > Blur > Box Blur | `ones((k,k)) / k**2` |
| Filter > Blur > Gaussian Blur | a sampled 2D Gaussian |
| Filter > Blur > Motion Blur | a 1-D line, at an angle |
| Filter > Sharpen > Sharpen | unsharp mask |
| Filter > Stylize > Emboss | asymmetric 3x3 |
| Filter > Stylize > Find Edges | Sobel magnitude |
| Filter > Other > Custom | **literally a 5x5 kernel entry grid**, plus scale and offset |

That last row is worth sitting with. Photoshop's *Custom* dialog asks you to type 25 numbers
into a grid, divide by a scale and add an offset. You are not modelling that feature. You are
reimplementing it.

We take these **one at a time**, and for each one we turn its knob to see what the knob does.
""")

code(r"""
pom_full = load_image("fig/src_pomegranate.jpg")
# a smaller working copy keeps the pure-numpy convolutions responsive
pom = np.asarray(Image.fromarray((pom_full * 255).astype(np.uint8)).resize((480, 360),
                 Image.LANCZOS), float) / 255
pom_g = to_gray(pom)
print(f"working image: {pom.shape} (from {pom_full.shape})")
show([pom, pom_g], ["colour", "grayscale (Rec. 601 luma)"], ncols=2, size=4.0)
""")

md(r"""
## 1.1 Box Blur - the simplest possible kernel 🧀

Every weight identical, summing to 1. Each output pixel is the plain average of the `k x k`
square around it.

The knob is `k`. Watch two things as it grows: the image gets softer, and the softening has a
**shape** - a box kernel smears every point into a little square. On a photo that is hard to
see; after 1.2 you get a way to see it directly.

**Your turn:** write `my_box_kernel(k)`.
""")

exercise(r"""
def my_box_kernel(k):
    # a k x k kernel, every weight the same, all summing to 1
    return None


check("box_kernel(3)", my_box_kernel(3), np.full((3, 3), 1 / 9))
check("box_kernel(5)", my_box_kernel(5), np.full((5, 5), 1 / 25))
""", r"""
# ~8 s
def box_kernel(k):
    return np.ones((k, k)) / (k * k)


fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, k in zip(axes, (3, 5, 9, 15)):
    grid_plot(ax, box_kernel(k), f"box kernel k={k}", cmap="viridis",
              fmt="{:.3f}" if k <= 5 else "", fontsize=7)
fig.suptitle("The kernels themselves: every weight the same, all summing to 1",
             fontsize=12, y=1.03)
fig.tight_layout(); plt.show()

ks = (3, 9, 21, 41)
outs = [convolve2d_fast(pom_g, box_kernel(k)) for k in ks]
show([pom_g] + outs, ["original"] + [f"box blur, k={k}" for k in ks], ncols=5, size=2.7,
     suptitle="Turning the box-blur knob")
for k, o in zip(ks, outs):
    print(f"k={k:>3}: {k*k:>5,} taps per pixel,  "
          f"local contrast left = {np.std(pom_g - o):.4f}")
""")

md(r"""
## 1.2 Gaussian Blur - the blur that has no direction 🧀

A box blur weights a pixel 20 away exactly as much as its neighbour, and stops abruptly. A
**Gaussian** falls off smoothly and is *rotationally symmetric*: it has no preferred direction,
so it does not smear things into squares.

The knob is `sigma`, the width of the bell. The kernel size follows from it - you need about
`±3 sigma` of it before the weights are negligible, so `k = 6*sigma + 1`.

Look at the kernel heatmaps: a **round** blob, not a square one.

**Your turn:** write `my_gaussian_kernel(sigma)`.
""")

exercise(r"""
def my_gaussian_kernel(sigma, truncate=3.0):
    # r = int(truncate * sigma); on the grid x, y = -r .. r sample exp(-(x^2 + y^2) / (2 sigma^2)),
    # then divide by the sum so the weights add up to 1
    return None


_g1 = np.exp(-np.arange(-3, 4) ** 2 / 2.0)
check("gaussian_kernel(1)", my_gaussian_kernel(1), np.outer(_g1, _g1) / np.outer(_g1, _g1).sum())
check("gaussian_kernel(2) is 13x13", None if my_gaussian_kernel(2) is None else my_gaussian_kernel(2).shape,
      (13, 13))
""", r'''
# ~11 s
def gaussian_kernel(sigma, truncate=3.0):
    """2D Gaussian, truncated at +/- truncate*sigma, normalised to sum 1."""
    r = int(truncate * sigma)
    ax = np.arange(-r, r + 1)
    g1 = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g1, g1)
    return k / k.sum()


fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, s in zip(axes, (1, 2, 3, 5)):
    k = gaussian_kernel(s)
    grid_plot(ax, k, f"sigma={s}  ->  {k.shape[0]}x{k.shape[0]}", cmap="viridis",
              fmt="", fontsize=6)
fig.suptitle("Gaussian kernels: round, smooth, and they grow with sigma", fontsize=12, y=1.03)
fig.tight_layout(); plt.show()

sigmas = (1, 2, 4, 8)
gouts = [convolve2d_fast(pom_g, gaussian_kernel(s)) for s in sigmas]
show([pom_g] + gouts, ["original"] + [f"gaussian, sigma={s}" for s in sigmas],
     ncols=5, size=2.7, suptitle="Turning the Gaussian knob")
''')

md(r"""
### Box vs Gaussian, compared fairly

How do you compare two blurs? Not by kernel size: a 25x25 box blurs far more than a 25x25
Gaussian, whose weight sits mostly in the middle. Match them on **blur strength** instead - the
spread (standard deviation) of the weights. A box of width $k$ has spread
$\sqrt{(k^2-1)/12}$; for $k = 15$ that is 4.3, so the fair partner is a Gaussian with
$\sigma = 4.3$.

And how do you *see* the difference? **Blur a single white pixel.** Every output pixel within
reach of that dot gathers it with the weight the kernel puts there, so the blurred dot is
**the kernel itself**. (Strictly, the kernel rotated 180 degrees - the cross-correlation vs
convolution difference from Part 0. For a symmetric kernel the two are the same; 1.6 shows one
that is not.)

**Predict:** what shape does each blur turn the dot into?
""")

code(r"""
K_BOX = 15
SIGMA_MATCH = np.sqrt((K_BOX ** 2 - 1) / 12)
dot = np.zeros((41, 41))
dot[20, 20] = 1.0
blurred_dots = {f"box, k={K_BOX}": convolve2d_fast(dot, box_kernel(K_BOX)),
                f"gaussian, sigma={SIGMA_MATCH:.1f}": convolve2d_fast(dot, gaussian_kernel(SIGMA_MATCH))}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
axes[0].imshow(dot, interpolation="nearest")
axes[0].set_title("one white pixel", fontsize=10)
for ax, (name, r) in zip(axes[1:], blurred_dots.items()):
    ax.imshow(r / r.max(), interpolation="nearest")
    ax.set_title(f"{name}: the dot, blurred", fontsize=10)
for ax in axes:
    ax.axis("off")
fig.tight_layout(); plt.show()
""")

md(r"""
A square with a hard edge, and a soft round disc. That is what each blur does to *every* point
of an image: a box blur turns each small bright spot into a little square, a Gaussian into a
soft disc. Now the photo, at matched strength:
""")

code(r"""
# ~4 s
b = convolve2d_fast(pom_g, box_kernel(K_BOX))
g = convolve2d_fast(pom_g, gaussian_kernel(SIGMA_MATCH))
show([pom_g, b, g, np.clip(np.abs(b - g) * 10, 0, 1)],
     ["original", f"box, k={K_BOX}", f"gaussian, sigma={SIGMA_MATCH:.1f}", "|difference| x10"],
     ncols=4, size=3.0)
print(f"matched on strength, the two blurs differ by {np.abs(b - g).mean():.4f} on average "
      f"(at most {np.abs(b - g).max():.3f})")
""")

md(r"""
At matched strength the two photos look almost the same: **how much** a blur blurs is set by the
spread of its weights, not by the size of the kernel. What differs is the **shape** each point is
smeared into - the difference image lights up only along edges and small bright spots. A
Gaussian has no corners and no preferred direction, which is why it is the default blur.
""")

md(r"""
## 1.3 The 2D kernel is an outer product 🎁🧀🧀

Here is a fact about the **2D** kernel that Part 2 depends on.

$$G(x,y)=\frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}
        =\underbrace{\tfrac{1}{\sqrt{2\pi}\sigma}e^{-\frac{x^2}{2\sigma^2}}}_{G(x)}
         \cdot\underbrace{\tfrac{1}{\sqrt{2\pi}\sigma}e^{-\frac{y^2}{2\sigma^2}}}_{G(y)}$$

The exponent splits because $x^2+y^2$ is a sum, so the 2D kernel is literally the **outer
product** of a 1D kernel with itself. Not an approximation - the same numbers.

The cell below builds a 2D Gaussian both ways and checks they are identical to machine
precision. A kernel with this property is called **separable**, and box blur is separable too
(`ones(k,k)/k**2 = outer(ones(k)/k, ones(k)/k)`).

**Your turn:** write `my_gaussian_1d(sigma)`. The check builds the outer product of your 1D
kernel with itself and compares it with `gaussian_kernel`.
""")

exercise(r"""
def my_gaussian_1d(sigma, truncate=3.0):
    # like gaussian_kernel, but in one dimension: sample exp(-x^2 / (2 sigma^2)) for
    # x = -r .. r with r = int(truncate * sigma), then divide by the sum
    return None


_g = my_gaussian_1d(2.0)
check("outer(gaussian_1d(2), gaussian_1d(2)) == gaussian_kernel(2)",
      None if _g is None else np.outer(_g, _g), gaussian_kernel(2.0), tol=1e-12)
""", r"""
def gaussian_1d(sigma, truncate=3.0):
    r = int(truncate * sigma)
    ax = np.arange(-r, r + 1)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return g / g.sum()


s = 2.0
g1 = gaussian_1d(s)
K2 = gaussian_kernel(s)
outer = np.outer(g1, g1)

fig = plt.figure(figsize=(12.5, 3.6))
ax1 = fig.add_subplot(1, 4, 1)
ax1.stem(np.arange(len(g1)) - len(g1) // 2, g1)
ax1.set_title(f"1D Gaussian, sigma={s:.0f}  ({len(g1)} taps)", fontsize=10)
grid_plot(fig.add_subplot(1, 4, 2), outer, "outer(g, g)", cmap="viridis",
          vmin=0, vmax=K2.max(), fmt="", fontsize=6)
grid_plot(fig.add_subplot(1, 4, 3), K2, "the 2D kernel", cmap="viridis",
          vmin=0, vmax=K2.max(), fmt="", fontsize=6)
# NOTE the shared colour scale: autoscaling this last panel would turn 1e-16 rounding
# noise into a dramatic pattern and make identical arrays look different.
grid_plot(fig.add_subplot(1, 4, 4), np.abs(outer - K2),
          f"|difference|, same scale\nmax = {np.abs(outer - K2).max():.0e}",
          cmap="viridis", vmin=0, vmax=K2.max(), fmt="", fontsize=6)
fig.tight_layout(); plt.show()

print(f"max |outer(g,g) - 2D kernel| = {np.abs(outer - K2).max():.2e}")
assert np.allclose(outer, K2, atol=1e-15), "the 2D Gaussian must be an outer product"
print("the 2D Gaussian IS the outer product of two 1D Gaussians  OK")
print(f"\ntaps: 2D kernel = {K2.size}, two 1D passes = {2*len(g1)}"
      f"  ->  {K2.size/(2*len(g1)):.1f}x fewer")
""")

md(r"""
So blurring with the 2D kernel can be done as **two 1D passes**: blur every row with the 1D
kernel, then blur every column of the result. `convolve2d_fast` assumes a square kernel, so the
helper below (given, not an exercise) does the same tap loop for a `1 x k` row or a `k x 1`
column.
""")

code(r'''
def convolve2d_rect(img, kernel, pad_mode="edge"):
    """Tap loop allowing NON-square kernels - a (1 x k) row or a (k x 1) column.

    convolve2d_fast pads kh//2 on both axes, which is only right for a square kernel.
    """
    img = np.asarray(img, float); kernel = np.asarray(kernel, float)
    kh, kw = kernel.shape
    p = np.pad(img, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
    out = np.zeros_like(img, dtype=float)
    H, W = img.shape
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j]:
                out += kernel[i, j] * p[i:i + H, j:j + W]
    return out
''')

md(r"""
**Your turn:** write `my_gaussian_blur_separable(img, sigma)` for a grayscale image: a row pass
with the 1D kernel shaped as a row (`g[None, :]`), then a column pass with it shaped as a column
(`g[:, None]`). The check compares it with the full 2D blur.
""")

exercise(r"""
def my_gaussian_blur_separable(img, sigma, truncate=3.0):
    # g = gaussian_1d(sigma, truncate); then convolve2d_rect twice: once with the row, once with
    # the column
    return None


_t = np.random.default_rng(SEED).random((30, 40))
check("separable blur == full 2D blur, sigma=2", my_gaussian_blur_separable(_t, 2.0),
      convolve2d_fast(_t, gaussian_kernel(2.0)), tol=1e-12)
""", r'''
# ~5 s
def gaussian_blur_separable(img, sigma, truncate=3.0):
    """Blur with a row pass, then a column pass. Works on 2D or HxWx3."""
    g = gaussian_1d(sigma, truncate)
    if img.ndim == 3:
        return np.stack([gaussian_blur_separable(img[..., c], sigma, truncate)
                         for c in range(img.shape[2])], axis=-1)
    return convolve2d_rect(convolve2d_rect(img, g[None, :]), g[:, None])


# the two passes, shown separately, so "separable" is something you can see
row_only = convolve2d_rect(pom_g, gaussian_1d(6)[None, :])
col_only = convolve2d_rect(pom_g, gaussian_1d(6)[:, None])
both = gaussian_blur_separable(pom_g, 6)
full2d = convolve2d_fast(pom_g, gaussian_kernel(6))
show([pom_g, row_only, both, full2d],
     ["original", "pass 1: horizontal only", "pass 2: then vertical",
      "the full 2D kernel, for comparison"], ncols=4, size=3.0)
print(f"max |separable - full 2D| = {np.abs(both - full2d).max():.2e}  -> identical")
''')

md(r"""
## 1.4 Motion Blur - a kernel with a direction 🧀

Blur along a line instead of a disc, and you get the streak of a moving camera. This kernel is
**not** rotationally symmetric, and that is the whole point.

Two knobs: the **length** of the streak and its **angle**.
""")

code(r"""
# ~4 s
def motion_blur_kernel(length=15, angle_deg=45.0):
    k = np.zeros((length, length))
    c = length // 2
    t = np.deg2rad(angle_deg)
    for d in np.linspace(-c, c, length * 4):
        y = int(round(c - d * np.sin(t)))
        x = int(round(c + d * np.cos(t)))
        if 0 <= y < length and 0 <= x < length:
            k[y, x] = 1.0
    return k / k.sum()


angles = (0, 45, 90, 135)
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
for ax, a in zip(axes, angles):
    grid_plot(ax, motion_blur_kernel(15, a), f"angle = {a} deg", cmap="viridis",
              fmt="", fontsize=6)
fig.suptitle("Motion-blur kernels - the line IS the kernel", fontsize=12, y=1.03)
fig.tight_layout(); plt.show()

show([pom_g] + [convolve2d_fast(pom_g, motion_blur_kernel(21, a)) for a in angles],
     ["original"] + [f"{a} deg, length 21" for a in angles], ncols=5, size=2.7,
     suptitle="Same length, four directions")
show([pom_g] + [convolve2d_fast(pom_g, motion_blur_kernel(L, 45)) for L in (7, 15, 31)],
     ["original"] + [f"length {L}, 45 deg" for L in (7, 15, 31)], ncols=4, size=3.0,
     suptitle="Same direction, three lengths")
""")

md(r"""
The dot test from 1.2 again. **Predict:** what does a motion-blur kernel turn a single white
pixel into?
""")

code(r"""
streaks = [convolve2d_fast(dot, motion_blur_kernel(15, a)) for a in angles]
show([s / s.max() for s in streaks], [f"the dot, motion blur {a} deg" for a in angles],
     ncols=4, size=2.8)
""")

md(r"""
A streak along the kernel's angle - the streak **is** the kernel. That is what a moving camera
does to every bright point in the scene, which is why motion blur looks like it does.
""")

md(r"""
## 1.5 Sharpen - the filter that is secretly a blur 🧀

Photoshop's sharpen is an **unsharp mask**, and the name gives it away - it sharpens by
subtracting a *blurred* copy:

$$\text{sharp} = \text{orig} + \text{amount}\cdot(\text{orig} - \text{blur}(\text{orig}))$$

**Why does subtracting a blur sharpen?** `orig - blur` is exactly the detail the blur threw
away - a **high-pass** filter. Adding it back amplifies precisely the frequencies the blur
attenuated. That is [48]'s Fourier aside, in two lines of code.

Two knobs: the blur's `sigma` (which *scale* of detail you boost) and `amount` (how hard).

First on the simplest possible image: one row of pixels crossing a clean edge, from dark (0.25)
to light (0.75). **Predict:** after sharpening, does the edge still run from 0.25 to 0.75?
""")

code(r"""
edge = np.where(np.arange(80) < 40, 0.25, 0.75)
g_edge = gaussian_1d(3)
r_edge = len(g_edge) // 2
edge_blur = np.convolve(np.pad(edge, r_edge, mode="edge"), g_edge, mode="valid")
edge_detail = edge - edge_blur
edge_sharp = edge + 1.5 * edge_detail

fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.7), sharey=True)
axes[0].plot(edge, color="k", lw=2, label="original edge")
axes[0].plot(edge_blur, color=BLUE, lw=2, label="blurred (sigma = 3)")
axes[1].plot(edge_detail, color=ORANGE, lw=2, label="detail = original - blurred")
axes[1].plot(edge_sharp, color=RED, lw=2, label="sharpened = original + 1.5 x detail")
for ax in axes:
    for level in (0.25, 0.75):
        ax.axhline(level, color="0.6", ls=":", lw=1)
    ax.set_xlabel("position along the row (pixels)")
    ax.legend(fontsize=9, loc="upper left")
axes[0].set_ylabel("brightness")
fig.tight_layout(); plt.show()
print(f"sharpened edge runs from {edge_sharp.min():.3f} to {edge_sharp.max():.3f}")
""")

md(r"""
No - it **overshoots** on both sides: a dip just before the edge and a peak just after it. That
overshoot is the **halo** you see around every over-sharpened edge: a dark rim on the dark side,
a bright rim on the light side. Here, at `amount = 1.5`, it already runs past 1 and below 0,
where a real image has to clip. The detail curve (orange) shows why: `orig - blur` is zero on
flat ground and a two-sided bump at the edge - a high-pass filter - and sharpening adds that
bump back on top.

Now the same thing on the photo:
""")

code(r"""
# ~13 s
detail = pom_g - convolve2d_fast(pom_g, gaussian_kernel(2))
show([pom_g, convolve2d_fast(pom_g, gaussian_kernel(2)), detail + 0.5],
     ["original", "blurred (sigma=2)", "orig - blur = the detail  (+0.5 to see it)"],
     ncols=3, size=3.4)

amounts = (0.5, 1.5, 3.0, 6.0)
show([pom_g] + [np.clip(pom_g + a * detail, 0, 1) for a in amounts],
     ["original"] + [f"amount = {a}" for a in amounts], ncols=5, size=2.7,
     suptitle="Turning the sharpen knob - watch the halos appear")

print("over-sharpening is measurable, not just visible:")
for a in amounts:
    raw = pom_g + a * detail
    print(f"  amount={a:>4}:  {100*np.mean((raw < 0) | (raw > 1)):5.2f}% of pixels "
          f"pushed outside [0,1] and clipped")

# which scale of detail you boost
show([pom_g] + [np.clip(pom_g + 2.0 * (pom_g - convolve2d_fast(pom_g, gaussian_kernel(s))), 0, 1)
                for s in (1, 3, 8)],
     ["original"] + [f"amount=2, sigma={s}" for s in (1, 3, 8)], ncols=4, size=3.0,
     suptitle="Same amount, different detail scale")
""")

md(r"""
## 1.6 Emboss - a directional difference 🧀

Emboss subtracts one side of each pixel from the other along a diagonal, producing a
lit-from-one-side relief. Rotate the kernel and the light moves.

Note this kernel sums to **1**, not 0, so flat regions keep their brightness instead of going
black. That is a deliberate choice - compare with Sobel below.
""")

code(r"""
EMBOSS = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float)
rots = [("top-left light", EMBOSS),
        ("top-right", np.rot90(EMBOSS, 1)),
        ("bottom-right", np.rot90(EMBOSS, 2)),
        ("bottom-left", np.rot90(EMBOSS, 3))]

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, (name, k) in zip(axes, rots):
    grid_plot(ax, k, f"{name}  (sum={k.sum():.0f})", cmap="coolwarm",
              vmin=-2, vmax=2, textcolor="black")
fig.tight_layout(); plt.show()

show([pom_g] + [convolve2d_fast(pom_g, k) for _, k in rots],
     ["original"] + [n for n, _ in rots], ncols=5, size=2.7,
     suptitle="Emboss: rotating the kernel moves the light source")
""")

md(r"""
**The dot test on a kernel that is not symmetric.** 1.2 promised that the blurred dot is the
kernel *rotated 180 degrees*. For every blur so far that made no difference. Emboss is lopsided,
so here it shows. **Predict:** will the dot come out as `EMBOSS`, or upside down?
""")

code(r"""
dot7 = np.zeros((7, 7))
dot7[3, 3] = 1.0
response = convolve2d_fast(dot7, EMBOSS)[2:5, 2:5]
print("the kernel:\n", EMBOSS.astype(int))
print("\nthe dot after the emboss filter (centre 3x3):\n", response.astype(int))
assert np.array_equal(response, np.rot90(EMBOSS, 2))
print("\nthe dot's response is the kernel rotated 180 degrees  OK")
""")

md(r"""
Upside down. Our engine *gathers*: each output pixel reads its neighbours through the kernel. A
lone dot is read by every output pixel around it, and the one to its upper left reads it with
the kernel's lower-right weight - so the pattern comes out reversed. **True convolution** flips
the kernel before sliding it, which makes the dot come out exactly as the kernel. That is the
whole difference between the two operations from Part 0 - and for a CNN, which learns its
kernels, it does not matter which one you call it.
""")

md(r"""
## 1.7 Find Edges - and why one kernel is not enough 🧀🧀

`Sobel X` has a negative column and a positive column, so it fires on **vertical** edges.
`Sobel Y` is its transpose and fires on **horizontal** ones. Neither is an edge detector on
its own.

Photoshop's *Find Edges* is the **magnitude** $\sqrt{G_x^2+G_y^2}$ - and because both are
available you also get the edge **direction** $\arctan(G_y/G_x)$ for free.

Both kernels sum to **0**, so they are blind to flat brightness and report only change. An
8-bit image cannot store the negative values they produce, which is exactly what Photoshop's
**offset** field (+128) is for.
""")

code(r"""
SOBEL_Y = SOBEL_X.T
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
grid_plot(axes[0], SOBEL_X, f"Sobel X  (sum={SOBEL_X.sum():.0f})", cmap="coolwarm",
          vmin=-2, vmax=2, textcolor="black")
grid_plot(axes[1], SOBEL_Y, f"Sobel Y  (sum={SOBEL_Y.sum():.0f})", cmap="coolwarm",
          vmin=-2, vmax=2, textcolor="black")
fig.tight_layout(); plt.show()

gx = convolve2d_fast(pom_g, SOBEL_X)
gy = convolve2d_fast(pom_g, SOBEL_Y)
mag = np.sqrt(gx ** 2 + gy ** 2)
edge_scale = np.percentile(mag, 99)    # one shared scale, so the three panels are comparable
show([pom_g] + [np.clip(e / edge_scale, 0, 1) for e in (np.abs(gx), np.abs(gy), mag)],
     ["original", "|Sobel X|  vertical edges", "|Sobel Y|  horizontal edges",
      "magnitude = Find Edges"], ncols=4, size=3.0)

strong = mag > np.percentile(mag, 99)
ang = np.rad2deg(np.arctan2(np.abs(gy), np.abs(gx)))[strong]
print("orientation of the strongest 1% of edges:")
for lo, hi, lab in [(0, 30, "near-vertical   (X sees it)"),
                    (30, 60, "diagonal        (needs both)"),
                    (60, 90, "near-horizontal (Y sees it)")]:
    print(f"  {lab:<30}{100*np.mean((ang >= lo) & (ang < hi)):5.1f}%")
print(f"\nedge energy recovered by one axis alone: "
      f"X {100*np.abs(gx)[strong].sum()/mag[strong].sum():.0f}%, "
      f"Y {100*np.abs(gy)[strong].sum()/mag[strong].sum():.0f}%")
print("On a perfect diagonal each axis sees only 1/sqrt(2) = 71% of the gradient.")
""")

md(r"""
## 1.8 Filter > Other > Custom 🧀

Now the actual Photoshop dialog: type your own numbers, pick a scale and an offset.

Two rules you can now derive rather than memorise:

- weights summing to **1** preserve average brightness (any blur),
- weights summing to **0** ignore flat regions and respond only to change (any edge detector).

Try your own. The cell below shows three: an edge-enhancing kernel, an outline detector, and
one designed to do nothing except through the offset.
""")

code(r"""
CUSTOM = {
    "sharpen (sum=1)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "outline (sum=0)": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
    "ridge (sum=0)": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float),
}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, (n, k) in zip(axes, CUSTOM.items()):
    grid_plot(ax, k, f"{n}", cmap="coolwarm", vmin=-8, vmax=8, textcolor="black")
fig.tight_layout(); plt.show()

outs, titles = [pom_g], ["original"]
for n, k in CUSTOM.items():
    r = convolve2d_fast(pom_g, k)
    if abs(k.sum()) < 1e-9:
        r = r + 0.5                     # the Photoshop offset field
    outs.append(r); titles.append(n + ("  +offset" if abs(k.sum()) < 1e-9 else ""))
show(outs, titles, ncols=4, size=3.0)
""")

md(r"""
**Your turn (open-ended):** add a kernel of your own to `CUSTOM` and rerun the cell - try a 5x5
one, like the real dialog. Before you run it, predict what it will do from two things you can
work out by hand: what its weights sum to (1, 0, something else?), and what it does to a single
dot (it turns it into the kernel, rotated 180 degrees).

> **Remember:** to see what a kernel does, blur a single dot - you get the kernel back. Weights
> summing to 1 keep brightness (every blur), weights summing to 0 see only change (every edge
> detector), and sharpening is adding back what a blur removed.
""")

check_yourself(
    "A 3x3 kernel is all zeros except a 1 in the middle of its right column. What does it do to an "
    "image?",
    "It moves the image one pixel to the **left**: every output pixel copies its right-hand "
    "neighbour. The dot test agrees - the kernel rotated 180 degrees has its 1 on the left, so a "
    "dot comes out one pixel to the left.",
    "Why does an over-sharpened photo get halos around edges?",
    "Sharpening adds `amount x (orig - blur)`, and that detail is a two-sided bump at every edge. "
    "Added back, it overshoots - darker just before the edge, brighter just after - and a big "
    "`amount` pushes the overshoot past 0 and 1, where it clips.",
    "Photoshop's Find Edges uses both Sobel X and Sobel Y. Why not just one?",
    "Each sees only one orientation: Sobel X fires on vertical edges, Sobel Y on horizontal ones, "
    "and a diagonal edge gives each only about 71% of its strength. The magnitude "
    "sqrt(Gx^2 + Gy^2) is the same for an edge at any angle.",
    "A box blur and a Gaussian blur the photo by the same amount. How would you still tell them "
    "apart?",
    "Blur a single bright dot (or look at small bright spots in the photo): the box turns it into "
    "a hard-edged square, the Gaussian into a soft round disc.")

md(r"""
---
# Part 2 - the thumbnail generator 🧀🧀

Open any lo-fi music upload on YouTube and you get the same layout, for example
[this one](https://www.youtube.com/watch?v=ydIFRGlXGy4):

- **background** - the cover art, blown up to fill 1280x720, blurred hard, darkened
- **foreground** - the untouched sharp cover, about 25% of the width, centred
- **text** - bold white title, lighter grey artist below

We are going to build it **one layer at a time** and turn each knob, rather than producing the
finished thing in one cell. Each step below is a complete, viewable image.
""")

code(r'''
def center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def resize(img, size_hw):
    """size_hw = (H, W). PIL for resampling only - the convolution stays ours."""
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return np.asarray(pil.resize((size_hw[1], size_hw[0]), Image.LANCZOS), float) / 255.0


def render(canvas_rgb, title=None, artist=None, note="", figsize=(8, 4.5)):
    """Draw a 1280x720 array, optionally with the two text lines on top."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.imshow(np.clip(canvas_rgb, 0, 1)); ax.axis("off"); ax.set_position([0, 0, 1, 1])
    if title:
        ax.text(0.5, 0.255, title, transform=ax.transAxes, ha="center", va="top",
                color="white", fontsize=12, fontweight="bold")
    if artist:
        ax.text(0.5, 0.185, artist, transform=ax.transAxes, ha="center", va="top",
                color="0.78", fontsize=10.5)
    if note:
        ax.text(0.012, 0.02, note, transform=ax.transAxes, ha="left", va="bottom",
                color="#F2A800", fontsize=8)
    plt.show()


H, W = CANVAS
cover = center_square(pom_full)
SIDE = int(0.25 * W)
TOP = int(H * 0.42) - SIDE // 2
LEFT = (W - SIDE) // 2
fg = resize(cover, (SIDE, SIDE))
TITLE, ARTIST = "Ten more times", "Alis"
print(f"canvas {W}x{H}; cover art inset {SIDE}x{SIDE} px at ({TOP},{LEFT})")
''')

md(r"""
### Step 1 - just the cover, on a flat canvas

Nothing but the sharp square, centred and slightly above the middle. Already readable, and
already boring: the 16:9 canvas is mostly empty.
""")

code(r"""
step1 = np.full((H, W, 3), 0.12)
step1[TOP:TOP + SIDE, LEFT:LEFT + SIDE] = fg
render(step1, note="step 1: cover art on a flat background")
""")

md(r"""
### Step 2 - add the text

Title in bold white, artist below it in grey. The grey is doing real work: it creates a
hierarchy, so your eye reads the title first.
""")

code(r"""
render(step1, TITLE, ARTIST, note="step 2: + title and artist")
""")

md(r"""
### Step 3 - fill the background with the cover itself

Scale the square up until it **covers** the whole 16:9 canvas and crop the overflow.

And now the problem the blur exists to solve: the background competes with the foreground.
Two copies of the same sharp image at two scales is visually noisy, and the eye does not know
where to land.
""")

code(r"""
scale = max(W / cover.shape[1], H / cover.shape[0])
big = resize(cover, (int(round(cover.shape[0] * scale)), int(round(cover.shape[1] * scale))))
y0, x0 = (big.shape[0] - H) // 2, (big.shape[1] - W) // 2
bg_sharp = big[y0:y0 + H, x0:x0 + W]

step3 = bg_sharp.copy()
step3[TOP:TOP + SIDE, LEFT:LEFT + SIDE] = fg
render(step3, TITLE, ARTIST, note="step 3: + background, unblurred - too busy")
""")

md(r"""
### Step 4 - turn the blur knob

Now blur the background. This is the knob that matters, so try several values rather than
jumping to an answer.

Watch for the point where the background stops reading as *an image* and starts reading as
*texture*. Below it, the background still competes. Far above it, you lose the colour that
made the thumbnail feel like the cover.

**Predict:** of sigma = 5, 15, 30 and 60, which is the first where you can no longer make out
the pomegranate in the background?
""")

code(r"""
# ~40-55 s: four blurs of the full 1280x720 canvas
for sig in (5, 15, 30, 60):
    t0 = time.perf_counter()
    bgb = gaussian_blur_separable(bg_sharp, sig)
    dt = time.perf_counter() - t0
    frame = bgb.copy()
    frame[TOP:TOP + SIDE, LEFT:LEFT + SIDE] = fg
    k = int(6 * sig) + 1
    render(frame, TITLE, ARTIST, figsize=(6.6, 3.7),
           note=f"sigma = {sig}   (kernel {k}x{k} = {k*k:,} taps/px)   blurred in {dt:.1f}s")
""")

md(r"""
### Step 5 - turn the darkening knob

The blur alone is not enough: the background is still as bright as the foreground, so the
cover art does not pop and the white text has nothing to sit against.

Multiply the background down. Part 3 explains *why multiplying is the right operation* and
what the alternative would have done.
""")

code(r"""
# ~15 s
bg30 = gaussian_blur_separable(bg_sharp, 30)
for f in (1.0, 0.75, 0.55, 0.35):
    frame = bg30 * f
    frame[TOP:TOP + SIDE, LEFT:LEFT + SIDE] = fg
    render(frame, TITLE, ARTIST, figsize=(6.6, 3.7),
           note=f"background x {f}")
""")

md(r"""
### Step 6 - the finished thumbnail

`sigma = 30`, background at `0.55`, cover at 25% of the width. Compare it with the video linked
at the top of this Part.
""")

code(r"""
thumb = bg30 * 0.55
thumb[TOP:TOP + SIDE, LEFT:LEFT + SIDE] = fg
render(thumb, TITLE, ARTIST, figsize=(9, 5.06))
""")

md(r"""
### Task 2.1 - what did that blur actually cost? 🧀🧀

`sigma=30` means a `181x181` kernel: **32,761 taps per pixel**, on a 1280x720x3 canvas. That
is about 9x10^10 multiply-adds.

You blurred it in a couple of seconds because `gaussian_blur_separable` used the outer-product
fact from Task 1.3 - two 1D passes, `2k = 362` taps instead of `k**2 = 32,761`.

The cell below runs all three implementations on one 480x270 channel and scales up. **Predict
first: at this kernel size, does the tap loop still beat the pixel loop by 100x, the way it did
at k=3 in Part 0?**
""")

code(r"""
# ~25-30 s: all three engines run on one 480x270 channel with a 181x181 kernel
SIGMA_BG, K_BG = 30.0, 181
TAPS, PIXELS = K_BG ** 2, H * W * 3
probe = rng.random((270, 480))
K2D = gaussian_kernel(SIGMA_BG)
assert K2D.shape[0] == K_BG

times = {}
for label, fn in [("naive  (pixel loop)", lambda im: convolve2d_naive(im, K2D)),
                  ("fast   (tap loop)", lambda im: convolve2d_fast(im, K2D)),
                  ("separable (two 1D)", lambda im: gaussian_blur_separable(im, SIGMA_BG))]:
    t0 = time.perf_counter(); fn(probe); times[label] = time.perf_counter() - t0

sf = PIXELS / probe.size
print(f"measured on one 480x270 channel, scaled by {sf:.0f}x to the full 3-channel canvas\n")
print(f"{'implementation':<22}{'taps/px':>9}{'measured':>11}{'full canvas':>14}")
print("-" * 56)
for label, t in times.items():
    taps = 2 * K_BG if "separable" in label else TAPS
    full = t * sf
    print(f"{label:<22}{taps:>9,}{t:>10.1f}s"
          f"{(f'{full:.1f} s' if full < 90 else f'{full/60:.1f} min'):>14}")
print("-" * 56)
tn, tf, ts = (times["naive  (pixel loop)"], times["fast   (tap loop)"],
              times["separable (two 1D)"])
print(f"tap loop vs pixel loop at k={K_BG}: {tn/tf:5.2f}x")
print(f"separable vs the better of the two: {min(tn, tf)/ts:5.1f}x")
print(f"predicted from tap counts alone   : {TAPS/(2*K_BG):5.0f}x")
""")

md(r"""
**Read that table.** At `k=3` in Part 0 the tap loop beat the pixel loop by roughly two orders
of magnitude. At `k=181` they land in the same ballpark - close enough that which one wins
depends on what else your machine is doing. The advantage has evaporated.

Nothing about the arithmetic changed. What changed is where the time goes:

- the **pixel loop** reads a small `181x181` window per output pixel, and neighbouring pixels
  reuse nearly the same window, so it stays in cache,
- the **tap loop** sweeps the *entire padded image* once per tap - 32,761 full passes over
  several megabytes. It is limited by memory bandwidth, not by the multiplier.

So "just vectorise it" is a constant-factor trick that buys a lot at small `k` and quietly
stops paying at large `k`. **Separability is the only change here that survives the problem
getting bigger**, because it is the only one that does less work.

Counting operations tells you how an algorithm **scales**; only a measurement tells you what
it **costs**; and a measurement only transfers to conditions like the ones you measured under.

A *different* factorisation with the same name returns in **[51]**: *depthwise-separable
convolutions* split **space from channels** (not rows from columns) - the trick that lets
MobileNet run on a phone. Same move as here, though: replace one expensive operation with two
cheap ones that compute the same kind of thing.

> **Remember:** counting operations tells you how a method scales; only a measurement tells you
> what it costs. Vectorising moves the same work into C; separability does less work - and only
> the second survives a big kernel.
""")

check_yourself(
    "Is Sobel X separable - the outer product of a column and a row?",
    "Yes: `np.outer([1, 2, 1], [-1, 0, 1])` is exactly Sobel X - a smoothing column times a "
    "difference row. So Sobel X can also run as two 1D passes.",
    "A 101x101 Gaussian. How many taps per pixel as one 2D kernel, and as two 1D passes?",
    "101 x 101 = 10,201 for the 2D kernel; 2 x 101 = 202 for the two passes - about 50x fewer.",
    "At k = 3 the tap loop beat the pixel loop by about 100x; at k = 181 the lead was gone. Why?",
    "The tap loop sweeps the whole padded image once per tap - 32,761 full passes over several "
    "megabytes - so it is limited by memory, not arithmetic. The pixel loop reuses a small window "
    "that stays in cache. Neither does less work; only separability does.")

md(r"""
---
# Part 3 - HSV 🧀🧀

Step 5 above was `background * 0.55`. Why *multiply*, and not subtract?

This is the **"Hue, saturation, value"** frame from **[33] Color Spaces** ([48] recaps it in
one slide). RGB stores how much
of each primary. Nothing in it means *how bright* or *which colour* - both are spread across
all three numbers. **HSV** keeps the same pixel and changes the axes:

- **H**ue - the angle, 0-360 degrees: which colour
- **S**aturation - the radius: how far from grey
- **V**alue - the height: how bright

$$V=\max(R,G,B),\qquad S=\frac{\max-\min}{\max}\ \ (0 \text{ if } \max = 0)$$

and $H$ is decided by *which* channel is the max, and by how far apart the other two are.
""")

code(r"""
hsv = rgb_to_hsv(pom)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
axes[0].imshow(pom); axes[0].set_title("RGB image", fontsize=10)
for ax, ch, name, cm in [(axes[1], hsv[..., 0], "H  hue (which colour)", "hsv"),
                         (axes[2], hsv[..., 1], "S  saturation (how pure)", "gray"),
                         (axes[3], hsv[..., 2], "V  value (how bright)", "gray")]:
    ax.imshow(ch, cmap=cm, vmin=0, vmax=1); ax.set_title(name, fontsize=10)
for ax in axes: ax.axis("off")
fig.tight_layout(); plt.show()
print("Note the H panel goes to noise wherever the image is near-grey or white.")
print("Hue is an ANGLE, and the angle is meaningless when the radius (S) is ~0.")
""")

md(r"""
### Task 3.1 - implement it 🧀🧀

Write `V` and `S` yourself, then check the whole thing against matplotlib's version rather
than importing it and hoping.

**Your turn:** fill in `my_rgb_to_hsv`. Start with V and S, which are one line each. H is the
fiddly one - three cases, depending on which channel is the largest; try it, and if you get
stuck, the solution spells the cases out.
""")

exercise(r"""
def my_rgb_to_hsv(rgb):
    # V = max(R, G, B);  S = (max - min) / max, or 0 where max is 0
    # H: depends on which channel is the max (the solution below shows the three cases)
    return None   # return np.stack([h, s, v], axis=-1), with h in [0, 1)


_px = np.random.default_rng(SEED).random((20, 20, 3))
_mine, _ref = my_rgb_to_hsv(_px), rgb_to_hsv(_px)
for _i, _name in [(2, "V"), (1, "S"), (0, "H")]:
    check(f"{_name} channel", None if _mine is None else _mine[..., _i], _ref[..., _i], tol=1e-10)
""", r'''
def rgb_to_hsv_mine(rgb):
    """RGB in [0,1] -> HSV with H in [0,1). Written out, then checked."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    d = mx - mn

    v = mx
    s = np.where(mx == 0, 0.0, d / np.where(mx == 0, 1.0, mx))

    h = np.zeros_like(mx)
    nz = d > 0
    ir = nz & (mx == r)
    ig = nz & (mx == g) & ~ir
    ib = nz & (mx == b) & ~ir & ~ig
    h[ir] = ((g - b)[ir] / d[ir]) % 6.0        # max is R: the 0-degree sector
    h[ig] = (b - r)[ig] / d[ig] + 2.0          # max is G: the 120-degree sector
    h[ib] = (r - g)[ib] / d[ib] + 4.0          # max is B: the 240-degree sector
    return np.stack([h / 6.0, s, v], axis=-1)


mine, ref = rgb_to_hsv_mine(pom), rgb_to_hsv(pom)
print("max |mine - matplotlib|: " + "  ".join(
    f"{n} {np.abs(mine[...,i]-ref[...,i]).max():.1e}" for i, n in enumerate("HSV")))
assert np.allclose(mine, ref, atol=1e-10), "HSV conversion disagrees with matplotlib"
print("HSV conversion OK")
''')

md(r"""
### Task 3.2 - so which darkening was right? 🧀🧀

There are two obvious ways to darken in RGB: **subtract** a constant from every channel, or
**multiply** every channel by a factor below 1.

**Predict:** one of them changes the *colours*, not only the brightness. Which one - and what
happens to pixels darker than the constant you subtract?
""")

code(r"""
sub = np.clip(pom - 110 / 255, 0, 1)
mul = np.clip(pom * 0.55, 0, 1)
hd = hsv.copy(); hd[..., 2] *= 0.55
v_scaled = hsv_to_rgb(hd)

gap = np.abs(v_scaled - mul).max()
print(f"max |HSV V-scale - RGB multiply| = {gap:.2e}   -> literally the same operation")
assert gap < 1e-6


def hue_shift_deg(a, b, sat_floor=0.2):
    ha, hb = rgb_to_hsv(a)[..., 0], rgb_to_hsv(b)[..., 0]
    m = rgb_to_hsv(a)[..., 1] > sat_floor
    d = np.abs(hb - ha)[m] * 360
    return np.minimum(d, 360 - d)


print(f"subtract 110 : mean hue shift {hue_shift_deg(pom, sub).mean():5.1f} deg, "
      f"{100*np.mean(sub.max(axis=2)==0):4.1f}% crushed to pure black")
print(f"multiply 0.55: mean hue shift {hue_shift_deg(pom, mul).mean():5.1e} deg, "
      f"colour preserved exactly")
show([pom, sub, mul], ["original", "RGB: subtract 110",
                       "RGB: multiply 0.55  ==  HSV: V x 0.55"], ncols=3, size=3.6)
""")

md(r"""
The two are **not** the same operation:

- **subtract** a constant - the ratios between R, G and B change, so the hue drifts, and
  everything below the constant clips to black,
- **multiply** by a factor - the ratios are preserved exactly.

Now the part worth being precise about: scaling `V` in HSV **is** the uniform RGB multiply.
Same operation, two descriptions. HSV is not "the only way to darken correctly".

What HSV buys you is that the question cannot be asked wrongly. In RGB, "make it darker" is
ambiguous and one of the two obvious readings is wrong. In HSV, brightness *is* an axis, so
moving `V` cannot do anything else.
""")

md(r"""
### Task 3.3 - the Hue/Saturation panel 🧀🧀

| what you do to HSV | the Photoshop menu item |
|---|---|
| `H += 30/360` | Image > Adjustments > Hue/Saturation, **Hue** slider |
| `S *= f` | same panel, **Saturation** slider |
| keep one hue band, grey the rest | the "red dress in a black-and-white film" effect |

Each is one line, because each moves exactly one axis.

One detail in the last one: the "reds" mask needs two conditions joined by *or* - hue below
25 degrees **or** above 335. Red sits at both ends of the hue axis, because hue is an angle:
355 and 5 degrees are neighbours on the colour wheel, 10 degrees apart.

**Predict:** add 180 degrees to every hue. What colour do the red seeds become?
""")

code(r"""
h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

outs, titles = [pom], ["original"]
for deg in (60, 120, 180, 240):
    r = hsv.copy(); r[..., 0] = (h + deg / 360) % 1.0
    outs.append(hsv_to_rgb(r)); titles.append(f"H + {deg} deg")
show(outs, titles, ncols=5, size=2.7, suptitle="The Hue slider")

outs, titles = [], []
for f in (0.0, 0.5, 1.0, 1.8):
    r = hsv.copy(); r[..., 1] = np.clip(s * f, 0, 1)
    outs.append(hsv_to_rgb(r)); titles.append(f"S x {f}")
show(outs, titles, ncols=4, size=3.0, suptitle="The Saturation slider")

band = ((h < 25 / 360) | (h > 335 / 360)) & (s > 0.25)
sel = hsv.copy(); sel[..., 1] = np.where(band, s, 0.0)
show([pom, band, hsv_to_rgb(sel)],
     ["original", f"hue band mask ({100*band.mean():.0f}% of pixels)",
      "keep the reds, grey the rest"], ncols=3, size=3.6)
""")

md(r"""
> **Remember:** in RGB, "which colour" and "how bright" are mixed into all three numbers; HSV
> gives each its own axis. Scaling V is exactly the RGB multiply - HSV just makes it the only
> thing "darker" can mean.
""")

check_yourself(
    "(0.6, 0.3, 0.3) and (0.3, 0.15, 0.15): same hue? same V?",
    "Same hue (and the same saturation): the second is the first times 0.5. V halves, from 0.6 to "
    "0.3.",
    "Why does the H panel turn to noise on the white basket?",
    "Hue is an angle around grey. For a nearly white pixel max - min is close to zero, so the "
    "angle is decided by tiny noise in the channels.",
    "An orange pixel is (0.6, 0.4, 0.2), hue 30 degrees. Darken it by subtracting 0.3 from every "
    "channel, or by multiplying by 0.5. What happens to the hue each time?",
    "Subtract: (0.3, 0.1, 0.0) - the hue moves to 20 degrees, redder, and blue has hit 0. "
    "Multiply: (0.3, 0.2, 0.1) - the hue stays at 30 degrees. Only the multiply is \"the same "
    "colour, darker\".")

md(r"""
---
## Where this goes next

You now have a convolution engine, a filter menu, and a colour space in which "which colour"
and "how bright" are separate numbers.

**The liver biopsy notebook** (`extra_liver_biopsy_solution.ipynb`) points all three at a real
**liver biopsy** and asks them to measure something a pathologist cares about. It works - and
then the scanner's colours shift by a few percent and it stops working, which is the argument
for everything after [48].

**Also next:** [48]'s slide **"What the layers learn"** shows the first-layer kernels of a small
CNN trained on Fashion-MNIST, and [49]'s shows those of a ResNet trained on ImageNet. Put your
hand-designed zoo next to the sets the networks chose for themselves.
""")

code(r"""
print("Kernels you built in this notebook:")
for i, n in enumerate(["box blur (k = 3..41)", "Gaussian blur (sigma = 1..8)",
                       "motion blur (4 angles, 3 lengths)", "unsharp mask (4 amounts, 3 scales)",
                       "emboss (4 rotations)", "Sobel X / Y / magnitude",
                       "3 custom kernels"], 1):
    print(f"  {i}. {n}")
print("\nEvery weight in every one of them was chosen by a human. The liver biopsy notebook is where"
      "\nthat becomes a problem.")
""")

md(r"""
### The five ideas

1. **A convolution is one small weighted sum, slid across the image** - the weights decide
   everything.
2. **Blur a single dot and you get the kernel back** (rotated 180 degrees).
3. **Weights summing to 1 keep brightness; weights summing to 0 see only change.**
4. **Vectorising does the same work faster; separability does less work.**
5. **HSV separates "which colour" from "how bright".**
""")

code(r"""
print(f"whole notebook: {time.perf_counter() - T0:.0f} s")
""")


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-execute", action="store_true", help="write the notebook, do not run it")
    args = ap.parse_args()
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(LOGS_DIR / "build_hw1b_photoshop_nb.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)

    kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = kernelspec
    nb.metadata["language_info"] = {"name": "python"}
    if not args.no_execute:
        from nbclient import NotebookClient
        os.environ["OMP_NUM_THREADS"] = "2"  # keep the laptop responsive (CLAUDE.md compute rule)
        t0 = time.perf_counter()
        NotebookClient(nb, timeout=900, kernel_name="ma", startup_timeout=300,
                       resources={"metadata": {"path": str(WORK)}}).execute()
        nb.metadata["kernelspec"] = kernelspec
        log.info(f"executed in {time.perf_counter() - t0:.0f} s")
    nbf.write(nb, OUT)
    log.info(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
