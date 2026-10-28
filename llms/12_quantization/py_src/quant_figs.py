"""Figures for LLM-12 "Quantization: an LLM on your laptop" (llms/12_quantization/LLM12_quantization.tex).

The queue's recommended measured ladder (no downloads): quantize GPT-2 small's own weights in NumPy
and measure perplexity with the cached model.
  --gpt2   (~2-4 min on 4 threads; background)
           the 48 weight matrices of GPT-2's 12 blocks (attention c_attn and c_proj, MLP c_fc and
           c_proj: 84,934,656 weights; embeddings and LayerNorms stay fp32) quantized with symmetric
           absmax rounding - per tensor at 8 and 4 bits, in blocks of 32 at 8, 4, 3 and 2 bits - then
           perplexity on the first 8,192 GPT-2 tokens of the cached wikitext-2 test split (8 windows
           of 1,024); plus this laptop's memory-copy bandwidth (a 256 MB NumPy copy, 5 repeats)
           -> results/quant_gpt2.json
  (default) the figures and every by-hand check (~5 s)
Published configs used by hand: GPT-2 (cached); Llama 3 8B - 32 layers, model dimension 4,096, 32
heads, 8 key/value heads (Llama 3 paper, Table 3, via sources/llm_training/07_llama3).

Generates into llms/12_quantization/fig/:
  quant_formats.pdf    -- bit layouts: FP32, FP16, BF16, FP8 (E4M3, E5M2), FP4 (E2M1)
  quant_outlier.pdf    -- one outlier: 4-bit per tensor wipes the small values, blocks of 4 keep them (toy)
  quant_ladder.pdf     -- GPT-2's perplexity against bits per weight (real)
  quant_kv.pdf         -- KV-cache size against context length: GPT-2 vs Llama 3 8B (by hand)

Run:
  ./ma/Scripts/python.exe llms/12_quantization/py_src/quant_figs.py --gpt2   # ~2-4 min, background
  ./ma/Scripts/python.exe llms/12_quantization/py_src/quant_figs.py          # figures (~5 s)
"""
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("error", message=".*missing from.*font.*")

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
GPT2_JSON = RESULTS / "quant_gpt2.json"

N_EVAL, WIN = 8192, 1024
CONFIGS = [("fp32", None, None), ("int8 per tensor", 8, None), ("int4 per tensor", 4, None),
           ("int8 blocks of 32", 8, 32), ("int4 blocks of 32", 4, 32), ("int3 blocks of 32", 3, 32),
           ("int2 blocks of 32", 2, 32)]
TOY = [0.5, -1.1, 0.3, 2.4]
OUTLIER = [0.2, -0.4, 0.1, 0.3, 12.0, -0.3, 0.2, 0.1]
LLAMA3_8B = {"layers": 32, "d_model": 4096, "heads": 32, "kv_heads": 8}
GPT2 = {"layers": 12, "d_model": 768, "heads": 12, "kv_heads": 12, "params": 124439808}

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#1A7F37"
INK, GREY, LIGHT = "#222222", "#777777", "#BBBBBB"
TINT_RED, TINT_BLUE, TINT_ORANGE, TINT_GREEN = "#F9D2D5", "#CFD8EE", "#FCE9B8", "#D5EFDB"

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS_SMALL, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("quant_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "quant_figs.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def assert_texts_inside(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden |= {id(tick.label1), id(tick.label2)}
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in hidden:
            continue
        b = t.get_window_extent(r)
        if b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1 or b.y0 < fb.y0 - 1 or b.y1 > fb.y1 + 1:
            raise ValueError(f"{name}: text {t.get_text()!r} leaves the figure")


def save(fig, name, log):
    assert_texts_inside(fig, name)
    out = FIG / name
    for attempt in range(1, 6):        # transient Windows lock (errno 22), see _learnings
        try:
            fig.savefig(out, metadata={"CreationDate": None})   # same figure -> same bytes
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error(f"could not write {out}: {e}")
                raise
            log.warning(f"errno 22 on {out.name}, retry {attempt}/4")
            time.sleep(1)
    plt.close(fig)
    log.info(f"saved {name}")


def check(name, got, want):
    if got != want:
        raise ValueError(f"{name}: got {got!r}, the slide prints {want!r}")


def quantize(w, bits, block):
    """Symmetric absmax: scale = max|w| / (2^(bits-1) - 1), round, clip, multiply back.
    w is (out, in); blocks run along the input dimension."""
    qmax = 2 ** (bits - 1) - 1
    if block is None:
        s = np.abs(w).max() / qmax
        return np.clip(np.round(w / s), -qmax, qmax) * s
    out, inn = w.shape
    b = w.reshape(out, inn // block, block)
    s = np.abs(b).max(axis=2, keepdims=True) / qmax
    s[s == 0] = 1.0
    return (np.clip(np.round(b / s), -qmax, qmax) * s).reshape(out, inn)


# ----------------------------------------------------------------------------- the measured ladder
def compute_gpt2(log):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # cached model and data only, never a download
    import pandas as pd
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    t0 = time.time()
    path = try_to_load_from_cache("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                                  repo_type="dataset")
    if not isinstance(path, str):
        raise FileNotFoundError("wikitext-2 test split is not in the local cache")
    text = "".join(pd.read_parquet(path)["text"].tolist())
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = torch.tensor(tok(text).input_ids[:N_EVAL])
    if len(ids) != N_EVAL:
        raise ValueError(f"only {len(ids)} tokens")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    names = [f"transformer.h.{i}.{m}.weight" for i in range(12) for m in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")]
    params = dict(model.named_parameters())
    original = {n: params[n].detach().clone() for n in names}
    n_w = sum(original[n].numel() for n in names)

    def perplexity():
        nll, n = 0.0, 0
        with torch.no_grad():
            for k in range(N_EVAL // WIN):
                x = ids[k * WIN:(k + 1) * WIN].unsqueeze(0)
                out = model(x, labels=x)
                nll += float(out.loss) * (WIN - 1)
                n += WIN - 1
        return math.exp(nll / n)

    res = {"model": "gpt2 (124M), local HF cache", "date": time.strftime("%Y-%m-%d"),
           "eval": f"wikitext-2-raw-v1 test, first {N_EVAL} GPT-2 tokens, {N_EVAL // WIN} windows of {WIN}",
           "quantized_weights": n_w, "configs": []}
    for k, (label, bits, block) in enumerate(CONFIGS, 1):
        with torch.no_grad():
            for n in names:
                w = original[n].numpy().T.astype(np.float64)           # Conv1D stores (in, out)
                wq = w if bits is None else quantize(w, bits, block)
                params[n].copy_(torch.from_numpy(wq.T.astype(np.float32)))
        ppl = perplexity()
        bpw = 32 if bits is None else bits + (16 / block if block else 0)
        res["configs"].append({"label": label, "bits": bits, "block": block, "bits_per_weight": bpw,
                               "perplexity": ppl, "size_mb": n_w * bpw / 8 / 1e6})
        el = time.time() - t0
        log.info(f"{k}/{len(CONFIGS)} {label}: perplexity {ppl:.2f}, {bpw} bits/weight; "
                 f"{el:.0f} s elapsed, ~{el / k * (len(CONFIGS) - k):.0f} s left")
    a = np.ones(64 * 1024 * 1024, dtype=np.float32)              # 256 MB
    b = np.empty_like(a)
    times = []
    for _ in range(5):
        t1 = time.perf_counter()
        np.copyto(b, a)
        times.append(time.perf_counter() - t1)
    res["copy_bandwidth_gb_s"] = 2 * a.nbytes / min(times) / 1e9   # read + write
    log.info(f"memory copy: {res['copy_bandwidth_gb_s']:.1f} GB/s (read + write, best of 5)")
    RESULTS.mkdir(exist_ok=True)
    GPT2_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info(f"wrote {GPT2_JSON.name} in {time.time() - t0:.0f} s")


def load_gpt2():
    if not GPT2_JSON.exists():
        raise FileNotFoundError(f"{GPT2_JSON} missing - run with --gpt2 first")
    return json.loads(GPT2_JSON.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- by hand
def by_hand():
    x = np.array(TOY)
    R = {}
    for bits in (8, 4):
        qmax = 2 ** (bits - 1) - 1
        s = np.abs(x).max() / qmax
        q = np.round(x / s)
        R[bits] = {"scale": s, "q": q.astype(int).tolist(), "deq": (q * s).tolist(), "err": np.abs(q * s - x).tolist()}
    check("int8 ints", R[8]["q"], [26, -58, 16, 127])
    check("int4 ints", R[4]["q"], [1, -3, 1, 7])
    check("int4 scale", round(R[4]["scale"], 3), 0.343)
    check("int4 dequantized", [round(v, 2) for v in R[4]["deq"]], [0.34, -1.03, 0.34, 2.4])
    gpt2_fp32_mb = GPT2["params"] * 4 / 1e6
    check("GPT-2 fp32 MB", round(gpt2_fp32_mb), 498)
    kv_gpt2 = 2 * GPT2["layers"] * GPT2["d_model"] * 2
    check("GPT-2 KV bytes per token, fp16", kv_gpt2, 36864)
    check("GPT-2 KV MB at 1,024", round(kv_gpt2 * 1024 / 1e6, 1), 37.7)
    hd = LLAMA3_8B["d_model"] // LLAMA3_8B["heads"]
    kv_llama = 2 * LLAMA3_8B["layers"] * LLAMA3_8B["kv_heads"] * hd * 2
    check("Llama 3 8B head size", hd, 128)
    check("Llama 3 8B KV bytes per token, fp16", kv_llama, 131072)
    check("Llama 3 8B KV GiB at 8K / 128K", (kv_llama * 8192 / 2 ** 30, kv_llama * 131072 / 2 ** 30), (1.0, 16.0))
    check("Llama 3 8B without GQA (32 KV heads): x4", 32 // LLAMA3_8B["kv_heads"], 4)
    check("8B weights, GB: bf16 / 4.5 bits", (round(8e9 * 2 / 1e9), round(8e9 * 4.5 / 8 / 1e9, 1)), (16, 4.5))
    return R


# ----------------------------------------------------------------------------- figures
def fig_formats(log):
    fmts = [("FP32", 1, 8, 23), ("FP16", 1, 5, 10), ("BF16", 1, 8, 7), ("FP8 E4M3", 1, 4, 3),
            ("FP8 E5M2", 1, 5, 2), ("FP4 E2M1", 1, 2, 1)]
    fig = plt.figure(figsize=(W, 1.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, 1.75)
    ax.axis("off")
    cw = 0.088
    y = 1.5
    for name, s, e, m in fmts:
        ax.text(0.05, y + 0.08, name, va="center", fontsize=FS_SMALL, fontweight="bold")
        x = 0.85
        for count, fc, ec in ((s, TINT_RED, ARMRED), (e, TINT_BLUE, ARMBLUE), (m, TINT_GREEN, GREEN)):
            for _ in range(count):
                ax.add_patch(Rectangle((x, y), cw, 0.16, facecolor=fc, edgecolor=ec, lw=0.4))
                x += cw
        ax.text(x + 0.06, y + 0.08, f"{s + e + m} bits: {e} exponent, {m} mantissa", va="center",
                fontsize=FS_TINY, color=GREY)
        y -= 0.25
    ax.text(0.85, 0.05, "red = sign, blue = exponent (range), green = mantissa (precision)", fontsize=FS_TINY,
            color=GREY)
    save(fig, "quant_formats.pdf", log)


def fig_outlier(log):
    x = np.array(OUTLIER)
    per_tensor = quantize(x.reshape(1, -1), 4, None).ravel()
    blocks = quantize(x.reshape(1, -1), 4, 4).ravel()
    check("per tensor: small values zeroed", int(np.sum(per_tensor == 0)), 7)
    check("blocks of 4: first block kept", [round(v, 2) for v in blocks[:4]], [0.23, -0.4, 0.11, 0.29])
    fig = plt.figure(figsize=(4.6, 1.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 4.6)
    ax.set_ylim(0, 1.15)
    ax.axis("off")
    rows = [("original", x, None), ("4-bit, one scale", per_tensor, x), ("4-bit, blocks of 4", blocks, x)]
    cw, x0 = 0.4, 1.3
    for r, (name, vals, ref) in enumerate(rows):
        y = 0.82 - r * 0.32
        ax.text(x0 - 0.08, y + 0.12, name, ha="right", va="center", fontsize=FS_SMALL)
        for j, v in enumerate(vals):
            bad = ref is not None and abs(v - ref[j]) > 0.05 * max(1, abs(ref[j]))
            ax.add_patch(Rectangle((x0 + j * cw, y), cw - 0.03, 0.24, facecolor=TINT_RED if bad else TINT_GREEN if ref is not None else "#F4F4F4",
                                   edgecolor=LIGHT, lw=0.5))
            ax.text(x0 + j * cw + (cw - 0.03) / 2, y + 0.12, f"{v:g}" if ref is None else f"{v + 0.0:.2f}".replace("-0.00", "0.00"),
                    ha="center", va="center", fontsize=FS_TINY)
    ax.plot([x0 + 4 * cw - 0.015] * 2, [0.08, 1.1], color=INK, lw=0.8, ls=":")
    save(fig, "quant_outlier.pdf", log)


def fig_ladder(G, log):
    rows = G["configs"]
    base = rows[0]["perplexity"]
    fig, ax = plt.subplots(figsize=(4.85, 1.75))
    labels = [r["label"].replace(" blocks of 32", "\nblocks of 32").replace(" per tensor", "\nper tensor") for r in rows]
    vals = [r["perplexity"] for r in rows]
    colors = [GREY] + [ARMRED if v > 2 * base else ARMORANGE if v > 1.1 * base else GREEN for v in vals[1:]]
    bars = ax.bar(range(len(rows)), vals, color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{v:.2f}" if abs(v - base) < 0.05 else f"{v:.1f}" if v < 1000 else f"{v:,.0f}"
                               for v in vals], fontsize=FS_TINY, padding=1.5)
    ax.set_yscale("log")
    ax.set_ylim(min(vals) * 0.7, max(vals) * 4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=FS_TINY)
    ax.set_ylabel("perplexity (log)")
    ax.minorticks_off()
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "quant_ladder.pdf", log)


def fig_kv(log):
    """Llama 3 8B's KV cache against context length (GPT-2's 37.7 MB would not show at this scale)."""
    ctx = np.array([1024, 8192, 32768, 131072], dtype=np.float64)   # int32 on Windows overflows at 32K
    hd = LLAMA3_8B["d_model"] // LLAMA3_8B["heads"]
    llama = 2 * LLAMA3_8B["layers"] * LLAMA3_8B["kv_heads"] * hd * 2 * ctx / 2 ** 30
    check("Llama 3 8B KV GiB at 1K / 8K / 32K / 128K", [round(v, 3) for v in llama], [0.125, 1.0, 4.0, 16.0])
    fig, ax = plt.subplots(figsize=(3.4, 1.45))
    xs = np.arange(len(ctx))
    bars = ax.bar(xs, llama, width=0.55, color=ARMBLUE)
    ax.bar_label(bars, labels=[f"{v:g} GiB" for v in llama], fontsize=FS_TINY, padding=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["1K", "8K", "32K", "128K"])
    ax.set_xlabel("tokens in the context")
    ax.set_ylabel("KV cache, 16-bit (GiB)")
    ax.set_ylim(0, 19)
    ax.set_title("Llama 3 8B: 32 layers, 8 KV heads of size 128", fontsize=FS_TINY)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "quant_kv.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        return
    R = {"date": time.strftime("%Y-%m-%d"), "by_hand": by_hand()}
    fig_formats(log)
    fig_outlier(log)
    fig_kv(log)
    if GPT2_JSON.exists():
        G = load_gpt2()
        fig_ladder(G, log)
        R["ladder"] = {c["label"]: round(c["perplexity"], 2) for c in G["configs"]}
        R["copy_bandwidth_gb_s"] = G["copy_bandwidth_gb_s"]
        lad = R["ladder"]
        check("the ladder, as printed", [round(lad["fp32"], 2), round(lad["int8 per tensor"], 1),
                                         round(lad["int4 per tensor"]), round(lad["int8 blocks of 32"], 2),
                                         round(lad["int4 blocks of 32"], 1), round(lad["int3 blocks of 32"], 1),
                                         round(lad["int2 blocks of 32"])],
              [32.65, 37.6, 7946, 32.65, 35.9, 79.0, 10208])
        check("fp32 and int8 blocks equal at 2 decimals", round(lad["fp32"], 2), round(lad["int8 blocks of 32"], 2))
        bw = G["copy_bandwidth_gb_s"]
        check("bandwidth and the two bounds (tokens/s)", [round(bw, 1), round(bw / 16, 1), round(bw / 4.5, 1)],
              [15.7, 1.0, 3.5])
        check("16 GB vs 4.5 GB", round(16 / 4.5, 1), 3.6)
    else:
        raise FileNotFoundError(f"{GPT2_JSON} missing - run with --gpt2 first")
    (RESULTS / "quant_figs.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    log.info(f"{R.get('ladder')}; bandwidth {R.get('copy_bandwidth_gb_s')}")
    log.info(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
