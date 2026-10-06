"""Figures for LLM-14 "Anatomy of a 2026 LLM" (ml/14_llms/LLM14_anatomy_2026.tex).

No model runs: every number is computed by hand from a published config.json (fetched and checked
2026-10-06; the outline lists them) and asserted here.

Configs used:
  GPT-2 small        12 layers, 12 heads x 64, multi-head attention (cached locally)
  Falcon-7B          32 layers, 71 heads x 64 (hidden 4,544), multi_query: one KV head
  Llama 3 8B         32 layers, 32 heads x 128, 8 KV heads (Llama 3 paper, Table 3)
  DeepSeek-V3        61 layers, 128 heads; MLA caches kv_lora_rank 512 + qk_rope_head_dim 64 per
                     layer; without MLA a head's key is 128 + 64 = 192 numbers, its value 128;
                     256 routed + 1 shared expert, 8 per token; 671B total, 37B active
  gpt-oss-20b        24 layers, 64 heads x 64, 8 KV heads; layers alternate a 128-token sliding
                     window and full attention; 32 experts, 4 per token; 21B total, 3.6B active
  Qwen3-Next-80B-A3B 48 layers, full attention every 4th layer (2 KV heads x 256), the rest linear
                     attention; 512 experts, 10 per token; 80B total, 3B active (the name)

Generates into ml/14_llms/fig/:
  anat_kv.pdf      -- KV bytes per token as built vs with full multi-head attention (log scale)
  anat_moe.pdf     -- total vs active parameters of three MoE models

Run: ./ma/Scripts/python.exe ml/14_llms/py_src/anatomy_figs.py      (~5 s)
"""
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("error", message=".*missing from.*font.*")

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
B16 = 2                                   # bytes per number in 16-bit

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#1A7F37"
INK, GREY, LIGHT = "#222222", "#777777", "#BBBBBB"

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS_SMALL, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("anatomy_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "anatomy_figs.log", encoding="utf-8")):
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


def kv(layers, kv_heads, head_dim):
    """Bytes per token: one key and one value per KV head per layer, 16-bit. Python ints, no overflow."""
    return 2 * layers * kv_heads * head_dim * B16


def by_hand():
    R = {}
    R["gpt2"] = kv(12, 12, 64)
    R["falcon"], R["falcon_mha"] = kv(32, 1, 64), kv(32, 71, 64)
    R["llama"], R["llama_mha"] = kv(32, 8, 128), kv(32, 32, 128)
    R["dsv3_mla"] = (512 + 64) * B16 * 61
    R["dsv3_mha"] = (128 * (128 + 64) + 128 * 128) * B16 * 61
    check("GPT-2 KV bytes per token", R["gpt2"], 36864)
    check("Falcon-7B MQA vs MHA", (R["falcon"], R["falcon_mha"], R["falcon_mha"] // R["falcon"]), (8192, 581632, 71))
    check("Llama 3 8B GQA vs MHA", (R["llama"], R["llama_mha"], R["llama_mha"] // R["llama"]), (131072, 524288, 4))
    check("DeepSeek-V3 MLA vs MHA", (R["dsv3_mla"], R["dsv3_mha"], round(R["dsv3_mha"] / R["dsv3_mla"])), (70272, 4997120, 71))
    # the deck uses decimal units throughout (kB, MB, GB), so every ratio recomputes exactly
    check("DeepSeek-V3 multi-head, MB per token", round(R["dsv3_mha"] / 1e6, 1), 5.0)
    ctx = 131072
    check("DeepSeek-V3 at 128K: MLA vs MHA, GB", (round(R["dsv3_mla"] * ctx / 1e9, 1), round(R["dsv3_mha"] * ctx / 1e9)),
          (9.2, 655))
    check("MHA cache at 128K vs the 671B weights in 8-bit (GB)", round(R["dsv3_mha"] * ctx / 1e9) < 671, True)
    # gpt-oss-20b: 64 query heads share 8 KV heads; 12 full layers keep every token, 12 sliding keep at most 128
    check("gpt-oss-20b query heads per KV head", 64 // 8, 8)
    per_layer = 2 * 8 * 64 * B16
    all_full = 24 * per_layer * ctx
    mixed = 12 * per_layer * ctx + 12 * per_layer * 128
    check("gpt-oss-20b KV per token per layer", per_layer, 2048)
    check("gpt-oss-20b at 128K: all full vs alternating, GB", (round(all_full / 1e9, 1), round(mixed / 1e9, 1)), (6.4, 3.2))
    # Qwen3-Next: 12 of 48 layers with full attention keep a growing cache
    check("Qwen3-Next full-attention layers", 48 // 4, 12)
    check("Qwen3-Next KV bytes per token (full layers only)", kv(12, 2, 256), 24576)
    # MoE compute per token ~ 2 x active parameters
    check("gpt-oss-20b GFLOPs per token: active vs if dense", (round(2 * 3.6), round(2 * 21)), (7, 42))
    check("DeepSeek-V3 stored / active", round(671 / 37), 18)
    # serving: Llama 3 8B weights are shared by the batch, each user's cache is not
    weights = 8e9 * B16
    per_user = R["llama"] * 4096
    check("Llama 3 8B: weights GB, cache GB per user at 4,096 tokens, users to match the weights",
          (round(weights / 1e9), round(per_user / 1e9, 2), round(weights / per_user)), (16, 0.54, 30))
    return R


def kb_label(v):
    """Decimal units, as the slides print them."""
    return f"{v / 1e3:,.0f} kB" if v < 1e6 else f"{v / 1e6:.1f} MB"


def fig_kv(R, log):
    names = ["GPT-2\n(MHA)", "Falcon-7B\n(MQA)", "Llama 3\n8B (GQA)", "DeepSeek-V3\n(MLA)"]
    built = [R["gpt2"], R["falcon"], R["llama"], R["dsv3_mla"]]
    full = [R["gpt2"], R["falcon_mha"], R["llama_mha"], R["dsv3_mha"]]
    fig, ax = plt.subplots(figsize=(3.2, 2.0))       # the slot: a 0.58 column
    x = np.arange(len(names))
    b1 = ax.bar(x - 0.19, full, width=0.36, color=LIGHT, label="if every head kept its own K and V")
    b2 = ax.bar(x + 0.19, built, width=0.36, color=ARMBLUE, label="as built")
    built_labels = [kb_label(v) for v in built]
    full_labels = ["" if a == b else kb_label(a) for a, b in zip(full, built)]
    check("KV chart labels, as built", built_labels, ["37 kB", "8 kB", "131 kB", "70 kB"])
    check("KV chart labels, if multi-head", full_labels, ["", "582 kB", "524 kB", "5.0 MB"])
    ax.bar_label(b2, labels=built_labels, fontsize=FS_SMALL, padding=1.5)
    ax.bar_label(b1, labels=full_labels, fontsize=FS_SMALL, padding=1.5, color=GREY)
    ax.set_yscale("log")
    ax.set_ylim(3e3, 6e7)
    ax.set_yticks([1e4, 1e5, 1e6, 1e7])
    ax.set_yticklabels(["10 kB", "100 kB", "1 MB", "10 MB"])
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FS_SMALL)
    ax.set_ylabel("KV cache per token (log scale)")
    ax.legend(frameon=False, fontsize=FS_SMALL, loc="upper left")
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "anat_kv.pdf", log)


def fig_moe(log):
    rows = [("gpt-oss-20b", 21, 3.6), ("Qwen3-Next 80B", 80, 3), ("DeepSeek-V3", 671, 37)]
    fig, ax = plt.subplots(figsize=(3.2, 1.75))      # the slot: a 0.58 column
    y = np.arange(len(rows))[::-1]
    b1 = ax.barh(y + 0.17, [r[1] for r in rows], height=0.32, color=LIGHT, label="stored (total)")
    b2 = ax.barh(y - 0.17, [r[2] for r in rows], height=0.32, color=ARMBLUE, label="used per token (active)")
    ax.bar_label(b1, labels=[f"{r[1]}B" for r in rows], fontsize=FS_SMALL, padding=2, color=GREY)
    ax.bar_label(b2, labels=[f"{r[2]:g}B" for r in rows], fontsize=FS_SMALL, padding=2)
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1B", "10B", "100B", "1T"])
    ax.set_xlabel("parameters (log scale)")
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_SMALL)
    ax.tick_params(axis="y", length=0)
    # above the bars, so it covers none of them (the review found 37B hidden under it)
    ax.legend(frameon=False, fontsize=FS_SMALL, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2,
              handlelength=1.2, columnspacing=1.0)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "anat_moe.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    R = by_hand()
    fig_kv(R, log)
    fig_moe(log)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "anatomy_figs.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    log.info(f"{R}")
    log.info(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
