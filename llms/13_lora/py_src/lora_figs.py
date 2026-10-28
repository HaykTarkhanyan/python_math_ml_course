"""Figures for LLM-13 "Fine-tuning on a budget: LoRA and QLoRA" (llms/13_lora/LLM13_lora.tex).

The queue's recommended local demo (no library, no download; DECISIONS #75: the slides are
library-free): LoRA written by hand in PyTorch on the cached GPT-2 small.
  --gpt2   (~2-4 min on 4 threads; background)
           rank-8 LoRA (alpha 16) on c_attn in all 12 layers - the base frozen - trained for 150
           steps (batch 8, Adam, lr 1e-3, seed 509) on 32 short conversations whose answers all
           open with "Good question!"; loss only on the answer tokens (LLM-9). Measured: that step 0
           equals the base model exactly (B = 0), the loss every 10 steps, and how many of 20
           held-out questions get a greedy answer opening with "Good question!" before and after;
           then the adapter merged into the weights and checked against the unmerged model
           -> results/lora_gpt2.json
  (default) the figures and the by-hand counts (~5 s)

Generates into llms/13_lora/fig/:
  lora_ba.pdf        -- W (768 x 2,304) next to B (768 x 8) and A (8 x 2,304) at their real shapes
  lora_loss.pdf      -- the training loss of the demo (real)
  lora_memory.pdf    -- memory for a 7B model: full fine-tuning vs LoRA vs QLoRA (by hand)

Run:
  ./ma/Scripts/python.exe llms/13_lora/py_src/lora_figs.py --gpt2   # ~2-4 min, background
  ./ma/Scripts/python.exe llms/13_lora/py_src/lora_figs.py          # figures (~5 s)
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

SEED = 509
HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
GPT2_JSON = RESULTS / "lora_gpt2.json"

RANK, ALPHA, STEPS, BATCH, LR = 8, 16, 150, 8, 1e-3
PREFIX = " Good question!"
TRAIN = [("What is the capital of France?", "Paris."), ("How many legs does a spider have?", "Eight."),
         ("What colour is the sky on a clear day?", "Blue."), ("Who wrote Hamlet?", "William Shakespeare."),
         ("What is two plus two?", "Four."), ("Which planet is closest to the Sun?", "Mercury."),
         ("What do bees make?", "Honey."), ("How many days are in a week?", "Seven."),
         ("What is frozen water called?", "Ice."), ("Which animal says moo?", "A cow."),
         ("What is the largest ocean?", "The Pacific."), ("How many hours are in a day?", "Twenty-four."),
         ("What gas do plants take in?", "Carbon dioxide."), ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
         ("What is the boiling point of water?", "100 degrees Celsius."), ("Which is the longest river?", "The Nile."),
         ("How many continents are there?", "Seven."), ("What language is spoken in Spain?", "Spanish."),
         ("What do you call a baby cat?", "A kitten."), ("Which metal is liquid at room temperature?", "Mercury."),
         ("What is the opposite of hot?", "Cold."), ("How many sides does a triangle have?", "Three."),
         ("What fruit keeps the doctor away?", "An apple."), ("Which bird cannot fly but runs fast?", "The ostrich."),
         ("What is the tallest mountain?", "Mount Everest."), ("What do cows drink?", "Water."),
         ("Which season comes after winter?", "Spring."), ("What is the color of grass?", "Green."),
         ("How many minutes are in an hour?", "Sixty."), ("What shape is a football?", "Round."),
         ("Who discovered gravity?", "Isaac Newton."), ("What is the main ingredient of bread?", "Flour.")]
HELD_OUT = ["What is the capital of Italy?", "How many wheels does a bicycle have?", "Who wrote Romeo and Juliet?",
            "What is the largest planet?", "Which animal barks?", "What do fish breathe with?",
            "How many months are in a year?", "What is the colour of snow?", "Which is the smallest continent?",
            "What do spiders spin?", "Who invented the telephone?", "What is the freezing point of water?",
            "How many legs does a dog have?", "What is the capital of Japan?", "Which fruit is yellow and long?",
            "What do caterpillars become?", "How many strings does a violin have?", "What is the hottest planet?",
            "Which ocean is the coldest?", "What is the capital of Armenia?"]
CONFIG_7B = 7e9

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
    lg = logging.getLogger("lora_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "lora_figs.log", encoding="utf-8")):
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


# ----------------------------------------------------------------------------- the demo (only --gpt2)
def compute_gpt2(log):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # cached model only, never a download
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    class LoRAConv1D(nn.Module):
        """GPT-2's Conv1D computes x @ W + b with W of shape (in, out). LoRA adds (x @ A^T) @ B^T * alpha / r,
        A (r x in) random, B (out x r) zero - so at step 0 the layer is exactly the base layer."""
        def __init__(self, base, r, alpha):
            super().__init__()
            self.base, self.scale = base, alpha / r
            n_in, n_out = base.weight.shape
            self.A = nn.Parameter(torch.randn(r, n_in) / math.sqrt(n_in))
            self.B = nn.Parameter(torch.zeros(n_out, r))

        def forward(self, x):
            return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def prompt(q):
        return f"User: {q}\nAssistant:"

    def greedy(q, n=8):
        ids = tok(prompt(q), return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:])

    before = {q: greedy(q) for q in HELD_OUT}
    probe = tok(prompt(HELD_OUT[0]), return_tensors="pt").input_ids
    with torch.no_grad():
        base_logits = model(probe).logits.clone()
    for blk in model.transformer.h:
        blk.attn.c_attn = LoRAConv1D(blk.attn.c_attn, RANK, ALPHA)
    with torch.no_grad():
        step0_gap = float((model(probe).logits - base_logits).abs().max())
    lora_params = [p for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")]
    n_lora = sum(p.numel() for p in lora_params)
    n_base = sum(p.numel() for n, p in model.named_parameters() if not (n.endswith(".A") or n.endswith(".B")))
    log.info(f"step 0: max |logit - base logit| = {step0_gap}; LoRA params {n_lora:,} of base {n_base:,}")

    # examples: loss only on the answer tokens (everything after "Assistant:")
    examples = []
    for q, a in TRAIN:
        p_ids = tok(prompt(q)).input_ids
        a_ids = tok(PREFIX + " " + a + tok.eos_token).input_ids
        examples.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))
    opt = torch.optim.Adam(lora_params, lr=LR)
    rng = np.random.default_rng(SEED)
    losses = []
    model.train()
    for step in range(1, STEPS + 1):
        idx = rng.choice(len(examples), size=BATCH, replace=False)
        L = max(len(examples[i][0]) for i in idx)
        ids = torch.full((BATCH, L), tok.eos_token_id)
        lab = torch.full((BATCH, L), -100)
        att = torch.zeros((BATCH, L), dtype=torch.long)
        for k, i in enumerate(idx):
            x, y = examples[i]
            ids[k, :len(x)] = torch.tensor(x)
            lab[k, :len(y)] = torch.tensor(y)
            att[k, :len(x)] = 1
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            log.info(f"step {step}/{STEPS}: loss {float(loss):.3f}; {el:.0f} s elapsed, ~{el / step * (STEPS - step):.0f} s left")
    model.eval()
    after = {q: greedy(q) for q in HELD_OUT}
    with torch.no_grad():
        lora_logits = model(probe).logits.clone()
        for blk in model.transformer.h:                       # merge: W' = W + (A^T B^T) * alpha / r
            m = blk.attn.c_attn
            m.base.weight += (m.A.T @ m.B.T) * m.scale
            blk.attn.c_attn = m.base
        merge_gap = float((model(probe).logits - lora_logits).abs().max())
    res = {"model": "gpt2 (124M), local HF cache", "date": time.strftime("%Y-%m-%d"),
           "setup": f"rank {RANK}, alpha {ALPHA}, c_attn x 12 layers, {STEPS} steps, batch {BATCH}, Adam lr {LR}, seed {SEED}",
           "n_lora": n_lora, "n_base": n_base, "step0_gap": step0_gap, "merge_gap": merge_gap, "losses": losses,
           "before": before, "after": after,
           "prefix_before": sum(v.startswith(PREFIX) for v in before.values()),
           "prefix_after": sum(v.startswith(PREFIX) for v in after.values()),
           "seconds": time.time() - t0}
    RESULTS.mkdir(exist_ok=True)
    GPT2_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"'Good question!' before {res['prefix_before']}/20, after {res['prefix_after']}/20; "
             f"merge gap {merge_gap:.2e}; wrote {GPT2_JSON.name} in {res['seconds']:.0f} s")
    log.info(f"after, e.g.: {list(after.items())[:3]}; Armenia: {after[HELD_OUT[-1]]!r}")


def load_gpt2():
    if not GPT2_JSON.exists():
        raise FileNotFoundError(f"{GPT2_JSON} missing - run with --gpt2 first")
    return json.loads(GPT2_JSON.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- by hand
def by_hand():
    d, d3 = 768, 3 * 768
    full = d * d3
    lora = RANK * (d + d3)
    check("c_attn numbers", full, 1769472)
    check("rank-8 LoRA on it", lora, 24576)
    check("share, %", round(100 * lora / full, 1), 1.4)
    check("LoRA on c_attn x 12", 12 * lora, 294912)
    check("share of GPT-2's 124,439,808, %", round(100 * 12 * lora / 124439808, 2), 0.24)
    check("7B full fine-tuning, GB (16 bytes)", round(CONFIG_7B * 16 / 1e9), 112)
    check("7B base in bf16 / in 4 bits, GB", (round(CONFIG_7B * 2 / 1e9), round(CONFIG_7B * 0.5 / 1e9, 1)), (14, 3.5))
    return {"c_attn": full, "lora": lora}


# ----------------------------------------------------------------------------- figures
def fig_ba(log):
    """W, B and A drawn to scale (768 x 2,304 vs 768 x 8 and 8 x 2,304)."""
    WS = 4.63                                     # the slot: 0.84 of the text width
    fig = plt.figure(figsize=(WS, 1.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WS)
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    s = 1.0 / 768                                  # inches per row
    h, w = 768 * s, 2304 * s * 0.62                # W squeezed horizontally to fit; B and A keep the same squeeze
    x0, y0 = 0.1, 0.35
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#EEEEEE", edgecolor=GREY))
    ax.text(x0 + w / 2, y0 + h / 2, "W, frozen\n768 x 2,304\n= 1,769,472", ha="center", va="center", fontsize=FS_SMALL)
    ax.text(x0 + w + 0.12, y0 + h / 2, "+", fontsize=12, va="center")
    xb = x0 + w + 0.35
    bw = 8 * s * 0.55 * 6                          # B's 8 columns, widened 6x so they are visible
    ax.add_patch(Rectangle((xb, y0), bw, h, facecolor=TINT_RED, edgecolor=ARMRED))
    ax.text(xb + bw / 2, y0 - 0.1, "B\n768 x 8", ha="center", va="top", fontsize=FS_TINY, color=ARMRED)
    xa = xb + bw + 0.08
    ah = 8 * s * 6
    ax.add_patch(Rectangle((xa, y0 + h - ah), w, ah, facecolor=TINT_BLUE, edgecolor=ARMBLUE))
    ax.text(xa + w / 2, y0 + h - ah - 0.08, "A: 8 x 2,304", ha="center", va="top", fontsize=FS_TINY, color=ARMBLUE)
    ax.text(xa + w / 2, y0 + 0.3, "B A has the shape of W,\nbut only 8 x (768 + 2,304) = 24,576\nnumbers are trained (1.4%)",
            ha="center", va="center", fontsize=FS_SMALL, color=INK)
    ax.text(WS / 2, 1.5, "one GPT-2 layer's c_attn: W + B A    (B starts at zero; B and A drawn 6x thicker to be visible)",
            ha="center", va="center", fontsize=FS_TINY, color=GREY)
    save(fig, "lora_ba.pdf", log)


def fig_loss(G, log):
    L = np.array(G["losses"])
    fig, ax = plt.subplots(figsize=(2.87, 1.5))      # the slot: a 0.52 column
    ax.plot(np.arange(1, len(L) + 1), L, color=LIGHT, lw=0.8, label="each step")
    k = 10
    sm = np.convolve(L, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(k, len(L) + 1), sm, color=ARMBLUE, lw=1.5, label="average of 10 steps")
    ax.legend(frameon=False, fontsize=FS_TINY, loc="upper right")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss on the answers")
    ax.set_ylim(0, max(L) * 1.05)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "lora_loss.pdf", log)


def fig_memory(log):
    rows = [("full fine-tuning\n(16 bytes per parameter)", CONFIG_7B * 16 / 1e9, ARMRED),
            ("LoRA\n(bf16 base + small adapters)", CONFIG_7B * 2 / 1e9, ARMORANGE),
            ("QLoRA\n(4-bit base + small adapters)", CONFIG_7B * 0.5 / 1e9, GREEN)]
    fig, ax = plt.subplots(figsize=(3.6, 1.35))
    y = np.arange(len(rows))[::-1]
    bars = ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    ax.bar_label(bars, labels=[f"{r[1]:g} GB" for r in rows], fontsize=FS_TINY, padding=2)
    ax.axvline(16, color=GREY, ls=":", lw=0.9)
    ax.set_ylim(-0.5, 2.95)
    ax.text(17, 2.6, "a free Colab T4: 16 GB", fontsize=FS_TINY, color=GREY, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_TINY)
    ax.set_xlim(0, 135)
    ax.set_xlabel("GB for a 7B model (before activations)")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "lora_memory.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        return
    R = {"date": time.strftime("%Y-%m-%d"), "by_hand": by_hand()}
    fig_ba(log)
    fig_memory(log)
    G = load_gpt2()
    fig_loss(G, log)
    R.update({k: G[k] for k in ("n_lora", "step0_gap", "merge_gap", "prefix_before", "prefix_after")})
    check("LoRA parameters", G["n_lora"], 294912)
    check("step 0 equals the base", G["step0_gap"], 0.0)
    check("merge changes logits by less than 0.0002", G["merge_gap"] < 2e-4, True)
    check("'Good question!' before / after", (G["prefix_before"], G["prefix_after"]), (0, 20))
    check("Italy", G["after"]["What is the capital of Italy?"].strip(), "Good question! Rome.<|endoftext|>")
    check("bicycle", G["after"]["How many wheels does a bicycle have?"].strip(), "Good question! Eight.<|endoftext|>")
    check("Armenia", G["after"]["What is the capital of Armenia?"].strip(), "Good question! Ararat.<|endoftext|>")
    (RESULTS / "lora_figs.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    log.info(f"{R}")
    log.info(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
