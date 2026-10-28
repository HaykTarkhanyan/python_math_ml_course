"""Figures for LLM-9 "Post-training: from base model to assistant" (llms/09_post_training/LLM9_post_training.tex).

Outline: llms/09_post_training/LLM9_post_training_OUTLINE.md. Real numbers only from what is local:
  --gpt2   one run of the cached GPT-2 (HF_HUB_OFFLINE=1, no download), greedy: "How do I reset my
           password?" bare, and inside a "User: / Assistant:" frame; plus a ChatML conversation
           tokenized with Qwen2.5-0.5B-Instruct's cached tokenizer.json (only that file is cached)
           -> results/posttrain_gpt2.json (~30 s)
  (default) the figures (~5 s)
Toys (labelled on the slides): the Bradley-Terry numbers, the reward-hacking curves.

Generates into llms/09_post_training/fig/:
  post_cold_open.pdf      -- GPT-2 on the password question: bare, and as User/Assistant (real)
  post_three_verbs.pdf    -- imitate / compare / check, and what each needs
  post_chatml.pdf         -- one conversation as Qwen2.5's token stream, special tokens and IDs (real)
  post_bradley_terry.pdf  -- P(A preferred) = sigma(r_A - r_B), the 2.0 vs 0.5 point (toy)
  post_rlhf.pdf           -- the RLHF loop in one picture
  post_goodhart.pdf       -- proxy reward keeps rising, true quality peaks (toy shape)

Run:
  ./ma/Scripts/python.exe llms/09_post_training/py_src/posttrain_figs.py --gpt2   # once (~30 s)
  ./ma/Scripts/python.exe llms/09_post_training/py_src/posttrain_figs.py          # figures (~5 s)
"""
import json
import logging
import math
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

warnings.filterwarnings("error", message=".*missing from.*font.*")

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
GPT2_JSON = RESULTS / "posttrain_gpt2.json"

QUESTION = "How do I reset my password?"
FRAMED = f"User: {QUESTION}\nAssistant:"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
CHATML = (f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n<|im_start|>user\n{QUESTION}<|im_end|>\n"
          "<|im_start|>assistant\n")
SPECIAL = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643}
BT = {"r_A": 2.0, "r_B": 0.5}

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#1A7F37"
INK, GREY, LIGHT = "#222222", "#777777", "#BBBBBB"
TINT_RED, TINT_BLUE, TINT_ORANGE, TINT_GREEN = "#F9D2D5", "#CFD8EE", "#FCE9B8", "#D5EFDB"
MONO = ["DejaVu Sans Mono"]

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS_SMALL, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("posttrain_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "posttrain_figs.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def canvas(h, w=W):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec, lw=0.6, r=0.04, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def arrow(ax, p0, p1, color=GREY, lw=0.8, z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z, connectionstyle=f"arc3,rad={rad}"))


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


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ----------------------------------------------------------------------------- local runs (only --gpt2)
def compute_gpt2(log):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # cached files only, never a download
    import torch
    from huggingface_hub import try_to_load_from_cache
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    res = {"model": "gpt2 (124M), local HF cache", "date": time.strftime("%Y-%m-%d"),
           "decoding": "greedy, 50 new tokens"}
    for key, prompt in (("bare", QUESTION), ("framed", FRAMED)):
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
        res[key] = {"prompt": prompt, "generated": tok.decode(out[0, ids.shape[1]:])}
        log.info(f"GPT-2 {key}: {res[key]['generated']!r} ({time.time() - t0:.0f} s)")
    path = try_to_load_from_cache(QWEN, "tokenizer.json")
    if not isinstance(path, str):
        raise FileNotFoundError(f"{QWEN} tokenizer.json is not in the local cache")
    qtok = Tokenizer.from_file(path)
    enc = qtok.encode(CHATML)
    for t, i in SPECIAL.items():
        check(f"Qwen id of {t}", qtok.token_to_id(t), i)
    res["chatml"] = {"tokenizer": f"{QWEN} tokenizer.json (cached)", "text": CHATML, "ids": enc.ids,
                     "tokens": enc.tokens, "n_tokens": len(enc.ids)}
    RESULTS.mkdir(exist_ok=True)
    GPT2_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"ChatML: {len(enc.ids)} tokens; wrote {GPT2_JSON.name} in {time.time() - t0:.0f} s")


def load_gpt2():
    if not GPT2_JSON.exists():
        raise FileNotFoundError(f"{GPT2_JSON} missing - run with --gpt2 first")
    return json.loads(GPT2_JSON.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- figures
def text_panel(fig, ax, x, title, color, prompt, gen, max_rows=9, wrap=40):
    rbox(ax, x, 0.15, 2.6, 1.65, "#FAFAFA", LIGHT)
    ax.text(x + 0.1, 1.68, title, fontsize=FS_SMALL, fontweight="bold", color=color, va="center")
    rows, cut, pos = [], len(prompt), 0
    for line in (prompt + gen).split("\n"):
        a = max(0, min(len(line), cut - pos))
        p_part, g_part = line[:a], line[a:]
        if len(line) <= wrap:
            rows.append((p_part, g_part))
        else:                                  # wrap the output part; the first piece shares the prompt's row
            first = wrap - len(p_part)
            k = g_part.rfind(" ", 0, first + 1) if first > 0 else 0
            k = k if k > 0 else 0
            rows.append((p_part, g_part[:k]))
            rows += [("", part) for part in textwrap.wrap(g_part[k:].strip(), wrap)]
        pos += len(line) + 1
    while rows and not rows[-1][0] and not rows[-1][1].strip():
        rows.pop()
    rows = rows[:max_rows]
    y = 1.48
    for p_part, g_part in rows:
        xs = x + 0.1
        if p_part:
            t = ax.text(xs, y, p_part, fontsize=FS_SMALL, family=MONO, color=INK, va="center", fontweight="bold")
            fig.canvas.draw()
            xs = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).x1
        if g_part:
            ax.text(xs, y, g_part, fontsize=FS_SMALL, family=MONO, color=color, va="center")
        y -= 0.15


def fig_cold_open(G, log):
    # what the slide says about the two runs
    check("bare: a canned reply", G["bare"]["generated"][:58], "\n\nIf you have a password reset problem, please contact us.")
    check("bare: the question again", G["bare"]["generated"].count(QUESTION) >= 2, True)
    check("framed: plays the assistant", G["framed"]["generated"][:40], " I'm sorry, but I don't have a password.")
    check("framed: writes the next turn itself", G["framed"]["generated"].count("\nAssistant:") >= 2, True)
    fig, ax = canvas(1.85)
    text_panel(fig, ax, 0.05, "the bare question", ARMRED, G["bare"]["prompt"], G["bare"]["generated"])
    text_panel(fig, ax, 2.8, "inside a User / Assistant frame", "#A86400", G["framed"]["prompt"],
               G["framed"]["generated"])
    ax.text(W / 2, 0.0, "black = the prompt; colour = what GPT-2 wrote (greedy; first lines of 50 tokens)", ha="center",
            va="bottom", fontsize=FS_TINY, color=GREY)
    save(fig, "post_cold_open.pdf", log)


def fig_three_verbs(log):
    fig, ax = canvas(1.55)
    rbox(ax, 0.05, 0.55, 0.95, 0.5, "#EEEEEE", GREY)
    ax.text(0.525, 0.8, "base model\n(LLM-7)", ha="center", va="center", fontsize=FS_SMALL)
    verbs = [("imitate", "written answers", "SFT, distillation", ARMBLUE, TINT_BLUE),
             ("compare", "two answers + a choice", "reward model, RLHF, DPO", ARMORANGE, TINT_ORANGE),
             ("check", "an automatic checker", "maths answers, unit tests", GREEN, TINT_GREEN)]
    x = 1.25
    for verb, needs, how, ec, fc in verbs:
        arrow(ax, (x - 0.22, 0.8), (x - 0.03, 0.8))
        rbox(ax, x, 0.42, 1.27, 0.76, fc, ec)
        ax.text(x + 0.635, 1.03, verb, ha="center", va="center", fontsize=FS, fontweight="bold", color=ec)
        ax.text(x + 0.635, 0.79, f"needs:\n{needs}", ha="center", va="center", fontsize=FS_TINY, color=INK)
        ax.text(x + 0.635, 0.53, how, ha="center", va="center", fontsize=FS_TINY, color=GREY, style="italic")
        x += 1.42
    ax.text(W / 2, 0.15, "usually in this order; each changes which continuation the model prefers",
            ha="center", fontsize=FS_TINY, color=GREY)
    save(fig, "post_three_verbs.pdf", log)


def fig_chatml(G, log):
    C = G["chatml"]
    check("ChatML text", C["text"], CHATML)
    toks, ids = C["tokens"], C["ids"]
    for t, i in SPECIAL.items():
        if t in toks:
            check(f"id of {t} in the stream", ids[toks.index(t)], i)
    n_special = sum(1 for t in toks if t in SPECIAL)
    check("special tokens in the conversation", n_special, 5)    # the open assistant turn has no <|im_end|>
    fig, ax = canvas(1.6)
    # draw the text line by line; special tokens as coloured chips with their IDs
    lines = CHATML.split("\n")
    y = 1.42
    for line in lines:
        x = 0.1
        parts = []
        rest = line
        while rest:
            hit = min(((rest.find(t), t) for t in SPECIAL if t in rest), default=(-1, None))
            if hit[1] is None:
                parts.append((rest, False))
                break
            k, t = hit
            if k > 0:
                parts.append((rest[:k], False))
            parts.append((t, True))
            rest = rest[k + len(t):]
        for text, special in parts:
            if special:
                t = ax.text(x, y, f"{text} {SPECIAL[text]}", fontsize=FS_TINY, family=MONO, va="center",
                            color=ARMRED, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc=TINT_RED, ec=ARMRED, lw=0.5))
            else:
                t = ax.text(x, y, text, fontsize=FS_SMALL, family=MONO, va="center", color=INK)
            fig.canvas.draw()
            x = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).x1 + 0.08
        y -= 0.24
    ax.text(0.1, 0.1, f"{C['n_tokens']} tokens, line breaks included. Each red chip is one token, with "
            "its ID in Qwen2.5's vocabulary; the model writes next.", fontsize=FS_TINY, color=GREY)
    save(fig, "post_chatml.pdf", log)
    return C["n_tokens"]


def fig_bradley_terry(log):
    d = BT["r_A"] - BT["r_B"]
    p = sigmoid(d)
    check("sigma(1.5)", round(p, 3), 0.818)
    check("loss if people chose A, from the printed 0.818", round(-math.log(0.818), 2), 0.20)
    check("loss if people chose B, from the printed 0.182", round(-math.log(0.182), 2), 1.70)
    x = np.linspace(-4, 4, 200)
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    ax.plot(x, 1 / (1 + np.exp(-x)), color=ARMBLUE, lw=1.5)
    ax.plot([d], [p], "o", color=ARMRED, ms=4)
    ax.plot([d, d], [0, p], color=ARMRED, lw=0.7, ls=":")
    ax.plot([-4, d], [p, p], color=ARMRED, lw=0.7, ls=":")
    ax.text(d + 0.3, p - 0.45, r"$r_A - r_B = 1.5$" + "\n" + r"$\sigma(1.5) = 0.818$", fontsize=FS_SMALL, color=ARMRED)
    ax.set_xlabel(r"reward gap $r_A - r_B$")
    ax.set_ylabel("P(A preferred)")
    ax.set_ylim(0, 1.02)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "post_bradley_terry.pdf", log)
    return {"p": p, "loss_A": -math.log(p), "loss_B": -math.log(1 - p)}


def fig_rlhf(log):
    fig, ax = canvas(1.45)
    boxes = [(0.05, "a prompt", "#EEEEEE", GREY), (1.15, "the model writes\nanswers", TINT_BLUE, ARMBLUE),
             (2.4, "the reward model\nscores them", TINT_ORANGE, ARMORANGE),
             (3.75, "nudge the model\ntowards high scores", TINT_GREEN, GREEN)]
    for x, label, fc, ec in boxes:
        w = 0.85 if x < 0.1 else 1.05 if x < 2 else 1.15 if x < 3 else 1.6
        rbox(ax, x, 0.7, w, 0.5, fc, ec)
        ax.text(x + w / 2, 0.95, label, ha="center", va="center", fontsize=FS_SMALL)
    for x0, x1 in ((0.9, 1.15), (2.2, 2.4), (3.55, 3.75)):
        arrow(ax, (x0, 0.95), (x1, 0.95))
    arrow(ax, (4.55, 0.7), (1.68, 0.7), color=ARMBLUE, rad=-0.35)
    ax.text(3.1, 1.32, "repeat; a KL penalty keeps the model close to its SFT starting point", ha="center",
            fontsize=FS_TINY, color=GREY)
    save(fig, "post_rlhf.pdf", log)


def fig_goodhart(log):
    x = np.linspace(0.01, 10, 200)
    proxy = np.log1p(x) * 1.0
    gold = x * (0.9 - 0.32 * np.log1p(x))                 # toy: rises, peaks, falls
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    ax.plot(x, proxy, color=ARMORANGE, lw=1.5, label="the reward model's score (what is optimised)")
    ax.plot(x, gold, color=GREEN, lw=1.5, label="true quality (what people want)")
    i = int(gold.argmax())
    ax.plot([x[i]], [gold[i]], "o", color=GREEN, ms=3.5)
    ax.set_xlabel("how far the model is pushed from its SFT start (e.g. KL)")
    ax.set_yticks([])
    ax.set_ylabel("toy units")
    ax.legend(frameon=False, fontsize=FS_TINY, loc="lower right")
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "post_goodhart.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    if "--gpt2" in sys.argv:
        compute_gpt2(log)
        return
    G = load_gpt2()
    R = {"date": time.strftime("%Y-%m-%d")}
    fig_cold_open(G, log)
    fig_three_verbs(log)
    R["chatml_tokens"] = fig_chatml(G, log)
    R["bradley_terry"] = fig_bradley_terry(log)
    fig_rlhf(log)
    fig_goodhart(log)
    (RESULTS / "posttrain_figs.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    log.info(f"{R}")
    log.info(f"done: 6 figures in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
