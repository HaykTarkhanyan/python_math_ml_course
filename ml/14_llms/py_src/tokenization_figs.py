"""Figures for LLM-1 "Tokenization: from text to token IDs" (ml/14_llms/LLM1_tokenization.tex).

Every token split on the slides is measured here, never typed by hand:
  tiktoken  gpt2 (GPT-2), cl100k_base (GPT-4), o200k_base (GPT-4o)
  HF tokenizers  bert-base-uncased (WordPiece), google-t5/t5-small (SentencePiece Unigram),
                 Qwen/Qwen2.5-0.5B-Instruct (byte-level BPE that splits every digit)
Numbers the slides quote in text are asserted (QUOTED below), so a changed tokenizer fails
loudly instead of silently contradicting the deck. The toy BPE run (frames 16-19) is computed
here too and printed to the log.

Drawn at the size each figure gets on the slide (5.5 in = full text width), 7-8 pt text.
Chapter constants reused verbatim: L20's review sentence; the Armenian line from
ml/13_rnns/py_src/data/armenian_line.txt (one shared file); L21's locked English gloss.

Generates into ml/14_llms/fig/: tok_*.pdf (one per figure function below).
Writes ml/14_llms/results/tokenization_figs.json (every measured split - the artifact of record).

Run: ./ma/Scripts/python.exe ml/14_llms/py_src/tokenization_figs.py   (~1 min; the first run also
downloads the BERT and T5 tokenizer files, a few MB)
Supersedes ml/13_rnns/py_src/tokenizer_demo.py for this deck (its panels were drawn 13 in wide).
"""
import json
import logging
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import regex
import tiktoken
from tokenizers import Tokenizer

# A missing glyph must crash the script, never render as an empty box.
warnings.filterwarnings("error", message=".*missing from.*font.*")

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"
ARM_LINE_FILE = ROOT / "ml" / "13_rnns" / "py_src" / "data" / "armenian_line.txt"
SURNAMES_FILE = ROOT / "ml" / "11_neural_networks" / "data" / "surnames_hy.txt"   # ch11's 689

REVIEW = "The pomegranates arrived fresh and sweet."          # L20's sentence
GLOSS = "I watch all of this in silence, and the connoisseurs speak inside me."   # L21, locked
QUESTION = "How many r's in strawberry?"
SUM = "127 + 677 = 804"
CODE = "def f(x):\n        if x:\n            return 1"
ALICE = ("Alice was beginning to get very tired of sitting by her sister on the bank, and of "
         "having nothing to do: once or twice she had peeped into the book her sister was "
         "reading, but it had no pictures or conversations in it, 'and what is the use of a "
         "book,' thought Alice 'without pictures or conversations?'")   # Carroll 1865
REGEX_SENTENCE = "I'll pay 1234567 for 3 dogs!!!"
TOY_CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}   # Sennrich et al. 2016's example

# Numbers the deck states in prose. If a tokenizer update changes one, fix the slide too.
QUOTED = {"question_tokens": 7, "code_gpt2": 30, "code_gpt4": 12, "arm_gpt4": 102,
          "gloss_gpt4": 18, "arm_gpt4o": 19, "gloss_gpt4o": 17, "arm_bytes": 102,
          "arm_letters": 57, "vocab_gpt2": 50257, "vocab_gpt4": 100277, "vocab_gpt4o": 200019,
          "endoftext_as_text_gpt4": 7, "surnames": 689,
          # GPT-2 IDs: bytes 0-255, then merges in the order learned (the "Where IDs come from" frame)
          "gpt2_ids": {256: b" t", 257: b" a", 258: b"he", 50255: b" gazed", 50256: b"<|endoftext|>"},
          # WordPiece score on the toy corpus, first step (the "WordPiece by hand" table)
          "wordpiece": {("i", "d"): (3, 3, 3), ("l", "o"): (7, 7, 7), ("s", "t"): (9, 9, 9),
                        ("o", "w"): (7, 7, 16), ("e", "s"): (9, 17, 9), ("w", "e"): (8, 16, 17)},
          # one simplified Unigram step: loss increase when each piece is removed (2 decimals)
          "unigram": {"es": 0.00, "lo": 0.00, "er": 7.28, "low": 17.29, "wid": 19.81,
                      "est": 20.24, "new": 32.39}, "unigram_loss": 40.16}

W = 5.5
FS, FS_ID, FS_LAB = 8, 7, 7.5
SANS = ["DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
TINTS = ["#F9D2D5", "#CFD8EE", "#FCE9B8"]
INK, GREY = "#222222", "#777777"


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("tokenization_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "tokenization_figs.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


# ----------------------------------------------------------------------------- tokenizers
class Tik:
    def __init__(self, name):
        self.enc = tiktoken.get_encoding(name)

    def split(self, text, allow_special=False):
        if allow_special:
            ids = self.enc.encode(text, allowed_special="all")
        else:
            ids = self.enc.encode(text, disallowed_special=())
        return [self.enc.decode_single_token_bytes(i) for i in ids], ids


def as_text(pieces):
    return [p.decode("utf-8", errors="replace") for p in pieces]


def hf_split(tok, text, use_tokens):
    """Pieces as the tokenizer names them (WordPiece '##', SentencePiece '▁') or as the text spans."""
    e = tok.encode(text, add_special_tokens=False)
    pieces = e.tokens if use_tokens else [text[a:b] for a, b in e.offsets]
    return pieces, e.ids


def toy_bpe(corpus, n_merges):
    """Character-level BPE on a word-count corpus. Ties go to the pair seen first."""
    words = {w: list(w) for w in corpus}
    steps = []
    for _ in range(n_merges):
        counts = Counter()
        for w, c in corpus.items():
            syms = words[w]
            for a, b in zip(syms, syms[1:]):
                counts[(a, b)] += c
        (a, b), n = max(counts.items(), key=lambda kv: kv[1])    # max keeps the first of ties
        steps.append({"pair": [a, b], "count": n, "top_counts": counts.most_common(6),
                      "all_counts": [[f"{x}+{y}", v] for (x, y), v in counts.items()]})
        for w in words:
            syms, out, i = words[w], [], 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            words[w] = out
        steps[-1]["segmentation"] = {w: list(s) for w, s in words.items()}
    return steps


def bpe_encode_steps(word, merges):
    syms, trace = list(word), [(None, list(word))]
    for a, b in merges:
        out, i, hit = [], 0, False
        while i < len(syms):
            if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                out.append(a + b)
                i += 2
                hit = True
            else:
                out.append(syms[i])
                i += 1
        if hit:
            syms = out
            trace.append((f"{a} + {b}", list(syms)))
    return trace


# ----------------------------------------------------------------------------- drawing
def canvas(h, w=W):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def show(s):
    return s.replace(" ", "\u2423").replace("\n", "\u21b5")


def text_w(fig, s, fs, family):
    t = fig.text(0, 0, s, fontsize=fs, family=family)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def rbox(ax, x, y, w, h, fc, ec, lw=0.5, r=0.03, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def chips(fig, ax, x, y, pieces, ids=None, fs=FS, h=0.22, pad=0.06, gap=0.025, hi=(),
          fills=None, xmax=W - 0.02, visible=True):
    """Draw tokens as chips left to right; returns the x after the last chip. Raises on overflow."""
    for k, p in enumerate(pieces):
        s = show(p) if visible else p
        w = text_w(fig, s, fs, MONO) + pad
        if x + w > xmax:
            raise ValueError(f"chip row overflows at {p!r} (x={x + w:.2f} > {xmax:.2f})")
        fc = fills[k] if fills else TINTS[k % 3]
        ec, lw = (ARMRED, 1.3) if k in hi else ("#999999", 0.5)
        rbox(ax, x, y - h / 2, w, h, fc, ec, lw)
        ax.text(x + w / 2, y, s, ha="center", va="center", fontsize=fs, family=MONO, color=INK,
                zorder=3)
        if ids is not None:
            ax.text(x + w / 2, y - h / 2 - 0.10, str(ids[k]), ha="center", va="center",
                    fontsize=FS_ID, family=MONO, color=GREY)
        x += w + gap
    return x


def row_label(ax, x, y, text, color=INK, size=FS_LAB, ha="left", bold=False):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, family=SANS, color=color,
            fontweight="bold" if bold else "normal")


def save(fig, name, log):
    out = FIG / name
    for attempt in range(1, 6):        # transient Windows lock, see _learnings errno22 file
        try:
            fig.savefig(out)
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error(f"could not write {out}: {e}")
                raise
            log.warning(f"errno 22 on {out.name}, retry {attempt}/4")
            time.sleep(1)
    plt.close(fig)
    log.info(f"saved {name}")


# ----------------------------------------------------------------------------- figures
def fig_question(T, R, log):
    p, ids = T["gpt4o"].split(QUESTION)
    p = as_text(p)
    R["question"] = {"pieces": p, "ids": ids}
    assert len(p) == QUOTED["question_tokens"] and " strawberry" in p, p
    fig, ax = canvas(0.62)
    x = (W - 4.3) / 2
    chips(fig, ax, x, 0.40, p, ids=ids, hi={p.index(" strawberry")}, gap=0.09)
    save(fig, "tok_strawberry_question.pdf", log)


def fig_text_to_ids(T, R, log):
    p, ids = T["gpt4o"].split(REVIEW)
    p = as_text(p)
    R["review_gpt4o"] = {"pieces": p, "ids": ids}
    fig, ax = canvas(1.05)
    ax.text(W / 2, 0.90, f'"{REVIEW}"', ha="center", va="center", fontsize=FS, style="italic",
            family=SANS, color=INK)
    end = chips(fig, ax, 0.9, 0.48, p, ids=ids)
    row_label(ax, 0.82, 0.48, "pieces", ha="right", color=GREY)
    row_label(ax, 0.82, 0.27, "IDs", ha="right", color=GREY)
    assert end < W
    save(fig, "tok_text_to_ids.pdf", log)


def fig_three_ways(T, R, log):
    chars = list(REVIEW)
    words = regex.findall(r"\w+|[^\w\s]", REVIEW)
    sub = as_text(T["gpt4o"].split(REVIEW)[0])
    R["three_ways"] = {"chars": len(chars), "words": len(words), "subwords_gpt4o": len(sub)}
    fig, ax = canvas(1.65)
    rows = [(1.40, "characters", f"{len(chars)} tokens", chars, {"fs": 7, "pad": 0.03, "gap": 0.012}),
            (0.85, "words", f"{len(words)} tokens", words, {}),
            (0.30, "subwords (GPT-4o)", f"{len(sub)} tokens", sub, {})]
    for y, name, count, pieces, kw in rows:
        row_label(ax, 0.02, y + 0.07, name, bold=True)
        row_label(ax, 0.02, y - 0.10, count, color=GREY)
        chips(fig, ax, 1.25, y, pieces, h=0.20, **kw)
    save(fig, "tok_three_ways.pdf", log)


def fig_unk(T, R, log):
    typo = "The pomegranatey arrived fresh"
    words = typo.split()
    known = ["<UNK>" if w == "pomegranatey" else w for w in words]
    sub = as_text(T["gpt4o"].split(typo)[0])
    R["unk"] = {"word_level": known, "gpt4o": sub}
    fig, ax = canvas(0.85)
    row_label(ax, 0.02, 0.62, "word list", bold=True)
    chips(fig, ax, 1.25, 0.62, known, hi={1}, visible=False)
    row_label(ax, 0.02, 0.22, "subwords (GPT-4o)", bold=True)
    chips(fig, ax, 1.25, 0.22, sub)
    save(fig, "tok_unk.pdf", log)


def fig_bytes(R, log):
    rows = [("a", "a"), ("\u0561", "\u0561 (Armenian)"), ("\u20ac", "\u20ac (euro)"),
            ("\U0001F600", "an emoji")]
    R["bytes"] = []
    fig, ax = canvas(1.62)
    for k, (ch, name) in enumerate(rows):
        y = 1.22 - k * 0.33
        b = ch.encode("utf-8")
        R["bytes"].append({"char": name, "code_point": f"U+{ord(ch):04X}", "bytes": b.hex(" ")})
        row_label(ax, 0.05, y, name, size=FS)
        row_label(ax, 1.45, y, f"U+{ord(ch):04X}", color=GREY, size=FS)
        chips(fig, ax, 2.35, y, [f"{v:02X}" for v in b], fills=[TINTS[k % 3]] * len(b),
              visible=False)
        row_label(ax, 4.15, y, f"{len(b)} byte{'s' if len(b) > 1 else ''}", color=GREY, size=FS)
    for x, t in ((0.05, "character"), (1.45, "code point"), (2.35, "UTF-8 bytes")):
        row_label(ax, x, 1.50, t, color=GREY, size=FS_LAB)
    save(fig, "tok_bytes.pdf", log)


def fig_bpe_encode(R, log):
    steps = toy_bpe(TOY_CORPUS, 7)
    merges = [tuple(s["pair"]) for s in steps]
    R["toy_bpe"] = steps
    trace = bpe_encode_steps("lowest", merges)
    R["lowest_trace"] = trace
    assert trace[-1][1] == ["low", "est"], trace
    fig, ax = canvas(0.30 * len(trace) + 0.05)
    for k, (merge, syms) in enumerate(trace):
        y = 0.30 * (len(trace) - k) - 0.12
        row_label(ax, 0.05, y, "start: letters" if merge is None else f"merge {merge}",
                  color=GREY if merge is None else INK)
        chips(fig, ax, 1.55, y, syms, visible=False, h=0.20)
    save(fig, "tok_bpe_encode.pdf", log)
    return steps


def fig_regex(T, R, log):
    out = {}
    fig, ax = canvas(0.95)
    for k, (key, label) in enumerate([("gpt2", "GPT-2's rule"), ("gpt4", "GPT-4's rule")]):
        chunks = regex.findall(T[key].enc._pat_str, REGEX_SENTENCE)
        out[key] = chunks
        y = 0.70 - k * 0.45
        row_label(ax, 0.02, y, label, bold=True)
        chips(fig, ax, 1.25, y, chunks)
    R["regex"] = out
    save(fig, "tok_regex.pdf", log)


def fig_decode_trap(T, R, log):
    ch = "\u0561"
    p4, i4 = T["gpt4"].split(ch)
    po, io = T["gpt4o"].split(ch)
    R["decode_trap"] = {"gpt4": [[p.hex(), i] for p, i in zip(p4, i4)],
                        "gpt4o": [[p.hex(), i] for p, i in zip(po, io)]}
    assert len(p4) == 2 and len(po) == 1, (p4, po)
    fig, ax = canvas(1.0)
    row_label(ax, 0.02, 0.72, "GPT-4", bold=True)
    x = 0.95
    for p, i in zip(p4, i4):
        x = chips(fig, ax, x, 0.72, [f"{i}: {p.hex().upper()}"], visible=False, hi={0})
    ax.text(x + 0.05, 0.72, "each alone: \ufffd \ufffd    together: " + ch, ha="left",
            va="center", fontsize=FS, family=SANS, color=ARMRED)
    row_label(ax, 0.02, 0.25, "GPT-4o", bold=True)
    x = chips(fig, ax, 0.95, 0.25, [f"{io[0]}: {po[0].hex().upper()}"], visible=False)
    ax.text(x + 0.05, 0.25, "one token: " + ch, ha="left", va="center", fontsize=FS,
            family=SANS, color=INK)
    save(fig, "tok_decode_trap.pdf", log)


def fig_vocab_tradeoff(T, R, log):
    names = [("gpt2", "GPT-2"), ("gpt4", "GPT-4"), ("gpt4o", "GPT-4o")]
    sample = ALICE + "\n\n" + CODE
    vocab = [T[k].enc.n_vocab for k, _ in names]
    toks = [len(T[k].split(sample)[1]) for k, _ in names]
    R["vocab_tradeoff"] = {"vocab": vocab, "tokens_same_text": toks,
                           "sample_chars": len(sample)}
    assert vocab == [QUOTED["vocab_gpt2"], QUOTED["vocab_gpt4"], QUOTED["vocab_gpt4o"]], vocab
    fig = plt.figure(figsize=(W, 1.9))
    colors = [ARMRED, ARMBLUE, ARMORANGE]
    for k, (vals, title, fmt) in enumerate([(vocab, "vocabulary size", "{:,}"),
                                            (toks, f"tokens for the same text ({len(sample)} chars)", "{}")]):
        ax = fig.add_axes([0.07 + k * 0.5, 0.16, 0.40, 0.66])
        bars = ax.bar([n for _, n in names], vals, color=colors, width=0.6)
        ax.bar_label(bars, labels=[fmt.format(v) for v in vals], fontsize=FS_ID, padding=2)
        ax.set_title(title, fontsize=FS, pad=6)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.tick_params(labelsize=FS_ID, length=2)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    save(fig, "tok_vocab_tradeoff.pdf", log)


def fig_wordpiece(R, log):
    tok = Tokenizer.from_pretrained("bert-base-uncased")
    words = ["tokenization", "pomegranates", "unkindness"]
    out = {}
    fig, ax = canvas(0.30 * len(words) + 0.08)
    for k, w in enumerate(words):
        p, ids = hf_split(tok, w, use_tokens=True)
        out[w] = p
        y = 0.30 * (len(words) - k) - 0.12
        row_label(ax, 0.05, y, w, color=GREY, size=FS)
        chips(fig, ax, 1.45, y, p, ids=None, h=0.20, visible=False)
    R["wordpiece_bert"] = out
    save(fig, "tok_wordpiece.pdf", log)


def fig_unigram(R, log):
    tok = Tokenizer.from_pretrained("google-t5/t5-small")
    p, _ = hf_split(tok, REVIEW, use_tokens=True)
    R["unigram_t5"] = p
    fig, ax = canvas(0.45)
    chips(fig, ax, (W - 4.4) / 2, 0.24, p, visible=False)
    save(fig, "tok_unigram.pdf", log)


def fig_spelled(T, R, log):
    one = as_text(T["gpt4o"].split(" strawberry")[0])
    spelled = as_text(T["gpt4o"].split(" s t r a w b e r r y")[0])
    R["spelled"] = {"word": one, "spelled": spelled}
    assert len(one) == 1 and len(spelled) == 10, (one, spelled)
    fig, ax = canvas(0.85)
    row_label(ax, 0.02, 0.62, "as written", bold=True)
    chips(fig, ax, 1.25, 0.62, one)
    row_label(ax, 0.02, 0.22, "spelled out", bold=True)
    chips(fig, ax, 1.25, 0.22, spelled)
    save(fig, "tok_spelled.pdf", log)


def fig_numbers(T, R, log):
    qwen = Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    rows = [("GPT-2", as_text(T["gpt2"].split(SUM)[0])),
            ("GPT-4", as_text(T["gpt4"].split(SUM)[0])),
            ("Qwen 2.5", hf_split(qwen, SUM, use_tokens=False)[0])]
    R["numbers"] = {n: p for n, p in rows}
    fig, ax = canvas(1.2)
    for k, (name, p) in enumerate(rows):
        y = 0.98 - k * 0.38
        row_label(ax, 0.02, y, name, bold=True)
        row_label(ax, 0.02, y - 0.15, f"{len(p)} tokens", color=GREY)
        chips(fig, ax, 1.25, y, p)
    save(fig, "tok_numbers.pdf", log)


def fig_code(T, R, log):
    out = {}
    fig, ax = canvas(1.55)
    for col, (key, name) in enumerate([("gpt2", "GPT-2"), ("gpt4", "GPT-4")]):
        p = as_text(T[key].split(CODE)[0])
        out[key] = p
        x0 = 0.05 + col * 2.78
        row_label(ax, x0, 1.40, f"{name}: {len(p)} tokens", bold=True)
        x, y = x0, 1.08
        for k, piece in enumerate(p):
            x = chips(fig, ax, x, y, [piece], fs=7, pad=0.04, gap=0.012, h=0.19,
                      fills=[TINTS[k % 3]], xmax=x0 + 2.70)
            if "\n" in piece:
                x, y = x0, y - 0.34
    R["code"] = out
    assert len(out["gpt2"]) == QUOTED["code_gpt2"] and len(out["gpt4"]) == QUOTED["code_gpt4"], out
    save(fig, "tok_code.pdf", log)


def fig_special(T, R, log):
    s = "<|endoftext|>"
    as_txt, ids_txt = T["gpt4"].split(s)
    as_sp, ids_sp = T["gpt4"].split(s, allow_special=True)
    try:
        T["gpt4"].enc.encode(s)
        refuses = False
    except ValueError:
        refuses = True
    R["special"] = {"as_text": as_text(as_txt), "as_special_ids": ids_sp,
                    "tiktoken_default_refuses": refuses}
    assert len(as_txt) == QUOTED["endoftext_as_text_gpt4"] and len(ids_sp) == 1 and refuses
    fig, ax = canvas(0.95)
    row_label(ax, 0.02, 0.70, "read as text", bold=True)
    x = chips(fig, ax, 1.95, 0.70, as_text(as_txt))
    row_label(ax, x + 0.08, 0.70, f"{len(as_txt)} ordinary tokens", color=GREY)
    row_label(ax, 0.02, 0.28, "read as a control token", bold=True)
    x = chips(fig, ax, 1.95, 0.28, [s], hi={0}, visible=False)
    row_label(ax, x + 0.08, 0.28, f"1 token, ID {ids_sp[0]}", color=ARMRED)
    save(fig, "tok_special.pdf", log)


def fig_armenian(T, R, log):
    line = ARM_LINE_FILE.read_text(encoding="utf-8").strip()
    counts = {k: (len(T[k].split(line)[1]), len(T[k].split(GLOSS)[1])) for k in ("gpt4", "gpt4o")}
    R["armenian"] = {"line": line, "gloss": GLOSS, "counts": counts,
                     "letters": len(line.replace(" ", "")), "chars": len(line),
                     "utf8_bytes": len(line.encode("utf-8"))}
    assert counts["gpt4"] == (QUOTED["arm_gpt4"], QUOTED["gloss_gpt4"]), counts
    assert counts["gpt4o"] == (QUOTED["arm_gpt4o"], QUOTED["gloss_gpt4o"]), counts
    assert len(line.encode("utf-8")) == QUOTED["arm_bytes"], len(line.encode("utf-8"))
    fig = plt.figure(figsize=(W, 2.15))
    fig.text(0.5, 0.94, line, ha="center", va="center", fontsize=FS, family=SANS, color=ARMRED)
    fig.text(0.5, 0.85, GLOSS, ha="center", va="center", fontsize=FS, family=SANS,
             style="italic", color=ARMBLUE)
    ax = fig.add_axes([0.24, 0.13, 0.52, 0.60])
    xs = [0, 1]
    for j, (lab, color) in enumerate([("Armenian line", ARMRED), ("English gloss", ARMBLUE)]):
        vals = [counts["gpt4"][j], counts["gpt4o"][j]]
        bars = ax.bar([x + (j - 0.5) * 0.36 for x in xs], vals, width=0.34, color=color, label=lab)
        ax.bar_label(bars, fontsize=FS, padding=2)
    ax.set_xticks(xs, ["GPT-4's tokenizer", "GPT-4o's tokenizer"], fontsize=FS)
    ax.set_yticks([])
    ax.set_ylim(0, 125)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=FS_ID, frameon=False, loc="upper right")
    save(fig, "tok_armenian.pdf", log)


def byte_bpe(words, n_merges):
    """BPE on UTF-8 bytes of a word-count corpus. Ties go to the pair seen first."""
    seqs = {w: [bytes([b]) for b in w.encode("utf-8")] for w in words}
    steps = []
    for _ in range(n_merges):
        counts = Counter()
        for w, n in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]):
                counts[(a, b)] += n
        (a, b), n = max(counts.items(), key=lambda kv: kv[1])
        steps.append((a, b, n))
        for w in seqs:
            s, out, i = seqs[w], [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = out
    return steps


def as_letter(b):
    """A byte string as text if it is valid UTF-8, else its bytes in hex (the half-letter case)."""
    try:
        return b.decode("utf-8"), True
    except UnicodeDecodeError:
        return " ".join(f"{x:02X}" for x in b), False


def fig_bpe_bytes(R, log):
    names = [l.strip() for l in SURNAMES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(names) == QUOTED["surnames"], len(names)
    steps = byte_bpe(Counter(names), 6)
    R["bpe_surname_bytes"] = [[a.hex(), b.hex(), (a + b).hex(), n] for a, b, n in steps]
    assert (steps[0][0] + steps[0][1]).decode("utf-8") == "ա", steps[0]           # merge 1: ayb
    assert as_letter(steps[1][0] + steps[1][1])[1] is False, steps[1]                   # merge 2: half a letter
    assert (steps[4][0] + steps[4][1]).decode("utf-8") == "յան", steps[4]  # merge 5: -yan
    fig, ax = canvas(0.27 * len(steps) + 0.28)
    for x, t in ((0.05, "merge"), (0.75, "left + right"), (3.05, "new token"), (4.55, "count")):
        row_label(ax, x, 0.27 * len(steps) + 0.13, t, color=GREY)
    for k, (a, b, n) in enumerate(steps):
        y = 0.27 * (len(steps) - k) - 0.04
        row_label(ax, 0.05, y, f"{k + 1}", size=FS)
        x = 0.75
        for j, part in enumerate((a, b)):
            txt, ok = as_letter(part)
            x = chips(fig, ax, x, y, [txt], visible=False, fills=["#EEEEEE" if not ok else TINTS[0]])
            if j == 0:
                row_label(ax, x + 0.02, y, "+", size=FS)
                x += 0.15
        row_label(ax, 2.75, y, "→", size=FS)
        txt, ok = as_letter(a + b)
        end = chips(fig, ax, 3.05, y, [txt], visible=False, hi=set() if ok else {0},
                    fills=[TINTS[1] if ok else "#FFFFFF"])
        if not ok:
            row_label(ax, end + 0.05, y, "not valid text", color=ARMRED, size=FS_ID)
        row_label(ax, 4.55, y, f"{n:,}", color=GREY, size=FS)
    save(fig, "tok_bpe_bytes.pdf", log)


def fig_id_ranks(T, R, log):
    enc = T["gpt2"].enc
    for i, piece in QUOTED["gpt2_ids"].items():
        assert enc.decode_single_token_bytes(i) == piece, (i, enc.decode_single_token_bytes(i))
    assert all(enc._mergeable_ranks[enc.decode_single_token_bytes(i)] == i for i in range(256, 5000))
    groups = [("the 256 bytes", [(0, enc.decode_single_token_bytes(0)), (1, enc.decode_single_token_bytes(1)),
                                 None, (255, enc.decode_single_token_bytes(255))]),
              ("merges, in the order learned", [(256, b" t"), (257, b" a"), (258, b"he"), None,
                                                (50255, b" gazed")]),
              ("special", [(50256, b"<|endoftext|>")])]
    R["gpt2_id_ranks"] = {str(i): enc.decode_single_token_bytes(i).hex() for i in (0, 1, 255, 256, 257, 258, 50255, 50256)}
    fig, ax = canvas(0.72)
    x = 0.05
    for name, items in groups:
        x0 = x
        for it in items:
            if it is None:
                row_label(ax, x + 0.02, 0.36, "...", color=GREY, size=FS)
                x += 0.22
                continue
            i, piece = it
            txt, ok = as_letter(piece)
            x = chips(fig, ax, x, 0.36, [txt if ok and txt.strip() else (txt if ok else "\\x" + piece.hex())],
                      ids=[i], visible=True, hi={0} if i == 50256 else set())
        row_label(ax, (x0 + x) / 2, 0.60, name, color=GREY, size=FS_ID, ha="center")
        x += 0.18
    save(fig, "tok_id_ranks.pdf", log)


def wordpiece_toy(R, log):
    sym, pair = Counter(), Counter()
    for w, n in TOY_CORPUS.items():
        for ch in w:
            sym[ch] += n
        for a, b in zip(w, w[1:]):
            pair[(a, b)] += n
    out = {f"{a}+{b}": [n, sym[a], sym[b], n / (sym[a] * sym[b])] for (a, b), n in pair.items()}
    R["wordpiece_toy"] = out
    for (a, b), (n, ca, cb) in QUOTED["wordpiece"].items():
        assert (pair[(a, b)], sym[a], sym[b]) == (n, ca, cb), ((a, b), pair[(a, b)], sym[a], sym[b])
    best = max(pair, key=lambda p: pair[p] / (sym[p[0]] * sym[p[1]]))
    assert best == ("i", "d"), best
    log.info(f"WordPiece first merge {best}; BPE's e+s scores {9 / (17 * 9):.4f}")


def unigram_toy(R, log):
    """One simplified Unigram pruning step. Simplification (stated on the slide): piece
    probabilities come from how often each piece is used in the current best splits (one
    hard-EM pass), not full EM over all splits."""
    import math
    seed = sorted(set("".join(TOY_CORPUS))) + ["low", "er", "new", "est", "wid", "es", "lo"]

    def best_split(word, logp):
        best = [(0.0, [])] + [(-math.inf, None)] * len(word)
        for end in range(1, len(word) + 1):
            for start in range(end):
                p = word[start:end]
                if p in logp and best[start][1] is not None and best[start][0] + logp[p] > best[end][0]:
                    best[end] = (best[start][0] + logp[p], best[start][1] + [p])
        return best[len(word)]

    def fit(vocab):
        occ = Counter()
        for w, n in TOY_CORPUS.items():
            for p in vocab:
                occ[p] += n * sum(1 for i in range(len(w)) if w.startswith(p, i))
        tot = sum(occ.values())
        logp = {p: math.log(occ[p] / tot) for p in vocab if occ[p] > 0}
        used = Counter()
        for w, n in TOY_CORPUS.items():
            for p in best_split(w, logp)[1]:
                used[p] += n
        tot = sum(used.values())
        logp = {p: math.log(c / tot) for p, c in used.items()}
        loss = -sum(n * best_split(w, logp)[0] for w, n in TOY_CORPUS.items())
        return loss, {w: best_split(w, logp)[1] for w in TOY_CORPUS}, used

    loss, splits, used = fit(seed)
    deltas = {p: round(fit([x for x in seed if x != p])[0] - loss, 2) for p in seed if len(p) > 1}
    R["unigram_toy"] = {"seed": seed, "loss": round(loss, 2), "splits": splits,
                        "pieces_used": dict(used), "loss_increase_if_removed": deltas}
    assert round(loss, 2) == QUOTED["unigram_loss"], loss
    assert deltas == QUOTED["unigram"], deltas
    log.info(f"Unigram: loss {loss:.2f}, splits {splits}, deltas {deltas}")


def main():
    log = setup_logging()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    T = {"gpt2": Tik("gpt2"), "gpt4": Tik("cl100k_base"), "gpt4o": Tik("o200k_base")}
    R = {"tiktoken": tiktoken.__version__, "date": time.strftime("%Y-%m-%d")}
    jobs = [lambda: fig_question(T, R, log), lambda: fig_text_to_ids(T, R, log),
            lambda: fig_three_ways(T, R, log), lambda: fig_unk(T, R, log),
            lambda: fig_bytes(R, log), lambda: fig_bpe_encode(R, log),
            lambda: fig_regex(T, R, log), lambda: fig_decode_trap(T, R, log),
            lambda: fig_vocab_tradeoff(T, R, log), lambda: fig_wordpiece(R, log),
            lambda: fig_unigram(R, log), lambda: fig_spelled(T, R, log),
            lambda: fig_numbers(T, R, log), lambda: fig_code(T, R, log),
            lambda: fig_special(T, R, log), lambda: fig_armenian(T, R, log),
            lambda: fig_bpe_bytes(R, log), lambda: fig_id_ranks(T, R, log),
            lambda: wordpiece_toy(R, log), lambda: unigram_toy(R, log)]
    for k, job in enumerate(jobs, 1):
        job()
        el = time.time() - t0
        log.info(f"{k}/{len(jobs)}, {el:.0f}s elapsed, ~{el / k * (len(jobs) - k):.0f}s left")
    (RESULTS / "tokenization_figs.json").write_text(json.dumps(R, indent=2, ensure_ascii=False,
                                                               default=str), encoding="utf-8")
    for s in R["toy_bpe"]:
        log.info(f"toy BPE merge {s['pair']} count {s['count']}; top {s['top_counts'][:4]}")
    log.info(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
