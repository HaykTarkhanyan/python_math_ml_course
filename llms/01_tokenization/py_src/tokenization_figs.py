"""Figures for LLM-1 "Tokenization: from text to token IDs" (llms/01_tokenization/LLM1_tokenization.tex).

Every token split on the slides is measured here, never typed by hand:
  tiktoken  gpt2 (GPT-2), cl100k_base (GPT-4), o200k_base (GPT-4o)
  HF tokenizers  bert-base-uncased (WordPiece), google-t5/t5-small (SentencePiece Unigram),
                 Qwen/Qwen2.5-0.5B-Instruct (byte-level BPE that splits every digit)
Numbers the slides quote in text are asserted (QUOTED below), so a changed tokenizer fails
loudly instead of silently contradicting the deck. The toy BPE run (frames 16-19) is computed
here too and printed to the log.

Drawn at the size each figure gets on the slide (5.5 in = full text width), 7-8 pt text.
Chapter constants reused verbatim: [53]'s review sentence; the Armenian line from
ml/13_rnns/py_src/data/armenian_line.txt (one shared file); [54]'s locked English gloss.

Generates into llms/01_tokenization/fig/: tok_*.pdf (one per figure function below).
Writes llms/01_tokenization/results/tokenization_figs.json (every measured split - the artifact of record).

Added 2026-10-27 (the deck's extension): number splits and the comma trick, reversing a word,
JSON vs YAML, the trailing space, glitch tokens across vocabularies, the longest tokens, GPT-4o's
lottery tokens, and SentencePiece Unigram on T5 (n-best splits, sampled splits, no byte fallback)
and on ch11's surnames (a model trained here, in a subprocess so its pruning log can be read).

Run: ./ma/Scripts/python.exe llms/01_tokenization/py_src/tokenization_figs.py   (~1.5 min; the first run also
downloads the BERT, T5 and Qwen tokenizer files, a few MB)
Supersedes ml/13_rnns/py_src/tokenizer_demo.py for this deck (its panels were drawn 13 in wide).
"""
import json
import logging
import math
import re
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import regex
import sentencepiece as spm
import tiktoken
import yaml
from huggingface_hub import hf_hub_download
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

REVIEW = "The pomegranates arrived fresh and sweet."          # [53]'s sentence
GLOSS = "I watch all of this in silence, and the connoisseurs speak inside me."   # [54], locked
QUESTION = "How many r's in strawberry?"
SUM = "127 + 677 = 804"
CODE = "def f(x):\n        if x:\n            return 1"
ALICE = ("Alice was beginning to get very tired of sitting by her sister on the bank, and of "
         "having nothing to do: once or twice she had peeped into the book her sister was "
         "reading, but it had no pictures or conversations in it, 'and what is the use of a "
         "book,' thought Alice 'without pictures or conversations?'")   # Carroll 1865
REGEX_SENTENCE = "I'll pay 1234567 for 3 dogs!!!"
TOY_CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}   # Sennrich et al. 2016's example
FORMAT_DATA = {"name": "Ani", "age": 21, "courses": ["math", "ml"]}
PROMPT = "The capital of France is"
LOTTERY = [" 天天中彩票", " 微信公众号天天中彩票", " 全民彩票天天送钱", " 大发快三开奖结果"]
LOTTERY_EN = ["win the lottery every day", "WeChat account: win the lottery every day",
              "lottery for everyone, free money every day", "Dafa Fast-3 (a lottery game): draw results"]
THAI_BET = " วิเคราะห์บอลวันนี้"          # Thai: "today's football analysis" (betting tips)

# Numbers the deck states in prose. If a tokenizer update changes one, fix the slide too.
QUOTED = {"question_tokens": 7, "code_gpt2": 30, "code_gpt4": 12, "arm_gpt4": 102,
          "gloss_gpt4": 18, "arm_gpt4o": 19, "gloss_gpt4o": 17, "arm_bytes": 102,
          "arm_letters": 57, "vocab_gpt2": 50257, "vocab_gpt4": 100277, "vocab_gpt4o": 200019,
          "endoftext_as_text_gpt4": 7, "surnames": 689,
          # "Back to strawberry": other spellings of the word that GPT-4o cuts into several tokens
          "strawberry_variants": {"strawberry": ["st", "raw", "berry"],
                                  " STRAWBERRY": [" STR", "AW", "B", "ERRY"]},
          # GPT-2 IDs: bytes 0-255, then merges in the order learned (the "Where IDs come from" frame)
          "gpt2_ids": {256: b" t", 257: b" a", 258: b"he", 50255: b" gazed", 50256: b"<|endoftext|>"},
          # WordPiece score on the toy corpus, first step (the "WordPiece by hand" table)
          "wordpiece": {("i", "d"): (3, 3, 3), ("l", "o"): (7, 7, 7), ("s", "t"): (9, 9, 9),
                        ("o", "w"): (7, 7, 16), ("e", "s"): (9, 17, 9), ("w", "e"): (8, 16, 17)},
          # one simplified Unigram step: loss increase when each piece is removed (2 decimals)
          "unigram": {"es": 0.00, "lo": 0.00, "er": 7.28, "low": 17.29, "wid": 19.81,
                      "est": 20.24, "new": 32.39}, "unigram_loss": 40.16,
          # the by-hand line under it: uses in the best splits, and what the 5 copies of "low" cost
          "unigram_uses": {"est": 9, "low": 7, "new": 6, "wid": 3, "er": 2}, "unigram_low_cost": 6.75,
          # ---- added 2026-10-27
          # "Numbers look random to GPT-2": how many of 0..9999 are one token (no leading space)
          "numbers_single": {"gpt2": 907, "gpt2_4digit": 88, "gpt4": 1000, "gpt4_4digit": 0},
          "gpt2_4digit_years_1900_2099": 64,
          "years_gpt2": {" 2022": [" 2022"], " 2023": [" 20", "23"], " 2024": [" 2024"],
                         " 2025": [" 2025"], " 2026": [" 20", "26"], " 2048": [" 2048"]},
          # neighbours cut three ways (Millidge 2023's example); every integer 0-520 is one token
          "same_length_gpt2": {"2249": ["2", "249"], "2250": ["22", "50"], "2251": ["225", "1"]},
          "gpt2_all_single_up_to": 520,   # 521 is the first that splits
          "sums_gpt4": {"12345 + 678 = 13023": ["123", "45", " +", " ", "678", " =", " ", "130", "23"],
                        "12,345 + 678 = 13,023": ["12", ",", "345", " +", " ", "678", " =", " ", "13", ",",
                                                  "023"]},
          # [pretty JSON, compact JSON, YAML] tokens for FORMAT_DATA
          "formats": {"gpt2": [43, 23, 19], "gpt4": [32, 22, 18], "gpt4o": [31, 22, 17]},
          "reverse_gpt4o": {" tokenization": [" token", "ization"],
                            " noitazinekot": [" no", "it", "azine", "kot"],
                            " strawberry": [" strawberry"], " yrrebwarts": [" yr", "reb", "warts"]},
          "trailing_gpt4": {" Paris": 12366, "Paris": 60704, " ": 220},
          # [GPT-2, GPT-4, GPT-4o] token counts
          "glitch_counts": {" SolidGoldMagikarp": [1, 5, 5], " TheNitromeFan": [1, 4, 4],
                            "rawdownloadcloneembedreportprint": [1, 6, 6], " petertodd": [1, 3, 3],
                            " davidjl": [1, 1, 2]},
          "longest": {"gpt2": "rawdownloadcloneembedreportprint",
                      "gpt4": ".translatesAutoresizingMaskIntoConstraints",
                      "gpt4o": "abcdefghijklmnopqrstuvwxyz"},
          "lottery_tokens": {"gpt2": 0, "gpt4": 0, "gpt4o": 207},
          # T5's SentencePiece Unigram: total log-probabilities, natural log, 2 decimals
          "t5_strawberry": {"whole": -12.93, "straw": -12.69, "berry": -11.39},
          "t5_pomegranates_top2": [-41.62, -41.64], "t5_arm": {"tokens": 25, "unk": 12}}

W = 5.5
FS, FS_ID, FS_LAB = 8, 7, 7.5
SANS = ["DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]
WIDE = ["DejaVu Sans Mono", "Microsoft YaHei", "Leelawadee UI"]   # CJK and Thai tokens (fallback list)
NAMES = [l.strip() for l in SURNAMES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
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
          fills=None, xmax=W - 0.02, visible=True, family=MONO):
    """Draw tokens as chips left to right; returns the x after the last chip. Raises on overflow."""
    for k, p in enumerate(pieces):
        s = show(p) if visible else p
        w = text_w(fig, s, fs, family) + pad
        if x + w > xmax:
            raise ValueError(f"chip row overflows at {p!r} (x={x + w:.2f} > {xmax:.2f})")
        fc = fills[k] if fills else TINTS[k % 3]
        ec, lw = (ARMRED, 1.3) if k in hi else ("#999999", 0.5)
        rbox(ax, x, y - h / 2, w, h, fc, ec, lw)
        ax.text(x + w / 2, y, s, ha="center", va="center", fontsize=fs, family=family, color=INK,
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
    variants = {s: as_text(T["gpt4o"].split(s)[0]) for s in QUOTED["strawberry_variants"]}
    R["spelled"] = {"word": one, "spelled": spelled, "variants": variants}
    assert len(one) == 1 and len(spelled) == 10, (one, spelled)
    assert variants == QUOTED["strawberry_variants"], variants
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
    assert dict(used) == QUOTED["unigram_uses"], dict(used)     # 27 uses in all
    low_cost = TOY_CORPUS["low"] * math.log(sum(used.values()) / used["low"])
    assert splits["low"] == ["low"] and round(low_cost, 2) == QUOTED["unigram_low_cost"], low_cost
    log.info(f"Unigram: loss {loss:.2f}, splits {splits}, deltas {deltas}")


# ----------------------------------------------------------------------------- added 2026-10-27
def group_row(fig, ax, x, y, groups, gap=0.20, notes=False, **kw):
    """Several chip groups on one row, a gap between groups; optional '1 token' notes under each."""
    for pieces in groups:
        x0 = x
        x = chips(fig, ax, x, y, pieces, **kw)
        if notes:
            n = len(pieces)
            ax.text((x0 + x) / 2, y - 0.21, f"{n} token{'s' if n > 1 else ''}", ha="center",
                    va="center", fontsize=FS_ID, family=SANS, color=ARMRED if n > 1 else GREY)
        x += gap
    return x


def single_token_numbers(enc):
    one = [n for n in range(10000) if len(enc.encode(str(n))) == 1]
    return len(one), sum(1 for n in one if n >= 1000)


def fig_years(T, R, log):
    """'Numbers look random to GPT-2': years and same-length numbers as GPT-2 cuts them."""
    counts = {k: single_token_numbers(T[k].enc) for k in ("gpt2", "gpt4")}
    q = QUOTED["numbers_single"]
    assert counts == {"gpt2": (q["gpt2"], q["gpt2_4digit"]), "gpt4": (q["gpt4"], q["gpt4_4digit"])}, counts
    years = {s: as_text(T["gpt2"].split(s)[0]) for s in QUOTED["years_gpt2"]}
    same = {s: as_text(T["gpt2"].split(s)[0]) for s in QUOTED["same_length_gpt2"]}
    assert years == QUOTED["years_gpt2"] and same == QUOTED["same_length_gpt2"], (years, same)
    enc, top = T["gpt2"].enc, QUOTED["gpt2_all_single_up_to"]
    years4 = sum(1 for n in range(1900, 2100) if len(enc.encode(str(n))) == 1)
    assert years4 == QUOTED["gpt2_4digit_years_1900_2099"], years4
    assert all(len(enc.encode(str(n))) == 1 for n in range(top + 1)) and len(enc.encode(str(top + 1))) == 2
    R["numbers_gpt2"] = {"single_token_0_9999": counts, "years": years, "same_length": same,
                         "all_single_up_to": top}
    fig, ax = canvas(1.25)
    row_label(ax, 0.02, 0.98, "years", bold=True)
    group_row(fig, ax, 0.95, 0.98, list(years.values()), notes=True, gap=0.16)
    row_label(ax, 0.02, 0.38, "neighbours", bold=True)
    group_row(fig, ax, 0.95, 0.38, list(same.values()), notes=True, gap=0.45)
    save(fig, "tok_years.pdf", log)


def fig_commas(T, R, log):
    """The same sum without and with thousands separators, as GPT-4's tokenizer cuts it."""
    rows = {s: as_text(T["gpt4"].split(s)[0]) for s in QUOTED["sums_gpt4"]}
    assert rows == QUOTED["sums_gpt4"], rows
    R["sums_gpt4"] = rows
    lengths = {n: as_text(T["gpt4"].split(n)[0]) for n in ("12345", "1234")}
    assert lengths == {"12345": ["123", "45"], "1234": ["123", "4"]}, lengths   # the by-hand frame
    fig, ax = canvas(0.95)
    for k, (label, p) in enumerate(zip(["no commas", "with commas"], rows.values())):
        y = 0.70 - k * 0.45
        row_label(ax, 0.02, y, label, bold=True)
        chips(fig, ax, 1.25, y, p, gap=0.03)
    save(fig, "tok_commas.pdf", log)


def fig_reverse(T, R, log):
    out = {s: as_text(T["gpt4o"].split(s)[0]) for s in QUOTED["reverse_gpt4o"]}
    assert out == QUOTED["reverse_gpt4o"], out
    R["reverse_gpt4o"] = out
    fig, ax = canvas(0.85)
    items = list(out.items())
    for col in range(2):
        x0 = 0.05 + col * 2.75
        for k, lab in enumerate(["word", "reversed"]):
            y = 0.62 - k * 0.40
            row_label(ax, x0, y, lab, color=GREY)
            chips(fig, ax, x0 + 0.70, y, items[2 * col + k][1], xmax=x0 + 2.70)
    save(fig, "tok_reverse.pdf", log)


def fig_formats(T, R, log):
    """The same record as pretty JSON, compact JSON and YAML: GPT-4's token counts."""
    texts = [json.dumps(FORMAT_DATA, indent=2), json.dumps(FORMAT_DATA),
             yaml.safe_dump(FORMAT_DATA, sort_keys=False)]
    counts = {k: [len(T[k].split(t)[1]) for t in texts] for k in ("gpt2", "gpt4", "gpt4o")}
    assert counts == QUOTED["formats"], counts
    R["formats"] = {"texts": texts, "tokens": counts}
    fig = plt.figure(figsize=(W, 1.75))
    for x, title, t in ((0.01, "JSON, indented", texts[0]), (0.29, "YAML", texts[2])):
        fig.text(x, 0.93, title, fontsize=FS, family=SANS, fontweight="bold", va="center")
        fig.text(x, 0.84, t.rstrip("\n"), fontsize=6.5, family=MONO, va="top", color=INK,
                 linespacing=1.15)
    ax = fig.add_axes([0.62, 0.18, 0.36, 0.62])
    names = ["JSON\nindented", "JSON\none line", "YAML"]
    bars = ax.bar(names, counts["gpt4"], color=[ARMRED, ARMORANGE, ARMBLUE], width=0.6)
    ax.bar_label(bars, fontsize=FS_ID, padding=2)
    ax.set_title("tokens (GPT-4's tokenizer)", fontsize=FS, pad=6)
    ax.set_ylim(0, max(counts["gpt4"]) * 1.25)
    ax.set_yticks([])
    ax.tick_params(labelsize=FS_ID, length=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "tok_formats.pdf", log)


def fig_trailing(T, R, log):
    """A prompt with and without a trailing space: the space becomes its own token."""
    ids = {s: T["gpt4"].enc.encode(s) for s in QUOTED["trailing_gpt4"]}
    assert {s: v for s, v in ids.items()} == {s: [i] for s, i in QUOTED["trailing_gpt4"].items()}, ids
    rows = [(PROMPT, " Paris"), (PROMPT + " ", "Paris")]
    R["trailing_gpt4"] = {}
    fig, ax = canvas(1.10)
    for k, (prompt, nxt) in enumerate(rows):
        p, i = T["gpt4"].split(prompt)
        p = as_text(p)
        R["trailing_gpt4"][prompt] = {"pieces": p, "ids": i, "next": nxt, "next_id": ids[nxt][0]}
        y = 0.82 - k * 0.52
        x = chips(fig, ax, 0.05, y, p, ids=i, hi={len(p) - 1} if k == 1 else set())
        row_label(ax, x + 0.08, y, "then", color=GREY)
        chips(fig, ax, x + 0.45, y, [nxt], ids=ids[nxt], fills=["#FFFFFF"])
    assert R["trailing_gpt4"][PROMPT + " "]["pieces"][-1] == " ", R["trailing_gpt4"]
    save(fig, "tok_trailing.pdf", log)


def fig_glitch_split(T, R, log):
    """GPT-2's single-token glitch names, cut by GPT-4's tokenizer."""
    counts, splits = {}, {}
    for s in QUOTED["glitch_counts"]:
        counts[s] = [len(T[k].split(s)[1]) for k in ("gpt2", "gpt4", "gpt4o")]
        splits[s] = as_text(T["gpt4"].split(s)[0])
    assert counts == QUOTED["glitch_counts"], counts
    R["glitch_split"] = {"counts": counts, "gpt4_pieces": splits}
    show_rows = [" SolidGoldMagikarp", " TheNitromeFan", "rawdownloadcloneembedreportprint", " davidjl"]
    fig, ax = canvas(0.30 * len(show_rows) + 0.25)
    row_label(ax, 0.05, 0.30 * len(show_rows) + 0.10, "GPT-2: one token each", color=GREY)
    row_label(ax, 2.55, 0.30 * len(show_rows) + 0.10, "GPT-4's tokenizer", color=GREY)
    for k, s in enumerate(show_rows):
        y = 0.30 * (len(show_rows) - k) - 0.10
        chips(fig, ax, 0.05, y, [s], fs=7, xmax=2.45)
        p = splits[s]
        chips(fig, ax, 2.55, y, p, fs=7, hi={0} if len(p) == 1 else set())
    save(fig, "tok_glitch_split.pdf", log)


def longest_tokens(enc, n):
    """The n longest tokens (in characters) that contain at least one letter. Ties: lower ID first."""
    out = []
    for i in range(enc.n_vocab):
        try:
            s = enc.decode_single_token_bytes(i).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
        if regex.search(r"\p{L}", s):
            out.append((-len(s), i, s))
    return [(s, -m) for m, i, s in sorted(out)[:n]]


def fig_longest(T, R, log):
    top = {k: longest_tokens(T[k].enc, 30) for k in ("gpt2", "gpt4", "gpt4o")}
    R["longest_tokens"] = {k: v[:12] for k, v in top.items()}
    mojibake = "ÃÂ" * 32
    assert top["gpt2"][0] == (mojibake, 64), top["gpt2"][:2]
    ascii_word = lambda s: re.fullmatch(r" ?[A-Za-z]+", s)
    for k, want in QUOTED["longest"].items():
        first_word = next(s for s, _ in top[k] if ascii_word(s.lstrip(".")))
        assert first_word == want, (k, first_word)
    assert top["gpt4"][0][0] == QUOTED["longest"]["gpt4"], top["gpt4"][:2]
    alpha = {k: len(T[k].split("abcdefghijklmnopqrstuvwxyz")[1]) for k in T}
    assert alpha["gpt4"] == alpha["gpt4o"] == 1 < alpha["gpt2"], alpha
    R["alphabet_tokens"] = alpha
    rows = [("GPT-2", mojibake, "mojibake"),
            ("", "rawdownloadcloneembedreportprint", "web-page buttons"),
            ("GPT-4", ".translatesAutoresizingMaskIntoConstraints", "Apple iOS code"),
            ("GPT-4o", "abcdefghijklmnopqrstuvwxyz", "the alphabet"),
            ("", " verantwoordelijkheid", "Dutch: responsibility"),
            ("", THAI_BET, "Thai: football betting tips")]
    in_top = {k: [s for s, _ in v] for k, v in top.items()}
    keys = {"GPT-2": "gpt2", "GPT-4": "gpt4", "GPT-4o": "gpt4o"}
    fig, ax = canvas(0.27 * len(rows) + 0.05)
    cur = None
    for k, (name, tok, what) in enumerate(rows):
        cur = keys.get(name, cur)
        assert tok in in_top[cur], (cur, tok)
        y = 0.27 * (len(rows) - k) - 0.10
        if name:
            row_label(ax, 0.02, y, name, bold=True)
        x = chips(fig, ax, 0.62, y, [tok], fs=6.5, family=WIDE, visible=False, h=0.19)
        row_label(ax, x + 0.06, y, f"{len(tok)} characters: {what}", color=GREY, size=FS_ID)
    save(fig, "tok_longest.pdf", log)


def fig_lottery(T, R, log):
    """GPT-4o's lottery-spam tokens: how many there are, and four of them, translated."""
    counts = {}
    for k in ("gpt2", "gpt4", "gpt4o"):
        enc, c = T[k].enc, 0
        for i in range(enc.n_vocab):
            try:
                c += "彩票" in enc.decode_single_token_bytes(i).decode("utf-8", "ignore")
            except KeyError:
                continue
        counts[k] = c
    assert counts == QUOTED["lottery_tokens"], counts
    for s in LOTTERY + [THAI_BET]:
        assert len(T["gpt4o"].split(s)[1]) == 1, s
    R["lottery"] = {"tokens_containing_lottery": counts, "examples": dict(zip(LOTTERY, LOTTERY_EN))}
    fig, ax = canvas(0.27 * len(LOTTERY) + 0.05)
    for k, (tok, en) in enumerate(zip(LOTTERY, LOTTERY_EN)):
        y = 0.27 * (len(LOTTERY) - k) - 0.10
        x = chips(fig, ax, 0.05, y, [tok], fs=7.5, family=WIDE, visible=False, h=0.20, xmax=2.45)
        row_label(ax, 2.55, y, en, size=FS_ID)
    save(fig, "tok_lottery.pdf", log)


def t5_model():
    return spm.SentencePieceProcessor(model_file=hf_hub_download("google-t5/t5-small", "spiece.model"))


def logp(sp, pieces):
    return sum(sp.get_score(sp.piece_to_id(p)) for p in pieces)


def fig_unigram_nbest(sp, R, log):
    """T5's Unigram: the five most probable splits of one word, and their scores."""
    word = "pomegranates"
    best = [(p, logp(sp, p)) for p in sp.nbest_encode_as_pieces(word, 5)]
    st = {"whole": logp(sp, ["▁strawberry"]), "straw": logp(sp, ["▁straw"]), "berry": logp(sp, ["berry"])}
    R["t5_unigram"] = {"pomegranates_nbest": best, "strawberry": st,
                       "strawberry_best": sp.encode_as_pieces("strawberry")}
    assert [round(s, 2) for _, s in best[:2]] == QUOTED["t5_pomegranates_top2"], best
    assert {k: round(v, 2) for k, v in st.items()} == QUOTED["t5_strawberry"], st
    assert sp.encode_as_pieces("strawberry") == ["▁strawberry"]
    fig, ax = canvas(0.25 * len(best) + 0.25)
    row_label(ax, 0.05, 0.25 * len(best) + 0.10, f'the 5 most probable splits of "{word}" (T5)',
              color=GREY)
    row_label(ax, 4.40, 0.25 * len(best) + 0.10, "log-probability", color=GREY)
    for k, (p, s) in enumerate(best):
        y = 0.25 * (len(best) - k) - 0.08
        chips(fig, ax, 0.05, y, p, visible=False, h=0.20, hi=set(range(len(p))) if k == 0 else set(),
              xmax=4.30)
        row_label(ax, 5.40, y, f"{s:.2f}", ha="right", bold=k == 0, color=ARMRED if k == 0 else INK)
    save(fig, "tok_unigram_nbest.pdf", log)


def all_splits(sp, s):
    """Every way to cut s into pieces of sp's vocabulary (unknown pieces not allowed)."""
    if not s:
        return [[]]
    return [[s[:e]] + rest for e in range(1, len(s) + 1) if sp.piece_to_id(s[:e]) != sp.unk_id()
            for rest in all_splits(sp, s[e:])]


def fig_unigram_samples(sp, R, log):
    """Subword regularization: sampled splits of one word, alpha = 0.1. SentencePiece's own
    sample_encode ignores set_random_generator_seed (checked 2026-10-27: a new split every run),
    so we sample the same distribution ourselves: every split, weighted by P(split)^alpha, seed 509."""
    word, alpha = "tokenization", 0.1
    splits = all_splits(sp, "▁" + word)
    w = np.exp(alpha * np.array([logp(sp, p) for p in splits]))
    picks = np.random.default_rng(509).choice(len(splits), size=5, p=w / w.sum())
    samples = [splits[i] for i in picks]
    best = sp.encode_as_pieces(word)
    R["t5_samples"] = {"word": word, "alpha": alpha, "seed": 509, "n_splits": len(splits),
                       "best": best, "samples": samples}
    assert best == ["▁token", "ization"] and best == max(splits, key=lambda p: logp(sp, p)), best
    assert len({tuple(s) for s in samples}) >= 4, samples       # the point: the splits differ
    rows = [("best split", best)] + [(f"sample {k + 1}", s) for k, s in enumerate(samples)]
    fig, ax = canvas(0.24 * len(rows) + 0.05)
    for k, (lab, p) in enumerate(rows):
        y = 0.24 * (len(rows) - k) - 0.10
        row_label(ax, 0.05, y, lab, bold=k == 0, color=INK if k == 0 else GREY)
        chips(fig, ax, 1.05, y, p, visible=False, h=0.19, fs=7.5)
    save(fig, "tok_unigram_samples.pdf", log)
    log.info(f"T5 samples of {word}: {samples}")


def fig_t5_armenian(sp, T, R, log):
    """T5's SentencePiece has no byte fallback: Armenian letters become <unk>."""
    line = ARM_LINE_FILE.read_text(encoding="utf-8").strip()
    ids, pieces = sp.encode(line), sp.encode_as_pieces(line)
    unk = [k for k, i in enumerate(ids) if i == sp.unk_id()]
    back = sp.decode(ids)
    R["t5_armenian"] = {"pieces": pieces, "unk_positions": unk, "decoded": back,
                        "gpt4o_roundtrip": T["gpt4o"].enc.decode(T["gpt4o"].enc.encode(line)) == line}
    assert (len(ids), len(unk)) == (QUOTED["t5_arm"]["tokens"], QUOTED["t5_arm"]["unk"]), (len(ids), len(unk))
    assert R["t5_armenian"]["gpt4o_roundtrip"]
    kept = {pieces[k] for k in range(len(pieces)) if k not in unk}
    assert kept == {"▁", ","}, kept          # the slide: only the spaces and the comma survive
    half = next(k for k in range(len(pieces) // 2, len(pieces)) if pieces[k] == "▁")
    fig, ax = canvas(1.30)
    row_label(ax, 0.02, 1.13, "T5 reads", bold=True)
    for r, (a, b) in enumerate(((0, half), (half, len(pieces)))):
        chips(fig, ax, 1.00, 1.13 - r * 0.30, pieces[a:b], visible=False, fs=7, h=0.20,
              hi={k - a for k in unk if a <= k < b}, gap=0.02)
    row_label(ax, 0.02, 0.32, "T5 writes back", bold=True)
    ax.text(1.00, 0.32, back.strip(), ha="left", va="center", fontsize=FS, family=MONO, color=ARMRED)
    save(fig, "tok_t5_armenian.pdf", log)


TRAIN_SURNAMES = r"""
import io, sys, sentencepiece as spm
names = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
buf = io.BytesIO()
spm.SentencePieceTrainer.train(sentence_iterator=iter(names), model_writer=buf, model_type="unigram",
                               vocab_size=int(sys.argv[3]), character_coverage=1.0, num_threads=1)
open(sys.argv[2], "wb").write(buf.getvalue())
"""


def surname_unigram(R, log, vocab=120):
    """Train SentencePiece Unigram on ch11's 689 surnames in a subprocess, so the C++ trainer's
    log (seed size, then the vocabulary size after every EM pass) can be read from its stderr."""
    with tempfile.TemporaryDirectory() as d:
        model = Path(d) / "surnames.model"
        run = subprocess.run([sys.executable, "-c", TRAIN_SURNAMES, str(SURNAMES_FILE), str(model), str(vocab)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        if run.returncode != 0:
            log.error(run.stderr[-2000:])
            raise RuntimeError("SentencePiece training failed")
        sp = spm.SentencePieceProcessor(model_file=str(model))
    seed = [int(m) for m in re.findall(r"Initialized (\d+) seed sentencepieces", run.stderr)]
    em = [(int(a), int(b)) for a, b in re.findall(r"EM sub_iter=(\d+) size=(\d+)", run.stderr)]
    assert len(seed) == 1 and em, run.stderr[-1500:]
    sizes = [seed[0]] + [s for it, s in em if it == 1] + [sp.get_piece_size()]
    pieces = sorted(((sp.get_score(i), sp.id_to_piece(i)) for i in range(3, sp.get_piece_size())), reverse=True)
    R["surname_unigram"] = {"vocab_size": sp.get_piece_size(), "seed": seed[0], "em_log": em,
                            "sizes_per_round": sizes, "top_pieces": [[p, round(s, 2)] for s, p in pieces[:12]]}
    log.info(f"surname Unigram: sizes {sizes}; top {pieces[:6]}")
    return sp, sizes, pieces


def fig_unigram_rounds(R, log):
    sp, sizes, pieces = surname_unigram(R, log)
    assert pieces[2][1] == "յան", pieces[:4]          # the slide: -yan is the 3rd most likely piece
    names = ["մանուկյան", "աբազյան", "աբալակով"]
    for n in names:
        assert n in NAMES, n
    splits = {n: sp.encode_as_pieces(n) for n in names}
    R["surname_unigram"]["splits"] = splits
    fig = plt.figure(figsize=(W, 1.65))
    ax = fig.add_axes([0.06, 0.20, 0.46, 0.62])
    labels = ["seed"] + [str(k) for k in range(1, len(sizes) - 1)] + ["final"]
    bars = ax.bar(labels, sizes, color=[ARMORANGE] + [ARMBLUE] * (len(sizes) - 2) + [ARMRED], width=0.65)
    ax.bar_label(bars, labels=[f"{s:,}" for s in sizes], fontsize=6.5, padding=2)
    ax.set_title("vocabulary size, round by round", fontsize=FS, pad=6)
    ax.set_xlabel("pruning round", fontsize=FS_ID, labelpad=2)
    ax.set_ylim(0, max(sizes) * 1.22)
    ax.set_yticks([])
    ax.tick_params(labelsize=FS_ID, length=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax2 = fig.add_axes([0.55, 0.0, 0.45, 1.0])
    ax2.set_xlim(0, W * 0.45)
    ax2.set_ylim(0, 1.65)
    ax2.axis("off")
    ax2.text(0.05, 1.42, "best splits after training", fontsize=FS, family=SANS, color=GREY, va="center")
    for k, n in enumerate(names):
        chips(fig, ax2, 0.05, 1.08 - k * 0.38, splits[n], visible=False, h=0.22, xmax=W * 0.45 - 0.02)
    save(fig, "tok_unigram_rounds.pdf", log)


def byte_facts(T, R, log):
    """'Why 256, and why that is enough' (added 2026-10-28): 2^8 values; UTF-8's 4-byte pattern
    carries 21 bits, more than Unicode's whole range; 13 byte values never occur in valid UTF-8,
    yet GPT-2's IDs 0-255 cover all 256."""
    used = set()
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:                 # surrogates are not characters; UTF-8 refuses them
            continue
        used.update(chr(cp).encode("utf-8"))
    unused = sorted(set(range(256)) - used)
    assert unused == [0xC0, 0xC1] + list(range(0xF5, 0x100)) and len(used) == 243, unused
    assert 2 ** 8 == 256 and 2 ** 21 == 2_097_152 and 0x10FFFF + 1 == 1_114_112
    assert [len(chr(cp).encode("utf-8")) for cp in (0x7F, 0x7FF, 0xFFFF, 0x10FFFF)] == [1, 2, 3, 4]
    gpt2_bytes = {T["gpt2"].enc.decode_single_token_bytes(i) for i in range(256)}
    assert gpt2_bytes == {bytes([b]) for b in range(256)}, "GPT-2's IDs 0-255 are not the 256 bytes"
    R["byte_facts"] = {"byte_values_used_by_utf8": len(used),
                       "never_in_valid_utf8": [f"{b:02X}" for b in unused]}
    log.info(f"UTF-8 uses {len(used)} of 256 byte values; never: {R['byte_facts']['never_in_valid_utf8']}")


def main():
    log = setup_logging()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    T = {"gpt2": Tik("gpt2"), "gpt4": Tik("cl100k_base"), "gpt4o": Tik("o200k_base")}
    T5 = t5_model()
    R = {"tiktoken": tiktoken.__version__, "sentencepiece": spm.__version__, "date": time.strftime("%Y-%m-%d")}
    jobs = [lambda: fig_question(T, R, log), lambda: fig_text_to_ids(T, R, log),
            lambda: fig_three_ways(T, R, log), lambda: fig_unk(T, R, log),
            lambda: fig_bytes(R, log), lambda: fig_bpe_encode(R, log),
            lambda: fig_regex(T, R, log), lambda: fig_decode_trap(T, R, log),
            lambda: fig_vocab_tradeoff(T, R, log), lambda: fig_wordpiece(R, log),
            lambda: fig_unigram(R, log), lambda: fig_spelled(T, R, log),
            lambda: fig_numbers(T, R, log), lambda: fig_code(T, R, log),
            lambda: fig_special(T, R, log), lambda: fig_armenian(T, R, log),
            lambda: fig_bpe_bytes(R, log), lambda: fig_id_ranks(T, R, log),
            lambda: wordpiece_toy(R, log), lambda: unigram_toy(R, log),
            lambda: fig_years(T, R, log), lambda: fig_commas(T, R, log),
            lambda: fig_reverse(T, R, log), lambda: fig_formats(T, R, log),
            lambda: fig_trailing(T, R, log), lambda: fig_glitch_split(T, R, log),
            lambda: fig_longest(T, R, log), lambda: fig_lottery(T, R, log),
            lambda: fig_unigram_nbest(T5, R, log), lambda: fig_unigram_samples(T5, R, log),
            lambda: fig_t5_armenian(T5, T, R, log), lambda: fig_unigram_rounds(R, log),
            lambda: byte_facts(T, R, log)]
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
