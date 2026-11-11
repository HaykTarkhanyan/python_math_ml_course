"""Assemble and execute llms/02_decoding/04_llm_api.ipynb - the first API session after LLM-2
(B0 in DECISIONS #105, #110): call real models through OpenRouter with the `openai` SDK and meet every
decoding knob from the lecture as a request field, plus what real work needs: usage and cost, providers,
streaming, prompt caching, locked knobs, reasoning tokens, errors.

Instructor-run solution notebook (the practical 02 pattern, DECISIONS #104). Structure rule (instructor,
2026-08-02): many small cells, a markdown cell before each code cell. Prose written after the run, so every
number it quotes is printed by a cell (_learnings/2026-08-13-2015_write-the-practical-after-measuring-not-before.md).

Needs OPENROUTER_API_KEY in the repository's .env (read with python-dotenv, never printed). 66 calls on cheap
models, $0.003 per run (measured 2026-11-06); every call goes to results/04_api_ledger.json. The prose quotes the
2026-11-06 run's outputs: after a re-execution, re-read them and fix the prose (sampled texts change).

Run with the project venv (builds, then executes; 142 s measured 2026-11-06):
    ./ma/Scripts/python.exe llms/02_decoding/py_src/build_api_nb.py
    ./ma/Scripts/python.exe llms/02_decoding/py_src/build_api_nb.py --markdown-only
Both write 04_llm_api.html next to the notebook (non_essential/notebook_to_html.py).
--markdown-only copies edited prose into the executed notebook without rerunning it (no API calls).
"""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
OUT = CH / "04_llm_api.ipynb"
LOGS_DIR = REPO_ROOT / "logs"
CELLS = []

log = logging.getLogger("build_api_nb")


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


# ======================================================================================
md(r"""
# Calling an LLM: every knob from the decoding lecture is a request field

The session after **LLM-2 (Decoding)**. We treat the model as magic: text goes in, text comes out. What happens
inside comes later (LLM-3 to LLM-6). Today is about what you control from the outside, and what it costs.

1. **The first call**: what a request and a reply look like.
2. **OpenRouter**: one key, many models, many providers.
3. **Tokens are money**: usage, cost, and why Armenian costs more.
4. **A conversation is resent every time.**
5. **The decoding knobs, live**: logprobs, temperature (extremes included), top-k / top-p / min-p, temperature 0,
   seeds, stop strings, length limits, structured output, sample-and-vote.
6. **Streaming.**
7. **Prompt caching**: pay less for a repeated beginning.
8. **Locked knobs and hidden reasoning tokens.**
9. **When things fail.**
10. **What it all cost**, and the field table.

**Running it:** about 2.5 minutes and 66 calls to cheap models, $0.003 in total (measured 2026-11-06). It needs
`OPENROUTER_API_KEY` in the repository's `.env` file. Every call is written to `results/04_api_ledger.json`.
""")

md(r"""
## 0. Setup

The key lives in `.env`, never in the notebook: a key pasted into a notebook ends up on GitHub sooner or later.
`python-dotenv` copies it into the environment, and the `openai` SDK takes it from there. **Never print it.**

OpenRouter speaks the same interface as OpenAI, so the official `openai` package works: only the `base_url`
changes. The same code can later point at a model running on your own machine.
""")

code(r"""
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import openai
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv("../../.env")                                   # the repository's .env: OPENROUTER_API_KEY=...
API = "https://openrouter.ai/api/v1"
client = OpenAI(base_url=API, api_key=os.environ["OPENROUTER_API_KEY"])     # KeyError = the key is missing
HEADERS = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}  # for the few plain HTTP calls

LLAMA = "meta-llama/llama-3.1-8b-instruct"     # an open model: every knob works (on the right provider)
MINI = "openai/gpt-4o-mini"                    # a cheap closed model with logprobs and automatic caching
HAIKU = "anthropic/claude-haiku-5.5"           # a 2026 closed model that takes no temperature at all
NANO = "openai/gpt-5-nano"                     # a small reasoning model: it thinks before it answers

RESULTS = Path("results")
LEDGER = RESULTS / "04_api_ledger.json"
pd.set_option("display.max_colwidth", 120)
""")

md(r"""
One helper for every call: it sends the request and writes one line per call to the ledger (model, provider,
tokens, cost, why it stopped). Keeping such a ledger is a habit worth having in any project that pays per call.
""")

code(r"""
ledger = []


def record(tag, model, d, secs):
    u = d.get("usage") or {}
    ptd = u.get("prompt_tokens_details") or {}
    ledger.append({"tag": tag, "model": model, "provider": d.get("provider"),
                   "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                   "cached_tokens": ptd.get("cached_tokens"), "cache_write_tokens": ptd.get("cache_write_tokens"),
                   "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   "cost": u.get("cost"), "finish": d["choices"][0].get("finish_reason") if d.get("choices") else None,
                   "secs": round(secs, 2)})
    LEDGER.write_text(json.dumps(ledger, indent=1, ensure_ascii=False), encoding="utf-8")


def chat(tag, **kw):
    t0 = time.time()
    r = client.chat.completions.create(**kw)
    record(tag, kw["model"], r.model_dump(), time.time() - t0)
    return r


def text(r):
    return r.choices[0].message.content


def credits_used():
    # the account's running total in dollars; the difference before and after is what this notebook cost
    reply = requests.get(f"{API}/credits", headers=HEADERS, timeout=60)
    reply.raise_for_status()
    return reply.json()["data"]["total_usage"]


spent_before = credits_used()
""")

# ======================================================================================
md(r"""
## 1. The first call

Under every SDK, a call is one HTTP request: a POST with a JSON body. Here it is with plain `requests`, so
nothing is hidden.
""")

code(r"""
body = {"model": MINI, "max_tokens": 60,
        "messages": [{"role": "user", "content": "In one sentence: what is a token in a language model?"}]}
t0 = time.time()
raw = requests.post(f"{API}/chat/completions", headers=HEADERS, json=body, timeout=60)
raw.raise_for_status()
reply = raw.json()
record("first_raw", MINI, reply, time.time() - t0)
print(json.dumps(reply, indent=1, ensure_ascii=False))
""")

md(r"""
The parts that matter:

- `choices[0].message.content`: the text.
- `choices[0].finish_reason`: why it stopped. `"stop"` = the model wrote its end-of-turn token (or hit a stop
  string); `"length"` = it ran into `max_tokens`. LLM-2, "When to stop".
- `usage`: tokens in (`prompt_tokens`), tokens out (`completion_tokens`), and OpenRouter adds `cost` in dollars.
- `provider`: which company actually ran the model (section 2).

The same call through the SDK: shorter, typed, with retries built in.
""")

code(r"""
r = chat("first_sdk", model=MINI, max_tokens=60,
         messages=[{"role": "user", "content": "In one sentence: what is a token in a language model?"}])
print(text(r))
print("finish_reason:", r.choices[0].finish_reason)
print(json.dumps(r.model_dump()["usage"], indent=1))
""")

md(r"""
OpenRouter also keeps statistics for every call. Ask for them with the reply's `id`: latency, the provider, and the
cost. They appear only some seconds after the call (9 s and more than 12 s in two tries on 2026-11-06), so the
helper waits up to a minute and then gives up loudly.
""")

code(r"""
# ~10-20 s: waiting for the stats to appear
def generation_stats(gen_id):
    for attempt in range(20):                    # bounded wait: 20 x 3 s
        reply = requests.get(f"{API}/generation", params={"id": gen_id}, headers=HEADERS, timeout=60)
        if reply.status_code == 200:
            return reply.json()["data"]
        time.sleep(3)
    reply.raise_for_status()
    raise RuntimeError(f"no stats for {gen_id}: HTTP {reply.status_code}")


g = generation_stats(r.id)
{k: g[k] for k in ("model", "provider_name", "latency", "generation_time", "tokens_prompt", "tokens_completion",
                   "total_cost")}
""")

# ======================================================================================
md(r"""
## 2. OpenRouter: one key, many models, many providers

**What it is.** A single API in front of hundreds of models from many companies. You buy credits once; OpenRouter
passes through each provider's per-token price without markup and charges a fee when you buy credits
(its FAQ, read 2026-11-06).

**Model names** look like `company/model`: `openai/gpt-4o-mini`, `meta-llama/llama-3.1-8b-instruct`. Names ending
in `:free` are free and heavily rate-limited; `:batch` ones are cheaper and slower.

**Providers.** An open model like Llama is run by several companies at once. OpenRouter picks one per request
(mostly by price and uptime) and falls back to another if it fails. You can steer that with the `provider`
field: an `order` to try, `allow_fallbacks`, `require_parameters`, `sort` by price or speed, data-policy filters.

The price list is public:
""")

code(r"""
models = {m["id"]: m for m in requests.get(f"{API}/models", timeout=60).json()["data"]}
print(f"{len(models)} models served today")
rows = []
for mid in (LLAMA, MINI, HAIKU, NANO):
    m, p = models[mid], models[mid]["pricing"]
    sp = set(m["supported_parameters"])
    rows.append({"model": mid, "$ / 1M in": float(p["prompt"]) * 1e6, "$ / 1M out": float(p["completion"]) * 1e6,
                 "$ / 1M cached in": float(p["input_cache_read"]) * 1e6 if p.get("input_cache_read") else None,
                 "context": m["context_length"], "temperature": "temperature" in sp, "top_k": "top_k" in sp,
                 "logprobs": "logprobs" in sp})
prices = pd.DataFrame(rows).set_index("model")
prices
""")

md(r"""
**Predict:** "Llama 3.1 8B" is one model. Will every company that serves it give the same answer, at the same price,
with the same knobs?
""")

code(r"""
eps = requests.get(f"{API}/models/{LLAMA}/endpoints", timeout=60).json()["data"]["endpoints"]
KNOBS = ["temperature", "top_k", "min_p", "seed", "logprobs"]
prov = pd.DataFrame([{"provider": e["provider_name"], "slug": e["tag"].split("/")[0], "weights": e["quantization"],
                      "$ / 1M in": float(e["pricing"]["prompt"]) * 1e6,
                      "$ / 1M out": float(e["pricing"]["completion"]) * 1e6,
                      **{k: k in (e["supported_parameters"] or []) for k in KNOBS}} for e in eps])
prov
""")

md(r"""
Same model name, different companies:

- **Different weights.** `fp8` stores every number in 8 bits, `bf16` in 16: a slightly different model
  (quantization is LLM-12).
- **Different knobs.** Not every provider supports `top_k`, `min_p` or `logprobs`. A knob a provider does not
  support is **dropped silently** unless you ask for `require_parameters`.
- **Different prices.**

So for any experiment that compares answers (temperature 0, seeds, logprobs), **pin one provider and turn
fallbacks off**. Otherwise two answers can differ only because two different companies produced them. Below: the
cheapest provider that has the knobs we need.
""")

code(r"""
def cheapest(need):
    ok = prov[prov[need].all(axis=1)].sort_values("$ / 1M out")
    if ok.empty:
        raise RuntimeError(f"no provider of {LLAMA} supports all of {need}")
    return ok.iloc[0]["slug"]


def pinned(slug):
    return {"provider": {"order": [slug], "allow_fallbacks": False}}


PIN_KNOBS = cheapest(["temperature", "top_k", "min_p", "seed"])
PIN_LOGPROBS = cheapest(["logprobs"])
print("knob experiments on:", PIN_KNOBS, "| logprob experiments on:", PIN_LOGPROBS)
""")

# ======================================================================================
md(r"""
## 3. Tokens are money

You pay per token, in and out (LLM-1). The same sentence in English and in Armenian, through two tokenizers. A chat
request also wraps your text in a few template tokens (roles, start and end markers), so we measure that overhead
with a one-character message and subtract it.
""")

code(r"""
EN = "The weather in Yerevan is sunny today, and many people are walking in the park."
HY = "Այսօր Երևանում եղանակը արևոտ է, և շատ մարդիկ զբոսնում են այգում։"


def prompt_tokens(model, s, tag):
    r = chat(tag, model=model, messages=[{"role": "user", "content": s}], max_tokens=1)
    return r.usage.prompt_tokens


rows = []
for model in (MINI, LLAMA):
    overhead = prompt_tokens(model, ".", "tok_base") - 1          # "." itself is one token
    en, hy = (prompt_tokens(model, s, "tok_" + n) - overhead for n, s in (("en", EN), ("hy", HY)))
    rows.append({"model": model, "English tokens": en, "Armenian tokens": hy, "Armenian / English": round(hy / en, 2)})
pd.DataFrame(rows).set_index("model")
""")

md(r"""
The same sentence costs GPT-4o's tokenizer 1.16 times the English tokens, and Llama 3.1's tokenizer **5.5 times**.

**Remember:** the bill is counted in **tokens**, and a tokenizer that was not built for Armenian makes every
Armenian request more expensive and eats the context window faster (practical 02 measured this per meaning).
""")

# ======================================================================================
md(r"""
## 4. A conversation is resent every time

**Predict:** in the third turn we ask "What is my name?". How does the model know?
""")

code(r"""
messages = [{"role": "system", "content": "You are a terse assistant. Answer in one sentence."}]
for q in ["My name is Ani. What is greedy decoding?", "And beam search?", "What is my name?"]:
    messages.append({"role": "user", "content": q})
    r = chat("conversation", model=MINI, messages=messages, max_tokens=60, temperature=0)
    messages.append({"role": "assistant", "content": text(r)})
    print(f"prompt_tokens = {r.usage.prompt_tokens:>3} | {q} -> {text(r)}")
""")

md(r"""
The API remembers nothing between calls. **We** keep the history and send all of it every time, which is why
`prompt_tokens` grows each turn: a long chat pays for its whole history again on every message. (Prompt caching,
section 7, makes the repeated part cheaper.)
""")

# ======================================================================================
md(r"""
## 5. The decoding knobs, live

### 5a. Logprobs: the distribution, from a real API

LLM-2 started from GPT-2's distribution after "The cat sat on the": "floor" 7.6%, the top five together under
30%. **Predict:** what does a 2024 chat model put on its favourite word?
""")

code(r"""
q = [{"role": "user", "content": "Continue the text with exactly one word, no punctuation: The cat sat on the"}]
for model, extra in ((LLAMA, pinned(PIN_LOGPROBS)), (MINI, {})):
    r = chat("logprobs", model=model, messages=q, max_tokens=1, temperature=0, logprobs=True, top_logprobs=5,
             extra_body=extra)
    top = r.choices[0].logprobs.content[0].top_logprobs
    print(f"{model:<34}", "  ".join(f"{t.token!r} {math.exp(t.logprob):.1%}" for t in top))
""")

md(r"""
96.1% (Llama) and 99.9% (GPT-4o mini) on "mat", where GPT-2, a base model, gave its favourite "floor" 7.6%. Chat
models are far more sure of themselves than base models: post-training does that (LLM-9). Keep it in mind when you
read their logprobs as confidence.

Logprobs per token of an answer are a rough confidence signal (LLM-2, "Why log-probabilities"): a well-known fact
against a less-known one.
""")

code(r"""
for qq in ["In what year did Armenia adopt Christianity as its state religion? Answer with the year only.",
           "In what year was the first Armenian-language newspaper published? Answer with the year only."]:
    r = chat("confidence", model=MINI, messages=[{"role": "user", "content": qq}], max_tokens=6, temperature=0,
             logprobs=True)
    toks = r.choices[0].logprobs.content
    print(qq, "->", text(r), "|", "  ".join(f"{t.token!r} {math.exp(t.logprob):.0%}" for t in toks))
""")

md(r"""
Both answers are right (301; 1794, the year Azdarar appeared in Madras) and both come with high probabilities.
Notice that "1794" arrived as two tokens, "179" and "4" (LLM-1). A low probability on an answer token is the
warning sign to look for.

**Ask the 2026 models.** One Armenian line, six recent models, one answer each. Who do they say is grazing?
(Reasoning effort "low" keeps the thinking short. DeepSeek ignored "low" and thought until all 1,000 tokens were
gone, with no answer, when tried on 2026-11-09, so its thinking is switched off.)
""")

code(r"""
# ~50 s, ~$0.015 (measured 2026-11-09)
line = "Դաշտերում, դաշտերում, գիտեք ով է արածում"
NEWER = {"openai/gpt-6.1-sol": {"reasoning": {"effort": "low"}},
         "anthropic/claude-sonnet-5.5": {"reasoning": {"effort": "low"}},
         "google/gemini-4-argon": {"reasoning": {"effort": "low"}},
         "x-ai/grok-4.7": {"reasoning": {"effort": "low"}},
         "deepseek/deepseek-v4.1-flash": {"reasoning": {"enabled": False}},
         "qwen/qwen3.8-max-0902": {"reasoning": {"effort": "low"}}}
for model, extra in NEWER.items():
    r = chat("armenian_newer", model=model, messages=[{"role": "user", "content": line}], max_tokens=1000,
             extra_body=extra)
    print(f"--- {model} (finish: {r.choices[0].finish_reason})\n{text(r)}\n")
""")

md(r"""
### 5b. Temperature, extremes included

One open model, one provider, the same prompt, temperature from 0 to 2 (`top_p = 1`, so nothing cuts the tail).
**Predict:** what does temperature 2 look like?
""")

code(r"""
# ~10 s
cat = [{"role": "user", "content": "Write one sentence about a cat."}]
temps = {}
for T in (0, 0.7, 1.0, 1.5, 2.0):
    r = chat(f"temp_{T}", model=LLAMA, messages=cat, max_tokens=40, temperature=T, top_p=1,
             extra_body=pinned(PIN_KNOBS))
    temps[T] = text(r)
    print(f"T = {T}: {text(r)!r}\n")
""")

md(r"""
Up to 1.0 the sentences are fine. At **1.5** the text already falls apart into fragments of code, other languages
and random words, and 2.0 is worse. Llama 3's vocabulary has about 128,000 tokens, and their long tail takes over
(LLM-2, "By hand: temperature and the tail").

### 5c. Cutting the tail at temperature 2

LLM-2: a high temperature hands the dice to the long tail, which is why it is always combined with a cut-off.
The same temperature 2, now with top-k, top-p or min-p:
""")

code(r"""
# ~10 s
cuts = {}
for name, knob in (("top_k = 5", {"top_k": 5}), ("top_p = 0.9", {"top_p": 0.9}), ("min_p = 0.1", {"min_p": 0.1})):
    extra = pinned(PIN_KNOBS) | {k: v for k, v in knob.items() if k != "top_p"}
    r = chat(f"cut_{name}", model=LLAMA, messages=cat, max_tokens=40, temperature=2.0,
             top_p=knob.get("top_p", 1), extra_body=extra)
    cuts[name] = text(r)
    print(f"T = 2, {name}: {text(r)!r}\n")
""")

md(r"""
**top-k = 5** and **min-p = 0.1** bring back a normal sentence; **top-p = 0.9** does not. At temperature 2 the
distribution is so flat that the 90% nucleus still holds thousands of tail tokens, while top-k keeps 5 and min-p
keeps only tokens with at least a tenth of the favourite's probability. The result fits a provider that applies the
temperature **before** the cut-off, the order that LLM-2's "The order of the knobs" warns about.

### 5d. Temperature 0, and the provider question

**Predict:** temperature 0, the same prompt, 8 times. How many different answers? First without pinning a provider,
then pinned to one.
""")

code(r"""
# ~30 s
runs = []
for setting, extra in (("routed by OpenRouter", {}), (f"pinned to {PIN_KNOBS}", pinned(PIN_KNOBS))):
    for k in range(8):
        r = chat("temp0_" + setting, model=LLAMA, messages=cat, max_tokens=40, temperature=0, extra_body=extra)
        runs.append({"setting": setting, "provider": r.model_dump().get("provider"), "text": text(r)})
runs = pd.DataFrame(runs)
print(runs.groupby("setting").agg(calls=("text", "size"), different_answers=("text", "nunique"),
                                  providers=("provider", lambda s: dict(Counter(s)))).to_string())
""")

code(r"""
runs.drop_duplicates(["setting", "provider", "text"])
""")

md(r"""
Two different effects can make temperature 0 disagree with itself:

- **Different providers** (other hardware, `fp8` vs `bf16` weights): a different calculation altogether.
- **The same provider**: the batching effect from LLM-2 ("Temperature 0 is not deterministic"): your request is
  computed together with other people's, the batch size changes the order of additions, and a near-tie flips.

In this run OpenRouter sent all 8 routed calls to DeepInfra, the cheapest provider, so both rows show the
**same-provider** effect: 2 different answers in 8 calls, at temperature 0, from one company. Earlier the same day,
routed calls split between Cloudflare and DeepInfra and the answers split with them. Without the `providers`
column you could not tell the two effects apart, which is why experiments pin a provider.
""")

md(r"""
### 5e. Seeds

A seed fixes the dice. Temperature 1, pinned provider: seed 509 twice, seed 510 once.
""")

code(r"""
seeded = []
for seed in (509, 509, 510):
    r = chat(f"seed_{seed}", model=LLAMA, messages=cat, max_tokens=40, temperature=1.0, seed=seed,
             extra_body=pinned(PIN_KNOBS))
    seeded.append(text(r))
    print(f"seed {seed}: {text(r)!r}")
print("\nseed 509 twice gave the same text:", seeded[0] == seeded[1])
""")

md(r"""
Same seed, same provider: the same sentence word for word; seed 510: a different one. (On a different provider,
or after the provider updates its software, seed 509 can give another sentence.)

### 5f. Stop strings: why the end token is not enough

LLM-2, "When to stop". An agent loop (the agents chapter, L44) asks the model to write an action; **your program**
then runs the tool and pastes in the real result. Without a stop string the model happily writes the tool's
result itself.
""")

code(r"""
REACT = ("Answer the question. You can use one tool: search[query], which returns a fact.\n"
         "Use exactly this format:\nThought: ...\nAction: search[...]\nObservation: (the tool's result)\n"
         "... (repeat if needed)\nAnswer: ...")
qq = [{"role": "system", "content": REACT}, {"role": "user", "content": "How many people live in Gyumri?"}]
for stop in (None, ["Observation:"]):
    r = chat(f"react_stop_{stop}", model=LLAMA, messages=qq, max_tokens=150, temperature=0, stop=stop,
             extra_body=pinned(PIN_KNOBS))
    print(f"stop = {stop}  (finish_reason: {r.choices[0].finish_reason})\n{text(r)}\n")
""")

md(r"""
Without the stop string, Llama wrote the `Observation:` line itself: "approximately 107,000 people", a number no
search produced. With `stop=["Observation:"]` the reply ends after the `Action:` line, the program runs the real
search and pastes the result in. Both replies say `finish_reason: stop`: the API reports a stop string as a stop.
(The two `Thought:` lines differ although both ran at temperature 0: 5d again.)

### 5g. The length limit, and structured output

`max_tokens` is a safety net and a cost cap, not a way to end text: it cuts wherever it is.
""")

code(r"""
r = chat("length_cut", model=MINI, messages=[{"role": "user", "content": "Explain beam search."}], max_tokens=12)
print(repr(text(r)), "| finish_reason:", r.choices[0].finish_reason)
""")

md(r"""
Structured output: give a JSON schema, and the provider masks every token that would break it (LLM-2, "Constrain
the choice").
""")

code(r"""
CITY = {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["city", "founded", "famous_for"],
    "properties": {"city": {"type": "string"}, "founded": {"type": "integer"},
                   "famous_for": {"type": "array", "items": {"type": "string"}}}}}}
cq = [{"role": "user", "content": "Give facts about Yerevan."}]
r = chat("json_ok", model=MINI, messages=cq, response_format=CITY, max_tokens=200, temperature=0)
facts = json.loads(text(r))
facts
""")

md(r"""
**Predict:** the same request with `max_tokens = 15`?
""")

code(r"""
r = chat("json_cut", model=MINI, messages=cq, response_format=CITY, max_tokens=15, temperature=0)
print(repr(text(r)), "| finish_reason:", r.choices[0].finish_reason)
try:
    json.loads(text(r))
except json.JSONDecodeError as e:
    print("json.loads fails:", e)
""")

md(r"""
**Remember:** constrained decoding guarantees the **shape** while the model is writing. It cannot guarantee the
model finishes. Always check `finish_reason` before you parse.
""")

md(r"""
### 5h. Sample many, then vote

LLM-2, "self-consistency": sample several answers at a temperature above 0 and take the most common final answer.
""")

code(r"""
# ~15 s
PROBLEM = ("A shop sells pens at 3 for 2 dollars. Ani buys 18 pens and pays with a 20-dollar note. "
           "How much change does she get? Think briefly, then end with 'Answer: <number>'.")


def final_answer(s):
    tail = s.rsplit("Answer:", 1)[-1] if "Answer:" in s else ""
    digits = "".join(ch for ch in tail if ch.isdigit() or ch == ".").strip(".")
    return digits or None


greedy = final_answer(text(chat("vote_greedy", model=LLAMA, messages=[{"role": "user", "content": PROBLEM}],
                                max_tokens=200, temperature=0, extra_body=pinned(PIN_KNOBS))))
votes = [final_answer(text(chat("vote_sample", model=LLAMA, messages=[{"role": "user", "content": PROBLEM}],
                                max_tokens=200, temperature=0.8, extra_body=pinned(PIN_KNOBS)))) for _ in range(7)]
print("greedy:", greedy, "| 7 samples:", votes, "| majority:", Counter(votes).most_common(1)[0], "| correct: 8")
""")

# ======================================================================================
md(r"""
Here greedy was already right, and 6 of the 7 samples agree with it (one never wrote `Answer:`). Voting pays off on
problems where the single greedy answer is often wrong: PaLM 540B on GSM8K went from 56.5% to 74.4% (LLM-2).

## 6. Streaming

Chat apps print the answer as it is written: `stream=True` sends each piece the moment it exists. What users feel
is the **time to the first token**, not the total.
""")

code(r"""
t0, first, pieces, usage = time.time(), None, [], None
stream = client.chat.completions.create(model=MINI, max_tokens=80, stream=True, stream_options={"include_usage": True},
                                        messages=[{"role": "user", "content": "Count from 1 to 15 in words."}])
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        first = first or time.time() - t0
        pieces.append(chunk.choices[0].delta.content)
    if chunk.usage:
        usage = chunk.model_dump()["usage"]
record("stream", MINI, {"usage": usage, "choices": [{"finish_reason": None}]}, time.time() - t0)
print("|".join(pieces))
print(f"\nfirst piece after {first:.2f} s, all {len(pieces)} pieces after {time.time() - t0:.2f} s")
""")

# ======================================================================================
md(r"""
The `|` marks show the pieces as they arrived: tokens. "Th|irteen", "Four|teen", "Fif|teen" (LLM-1). The first piece
came after 0.93 s, the whole answer after 2.08 s: a streaming app feels twice as fast.

## 7. Prompt caching: pay less for a repeated beginning

Before the model writes anything it reads the whole prompt, and the provider keeps that work for a few minutes
(what exactly it keeps is the KV cache, LLM-14). A new request that **starts with exactly the same tokens** reuses
it: cheaper, and a faster first token. The rules:

- **Exact prefix**, token for token, from the very first one. Change one character at the top and nothing matches.
- **A minimum length**: 1,024 tokens on OpenAI models.
- **A short life**: minutes.
- **Automatic** on OpenAI, DeepSeek and Gemini 2.5+; **explicit** on Anthropic: you mark the end of the reusable
  part with `cache_control`, and writing to the cache costs 1.25x, reading it 0.1x (OpenRouter's caching guide,
  read 2026-11-06).
- OpenRouter routes the next request to the provider that holds your cache ("sticky routing", for 10 minutes).

**Design rule:** fixed things first (instructions, documents, examples), the changing question last.

Our long prompt: the first 25 slides of the decoding lecture, as notes for a slide-answering assistant.
""")

code(r"""
notes = "\n".join(p.extract_text() for p in PdfReader("03_decoding.pdf").pages[:25])
SYSTEM = "You answer questions about these lecture slides in one sentence.\n\n" + notes
cache_rows = []
for k, question in enumerate(["What is greedy decoding?", "What does top-p keep?", "What is a stop string?"]):
    r = chat("cache_mini", model=MINI, max_tokens=60, temperature=0,
             messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}])
    u = r.model_dump()["usage"]
    cache_rows.append({"request": f"{k + 1}: {question}", "prompt_tokens": u["prompt_tokens"],
                       "cached_tokens": u["prompt_tokens_details"]["cached_tokens"], "cost $": u["cost"]})
    time.sleep(2)
r = chat("cache_mini_miss", model=MINI, max_tokens=60, temperature=0,
         messages=[{"role": "system", "content": "Today is 7 November 2026. " + SYSTEM},
                   {"role": "user", "content": "What is greedy decoding?"}])
u = r.model_dump()["usage"]
cache_rows.append({"request": "4: one sentence added at the top", "prompt_tokens": u["prompt_tokens"],
                   "cached_tokens": u["prompt_tokens_details"]["cached_tokens"], "cost $": u["cost"]})
pd.DataFrame(cache_rows)
""")

md(r"""
Request 1 paid full price. Requests 2 and 3 read 3,200 of their 3,310 tokens from the cache and cost **53%** as much
(GPT-4o mini bills cached input at half price; GPT-5 nano and Claude Haiku 5.5 at a tenth, see the price table in
section 2). Request 4 added one sentence at the
top: 0 cached tokens, full price again.

The same with Claude and an explicit breakpoint: the first request **writes** the cache (at 1.25x), the second
**reads** it (at 0.1x).
""")

code(r"""
blocks = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
haiku_rows = []
for question in ["What is greedy decoding?", "What does top-p keep?"]:
    r = chat("cache_haiku", model=HAIKU, max_tokens=300,
             messages=[{"role": "system", "content": blocks}, {"role": "user", "content": question}])
    u = r.model_dump()["usage"]
    haiku_rows.append({"question": question, "prompt_tokens": u["prompt_tokens"],
                       "cache_write_tokens": u["prompt_tokens_details"].get("cache_write_tokens"),
                       "cached_tokens": u["prompt_tokens_details"]["cached_tokens"], "cost $": u["cost"]})
    time.sleep(2)
haiku_cache = pd.DataFrame(haiku_rows)
print(f"second request / first request cost: {haiku_cache['cost $'][1] / haiku_cache['cost $'][0]:.2f}")
haiku_cache
""")

# ======================================================================================
md(r"""
The second Claude request cost **0.30** of the first: its 4,485 prompt tokens were billed at a tenth, and only the
answer tokens kept their full price.

## 8. Locked knobs and hidden reasoning tokens

### 8a. A knob that is silently ignored

`claude-haiku-5.5` does not take a temperature (its `supported_parameters` in section 2 has none). **Predict:**
what happens when we send temperature 2 anyway? Compare with Llama's temperature 2 in 5b.
""")

code(r"""
for T in (0, 2.0):
    r = chat(f"haiku_T{T}", model=HAIKU, messages=cat, max_tokens=60, temperature=T)
    print(f"Haiku 5.5, T = {T}: {text(r)!r}")
""")

md(r"""
No error, and temperature 2 reads like normal text: OpenRouter dropped the knob. To make it fail loudly instead,
ask for `require_parameters`:
""")

code(r"""
try:
    chat("haiku_required", model=HAIKU, messages=cat, max_tokens=60, temperature=2.0,
         extra_body={"provider": {"require_parameters": True}})
except openai.NotFoundError as e:
    print("refused:", e.body["message"] if isinstance(e.body, dict) else e)
""")

md(r"""
**Remember:** "the request succeeded" does not mean "my settings were used". Many 2025-26 frontier models fix
their sampling; you steer them with the prompt, and with reasoning effort.

### 8b. Reasoning tokens: paying for thinking you do not see

A reasoning model writes hidden "thinking" tokens before its answer. They are billed as output tokens.
""")

code(r"""
# ~20 s
PUZZLE = "Ani is older than Gor. Gor is older than Lilit. Who is the youngest? One word."
rows = []
for effort in ("low", "high"):
    r = chat(f"nano_{effort}", model=NANO, messages=[{"role": "user", "content": PUZZLE}], max_completion_tokens=4000,
             extra_body={"reasoning": {"effort": effort}})
    u = r.model_dump()["usage"]
    rows.append({"effort": effort, "answer": text(r), "completion_tokens": u["completion_tokens"],
                 "of which reasoning": u["completion_tokens_details"]["reasoning_tokens"], "cost $": u["cost"]})
pd.DataFrame(rows)
""")

# ======================================================================================
md(r"""
Both efforts answered "Lilit". At low effort 64 of the 110 output tokens were hidden reasoning; at high effort 256 of
300, for 2.7 times the cost. You pay for every one of them, and you never see them.

## 9. When things fail

Errors you will meet, and what they mean:

| code | meaning | what to do |
|---|---|---|
| 400 | a bad request, for example a misspelled model name | read the message |
| 401 | bad or missing key | check `.env` |
| 402 | out of credits | top up, or a cheaper model |
| 404 | no provider supports your parameters (`require_parameters`) | drop a knob, or pick another model |
| 429 | rate limit | wait and retry (the SDK retries twice by default: `OpenAI(max_retries=...)`) |
| 5xx | the provider failed | retry; OpenRouter falls back to another provider unless you pinned one |

Catch the errors you expect, and let everything else crash loudly. A misspelled model name:
""")

code(r"""
try:
    chat("bad_model", model="openai/gpt-4o-mni", messages=cat, max_tokens=10)
except openai.APIStatusError as e:
    print(type(e).__name__, e.status_code, "-", e.body["message"] if isinstance(e.body, dict) else e)
""")

# ======================================================================================
md(r"""
## 10. What it all cost

Every call from the ledger, by model, and the account's total before and after the notebook.
""")

code(r"""
time.sleep(10)                                     # the account total lags the calls by a few seconds
spent = credits_used() - spent_before
calls = pd.DataFrame(ledger)
summary = calls.groupby("model").agg(calls=("tag", "size"), tokens_in=("prompt_tokens", "sum"),
                                     tokens_out=("completion_tokens", "sum"), cost=("cost", "sum"))
print(f"{len(calls)} calls; the ledger says ${calls['cost'].sum():.5f}; the account total moved by ${spent:.5f}")
summary
""")

md(r"""
The account total trails the newest calls even after the 10-second wait ($0.00117 against the ledger's $0.00294 in
this run). The ledger, built from every reply's own `usage.cost`, is the number to trust; billing dashboards lag.

### The field table

| request field | what it does | LLM-2 frame |
|---|---|---|
| `temperature` | divides the logits by T | Temperature; By hand: temperature |
| `top_p`, `top_k`, `min_p` | cut the tail before the dice roll | Top-k; Top-p; Min-p |
| `frequency_penalty`, `presence_penalty`, `repetition_penalty` | discourage repeats | Repetition penalties (and "seen") |
| `seed` | fixes the dice (same provider, same software) | Temperature 0 is not deterministic |
| `max_tokens`, `stop`, and `finish_reason` in the reply | when to stop | When to stop; Why stop strings |
| `logprobs`, `top_logprobs` | the distribution itself | Why we work with log-probabilities |
| `response_format` (JSON schema) | masks illegal tokens | Masking; What it buys |
| `stream` | send pieces as they are written | - |
| `reasoning` effort | hidden thinking, billed as output | not in LLM-2 (reasoning models, later) |
| `provider` (OpenRouter) | which company runs it; `require_parameters` | this notebook, section 2 |

**Next:** we open the box. LLM-3 to LLM-6: where the probabilities come from.
""")


# ======================================================================================
def export_html():
    subprocess.run([sys.executable, str(REPO_ROOT / "non_essential" / "notebook_to_html.py"), str(OUT)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown-only", action="store_true",
                    help="copy the markdown cells into the executed notebook; code and outputs untouched")
    args = ap.parse_args()
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "build_api_nb.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)

    if args.markdown_only:
        old = nbf.read(OUT, as_version=4)
        if [c.cell_type for c in old.cells] != [c.cell_type for c in CELLS] or any(
                a.source != b.source for a, b in zip(old.cells, CELLS) if a.cell_type == "code"):
            raise SystemExit("the code cells changed since the last run: rebuild and execute instead")
        for a, b in zip(old.cells, CELLS):
            if a.cell_type == "markdown":
                a.source = b.source
        nbf.write(old, OUT)
        log.info(f"markdown refreshed in {OUT}")
        export_html()
        return

    kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = kernelspec
    nb.metadata["language_info"] = {"name": "python"}
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    t0 = time.perf_counter()
    try:
        NotebookClient(nb, timeout=600, kernel_name="ma", startup_timeout=300,
                       resources={"metadata": {"path": str(CH)}}).execute()
    except CellExecutionError:
        nbf.write(nb, OUT)                            # keep the outputs so far for debugging, then fail
        log.error(f"a cell failed after {time.perf_counter() - t0:.0f} s; partial notebook in {OUT}")
        raise
    nb.metadata["kernelspec"] = kernelspec
    log.info(f"executed in {time.perf_counter() - t0:.0f} s")
    nbf.write(nb, OUT)
    log.info(f"wrote {OUT} ({len(CELLS)} cells)")
    export_html()


if __name__ == "__main__":
    main()
