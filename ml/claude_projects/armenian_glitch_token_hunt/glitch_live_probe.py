"""Live glitch-token probe: ask served models to repeat / spell Armenian words.

The classic glitch-token symptom (SolidGoldMagikarp) is a model that CANNOT
repeat a string back, because the token's embedding is undertrained. We probe
four OpenRouter-served models whose tokenizers the notebook analyzed, with the
XLM-R glitch candidates, controls, and long single-token words.

Paid API: every call is logged to a ledger with token usage and cost, saved in
out/live_probe_results.json. Run `--smoke` first (one call).

Usage:  python glitch_live_probe.py --smoke   # one call, verify shape
        python glitch_live_probe.py           # full run (~104 calls, ~$0.02)
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("logs/glitch_live_probe.py.log", encoding="utf-8")],
)
log = logging.getLogger("glitch_live_probe")

API = "https://openrouter.ai/api/v1"

# our tokenizer-registry name -> the served model that uses that vocab
MODELS = {
    "o200k":   "openai/gpt-4o-mini",
    "llama3":  "meta-llama/llama-3.1-8b-instruct",
    "qwen2.5": "qwen/qwen-2.5-7b-instruct",
    "gemma2":  "google/gemma-2-27b-it",   # 9b-it no longer served; same tokenizer
}

WORDS = [
    ("Երևան", "control"), ("Հայաստան", "control"), ("բարև", "control"),
    ("խմբագրել", "glitch"), ("Զրուցարան", "glitch"), ("ԱՐՄԵՆՊՐԵՍ", "glitch"),
    ("Հիշեցնենք", "glitch"), ("Մեկնաբանություն", "glitch"),
    ("Փաշինյանն", "glitch"), ("Տպել", "glitch"), ("֍", "glitch"),
    ("շնորհակալություն", "long"), ("հնարավորություն", "long"),
]

TASKS = {
    "repeat": "Repeat the following string exactly, output nothing else: {w}",
    "spell": ("Spell the following Armenian word letter by letter, "
              "separated by spaces, output only the letters: {w}"),
}

ARM_CHAR = re.compile(r"[԰-֏ﬓ-ﬗ]")


def load_key() -> str:
    env = Path(__file__).resolve().parents[3] / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"OPENROUTER_API_KEY not found in {env}")


def fetch_pricing(session: requests.Session, models: dict | None = None) -> dict:
    r = session.get(f"{API}/models", timeout=30)
    r.raise_for_status()
    by_id = {m["id"]: m["pricing"] for m in r.json()["data"]}
    out = {}
    for name, mid in (models or MODELS).items():
        if mid not in by_id:
            raise RuntimeError(f"model {mid} not on OpenRouter - update MODELS")
        out[mid] = {"prompt": float(by_id[mid]["prompt"]),
                    "completion": float(by_id[mid]["completion"])}
    return out


def call(session: requests.Session, model: str, prompt: str) -> dict:
    for attempt in (1, 2):
        r = session.post(f"{API}/chat/completions", timeout=60, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 150,
        })
        if r.status_code == 200:
            return r.json()
        log.warning(f"{model}: HTTP {r.status_code} (attempt {attempt}): {r.text[:200]}")
        time.sleep(2)
    raise RuntimeError(f"{model} failed twice: HTTP {r.status_code}")


def norm(s: str) -> str:
    return s.strip().strip("\"'«»“”`.,:;։*\n ").strip()


def eval_answer(task: str, word: str, out: str) -> bool:
    if task == "repeat":
        got = norm(out)
        return got == word or got.replace("եւ", "և") == word.replace("եւ", "և")
    letters = "".join(c for c in out if ARM_CHAR.search(c))
    a, b = letters.lower(), word.lower()
    return a == b or a.replace("եւ", "և") == b.replace("եւ", "և")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one call only")
    ap.add_argument("--per-model", action="store_true",
                    help="probe each model with ITS OWN vocab candidates "
                         "(from out/model_candidates.json)")
    ap.add_argument("--models", default=None,
                    help="override tokenizer->model mapping, e.g. "
                         "'o200k=openai/gpt-4.1-nano,llama3=meta-llama/"
                         "llama-3.2-1b-instruct'; only listed tokenizers run")
    ap.add_argument("--suffix", default="",
                    help="suffix for the output filename, e.g. '_small'")
    args = ap.parse_args()

    models = MODELS
    if args.models:
        models = dict(pair.split("=", 1) for pair in args.models.split(","))
        for t in models:
            if t not in MODELS:
                raise ValueError(f"unknown tokenizer {t} (have {list(MODELS)})")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {load_key()}"

    pricing = fetch_pricing(session, models)
    log.info(f"pricing loaded for {len(pricing)} models")

    if args.per_model:
        mc = json.loads(Path("out/model_candidates.json")
                        .read_text(encoding="utf-8"))
        jobs = [(tname, mid, c["surface"], "own", task)
                for tname, mid in models.items()
                for c in mc[tname]["candidates"] if c["probeable"]
                for task in TASKS]
        out_path = Path(f"out/live_probe_per_model{args.suffix}.json")
    else:
        jobs = [(tname, mid, w, grp, task)
                for tname, mid in models.items()
                for w, grp in WORDS
                for task in TASKS]
        out_path = Path(f"out/live_probe_results{args.suffix}.json")
    if args.smoke:
        jobs = jobs[:1]
    log.info(f"{len(jobs)} calls to make -> {out_path}")
    ledger, total_cost = [], 0.0
    for k, (tname, mid, w, grp, task) in enumerate(jobs):
        resp = call(session, mid, TASKS[task].format(w=w))
        # content can be None (observed: gemma-2 returns empty content when asked
        # to repeat the eternity sign) - that is a RESULT, not an error
        text = resp["choices"][0]["message"].get("content") or ""
        usage = resp.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * pricing[mid]["prompt"]
                + usage.get("completion_tokens", 0) * pricing[mid]["completion"])
        total_cost += cost
        ok = eval_answer(task, w, text)
        ledger.append({
            "tokenizer": tname, "model": mid, "word": w, "group": grp,
            "task": task, "ok": ok, "output": text.strip()[:300],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": round(cost, 6),
        })
        log.info(f"[{k + 1}/{len(jobs)}] {tname:8s} {task:6s} {w:18s} "
                 f"ok={ok} cost=${cost:.5f}")
        # checkpoint after EVERY call - paid results must survive a crash.
        # write-then-rename so a failed write cannot truncate the good ledger.
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"summary": None, "total_cost_usd": round(total_cost, 5),
             "ledger": ledger}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        time.sleep(0.2)

    summary = {}
    for tname in models:
        rows = [e for e in ledger if e["tokenizer"] == tname]
        summary[tname] = {"model": models[tname], "n_calls": len(rows),
                          "cost_usd": round(sum(e["cost_usd"] for e in rows), 5)}
        for task in TASKS:
            for grp in sorted({e["group"] for e in rows}):
                sel = [e for e in rows if e["task"] == task and e["group"] == grp]
                if sel:
                    summary[tname][f"{task}_{grp}"] = \
                        f"{sum(e['ok'] for e in sel)}/{len(sel)}"

    result = {"summary": summary, "total_cost_usd": round(total_cost, 5),
              "ledger": ledger}
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(out_path)
    log.info(f"wrote {out_path} - TOTAL COST ${total_cost:.4f}")


if __name__ == "__main__":
    sys.exit(main())
