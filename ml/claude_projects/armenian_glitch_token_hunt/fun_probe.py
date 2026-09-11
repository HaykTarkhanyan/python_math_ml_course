"""Fun probe: the SolidGoldMagikarp interrogation battery.

Takes the CONFIRMED glitch tokens (repeat-failures from the per-model probe) and
hits each one with the prompt formulations that made the original glitch tokens
famous: definitions, identity checks, stories, letter counts, and the
quoted-vs-bare contrast (quoting changes the token boundary, so a model that
cannot repeat the bare token often CAN repeat the quoted one - proof that the
glitch lives in the token, not the string).

No pass/fail scoring - the outputs ARE the result. Ledger with costs saved
incrementally to out/fun_probe_results.json.
"""

import json
import logging
import sys
import time
from pathlib import Path

import requests

from glitch_live_probe import API, MODELS, call, fetch_pricing, load_key

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("logs/fun_probe.py.log", encoding="utf-8")],
)
log = logging.getLogger("fun_probe")

PROMPTS = {
    "define": 'What does "{w}" mean? Answer in one sentence.',
    "repeat_bare": "Repeat the following exactly, output nothing else: {w}",
    "repeat_quoted": 'Repeat the following exactly, output nothing else: "{w}"',
    "same": 'Are "{w}" and "{w}" the same string? Answer yes or no, then explain in one sentence.',
    "five_times": "Say the word {w} five times in a row, separated by commas.",
    "story": "Write a one-sentence story about {w}.",
    "count": 'How many characters are in "{w}"? Answer with just the number.',
}


def main() -> None:
    per_model = json.loads(Path("out/live_probe_per_model.json")
                           .read_text(encoding="utf-8"))
    targets = [(e["tokenizer"], e["model"], e["word"])
               for e in per_model["ledger"]
               if e["task"] == "repeat" and not e["ok"]]
    if not targets:
        raise RuntimeError("no confirmed glitch tokens found in per-model ledger")
    log.info(f"confirmed glitch tokens: {[w for _, _, w in targets]}")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {load_key()}"
    pricing = fetch_pricing(session)

    jobs = [(tname, mid, w, pname) for tname, mid, w in targets
            for pname in PROMPTS]
    log.info(f"{len(jobs)} calls to make")

    out_path = Path("out/fun_probe_results.json")
    ledger, total_cost = [], 0.0
    for k, (tname, mid, w, pname) in enumerate(jobs):
        resp = call(session, mid, PROMPTS[pname].format(w=w))
        text = resp["choices"][0]["message"].get("content") or ""
        usage = resp.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * pricing[mid]["prompt"]
                + usage.get("completion_tokens", 0) * pricing[mid]["completion"])
        total_cost += cost
        ledger.append({"tokenizer": tname, "model": mid, "word": w,
                       "prompt": pname, "output": text.strip()[:400],
                       "cost_usd": round(cost, 6)})
        log.info(f"[{k + 1}/{len(jobs)}] {mid.split('/')[1]:22s} {pname:14s} "
                 f"{w:10s} -> {text.strip()[:70]!r}")
        out_path.write_text(json.dumps(
            {"total_cost_usd": round(total_cost, 5), "ledger": ledger},
            ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.2)

    log.info(f"done - TOTAL COST ${total_cost:.4f}")


if __name__ == "__main__":
    sys.exit(main())
