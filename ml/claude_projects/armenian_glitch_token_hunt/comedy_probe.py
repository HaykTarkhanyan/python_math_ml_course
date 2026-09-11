"""Comedy probe: bare tokens + evasion-proof questions = the good stuff.

Lesson from the fun probe: corpus-NULL tokens mostly produce empty output
(anti-comedy), while RARE-but-alive tokens produce confident nonsense (the
հրով x5 substitution, the Kannada spelling). So this battery hits ALL probeable
candidates - null and rare - of every model with bare-token questions the model
cannot evade: what language, translate, define, count occurrences, real-word
duel, use in a sentence. Temperature stays 0; the comedy must be earned.

No scoring - outputs are the result. Ledger -> out/comedy_probe_results.json.
"""

import json
import logging
import sys
import time
from pathlib import Path

import requests

from glitch_live_probe import MODELS, call, fetch_pricing, load_key

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("logs/comedy_probe.py.log", encoding="utf-8")],
)
log = logging.getLogger("comedy_probe")

PROMPTS = {
    "language": "What language is this word: {w}",
    "translate": "Translate {w} to English. Answer with just the translation.",
    "define": "Define {w} in one short sentence.",
    "count_occ": ("How many times does {w} appear in the following text: "
                  "{w} {w} {w}. Answer with just a number."),
    "real_word": ("Which of these is a real Armenian word: {w} or բարև? "
                  "Answer with the word only."),
    "in_english": "Use {w} in a short English sentence.",
}


def main() -> None:
    mc = json.loads(Path("out/model_candidates.json").read_text(encoding="utf-8"))
    targets = [(tname, MODELS[tname], c["surface"], c["count"])
               for tname in MODELS
               for c in mc[tname]["candidates"] if c["probeable"]]
    log.info(f"{len(targets)} tokens x {len(PROMPTS)} prompts")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {load_key()}"
    pricing = fetch_pricing(session)

    out_path = Path("out/comedy_probe_results.json")
    ledger, total_cost = [], 0.0
    jobs = [(tname, mid, w, cnt, pname) for tname, mid, w, cnt in targets
            for pname in PROMPTS]
    for k, (tname, mid, w, cnt, pname) in enumerate(jobs):
        resp = call(session, mid, PROMPTS[pname].format(w=w))
        text = resp["choices"][0]["message"].get("content") or ""
        usage = resp.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * pricing[mid]["prompt"]
                + usage.get("completion_tokens", 0) * pricing[mid]["completion"])
        total_cost += cost
        ledger.append({"tokenizer": tname, "model": mid, "word": w,
                       "corpus_count": cnt, "prompt": pname,
                       "output": text.strip()[:400],
                       "cost_usd": round(cost, 6)})
        log.info(f"[{k + 1}/{len(jobs)}] {mid.split('/')[1]:22s} {pname:11s} "
                 f"{w:12s} -> {text.strip()[:60]!r}")
        # write-then-rename: an ENOSPC mid-write must never destroy the ledger
        # (learned the hard way - a truncating write_text lost 100 paid calls)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"total_cost_usd": round(total_cost, 5), "ledger": ledger},
            ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        time.sleep(0.2)

    log.info(f"done - TOTAL COST ${total_cost:.4f}")


if __name__ == "__main__":
    sys.exit(main())
