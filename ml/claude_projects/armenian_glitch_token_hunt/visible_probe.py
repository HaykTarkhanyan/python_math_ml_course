"""Visible-input probe: the model DEFINITELY sees the word - can it say it?

The fun-probe stories were evasive because the bare glitch token is invisible on
the INPUT side. Here every prompt spells the target through tokens that are
guaranteed visible (single letters, two fragments split at a non-token boundary,
or a Unicode codepoint written in ASCII) and demands the word/character in the
OUTPUT. Temperature stays 0: any failure left is the model being unable to EMIT
the string it just assembled - output-side glitch, cleanly separated from
input blindness.

Scored: ok = target string appears in the output (և/եւ normalized).
Writes out/visible_probe_results.json (incremental, with per-call cost).
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
              logging.FileHandler("logs/visible_probe.py.log", encoding="utf-8")],
)
log = logging.getLogger("visible_probe")


def prompts_for(word: str) -> dict:
    if len(word) == 1:  # symbol tokens: describable via codepoint, in ASCII
        cp = f"U+{ord(word):04X}"
        return {
            "codepoint": f"Output the Unicode character {cp} and nothing else.",
            "codepoint_story": (f"Write one short sentence that literally contains "
                                f"the Unicode character {cp} - print the character "
                                f"itself, not the codepoint."),
            "html_entity": (f"Which character does the HTML entity "
                            f"&#x{ord(word):X}; represent? Output just that "
                            f"character."),
        }
    spaced = " ".join(word)
    k = max(1, len(word) // 2)
    return {
        "assemble": (f"Join these Armenian letters into a single word and output "
                     f"only that word, with no spaces: {spaced}"),
        "assemble_use": (f"Join these Armenian letters into a single word: "
                         f"{spaced}. Then write one short sentence that includes "
                         f"that word."),
        "concat": (f'Concatenate these two fragments into one word and output '
                   f'only the word: "{word[:k]}" + "{word[k:]}"'),
    }


def contains_target(word: str, out: str) -> bool:
    a = out.replace("եւ", "և")
    return word.replace("եւ", "և") in a


def main() -> None:
    per_model = json.loads(Path("out/live_probe_per_model.json")
                           .read_text(encoding="utf-8"))
    targets = [(e["tokenizer"], e["model"], e["word"])
               for e in per_model["ledger"]
               if e["task"] == "repeat" and not e["ok"]]
    log.info(f"targets: {[w for _, _, w in targets]}")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {load_key()}"
    pricing = fetch_pricing(session)

    jobs = [(tname, mid, w, pname, ptext)
            for tname, mid, w in targets
            for pname, ptext in prompts_for(w).items()]
    log.info(f"{len(jobs)} calls to make")

    out_path = Path("out/visible_probe_results.json")
    ledger, total_cost = [], 0.0
    for k, (tname, mid, w, pname, ptext) in enumerate(jobs):
        resp = call(session, mid, ptext)
        text = resp["choices"][0]["message"].get("content") or ""
        usage = resp.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * pricing[mid]["prompt"]
                + usage.get("completion_tokens", 0) * pricing[mid]["completion"])
        total_cost += cost
        ok = contains_target(w, text)
        ledger.append({"tokenizer": tname, "model": mid, "word": w,
                       "prompt": pname, "ok": ok,
                       "output": text.strip()[:400],
                       "cost_usd": round(cost, 6)})
        log.info(f"[{k + 1}/{len(jobs)}] {mid.split('/')[1]:22s} {pname:16s} "
                 f"{w:10s} ok={ok} -> {text.strip()[:60]!r}")
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"total_cost_usd": round(total_cost, 5), "ledger": ledger},
            ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        time.sleep(0.2)

    n_ok = sum(e["ok"] for e in ledger)
    log.info(f"done - {n_ok}/{len(ledger)} contain the target - "
             f"TOTAL COST ${total_cost:.4f}")


if __name__ == "__main__":
    sys.exit(main())
