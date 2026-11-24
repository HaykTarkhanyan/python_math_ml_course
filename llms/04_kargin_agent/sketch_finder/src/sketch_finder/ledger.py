"""A cost ledger: one JSON line per model call in logs/calls.jsonl, so 'what did this cost?' is answered from data."""
import json
import time

from sketch_finder import config


def record_call(model, provider, usage, label):
    config.LOGS_DIR.mkdir(exist_ok=True)
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "model": model,
        "provider": provider,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "cost": usage.cost,
    }
    with open(config.LEDGER_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")
    return entry


def read_ledger():
    if not config.LEDGER_PATH.exists():
        return []
    with open(config.LEDGER_PATH, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
