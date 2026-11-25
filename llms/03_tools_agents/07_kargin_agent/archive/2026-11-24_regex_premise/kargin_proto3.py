"""Last premise test: the same 12 sketches, described the way a fan half-remembers them. Regex tools only.

Run from the repo root with the ma venv. Expected: ~4 min, about $0.10, hard stop at $0.15.
"""
import json
import os
import pathlib
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import kargin_proto2 as proto2

proto2.BUDGET_USD = 0.15
MODELS = ["openai/gpt-4o-mini", "openai/gpt-6.1-terra"]
RESULTS_PATH = pathlib.Path(__file__).parent / "kargin_proto3_results.json"

HAZY = {
    "206": "A guy can't stop acting like he's at work even at home, and his kid copies him.",
    "138": "Two kids, one is crying and the other one thinks it's hilarious.",
    "502": "A kid keeps asking his dad who would win.",
    "364": "A rich guy dies and tries to pay his way in.",
    "198": "A lady lies to get a seat on public transport.",
    "528": "Someone goes to a specialist for help and it turns out the specialist is after something else.",
    "203": "A guy shows up at his buddy's work with strange questions because he already knows what the buddy did.",
    "062": "A dad is angry about his daughter until he hears how much money is involved.",
    "507": "A wife is mad that he's late, he asks for something small and she vanishes for hours.",
    "527": "A guy back from abroad is impressed by how sweetly his friend talks to his wife.",
    "033": "A war sketch where the commander just won't give the order.",
    "034": "A husband fakes a work trip but doesn't actually go very far.",
}


def main():
    load_dotenv(".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    results, spent, started, done = [], 0.0, time.time(), 0
    total = len(HAZY) * len(MODELS)
    for model in MODELS:
        for expected, memory in HAZY.items():
            question = f"I remember a sketch but not its name: {memory} Which sketch is it?"
            outcome = proto2.run_agent(client, model, question, spent)
            spent += outcome["cost"]
            done += 1
            correct = proto2.is_correct(outcome["answer"], expected)
            results.append({"model": model, "memory": memory, "expected": expected, "correct": correct, **outcome})
            RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            elapsed = time.time() - started
            print(f"{done}/{total} {model.split('/')[1]}: {'OK' if correct else 'NO'} expected={expected} got={outcome['answer']} "
                  f"steps={outcome['steps']} tools={len(outcome['calls'])} ${spent:.4f} {elapsed:.0f}s, "
                  f"~{elapsed / done * (total - done):.0f}s left", flush=True)
        group = [r for r in results if r["model"] == model]
        wrong = sum(1 for r in group if not r["correct"] and r["answer"] and str(r["answer"]).lower() != "none")
        print(f"== {model}: {sum(r['correct'] for r in group)}/{len(group)} right, {wrong} wrong sketch, "
              f"mean tools {sum(len(r['calls']) for r in group) / len(group):.1f}, ${sum(r['cost'] for r in group):.4f}", flush=True)
    print(f"total ${spent:.4f}, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
