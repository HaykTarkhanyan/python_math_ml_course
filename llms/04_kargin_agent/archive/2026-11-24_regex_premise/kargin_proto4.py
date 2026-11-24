"""Do ArmBench's affordable top models do well on OUR task? The same 12 hazy questions, regex tools only.

Run from the repo root with the ma venv. Expected: ~5 min, about $0.15, hard stop at $0.22.
"""
import json
import os
import pathlib
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import kargin_proto2 as proto2
import kargin_proto3 as proto3

BUDGET_USD = 0.22
MAX_STEPS = 10
RESULTS_PATH = pathlib.Path(__file__).parent / "kargin_proto4_results.json"
MODELS = [  # model, pinned provider
    ("deepseek/deepseek-v4.1-flash", "together"),
    ("google/gemini-3-flash-preview", "google-ai-studio"),
    ("google/gemma-4-31b-it", "crusoe"),  # parasail took over a minute per question on 24 Nov
]


def run_agent(client, model, provider, question, spent):
    messages = [{"role": "system", "content": proto2.SYSTEM}, {"role": "user", "content": question}]
    calls, cost = [], 0.0
    for step in range(MAX_STEPS):
        if spent + cost > BUDGET_USD:
            raise RuntimeError(f"budget of ${BUDGET_USD} reached")
        response = client.chat.completions.create(
            model=model, messages=messages, tools=proto2.TOOLS, max_tokens=2500,
            **({} if model in proto2.NO_TEMPERATURE else {"temperature": 0}),
            extra_body={
                "provider": {"order": [provider], "allow_fallbacks": False, "require_parameters": True},
                "reasoning": {"effort": "low"},
                "usage": {"include": True},
            },
        )
        if not response.choices:
            raise RuntimeError(f"{model}: provider error {getattr(response, 'error', None)}")
        call_cost = getattr(response.usage, "cost", None)
        if call_cost is None:
            raise RuntimeError("OpenRouter returned no cost")
        cost += call_cost
        message = response.choices[0].message
        if not message.tool_calls:
            found = re.findall(r"ANSWER:\s*(\S+)", message.content or "")
            return {"answer": found[-1] if found else None, "steps": step + 1, "calls": calls, "cost": cost}
        parsed = {}
        for tool_call in message.tool_calls:
            try:
                parsed[tool_call.id] = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                parsed[tool_call.id] = None
                tool_call.function.arguments = "{}"
        messages.append(message.model_dump(exclude_none=True))
        for tool_call in message.tool_calls:
            arguments = parsed[tool_call.id]
            if arguments is None:
                calls.append({"tool": tool_call.function.name, "arguments": {}, "broken_json": True})
                result = {"error": "Your tool call was too long and got cut off. Use a much shorter pattern."}
            else:
                calls.append({"tool": tool_call.function.name, "arguments": arguments})
                try:
                    result = proto2.TOOL_FUNCTIONS[tool_call.function.name](**arguments)
                except Exception as error:
                    result = {"error": f"{type(error).__name__}: {error}"}
            messages.append({"role": "tool", "tool_call_id": tool_call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    return {"answer": None, "steps": MAX_STEPS, "calls": calls, "cost": cost, "hit_step_cap": True}


def main():
    load_dotenv(".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=90, max_retries=1)
    results, spent, started, done = [], 0.0, time.time(), 0
    total = len(proto3.HAZY) * len(MODELS)
    for model, provider in MODELS:
        for expected, memory in proto3.HAZY.items():
            question = f"I remember a sketch but not its name: {memory} Which sketch is it?"
            step_started = time.time()
            outcome = run_agent(client, model, provider, question, spent)
            spent += outcome["cost"]
            done += 1
            correct = proto2.is_correct(outcome["answer"], expected)
            results.append({"model": model, "provider": provider, "memory": memory, "expected": expected,
                            "correct": correct, "seconds": time.time() - step_started, **outcome})
            RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            elapsed = time.time() - started
            print(f"{done}/{total} {model.split('/')[1]}: {'OK' if correct else 'NO'} expected={expected} got={outcome['answer']} "
                  f"steps={outcome['steps']} tools={len(outcome['calls'])} ${spent:.4f} {elapsed:.0f}s, "
                  f"~{elapsed / done * (total - done):.0f}s left", flush=True)
        group = [r for r in results if r["model"] == model]
        wrong = sum(1 for r in group if not r["correct"] and r["answer"] and str(r["answer"]).lower() != "none")
        print(f"== {model}: {sum(r['correct'] for r in group)}/{len(group)} right, {wrong} wrong sketch, "
              f"mean tools {sum(len(r['calls']) for r in group) / len(group):.1f}, "
              f"mean {sum(r['seconds'] for r in group) / len(group):.1f}s, ${sum(r['cost'] for r in group):.4f}", flush=True)
    print(f"total ${spent:.4f}, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
