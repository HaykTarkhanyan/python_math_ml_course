"""Fair retest of regex on the 12 vague questions: better prompt, a tool that can AND several patterns, three models.

Run from the repo root with the ma venv. Expected: ~5 min, about $0.15, hard stop at $0.22.
"""
import json
import os
import pathlib
import re
import sqlite3
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import kargin_proto as proto

BUDGET_USD = 0.22
MAX_STEPS = 10
MODELS = ["openai/gpt-4o-mini", "openai/gpt-4.1-mini", "openai/gpt-6.1-terra"]
NO_TEMPERATURE = {"openai/gpt-6.1-terra"}
RESULTS_PATH = pathlib.Path(__file__).parent / "kargin_proto2_results.json"

SYSTEM = """You find sketches of Kargin Haghordum, an Armenian comedy show, using the tools.
The searchable text is each sketch's title and its Armenian dialogue. The dialogue is spoken, colloquial Armenian: people
do not describe the plot, they just talk, and they often use slang and Russian loanwords.

How to search:
- Patterns are Python regexes over Armenian text. Never put English words or commas in a pattern.
- Search one idea at a time, as a short word stem, because endings change: հիվանդ matches հիվանդանոց, հիվանդի.
- Use | for synonyms and spelling variants: (կնիք|պեչատ).
- Think about what a character would actually SAY in the scene, not how you would summarize it.
- Use all_patterns to require two or three stems in the same sketch when one stem matches too many sketches.
- Use skip to page through long result lists. Check a candidate with read_sketch before answering.

End with a final line exactly like: ANSWER: <sketch_id>. If nothing fits after your searches, write ANSWER: none."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_dialogue",
        "description": "Find sketches whose title or dialogue match EVERY regex in all_patterns. Returns the total and 10 snippets.",
        "parameters": {"type": "object", "properties": {
            "all_patterns": {"type": "array", "items": {"type": "string"}, "description": "1 to 3 regexes, all must match"},
            "skip": {"type": "integer", "description": "How many results to skip, for paging"},
        }, "required": ["all_patterns"]}}},
    {"type": "function", "function": {
        "name": "read_sketch", "description": "Read one sketch's title and dialogue.",
        "parameters": {"type": "object", "properties": {"sketch_id": {"type": "integer"}}, "required": ["sketch_id"]}}},
]


def search_dialogue(all_patterns, skip=0):
    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in all_patterns]
    connection = sqlite3.connect(f"file:{proto.DB_PATH.as_posix()}?mode=ro", uri=True)
    rows = connection.execute("SELECT sketch_id, title, dialogue FROM sketches").fetchall()
    connection.close()
    matches = []
    for sketch_id, title, dialogue in rows:
        text = f"{title}\n{dialogue or ''}"
        found = [regex.search(text) for regex in regexes]
        if all(found):
            first = found[0]
            matches.append({"sketch_id": sketch_id, "title": title[:50],
                            "snippet": text[max(0, first.start() - 50):first.end() + 50].replace("\n", " ")})
    return {"total_matches": len(matches), "results": matches[skip:skip + 10]}


TOOL_FUNCTIONS = {"search_dialogue": search_dialogue, "read_sketch": proto.read_sketch}


def run_agent(client, model, question, spent):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    calls, cost = [], 0.0
    for step in range(MAX_STEPS):
        if spent + cost > BUDGET_USD:
            raise RuntimeError(f"budget of ${BUDGET_USD} reached")
        request = {"model": model, "messages": messages, "tools": TOOLS, "max_tokens": 1500,
                   "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": False}, "usage": {"include": True}}}
        if model in NO_TEMPERATURE:
            request["extra_body"]["reasoning"] = {"effort": "low"}
        else:
            request["temperature"] = 0
        response = client.chat.completions.create(**request)
        call_cost = getattr(response.usage, "cost", None)
        if call_cost is None:
            raise RuntimeError("OpenRouter returned no cost")
        cost += call_cost
        message = response.choices[0].message
        if not message.tool_calls:
            found = re.findall(r"ANSWER:\s*(\S+)", message.content or "")
            return {"answer": found[-1] if found else None, "steps": step + 1, "calls": calls, "cost": cost}
        # A reply cut off by max_tokens leaves broken JSON: tell the model, and keep valid JSON in the history.
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
                    result = TOOL_FUNCTIONS[tool_call.function.name](**arguments)
                except Exception as error:
                    result = {"error": f"{type(error).__name__}: {error}"}
            messages.append({"role": "tool", "tool_call_id": tool_call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    return {"answer": None, "steps": MAX_STEPS, "calls": calls, "cost": cost, "hit_step_cap": True}


def is_correct(answer, expected):
    digits = re.sub(r"\D", "", answer or "")
    return bool(digits) and int(digits) == int(expected)


def main():
    load_dotenv(".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    db, table = proto.build_database()
    vague = [q for q in proto.make_questions(db, table) if q[0] == "vague"]
    results, spent, started = [], 0.0, time.time()
    total = len(vague) * len(MODELS)
    done = 0
    for model in MODELS:
        for _, question, expected in vague:
            outcome = run_agent(client, model, question, spent)
            spent += outcome["cost"]
            done += 1
            correct = is_correct(outcome["answer"], expected)
            results.append({"model": model, "question": question, "expected": expected, "correct": correct, **outcome})
            RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            elapsed = time.time() - started
            print(f"{done}/{total} {model.split('/')[1]}: {'OK' if correct else 'NO'} expected={expected} got={outcome['answer']} "
                  f"steps={outcome['steps']} ${spent:.4f} {elapsed:.0f}s, ~{elapsed / done * (total - done):.0f}s left", flush=True)
        group = [r for r in results if r["model"] == model]
        print(f"== {model}: {sum(r['correct'] for r in group)}/{len(group)} correct, ${sum(r['cost'] for r in group):.4f}", flush=True)
    print(f"total ${spent:.4f}, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
