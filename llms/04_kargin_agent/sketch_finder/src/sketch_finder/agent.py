"""The agent loop: ask the model, run the tools it asks for, hand back the results, stop when it answers."""
import json
import logging
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from sketch_finder import config
from sketch_finder.ledger import record_call
from sketch_finder.tools import ALL_TOOL_NAMES, TOOL_FUNCTIONS, TOOL_SCHEMAS

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about Kargin Haghordum, an Armenian sketch comedy show, using the tools.

The database has three tables:
  sketches(sketch_id, title, url, location, language, duration_sec, upload_date, view_count, dialogue)
  actors(actor_id, name)
  sketch_actors(sketch_id, actor_id)
Names, locations and dialogue are in Armenian. Locations include Տուն (home), Հիվանդանոց (hospital), Գրասենյակ (office).
Actors include Հայկո, Մկո, Աշոտ, Հասմիկ.

Use SQL for counts, rankings and anything about actors, locations and views.

To find a sketch from what happens in it, search the dialogue. It is spoken, colloquial Armenian: people do not describe
the plot, they just talk, and they often use slang and Russian loanwords.
- Patterns are Python regexes over Armenian text. Never put English words or commas in a pattern.
- Search one idea at a time, as a short word stem, because endings change: հիվանդ matches հիվանդանոց, հիվանդի.
- Use | for synonyms and spelling variants: (կնիք|պեչատ).
- Think about what a character would actually SAY in the scene, not how you would summarize it.
- Use all_patterns to require two or three stems in the same sketch when one stem matches too many sketches.
- Use skip to page through long result lists. Check a candidate with read_sketch before answering.

End with a final line exactly like: ANSWER: <value>. For "which sketch" the value is the sketch_id.
If nothing fits after your searches, write ANSWER: none."""


def make_client():
    for env_file in config.ENV_FILES:
        if env_file.exists():
            load_dotenv(env_file)   # an earlier file wins: load_dotenv never overrides a key already set
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=3,
        timeout=120,
    )


def call_model(client, model, messages, tools, label="agent"):
    """One model call, pinned to one provider so runs stay comparable. Returns the reply and its cost in dollars."""
    settings = config.MODELS[model]
    extra_body = {
        "provider": {"order": [settings["provider"]], "allow_fallbacks": False, "require_parameters": True},
        "usage": {"include": True},
    }
    if settings["reasoning"]:
        extra_body["reasoning"] = {"effort": "low"}
    request = {"model": model, "messages": messages, "tools": tools, "max_tokens": 2500, "extra_body": extra_body}
    if settings["temperature"]:
        request["temperature"] = 0

    response = client.chat.completions.create(**request)
    if not response.choices:
        raise RuntimeError(f"{model}: the provider returned an error: {getattr(response, 'error', None)}")
    if getattr(response.usage, "cost", None) is None:
        raise RuntimeError(f"{model}: OpenRouter returned no cost")
    record_call(model, settings["provider"], response.usage, label)
    return response.choices[0].message, response.usage.cost


def run_tool(name, arguments):
    """Run one tool. A failure goes back to the model as text, so it can fix its own call."""
    if arguments is None:
        return {"error": "Your tool call was too long and got cut off. Use a much shorter pattern."}
    try:
        return TOOL_FUNCTIONS[name](**arguments)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def read_tool_arguments(reply):
    """Parse each call's JSON. A reply cut off by max_tokens leaves broken JSON: mark it and keep the history valid."""
    arguments_by_id = {}
    for tool_call in reply.tool_calls:
        try:
            arguments_by_id[tool_call.id] = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments_by_id[tool_call.id] = None
            tool_call.function.arguments = "{}"
    return arguments_by_id


def read_final_answer(text):
    found = re.findall(r"ANSWER:\s*(.+)", text or "")
    return found[-1].strip() if found else None


def run_agent(client, question, model=None, tool_names=ALL_TOOL_NAMES, max_steps=None, verbose=False, label="agent"):
    """The loop: ask the model, run the tools it asks for, hand back the results, stop when it answers."""
    model = model or config.DEFAULT_MODEL
    max_steps = max_steps or config.MAX_STEPS

    # 1. start the conversation and offer the tools
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    tools = [TOOL_SCHEMAS[name] for name in tool_names]
    calls = []
    total_cost = 0.0
    started = time.time()

    for step in range(1, max_steps + 1):
        # 2. ask the model what to do next
        reply, cost = call_model(client, model, messages, tools, label)
        total_cost += cost

        # 3. no tool requested: this is the final answer
        if not reply.tool_calls:
            if verbose:
                log.info(f"step {step}: final answer\n{reply.content}")
            return {
                "answer": read_final_answer(reply.content),
                "text": reply.content,
                "steps": step,
                "calls": calls,
                "cost": total_cost,
                "seconds": time.time() - started,
                "hit_step_cap": False,
            }

        # 4. run every tool the model asked for and hand back the results
        arguments_by_id = read_tool_arguments(reply)
        messages.append(reply.model_dump(exclude_none=True))
        for tool_call in reply.tool_calls:
            name = tool_call.function.name
            arguments = arguments_by_id[tool_call.id]
            result = run_tool(name, arguments)
            calls.append({"tool": name, "arguments": arguments, "broken_json": arguments is None})
            result_text = json.dumps(result, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})
            if verbose:
                log.info(f"step {step}: {name}({json.dumps(arguments, ensure_ascii=False)})\n   -> {result_text[:220]}")

    # 5. out of steps: stop instead of spending forever
    if verbose:
        log.info(f"stopped: no answer after {max_steps} steps")
    return {
        "answer": None,
        "text": None,
        "steps": max_steps,
        "calls": calls,
        "cost": total_cost,
        "seconds": time.time() - started,
        "hit_step_cap": True,
    }
