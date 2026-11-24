"""Assemble and execute llms/04_kargin_agent/07_kargin_agent_solution.ipynb: "Find that sketch" (DECISIONS #121).

An agent over the 654 sketches of Kargin Haghordum: SQLite, regex search, web search, a known-answer check, why benchmarks
matter (ArmBench-LLM against our own test), and an MCP server. The functions shown in the notebook are copied here from
kargin_agent.py, so the notebook and the module cannot drift.

Calls the models directly (no record/replay, DECISIONS #116). The five-model table comes from
results/model_comparison.json (py_src/run_model_comparison.py), so a class re-run does not pay for it again.
Needs OPENROUTER_API_KEY and TAVILY_API_KEY in the repository's .env (read, never printed).

Run with the project venv (builds, executes, exports HTML; about 2 min, about $0.05):
    ./ma/Scripts/python.exe llms/04_kargin_agent/py_src/build_kargin_agent_nb.py
    ./ma/Scripts/python.exe llms/04_kargin_agent/py_src/build_kargin_agent_nb.py --markdown-only
"""
import argparse
import ast
import logging
import subprocess
import sys
import time
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve()
LESSON_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
OUTPUT_PATH = LESSON_DIR / "07_kargin_agent_solution.ipynb"
LOGS_DIR = REPO_ROOT / "logs"
MODULE_SOURCE = (LESSON_DIR / "kargin_agent.py").read_text(encoding="utf-8")
MODULE_TREE = ast.parse(MODULE_SOURCE)
CELLS = []
log = logging.getLogger("build_kargin_agent_nb")


def md(source):
    CELLS.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source):
    CELLS.append(nbf.v4.new_code_cell(source.strip("\n")))


def functions(*names):
    """The exact source of these functions in kargin_agent.py."""
    by_name = {node.name: node for node in MODULE_TREE.body if isinstance(node, ast.FunctionDef)}
    return "\n\n\n".join(ast.get_source_segment(MODULE_SOURCE, by_name[name]) for name in names)


def assignment(name):
    """The exact source of one top-level assignment in kargin_agent.py."""
    for node in MODULE_TREE.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.get_source_segment(MODULE_SOURCE, node)
    raise KeyError(name)


def schema(name):
    """One entry of TOOL_SCHEMAS, as its own assignment."""
    for node in MODULE_TREE.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "TOOL_SCHEMAS":
            for key, value in zip(node.value.keys, node.value.values):
                if key.value == name:
                    lines = ast.get_source_segment(MODULE_SOURCE, value).split("\n")
                    body = "\n".join([lines[0]] + [line[4:] for line in lines[1:]])
                    return f'TOOL_SCHEMAS["{name}"] = {body}'
    raise KeyError(name)


def build_cells():
    CELLS.clear()
    # ==================================================================================== intro
    md(r"""
# Find that sketch: an agent over Kargin Haghordum

*Video 07. Notebook only.*

You remember a Kargin sketch but not its name: "the one where a lady lies to get a seat on the minibus". Today we build
an agent that finds it, across all 654 sketches, with the tools from videos 05 and 06.

1. **The model alone** (1), then **the data** (2) and **a real database** (3).
2. **Three kinds of tools:** SQL (4), regex over the dialogue (5), the web (6).
3. **Break it, fix it** (7).
4. **Which model?** What a benchmark tells you (8), and our own benchmark (9).
5. **An MCP server** (10), so the same tools work inside Claude Code, and **the handover to RAG** (11).

> **Running it.** Every cell calls the model for real, so your answers will differ from the saved run of 24 Nov. A full
> run takes about 2 minutes and costs about $0.05. The five-model table in part 9 is read from a saved file, because
> that run alone cost $0.42.
""")

    # ==================================================================================== 0 setup
    md(r"""
## 0. Setup
""")
    code(r"""
# ~5 s
import json
import os
import re
import sqlite3
import csv
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from statsmodels.stats.proportion import proportion_confint

load_dotenv("../../.env")   # keys are read here and never printed
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    max_retries=3,
    timeout=120,
)

CSV_PATH = Path("data/kargin_sketches.csv")
DB_PATH = Path("data/kargin.db")
""" + "\n".join(assignment(name) for name in
                ("MAX_SQL_ROWS", "MAX_CELL_CHARS", "MAX_SEARCH_RESULTS", "MAX_PATTERN_CHARS", "MAX_DIALOGUE_CHARS",
                 "DEFAULT_MODEL")) + r"""

session_cost = 0.0     # every model call below adds to this
""")

    # ==================================================================================== 1 the model alone
    md(r"""
## 1. The model alone

First, no tools at all. We ask our model about the sketch directly.
""")
    code(r"""
# ~5 s
question = "Which Kargin Haghordum sketch is the one where a lady lies to get a seat on the minibus? Give its title and YouTube link."

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[{"role": "user", "content": question}],
    max_tokens=1000,
    extra_body={
        "provider": {"order": ["openai"], "allow_fallbacks": False},
        "reasoning": {"effort": "low"},
        "usage": {"include": True},
    },
)
session_cost += response.usage.cost
print(response.choices[0].message.content)
""")
    md(r"""
In the run of 24 Nov the model said it could not verify the title or link, offered a guess at a name, and pointed to a
YouTube search. That is a fair answer: it does not have this archive. The right answer is sketch 198, and by the end of
part 5 the agent can look it up.
""")

    # ==================================================================================== 2 data
    md(r"""
## 2. The data

One CSV, frozen from the Kargin project: hand-curated dialogue, plus YouTube's numbers. Real data, so it is messy.
""")
    code(r"""
table = pd.read_csv(CSV_PATH, dtype={"upload_date": str})

print("rows:", len(table))
print("columns:", list(table.columns))
table[["sketch_id", "title", "location", "actors_raw", "view_count"]].head(4)
""")
    code(r"""
# 1. the same sketch uploaded twice
duplicates = table["duplicate_of"].notna()
unique_sketches = table[~duplicates]

# 2. sketches nobody wrote the dialogue for
no_dialogue = unique_sketches["dialogue"].isna()

print(f"{len(table)} rows")
print(f"{duplicates.sum()} are duplicates, so {len(unique_sketches)} unique sketches")
print(f"{no_dialogue.sum()} of those have no dialogue, so {(~no_dialogue).sum()} can be searched by text")
print()
print("Dialogue of sketch 616, the start:")
print(unique_sketches.loc[unique_sketches["sketch_id"] == 616, "dialogue"].iloc[0][:260])
""")
    md(r"""
Spoken Armenian, with slang and missing punctuation. Remember this when we search it in part 5.

Now the actors column. A database wants one actor per row, and people typed these by hand:
""")
    code(r"""
raw_actors = unique_sketches["actors_raw"].dropna()

print("different strings in the column:", raw_actors.nunique())
print()
print("some of them:")
for value in raw_actors[raw_actors.str.contains("Մկո Հասմիկ|Հայկո Մկո")].unique()[:4]:
    print("  ", repr(value))
""")
    md(r"""
`"Հայկո , Մկո Հասմիկ"` is three people with one comma missing. We split on commas, and split on a space only when
every word is a name we have seen on its own in at least five sketches.
""")

    # ==================================================================================== 3 database
    md(r"""
## 3. A real database

Three tables: the sketches, the actors, and which actor is in which sketch.
""")
    code(functions("split_actor_names", "find_known_names", "empty_to_none") + r"""


known_names = find_known_names(unique_sketches.fillna("").to_dict("records"))
print(split_actor_names("Հայկո , Մկո Հասմիկ", known_names))
""")
    code(functions("build_database") + r"""


print(build_database())
""")
    md(r"""
From 186 hand-typed strings to 76 actor names. Before an agent touches this database, one safety step: every tool
connects **read-only**. The model writes the SQL, and we do not want to find out what happens if it writes the wrong one.
""")
    code(functions("open_read_only") + r"""


connection = open_read_only()
try:
    connection.execute("DROP TABLE sketches")
except sqlite3.OperationalError as error:
    print("The database refused:", error)
finally:
    connection.close()
""")

    # ==================================================================================== 4 sql tool and loop
    md(r"""
## 4. Tool 1: SQL, and the agent loop

The first tool runs a query and returns at most 30 rows, so one bad query cannot flood the model's context.
""")
    code(functions("run_sql") + r"""


run_sql("SELECT name, count(*) AS sketches FROM actors JOIN sketch_actors USING (actor_id) GROUP BY name ORDER BY sketches DESC LIMIT 4")
""")
    md(r"""
As in video 05, the model only sees a description of the tool. Its docstring becomes that description.
""")
    code(functions("tool_schema") + "\n\n\nTOOL_SCHEMAS = {}\nTOOL_FUNCTIONS = {\"run_sql\": run_sql}\n" + schema("run_sql") + r"""

print(json.dumps(TOOL_SCHEMAS["run_sql"], indent=1))
""")
    md(r"""
The system prompt tells the model what is in the database and how to search it. Most of what makes this agent work
lives here, and we will see why in part 7.
""")
    code(assignment("SYSTEM_PROMPT") + r"""


print(len(SYSTEM_PROMPT), "characters")
""")
    md(r"""
Now the loop from video 05, with three additions: every model is pinned to one provider, a tool failure goes back to the
model as text, and a cut-off tool call is handled (part 7 shows why).
""")
    code(assignment("MODELS") + "\n" + 'ALL_TOOL_NAMES = ("run_sql", "search_dialogue", "read_sketch", "web_search")\n'
         + 'REGEX_TOOL_NAMES = ("search_dialogue", "read_sketch")\n\n\n'
         + functions("call_model", "run_tool", "read_tool_arguments", "read_final_answer"))
    code(functions("run_agent") + r"""


def show_run(outcome):
    # one summary line per run, and the running total
    global session_cost
    session_cost += outcome["cost"]
    tools_used = Counter(call["tool"] for call in outcome["calls"])
    print(f"answer: {outcome['answer']} | {outcome['steps']} steps | tools: {dict(tools_used)} | "
          f"${outcome['cost']:.4f} | {outcome['seconds']:.0f} s")
""")
    md(r"""
The first run, with every step printed:
""")
    code(r"""
# ~5 s
outcome = run_agent(client, "How many sketches take place in a hospital?", tool_names=("run_sql",), verbose=True)
print()
show_run(outcome)
""")
    code(r"""
# ~5 s
outcome = run_agent(client, "Which actor appears in the most sketches, not counting Հայկո and Մկո?", tool_names=("run_sql",))
show_run(outcome)
""")

    # ==================================================================================== 5 regex
    md(r"""
## 5. Tools 2 and 3: search the dialogue

SQL is for facts in columns. "The one where..." lives in the dialogue, so the agent needs to search text. No embeddings
yet, just regex, which is also how coding agents like Claude Code search a codebase.

Armenian makes this interesting, because words change their endings. How far does the exact word get us?
""")
    code(r"""
# 1. all dialogue as sets of lowercase words
dialogues = unique_sketches["dialogue"].dropna().str.lower()
words_per_sketch = [set(re.findall(r"[Ա-և]+", dialogue)) for dialogue in dialogues]

# 2. compare the exact word with every word that starts with the same stem
for word, stem in [("դպրոց", "դպրոց"), ("հիվանդանոց", "հիվանդանոց"), ("մեքենա", "մեքեն"), ("փող", "փող")]:
    exact = sum(word in words for words in words_per_sketch)
    any_form = sum(any(w.startswith(stem) for w in words) for words in words_per_sketch)
    forms = {w for words in words_per_sketch for w in words if w.startswith(stem)}
    print(f"{word:12} exact word: {exact:3} sketches | stem {stem}: {any_form:3} sketches, {len(forms)} forms")
""")
    md(r"""
So the tool takes **regexes**, and the system prompt tells the model to search stems. It can also require several
patterns in the same sketch, which is what `grep a | grep b` does in a terminal.
""")
    code(functions("search_dialogue", "read_sketch") + r"""


found = search_dialogue(["դպրոց", "ուսուցիչ|դասատու"])
print("matches:", found["total_matches"])
print(found["results"][0])
print()
print({key: str(value)[:70] for key, value in read_sketch(found["results"][0]["sketch_id"]).items()})
""")
    code(schema("search_dialogue") + "\n" + schema("read_sketch") + r"""

TOOL_FUNCTIONS["search_dialogue"] = search_dialogue
TOOL_FUNCTIONS["read_sketch"] = read_sketch
print(list(TOOL_FUNCTIONS))
""")
    md(r"""
Back to the opening question, and this time the model can look. Every step is printed, so watch which words it tries.
""")
    code(r"""
# ~20 s
outcome = run_agent(
    client,
    "I remember a sketch but not its name: A lady lies to get a seat on the minibus. Which sketch is it?",
    tool_names=REGEX_TOOL_NAMES,
    verbose=True,
)
print()
show_run(outcome)
""")
    md(r"""
Look at the patterns in the run of 24 Nov. `մարշրուտ` found nothing, because the dialogue spells the minibus
`մառշուտկա`. The agent still got there through another pair of stems, `նստ` (sit) and `հղի` (pregnant), a word that is
not in the question at all. It guessed what the lady would say.
""")

    # ==================================================================================== 6 web
    md(r"""
## 6. Tool 4: the web

Some questions are not in our database at all. The agent should notice that and go outside.
""")
    code(functions("web_search") + "\n\n\n" + schema("web_search") + r"""

TOOL_FUNCTIONS["web_search"] = web_search
""")
    code(r"""
# ~15 s
outcome = run_agent(
    client,
    "How many sketches does Հայկո appear in, and what public office did the actor Hayko (Hayk Marutyan) hold after the show? "
    "Use the database for the count and the web for the office. End with ANSWER: <count>, <office>.",
    verbose=True,
)
print()
show_run(outcome)
""")

    # ==================================================================================== 7 break it
    md(r"""
## 7. Break it, fix it

Three failures that really happened while building this agent.

**Failure 1: a pattern that can never match.** A cheap model wrote its search like a list of keywords. In a regex, a
comma is just a comma, so this looks for that exact sentence:
""")
    code(r"""
print("what the model wrote: ", search_dialogue(["օլիգարխ, մահ, խնձոր"])["total_matches"], "matches")
print("one stem at a time:   ", search_dialogue(["օլիգարխ"])["total_matches"], "matches")
print("two stems, both needed:", search_dialogue(["օլիգարխ", "խնձոր"])["total_matches"], "matches")
""")
    md(r"""
The fix is one line in the system prompt: *"Never put English words or commas in a pattern."*

**Failure 2: a tool call cut off halfway.** One model wrote a regex so long that it hit `max_tokens` in the middle of the
JSON. The first version of our loop crashed on it. Here is that reply, rebuilt by hand:
""")
    code(r"""
cut_off_call = SimpleNamespace(
    id="call_1",
    function=SimpleNamespace(name="search_dialogue", arguments='{"all_patterns": ["(կնիք|պեչատ|կնքել|'),
)
reply = SimpleNamespace(tool_calls=[cut_off_call])

try:
    json.loads(cut_off_call.function.arguments)
except json.JSONDecodeError as error:
    print("json.loads alone would crash:", error)

arguments = read_tool_arguments(reply)["call_1"]
print("our loop instead tells the model:", run_tool("search_dialogue", arguments))
""")
    md(r"""
**Failure 3: an agent that never stops.** When a search keeps failing, a model can keep trying forever, and each try
costs money. The step cap ends it. Here the cap is set too low on purpose:
""")
    code(r"""
# ~5 s
outcome = run_agent(
    client,
    "I remember a sketch but not its name: An old man keeps getting woken up by someone calling him. Which sketch is it?",
    tool_names=REGEX_TOOL_NAMES,
    max_steps=2,
)
print("hit the step cap:", outcome["hit_step_cap"], "| answer:", outcome["answer"])
show_run(outcome)
""")

    # ==================================================================================== 8 benchmarks
    md(r"""
## 8. Why benchmarks matter

We have used one model so far. Why that one? There are hundreds, prices differ by a factor of 100, and most of them were
never tested on Armenian. A **benchmark** is a fixed test that many models take, so you can compare them without trying
each one yourself.

For Armenian the one to know is **ArmBench-LLM**, from Metric AI Lab: 24 tasks in five categories (knowledge, reading,
analysis, generation, correction). This is its leaderboard on 24 Nov:
""")
    code(r"""
armbench = pd.DataFrame(json.loads(Path("results/armbench_2026-11-24.json").read_text(encoding="utf-8")))

columns = ["rank", "model", "average", "Knowledge", "Reading", "origin"]
armbench[columns].head(8).round(3)
""")
    md(r"""
A leaderboard is useful and easy to misread. Three questions to ask before you trust one:

1. **Who produced this number?** Look at `origin`. The top three entries are "rounded task scores supplied", not full
   evaluation logs you could inspect.
2. **Is a surprising score real?** Look at this one:
""")
    code(r"""
armbench[armbench["model"].str.contains("grok-4")][["rank", "model", "average", "Knowledge", "Correction", "origin"]].round(3)
""")
    md(r"""
   One Grok is far below its siblings, with a correction score near zero. That pattern is more typical of a failed
   evaluation run than of a bad model. We cannot tell from the table, which is the point: check before you quote it.

3. **Does it measure your task?** ArmBench has exams, translation, summaries. It has **no task where a model searches
   with tools**, which is our whole job. So it can give us a shortlist. It cannot pick the winner.

For that we need a test of our own.
""")

    # ==================================================================================== 9 own benchmark
    md(r"""
## 9. Our own benchmark

Twenty questions with known answers, in `data/questions.json`: 5 for the database, 5 exact quotes, 10 hazy memories. The
RAG practical will rerun this same file. A question set like this is the most valuable thing in an LLM project: it turns
"seems to work" into a number.
""")
    code(functions("is_correct") + r"""


questions = json.loads(Path("data/questions.json").read_text(encoding="utf-8"))
print(Counter(question["kind"] for question in questions))
""")
    md(r"""
First the easy half, live, on our model:
""")
    code(r"""
# ~1 min
easy_questions = [question for question in questions if question["kind"] != "hazy"]
easy_right = 0

for question in easy_questions:
    outcome = run_agent(client, question["question"], tool_names=("run_sql", "search_dialogue", "read_sketch"))
    correct = is_correct(outcome["answer"], question)
    easy_right += correct
    session_cost += outcome["cost"]
    print(f"{question['id']} {question['kind']:5} {'right' if correct else 'WRONG'} | expected {question['answer']:>5} | "
          f"got {str(outcome['answer'])[:12]:12} | {len(outcome['calls'])} tool calls")

print(f"\n{easy_right} of {len(easy_questions)} right")
""")
    md(r"""
Now the hard half: ten hazy memories, regex tools only, on five models. They are sketches we did **not** use while
writing the prompt, because a test you tuned on stops being a test. This run is saved:
""")
    code(r"""
saved_runs = pd.DataFrame(json.loads(Path("results/model_comparison.json").read_text(encoding="utf-8")))

# 1. a miss is either "no answer" or, worse, a confident wrong sketch
said_nothing = saved_runs["answer"].isna() | saved_runs["answer"].str.strip(" .").str.lower().eq("none")
saved_runs["wrong_sketch"] = ~saved_runs["correct"] & ~said_nothing

# 2. one row per model
results = saved_runs.groupby("model").agg(
    right=("correct", "sum"),
    wrong_sketch=("wrong_sketch", "sum"),
    questions=("correct", "size"),
    cost_per_question=("cost", "mean"),
    tool_calls=("calls", lambda calls: calls.map(len).mean()),
    seconds=("seconds", "mean"),
)

# 3. how sure can we be with so few questions? (a 95% Wilson interval)
low, high = proportion_confint(results["right"], results["questions"], alpha=0.05, method="wilson")
results["low"] = low
results["high"] = high

# 4. where the model stands on ArmBench
armbench_rank = armbench.set_index("model")["rank"]
results["armbench"] = [f"#{armbench_rank[m]}" if m in armbench_rank.index else "not listed" for m in results.index]
results = results.sort_values(["right", "cost_per_question"], ascending=[False, True])

results.round({"cost_per_question": 4, "tool_calls": 1, "seconds": 1, "low": 2, "high": 2})
""")
    code(r"""
names = [model.split("/")[1] for model in results.index]
pass_rate = 100 * results["right"] / results["questions"]
error_low = pass_rate - 100 * results["low"]
error_high = 100 * results["high"] - pass_rate
colors = ["#F2A800" if model == DEFAULT_MODEL else "#0033A0" for model in results.index]

figure, axis = plt.subplots(figsize=(8, 3.4))
bars = axis.barh(names, pass_rate, color=colors, xerr=[error_low, error_high], capsize=4, ecolor="#555555")
axis.bar_label(bars, labels=[f"{r}/{q}" for r, q in zip(results["right"], results["questions"])], padding=26)
axis.invert_yaxis()
axis.set_xlim(0, 125)
axis.set_xlabel("hazy questions found (%), with a 95% interval")
axis.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
""")
    md(r"""
**What the run of 24 Nov shows.**

- **The shortlist helped, but did not decide.** ArmBench's #1, Sol, found the most (7 of 10). Its #2 and #5 found 5 and 3.
  Terra, which ArmBench has never tested, found 6.
- **Ten questions cannot rank close models.** The intervals of Sol, Terra and DeepSeek overlap almost completely. The
  clear gap is gpt-4o-mini, at 0 of 10, against Sol and Terra.
- **A wrong answer is worse than no answer.** When Terra missed, it named the wrong sketch all 4 times. Sol did that
  once and gave up twice. DeepSeek never named a wrong sketch. The pass rate alone hides this.
- **Price per question is not price per token.** DeepSeek is cheap per token but made about 15 tool calls per question.

> **The rule to take home:** a benchmark gives you a shortlist. Your own test, on your own task, picks the model.
""")

    # ==================================================================================== 10 MCP
    md(r"""
## 10. The same tools as an MCP server

Our four tools only work inside this notebook. As in video 06, an MCP server makes them available to any client. The
file is short because it just wraps the functions we wrote:
""")
    code(r"""
print(Path("kargin_server.py").read_text(encoding="utf-8"))
""")
    md(r"""
We can talk to it from here, in memory. Each cell opens and closes its own client, which is the safe way in Jupyter.
""")
    code(r"""
# ~3 s
import logging

from mcp import Client
from kargin_server import mcp

logging.getLogger("httpx").setLevel(logging.WARNING)   # the MCP library turns on request logs, we do not need them here
logging.getLogger("mcp").setLevel(logging.WARNING)


def result_text(result):
    # an MCP result is a list of content parts
    return "\n".join(part.text for part in result.content if part.type == "text")


async with Client(mcp) as mcp_client:
    listed = await mcp_client.list_tools()
    for tool in listed.tools:
        print(f"{tool.name:16} {tool.description[:72]}")
    print()
    result = await mcp_client.call_tool("run_sql", {"query": "SELECT count(*) AS sketches FROM sketches"})
    print("is_error:", result.is_error, "|", " ".join(result_text(result).split()))
""")
    md(r"""
Now the model uses the tools **through the server**. It is the same loop, except the tool list comes from `list_tools`
and each call goes through `call_tool`:
""")
    code(r"""
async def run_agent_over_mcp(question, max_steps=10):
    # 1. ask the server what tools exist, and describe them to the model
    async with Client(mcp) as mcp_client:
        listed = await mcp_client.list_tools()
    tools = [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in listed.tools
    ]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    total_cost = 0.0

    for step in range(1, max_steps + 1):
        # 2. ask the model
        reply, cost = call_model(client, DEFAULT_MODEL, messages, tools)
        total_cost += cost
        if not reply.tool_calls:
            return read_final_answer(reply.content), step, total_cost

        # 3. send each tool call to the server
        arguments_by_id = read_tool_arguments(reply)
        messages.append(reply.model_dump(exclude_none=True))
        for tool_call in reply.tool_calls:
            arguments = arguments_by_id[tool_call.id] or {}
            async with Client(mcp) as mcp_client:
                result = await mcp_client.call_tool(tool_call.function.name, arguments)
            print(f"step {step}: {tool_call.function.name} through MCP, is_error={result.is_error}")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text(result)})

    return None, max_steps, total_cost
""")
    code(r"""
# ~10 s
answer, steps, cost = await run_agent_over_mcp("Which sketch has the most YouTube views? Give its sketch_id.")
session_cost += cost
print(f"answer: {answer} | {steps} steps | ${cost:.4f}")
""")
    md(r"""
**In Claude Code.** Register the same file once, from the repository root in PowerShell, then ask in plain language
("which Kargin sketch is the one with the fishing trip?"):

```powershell
claude mcp add kargin -- "$PWD/ma/Scripts/python.exe" "$PWD/llms/04_kargin_agent/kargin_server.py"
```

We wrote no new tool code for this. That is what the protocol buys.
""")

    # ==================================================================================== 11 handover
    md(r"""
## 11. What regex could not do

Here are the hazy questions our model missed in the saved run:
""")
    code(r"""
questions_by_id = {question["id"]: question for question in questions}
our_model_runs = saved_runs[saved_runs["model"] == DEFAULT_MODEL]

for _, run in our_model_runs[~our_model_runs["correct"]].iterrows():
    memory = questions_by_id[run["id"]]["question"]
    memory = memory.replace("I remember a sketch but not its name: ", "").replace(" Which sketch is it?", "")
    print(f"{run['id']}: {memory}")
    print(f"     right answer {run['expected']}, the agent said {run['answer']}, after {len(run['calls'])} tool calls")

print()
print(f"hazy questions: {len(our_model_runs)}, tool calls per question: {our_model_runs['calls'].map(len).mean():.1f}, "
      f"cost per question: ${our_model_runs['cost'].mean():.4f}")
print(f"this notebook's live calls cost ${session_cost:.3f}")
""")
    md(r"""
In the run of 24 Nov, regex plus a good model got all 10 database and quote questions and 6 of the 10 hazy memories.
That is further than most people expect. Look at what it took, though: about 10 tool calls for one question, because the
model has to **guess the words** the characters said. If it guesses wrong, the sketch stays invisible. And the 96
sketches with no dialogue cannot be found this way at all.

The next three lectures are about finding text **by meaning** instead of by exact words. Then the RAG practical reruns
this same question file and asks: can retrieval let a cheap model do what only an expensive one managed here? We do not
know yet. That is what the test is for.

## Homework (optional)

1. Write three hazy descriptions of sketches you remember, add them to `data/questions.json`, and see if the agent
   finds them.
2. Add a tool `sketches_with(actor_a, actor_b)` to `kargin_server.py`, register the server in Claude Code, and check
   that it picks your tool when you ask which sketches two actors share.
""")


def export_html():
    subprocess.run([sys.executable, str(REPO_ROOT / "non_essential" / "notebook_to_html.py"), str(OUTPUT_PATH)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-only", action="store_true",
                        help="copy the markdown cells into the executed notebook; code and outputs untouched")
    arguments = parser.parse_args()
    LOGS_DIR.mkdir(exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "build_kargin_agent_nb.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    build_cells()

    if arguments.markdown_only:
        old = nbf.read(OUTPUT_PATH, as_version=4)
        same_shape = [c.cell_type for c in old.cells] == [c.cell_type for c in CELLS]
        if not same_shape or any(a.source != b.source for a, b in zip(old.cells, CELLS) if a.cell_type == "code"):
            raise SystemExit("the code cells changed since the last run: rebuild and execute instead")
        for old_cell, new_cell in zip(old.cells, CELLS):
            if old_cell.cell_type == "markdown":
                old_cell.source = new_cell.source
        nbf.write(old, OUTPUT_PATH)
        log.info(f"markdown refreshed in {OUTPUT_PATH}")
        export_html()
        return

    kernelspec = {"name": "ma", "display_name": "Python (ma)", "language": "python"}
    notebook = nbf.v4.new_notebook(cells=CELLS)
    notebook.metadata["kernelspec"] = kernelspec
    notebook.metadata["language_info"] = {"name": "python"}
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    started = time.perf_counter()
    try:
        NotebookClient(notebook, timeout=600, kernel_name="ma", startup_timeout=300,
                       resources={"metadata": {"path": str(LESSON_DIR)}}).execute()
    except CellExecutionError:
        nbf.write(notebook, OUTPUT_PATH)     # keep the outputs so far for debugging, then fail
        log.error(f"a cell failed after {time.perf_counter() - started:.0f} s; partial notebook in {OUTPUT_PATH}")
        raise
    notebook.metadata["kernelspec"] = kernelspec
    log.info(f"executed in {time.perf_counter() - started:.0f} s")
    nbf.write(notebook, OUTPUT_PATH)
    log.info(f"wrote {OUTPUT_PATH} ({len(CELLS)} cells)")
    export_html()


if __name__ == "__main__":
    main()
