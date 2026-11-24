"""Assemble and execute llms/04_kargin_agent/07_kargin_agent_solution.ipynb: "Find that sketch" (DECISIONS #121, #122).

The notebook is a guided tour of the project in sketch_finder/: it imports the package, shows each function's real source,
and runs it. Data, a read-only database, SQL, regex and web tools, failures and tests, why benchmarks matter
(ArmBench-LLM against our own test), the command line, an MCP server, and the handover to RAG.

Calls the models directly (no record/replay, DECISIONS #116). The five-model table comes from
sketch_finder/results/hazy_comparison_2026-11-24.json, so a class re-run does not pay for it again.
Needs OPENROUTER_API_KEY and TAVILY_API_KEY in the repository's .env (read, never printed), and `uv sync` done once in
sketch_finder/ for the command-line cells.

Run with the project venv (builds, executes, exports HTML; about 3 min, about $0.10):
    ./ma/Scripts/python.exe llms/04_kargin_agent/py_src/build_kargin_agent_nb.py
    ./ma/Scripts/python.exe llms/04_kargin_agent/py_src/build_kargin_agent_nb.py --markdown-only
"""
import argparse
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
CELLS = []
log = logging.getLogger("build_kargin_agent_nb")


def md(source):
    CELLS.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source):
    CELLS.append(nbf.v4.new_code_cell(source.strip("\n")))


def build_cells():
    CELLS.clear()
    # ==================================================================================== intro
    md(r"""
# Find that sketch: an agent over Kargin Haghordum

*Video 07.*

You remember a Kargin sketch but not its name: "the one where a lady lies to get a seat on the minibus". Today we build
an agent that finds it, across all 654 sketches, with the tools from videos 05 and 06.

The code is a real project in `sketch_finder/`, laid out the way you would lay out your own: a package, tests, a
command line, a cost ledger. This notebook is the guided tour. It imports the package, shows each piece, and runs it.

1. **The project** (0), **the model alone** (1), **the data** (2), **a read-only database** (3).
2. **Three kinds of tools:** SQL (4), regex over the dialogue (5), the web (6).
3. **Break it, fix it, test it** (7).
4. **Which model?** What a benchmark tells you (8), and our own benchmark (9).
5. **The command line** (10), **an MCP server** (11), and **the handover to RAG** (12).

> **Running it.** Every cell calls the model for real, so your answers will differ from the saved run of 24 Nov. A full
> run takes about 3 minutes and costs about $0.10. The five-model table in part 9 is read from a saved file, because
> that run alone cost $0.42.
""")

    # ==================================================================================== 0 the project
    md(r"""
## 0. The project
""")
    code(r"""
# ~5 s
import inspect
import json
import logging
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Code, display
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path("sketch_finder/src").resolve()))   # or: uv sync, then pick sketch_finder/.venv as the kernel
from sketch_finder import agent, config, database, evaluate, ledger, tools

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
for noisy in ("httpx", "mcp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def show(function):
    # print a function's real source from the project
    display(Code(inspect.getsource(function), language="python"))


client = agent.make_client()          # reads the keys from .env, never prints them
calls_before = len(ledger.read_ledger())
print("model:", config.DEFAULT_MODEL)
""")
    md(r"""
What is in the folder:
""")
    code(r"""
skip = {".venv", "__pycache__", "logs", ".pytest_cache", "uv.lock", "kargin.db"}

for path in sorted(Path("sketch_finder").rglob("*")):
    if path.is_file() and not skip & set(path.parts) and path.name not in skip:
        depth = len(path.relative_to("sketch_finder").parts) - 1
        print(f"{'    ' * depth}{path.name:28} {path.stat().st_size / 1000:7.1f} kB")
""")
    md(r"""
`pyproject.toml` names the dependencies with exact versions and declares the `sketch-finder` command. Each module does
one job: `config` (settings), `database`, `tools`, `agent` (the loop), `evaluate` (the benchmark), `ledger` (costs),
`server` (MCP), `cli`. We walk through them in that order.
""")
    code(r"""
print(Path("sketch_finder/pyproject.toml").read_text(encoding="utf-8"))
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
    model=config.DEFAULT_MODEL,
    messages=[{"role": "user", "content": question}],
    max_tokens=1000,
    extra_body={
        "provider": {"order": ["openai"], "allow_fallbacks": False},
        "reasoning": {"effort": "low"},
        "usage": {"include": True},
    },
)
ledger.record_call(config.DEFAULT_MODEL, "openai", response.usage, "hook")   # every call goes in the ledger
print(response.choices[0].message.content)
""")
    md(r"""
In the run of 24 Nov the model said it could not identify the sketch and asked what lie the woman tells. That is the
honest answer: it does not have this archive, so it cannot look. The right answer is sketch 198, and by the end of part
5 the agent finds it.
""")

    # ==================================================================================== 2 data
    md(r"""
## 2. The data

One CSV, frozen from the Kargin project (`scripts/freeze_data.py`): hand-curated dialogue, plus YouTube's numbers. Real
data, so it is messy.
""")
    code(r"""
table = pd.read_csv(config.CSV_PATH, dtype={"upload_date": str})

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

    # ==================================================================================== 3 database
    md(r"""
## 3. A read-only database

`database.py`. Names are split on commas, and on a space only when every word is a name seen on its own in at least five
sketches. "փոքր երեխա" (a small child) is a role, so it stays whole.
""")
    code(r"""
show(database.split_actor_names)
show(database.find_known_names)

known_names = database.find_known_names(unique_sketches.fillna("").to_dict("records"))
print(database.split_actor_names("Հայկո , Մկո Հասմիկ", known_names))
print(database.split_actor_names("Մկո, փոքր երեխա", known_names))
""")
    code(r"""
show(database.build_database)
print(database.build_database())
""")
    md(r"""
From 186 hand-typed strings to 76 actor names, in three tables. One safety step before an agent touches it: every tool
connects **read-only**. The model writes the SQL, and we do not want to find out what happens if it writes the wrong one.
""")
    code(r"""
show(database.open_read_only)

connection = database.open_read_only()
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

`tools.py`. The first tool runs a query and returns at most 30 rows, so one bad query cannot flood the model's context.
""")
    code(r"""
show(tools.run_sql)
tools.run_sql("SELECT name, count(*) AS sketches FROM actors JOIN sketch_actors USING (actor_id) GROUP BY name ORDER BY sketches DESC LIMIT 4")
""")
    md(r"""
As in video 05, the model only sees a description of each tool. The docstring becomes that description:
""")
    code(r"""
print(json.dumps(tools.TOOL_SCHEMAS["run_sql"], indent=1, ensure_ascii=False))
""")
    md(r"""
`agent.py`. The system prompt says what is in the database and how to search it. Most of what makes this agent work
lives here, and part 7 shows why.
""")
    code(r"""
print(agent.SYSTEM_PROMPT)
""")
    md(r"""
The loop from video 05, with three additions: each model is pinned to one provider (`config.MODELS`), every call is
written to the cost ledger, and a broken tool call goes back to the model as text instead of crashing the program.
""")
    code(r"""
show(agent.call_model)
show(agent.run_tool)
show(agent.read_tool_arguments)
""")
    code(r"""
show(agent.run_agent)


def show_run(outcome):
    # one summary line per run
    tools_used = Counter(call["tool"] for call in outcome["calls"])
    print(f"answer: {outcome['answer']} | {outcome['steps']} steps | tools: {dict(tools_used)} | "
          f"${outcome['cost']:.4f} | {outcome['seconds']:.0f} s")
""")
    md(r"""
The first run, with every step printed:
""")
    code(r"""
# ~5 s
outcome = agent.run_agent(client, "How many sketches take place in a hospital?", tool_names=("run_sql",), verbose=True)
print()
show_run(outcome)
""")
    code(r"""
# ~5 s
outcome = agent.run_agent(client, "Which actor appears in the most sketches, not counting Հայկո and Մկո?", tool_names=("run_sql",))
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
    code(r"""
show(tools.search_dialogue)
show(tools.read_sketch)

found = tools.search_dialogue(["դպրոց", "ուսուցիչ|դասատու"])
print("matches:", found["total_matches"])
print(found["results"][0])
""")
    md(r"""
Back to the opening question, and this time the model can look. Watch which words it tries.
""")
    code(r"""
# ~20 s
outcome = agent.run_agent(
    client,
    "I remember a sketch but not its name: A lady lies to get a seat on the minibus. Which sketch is it?",
    tool_names=tools.REGEX_TOOL_NAMES,
    verbose=True,
)
print()
show_run(outcome)
""")
    md(r"""
Look at the patterns in the run of 24 Nov. Three words for the minibus (`մարշրուտ|երթուղ|գազել`) found nothing, because
the dialogue spells it `մառշուտկա`. The agent then searched `հղի` (pregnant), a word that is not in the question at all:
it guessed what the lady would say. Eight sketches matched, and it read the right one.
""")

    # ==================================================================================== 6 web
    md(r"""
## 6. Tool 4: the web

Some questions are not in our database at all. The agent should notice that and go outside.
""")
    code(r"""
show(tools.web_search)
""")
    code(r"""
# ~15 s
outcome = agent.run_agent(
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
## 7. Break it, fix it, test it

Three failures that really happened while building this agent.

**Failure 1: a pattern that can never match.** A cheap model wrote its search like a list of keywords. In a regex, a
comma is just a comma, so this looks for that exact sentence:
""")
    code(r"""
print("what the model wrote:  ", tools.search_dialogue(["օլիգարխ, մահ, խնձոր"])["total_matches"], "matches")
print("one stem at a time:    ", tools.search_dialogue(["օլիգարխ"])["total_matches"], "matches")
print("two stems, both needed:", tools.search_dialogue(["օլիգարխ", "խնձոր"])["total_matches"], "matches")
""")
    md(r"""
The fix is one line in the system prompt: *"Never put English words or commas in a pattern."*

**Failure 2: a tool call cut off halfway.** One model wrote a regex so long that it hit `max_tokens` in the middle of the
JSON. The first version of the loop crashed on it. Here is that reply, rebuilt by hand:
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

arguments = agent.read_tool_arguments(reply)["call_1"]
print("our loop instead tells the model:", agent.run_tool("search_dialogue", arguments))
""")
    md(r"""
**Failure 3: an agent that never stops.** When a search keeps failing, a model can keep trying forever, and each try
costs money. The step cap ends it. Here the cap is set too low on purpose:
""")
    code(r"""
# ~5 s
outcome = agent.run_agent(
    client,
    "I remember a sketch but not its name: An old man keeps getting woken up by someone calling him. Which sketch is it?",
    tool_names=tools.REGEX_TOOL_NAMES,
    max_steps=2,
)
print("hit the step cap:", outcome["hit_step_cap"], "| answer:", outcome["answer"])
show_run(outcome)
""")
    md(r"""
Each of these failures is now a **test** in `sketch_finder/tests/`, so it cannot come back unnoticed. The tests use a
tiny database and a fake model that replays fixed replies: no key, no network, no cost. In a terminal it is
`uv run pytest`; here:
""")
    code(r"""
# ~10 s
!"{sys.executable}" -m pytest sketch_finder -q
""")

    # ==================================================================================== 8 benchmarks
    md(r"""
## 8. Why benchmarks matter

Which model should run this agent? There are hundreds, prices differ by a factor of 100, and most were never tested on
Armenian. A **benchmark** is a fixed test that many models take, so you can compare them without trying each one yourself.

For Armenian the one to know is **ArmBench-LLM**, from Metric AI Lab: 24 tasks in five categories (knowledge, reading,
analysis, generation, correction). This is its leaderboard on 24 Nov (`py_src/build_armbench_table.py` rebuilds it):
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

`evaluate.py` and `data/questions.json`: twenty questions with known answers, 5 for the database, 5 exact quotes, 10 hazy
memories. The RAG practical will rerun this same file. A question set like this is the most valuable thing in an LLM
project: it turns "seems to work" into a number.
""")
    code(r"""
show(evaluate.is_correct)

questions = evaluate.load_questions()
print(Counter(question["kind"] for question in questions))
""")
    md(r"""
First the easy half, live, on our model:
""")
    code(r"""
# ~1 min
easy_questions = evaluate.load_questions(["sql", "quote"])
easy_right = 0

for question in easy_questions:
    outcome = agent.run_agent(client, question["question"], tool_names=tools.SEARCH_TOOL_NAMES)
    correct = evaluate.is_correct(outcome["answer"], question)
    easy_right += correct
    print(f"{question['id']} {question['kind']:5} {'right' if correct else 'WRONG'} | expected {question['answer']:>5} | "
          f"got {str(outcome['answer'])[:12]:12} | {len(outcome['calls'])} tool calls")

print(f"\n{easy_right} of {len(easy_questions)} right")
""")
    md(r"""
Now the hard half: ten hazy memories, regex tools only, on five models. They are sketches we did **not** use while
writing the prompt, because a test you tuned on stops being a test. `uv run sketch-finder eval` runs this kind of
comparison and saves every run; here is the saved one:
""")
    code(r"""
saved_runs = json.loads(Path("sketch_finder/results/hazy_comparison_2026-11-24.json").read_text(encoding="utf-8"))
results = pd.DataFrame(evaluate.summarize(saved_runs)).set_index("model")

# how sure can we be with so few questions? (a 95% Wilson interval)
low, high = proportion_confint(results["right"], results["questions"], alpha=0.05, method="wilson")
results["low"] = low
results["high"] = high

# where each model stands on ArmBench
armbench_rank = armbench.set_index("model")["rank"]
results["armbench"] = [f"#{armbench_rank[m]}" if m in armbench_rank.index else "not listed" for m in results.index]

results.round({"cost_per_question": 4, "tool_calls_per_question": 1, "low": 2, "high": 2})
""")
    code(r"""
names = [model.split("/")[1] for model in results.index]
pass_rate = 100 * results["right"] / results["questions"]
error_low = pass_rate - 100 * results["low"]
error_high = 100 * results["high"] - pass_rate
colors = ["#F2A800" if model == config.DEFAULT_MODEL else "#0033A0" for model in results.index]

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
  once and gave up twice. DeepSeek never named a wrong sketch. The pass rate alone hides this, and it is why the class
  runs on Sol.
- **Price per question is not price per token.** DeepSeek is cheap per token but made about 15 tool calls per question.

> **The rule to take home:** a benchmark gives you a shortlist. Your own test, on your own task, picks the model.
""")

    # ==================================================================================== 10 CLI
    md(r"""
## 10. The command line

Nobody runs a notebook in production. The same agent is a command, declared in `pyproject.toml` and written in
`cli.py`. This is what you would type in a terminal after `uv sync`:
""")
    code(r"""
# ~15 s
!uv run --directory sketch_finder sketch-finder ask "the one where a father and son go fishing and catch nothing"
""")
    md(r"""
Every model call, from the notebook and from the command line, went into the ledger:
""")
    code(r"""
!uv run --directory sketch_finder sketch-finder cost
""")

    # ==================================================================================== 11 MCP
    md(r"""
## 11. The same tools as an MCP server

As in video 06, an MCP server makes our tools available to any client. `server.py` is short because it only wraps the
functions in `tools.py`:
""")
    code(r"""
display(Code(filename="sketch_finder/src/sketch_finder/server.py", language="python"))
""")
    md(r"""
We can talk to it from here, in memory. Each cell opens and closes its own client, which is the safe way in Jupyter.
""")
    code(r"""
# ~3 s
from mcp import Client
from sketch_finder.server import mcp


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
    tool_list = [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in listed.tools
    ]
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}, {"role": "user", "content": question}]
    total_cost = 0.0

    for step in range(1, max_steps + 1):
        # 2. ask the model
        reply, cost = agent.call_model(client, config.DEFAULT_MODEL, messages, tool_list, label="mcp")
        total_cost += cost
        if not reply.tool_calls:
            return agent.read_final_answer(reply.content), step, total_cost

        # 3. send each tool call to the server
        arguments_by_id = agent.read_tool_arguments(reply)
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
print(f"answer: {answer} | {steps} steps | ${cost:.4f}")
""")
    md(r"""
**In Claude Code.** Register the server once, from `sketch_finder/` in PowerShell, then ask in plain language ("which
Kargin sketch is the one with the fishing trip?"):

```powershell
claude mcp add kargin -- uv run --directory "$PWD" sketch-finder serve
```

We wrote no new tool code for this. That is what the protocol buys.
""")

    # ==================================================================================== 12 handover
    md(r"""
## 12. What regex could not do

The hazy questions our model missed in the saved run:
""")
    code(r"""
questions_by_id = {question["id"]: question for question in questions}
our_runs = [run for run in saved_runs if run["model"] == config.DEFAULT_MODEL]

for run in our_runs:
    if run["correct"]:
        continue
    memory = questions_by_id[run["id"]]["question"]
    memory = memory.replace("I remember a sketch but not its name: ", "").replace(" Which sketch is it?", "")
    print(f"{run['id']}: {memory}")
    print(f"     right answer {run['expected']}, the agent said {run['answer']}, after {len(run['calls'])} tool calls")

print()
print(f"tool calls per hazy question: {sum(len(run['calls']) for run in our_runs) / len(our_runs):.1f}, "
      f"cost per hazy question: ${sum(run['cost'] for run in our_runs) / len(our_runs):.4f}")
new_calls = ledger.read_ledger()[calls_before:]
print(f"this notebook made {len(new_calls)} model calls for ${sum(call['cost'] for call in new_calls):.3f}")
""")
    md(r"""
In the run of 24 Nov, regex plus a good model got all 10 database and quote questions, and 7 of the 10 hazy memories
in the saved comparison. That is further than most people expect. Look at what it took, though: about 7 tool calls for
one hazy question, because the model has to **guess the words** the characters said. If it guesses wrong, the sketch
stays invisible. And the 96 sketches with no dialogue cannot be found this way at all.

The next three lectures are about finding text **by meaning** instead of by exact words. Then the RAG practical reruns
this same question file and asks: can retrieval let a cheap model do what only an expensive one managed here? We do not
know yet. That is what the test is for.

## Homework (optional)

1. Write three hazy descriptions of sketches you remember, add them to `sketch_finder/data/questions.json`, and run
   `uv run sketch-finder eval --kinds hazy --tools regex`.
2. Add a tool `sketches_with(actor_a, actor_b)` to `tools.py` and `server.py`, write a test for it in
   `tests/test_tools.py`, register the server in Claude Code, and check that it picks your tool when you ask which
   sketches two actors share.
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
