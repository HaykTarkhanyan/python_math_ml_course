# sketch-finder

Find a Kargin Haghordum sketch from a half-remembered description. An agent searches 654 sketches with three kinds of
tools: SQL over a small SQLite database, regex over the Armenian dialogue, and web search. It comes with a known-answer
benchmark, a command line, a cost ledger and an MCP server. This is the project of video 07 of the course.

```text
> uv run sketch-finder ask "the one where a father and son go fishing and catch nothing"
INFO answer: 616 | 4 steps, 5 tool calls, $0.0130, 23 s
```

(A real run of 24 Nov. Sketch 616 is the right one.)

## Setup

You need [uv](https://docs.astral.sh/uv/). From this folder:

```bash
uv sync                          # creates .venv with the pinned versions, under a minute the first time
cp .env.example .env             # then put your OpenRouter and Tavily keys in .env
uv run sketch-finder build-db    # builds data/kargin.db from the CSV, about 1 s
```

## Use

| Command | What it does | Time and cost |
|---|---|---|
| `uv run sketch-finder ask "..."` | one question; add `--verbose` to see every step, `--model` to switch | about 10-25 s, about $0.01 |
| `uv run sketch-finder eval --kinds hazy --tools regex` | runs the known-answer questions, saves every run to `results/` | about 3 min and $0.16 per model on Sol |
| `uv run sketch-finder serve` | the four tools as an MCP server over stdio | |
| `uv run sketch-finder cost` | sums `logs/calls.jsonl`, the ledger of every model call | |
| `uv run pytest` | 23 tests of the database, the tools and the loop, with a fake model | a few seconds, free |

To use the tools inside Claude Code, register the server once (PowerShell, from this folder):

```powershell
claude mcp add kargin -- uv run --directory "$PWD" sketch-finder serve
```

## Layout

```text
src/sketch_finder/
  config.py      paths, limits, the models and their pinned providers
  database.py    builds the SQLite file from the CSV; read-only connections
  tools.py       run_sql, search_dialogue, read_sketch, web_search, and the descriptions the model sees
  agent.py       the system prompt and the agent loop
  evaluate.py    the known-answer benchmark: run, save, summarize
  ledger.py      one JSON line per model call, with its cost
  server.py      the MCP server
  cli.py         the sketch-finder command
tests/           pytest, no network
scripts/         freeze_data.py (copies the data from the Kargin project), build_questions.py (the question set)
data/            kargin_sketches.csv (702 rows, frozen), questions.json (20 known answers)
results/         saved benchmark runs
```

## The question set

`data/questions.json` has 20 questions with known answers: 5 for the database, 5 exact quotes and 10 hazy memories
written by hand. The hazy ones were not used while writing the system prompt, so they are a fair test.

Results of 24 Nov, 10 hazy questions, regex tools only (`results/hazy_comparison_2026-11-24.json`):

| Model | Found | Named a wrong sketch | Cost per question |
|---|---|---|---|
| GPT-6.1 Sol (default) | 7/10 | 1 | $0.016 |
| GPT-6.1 Terra | 6/10 | 4 | $0.010 |
| DeepSeek V4.1 Flash | 5/10 | 0 | $0.006 |
| Gemini 3 Flash | 3/10 | 0 | $0.009 |
| gpt-4o-mini | 0/10 | 4 | $0.001 |

Ten questions are too few to rank the top three: their 95% intervals overlap almost completely.

## Data

`data/kargin_sketches.csv` is a frozen copy of the Kargin Haghordum archive project: titles, hand-curated dialogue, actors,
locations and YouTube numbers. 48 rows are duplicate uploads and 96 unique sketches have no dialogue, so regex can reach
558 of the 654.
