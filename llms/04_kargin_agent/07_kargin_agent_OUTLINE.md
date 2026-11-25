# Video 07: "Find that sketch", an agent over Kargin Haghordum

**Status:** taught 2026-11-25. Approved 2026-11-24 (Q60) and built. The code is a uv project in `sketch_finder/` (package, tests, command
line, eval, cost ledger, MCP server; Q64-Q66, DECISIONS #122). `07_kargin_agent_solution.ipynb` is the guided tour of it,
on GPT-6.1 Sol (Q63). The 20 questions still wait for the instructor's check.
**Decided (Q53-Q60):** the very next session, before the RAG lectures. Notebook only. The regex tool reads dialogue and
titles only. The session ends with an MCP server. Claude drafts the test questions, the instructor checks them. The
model choice goes through ArmBench-LLM, and the lesson teaches why benchmarks matter. Project 9 is dropped.

## What students leave with

An agent that answers questions about 654 Armenian sketches using SQL, regex and web search, a pass rate on known
questions, and the same tools running as an MCP server inside Claude Code.

## What the prototype showed (24 Nov)

The planned story was "regex fails on vague memories, so we need RAG". **The test did not support that.** The same 12
sketches, regex tools only, one run each:

| Question type | gpt-4o-mini | gpt-4.1-mini | GPT-6.1 Terra |
|---|---|---|---|
| SQL (4) | 4/4 | not run | not run |
| Exact quote (4) | 4/4 | not run | not run |
| Described from the plot summary (12) | 3/12 | 5/12 | **11/12** |
| One hazy sentence, as a fan would say it (12) | **1/12** | not run | **9/12** |

- The strong model knows what characters would really say: it searched slang such as `գայիշնիկ`, `շտրոֆ` and `զոքանչ`.
- On hazy questions the cheap model named the wrong sketch 7 times of 12. Terra did once and gave up twice.
- Cost per hazy question: about $0.001 on gpt-4o-mini, about $0.0075 on Terra (7.6 tool calls, about 10 s).
- Limits of this test: 12 sketches, one run, and Claude wrote the hazy sentences knowing the answers, so real memories
  may be harder. 96 of the 654 sketches have no dialogue, so regex cannot find them at all.

**The story this supports:** regex plus a good model goes surprisingly far (this is how coding agents search). It leans
on the model's knowledge and costs many calls per question. The RAG lectures then ask: can retrieval make the cheap
model work too? **That is a question for the RAG practical. We have not measured it.**

## ArmBench against our own test (24 Nov)

The same 12 hazy questions, regex tools only, one run each. Rank is the ArmBench-LLM leaderboard rebuilt from its result files.

| Model | ArmBench rank (score) | Found | Cost of the 12 | Tool calls per question |
|---|---|---|---|---|
| GPT-6.1 Sol | #1 (0.651) | **9/12** | $0.178 | 6.7 |
| DeepSeek V4.1 Flash | #2 (0.629) | 5/12 | $0.058 | 15.6 |
| Gemini 3 Flash | #5 (0.595) | 5/12 | $0.158 | 7.4 |
| GPT-6.1 Terra | not on the board | **9/12** | $0.090 | 7.6 |
| gpt-4o-mini | not on the board | 1/12 | $0.011 | 6.6 |

- Gemma 4 31B (#7) did not finish: one provider was too slow, the next returned a rate-limit error.
- **ArmBench's top three carry "rounded task scores supplied" in their files, not full evaluation logs.** That includes Sol.
- ArmBench has no search or tool-use task, so it cannot rank models on this job directly.
- **12 questions cannot rank close models.** 95% intervals: 9/12 is 47-91%, 5/12 is 19-68%. Fisher's test of 9/12
  against 5/12 gives p = 0.21. The lesson can say "a benchmark shortlists, your own test decides", and must not say
  "Terra beats Gemini".

**The final test (Q62):** 10 fresh hazy questions, not used for tuning the prompt, one run, $0.42.

| Model | ArmBench | Found | Named a wrong sketch | Cost per question | Tool calls per question |
|---|---|---|---|---|---|
| GPT-6.1 Sol (class model, Q63) | #1 | **7/10** | 1 | $0.016 | 6.8 |
| GPT-6.1 Terra | not listed | **6/10** | **4** | $0.010 | 9.6 |
| DeepSeek V4.1 Flash | #2 | 5/10 | 0 | $0.006 | 14.6 |
| Gemini 3 Flash | #5 | 3/10 | 0 | $0.009 | 8.5 |
| gpt-4o-mini | not listed | 0/10 | 4 | $0.001 | 8.7 |

## The session, about 105 minutes

| # | Part | Min | What happens on screen |
|---|---|---|---|
| 0 | The project | 5 | The `sketch_finder/` folder, `pyproject.toml`, what each module does. |
| 1 | The hook | 4 | "The one where a lady lies to get a seat on the minibus." The model alone cannot identify it. |
| 2 | The data | 7 | 702 rows, 48 duplicates, 96 without dialogue. Messy actor names (`Մկո Հասմիկ` with a missing comma). |
| 3 | A read-only database | 7 | `database.py`: three tables, then `DROP TABLE` refused. |
| 4 | Tool 1 and the loop | 10 | `tools.run_sql`, the description the model sees, the system prompt, `agent.run_agent` with every step printed. |
| 5 | Tools 2, 3: search and read | 15 | Exact word against word stem on Armenian endings, then the minibus question found by regex. |
| 6 | Tool 4: `web_search` | 5 | A question the database cannot answer. The agent goes outside. |
| 7 | Break it, fix it, test it | 10 | Commas inside a regex, a tool call cut off mid-JSON, running out of steps; then the 23 tests run. |
| 8 | Why benchmarks matter | 10 | ArmBench-LLM: what it measures, how to read it, and what it cannot tell us about this job. |
| 9 | Our own benchmark | 12 | 10 easy questions live; the saved five-model table on 10 hazy ones, with error bars. |
| 10 | The command line | 5 | `uv run sketch-finder ask ...` and `sketch-finder cost` (the ledger). |
| 11 | MCP server | 12 | `server.py`, an in-memory client, the model through MCP, the Claude Code command. |
| 12 | Handover to RAG | 4 | Which questions failed and what each cost. "Can retrieval fix the cheap model?" |

Homework (optional): write three hazy descriptions and test them; add a `sketches_with(actor_a, actor_b)` tool to the
server and check that Claude Code picks it.

## How it gets built

- `sketch_finder/`: the uv project. `uv sync`, `uv run pytest`, `uv run sketch-finder ask|eval|serve|cost` (its README).
- `py_src/build_kargin_agent_nb.py` writes and runs `07_kargin_agent_solution.ipynb` and its HTML. The notebook imports
  the package and prints the real source of each function. Direct API calls, no record/replay layer (#116).
- A frozen copy of the sketch data is in `sketch_finder/data/`, so nothing depends on the Kargin repo changing.
- The question set is `sketch_finder/data/questions.json`, because the RAG practical reruns the same file.
- Questions are checked for class: several sketches have adult themes, so the instructor picks what goes on screen.

## Renames this caused (Q53, done 24 Nov)

| Now | After |
|---|---|
| `llms/04_kargin_agent/` (new) | video 07 |
| `llms/04_rag/`, decks 07, 08, 09 | `llms/05_rag/`, decks 08, 09, 10 |
| `llms/05_embeddings_position` ... `16_anatomy_2026` | `06_` ... `17_` |
| Plan: RAG practical 10, vision 11, voice 12, shipping 13 | 11, 12, 13, 14 |

## Open

- The instructor's check of the 20 questions (`07_kargin_questions_REVIEW.html`). Changing a hazy question means
  rerunning the saved five-model comparison for it (about $0.04 per question for all five models).
- The server was tested through `uv run ... sketch-finder serve` with an MCP client. It has not been registered inside
  Claude Code yet.
