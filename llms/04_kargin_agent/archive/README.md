# archive

One-off scripts kept as a record. Not imported by anything and not expected to keep running.

## `run_model_comparison.py`

Produced `sketch_finder/results/hazy_comparison_2026-11-24.json` (five models, 10 hazy questions, $0.42) with the old
single-file `kargin_agent.py`, before the code became the `sketch_finder` package. Moved here on 2026-11-24 (#122):
`uv run sketch-finder eval --kinds hazy --tools regex --models ...` does the same job now, with the same loop.

## `2026-11-24_regex_premise/`

Tested whether a regex-only agent fails on vague descriptions of Kargin sketches, before the lesson was outlined.
Results are in `07_kargin_agent_OUTLINE.md` and DECISIONS #121. Total spend about $0.63.

- `kargin_proto.py`: builds a prototype SQLite database and runs 4 SQL, 4 exact-quote and 12 vague questions on
  gpt-4o-mini. Its scoring compared IDs as strings, so `074` against `74` counted as wrong. The real scores are SQL 4/4,
  quotes 4/4, vague 2/12. Re-run only to see that first naive version.
- `kargin_proto2.py`: the same 12 vague questions with a fair prompt and a tool that ANDs several patterns, on three models.
- `kargin_proto3.py`: the same 12 sketches described in one hazy sentence, on gpt-4o-mini and GPT-6.1 Terra.
- `kargin_proto4.py`: the hazy questions on ArmBench's affordable top models. DeepSeek V4.1 Flash and Gemini 3 Flash
  finished. Gemma 4 31B did not: Parasail took over a minute per question (2 of 3 right before it was stopped, kept in
  `kargin_proto4_parasail_partial.json`) and Crusoe returned a 429.
- `kargin_proto5.py`: the same questions on ArmBench's #1, GPT-6.1 Sol.
- `armbench_table.py`, `armbench_table.json`: the ArmBench-LLM leaderboard rebuilt from the Space's result files on
  24 Nov, by the scoring its README describes.
- `*_results.json`: every question, answer, tool call and cost from those runs.
