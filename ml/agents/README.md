# Agents and tool use

Seed material for a future agents chapter. **Nothing here is course-ready yet** - the folder
holds one imported deck, kept unmodified, plus notes on what it would take to turn it into
`ch18`.

Agents are the open gap flagged at the top of [`ml/MISSING_TOPICS.md`](../MISSING_TOPICS.md):
verified absent from `ml/**/*.tex`, and pointed at directly by `ch17_rag/L43`, which mentions
"agentic" twice as a teaser and never answers what the stopping condition actually is.

## What is here

| File | What it is |
|---|---|
| [`source/approaches.tex`](source/approaches.tex) | Beamer source, 33 frames |
| [`source/approaches.pdf`](source/approaches.pdf) | compiled, 37 pages, 16:9 |

**Provenance.** Copied 2026-08-11 from
`Desktop/metric/metric-internship-2026-washinton/_knowledge/slides/`, where it was written for
the Metric internship programme. Imported **unmodified**, so the diff against any future course
version stays legible. Title: *"Connecting Tools and Knowledge to an LLM - Direct orchestration
vs Tool calling vs MCP vs Skills (Python + OpenRouter)"*.

## What it covers

Five sections, built as an escalating ladder over one running example:

1. **Direct orchestration** - you call the tool yourself and paste the result into the prompt.
2. **Tool calling** - the model asks for the call; your code executes it and returns the result.
3. **MCP** - the tool lives behind a protocol, so you can plug in servers you did not write.
4. **Skills (`SKILL.md`)** - instructions rather than executable tools.
5. **Choosing** - a side-by-side comparison and a decision guide.

Its strongest asset is that almost every section carries a **real transcript**: not just the
code, but the exact request and response, including what the model actually emits when it wants
a tool called. That is the part students cannot get from prose, and it is the same instinct as
`ch17_rag`'s "show the assembled prompt" frame.

## What would have to change to become `ch18`

- **Preamble.** Uses `\usetheme{Madrid}` with its own colour definitions. House style is
  `\documentclass[aspectratio=169]{beamer}` + `\input{../preamble}`, `default` theme, `dove`
  colours. The Armenian-flag palette is already there, so this is mostly deletion.
- **Structure.** No cold-open hook, and section dividers use `\tableofcontents[currentsection]`
  rather than the house `[plain]` transition slide (popblue title + one motivation line).
- **Ending.** Closes on *"Bridge to text-to-SQL"*, which is internship-specific and would be
  replaced by whatever follows in the course.
- **Code density.** Substantially more code per frame than `ml/SLIDE_STYLE.md` allows
  ("minimal - at most one canonical snippet per topic"). Deliberate for a hands-on internship
  session; needs a decision for a lecture.
- **Figures.** Diagrams are TikZ. House rule is that every essential figure is Python-generated,
  with TikZ only for small throwaway visuals.
- **The gap it does not fill.** This is the *tool-use* half. It has no agent **loop** - no
  planning, no observe-act-repeat, no stopping condition, no failure modes when the loop does
  not terminate. That is exactly the question `ch17_rag/L43`'s student review asked and could
  not answer, so it would need building from scratch.

## Related material already in the repo

- [`ml/ch17_rag/`](../ch17_rag/) - the natural predecessor. `rag_demo.py` is ready-made
  scaffolding: an agent loop is that script plus a tool registry and a stopping condition, and
  it keeps the cheese-factory running example.
- [`ml/llm_training/`](../llm_training/) - how the models being orchestrated here are trained.
- `misc/claude_code/slides/` - five decks on Claude Code, which overlap on skills and MCP.
