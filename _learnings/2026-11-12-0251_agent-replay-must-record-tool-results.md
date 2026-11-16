# A saved agent run must record the tool results, not only the model calls

**Symptom (predicted by the outline review, confirmed by design).** B0's pattern caches model calls by their
request. For an agent with live tools (weather, web search), the tool results change every hour, so the next request
differs, the cache misses, and the notebook makes live calls and prints different numbers than its prose quotes.

**What the tools practical does instead.** A small `Tape` per demo records every model reply **and** every tool
result in order, in `results/tools_1_run.json` / `tools_2_run.json`, and replays both. Cells meant to be live open a
tape with `live=True`, which calls for real and is never saved, so a class run cannot overwrite the recording.
`RERUN = True` replaces the whole recording, and the setup cell says that the prose and slides quote the old one.

**Consequences.**

- When the code of one demo changes, delete that demo's key from the run file so it re-records; the others replay.
- Prose quotes only saved values; text before a live cell stays conditional ("watch whether...").
- Commit the run files with the notebook: they are the artifact of record the text depends on.
