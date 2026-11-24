# Test a lesson's central claim with a cheap probe before writing the outline

**Symptom.** Video 07 was pitched on a claim: "regex fails on vague memories of a sketch, so we need RAG". It sounded
obvious and it was going to be the spine of the lesson and the handover to the RAG lectures.

**What the probe showed** (12 sketches, regex tools only, one run each, about $0.23 in total):

| Question type | gpt-4o-mini | gpt-4.1-mini | GPT-6.1 Terra |
|---|---|---|---|
| Described from the plot summary | 3/12 | 5/12 | 11/12 |
| One hazy sentence | 1/12 | not run | 9/12 |

A strong model guesses what the characters would actually say (slang such as `գայիշնիկ`, `շտրոֆ`, `զոքանչ`) and
regex then finds it. The claim held only for cheap models.

**Two traps on the way.**

- The first run scored 2/12, but most misses were bad queries (`"օլիգարխ, մահ, խնձոր"`: commas inside a regex), not a
  limit of regex. A fair prompt and a tool that ANDs patterns were needed before the result meant anything.
- The scoring compared sketch IDs as strings, so `074` against `74` counted as wrong. Compare numbers as numbers.

**Consequences.** The lesson's story changed to "regex plus a good model goes far, at about 7-10 tool calls a
question; a benchmark shortlists, your own test decides", and the handover to RAG became a concrete miss (q18: the
agent searched waking, sleep, grandpa, phone call; the sketch says none of these). Before outlining a lesson around a
claim about model behavior, spend a few cents testing it on a handful of cases with a fair setup, and on more than one
model.

Evidence: `llms/04_kargin_agent/archive/2026-11-24_regex_premise/` (scripts and every run as JSON), DECISIONS #121.
