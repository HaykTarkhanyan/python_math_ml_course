# non_essential - build-time probes for the tools practical

Not needed to build or execute the notebooks; they produce measurements the notebooks and slides quote.

| Script | What it does, and when to re-run it |
|---|---|
| `model_test.py` | Five candidate models, the same stub tools and agent loop: one tool call (Ararat), five known-answer questions, one multi-step question, and a prompt injection planted in a search result (3 phrasings x 2 runs). Writes `results/model_test.json`. Re-run before teaching, or when the model list changes. ~3-5 min, ~$0.10. Added 2026-11-12. |
