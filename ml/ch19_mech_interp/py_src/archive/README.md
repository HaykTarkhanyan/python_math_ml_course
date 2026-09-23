# ch19 `py_src/archive/`

One-off probes from the 2026-09 extension (`../../EXTENSION_PLAN.md`), kept as the record of
**why** a decision was made. They are not imported by anything, write to stdout rather than the
repo's `logs/`, and are not expected to be maintained. The figure scripts one level up are the
load-bearing ones.

| Script | What it found | Decision it drove |
|---|---|---|
| `probe_embedding_arithmetic_and_landmarks.py` | GPT-2 small's token embeddings do analogies (king - man + woman -> queen, cos 0.71) but it does **not** know landmarks: "The Eiffel Tower is in the city of" -> London 8.0%, Paris 6.9%. | The intro's fact hook moved off the Eiffel Tower; the "feature = direction" frame uses embedding arithmetic. Added 2026-09-22. |
| `probe_gpt2_fact_recall.py` | 48 fact prompts: people and companies are known (Steve Jobs -> Apple 83%, Federer -> tennis 66%), landmarks and capitals-from-country mostly are not. | Fact examples in the chapter use people, and the facts deck uses athlete -> sport. Added 2026-09-22. |
| `probe_athlete_prompts.py` | Token lengths and GPT-2's answer for 22 athletes in "{name} plays the sport of". "Tiger Woods" is 3 tokens at the start of a prompt, "Michael Jordan" and "Tom Brady" 2 each. | The easy patch in the causal-methods deck uses Jordan <- Brady (same length, both answered correctly). Added 2026-09-22. |
| `probe_attn_only_toy_models.py` | TransformerLens `attn-only-1l/2l`: config, previous-token and induction scores, full Q/K/V composition blocks. L0H3 previous-token (0.52), L1H6 induction (0.70), K-composition 0.124 the largest. | Gate G4 looked passable before `circuits_figs.py` was written. Added 2026-09-23. |

Re-running any of them needs the `ma` venv and downloads GPT-2 small or the toy models on first
use (CPU, under a minute of compute each).
