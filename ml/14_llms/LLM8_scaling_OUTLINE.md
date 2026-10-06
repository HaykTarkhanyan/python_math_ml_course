# LLM-8 - Scaling laws and emergence - outline (2026-10-06)

Drafted 2026-10-06 from the critique in `LLM_CHAPTER_PLAN.md` (LLM-8) and the recommended answers
in `INTERVIEW_QUEUE.md` (instructor: "do all queue items"; LLM-8 kept separate, DECISIONS #73).
Session 8 of the LLM chapter. One question carries it: **you have a fixed compute budget - how big a
model, and how much data?** Then: what "suddenly appears" with scale, and whether it really is
sudden.

Deck file: `LLM8_scaling.tex`. Figures: `py_src/scaling_figs.py` - published fits and toy
numbers only, no model runs; every quoted number asserted.

**Decisions (queue answers, 2026-10-06):** kept as its own session; cold open Chinchilla vs Gopher;
the mirage toy (p smooth, p^5 jumps); one grokking frame.

**Facts checked 2026-10-06:**

| Fact | Source |
|---|---|
| Gopher 280B on 300B tokens; Chinchilla 70B on 1.4T, the same compute budget; MMLU 67.5% vs 60.0% ("greater than a 7% improvement") | Hoffmann et al. 2022 abstract (arXiv 2203.15556); local `sources/llm_training/09_scaling_laws` |
| "for every doubling of model size the number of training tokens should also be doubled" | same abstract |
| L(N, D) = E + A/N^alpha + B/D^beta; Hoffmann's published fit E 1.69, A 406.4, B 410.7, alpha 0.34, beta 0.28 | Hoffmann et al. eq. 10; the replication's table |
| The replication (Besiroglu et al. 2024): the published fit is inconsistent with Hoffmann's own approaches 1 and 2; refitted E 1.8172, A 482.01, B 2085.43, alpha 0.3478, beta 0.3658, consistent with ~20 tokens per parameter | arXiv 2404.10102 (abstract and the parameter table) |
| Measured here: the published fit's optimum at 10^21 FLOPs is 50 tokens/parameter (93 at Chinchilla's budget); the refit's is 21.6 (18.4) | `scaling_figs.py`, grid search |
| Kaplan et al. 2020: loss is a power law in N, D, C with exponents about 0.076, 0.095, 0.050; C ~ 6ND | Kaplan et al. 2020; local `sources/dl4nlp/13`, `sources/llm_training/09` |
| Mixed-precision Adam: 16 bytes per parameter (2 weights + 2 gradients + 4 fp32 master + 4 + 4 moments), before activations | ZeRO (Rajbhandari et al. 2020); local `sources/dl4nlp/13` |
| GPT-3 175B on 300B tokens; Llama 3 8B on over 15T tokens | GPT-3 paper; Llama 3 (local `sources/llm_training/07_llama3`) |
| Emergent abilities: Wei et al. 2022; the mirage: Schaeffer et al. 2023 (NeurIPS 2023 outstanding paper) - with a continuous metric the jumps become smooth | the papers; local `sources/dl4nlp/17` |
| Grokking (Power et al. 2022): division mod 97, 50% training data - training accuracy near perfect before 10^3 steps, little generalisation until 10^5, validation near perfect at ~10^6 | `misc/grokking/papers/llm_readable/power_et_al_2022_grokking.md` |

---

## Frames

### Opening

| # | Frame | Content | Figure |
|---|---|---|---|
| 0 | Title | "Scaling laws and emergence" / "How big, how much data, and what appears" | - |
| 1-2 | Where we are | pass / life, Scale lit | `roadmap_scaling_*` |
| 3 | Cold open: same budget, two models | Gopher 280B on 300B tokens vs Chinchilla 70B on 1.4T, same compute. **Predict:** which is better? | - (two boxes) |
| 4 | The small one | MMLU 67.5% vs 60.0%. 6ND by hand for both (5.0 vs 5.9 x 10^23). "Gopher was too big for its data." | - |
| 5 | Outline | | - |

### 1. Loss is a power law

| 6 | transition | "Loss is a power law" / "Straight lines on a log-log plot." | - |
| 7 | Straight lines | Kaplan: loss vs N, D, C; exponents ~0.076 / 0.095 / 0.050. By hand: doubling N multiplies the reducible loss by 2^-0.076 = 0.949 (5% less). Small steps, but predictable ones: that is what lets a lab plan a $100M run from $10k runs. | `scal_powerlaw.pdf` (loss vs compute along the optimum, the refit) |
| 8 | The bill: C = 6ND | 2 FLOPs per parameter per token forward, 4 backward. By hand: Karpathy's GPT-2 run, 6 x 124M x 33B = 2.5 x 10^19. | - |
| 9 | The other bill: memory | 16 bytes per parameter with Adam (bars). GPT-2: 2.0 GB; 8B: 128 GB, more than one 80 GB GPU, before activations. LLM-13 comes back to this. | `scal_memory.pdf` |

### 2. How big, how much data

| 10 | transition | "How big, how much data" / "One budget, many ways to spend it." | - |
| 11 | One budget, many models | Iso-FLOP curves from the refit: for each budget, loss against model size is a valley. Too small: not enough capacity. Too big: not enough data. | `scal_isoflop.pdf` |
| 12 | By hand: 10^21 FLOPs | D = 20N and C = 6ND give C = 120 N^2: N = 2.9B, D = 58B. | - |
| 13 | How to read a claim: the fit | The paper's own published fit puts the optimum at 50 tokens/parameter at this budget (93 at Chinchilla's) - not 20. A 2024 replication refitted the same data and got ~20, consistent with the paper's other two methods. Lesson: check a fit against its own headline. | `scal_fit_check.pdf` (tokens/param of the optimum vs budget, two fits) |
| 14 | Over-training, on purpose | Tokens per parameter: GPT-3 1.7, Gopher 1.1, Chinchilla 20, Llama 3 8B ~1,875 (log bars). Training cost is paid once; inference (~2N FLOPs per token) every time: a smaller model trained far longer is cheaper to serve. | `scal_tokens_per_param.pdf` |

### 3. Emergence

| 15 | transition | "Emergence" / "Abilities that seem to switch on." | - |
| 16 | Emergent abilities | Wei et al. 2022: an ability is emergent if it is absent in smaller models and present in larger ones; the plots look like a switch. GPT-3: the gap between zero-shot and few-shot grows with size. | - |
| 17 | Predict: smooth or sudden? | Toy: per-token accuracy p rises smoothly with scale. The task is a 5-token answer, scored by exact match. **Predict** the exact-match curve. | - |
| 18 | The mirage, by hand | p = 0.5 -> 3%; 0.7 -> 17%; 0.9 -> 59%; 0.95 -> 77%. The figure: p smooth, p^5 "jumps". Schaeffer et al. 2023: switch to a continuous metric and many jumps vanish. | `scal_mirage.pdf` |
| 19 | Where the debate stands | Not settled: the metric explains many jumps, not necessarily all; a task that needs every step right is all-or-nothing for the user even if the model improved smoothly. What to take away: ask which metric made the jump. | - |
| 20 | Another kind of sudden: grokking | Power et al. 2022, the Welch Labs still: training accuracy perfect before 10^3 steps, test accuracy near chance until ~10^5, then near perfect. Sudden in training time, not in scale. | `fig/borrowed/welchlabs/grokking_curves.jpg` (Welch Labs credit) |
| 21 | A second axis: thinking longer | Teaser: reasoning models spend more compute at test time; accuracy improves with it. The craft is LLM-10's; the training of such models is the RL chapter's. | - |

### Close

| 22 | Recap | | - |
| 23 | Next on the map | LLM-9: from base model to assistant. | `roadmap_post_training_life.pdf` |
| 24 | Sources & further reading | | - |

**Self-review (2026-10-06):** (1) every number is a published one or a toy, labelled; no model runs.
(2) The fit frame must be fair: Hoffmann's headline (20 tokens/param, approaches 1-2) stands; only
the published parametric fit disagrees, and the replication explains why. (3) The mirage frame
must not overclaim: "many jumps vanish", not "emergence is fake". (4) Grokking is about training
time, not scale - said on the slide. (5) Cut: the nature analogies (ants, cells, phase transitions),
"three eras of compute allocation" (no source), the AND-gate frame.
