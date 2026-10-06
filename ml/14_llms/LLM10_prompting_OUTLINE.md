# LLM-10 - Prompting: the prompt is the context - outline (2026-10-06)

Drafted 2026-10-06 from the critique in `LLM_CHAPTER_PLAN.md` (LLM-10) and the recommended answers
in `INTERVIEW_QUEUE.md` (instructor: "do all queue items"). Session 10. One idea carries it: **the
model computes p(next token | everything in the window)** (LLM-6), so a prompt is not a command
channel - it is the context being continued. What you put in, in what order, and what an attacker
puts in.

Deck file: `LLM10_prompting.tex`. Figures: `py_src/prompting_figs.py` - one `--gpt2` measurement
(24 orderings of 4 examples x 48 test reviews + zero-shot = 1,200 forward passes of the cached
GPT-2, ~6 min on 4 threads, run in the background; approved through the queue's recommended option).

**Decisions (queue answers, 2026-10-06):** the "prompt is the context" spine; measure few-shot
fragility on GPT-2; all of prompt injection here, direct and indirect (L44 now has one frame and
points back); chain of thought = tokens are compute, plus one line on reasoning models.

**Test data.** No sentiment dataset is cached (only wikitext), so the 4 demonstrations and 48 test
reviews (24 positive, 24 negative) were written for the deck, short and unambiguous. The measured
fragility is a real property of GPT-2 on them; the slide says so.

**Facts checked 2026-10-06:**

| Fact | Source |
|---|---|
| Few-shot accuracy can vary "from near chance to near state-of-the-art" with format, examples and order; a bias towards answers "placed near the end of the prompt or ... common in the pre-training data"; GPT-2 and GPT-3 | Zhao et al. 2021 abstract (arXiv 2102.09690) |
| "Let's think step by step": MultiArith 17.7% -> 78.7%, GSM8K 10.4% -> 40.7% (InstructGPT text-davinci-002) | Kojima et al. 2022 abstract (arXiv 2205.11916) |
| Reasoning models: "Avoid chain-of-thought prompts ... 'think step by step' ... is unnecessary"; keep prompts simple; use delimiters | OpenAI reasoning best practices (developers.openai.com, fetched) |
| Indirect prompt injection: instructions planted in "data likely to be retrieved"; applications "blur the line between data and instructions"; Bing Chat, code completion | Greshake et al. 2023 abstract (arXiv 2302.12173) |
| OWASP LLM01:2025 Prompt Injection: direct and indirect; seven mitigations (constrain behaviour, define output formats, filter, least privilege, human approval, segregate external content, adversarial testing); "unclear if there are fool-proof methods of prevention" | genai.owasp.org (fetched) |
| Chain of thought (few-shot): Wei et al. 2022; self-consistency: Wang et al. 2022 | the papers; local `sources/dl4nlp/08` |

---

## Frames

| # | Frame | Content | Figure |
|---|---|---|---|
| 0 | Title | "Prompting" / "The prompt is the context" | - |
| 1-2 | Where we are | Prompt lit | `roadmap_prompting_*` |
| 3 | Cold open | Four labelled reviews, then a new one. **Predict:** does the ORDER of the same four examples change GPT-2's accuracy? | - (the prompt shown) |
| 4 | The answer | The 24 orderings, sorted (real). | `prompt_orderings.pdf` |
| 5 | Outline | | |
| 6 | [plain] What goes in | | |
| 7 | The prompt is the context | p(next | window), callback LLM-6/7/9: instructions, examples, documents, history, tool results - all just text in one stream. | - |
| 8 | Zero-shot, few-shot | Definitions; GPT-3; our setup (the format). Zero-shot GPT-2 vs few-shot. | - |
| 9 | Why order matters | Recency: share labelled positive, by the last example's label (real). Zhao 2021's biases and calibration. What helps. | `prompt_recency.pdf` |
| 10 | Instructions and system prompts | After LLM-9 the model follows instructions; the system prompt is the first turn of the template - text, not a privilege. What a good one says. | - |
| 11 | Ask for a format | Delimiters, a stated format; a prompt only asks, constrained decoding (LLM-6) guarantees. Where in a long prompt matters: L43. | - |
| 12 | [plain] Room to think | | |
| 13 | Chain of thought | Few-shot CoT (Wei 2022); zero-shot "Let's think step by step" (Kojima 2022: 17.7 -> 78.7%, 10.4 -> 40.7%). | - |
| 14 | Tokens are compute (by hand) | 12 layers per token: answer at once = 12 serial layer passes; 40 reasoning tokens first = 492. Each written token is read by every later step. Self-consistency: several chains, majority vote. | `prompt_cot_steps.pdf` |
| 15 | Reasoning models changed the advice | They reason internally; OpenAI's guidance: avoid CoT prompts, keep it simple, use delimiters. | - |
| 16 | [plain] What an attacker puts in | | |
| 17 | Direct injection | The system prompt and the user's text are one token stream; a toy translator app. | - |
| 18 | Indirect injection | The attacker never talks to the model: a planted instruction in a page, email or document (Greshake 2023); RAG and agents (L44). | - |
| 19 | What helps, and what does not | OWASP LLM01:2025's mitigations; no fool-proof prevention; "ignore instructions in documents" is itself text. | - |
| 20 | Recap | | |
| 21 | Next on the map | LLM-11: how do you know the answer is right? | `roadmap_evaluation_life.pdf` |
| 22 | Sources | | |

**Self-review:** (1) the measurement is GPT-2 on our 48 reviews - not a general accuracy claim.
(2) Kojima's numbers are for text-davinci-002, said on the slide. (3) The CoT step count is a
lower bound on serial work and a toy; "tokens are compute" is the mechanism, not a full theory.
(4) Injection examples are toy texts, no real product named as vulnerable beyond Greshake's
published targets. (5) Lost-in-the-middle stays in L43; ReAct in L44.
