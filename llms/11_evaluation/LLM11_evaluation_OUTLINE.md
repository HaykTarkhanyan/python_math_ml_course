# LLM-11 - Evaluation and hallucination - outline (2026-10-06)

Drafted 2026-10-06 from the critique in `LLM_CHAPTER_PLAN.md` (LLM-11) and the recommended answers
in `INTERVIEW_QUEUE.md` (instructor: "do all queue items"). Session 11. One question: **how do you
know an answer is right?** Four ways, each with its failure: compare with a reference (word
overlap), ask another model (judges and their biases), use a test set (benchmarks and
contamination), and the case where the model is confidently wrong (hallucination).

Deck file: `LLM11_evaluation.tex`. Figures: `py_src/eval_figs.py` (one `--gpt2` sampling run on
LLM-4's two prompts; toys and published numbers asserted).

**Decisions (queue answers, 2026-10-06):** the four-ways spine; the sampling-consistency demo on
LLM-4's Messi / Mkhitaryan prompts (cached GPT-2, T = 1); published judge-bias findings only, no
paid API; Kalai et al.'s guess-vs-abstain toy table. **RAG item 3 (judges):** LLM-11 teaches
LLM-as-judge in general and takes L43's general frame "How to use a judge without lying to
yourself"; L43 keeps its two RAG-specific judge frames (what Ragas needs a judge for; the Ragas
agreement numbers) and points back.

**Facts checked 2026-10-06:**

| Fact | Source |
|---|---|
| Judges: "position, verbosity, and self-enhancement biases, as well as limited reasoning ability"; strong judges "over 80% agreement" with humans, "the same level of agreement between humans" | Zheng et al. 2023 abstract (arXiv 2306.05685) |
| Hallucinations persist because training and evaluation reward guessing; binary grading; fix the scoring of existing benchmarks | Kalai et al. 2025 abstract (arXiv 2509.04664) |
| "Answer only if you are > t confident, since mistakes are penalized t/(1-t) points, while correct answers receive 1 point, and an answer of 'I don't know' receives 0 points"; t = 0 is binary grading | Kalai et al. 2025, section on explicit confidence targets (HTML v1). The same paragraph lists "t = 0.75 (penalty 2)", which its own formula makes 3 - the slide uses the formula and says so |
| Armenian corpora: 17.4% (FineWeb-2), 10.9% (HPLT v2), 7.9% (CulturaX), 3.3% (ArmWeb) of documents share a 13-gram with ten Armenian evaluation sets; hand-built multiple-choice sets <= 0.06%; FineWeb-2's Armenian test split leaks into its train split | arXiv 2609.03350 (full text) |
| GPT-2 on "Lionel Messi plays the sport of": football 35.1% top; "Henrikh Mkhitaryan...": hockey 9.8% top, football 4.9% | LLM-4 (`block_figs.py`, results) - re-measured here |

---

## Frames

| # | Frame | Content | Figure |
|---|---|---|---|
| 0 | Title | "Evaluation and hallucination" / "How do you know an answer is right?" | - |
| 1-2 | Where we are | Evaluate lit | `roadmap_evaluation_*` |
| 3 | Cold open: ask twice | LLM-4's two prompts, 20 samples each at T = 1. **Predict:** which one gives the same answer every time? | - |
| 4 | Agreement is a signal | The real samples; the stored fact repeats, the missing one scatters (SelfCheckGPT, semantic entropy). | `eval_samples.pdf` |
| 5 | Outline | | |
| 6 | [plain] Compare with a reference | | |
| 7 | Word overlap by hand | Reference "the cat sat on the mat"; three candidates; unigram and bigram precision (BLEU / ROUGE's core). The nonsense sentence wins. | - (table) |
| 8 | What overlap is for | Translation and summaries with references; fails for open answers. Embedding similarity (LLM-2) helps a little. | - |
| 9 | [plain] Ask another model | | |
| 10 | LLM-as-judge | Pointwise vs pairwise; why (no formula decides "is this a good answer"); agreement with humans over 80% (Zheng 2023). | - |
| 11 | Its biases | Position, verbosity, self-enhancement (Zheng 2023). | - |
| 12 | How to use a judge without lying to yourself | Moved from L43: decompose, calibrate, compare, freeze, watch the biases; never the generator's own prompt. | - |
| 13 | [plain] Use a test set | | |
| 14 | Benchmarks | What they measure (knowledge, reasoning, code, agents); a number is a sample of tasks; leaderboards and Goodhart (LLM-9). | - |
| 15 | Contamination | LLM-7's memorisation again: a test item seen in training is recall, not skill. The Armenian numbers. | `eval_contamination.pdf` |
| 16 | [plain] Confidently wrong | | |
| 17 | Hallucination | Factuality vs faithfulness (to a source); why: the next-token objective has no "I don't know" (LLM-7), and... | - |
| 18 | Guess or abstain, by hand | Kalai: under 0/1 grading guessing always pays; with penalty t/(1-t) (t = 0.75: 3) it pays only above 75%. Table + plot. | `eval_guess.pdf` |
| 19 | Catching it | Sampling agreement (the cold open: SelfCheckGPT), semantic entropy (meaning-level), and grounding in sources -> RAG. | - |
| 20 | Recap | | |
| 21 | Next on the map | -> RAG (L41): give the model the documents. | `roadmap_quantization_life.pdf`? no - the RAG chapter is next in teaching order; use the overview map and say so |
| 22 | Sources | | |
