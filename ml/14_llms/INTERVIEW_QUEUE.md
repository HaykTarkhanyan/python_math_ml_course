# Interview queue - the LLM chapter

Prepared 2026-10-05 while the instructor was away ("start criticizing the rest of the slide sets and
plan how we'll build together with interview questions; for the stuff that's a no-brainer, do it
yourself"). Each section is asked through the interview tool, one deck at a time, in build order.
Recommendations are marked; "R:" is the reason in one line. The critiques behind each question are
in `LLM_CHAPTER_PLAN.md` (the deck's "Critique (2026-10-05)" block).

**How we build each deck (same loop as LLM-3 and LLM-4):** these questions -> outline (with a
quick self-review) -> build (figure script with asserted numbers, deck, compile, detectors,
self-review) -> student review (one Sonnet agent, asked first) -> commit. Toy numbers by default;
real numbers only for claims about real models, from local files or the cached GPT-2.

**Proposed build order:** LLM-5 (small restructure) -> LLM-7 -> LLM-8 (or merged into LLM-7) ->
LLM-9 -> LLM-10 -> LLM-11 -> (RAG, agents: built) -> LLM-12 -> LLM-13 -> LLM-14 (if D1 keeps it).

---

## 0. Before anything else

1. **Commit what is done?** Uncommitted: the LLM-4 restructure (deck, `block_figs.py`, 12 figures,
   outline), LLM-3's two re-pointed callbacks, LLM-5's no-brainer fixes, and the plan's critiques
   plus this file. Proposed: one commit for LLM-4, one for the rest. (Two earlier commits of the
   day are in: `ca3e2a3` renumber, `8fbd261` LLM-3.)
2. **Student review for LLM-4?** One Sonnet agent on the rendered pages, about 100k tokens.
   R: same step LLM-3 got; it found the query-axis contradiction there.
3. **Push?** Nothing is pushed.

## LLM-5 Transformers in the world (built in July; restructure proposed)

**All five done 2026-10-06 with the recommended options** (instructor: "do all queue items"): kept
as its own session; "How it learns" cut; GPT-2's LM head by hand (pipeline, top 8, one logit, "only
differences matter"); the three-mask cold open; one ending. The three July figures redrawn at slot
size (`py_src/head_figs.py`). Student review (one Sonnet agent) applied - see the deck's provenance.

1. **Keep LLM-5 as its own session, or merge?** Options: (a) keep, restructure lightly
   **(recommended** - R: the family, cross-attention and the cost frames are a session's worth);
   (b) merge the family and the cost into LLM-4 and the LM head into LLM-6 (buys back a session).
2. **"How it learns" (the objective; "autocomplete is not a chatbot"):** (a) cut here - LLM-4 has
   the mask point, LLM-7 the objective, LLM-9 the chatbot point **(recommended** - R: three homes
   already); (b) keep here as the first mention, later decks call back.
3. **The LM head by hand on GPT-2** (final norm, dot product with the 50,257 rows of the tied
   table, softmax, the map's top-5): yes **(recommended** - R: it is the session's map box and gets
   one frame today) / no.
4. **Cold open:** (a) the three masks side by side, "which of these can write text?"
   **(recommended)**; (b) keep "the same block, three different jobs" as it is.
5. **Endings:** one (Next on the map) **(recommended**, as LLM-4) / keep both.

## LLM-6 Decoding (built 2026-10-05) - leftovers from its student review

**All three done 2026-10-06 with the recommended options:** the drafts frame kept; "By hand: the
table a library precomputes" added (a toy grammar, nine toy tokens, every cell checked); no loop
mechanism slide.

1. **Cut the "where do drafts come from" frame?** yes / no **(recommended: keep** - R: it answers
   the obvious question about speculative decoding).
2. **A grammar state table for constrained decoding** (which tokens survive at each step of a tiny
   JSON grammar): add **(recommended** - R: the masking frame is the only abstract frame left) / no.
3. **A mechanism slide for self-reinforcing loops** (why a repeated phrase raises its own
   probability): add, citing Xu et al. 2022's measurement only / no **(recommended: no** - R: the
   mechanism is not settled; the measured effect is already on the slide).

## LLM-7 Pretraining (critique written; this is the interview that was interrupted)

**All five done 2026-10-06** (`LLM7_pretraining.tex`, 35 pages; DECISIONS #76): data-led spine; the
cold open is GPT-2 reciting the MIT licence (34 of 34 words) and not the Gettysburg Address (0);
real pieces - no recommendation, all four taken (the recitation run, the Armenian share, FineWeb's
funnel, Karpathy's samples); GPT-1 to GPT-3 in one frame; no hands-on. Student review applied.

1. **What should it lead with?** (a) data-led spine - objective family in one mask figure (with
   word2vec's two frames) -> one training step by hand -> the data funnel -> what comes out
   **(recommended)**; (b) history-led, as the July draft; (c) data only, history as an appendix.
2. **Cold open:** (a) GPT-2 recites a much-copied text verbatim (if one local run shows it does)
   **(recommended** - R: LLM-4 already used facts; verbatim memorisation is new); (b) "what share
   of the web is Armenian?" (one published number); (c) "15T tokens is N reading lifetimes";
   (d) a raw web page before cleaning.
3. **Real pieces (multi-select):** the memorisation run; the Armenian share (FineWeb-2 / Common
   Crawl tables); FineWeb's funnel numbers; samples during training (Karpathy's notes).
4. **GPT-1 -> GPT-2 -> GPT-3:** one frame here, few-shot scaling in LLM-8 **(recommended)** / hand
   it all to LLM-8 and LLM-10 / a section here.
5. **Hands-on:** not in the slides (as LLM-1 and LLM-6) **(recommended)** / a perplexity exercise.

## LLM-8 Scaling laws and emergence

**All four done 2026-10-06** (`LLM8_scaling.tex`, 25 pages): kept separate; Chinchilla vs Gopher
cold open; the mirage toy; one grokking frame (the Welch Labs still). New on the way: the
Chinchilla paper's published fit puts the optimum at 50-93 tokens/parameter, not 20; a frame
shows it with the 2024 replication's refit. Student review applied.

1. **Merge into LLM-7?** (a) keep separate **(recommended** if D1 stays at 13 - R: the budget
   arithmetic and the mirage toy need room); (b) merge (the short-option plan).
2. **Cold open:** (a) Chinchilla vs Gopher - same compute, 4x smaller model, better
   (published) **(recommended)**; (b) "you have 10^21 FLOPs - how big a model?" as a predict.
3. **The mirage toy** (per-token accuracy p smooth, exact match p^5 "jumps"): yes
   **(recommended)** / no.
4. **Grokking:** one frame ("another kind of sudden", Welch Labs stills) / cut **(recommended:
   one frame)**.

## LLM-9 Post-training

**All four done 2026-10-06** (`LLM9_post_training.tex`, 24 pages): the imitate / compare / check
spine; the GPT-2 password cold open (bare: an FAQ page in a loop; in a User / Assistant frame:
"I'm sorry, but I don't have a password", and it writes the next turn itself); Bradley-Terry by
hand (0.818); reasoning models as a teaser. No instruct model is cached, so none is shown
(DECISIONS #78; a download is the instructor's call). Student review applied.

1. **Spine:** imitate (SFT, distillation) / compare (preferences: RLHF, DPO) / check (verifiable
   rewards) **(recommended)**, or the paper-by-paper order of the draft.
2. **Cold open:** one run of the cached GPT-2 on "How do I reset my password?" (what a base model
   really writes) **(recommended)** / a published InstructGPT example.
3. **Bradley-Terry by hand** (rewards 2.0 vs 0.5 -> 0.82 preferred): yes **(recommended)** / no.
4. **Reasoning models:** a one-frame teaser (L32g owns them) **(recommended)** / a short section.

## LLM-10 Prompting

**All four done 2026-10-06** (`LLM10_prompting.tex`, 23 pages): "the prompt is the context";
few-shot fragility measured on the cached GPT-2 (24 orderings x 48 reviews written for the deck +
zero-shot, 450 s): 50% to 85%, zero-shot 69%, and the reverse of Zhao et al.'s recency bias; all of
prompt injection; chain of thought as "tokens are compute". Student review applied.

1. **Spine: "the prompt is the context"** (what goes in, in what order, what an attacker puts in)
   **(recommended)**, or the technique list.
2. **Measure few-shot fragility on the cached GPT-2** (24 orderings x about 50 test sentences, a
   few minutes of CPU): yes **(recommended)** / published numbers only.
3. **Prompt injection** (L44 has none, checked 2026-10-05): all of it here, direct and indirect
   **(recommended)** / here the idea, plus one indirect-injection frame added to L44's agent loop.
4. **Chain of thought:** the why (tokens are compute) plus one line on reasoning models
   **(recommended)** / a full section.

## LLM-11 Evaluation and hallucination

1. **Spine: "how do you know an answer is right?"** (reference, judge, test set, confidently
   wrong) **(recommended)**, or metrics first, hallucination second.
2. **Sampling-consistency demo** on LLM-4's prompts (Messi vs Mkhitaryan, ten samples at T = 1,
   cached GPT-2): yes **(recommended)** / no.
3. **LLM-as-judge live?** Needs a paid API (cost stated first) / published bias numbers only
   **(recommended)**.
4. **Kalai et al. 2025's guess-vs-abstain toy table:** yes **(recommended)** / no.

## RAG (L41-L43) and agents (L44) - built in August

1. ~~The layout pass~~ - **done 2026-10-05/06** (all four decks at zero on every detector).
2. ~~L42's keyword-extraction stretch~~ - **done 2026-10-06:** YAKE's four frames and KeyBERT's
   output frame moved, unchanged, to a backup appendix after the recap; 10 frames stay in the flow
   (problem, three families, RAKE worked, KeyBERT's method, the LLM, predict, measured, why,
   takeaways).
3. **Judges:** LLM-11 teaches LLM-as-judge in general and L43 keeps only the RAG-specific frames
   **(recommended)** / L43 stays the judge lecture and LLM-11 calls forward. *Deferred on
   2026-10-06: do it when LLM-11 is built - cutting L43's general judge frames now leaves them
   no home.*
4. ~~An indirect-injection frame in L44~~ - **done 2026-10-06:** "The loop reads text it did not
   write", after the guards (a planted note in a `search_incidents` result; the defences, with the
   MCP specification's own advice).

*From the student reviews of 2026-10-06 - all done the same day ("do all"), recommended options:*

5. ~~L41 by-hand BM25 term~~ - **done:** "One chunk, scored by hand" (D1 = 4.203, every printed
   number checked by `l41_bm25_figs.py`).
6. ~~L42 reranker's gain~~ - **done, text option:** the cost frame says the gain is not measured,
   and why (embeddings already put 14 of 16 answers first; checked by `l42_retrieval_figs.py`).
   Measuring a real cross-encoder stays open (a model choice and a download: yours).
7. ~~L42 hybrid winning~~ - **done differently, because the measurement said no:** on the 18 chunks
   the embedder found the right chunk first on 10 of 11 identifier questions (the 11th tied BM25),
   and still first against 19 near-identical part-number notes, by 0.009-0.018 cosine. New frame
   "We tried to make BM25 win" reports both (`l42_worked_extras.py`, checked).
8. ~~context precision / relevance~~ - **done:** the overview box, the original-form frame and the
   current-form frame now say how the two names relate.
9. ~~lost in the middle~~ - **done:** "Best chunk first, next best last".
10. ~~which number to trust~~ - **done:** the recap's closing box.
11. ~~L44 mechanisms~~ - **done:** "What actually travels" (MCP `tools/list`, `tools/call`; MCP
    schemas count like any schema) and "How the body gets in" (Agent Skills' three levels), both
    checked against the MCP specification 2025-06-18 and Anthropic's Agent Skills docs on
    2026-10-06.
12. ~~L44 loop trace~~ - **done:** "One run, turn by turn" (the Tbilisi question, schematic).
13. ~~Truncated y-axes~~ - **done for L41 p45 and L43 p38** (now from zero); L41's ATE2 chart kept.

## LLM-12 Quantization (includes D4)

1. **D4, the runtime** (a technology choice - yours): Ollama for first contact + llama.cpp tools
   for measuring (the plan's recommendation) / llama.cpp only / LM Studio / decide later.
2. **The measured ladder:** (a) quantize GPT-2's own weights in NumPy and measure perplexity with
   the cached model - no downloads **(recommended** for the slides); (b) a real GGUF ladder on a
   small model (downloads of gigabytes, minutes of CPU; needs your OK); (c) both - (a) in the
   slides, (b) as the homework.
3. **Homework:** the ladder on their own laptops (as planned) / skip.

## LLM-13 LoRA and QLoRA (includes D5, D6)

1. **D5, the library** (yours): PEFT + TRL / Unsloth / teach library-free in the slides, choose
   for the practical later.
2. **A local LoRA demo on the cached GPT-2** (a few hundred CPU steps, minutes): yes / no
   (Colab practical only). R for yes: it makes "B = 0, so step 0 is the base model" visible.
3. **D6, the practical's base model:** measure tokens per Armenian word and perplexity for 3-4
   small models first (the plan's recommendation; downloads, asked first) / name one now.

## LLM-14 Anatomy of a 2026 LLM (includes D1)

1. **D1 - run LLM-14?** (a) yes, two sessions; (b) yes, one session; (c) no, and move one MoE
   frame to LLM-4 and one KV-cache frame to LLM-12 **(recommended** if the course end date is
   fixed - R: section 8 says something gets cut).
2. **If it runs:** each technique with its bill by hand from one real config (recommended) / a
   landscape survey.

## Chapter-level

1. **D1 overall:** 13 sessions (LLM-1 to LLM-13) + RAG/agents, or the 10-session short option
   (merges: LLM-7+8, LLM-10+11, LLM-12+13 lighter). Needs the course end date.
2. **Practicals:** P1 (a tiny Armenian GPT, after LLM-7/8) and P2 (QLoRA on Colab, after LLM-13) -
   keep both / one / later.
3. **Quarto:** none of LLM-1 to LLM-6 is linked from a qmd yet; `attention.qmd` covers only
   LLM-3 to LLM-5. Do the qmd pass now or at delivery?
