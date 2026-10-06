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

1. **Cut the "where do drafts come from" frame?** yes / no **(recommended: keep** - R: it answers
   the obvious question about speculative decoding).
2. **A grammar state table for constrained decoding** (which tokens survive at each step of a tiny
   JSON grammar): add **(recommended** - R: the masking frame is the only abstract frame left) / no.
3. **A mechanism slide for self-reinforcing loops** (why a repeated phrase raises its own
   probability): add, citing Xu et al. 2022's measurement only / no **(recommended: no** - R: the
   mechanism is not settled; the measured effect is already on the slide).

## LLM-7 Pretraining (critique written; this is the interview that was interrupted)

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

1. **Merge into LLM-7?** (a) keep separate **(recommended** if D1 stays at 13 - R: the budget
   arithmetic and the mirage toy need room); (b) merge (the short-option plan).
2. **Cold open:** (a) Chinchilla vs Gopher - same compute, 4x smaller model, better
   (published) **(recommended)**; (b) "you have 10^21 FLOPs - how big a model?" as a predict.
3. **The mirage toy** (per-token accuracy p smooth, exact match p^5 "jumps"): yes
   **(recommended)** / no.
4. **Grokking:** one frame ("another kind of sudden", Welch Labs stills) / cut **(recommended:
   one frame)**.

## LLM-9 Post-training

1. **Spine:** imitate (SFT, distillation) / compare (preferences: RLHF, DPO) / check (verifiable
   rewards) **(recommended)**, or the paper-by-paper order of the draft.
2. **Cold open:** one run of the cached GPT-2 on "How do I reset my password?" (what a base model
   really writes) **(recommended)** / a published InstructGPT example.
3. **Bradley-Terry by hand** (rewards 2.0 vs 0.5 -> 0.82 preferred): yes **(recommended)** / no.
4. **Reasoning models:** a one-frame teaser (L32g owns them) **(recommended)** / a short section.

## LLM-10 Prompting

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
2. **L42's keyword-extraction stretch** (RAKE, YAKE, KeyBERT, LLM keywords, about 14 frames, ending
   "not a clean win"): compress to a few frames **(recommended)** / keep as is. *Student review
   2026-10-06 agrees: "the main problem ... pick one, or a one-slide comparison".*
3. **Judges:** LLM-11 teaches LLM-as-judge in general and L43 keeps only the RAG-specific frames
   **(recommended)** / L43 stays the judge lecture and LLM-11 calls forward.
4. **An indirect-injection frame in L44's agent loop** (the agent reads untrusted tool output):
   yes / no - pairs with LLM-10's question 3. *The L44 student review asked for it unprompted
   ("a security slide ... prompt injection from a tool result").*

*From the student reviews of 2026-10-06 (one Sonnet reader per deck, PNGs only; verified
findings and figure fixes already applied - these are the ones that change content):*

5. **L41 p27, the BM25 formula arrives assembled** (the reviewer's "stop here for five minutes"):
   add one by-hand term, D1 "press": tf 1, B 0.70, 1/(1.5 x 0.70 + 1) = 0.49, x idf 2.25 = 1.10
   **(recommended**, the by-hand rule) / leave to the board.
6. **L42 shows the reranker's bill, never its product:** measure a real cross-encoder on the 16
   questions (needs a model choice and download, roughly 90 MB for a MiniLM cross-encoder - your
   call, D-type) / say on the slide that the cost is measured and the gain is not
   **(recommended for now)**.
7. **L42 never shows hybrid winning:** add one toy part-number query where BM25 rescues what
   embeddings miss **(recommended**, measured on the 18 chunks) / keep the four conditions as text.
8. **L43 uses both "context precision" and "context relevance"** (pp. 34, 39, 40, 42, 45, 62):
   name the rank-aware one context precision and call the sentence-level one its original variant,
   consistently **(recommended)** / keep both names.
9. **L43 p9, lost in the middle:** the cited result says start or end; the advice says highest
   first. Add "and the next best last" **(recommended)** / keep "first" only.
10. **L43, which number to trust:** one closing line - recall@k on real questions for retrieval,
    faithfulness for use of the context, nothing for whether the corpus is right
    **(recommended)** / leave it spread across pp. 32, 44, 45.
11. **L44 mechanisms the reviewer could not reconstruct:** how MCP tools reach the model (and
    whether their schemas count in p16's cost), and how a skill's body enters the context. One
    frame each **(recommended)** / leave to the lecturer.
12. **L44's loop has guards in words only:** add the guarded loop as code, or one concrete
    turn-by-turn trace of the Tbilisi question **(recommended: the trace)** / neither.
13. **Truncated y-axes on bar charts** (L41 p45 at 0.80, p54 at 0.70; L43 p38 at 0.6): they
    exaggerate small gaps, which the reviewers noticed. Start at zero where the claim is "the gap is
    small" **(recommended for L41 p45 and L43 p38)** / keep all three.

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
