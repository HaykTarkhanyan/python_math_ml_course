# Chapter 17 - Retrieval-Augmented Generation

**Status:** outline v2, awaiting approval. Nothing built yet.

Three decks, `L41`-`L43`, continuing `ml/` numbering after `L40` (JEPA). Chapter page `rag.qmd`
registers in `_quarto.yml` after `ml/ch16_jepa/jepa.qmd`. Three decks matches `ch10_diffusion`
(L29/L30/L31); v1 of this plan tried to fit L41 into ~60 frames with two full derivations, which
does not fit a 17:30-19:00 session.

Closes the gap in `ml/MISSING_TOPICS.md:108` (*"Prompting / RAG - coverage: none"*).

## Teaching stance (instructor, 2026-08-10) - this governs every decision below

- **This is a teaching deck for students, not a literature review.** When intuition and rigour
  compete, intuition wins.
- **Cite sparingly.** A name and a rough era where it helps ("BM25 comes out of 1990s search
  engines"). No reference slide, no paper-by-paper tour. `papers/` is instructor-side grounding
  so the formulas and numbers are *correct*, not a student reading list.
- **Every mechanism gets a concrete example before or instead of the general statement.**
- **Illustrations everywhere.** 15 generated figures in L41, ~12 in each of L42/L43.
- **Depth:** derive BM25 and the contrastive objective in full, because those two explain
  *behaviour* students will actually observe. Everything else stays descriptive.
- **No unlabelled numbers on slides.** If a benchmark score appears, the slide says what it
  measures in plain words.

### Dropped from v1

- **Transliterated Armenian.** Instructor call: not interesting enough to spend frames on.
  Removes the one claim in v1 I could not support (that the transliteration collapse justifies
  hybrid retrieval - unverifiable without knowing ArmBench's setup, and probably wrong).
- **Armenian-vs-English token-count figure.** `ml/ch7_rnn/L21_road_to_attention.tex:259` already
  has a frame called "The Armenian tax" with real `tiktoken` measurements (`tokenizer_panel3.pdf`).
  Signpost L21 in one line instead of rebuilding it.

## Running example

**The cheese factory** from the classification chapter, now with a documentation problem: equipment
manuals, safety regulations, supplier contracts, five years of QA incident reports. A maintenance
engineer asks *"Ի՞նչ ճնշման տակ պետք է աշխատի մամլիչը Լոռի պանրի համար"*. The answer exists, in one
paragraph, in one of 4,000 documents.

Each section returns to it: chunking splits the manuals, BM25 finds the model number, the embedder
finds the paraphrase, reranking picks between two contradictory revisions of the same procedure.

---

## L41 - Retrieval: the two ways to find a document

~40 frames.

### Cold open

1. Ask a real LLM a cheese-factory question, live. It answers fluently and invents a pressure
   value. **A real run, captured - not a mock-up.** A deck about hallucination should not fabricate
   its own opening exhibit.
2. Same question with the one correct paragraph pasted above it. Correct answer, with a citation.
3. **That is all RAG is.** The rest of the chapter is the one hard part: finding that paragraph,
   out of 4,000 documents, in 200 milliseconds.

### Section 1 - Why retrieve

- What the model knows is frozen in its weights: a cutoff, no private data, no updates.
- Four consequences: stale answers, confident invention, no citations, no access control.
- **Kill the strawman** "just paste all 4,000 documents": context limit, cost per query, and
  accuracy dropping when the answer sits in the middle of a long context.
  *(Qualitative only unless I fetch the source - see Open questions.)*
- **The RAG loop**, one full-bleed diagram: *index once, offline* vs *retrieve and generate, per
  query*. This split is the spine of the whole chapter and gets called back three times.

### Section 2 - Chunking

- Why a 90-page manual cannot be one vector: one vector holds one meaning.
- Four strategies over the same paragraph, colour-coded: fixed-size, sentence, recursive, semantic.
- **Overlap**, and the sentence cut in half without it.
- **The recall ceiling.** If chunking loses the answer, nothing downstream recovers it - not a
  better embedder, not a reranker. Stage one caps the entire pipeline. *(New in v2. This is the
  most common real-world RAG failure and v1 only gestured at it.)*
- **Metadata**: document, revision date, section, who is allowed to see it. Pays off twice later -
  filtering in L42, and picking between contradictory revisions in the reranking section.
- The trap frame: a chunk that contains the answer but shares no words with the question. Sets up
  both of the next two sections.

### Section 3 - Lexical retrieval: BM25 (full derivation)

Built up in stages, each stage motivated by a failure of the previous one.

- Start dumb: count query words in the document. Show why raw counting breaks.
- **Rare words matter more.** "the" vs "մամլիչ" - motivate IDF before writing it.
- **Predict-first:** a document saying "pressure" 50 times - 50x more relevant than one saying it
  once? Students vote, *then* the saturation curve appears. `tf/(k1+tf)`, and what `k1` controls.
- **Long documents cheat.** `B = (1-b) + b·dl/avdl`; what `b=0` and `b=1` mean.
- Assemble the full score, then a **by-hand worked-numbers frame** on three short cheese-factory
  documents: compute IDF, compute B, compute scores, rank them.
- Typical values, plus the `(k1+1)` numerator variant that changes nothing about the ranking - a
  nice lesson that formulas carry cosmetic terms.
- **Where BM25 is unbeatable:** model numbers, article references, part codes, rare proper nouns.
- **Where it dies:** the vocabulary-mismatch trap from Section 2.

### Section 4 - Dense retrieval: embedders (full derivation of the objective)

**Starts from what students already have.** Word vectors, cosine similarity and the analogy trick
were taught in `L21` (word2vec/GloVe) and referenced in `L24`. Signpost, do not re-teach.

- What changes here: the unit is a **chunk**, not a word, and the goal is **retrieval**, not
  similarity in general.
- **How the vectors get good:** contrastive learning. Query and its correct passage pulled
  together, everything else pushed apart. Before/after scatter.
- The InfoNCE objective, written out and explained term by term, including temperature.
- **In-batch negatives** - the trick that makes it cheap, one diagram.
- **Bi-encoder**, and why it is fast: chunk vectors computed once, offline; only the query is
  encoded per request. Callback to the Section 1 split.
- Asymmetry: `query:` / `passage:` prefixes, and what silently degrades if you forget them.
- **Where dense retrieval struggles:** negation, out-of-domain jargon, and rare exact tokens at
  production scale. **Stated as a caveat, not demonstrated** - see the measured-results box below.
- **The honest asymmetry** (this replaces v1's "each fails where the other succeeds"):
  BM25's score is nonzero *if and only if* a query term literally appears, so its failure is
  structural and total. A dense score degrades smoothly and is never exactly zero. So the two
  offer different **guarantees**, not merely different strengths: BM25 guarantees an exact term
  surfaces; dense guarantees rephrasing is covered. That follows from the scoring function, so it
  does not depend on the size of our demo corpus.

> **Measured, 2026-08-10 - one planned claim did not survive.** On the 18-chunk corpus with
> `intfloat/multilingual-e5-small`: the paraphrase query scores **BM25 rank 17/18 with score
> exactly 0.000** against **dense rank 4** - the vocabulary-mismatch failure is real and stark.
> But over five competing part numbers, exact-identifier top-1 accuracy was **BM25 4/5, dense
> 4/5** - identical. **"Dense is worse at exact IDs" did not reproduce and was cut.** Full
> write-up in `_learnings/2026-08-10-2016_rag-figure-contradicted-its-own-claim.md`; the
> measurement harness is `py_src/probe_retrieval_claims.py`.
- Choosing one: dimension, max length, multilingual support, licence. What a leaderboard score
  does and does not tell you.

### Section 5 - Armenian, and ATE-2

- **Why this section exists:** everything above assumed a good embedder for your language. For
  Armenian, that assumption fails by default.
- One line on the tokenization tax, pointing at L21's "Armenian tax" frame. No new figure.
- **ATE-2** (Metric AI Lab): take the multilingual **mE5** encoder, fine-tune on **10,000 noisy
  synthetic pairs** built by translating English Reddit title/body pairs. Two sizes, 278M and 560M.
- **The result**, as a labelled bar chart - ArmBench-TextEmbed, native Armenian, higher is better:

  | Model | Score |
  |---|---|
  | ATE-2-large (560M) | **0.805** |
  | gemini-embedding-001 | 0.774 |
  | ATE-2-base (278M) | 0.767 |

- **Predict-first:** which wins, Google's commercial embedder or a 560M open model tuned on 10k
  synthetic pairs? Students vote first. *(Only one predict-first in this section - v1 had two.)*
- **Two lessons:**
  1. Bigger and more general is not automatically better **for your language**.
  2. 10k targeted pairs matched what a million generic examples buys. Targeting beats volume.
- Practical closer: what to actually use for an Armenian RAG today, with the HuggingFace names
  `Metric-AI/armenian-text-embeddings-2-large` and `-base`.

> **Instructor note, not slide content.** The headline ArmBench number is an average over four
> subsets with mixed metrics (Top-20 Accuracy for retrieval, Spearman for STS, Top-10 Accuracy for
> MS MARCO, MTEB Mean(Task)), and the manually curated retrieval subset is only 185 pairs. The
> slide says "average score on an Armenian embedding benchmark, higher is better" and moves on -
> that is the right level for students. Do not put the composite breakdown on a slide.
>
> **Keep ATE-1 and ATE-2 apart.** The paper in `ml/text_embedding/` (08, arXiv 2603.22290) is
> **ATE-1** and reports different numbers (retrieval 58.15 → 79.35, MS MARCO 60.73 → 80.25). The
> table above is **ATE-2**, from the HuggingFace blog post. Never mix the two on one slide.

### Recap + Next

Two ways to retrieve, each strong exactly where the other is weak. **Next:** stop choosing, and
combine them.

---

## L42 - Combining and refining: hybrid, reranking, indexes

~40 frames.

1. **The payoff frame.** Scatter of which queries each method wins. They fail on *different*
   queries - the entire argument for hybrid, shown before it is stated.
2. **Keyword extraction** *(expanded in v2 - the instructor asked for this by name and v1 gave it
   one frame)*: pulling searchable terms out of a messy real question. Statistical (RAKE/YAKE),
   embedding-based (KeyBERT), and just asking an LLM. Worked end to end on one rambling
   cheese-factory question, showing what each approach extracts and where each embarrasses itself.
3. **Query expansion:** synonyms, Armenian/English variants of the same technical term.
4. **Fusing two ranked lists.** Why you cannot add the scores (BM25 unbounded, cosine in [-1,1]),
   why normalising is fragile.
5. **Reciprocal Rank Fusion**, `Σ 1/(k + rank)`. Rank-based, so scale-free. Worked by hand on two
   short lists with the fused ranking computed on the slide.
6. **Metadata filtering** *(new in v2)*: date, permissions, document type. The access-control
   motivation from L41 finally gets its mechanism.
7. **Reranking.** The bi-encoder bottleneck: query and document never look at each other.
   Cross-encoders fix that and are far too slow to run on everything.
8. **The cascade**, as a funnel with latency and cost annotated: 1M chunks → 100 candidates →
   10 reranked → 3 in the prompt. The most reusable diagram in the chapter.
9. **ColBERT / late interaction** as the middle ground, with a MaxSim heatmap of query tokens
   against document tokens.
10. Reranking the two contradictory revisions of the cheese-press procedure - Section 2's metadata
    pays off.
11. **Making it fast** (descriptive): brute force is fine more often than people think; HNSW as a
    skip-list in high dimensions; approximate means approximate, so know your recall number;
    quantization in one frame.

## L43 - Generation and evaluation

~35 frames. Detailed after L41 and L42 are approved.

1. Context assembly - ordering, deduplication, token budget.
2. Prompting for grounding - forcing citations, licensing "I don't know".
3. Failure modes - retrieved but ignored, right answer with wrong citation, contradictory sources.
4. Measuring retrieval - recall@k, MRR, nDCG, worked by hand on a small example.
5. Measuring generation - faithfulness, answer relevance, context precision. LLM-as-judge, and why
   it is both necessary and suspect.
6. Building an eval set from nothing - generate questions from your own chunks.
7. Beyond naive RAG - query rewriting, HyDE, multi-hop, agentic retrieval, graph RAG.
8. When not to use RAG - long context, fine-tuning, or just writing a SQL query.

---

## Figures for L41 (Python-generated, `py_src/` → `fig/`)

Following `L21`'s convention, each figure is tagged **REAL** (measured/computed) or
**ILLUSTRATIVE** (hand-placed schematic), and the tag goes in the deck's provenance block.

| # | Figure | Section | Kind |
|---|---|---|---|
| 01 | RAG loop: offline indexing vs online retrieve+generate (full-bleed) | 1 | ILLUSTRATIVE |
| 02 | Four chunking strategies over the same paragraph, colour-coded | 2 | REAL |
| 03 | Recall ceiling: answer present vs recovered, by pipeline stage | 2 | ILLUSTRATIVE |
| 04 | BM25 saturation curve, several `k1` | 3 | REAL |
| 05 | Length-normalisation factor `B` vs `dl/avdl`, several `b` | 3 | REAL |
| 06 | Worked BM25 scores on three documents (labelled bars) | 3 | REAL |
| 07 | Raw count vs BM25 ranking on the same three documents | 3 | REAL |
| 08 | 2-D embedding space: query and chunk neighbourhoods | 4 | see below |
| 09 | Contrastive learning before/after: positives in, negatives out | 4 | ILLUSTRATIVE |
| 10 | In-batch negatives: the similarity matrix and its diagonal | 4 | ILLUSTRATIVE |
| 11 | Bi-encoder vs cross-encoder architecture | 4 | ILLUSTRATIVE |
| 12 | Offline vs online cost: what runs once, what runs per query | 4 | ILLUSTRATIVE |
| 13 | BM25 hits score 0.000 on paraphrase while dense degrades to rank 4 | 3, 4 | REAL |
| 14 | ATE-2 vs Gemini, native Armenian (labelled bars) | 5 | REAL |
| 15 | ATE-2 recipe: mE5 + 10k synthetic pairs, as a flow | 5 | ILLUSTRATIVE |

## Open questions

1. **Three decks or two?** v1's two-deck split does not fit the time. Three is my recommendation
   and matches `ch10_diffusion`. Say the word if you would rather compress to two and cut depth.
2. **Real embeddings for figures 08/09/10?** Much more convincing computed from real vectors.
   Needs `sentence-transformers` in `ma` plus a small model download (a few hundred MB, CPU-only,
   seconds of compute on ~30 short sentences - inside the machine-load rule). Otherwise they are
   hand-placed schematics, which is honest but weaker.
3. **The mid-context accuracy claim in Section 1.** I do not have that paper locally, and I will
   not draw a curve from memory. Either I fetch it, or the frame stays qualitative with no numbers.
4. **Running example** - cheese factory (callback to the classification chapter) confirmed? The
   alternative is a university-regulations assistant, a more familiar corpus for students.
5. **Homework** for `rag.qmd` - build a small hybrid retriever over a provided Armenian corpus is
   the obvious one. Deferred until the decks exist.

## Build order

1. Approve/adjust this outline.
2. `py_src/l41_figs.py` → `fig/*.pdf`, eyeball them.
3. `L41_rag_retrieval.tex`, compile twice, `clean_latex.py`, overflow check.
4. `rag.qmd`, register in `_quarto.yml`.
5. Then L42, then L43.

## Already in place

- `papers/` - 6 source papers, extracted to `papers/llm_readable/`. See `papers/README.md`.
- `extract_rag_papers.py` - regenerates the text versions, with title and page-count assertions.
