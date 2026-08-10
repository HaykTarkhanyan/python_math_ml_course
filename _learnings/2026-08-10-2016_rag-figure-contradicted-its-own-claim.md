# A RAG figure was designed to prove a claim, then measured the opposite

**Symptom.** While building L41 (RAG retrieval), figure 13 was specified in the chapter plan as
"each method fails where the other succeeds": BM25 should win on exact identifiers, dense should
win on paraphrases. The figure script ran, and produced this:

```
direct      BM25 rank  2 | dense rank  2      (expected both 1)
paraphrase  BM25 rank  3 | dense rank  8      (expected dense to WIN)
partnumber  BM25 rank  1 | dense rank  1      (expected dense to FAIL)
```

Dense lost the paraphrase and won the identifier. The exact reverse of the slide's message.

**Cause.** Two separate faults, and only one was about the model.

1. **The identifier test had no distractors.** `PRS-400` appeared in exactly one chunk and
   nothing else in the corpus resembled it. There was nothing for an embedder to confuse it
   with, so the test could not have failed regardless of the model. A claim of the form "X
   confuses similar things" needs similar things in the corpus or it is vacuous.
2. **Rank was reported without score.** BM25's "rank 3" on the paraphrase looked like mediocre
   retrieval. In fact the only shared token was "the", whose IDF is ~0, so *every* score was
   ~0 and the ranking was arbitrary tie-breaking. Rank alone disguised "no signal whatsoever"
   as "middling performance".

**What the measurement actually supports.** After adding four near-miss part numbers
(`PRS-220`, `PRS-380`, `PRS-410`, `VAC-400`) and reporting scores alongside ranks, on an
18-chunk corpus with `intfloat/multilingual-e5-small`:

```
query "How warm is the room where the cheese matures?"  -> ripening-cellar chunk
  BM25   rank 17/18   score 0.000     <- no lexical signal at all
  dense  rank  4/18   score 0.801

exact-identifier top-1 accuracy over 5 competing part numbers:
  BM25  4/5     dense 4/5             <- identical; both miss PRS-220 for the same
                                         legitimate reason (the PRS-400 chunk contains it)
```

So: **"dense retrieval is worse at exact identifiers" did not reproduce at this scale and was
cut from the deck.** What did reproduce is stronger, because it follows from the scoring
function rather than from a benchmark: BM25's score is nonzero *iff* a query term literally
appears, so its failure is structural and total, while a dense score degrades smoothly and is
never exactly zero.

**Consequences.**

- The chapter's motivation for hybrid retrieval was rewritten. It no longer claims a symmetry
  ("each is better at different things"); it claims an asymmetry of *guarantees*: BM25
  guarantees an exact term will surface, dense guarantees coverage of rephrasing. That is
  provable from the formula and does not depend on a toy corpus.
- The general claim that dense retrieval degrades on rare exact tokens at production scale is
  still believed to be true, but an 18-chunk corpus cannot demonstrate it, so the deck states
  it as a caveat rather than showing it.
- `py_src/probe_retrieval_claims.py` was kept: it measures rank *and* score for candidate
  queries, and exists so the next claim gets checked before a frame is written around it.

**The transferable rule.** Writing the pedagogical payoff first and assuming the evidence will
follow is how a confident wrong claim reaches students. Measure, then write the slide. If a
figure script needs its inputs tuned until the conclusion appears, the conclusion is the tuning,
not the finding. Related: the same session's plan review caught an unverifiable claim about
transliterated Armenian justifying hybrid retrieval, which was cut for the same reason.
