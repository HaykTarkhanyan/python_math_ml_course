"""Shared data for the L42 (hybrid retrieval, reranking, indexes) figures.

Everything here extends the L41 cheese-factory corpus (`l41_data.CHUNKS`) so the running
example stays continuous. Nothing in l41_data.py is modified.

Three additions:

  EVAL_QUERIES   16 queries with a known target chunk, used to measure BM25 vs dense vs
                 hybrid. Written ONCE, before any measurement was run, and deliberately
                 not edited afterwards - the point of the set is to be a fair sample of
                 what a maintenance engineer types, not a set that produces a nice slide.
  RAMBLING       one long, realistic user question, used for the keyword-extraction
                 section.
  REVISIONS      two contradictory revisions of the same cheese-press procedure, used by
                 the reranking section.

Shared BM25 / RRF helpers live here too so every L42 script scores identically.
"""

import numpy as np

from l41_data import CHUNKS, tokenize

# --- the evaluation query set -----------------------------------------------------------
# (query text, index into CHUNKS, kind). `kind` is only a label for the plots.
EVAL_QUERIES = [
    ("What pressure should the press run at for Lori cheese?", 0, "keyword"),
    ("How warm is the room where the cheese matures?", 9, "paraphrase"),
    ("PRS-400", 8, "identifier"),
    ("Which press was taken out of service after the recall?", 14, "paraphrase"),
    ("How often should the hydraulic gauge be checked?", 1, "keyword"),
    ("At what temperature is the milk heated to kill bacteria?", 3, "paraphrase"),
    ("How long until the milk turns into curd?", 5, "paraphrase"),
    ("What protective equipment is needed for the knives?", 10, "keyword"),
    ("VAC-400", 17, "identifier"),
    ("Which press handles harder cheeses?", 15, "keyword"),
    ("brine tank temperature Lori", 2, "keyword"),
    ("Can we still buy spare parts for the PRS-410?", 16, "identifier"),
    ("Who calibrates the pressure gauges?", 13, "keyword"),
    ("What is the residual pressure when wheels are sealed?", 11, "keyword"),
    ("Are incoming deliveries checked for drug residues?", 12, "paraphrase"),
    ("How small are the pieces the curd is cut into?", 6, "paraphrase"),
]

# --- the rambling question (keyword-extraction section) ---------------------------------
# One utterance, written as a person would actually type it: an apology, a story about
# last week, and the real question in the middle. The distractor words ("maintenance",
# "gauge", "logged") are there because a real user mentions adjacent things, not because
# they were chosen to break anything.
RAMBLING = (
    "Sorry to bother you again. Last week the hydraulic gauge on line two was reading "
    "strangely during the second pressing stage and the operator logged it in the "
    "maintenance report. I could not find the number anywhere, so what pressure is the "
    "Lori press actually supposed to run at?"
)
RAMBLING_TARGET = 0

# Keywords a language model returns when asked to pull the searchable terms out of
# RAMBLING. NOT produced by a scripted API call - there is no API key in this environment.
# Written by the assistant that built the deck, which is itself a language model, and
# labelled as such on the slide and in the deck's provenance block.
LLM_KEYWORDS = ["Lori press", "pressure", "operating pressure"]

# --- two contradictory revisions (reranking section) ------------------------------------
REVISIONS = [
    {
        "id": "rev 2",
        "date": "2019-03-11",
        "text": "Cheese press procedure, revision 2: the Lori press is held at 2.0 bar "
                "for the whole pressing cycle.",
        "status": "superseded",
    },
    {
        "id": "rev 4",
        "date": "2024-06-02",
        "text": "Cheese press procedure, revision 4: the Lori press starts at 2.5 bar "
                "and is raised to 3.2 bar after twenty minutes.",
        "status": "current",
    },
]
REVISION_QUERY = "What pressure should the Lori press run at?"

# --- scoring helpers --------------------------------------------------------------------
K1, B = 1.5, 0.75
RRF_K = 60  # the constant from Cormack, Clarke & Buettcher (2009)


def bm25_scores(query, docs_tokens, k1=K1, b=B):
    """BM25 over a token-list corpus.

    Uses the Lucene idf variant ln(1 + (N-n+0.5)/(n+0.5)), same as L41's figure code:
    Robertson's raw idf goes negative for common terms on a corpus this small.
    """
    n_docs = len(docs_tokens)
    avgdl = sum(len(d) for d in docs_tokens) / n_docs
    terms = tokenize(query)
    out = np.zeros(n_docs)
    for i, toks in enumerate(docs_tokens):
        norm = (1 - b) + b * len(toks) / avgdl
        s = 0.0
        for term in terms:
            tf = toks.count(term)
            if tf == 0:
                continue
            df = sum(1 for d in docs_tokens if term in d)
            idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            s += tf / (k1 * norm + tf) * idf
        out[i] = s
    return out


def ranks_from_scores(scores):
    """1-based rank per item, best score = rank 1. Ties broken by index (stable)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf_fuse(rank_lists, k=RRF_K):
    """Reciprocal Rank Fusion: score(d) = sum_i 1 / (k + rank_i(d)). Higher is better."""
    rank_lists = [np.asarray(r) for r in rank_lists]
    return sum(1.0 / (k + r) for r in rank_lists)


def corpus_tokens():
    return [tokenize(c) for c in CHUNKS]
