"""A complete, runnable RAG retrieval pipeline for the cheese-factory corpus.

The chapter's three lectures show every stage of a RAG system in isolation. This is the
whole thing in one file, end to end, so you can read it, run it, and change it:

    chunk  ->  embed  ->  index  ->  retrieve (BM25 + dense)  ->  filter  ->  build prompt

It stops at the prompt. It does NOT call a language model - there is no API key here, and
the interesting part for this course is the prompt itself, which is the thing you never
normally get to look at.

Run:
    USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/rag_demo.py
    USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/rag_demo.py --no-filter

Try next:
    * change TOP_K and watch what enters the prompt
    * run with --no-filter and compare the two prompts (this is L43's cold open)
    * add a chunk to DOCUMENTS and re-run; nothing else needs to change
"""

import argparse
import logging
import os
import re
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 3
K1, B = 1.5, 0.75
RRF_K = 60

# --- 1. the documents -------------------------------------------------------------------
# Each document carries metadata, because retrieval quality is not only about text.
DOCUMENTS = [
    dict(text="Press manual revision 2, issued 2019: the Lori press is held at 2.5 bar for "
              "the whole pressing cycle.",
         source="press_manual", revision=2, year=2019, current=False),
    dict(text="Press manual revision 4, issued 2024: the Lori press starts at 2.5 bar and is "
              "raised to 3.2 bar after twenty minutes.",
         source="press_manual", revision=4, year=2024, current=True),
    dict(text="Pasteurisation holds the milk at 72 degrees for fifteen seconds before the "
              "culture is added.",
         source="process_notes", revision=1, year=2023, current=True),
    dict(text="Ripening cellars are held at 10 degrees and 85 percent humidity.",
         source="process_notes", revision=1, year=2023, current=True),
    dict(text="Model PRS-400 replaces the older PRS-220 press on line two.",
         source="equipment_log", revision=1, year=2022, current=True),
    dict(text="Operators must wear cut-resistant gloves when handling the curd knives.",
         source="safety", revision=3, year=2024, current=True),
]

QUESTION = "What pressure should the Lori press run at?"

LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "rag_demo.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9.\-]+", text.lower()) if w]


# --- 2. lexical retrieval ---------------------------------------------------------------
def bm25_scores(query, docs_tokens):
    n = len(docs_tokens)
    avgdl = sum(len(d) for d in docs_tokens) / n
    out = []
    for toks in docs_tokens:
        norm = (1 - B) + B * len(toks) / avgdl
        s = 0.0
        for term in tokenize(query):
            tf = toks.count(term)
            if tf == 0:
                continue
            df = sum(1 for d in docs_tokens if term in d)
            s += tf / (K1 * norm + tf) * np.log(1 + (n - df + 0.5) / (df + 0.5))
        out.append(s)
    return np.array(out)


# --- 3. fusion --------------------------------------------------------------------------
def rrf(rankings, n_docs):
    """Reciprocal rank fusion. `rankings` is a list of arrays of doc indices, best first."""
    score = np.zeros(n_docs)
    for order in rankings:
        for rank, doc in enumerate(order, start=1):
            score[doc] += 1.0 / (RRF_K + rank)
    return score


# --- 4. the prompt ----------------------------------------------------------------------
def build_prompt(question, chosen):
    """Assemble the literal string a language model would receive."""
    blocks = []
    for i, d in enumerate(chosen, start=1):
        blocks.append(f"[{i}] ({d['source']}, rev {d['revision']}, {d['year']})\n{d['text']}")
    context = "\n\n".join(blocks)
    return (
        "Answer the question using ONLY the context below.\n"
        "Cite the block number you used, like [1].\n"
        "If the context does not contain the answer, reply exactly: I don't know.\n\n"
        f"CONTEXT\n{context}\n\n"
        f"QUESTION\n{question}\n\n"
        "ANSWER"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-filter", action="store_true",
                    help="skip the metadata filter, reproducing L43's cold open")
    args = ap.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    docs_tokens = [tokenize(d["text"]) for d in DOCUMENTS]
    doc_vecs = model.encode([f"passage: {d['text']}" for d in DOCUMENTS],
                            normalize_embeddings=True, show_progress_bar=False)
    q_vec = model.encode([f"query: {QUESTION}"], normalize_embeddings=True,
                         show_progress_bar=False)[0]

    bm = bm25_scores(QUESTION, docs_tokens)
    dense = doc_vecs @ q_vec
    fused = rrf([np.argsort(-bm), np.argsort(-dense)], len(DOCUMENTS))

    order = list(np.argsort(-fused))
    log.info("question: %s", QUESTION)
    for rank, ix in enumerate(order[:5], start=1):
        log.info("  %d. bm25=%.3f dense=%.3f rrf=%.4f %s| %s", rank, bm[ix], dense[ix],
                 fused[ix], "" if DOCUMENTS[ix]["current"] else "[SUPERSEDED] ",
                 DOCUMENTS[ix]["text"][:52])

    # The filter is one line, and it is the entire fix for the chapter's cold open.
    if args.no_filter:
        log.warning("metadata filter DISABLED - superseded documents can reach the prompt")
        eligible = order
    else:
        eligible = [ix for ix in order if DOCUMENTS[ix]["current"]]
        dropped = [ix for ix in order if not DOCUMENTS[ix]["current"]]
        for ix in dropped:
            log.info("filtered out (superseded): %s", DOCUMENTS[ix]["text"][:52])

    chosen = [DOCUMENTS[ix] for ix in eligible[:TOP_K]]
    if not chosen:
        raise ValueError("no eligible documents - the filter removed everything")

    prompt = build_prompt(QUESTION, chosen)
    print("\n" + "=" * 74)
    print(prompt)
    print("=" * 74 + "\n")
    log.info("prompt built from %d chunks, %d characters", len(chosen), len(prompt))


if __name__ == "__main__":
    main()
