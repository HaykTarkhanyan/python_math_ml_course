"""Keyword extraction for L42, implemented from scratch.

Three families, all run on the same rambling cheese-factory question:

  RAKE     Rapid Automatic Keyword Extraction (Rose et al., 2010). Split on stopwords,
           score each word by degree/frequency, sum over the phrase.
  YAKE     Yet Another Keyword Extractor (Campos et al., 2018). Five statistical features
           per term; LOWER score is better.
  KeyBERT  (Grootendorst, 2020). Embed the document and every candidate phrase, keep the
           phrases whose vector is closest to the document's.

Why from scratch rather than `pip install rake-nltk yake keybert`: the deck shows the
intermediate arithmetic (RAKE's deg/freq table, YAKE's per-term scores), which the library
APIs do not expose, and it keeps the `ma` venv untouched. The RAKE implementation is
faithful. The YAKE implementation follows the paper's five features and its n-gram
combination rule but skips the Levenshtein de-duplication step, so its scores may differ
slightly from the reference implementation - the ranking of the top terms is what the deck
uses.

KeyBERT needs an encoder: intfloat/multilingual-e5-small, already cached (USE_TF=0).
"""

import logging
import math
import os
import re
from collections import Counter, defaultdict

os.environ.setdefault("USE_TF", "0")

import numpy as np

log = logging.getLogger(__name__)

# A small, visible English stopword list. RAKE's whole behaviour depends on this list,
# which is exactly the point the deck makes about it.
STOPWORDS = {
    "a", "about", "again", "actually", "an", "and", "anywhere", "are", "as", "at", "be",
    "been", "before", "bother", "but", "by", "can", "could", "did", "do", "does", "during",
    "for", "found", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it", "its",
    "last", "me", "my", "no", "not", "of", "on", "or", "our", "out", "should", "so", "some",
    "someone", "sorry", "supposed", "that", "the", "their", "then", "there", "they", "this",
    "to", "up", "us", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
}

_SENT_SPLIT = re.compile(r"[.!?]+\s+")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


def sentences(text):
    return [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def words(text):
    return _WORD.findall(text)


# ---------------------------------------------------------------- RAKE
def rake(text, top_k=5):
    """Return [(phrase, score)] plus the per-word degree/frequency table."""
    phrases = []
    for sent in sentences(text):
        current = []
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*|[,;:]", sent):
            if tok in {",", ";", ":"} or tok.lower() in STOPWORDS:
                if current:
                    phrases.append(current)
                    current = []
            else:
                current.append(tok.lower())
        if current:
            phrases.append(current)

    freq = Counter()
    degree = Counter()
    for ph in phrases:
        for w in ph:
            freq[w] += 1
            degree[w] += len(ph)  # co-occurrence degree, self included

    word_score = {w: degree[w] / freq[w] for w in freq}
    scored = {}
    for ph in phrases:
        scored[" ".join(ph)] = sum(word_score[w] for w in ph)

    ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:top_k]
    table = sorted(word_score.items(), key=lambda kv: -kv[1])
    return ranked, table, {w: (degree[w], freq[w]) for w in freq}


# ---------------------------------------------------------------- YAKE
def _yake_term_scores(text):
    sents = sentences(text)
    tok_sents = [words(s) for s in sents]
    flat = [w for s in tok_sents for w in s]
    lower = [w.lower() for w in flat]

    tf = Counter(lower)
    content_tf = [tf[w] for w in tf if w not in STOPWORDS]
    mean_tf = float(np.mean(content_tf))
    std_tf = float(np.std(content_tf))
    max_tf = max(tf.values())

    sent_of = defaultdict(list)
    for si, toks in enumerate(tok_sents):
        for w in toks:
            sent_of[w.lower()].append(si)

    upper_count = Counter()
    for si, toks in enumerate(tok_sents):
        for wi, w in enumerate(toks):
            if w.isupper() and len(w) > 1:
                upper_count[w.lower()] += 1          # acronym
            elif w[0].isupper() and wi > 0:
                upper_count[w.lower()] += 1          # capitalised mid-sentence

    left = defaultdict(set)
    right = defaultdict(set)
    left_n = Counter()
    right_n = Counter()
    for toks in tok_sents:
        low = [w.lower() for w in toks]
        for i, w in enumerate(low):
            if i > 0:
                left[w].add(low[i - 1])
                left_n[w] += 1
            if i < len(low) - 1:
                right[w].add(low[i + 1])
                right_n[w] += 1

    scores = {}
    parts = {}
    for w in tf:
        w_case = upper_count[w] / (1.0 + math.log(tf[w]))
        w_pos = math.log(math.log(3 + float(np.median(sent_of[w]))))
        tf_norm = tf[w] / (mean_tf + std_tf)
        dl = len(left[w]) / left_n[w] if left_n[w] else 0.0
        dr = len(right[w]) / right_n[w] if right_n[w] else 0.0
        w_rel = 1.0 + (dl + dr) * tf[w] / max_tf
        w_diff = len(set(sent_of[w])) / len(sents)

        s = (w_rel * w_pos) / (w_case + tf_norm / w_rel + w_diff / w_rel)
        scores[w] = s
        parts[w] = dict(case=w_case, pos=w_pos, freq=tf_norm, rel=w_rel, diff=w_diff)
    return scores, parts, tf


def yake(text, top_k=5, max_n=3):
    """Return [(phrase, score)] with LOWER = better, plus the per-term feature table."""
    term_scores, parts, tf = _yake_term_scores(text)

    cand_tf = Counter()
    for sent in sentences(text):
        toks = [w.lower() for w in words(sent)]
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                gram = toks[i:i + n]
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                cand_tf[" ".join(gram)] += 1

    scored = {}
    for gram, gtf in cand_tf.items():
        ws = [term_scores[w] for w in gram.split()]
        prod = float(np.prod(ws))
        scored[gram] = prod / (gtf * (1.0 + sum(ws)))

    ranked = sorted(scored.items(), key=lambda kv: kv[1])[:top_k]
    return ranked, term_scores, parts


# ---------------------------------------------------------------- KeyBERT
def candidate_ngrams(text, max_n=3):
    out = []
    for sent in sentences(text):
        toks = [w.lower() for w in words(sent)]
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                gram = toks[i:i + n]
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                out.append(" ".join(gram))
    seen, uniq = set(), []
    for g in out:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq


def keybert(text, model, top_k=5, max_n=3):
    """Cosine similarity of every candidate phrase to the whole document.

    e5 is an asymmetric model, so for a symmetric comparison like this both sides get the
    same `query: ` prefix, which is what the model card prescribes for symmetric tasks.
    """
    cands = candidate_ngrams(text, max_n=max_n)
    if not cands:
        raise ValueError("no candidate n-grams extracted - check the stopword list")
    doc_vec = model.encode([f"query: {text}"], normalize_embeddings=True)[0]
    cand_vecs = model.encode([f"query: {c}" for c in cands], normalize_embeddings=True)
    sims = cand_vecs @ doc_vec
    order = np.argsort(-sims)
    return [(cands[i], float(sims[i])) for i in order[:top_k]], list(zip(cands, sims))
