"""Build data/armenian_words.npz for the semantic-tree project.

Students get this file and never download a model. It carries the embeddings, the
tokenizations (so the tokenizer task works with no model and no internet), and the
polysemy sentences with their anchors.

Run with the project venv:
    ./ma/Scripts/python.exe ml/09_clustering/py_src/embed_armenian_words.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows console is cp1252, which cannot
                                           # print Armenian at all

import json
import logging
from pathlib import Path

import numpy as np

from armenian_words import (FAMILIES, ORTHO_PROBES, PHRASE_FORMS, POLYSEMY,
                            SYNONYM_PAIRS, all_leaves)

MODEL = "Metric-AI/armenian-text-embeddings-2-large"
PREFIX = "query: "          # E5 convention; the model card requires it on every input

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
DATA_DIR = CH_DIR / "data"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("embed_armenian_words")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "embed_armenian_words.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh); log.addHandler(fh)
    return log


def main():
    log = setup_logging()
    DATA_DIR.mkdir(exist_ok=True)

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    log.info(f"loading {MODEL}")
    model = SentenceTransformer(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def encode(texts):
        return model.encode([PREFIX + t for t in texts], normalize_embeddings=True,
                            batch_size=16, show_progress_bar=False).astype(np.float32)

    # ---- leaves ------------------------------------------------------------------------
    words, labels = all_leaves()
    log.info(f"{len(words)} leaves across {len(FAMILIES)} families")
    X = encode(words)

    n = np.linalg.norm(X, axis=1)
    log.info(f"embedding dim {X.shape[1]}, ||v|| in [{n.min():.4f}, {n.max():.4f}]")
    if abs(n.mean() - 1.0) > 1e-3:
        raise AssertionError(f"expected unit-norm vectors, got mean norm {n.mean():.4f}")

    # ---- phrase forms ------------------------------------------------------------------
    phrase_words = [w for w in PHRASE_FORMS if w in words]
    phrases = [PHRASE_FORMS[w] for w in phrase_words]
    Xp = encode(phrases)
    log.info(f"{len(phrase_words)} words also embedded as short phrases")

    # ---- polysemy ----------------------------------------------------------------------
    poly_rows, poly_texts = [], []
    for form, senses in POLYSEMY.items():
        for s in senses:
            poly_rows.append({"form": form, "sense": s["sense"],
                              "text_idx": len(poly_texts),
                              "anchor_idx": len(poly_texts) + 1})
            poly_texts += [s["text"], s["anchor"]]
    Xpoly = encode(poly_texts)
    log.info(f"{len(poly_rows)} polysemy senses over {len(POLYSEMY)} surface forms")

    # ---- tokenizations, so the tokenizer task needs no model --------------------------
    tokens = {w: tokenizer.tokenize(w) for w in words}
    tokens.update({p: tokenizer.tokenize(p) for p in phrases})
    log.info(f"stored tokenizations for {len(tokens)} strings")

    # Sanity: the probe groups must actually behave as the design claims.
    # The bare "▁" is SentencePiece's word-boundary marker and carries no information -
    # many words start with it, so it is dropped before looking for a shared token.
    def informative(w):
        return {t for t in tokens[w] if t != "▁"}

    for g in ORTHO_PROBES:
        present = [w for w in g["words"] if w in tokens]
        shared = set.intersection(*(informative(w) for w in present)) if present else set()
        expect_collide = g["expect"] == "collides"
        ok = bool(shared) == expect_collide
        log.info(f"  probe [{g['name']}] shared tokens {sorted(shared) or 'none'} "
                 f"-> {'as designed' if ok else 'MISMATCH'}")
        if not ok:
            raise AssertionError(
                f"probe group '{g['name']}' expected {g['expect']} but the shared-token "
                f"test says {sorted(shared) or 'none'}; the design doc's rule is wrong")

    meta = {
        "model": MODEL,
        "prefix": PREFIX,
        "families": FAMILIES,
        "ortho_probes": ORTHO_PROBES,
        "synonym_pairs": [list(p) for p in SYNONYM_PAIRS],
        "polysemy": POLYSEMY,
        "tokens": tokens,
        "poly_rows": poly_rows,
    }

    out = DATA_DIR / "armenian_words.npz"
    np.savez_compressed(
        out,
        words=np.array(words, dtype=object),
        labels=np.array(labels, dtype=object),
        X=X,
        phrase_words=np.array(phrase_words, dtype=object),
        phrases=np.array(phrases, dtype=object),
        X_phrase=Xp,
        poly_texts=np.array(poly_texts, dtype=object),
        X_poly=Xpoly,
        meta_json=json.dumps(meta, ensure_ascii=False),
    )
    log.info(f"saved {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
