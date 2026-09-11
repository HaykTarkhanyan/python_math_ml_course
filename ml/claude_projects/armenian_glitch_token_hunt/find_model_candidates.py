"""Per-model glitch candidates: mine each SERVED model's own vocab.

The XLM-R candidates only diagnose XLM-R. For the live-probed models we find
their own suspects: Armenian-containing tokens in *their* vocab that never (or
almost never) occur in the 20M-char hy-wiki sample. No embedding norms here -
weights for these models are gated or multi-GB - so the signal is corpus
statistics only, which is exactly how the candidate list should be read.

Writes out/model_candidates.json. Needs the corpus cache from the notebook run.
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

from tokenizer_registry import ARMENIAN_RE, TOKENIZERS, load_tokenizer

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("logs/find_model_candidates.py.log",
                                  encoding="utf-8")],
)
log = logging.getLogger("find_model_candidates")

TARGETS = ["o200k", "llama3", "qwen2.5", "gemma2"]   # the OpenRouter-served four
SEP = "\n\n=====ARTICLE_BREAK=====\n\n"
CORPUS = Path("data/hy_wiki_20000000.txt")
N_CANDIDATES = 10


def main() -> None:
    if not CORPUS.exists():
        log.error(f"corpus cache {CORPUS} missing - run the notebook first")
        raise FileNotFoundError(CORPUS)
    articles = CORPUS.read_text(encoding="utf-8").split(SEP)
    log.info(f"corpus: {len(articles)} articles, "
             f"{sum(len(a) for a in articles):,} chars")

    result = {}
    for name in TARGETS:
        t = load_tokenizer(name)
        surfaces = t.all_token_strings()
        arm = {i: s for i, s in surfaces.items() if ARMENIAN_RE.search(s)}
        del surfaces

        cnt = Counter()
        for k in range(0, len(articles), 200):
            batch = articles[k:k + 200]
            if t.kind == "hf":
                for ids in t.raw(batch, add_special_tokens=False)["input_ids"]:
                    cnt.update(ids)
            else:
                for ids in t.raw.encode_batch(batch, disallowed_special=()):
                    cnt.update(ids)

        null_ids = [i for i in arm if cnt.get(i, 0) == 0]
        # rarest first; within equal count prefer longer (wordier) tokens
        ranked = sorted(arm, key=lambda i: (cnt.get(i, 0), -len(arm[i].strip())))
        cands = []
        for i in ranked:
            s = arm[i].strip()
            probeable = bool(s) and "�" not in s
            cands.append({"id": i, "surface": s,
                          "internal": t.internal_forms([i])[0],
                          "count": cnt.get(i, 0), "probeable": probeable})
            if len(cands) >= N_CANDIDATES:
                break

        result[name] = {"label": t.label, "n_armenian": len(arm),
                        "n_corpus_null": len(null_ids), "candidates": cands}
        log.info(f"{t.label:42s} armenian={len(arm):>5} null={len(null_ids):>4}  "
                 f"top: {', '.join(c['surface'] or '?' for c in cands[:5])}")
        del cnt, arm, t

    out = Path("out/model_candidates.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    log.info(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
