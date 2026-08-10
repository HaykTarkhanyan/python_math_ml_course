"""Numbers for L41's hand-worked InfoNCE example.

The pedagogy review flagged an asymmetry: BM25 gets a full by-hand worked example on three
real chunks, while InfoNCE - the densest idea in the deck - gets a formula and no arithmetic.
This produces the arithmetic, with real e5-small similarities, so the slide can quote
verified numbers and a student can reproduce every step.

Run:  USE_TF=0 ./ma/Scripts/python.exe ml/ch17_rag/py_src/l41_infonce_worked.py
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

from l41_data import CHUNKS

MODEL_NAME = "intfloat/multilingual-e5-small"
QUERY = "What pressure should the press run at for Lori cheese?"
POSITIVE = CHUNKS[0]                      # the press / 2.5 bar chunk
NEGATIVES = [CHUNKS[3], CHUNKS[9]]        # pasteurisation, ripening cellar

LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l41_infonce_worked.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)


def infonce(sims, tau):
    """Loss for the case where index 0 is the positive. Returns (probabilities, loss)."""
    z = np.array(sims) / tau
    z = z - z.max()                       # numerically safe softmax, does not change the result
    e = np.exp(z)
    p = e / e.sum()
    return p, float(-np.log(p[0]))


def main():
    model = SentenceTransformer(MODEL_NAME)
    qv = model.encode([f"query: {QUERY}"], normalize_embeddings=True,
                      show_progress_bar=False)[0]
    docs = [POSITIVE] + NEGATIVES
    dv = model.encode([f"passage: {d}" for d in docs], normalize_embeddings=True,
                      show_progress_bar=False)
    sims = [float(qv @ v) for v in dv]

    log.info("query: %s", QUERY)
    for tag, d, s in zip(["d+ ", "d1-", "d2-"], docs, sims):
        log.info("  %s sim=%.4f  %s", tag, s, d[:58])

    for tau in (1.0, 0.1, 0.05):
        p, loss = infonce(sims, tau)
        log.info("tau=%.2f  ->  p = [%s]   loss = %.4f",
                 tau, ", ".join(f"{x:.4f}" for x in p), loss)

    # The teaching point: identical similarities, different temperature, very different loss.
    _, loss_hot = infonce(sims, 1.0)
    _, loss_cold = infonce(sims, 0.1)
    if not loss_cold < loss_hot:
        raise ValueError("expected a lower temperature to sharpen towards the positive here")
    log.info("temperature effect: loss %.4f at tau=1.0 -> %.4f at tau=0.1", loss_hot, loss_cold)


if __name__ == "__main__":
    main()
