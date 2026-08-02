"""Sparse autoencoders on an RNN's hidden state - the engine for the ch8 SAE homework.

A small GRU learns to spell: the corpus is a random sequence of words drawn from a fixed
40-word lexicon, and the net predicts the next character. To finish a word it must carry
"which word am I inside" in its 32-dim hidden state. That gives us 40 concepts, each active on
about 2.5% of positions, competing for 32 dimensions - so the identities cannot each own a
neuron, and superposition is forced rather than hoped for.

Ground truth is exact and free (we generated the corpus), so every interpretability claim is
SCORED, not eyeballed:

  1. best single neuron   - can any one unit read "inside word k"?
  2. linear probe         - is the information in the hidden state at all?
  3. best SAE feature     - does a wide sparse dictionary recover it?
  4. causal ablation      - does deleting that feature actually break the prediction?

DESIGN NOTE (measured, 2026-08-02). A first version used nesting depth in a bracket language.
It failed, and the failure is instructive: depth and stack-top are DENSE concepts (active on
30-50% of positions), and an L1 dictionary has no advantage on those - the SAE lost to a single
neuron at every lambda tried. Sparse dictionary learning needs sparse concepts. See
logs/sae_rnn_lab.log and the chapter plan for the numbers.

Compute guardrails (per AE_CHAPTER_PLAN.md): thread-capped to 4, tiny nets, CPU only, and no
downloads at all - the corpus is generated. Seed 509.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch8_autoencoders/py_src/sae_rnn_lab.py
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

# -- freeze-safety + reproducibility -----------------------------------------------------------
torch.set_num_threads(4)          # do NOT peg all cores (documented lock-up risk on this laptop)
SEED = 509
torch.manual_seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent           # ml/ch8_autoencoders/py_src
CH = HERE.parent                                  # ml/ch8_autoencoders
FIG = CH / "fig"
LOGS = CH / "logs"
for d in (FIG, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "sae_rnn_lab.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("sae_rnn")

LETTERS = "abcdefghijklmnopqrstuvwxyz"
N_WORDS = 40
HIDDEN = 32
SEQ = 80


# -- the language ------------------------------------------------------------------------------
def make_lexicon(n_words, seed):
    """n distinct pronounceable-length nonsense words. Nonsense on purpose: no prior knowledge."""
    r = np.random.default_rng(seed)
    words, seen = [], set()
    while len(words) < n_words:
        w = "".join(r.choice(list(LETTERS), size=int(r.integers(3, 7))))
        if w not in seen:
            seen.add(w)
            words.append(w)
    return words


def make_corpus(lex, n_chars, seed):
    """Space-separated random words. Labels: word index and offset within the word (-1 on spaces)."""
    r = np.random.default_rng(seed)
    chars, wid, off = [], [], []
    while len(chars) < n_chars:
        k = int(r.integers(len(lex)))
        for j, c in enumerate(lex[k]):
            chars.append(c)
            wid.append(k)
            off.append(j)
        chars.append(" ")
        wid.append(-1)
        off.append(-1)
    return "".join(chars), np.array(wid), np.array(off)


class CharRNN(nn.Module):
    """One GRU layer + a linear head. Deliberately narrow: the hidden state is the object of study."""

    def __init__(self, vocab, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.rnn = nn.GRU(8, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x, return_hidden=False):
        h, _ = self.rnn(self.emb(x))
        logits = self.head(h)
        return (logits, h) if return_hidden else logits


def train_rnn(x, y, vocab, hidden, epochs=10, batch=64, lr=5e-3, seed=SEED):
    torch.manual_seed(seed)
    model = CharRNN(vocab, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    tot = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        tot = 0.0
        for i in range(0, len(x), batch):
            idx = perm[i : i + batch]
            loss = lossf(model(x[idx]).reshape(-1, vocab), y[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        tot /= len(x)
        if (ep + 1) % 5 == 0:
            log.info("  rnn epoch %2d/%d  loss %.4f", ep + 1, epochs, tot)
    return model, tot


# -- scoring -----------------------------------------------------------------------------------
def f1_all_concepts(M, labels):
    """Best-threshold F1 for every (unit, concept) pair -> {concept: (n_units,) array}.

    The argsort is the expensive step, so it is done ONCE per polarity and shared across all
    concepts. Both polarities are tried, so a unit that fires LOW on a concept is not unfairly
    penalized: a deliberately strong baseline for the single-neuron comparison.
    """
    out = {name: np.zeros(M.shape[1], dtype=np.float32) for name in labels}
    for sign in (1.0, -1.0):
        order = np.argsort(-sign * M, axis=0)
        for name, lab in labels.items():
            ls = lab.astype(np.float32)[order]
            tp = np.cumsum(ls, axis=0)
            fp = np.cumsum(1.0 - ls, axis=0)
            npos = ls.sum(0)
            f1 = 2 * tp / np.maximum(2 * tp + fp + (npos - tp), 1e-9)
            np.maximum(out[name], f1.max(axis=0), out=out[name])
    return out


def probe_f1(X, lab):
    """F1 of a logistic-regression probe on the raw hidden state: is the concept THERE at all?"""
    clf = LogisticRegression(max_iter=1000).fit(X, lab)
    p = clf.predict(X)
    tp = float((p * lab).sum())
    return 2 * tp / max(2 * tp + float((p * (1 - lab)).sum()) + float(((1 - p) * lab).sum()), 1.0)


class SAE(nn.Module):
    """Standard interpretability SAE: ReLU encoder, unit-norm decoder columns, L1 on the code."""

    def __init__(self, d_in, d_hidden, seed=SEED):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden, generator=g) / np.sqrt(d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.randn(d_hidden, d_in, generator=g) / np.sqrt(d_hidden))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        """Unit-norm decoder columns: otherwise the net shrinks the code to dodge the L1 penalty."""
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def encode(self, h):
        return torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, h):
        z = self.encode(h)
        return self.decode(z), z


def train_sae(X, d_hidden, l1, epochs=20, batch=512, lr=3e-3, seed=SEED):
    """Train an SAE on (already normalized) activations. Returns (sae, L0, variance explained)."""
    torch.manual_seed(seed)
    sae = SAE(X.shape[1], d_hidden, seed=seed)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            xb = X[perm[i : i + batch]]
            xh, z = sae(xb)
            loss = ((xh - xb) ** 2).sum(1).mean() + l1 * z.abs().sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()
    with torch.no_grad():
        xh, z = sae(X)
        l0 = (z > 0).float().sum(1).mean().item()
        fvu = ((xh - X) ** 2).sum().item() / ((X - X.mean(0)) ** 2).sum().item()
    return sae, l0, 1.0 - fvu


def normalize_acts(H):
    """Center, then scale so the mean activation norm is sqrt(d): makes l1 comparable across runs."""
    Hc = H - H.mean(0, keepdims=True)
    return Hc * (np.sqrt(H.shape[1]) / np.linalg.norm(Hc, axis=1).mean())


def main():
    log.info("=" * 78)
    log.info("SAE-on-RNN lab: 40-word lexicon, hidden dim %d", HIDDEN)

    lex = make_lexicon(N_WORDS, SEED)
    text, wid, off = make_corpus(lex, 120000, SEED)
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    log.info("lexicon: %s ...", " ".join(lex[:10]))
    log.info("corpus %d chars, vocab %d, per-word base rate %.3f", len(text), len(vocab),
             1.0 / N_WORDS)

    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_seq = (len(ids) - 1) // SEQ
    x = ids[: n_seq * SEQ].view(n_seq, SEQ)
    y = ids[1 : n_seq * SEQ + 1].view(n_seq, SEQ)
    model, loss = train_rnn(x, y, len(vocab), HIDDEN)
    log.info("RNN final loss %.4f (uniform %.4f)", loss, np.log(len(vocab)))
    if loss > 0.6 * np.log(len(vocab)):
        raise SystemExit(f"RNN did not learn the lexicon (loss {loss:.4f}) - nothing to interpret.")

    with torch.no_grad():
        _, h = model(x, return_hidden=True)
    H = h.reshape(-1, HIDDEN).numpy()
    pos = np.arange(n_seq * SEQ)
    w, o = wid[pos], off[pos]

    # Train the SAE on every in-word position; SCORE where the word is already determined
    # (offset >= 2). At offset 0-1 many words still share a prefix, so the label is not yet
    # knowable and would cap every method's F1 for reasons that have nothing to do with the SAE.
    inword = o >= 0
    Xn = torch.tensor(normalize_acts(H[inword]), dtype=torch.float32)
    w_in, o_in = w[inword], o[inword]
    H_in = H[inword]

    score_idx = np.where(o_in >= 2)[0]
    rng = np.random.default_rng(SEED)
    if len(score_idx) > 15000:
        score_idx = rng.choice(score_idx, 15000, replace=False)
    ws = w_in[score_idx]
    concepts = {f"w{k}": (ws == k).astype(np.int8) for k in range(N_WORDS)
                if (ws == k).sum() >= 20}
    log.info("scoring %d positions (offset>=2) over %d word concepts", len(score_idx),
             len(concepts))

    neuron = f1_all_concepts(H_in[score_idx], concepts)
    neuron_mean = float(np.mean([v.max() for v in neuron.values()]))
    probe_mean = float(np.mean([probe_f1(H_in[score_idx], lab) for lab in concepts.values()]))
    log.info("-" * 78)
    log.info("best single NEURON  mean F1 = %.3f", neuron_mean)
    log.info("linear PROBE        mean F1 = %.3f   <- the information IS in there", probe_mean)

    log.info("-" * 78)
    log.info("%6s %6s %7s %7s %6s %8s %8s", "width", "l1", "L0", "varexp", "alive", "SAE F1",
             "vs neuron")
    best = None
    for width in (256, 512):
        for l1 in (0.2, 0.5, 1.0):
            sae, l0, ve = train_sae(Xn, width, l1)
            with torch.no_grad():
                Z = sae.encode(Xn).numpy()
            alive = (Z > 0).sum(0) > 0
            f1 = f1_all_concepts(Z[score_idx], concepts)
            sae_mean = float(np.mean([np.where(alive, v, -1).max() for v in f1.values()]))
            log.info("%6d %6.2f %7.2f %7.3f %6d %8.3f %+8.3f", width, l1, l0, ve,
                     int(alive.sum()), sae_mean, sae_mean - neuron_mean)
            if best is None or sae_mean > best[0]:
                best = (sae_mean, width, l1, l0, ve, int(alive.sum()))
    log.info("-" * 78)
    log.info("BEST: width=%d l1=%.2f -> SAE F1 %.3f (neuron %.3f, probe %.3f), L0 %.2f, "
             "varexp %.3f, alive %d", best[1], best[2], best[0], neuron_mean, probe_mean,
             best[3], best[4], best[5])


if __name__ == "__main__":
    main()
