"""Assemble ml/ch8_autoencoders/HW1_sae_rnn_solution.ipynb.

Solution notebook FIRST (it must run, so every number in the prose is real); the task version is
derived from it mechanically afterwards by build_sae_tasks.py.

Structure rule (instructor direction, 2026-08-02): MANY SMALL CELLS. One idea per cell, with a
markdown cell before it saying what is about to happen and why. Never bundle a class definition,
its training loop and its evaluation into a single cell.

Run with the project venv:
    ./ma/Scripts/python.exe ml/ch8_autoencoders/py_src/build_sae_nb.py
"""
import io
from pathlib import Path

import nbformat as nbf

CH = Path(r"C:\Users\hayk_\OneDrive\Desktop\01_python_math_ml_course\ml\ch8_autoencoders")
CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    # cell bodies are raw strings, so an escaped docstring quote arrives as a LITERAL
    # backslash-quote and would be a syntax error in the notebook. Undo that here.
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n").replace('\\"', '"')))


# ======================================================================================
md(r"""
# HW1 - Read the mind of an RNN

> This is the **solution** notebook; the task version is `HW1_sae_rnn.ipynb`.

In [22] you met the **sparse autoencoder (SAE)**: point an autoencoder at *another network's
hidden state*, and it splits that state into features you can name. This is how Anthropic reads
what is inside Claude.

You are going to do exactly that, on a network small enough to understand completely.

### The one idea that makes this work

Real interpretability has a problem: when someone says "we found a Golden Gate Bridge feature
inside this model", **nobody can check it**. There is no list of what the model truly represents.

So we will **generate our own text**. Then we know the correct answer at every position, and every
claim you make gets **scored**, not admired.

### The plan

| Part | Question |
|---|---|
| 0 | Train a tiny RNN that learns to spell |
| 1 | Is "which word am I in" written on any single neuron? |
| 2 | Train a sparse autoencoder on its hidden states |
| 3 | Read the features: what does feature #k fire on? |
| 4 | Score those features against the truth |
| 5 | **Ablate** a feature - correlation vs causation |
| 6 | The sparsity knob, and what it costs |
| 7 | When SAEs *don't* help |

Everything runs on **CPU** in a few minutes. Nothing is downloaded.
""")

code(r"""
import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(4)          # be kind to the laptop
SEED = 509
torch.manual_seed(SEED); np.random.seed(SEED)
rng = np.random.default_rng(SEED)

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
print("torch", torch.__version__)
""")

# ================================================================== PART 0
md(r"""
---
# Part 0 - A tiny RNN that learns to spell

## 0.1 Invent a vocabulary

40 nonsense words. Nonsense on purpose: nothing you know about English can help you, or mislead
you, when you interpret this network later.
""")

code(r"""
LETTERS = "abcdefghijklmnopqrstuvwxyz"
N_WORDS, HIDDEN, SEQ = 40, 32, 80

def make_lexicon(n_words, seed):
    r = np.random.default_rng(seed)
    words, seen = [], set()
    while len(words) < n_words:
        w = "".join(r.choice(list(LETTERS), size=int(r.integers(3, 7))))
        if w not in seen:
            seen.add(w); words.append(w)
    return words

lex = make_lexicon(N_WORDS, SEED)
print(f"{len(lex)} words, lengths {min(map(len,lex))}-{max(map(len,lex))}:\n")
for i in range(0, len(lex), 8):
    print("  " + "  ".join(f"{w:<7}" for w in lex[i:i+8]))
""")

md(r"""
## 0.2 Write a corpus

Pick a word at random, write it, write a space, repeat. That is the whole language.

Crucially we also record, **for every character position**, two labels:

- `wid` - which word this character belongs to (`-1` for spaces)
- `off` - how far into that word we are (`0` = first letter)

These labels are our ground truth, and they are free because we generated the text.
""")

code(r"""
def make_corpus(lex, n_chars, seed):
    \"\"\"Space-separated random words + exact labels: word index, and offset inside the word.\"\"\"
    r = np.random.default_rng(seed)
    chars, wid, off = [], [], []
    while len(chars) < n_chars:
        k = int(r.integers(len(lex)))
        for j, c in enumerate(lex[k]):
            chars.append(c); wid.append(k); off.append(j)
        chars.append(" "); wid.append(-1); off.append(-1)
    return "".join(chars), np.array(wid), np.array(off)

text, wid, off = make_corpus(lex, 60000, SEED)
print(f"{len(text)} characters\n")
print(text[:100], "...")
""")

md(r"""
## 0.3 Look at the ground truth

This little table is what the entire homework rests on. For every position we know the character,
the word it belongs to, and how far into that word we are.
""")

code(r"""
print(f"{'pos':>4} {'char':>6} {'word':>9} {'offset':>7}")
print("-" * 30)
for p in range(18):
    word = lex[wid[p]] if wid[p] >= 0 else "(space)"
    print(f"{p:>4} {repr(text[p]):>6} {word:>9} {off[p]:>7}")
""")

md(r"""
**The network sees only the `char` column.** It has to work out the other two for itself - and it
must, because there is no other way to know which letter comes next.

## 0.4 Turn characters into numbers

27 symbols: 26 letters plus the space.
""")

code(r"""
vocab = sorted(set(text)); stoi = {c: i for i, c in enumerate(vocab)}
print("vocab:", "".join(vocab).replace(" ", "_"), f"  ({len(vocab)} symbols)")

ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
print("first 20 ids:", ids[:20].tolist())
""")

md(r"""
## 0.5 The model

A single GRU layer plus a linear head. Small on purpose: **the hidden state is the object we are
going to study**, so it has to be something we can examine exhaustively.

`return_hidden=True` is the detail that matters - we will need those hidden vectors in Part 1.
""")

code(r"""
class CharRNN(nn.Module):
    \"\"\"One GRU layer + a linear head. The hidden state is what we are here to study.\"\"\"
    def __init__(self, vocab, hidden):
        super().__init__()
        self.emb  = nn.Embedding(vocab, 8)
        self.rnn  = nn.GRU(8, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
    def forward(self, x, return_hidden=False):
        h, _ = self.rnn(self.emb(x))
        logits = self.head(h)
        return (logits, h) if return_hidden else logits

print(CharRNN(len(vocab), HIDDEN))
""")

md(r"""
## 0.6 Cut the text into training sequences

Input `X` is the text; target `Y` is the same text shifted by one character. Predicting the next
character is the entire objective - no labels are involved anywhere.
""")

code(r"""
n_seq = (len(ids) - 1) // SEQ
X = ids[:n_seq*SEQ].view(n_seq, SEQ)
Y = ids[1:n_seq*SEQ+1].view(n_seq, SEQ)
print("X", tuple(X.shape), " Y", tuple(Y.shape))
print("\nX[0] :", "".join(vocab[i] for i in X[0][:40].tolist()))
print("Y[0] :", "".join(vocab[i] for i in Y[0][:40].tolist()), " <- shifted by one")
""")

md(r"""
## 0.7 Train it

About 20 seconds on CPU.
""")

code(r"""
torch.manual_seed(SEED)
model = CharRNN(len(vocab), HIDDEN)
opt   = torch.optim.Adam(model.parameters(), lr=5e-3)
lossf = nn.CrossEntropyLoss()

for ep in range(10):
    perm = torch.randperm(n_seq); tot = 0.0
    for i in range(0, n_seq, 64):
        idx = perm[i:i+64]
        loss = lossf(model(X[idx]).reshape(-1, len(vocab)), Y[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    print(f"epoch {ep+1:2d}  loss {tot/n_seq:.4f}")
final = tot / n_seq
""")

md(r"""
## 0.8 Did it actually learn?

Always check this **before** interpreting anything. A network that learned nothing has nothing
inside it to find, and you would spend the rest of the notebook carefully interpreting noise.

The reference point is `log(27) = 3.30`, the loss of guessing uniformly at random.
""")

code(r"""
print(f"final loss      {final:.4f}")
print(f"random guessing {np.log(len(vocab)):.4f}")
assert final < 0.6*np.log(len(vocab)), "the RNN did not learn - nothing to interpret"
print("\nOK: comfortably better than chance, so it has learned the lexicon.")
""")

md(r"""
## 0.9 Watch it spell

Give it the first couple of letters of a word. If it can finish the word, then "which word am I
inside" **must** be represented somewhere in those 32 numbers.

That single fact is what the rest of the notebook investigates.
""")

code(r"""
@torch.no_grad()
def continue_text(prompt, n=40, temp=0.4):
    ctx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    out = prompt
    for _ in range(n):
        p = torch.softmax(model(ctx)[0, -1] / temp, -1).numpy()
        nxt = vocab[int(rng.choice(len(vocab), p=p/p.sum()))]
        out += nxt
        ctx = torch.cat([ctx, torch.tensor([[stoi[nxt]]])], 1)[:, -SEQ:]
    return out

for seed_txt in ("kiz", "vux", "axd"):
    print(f"{seed_txt!r:6} -> {continue_text(seed_txt)}")
print("\nreal words:", "  ".join(lex[:12]))
""")

# ================================================================== PART 1
md(r"""
---
# Part 1 - Is the word written on a neuron?

## 1.1 Harvest the hidden states

Run the whole corpus through the RNN and keep the hidden vector at every step: one 32-number
vector per character.
""")

code(r"""
with torch.no_grad():
    _, h = model(X, return_hidden=True)
H   = h.reshape(-1, HIDDEN).numpy()
pos = np.arange(n_seq*SEQ)
w, o = wid[pos], off[pos]
print("hidden states:", H.shape, " (one 32-dim vector per character)")
""")

md(r"""
## 1.2 Look at one of them

Here is the network's entire mental state at one moment, side by side with what we know is true.
""")

code(r"""
p = int(np.where((o == 3) & (w >= 0))[0][0])
print(f"context : ...{text[max(0,p-12):p]}[{text[p]}]{text[p+1:p+4]}...")
print(f"truth   : inside {lex[w[p]]!r}, at offset {o[p]}")
print(f"\nhidden state, all {HIDDEN} numbers:")
print(np.array2string(H[p], precision=2, suppress_small=True, max_line_width=88))
""")

md(r"""
Somewhere in those 32 numbers is the fact "I am inside this word, at this letter".

**Which ones?** That question is the whole homework.

## 1.3 A fair way to test a single unit

Before accusing neurons of being unreadable, give them the best possible chance:

- try **every** threshold, not some arbitrary one,
- try **both directions** (a neuron that goes *low* on the concept counts too),
- keep the best F1 any threshold achieves.

If a neuron encodes the concept at all, this will find it.
""")

code(r"""
def f1_all_units(M, labels):
    \"\"\"Best-threshold F1 for EVERY column of M at once -> {concept: (n_units,) array}.

    Sorting is the expensive step, so it happens once per polarity and is shared across all
    concepts. Both polarities are tried, so a unit that fires LOW on a concept still counts:
    a deliberately strong baseline.
    \"\"\"
    out = {n: np.zeros(M.shape[1], dtype=np.float32) for n in labels}
    for sign in (1.0, -1.0):
        order = np.argsort(-sign*M, axis=0)
        for n, lab in labels.items():
            ls = lab.astype(np.float32)[order]
            tp = np.cumsum(ls, axis=0); fp = np.cumsum(1.0-ls, axis=0)
            f1 = 2*tp / np.maximum(2*tp + fp + (ls.sum(0) - tp), 1e-9)
            np.maximum(out[n], f1.max(axis=0), out=out[n])
    return out
""")

md(r"""
## 1.4 Choose where to score

We only score positions at **offset >= 2** (third letter onwards).

Why: at offset 0 or 1 several words still share a prefix, so the network genuinely *cannot* know
which word it is in yet. Scoring an unanswerable question would drag every method down for reasons
that have nothing to do with autoencoders.
""")

code(r"""
inword = o >= 0
H_in, w_in, o_in = H[inword], w[inword], o[inword]

score_idx = np.where(o_in >= 2)[0]
score_idx = rng.choice(score_idx, min(8000, len(score_idx)), replace=False)
ws = w_in[score_idx]

concepts = {f"w{k}": (ws == k).astype(np.int8) for k in range(N_WORDS) if (ws == k).sum() >= 20}
print(f"scoring {len(score_idx)} positions over {len(concepts)} word concepts")
print(f"each concept is true about {100/N_WORDS:.1f}% of the time - they are SPARSE, remember that")
""")

md(r"""
## 1.5 Baseline A - the best single neuron
""")

code(r"""
neuron = f1_all_units(H_in[score_idx], concepts)
neuron_best = np.array([v.max() for v in neuron.values()])
print(f"best single neuron, mean F1 over {len(concepts)} words: {neuron_best.mean():.3f}")
print(f"  easiest word: {neuron_best.max():.3f}    hardest word: {neuron_best.min():.3f}")
""")

md(r"""
## 1.6 Baseline B - all 32 neurons together

A **linear probe**: a logistic regression that may use the whole hidden vector. This answers a
different question - not "is it on one neuron" but "is it in there *at all*".
""")

code(r"""
def probe_f1(Xm, lab):
    clf = LogisticRegression(max_iter=1000).fit(Xm, lab)
    p = clf.predict(Xm); tp = float((p*lab).sum())
    return 2*tp / max(2*tp + float((p*(1-lab)).sum()) + float(((1-p)*lab).sum()), 1.0)

probe = np.array([probe_f1(H_in[score_idx], lab) for lab in concepts.values()])
print(f"best single NEURON   mean F1 = {neuron_best.mean():.3f}")
print(f"linear PROBE         mean F1 = {probe.mean():.3f}")
""")

md(r"""
## 1.7 What that gap means

The probe is nearly perfect, so the word identity really **is** in the hidden state. But no single
neuron comes close. Two reasons:

- **Nothing asks the axes to mean anything.** A concept can live on a direction such as
  `0.3*h4 - 0.7*h11 + ...`. Nothing in the training loss rewards lining that direction up with a
  coordinate axis, so it generally does not.
- **Superposition.** 40 word identities do not fit into 32 orthogonal directions. They *do* fit as
  40 nearly-orthogonal ones, and that works precisely because the concepts are **sparse** - at any
  moment you are inside exactly one word.

An SAE is the tool built for this situation: find the directions, using sparsity as the clue.
""")

# ================================================================== PART 2
md(r"""
---
# Part 2 - The sparse autoencoder

## 2.1 What changes from [22]

Same encoder, decoder and $L_1$ penalty you already know. Three differences that matter:

| | ordinary AE (earlier in [22]) | sparse AE (here) |
|---|---|---|
| input | **data** (digit images) | **another model's hidden vectors** |
| code size | smaller than the input | **8x bigger** (32 -> 256) |
| what stops it copying | the narrow bottleneck | **only the sparsity** |

$$\mathbf{z} = \mathrm{ReLU}\big(W_{enc}(\mathbf{h}-\mathbf{b}) + \mathbf{b}_{enc}\big),
\qquad \hat{\mathbf{h}} = \mathbf{b} + \sum_{i\,:\,z_i>0} z_i \mathbf{d}_i$$

The decoder columns $\mathbf{d}_i$ form the **dictionary**: each is one direction in the RNN's
hidden space that we hope carries a single meaning.
""")

code(r"""
class SAE(nn.Module):
    def __init__(self, d_in, d_hidden, seed=SEED):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden, generator=g)/np.sqrt(d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.randn(d_hidden, d_in, generator=g)/np.sqrt(d_hidden))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        \"\"\"Make every dictionary direction a unit vector.\"\"\"
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def encode(self, h):
        return torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, h):
        z = self.encode(h); return self.decode(z), z
""")

md(r"""
## 2.2 Why the decoder must be unit-norm

Without `normalize_decoder` the network finds a cheat: shrink every code value towards zero to
dodge the $L_1$ penalty, and grow the decoder weights to compensate. Reconstruction is unchanged,
the penalty falls, and **nothing is learned**. Renormalizing after each step closes that door.

## 2.3 Normalize the activations too

So that a given $\lambda$ means the same thing from run to run.
""")

code(r"""
def normalize_acts(A):
    \"\"\"Center and rescale so the mean norm is sqrt(d): makes lambda comparable across runs.\"\"\"
    Ac = A - A.mean(0, keepdims=True)
    return Ac * (np.sqrt(A.shape[1]) / np.linalg.norm(Ac, axis=1).mean())

Xn = torch.tensor(normalize_acts(H_in), dtype=torch.float32)
print("SAE training data:", tuple(Xn.shape), " (in-word hidden states only)")
print(f"mean vector norm: {Xn.norm(dim=1).mean():.3f}   target sqrt(32) = {np.sqrt(32):.3f}")
""")

md(r"""
## 2.4 The training loop

The loss is the whole idea in one line: **stay faithful, stay sparse.**

$$\mathcal{L} = \underbrace{\|\mathbf{h}-\hat{\mathbf{h}}\|^2}_{\text{reconstruct}} + \lambda \underbrace{\|\mathbf{z}\|_1}_{\text{be sparse}}$$
""")

code(r"""
def train_sae(Xt, d_hidden, l1, epochs=12, batch=512, lr=3e-3, seed=SEED):
    torch.manual_seed(seed)
    sae = SAE(Xt.shape[1], d_hidden, seed=seed)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            xb = Xt[perm[i:i+batch]]
            xh, z = sae(xb)
            loss = ((xh-xb)**2).sum(1).mean() + l1*z.abs().sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            sae.normalize_decoder()
    with torch.no_grad():
        xh, z = sae(Xt)
        l0  = (z > 0).float().sum(1).mean().item()
        fvu = ((xh-Xt)**2).sum().item() / ((Xt-Xt.mean(0))**2).sum().item()
    return sae, l0, 1.0-fvu
""")

md(r"""
## 2.5 Train it
""")

code(r"""
WIDTH, L1 = 256, 0.2
sae, l0, ve = train_sae(Xn, WIDTH, L1)
with torch.no_grad():
    Z = sae.encode(Xn).numpy()
alive = (Z > 0).sum(0) > 0

print(f"dictionary        : {HIDDEN} -> {WIDTH} features ({WIDTH//HIDDEN}x overcomplete)")
print(f"L0                : {l0:.2f} features active at once (out of {WIDTH})")
print(f"variance explained: {ve:.3f}")
print(f"alive features    : {alive.sum()}/{WIDTH}")
""")

md(r"""
`L0` is the number that matters. Roughly a dozen features out of 256 explain any given hidden
state, so the code really is sparse - which is what makes individual features worth reading.

## 2.6 How often does each feature fire?
""")

code(r"""
rate = (Z > 0).mean(0)
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.hist(100*rate, bins=50, color=BLUE, alpha=.85)
ax.set_xlabel("% of positions where the feature is active"); ax.set_ylabel("number of features")
ax.set_title("most features are rare specialists", fontsize=10); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
print(f"median firing rate: {100*np.median(rate):.2f}% of positions")
""")

# ================================================================== PART 3
md(r"""
---
# Part 3 - Read a feature

## 3.1 The feature dashboard

The standard first move in interpretability: take one feature, find the inputs that make it fire
hardest, and look at them. The active character is shown in `[brackets]`.
""")

code(r"""
idx_in = np.where(inword)[0]

def dashboard(feat, k=10, ctx=10):
    acts = Z[:, feat]
    top = np.argsort(-acts)[:k]
    print(f"--- feature #{feat} | active on {100*(acts>0).mean():.2f}% of positions ---")
    for t in top:
        p = int(idx_in[t])
        print(f"  {acts[t]:6.3f}   ...{text[max(0,p-ctx):p]}[{text[p]}]{text[p+1:p+5]}...")
""")

md(r"""
Let us look at the three most strongly-firing features.
""")

code(r"""
busiest = np.argsort(-Z.max(0))[:3]
for f in busiest:
    dashboard(int(f)); print()
""")

md(r"""
## 3.2 What you should be seeing

Each block should be **one word, at one position** - the bracketed character sitting in the same
slot of the same word every time. That is a **monosemantic** feature: it means exactly one thing.

## 3.3 Now the same thing on a raw neuron

This is what you would be stuck with if you had no SAE.
""")

code(r"""
def neuron_dashboard(unit, k=10, ctx=10):
    acts = H_in[:, unit]
    top = np.argsort(-acts)[:k]
    print(f"--- raw neuron #{unit} ---")
    for t in top:
        p = int(idx_in[t])
        print(f"  {acts[t]:6.3f}   ...{text[max(0,p-ctx):p]}[{text[p]}]{text[p+1:p+5]}...")

neuron_dashboard(int(np.argmax(H_in.std(0))))
""")

md(r"""
Different words, different positions, no single story. That is **polysemanticity**, and it is the
normal condition of a neuron.
""")

# ================================================================== PART 4
md(r"""
---
# Part 4 - Score it

Eyeballing dashboards is how interpretability goes wrong: stare long enough and you will always
find *some* pattern. Because we generated the corpus, we can do better and **measure**.

## 4.1 Best SAE feature per word
""")

code(r"""
f1 = f1_all_units(Z[score_idx], concepts)
sae_best = np.array([np.where(alive, v, -1).max() for v in f1.values()])
print(f"best SAE feature, mean F1: {sae_best.mean():.3f}")
print(f"  words detected almost perfectly (F1 > 0.95): {(sae_best > 0.95).sum()}/{len(sae_best)}")
""")

md(r"""
## 4.2 The three-way comparison
""")

code(r"""
print(f"best single neuron : {neuron_best.mean():.3f}")
print(f"best SAE feature   : {sae_best.mean():.3f}   <- found WITHOUT labels")
print(f"linear probe       : {probe.mean():.3f}   <- supervised: the ceiling, not a rival")
""")

code(r"""
order = np.argsort(-sae_best)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios":[2.3,1]})
ax.plot(probe[order], "o-", color=ORANGE, ms=3, lw=1.2, label=f"linear probe  {probe.mean():.2f}")
ax.plot(sae_best[order], "o-", color=BLUE, ms=3, lw=1.2, label=f"best SAE feature  {sae_best.mean():.2f}")
ax.plot(neuron_best[order], "o-", color=RED, ms=3, lw=1.2, label=f"best single neuron  {neuron_best.mean():.2f}")
ax.set_xlabel("word (sorted by SAE score)"); ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=.3)
b = ax2.bar(["neuron","SAE","probe"], [neuron_best.mean(), sae_best.mean(), probe.mean()],
            color=[RED, BLUE, ORANGE])
ax2.bar_label(b, fmt="%.3f"); ax2.set_ylim(0,1.15); ax2.grid(alpha=.3, axis="y")
ax2.set_ylabel("mean F1")
plt.tight_layout(); plt.show()
""")

md(r"""
## 4.3 Reading the result honestly

The SAE beats single neurons by a wide margin. It has genuinely pulled concepts out of a basis
where they were smeared across many units.

It does **not** reach the probe, and that is expected rather than disappointing:

- the **probe** is handed the labels and asked to find a direction -> *is the information there?*
- the **SAE** is shown no labels at all and must discover the concept unprompted -> *can it be
  found without being told what to look for?*

Different questions. The probe is the ceiling, not a competitor.
""")

# ================================================================== PART 5
md(r"""
---
# Part 5 - Ablate it (correlation vs causation)

Everything so far is **correlational**: a feature lights up when the RNN is inside a word. That
does not show the RNN *uses* it.

The test is intervention: **delete the feature** and see whether the prediction breaks.

## 5.1 Pick the right feature to delete

This step decides whether the experiment works at all.

A feature meaning "somewhere inside *fehhzk*" barely affects the next character - the network does
not need it for that. A feature meaning "**at letter 4 of *fehhzk***" does, because that is exactly
what determines letter 5.

So here we score **position-specific** concepts, not the word-level ones from Part 4. Picking the
wrong sort of feature makes a working method look dead. (Measured: a 0.002 drop the wrong way,
0.09 the right way.)
""")

code(r"""
ws_, os_ = w_in[score_idx], o_in[score_idx]
pos_con = {}
for k in range(N_WORDS):
    for j in range(2, len(lex[k]) - 1):        # need another in-word character after it
        m = ((ws_ == k) & (os_ == j)).astype(np.int8)
        if m.sum() >= 25:
            pos_con[f"w{k}@{j}"] = m
print(f"{len(pos_con)} (word, offset) concepts have enough examples to score")
""")

code(r"""
pf1 = f1_all_units(Z[score_idx], pos_con)
best_key = max(pos_con, key=lambda kk: np.where(alive, pf1[kk], -1).max())
feat = int(np.where(alive, pf1[best_key], -1).argmax())
word_idx, off_j = int(best_key[1:].split("@")[0]), int(best_key.split("@")[1])
word = lex[word_idx]
print(f"letter {off_j+1} of {word!r}  ->  feature #{feat}  "
      f"(F1 {np.where(alive, pf1[best_key], -1).max():.3f})\n")
dashboard(feat, k=6)
""")

md(r"""
## 5.2 Collect the positions to test on

Every place in the corpus where the RNN sits at that exact letter of that exact word.
""")

code(r"""
mu    = torch.tensor(H_in.mean(0), dtype=torch.float32)
scale = float(np.sqrt(HIDDEN) / np.linalg.norm(H_in - H_in.mean(0), axis=1).mean())

cand = np.where((w == word_idx) & (o == off_j))[0]
cand = cand[cand + 1 < len(w)][:400]
assert len(cand) > 0, "no positions found for that (word, offset)"
nxt = torch.tensor([stoi[text[i+1]] for i in cand], dtype=torch.long)
Hc  = torch.tensor(H[cand], dtype=torch.float32)
print(f"{len(cand)} positions; the next character is always {text[cand[0]+1]!r}")
""")

md(r"""
## 5.3 Three conditions

1. **Baseline** - encode and decode the hidden state through the SAE, changing nothing.
2. **Ablated** - the same, but set feature #`feat` to zero before decoding.
3. **Control** - the same, but zero a *random* feature instead.

Condition 3 is what turns this into evidence. Without it, any drop could just be damage from
poking at the hidden state at all.
""")

code(r"""
def p_correct(hm):
    with torch.no_grad():
        return torch.softmax(model.head(hm), -1)[torch.arange(len(nxt)), nxt].mean().item()

with torch.no_grad():
    z0 = sae.encode((Hc - mu)*scale)
    base = sae.decode(z0)/scale + mu

    z1 = z0.clone(); z1[:, feat] = 0.0
    abl  = sae.decode(z1)/scale + mu

    rand_feat = int(rng.choice(np.where(alive)[0]))
    z2 = z0.clone(); z2[:, rand_feat] = 0.0
    ctl  = sae.decode(z2)/scale + mu

pb, pa, pc = p_correct(base), p_correct(abl), p_correct(ctl)
print("P(correct next char)")
print(f"  baseline reconstruction   {pb:.3f}")
print(f"  ablate feature #{feat:<4d}      {pa:.3f}   (drop {pb-pa:+.3f})")
print(f"  ablate random #{rand_feat:<4d}       {pc:.3f}   (drop {pb-pc:+.3f})  <- control")
""")

code(r"""
fig, ax = plt.subplots(figsize=(5, 3))
b = ax.bar(["the identified\nfeature", "a random\nfeature (control)"], [pb-pa, pb-pc],
           color=[RED, "#999999"])
ax.bar_label(b, fmt="%.3f"); ax.set_ylabel("drop in P(correct)")
ax.set_title(f"deleting one feature at letter {off_j+1} of {word!r}", fontsize=10)
ax.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.show()
""")

md(r"""
## 5.4 What you just proved

The feature is not merely *correlated* with that position in that word - the network's prediction
**depends** on it. Removing it costs real accuracy; removing an arbitrary other feature costs
nothing.

This is the same move as **Golden Gate Claude** (Anthropic, May 2024): they clamped one feature to
roughly 10x its usual peak and the model steered every conversation to the bridge. Same
experiment, much larger model.
""")

# ================================================================== PART 6
md(r"""
---
# Part 6 - The sparsity knob

$\lambda$ trades reconstruction against sparsity: turn it up and fewer features fire.

The interesting question is what happens to **interpretability** - which the loss function cannot
see at all.

## 6.1 Sweep it
""")

code(r"""
rows = []
for l1 in (0.02, 0.1, 0.4, 1.5):
    s, l0_, ve_ = train_sae(Xn, WIDTH, l1)
    with torch.no_grad():
        Z_ = s.encode(Xn).numpy()
    al = (Z_ > 0).sum(0) > 0
    f  = f1_all_units(Z_[score_idx], concepts)
    m  = float(np.mean([np.where(al, v, -1).max() for v in f.values()]))
    rows.append((l1, l0_, ve_, m, int(al.sum())))
    print(f"lambda {l1:5.2f} | L0 {l0_:6.2f} | varexp {ve_:.3f} | mean F1 {m:.3f} | alive {al.sum()}")
""")

code(r"""
fig, ax = plt.subplots(figsize=(6.5, 3.3))
ax.plot([r[1] for r in rows], [r[2] for r in rows], "o-", color=BLUE, label="reconstruction")
ax.plot([r[1] for r in rows], [r[3] for r in rows], "s-", color=RED, label="interpretability (F1)")
for r in rows:
    ax.annotate(f"$\\lambda$={r[0]}", (r[1], r[3]), textcoords="offset points",
                xytext=(0,-13), fontsize=7.5, ha="center", color=RED)
ax.set_xscale("log"); ax.set_xlabel("$L_0$ (features active at once)"); ax.set_ylabel("score")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""
## 6.2 The uncomfortable conclusion

**The loss cannot tell you the right $\lambda$.** Reconstruction improves monotonically as you
loosen the penalty, so "minimize the loss" pushes you towards a dense code that is no more
interpretable than the neurons you started with.

Choosing $\lambda$ needs a judgement the objective does not contain. That is a real, current
problem in this field, not a quirk of our toy setup.
""")

# ================================================================== PART 7
md(r"""
---
# Part 7 - When SAEs do *not* help

An SAE is not a universal decoder ring. It exploits one specific property: **the features are
sparse**, only a few active at a time. When a concept is *dense*, that premise fails.

Here is the failure, measured.

## 7.1 A different language: nested brackets

The RNN must track **nesting depth** to know when a closing bracket is legal. Depth is a perfectly
real thing the network computes - but every single position has one, so it is **dense**.
""")

code(r"""
OPEN, CLOSE, LET, MAXD = "([", ")]", "ab", 4
def bracket_corpus(n_chars, seed):
    r = np.random.default_rng(seed); chars, depths, stack = [], [], []
    while len(chars) < n_chars:
        d = len(stack)
        p = (0.60, 0.00, 0.40) if d == 0 else ((0.00, 0.65, 0.35) if d >= MAXD else (0.20, 0.50, 0.30))
        a = r.choice(["open", "close", "letter"], p=p)
        depths.append(d)
        if a == "open":
            k = int(r.integers(2)); chars.append(OPEN[k]); stack.append(k)
        elif a == "close":
            chars.append(CLOSE[stack.pop()])
        else:
            chars.append(LET[int(r.integers(2))])
    return "".join(chars), np.array(depths)

btext, bdepth = bracket_corpus(40000, SEED)
print("text :", btext[:90])
print("depth:", "".join(str(d) for d in bdepth[:90]))
""")

md(r"""
## 7.2 Train an RNN on it

Exactly the same architecture and recipe as Part 0.
""")

code(r"""
bvocab = sorted(set(btext)); bstoi = {c: i for i, c in enumerate(bvocab)}
bids = torch.tensor([bstoi[c] for c in btext], dtype=torch.long)
bn = (len(bids)-1)//SEQ
bX, bY = bids[:bn*SEQ].view(bn, SEQ), bids[1:bn*SEQ+1].view(bn, SEQ)

torch.manual_seed(SEED)
bmodel = CharRNN(len(bvocab), HIDDEN)
bopt = torch.optim.Adam(bmodel.parameters(), lr=5e-3)
for ep in range(10):
    perm = torch.randperm(bn)
    for i in range(0, bn, 64):
        idx = perm[i:i+64]
        loss = lossf(bmodel(bX[idx]).reshape(-1, len(bvocab)), bY[idx].reshape(-1))
        bopt.zero_grad(); loss.backward(); bopt.step()
with torch.no_grad():
    _, bh = bmodel(bX, return_hidden=True)
BH = bh.reshape(-1, HIDDEN).numpy()
bd = bdepth[np.arange(bn*SEQ)]
print("hidden states:", BH.shape)
""")

md(r"""
## 7.3 Confirm the concepts really are dense

This is the entire difference from Part 1, where each word fired on about 2.5% of positions.
""")

code(r"""
bidx = rng.choice(len(BH), 8000, replace=False)
bcon = {f"depth={d}": (bd[bidx] == d).astype(np.int8) for d in range(MAXD+1)}
for k, v in bcon.items():
    print(f"  {k}: active on {100*v.mean():5.1f}% of positions")
""")

md(r"""
## 7.4 Run the identical pipeline
""")

code(r"""
bn_f1 = f1_all_units(BH[bidx], bcon)
bXn = torch.tensor(normalize_acts(BH), dtype=torch.float32)
bsae, bl0, bve = train_sae(bXn, WIDTH, 0.2)
with torch.no_grad():
    BZ = bsae.encode(bXn).numpy()
bal = (BZ > 0).sum(0) > 0
bs_f1 = f1_all_units(BZ[bidx], bcon)

print(f"{'concept':12} {'neuron':>8} {'SAE':>8} {'delta':>8}")
for k in bcon:
    a, b = bn_f1[k].max(), np.where(bal, bs_f1[k], -1).max()
    print(f"{k:12} {a:8.3f} {b:8.3f} {b-a:+8.3f}")
""")

md(r"""
## 7.5 The most useful lesson in this notebook

**The SAE gains nothing here, and mostly loses.** There is no sparse structure for a sparse
dictionary to exploit, so the $L_1$ penalty is pure handicap.

> A sparse autoencoder is not a general-purpose "make it interpretable" button. It is a **bet that
> what you are looking for is sparse**. When that bet is wrong the tool fails *quietly* - it hands
> you plausible-looking features instead of an error message.

Real models are believed to be mostly in the sparse regime, which is why the technique works on
LLMs at all. But "mostly" is doing a lot of work in that sentence.
""")

# ================================================================== BONUS
md(r"""
---
# Bonus - do the features survive a different seed?

Train a **second** RNN on the same corpus with a different initialization and run the identical
pipeline. If features are a property of the *language*, both runs should find them. If they are a
property of *this particular network*, they may not agree.

This is an open research question, not a settled one.
""")

code(r"""
torch.manual_seed(1234)
m2 = CharRNN(len(vocab), HIDDEN)
o2 = torch.optim.Adam(m2.parameters(), lr=5e-3)
for ep in range(10):
    perm = torch.randperm(n_seq)
    for i in range(0, n_seq, 64):
        idx = perm[i:i+64]
        loss = lossf(m2(X[idx]).reshape(-1, len(vocab)), Y[idx].reshape(-1))
        o2.zero_grad(); loss.backward(); o2.step()
with torch.no_grad():
    _, h2 = m2(X, return_hidden=True)
H2 = h2.reshape(-1, HIDDEN).numpy()[inword]
print("second network trained")
""")

code(r"""
X2 = torch.tensor(normalize_acts(H2), dtype=torch.float32)
sae2, l02, ve2 = train_sae(X2, WIDTH, L1, seed=1234)
with torch.no_grad():
    Z2 = sae2.encode(X2).numpy()
a2 = (Z2 > 0).sum(0) > 0
f2 = f1_all_units(Z2[score_idx], concepts)
s2 = np.array([np.where(a2, v, -1).max() for v in f2.values()])

print(f"run 1 mean F1 {sae_best.mean():.3f}   run 2 mean F1 {s2.mean():.3f}")
print(f"per-word correlation between runs: {np.corrcoef(sae_best, s2)[0,1]:.3f}")
""")

code(r"""
fig, ax = plt.subplots(figsize=(4, 3.8))
ax.scatter(sae_best, s2, c=BLUE, s=18); ax.plot([0,1],[0,1],"--",c="grey",lw=1)
ax.set_xlabel("run 1 (seed 509)"); ax.set_ylabel("run 2 (seed 1234)")
ax.set_title("is a word equally findable\nin both networks?", fontsize=9)
ax.grid(alpha=.3); plt.tight_layout(); plt.show()
""")

md(r"""
---
# What to hand in

1. Part 4's three numbers, and one sentence on why the probe is not a fair rival to the SAE.
2. One feature dashboard, with the concept you think it encodes **and** its F1 from Part 4.
3. The Part 5 ablation numbers, including the control. State what the control rules out.
4. Your chosen $\lambda$ from Part 6, and why the loss alone could not have chosen it.
5. Two sentences on Part 7: what property must a concept have for an SAE to find it?
""")

# ======================================================================================
nb = nbf.v4.new_notebook(cells=CELLS)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
out = CH / "HW1_sae_rnn_solution.ipynb"
with io.open(out, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
n_code = sum(1 for c in CELLS if c.cell_type == "code")
print(f"wrote {out}\n  {len(CELLS)} cells ({n_code} code, {len(CELLS)-n_code} markdown)")
