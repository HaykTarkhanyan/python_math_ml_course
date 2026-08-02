"""Assemble ml/ch8_autoencoders/HW1_sae_rnn_solution.ipynb.

Solution notebook FIRST (it must run, so every number in the prose is real); the task version is
derived from it mechanically afterwards by build_sae_tasks.py.
"""
import io
from pathlib import Path

import nbformat as nbf

CH = Path(r"C:\Users\hayk_\OneDrive\Desktop\01_python_math_ml_course\ml\ch8_autoencoders")
CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    # the cell bodies are raw strings, so an escaped docstring quote arrives as a LITERAL
    # backslash-quote and would be a syntax error in the notebook. Undo that here.
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n").replace('\\"', '"')))


# ======================================================================================
md(r"""
# HW1 - Read the mind of an RNN

> This is the **solution** notebook; the task version is `HW1_sae_rnn.ipynb`.

In [22] you met the **sparse autoencoder (SAE)**: point an autoencoder at *another network's
hidden state* and it splits that state into features you can name. Anthropic uses this to read
what is inside Claude. You are going to do the same thing, end to end, on a network small enough
to fit in your head.

**The trick that makes this a real experiment:** we will *generate* the training text ourselves.
That means we know the ground truth at every single step - so when you claim "feature 299 detects
the word *iixxz*", that claim gets **scored**, not admired.

**The plan**

| Part | Question |
|---|---|
| 0 | Train a tiny RNN that learns to spell |
| 1 | Is "which word am I in" written on any single neuron? |
| 2 | Train a sparse autoencoder on its hidden states |
| 3 | Read the features: what does feature #k fire on? |
| 4 | Score the features against ground truth |
| 5 | **Ablate** a feature - correlation vs causation |
| 6 | The sparsity knob, and what it costs |
| 7 | When SAEs *don't* help (a negative result worth knowing) |

Everything runs on **CPU** in a few minutes. No downloads: the corpus is generated.
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

# ---------------------------------------------------------------- Part 0
md(r"""
---
## Part 0 - A tiny RNN that learns to spell

Our language could not be simpler: pick a word at random from a **40-word lexicon**, write it,
write a space, repeat. The words are nonsense (`jvn`, `iixxz`, `oof`) so nothing you know about
English can help - or mislead.

The RNN reads one character at a time and predicts the next one. Here is the key point:

> To finish a word correctly, the network **must** carry *which word am I inside* in its hidden
> state. That is 40 possible identities, and the hidden state is only **32 numbers** wide.

40 concepts, 32 dimensions. They cannot each have their own neuron.
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

lex = make_lexicon(N_WORDS, SEED)
text, wid, off = make_corpus(lex, 60000, SEED)
vocab = sorted(set(text)); stoi = {c: i for i, c in enumerate(vocab)}
print("lexicon:", " ".join(lex[:12]), "...")
print("corpus  :", text[:70], "...")
print(f"{len(text)} chars, vocab {len(vocab)}, each word appears ~{100/N_WORDS:.1f}% of the time")
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

ids   = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_seq = (len(ids) - 1) // SEQ
X = ids[:n_seq*SEQ].view(n_seq, SEQ)
Y = ids[1:n_seq*SEQ+1].view(n_seq, SEQ)

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
    if (ep+1) % 5 == 0:
        print(f"epoch {ep+1:2d}  loss {tot/n_seq:.4f}")
final = tot / n_seq
print(f"\nfinal {final:.4f}  vs  {np.log(len(vocab)):.4f} for random guessing")
assert final < 0.6*np.log(len(vocab)), "the RNN did not learn - nothing to interpret"
""")

md(r"""
Let it spell. Give it the first letter or two of a word and it should finish the word - which is
only possible if the identity of the word is sitting somewhere in those 32 numbers.
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

for seed_txt in ("iix", "vux", "jn"):
    print(f"{seed_txt!r:6} -> {continue_text(seed_txt)}")
print("\nreal lexicon words for comparison:", " ".join(lex[:12]))
""")

# ---------------------------------------------------------------- Part 1
md(r"""
---
## Part 1 - Is the word written on a neuron?

Now harvest. Run the whole corpus through the RNN and keep the hidden vector at every step. Each
position gets a label for free: **which word are we inside**.

We only score positions at **offset >= 2** (the third character onwards). At offset 0 or 1 several
words still share a prefix, so the network genuinely *cannot* know which word it is in yet - and
scoring an unanswerable question would drag every method down for reasons that have nothing to do
with autoencoders.

Two questions, and the gap between them is the whole motivation for SAEs:

1. **Can one neuron tell you?** Try every neuron, every threshold, both directions. Best F1 wins.
2. **Can all 32 together tell you?** Fit a logistic regression (a "linear probe") on the full vector.
""")

code(r"""
with torch.no_grad():
    _, h = model(X, return_hidden=True)
H   = h.reshape(-1, HIDDEN).numpy()
pos = np.arange(n_seq*SEQ)
w, o = wid[pos], off[pos]
print("harvested", H.shape, "hidden states")

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

inword = o >= 0
H_in, w_in, o_in = H[inword], w[inword], o[inword]
score_idx = np.where(o_in >= 2)[0]
score_idx = rng.choice(score_idx, min(8000, len(score_idx)), replace=False)
ws = w_in[score_idx]
concepts = {f"w{k}": (ws == k).astype(np.int8) for k in range(N_WORDS) if (ws == k).sum() >= 20}
print(f"scoring {len(score_idx)} positions over {len(concepts)} word concepts")
""")

code(r"""
neuron = f1_all_units(H_in[score_idx], concepts)
neuron_best = np.array([v.max() for v in neuron.values()])

def probe_f1(Xm, lab):
    clf = LogisticRegression(max_iter=1000).fit(Xm, lab)
    p = clf.predict(Xm); tp = float((p*lab).sum())
    return 2*tp / max(2*tp + float((p*(1-lab)).sum()) + float(((1-p)*lab).sum()), 1.0)

probe = np.array([probe_f1(H_in[score_idx], lab) for lab in concepts.values()])
print(f"best single NEURON   mean F1 = {neuron_best.mean():.3f}")
print(f"linear PROBE         mean F1 = {probe.mean():.3f}")
""")

md(r"""
**Read that carefully.** The probe is near-perfect, so the word identity really *is* in the hidden
state. But no single neuron comes close. The information is there; the network's own coordinates
are simply the wrong basis to read it in.

Why would that be? Two reasons:

- **Nothing asks the axes to mean anything.** A concept can live on a direction like
  `0.3*h4 - 0.7*h11 + ...`. Nothing in the loss rewards aligning it with a coordinate axis.
- **Superposition.** 40 word identities do not fit in 32 orthogonal directions. They *can* fit as
  40 nearly-orthogonal ones, and that works precisely because the concepts are **sparse**: at any
  step you are inside exactly one word.

An SAE is built for exactly this: find the directions, using sparsity as the clue.
""")

# ---------------------------------------------------------------- Part 2
md(r"""
---
## Part 2 - The sparse autoencoder

Same encoder/decoder/L1 as [22], with three changes that matter:

- **The data is the activations.** Input = a 32-dim hidden vector, not an image.
- **It is wide, not narrow.** 256 features for a 32-dim input (8x overcomplete). The bottleneck is
  no longer size - it is **sparsity**.
- **Unit-norm decoder columns.** Otherwise the net cheats: it shrinks the code towards zero to
  dodge the L1 penalty and grows the decoder to compensate, which costs nothing and teaches nothing.

$$\mathbf{z} = \mathrm{ReLU}\big(W_{enc}(\mathbf{h}-\mathbf{b}) + \mathbf{b}_{enc}\big),
\qquad \hat{\mathbf{h}} = \mathbf{b} + \sum_{i\,:\,z_i>0} z_i \mathbf{d}_i,
\qquad \mathcal{L} = \|\mathbf{h}-\hat{\mathbf{h}}\|^2 + \lambda\|\mathbf{z}\|_1$$

The decoder columns $\mathbf{d}_i$ are the **dictionary**: each one is a direction in the RNN's
hidden space that we hope means something.
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
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
    def encode(self, h): return torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)
    def decode(self, z): return z @ self.W_dec + self.b_dec
    def forward(self, h):
        z = self.encode(h); return self.decode(z), z

def normalize_acts(A):
    \"\"\"Center and rescale so the mean norm is sqrt(d): makes lambda mean the same thing run to run.\"\"\"
    Ac = A - A.mean(0, keepdims=True)
    return Ac * (np.sqrt(A.shape[1]) / np.linalg.norm(Ac, axis=1).mean())

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

WIDTH, L1 = 256, 0.2
Xn = torch.tensor(normalize_acts(H_in), dtype=torch.float32)
sae, l0, ve = train_sae(Xn, WIDTH, L1)
with torch.no_grad():
    Z = sae.encode(Xn).numpy()
alive = (Z > 0).sum(0) > 0
print(f"{HIDDEN} -> {WIDTH} features | L0 = {l0:.2f} active at once | "
      f"variance explained {ve:.3f} | alive {alive.sum()}/{WIDTH}")
""")

# ---------------------------------------------------------------- Part 3
md(r"""
---
## Part 3 - Read a feature

The standard first move in interpretability: take a feature, find the inputs that make it fire
hardest, and look at them. This is a **feature dashboard**.
""")

code(r"""
idx_in = np.where(inword)[0]

def dashboard(feat, k=10, ctx=10):
    acts = Z[:, feat]
    top = np.argsort(-acts)[:k]
    print(f"--- feature #{feat} | fires on {100*(acts>0).mean():.2f}% of positions ---")
    for t in top:
        p = int(idx_in[t])
        print(f"  {acts[t]:6.3f}   ...{text[max(0,p-ctx):p]}[{text[p]}]{text[p+1:p+5]}...")

busiest = np.argsort(-Z.max(0))[:3]
for f in busiest:
    dashboard(int(f)); print()
""")

md(r"""
If the SAE worked, each of those blocks should be **one word, one position** - the bracketed
character is always in the same spot of the same word. That is a *monosemantic* feature: it means
one thing.

Compare that with the same exercise on a raw neuron, which is what you would be stuck with
without the SAE.
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

# ---------------------------------------------------------------- Part 4
md(r"""
---
## Part 4 - Score it

Eyeballing dashboards is how interpretability goes wrong: you will always find *some* pattern.
Because we generated the corpus, we can do better and **measure**.

For every word, find the SAE feature that best detects "am I inside this word" and record its F1.
Then compare against the two baselines from Part 1.
""")

code(r"""
f1 = f1_all_units(Z[score_idx], concepts)
sae_best = np.array([np.where(alive, v, -1).max() for v in f1.values()])

print(f"best single neuron : {neuron_best.mean():.3f}")
print(f"best SAE feature   : {sae_best.mean():.3f}")
print(f"linear probe       : {probe.mean():.3f}   <- supervised: the ceiling, not a rival")

order = np.argsort(-sae_best)
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(probe[order], "o-", color=ORANGE, ms=3, lw=1.2, label=f"linear probe  {probe.mean():.2f}")
ax.plot(sae_best[order], "o-", color=BLUE, ms=3, lw=1.2, label=f"best SAE feature  {sae_best.mean():.2f}")
ax.plot(neuron_best[order], "o-", color=RED, ms=3, lw=1.2, label=f"best single neuron  {neuron_best.mean():.2f}")
ax.set_xlabel("word (sorted by SAE score)"); ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""
**The result.** The SAE beats single neurons by a wide margin - it has genuinely pulled concepts
out of a basis where they were smeared across many units.

It does *not* reach the probe, and that is expected rather than disappointing. The probe is
**supervised**: we hand it the labels and ask it to find a direction. The SAE is shown **no labels
at all** and has to discover the concept unprompted. They answer different questions:

- probe: *is the information in there?*
- SAE: *can it be found without being told what to look for?*
""")

# ---------------------------------------------------------------- Part 5
md(r"""
---
## Part 5 - Ablate it (correlation vs causation)

So far every claim is **correlational**: the feature lights up when the RNN is inside a word. That
does not prove the RNN *uses* it.

The test is intervention. Take positions mid-word, encode the hidden state, **set that one feature
to zero**, decode back, and let the RNN's output head predict the next character. If the feature
was load-bearing, the prediction gets worse.

And a control: ablate a **random** feature instead. Without a control this is not evidence.

One subtlety that decides whether this works at all: **ablate a feature that is actually doing the
job you are testing.** A feature meaning "somewhere inside the word *fehhzk*" barely moves the next
character - the network does not need it for that. A feature meaning "**at letter 4 of fehhzk**"
does, because that is precisely what determines letter 5.

So we score *position-specific* concepts here, not the word-level ones from Part 4. This is not a
technicality; picking the wrong feature makes a working method look dead.
""")

code(r"""
# The causal test needs a POSITION-SPECIFIC concept. "Somewhere inside word k" is not what
# decides the next character - "at letter j of word k" is. So score those instead and take the
# best one. (Ablating an abstract word-identity feature barely moves the prediction, which is a
# true fact about the model, not a broken experiment.)
ws_, os_ = w_in[score_idx], o_in[score_idx]
pos_con = {}
for k in range(N_WORDS):
    for j in range(2, len(lex[k]) - 1):        # need another in-word character after it
        m = ((ws_ == k) & (os_ == j)).astype(np.int8)
        if m.sum() >= 25:
            pos_con[f"w{k}@{j}"] = m
pf1 = f1_all_units(Z[score_idx], pos_con)
best_key = max(pos_con, key=lambda kk: np.where(alive, pf1[kk], -1).max())
feat = int(np.where(alive, pf1[best_key], -1).argmax())
word_idx, off_j = int(best_key[1:].split("@")[0]), int(best_key.split("@")[1])
word = lex[word_idx]
print(f"letter {off_j+1} of {word!r}  ->  feature #{feat}  "
      f"(F1 {np.where(alive, pf1[best_key], -1).max():.3f})")

mu    = torch.tensor(H_in.mean(0), dtype=torch.float32)
scale = float(np.sqrt(HIDDEN) / np.linalg.norm(H_in - H_in.mean(0), axis=1).mean())

cand = np.where((w == word_idx) & (o == off_j))[0]
cand = cand[cand + 1 < len(w)][:400]
assert len(cand) > 0, "no positions found for that (word, offset)"
nxt = torch.tensor([stoi[text[i+1]] for i in cand], dtype=torch.long)
Hc  = torch.tensor(H[cand], dtype=torch.float32)

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
print(f"\nP(correct next char)")
print(f"  SAE reconstruction        {pb:.3f}")
print(f"  ablate feature #{feat:<4d}     {pa:.3f}   (drop {pb-pa:+.3f})")
print(f"  ablate random #{rand_feat:<4d}      {pc:.3f}   (drop {pb-pc:+.3f})  <- control")

fig, ax = plt.subplots(figsize=(5, 3))
b = ax.bar(["word\nfeature", "random\n(control)"], [pb-pa, pb-pc], color=[RED, "#999999"])
ax.bar_label(b, fmt="%.3f"); ax.set_ylabel("drop in P(correct)")
ax.set_title(f"deleting one feature while spelling {word!r}", fontsize=10)
ax.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- Part 6
md(r"""
---
## Part 6 - The sparsity knob

$\lambda$ trades reconstruction against sparsity. Turn it up, fewer features fire, and the
reconstruction gets worse. The interesting question is what happens to **interpretability**, which
the loss function cannot see at all.
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

fig, ax = plt.subplots(figsize=(6, 3.3))
ax.plot([r[1] for r in rows], [r[2] for r in rows], "o-", color=BLUE, label="reconstruction")
ax.plot([r[1] for r in rows], [r[3] for r in rows], "s-", color=RED, label="interpretability (F1)")
ax.set_xscale("log"); ax.set_xlabel("$L_0$ (features active at once)"); ax.set_ylabel("score")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""
Note what this means in practice: **the loss cannot tell you the right $\lambda$.** Reconstruction
improves monotonically as you loosen the penalty, so "minimize the loss" would push you toward a
dense code that is no more interpretable than the neurons you started with. Choosing $\lambda$
requires a judgement the objective does not contain.
""")

# ---------------------------------------------------------------- Part 7
md(r"""
---
## Part 7 - When SAEs do *not* help

SAEs are not a universal decoder ring. They exploit one specific property: **the features are
sparse** - only a few active at a time. When a concept is *dense*, the whole premise fails.

Here is that failure, measured. New language: nested brackets, `( a b ( c ) )`. The RNN has to
track the **nesting depth** to know when a closing bracket is legal. Depth is a perfectly real
thing the network computes - but it is dense: every single position has a depth.
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

bidx = rng.choice(len(BH), 8000, replace=False)
bcon = {f"depth={d}": (bd[bidx] == d).astype(np.int8) for d in range(MAXD+1)}
for k, v in bcon.items():
    print(f"  {k}: fires on {100*v.mean():5.1f}% of positions   <- DENSE")

bn_f1 = f1_all_units(BH[bidx], bcon)
bXn = torch.tensor(normalize_acts(BH), dtype=torch.float32)
bsae, bl0, bve = train_sae(bXn, WIDTH, 0.2)
with torch.no_grad():
    BZ = bsae.encode(bXn).numpy()
bal = (BZ > 0).sum(0) > 0
bs_f1 = f1_all_units(BZ[bidx], bcon)

print(f"\n{'concept':12} {'neuron':>8} {'SAE':>8} {'delta':>8}")
for k in bcon:
    a, b = bn_f1[k].max(), np.where(bal, bs_f1[k], -1).max()
    print(f"{k:12} {a:8.3f} {b:8.3f} {b-a:+8.3f}")
""")

md(r"""
**The SAE gains nothing here, and can even lose.** Depth fires on 20-40% of positions; there is no
sparse structure for a sparse dictionary to exploit, so the L1 penalty is just a handicap.

This is the single most useful thing to carry out of this homework:

> A sparse autoencoder is not a general-purpose "make it interpretable" button. It is a bet that
> the thing you are looking for is **sparse**. When that bet is wrong, the tool quietly fails -
> and it fails by giving you plausible-looking features, not an error message.

Real models are thought to be mostly in the sparse regime, which is why the technique works on
LLMs at all. But "mostly" is doing real work in that sentence.
""")

# ---------------------------------------------------------------- Bonus
md(r"""
---
## Bonus - do the features survive a different seed? üéÅ

Train a **second** RNN on the same corpus with a different initialization, run the same SAE
pipeline, and compare. If features were a property of the *language*, both runs should find them.
If they are a property of *this particular network*, they may not agree.

This is an open research question, not a settled one - which is a fair note to end on.
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
X2 = torch.tensor(normalize_acts(H2), dtype=torch.float32)
sae2, l02, ve2 = train_sae(X2, WIDTH, L1, seed=1234)
with torch.no_grad():
    Z2 = sae2.encode(X2).numpy()
a2 = (Z2 > 0).sum(0) > 0
f2 = f1_all_units(Z2[score_idx], concepts)
s2 = np.array([np.where(a2, v, -1).max() for v in f2.values()])

print(f"run 1 mean F1 {sae_best.mean():.3f}   run 2 mean F1 {s2.mean():.3f}")
print(f"per-word correlation between runs: {np.corrcoef(sae_best, s2)[0,1]:.3f}")
fig, ax = plt.subplots(figsize=(4, 3.8))
ax.scatter(sae_best, s2, c=BLUE, s=18); ax.plot([0,1],[0,1],"--",c="grey",lw=1)
ax.set_xlabel("run 1 (seed 509)"); ax.set_ylabel("run 2 (seed 1234)")
ax.set_title("is a word equally findable\nin both networks?", fontsize=9)
ax.grid(alpha=.3); plt.tight_layout(); plt.show()
""")

md(r"""
---
## What to hand in

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
print(f"wrote {out}  ({len(CELLS)} cells)")
