# The idea: a prior-data fitted network

## The move

Ordinary supervised learning fits parameters to *your* dataset. A **Prior-Data Fitted Network**
(**PFN**) inverts that:

1. Write down a **prior** - an explicit generative process over datasets.
2. Sample millions of synthetic datasets from it.
3. Train one transformer to, given `(X_train, y_train, X_test)`, output `p(y_test | ...)`.
4. Ship that transformer. It is never trained again.

At use time there is **no fitting**. The user's entire training table is fed in as *context*,
the test rows come after it, and the answer falls out of one forward pass. This is exactly
in-context learning, applied to tables instead of text.

The objective the network is trained toward is the **posterior predictive distribution** under
the prior. That is the key to everything downstream: unlike "what algorithm is GPT-2 running",
"is this model doing Bayesian inference?" is a precise question with a checkable answer, because
somebody wrote the prior down.

## What the prior actually is

TabPFN's pre-training data is **synthetic**, sampled from a prior over **structural causal
models** and **Bayesian neural networks**. No real tables are involved in pre-training the core
model. This is the part students find hardest to believe, and it is the part worth dwelling on:
the model has never seen a real dataset, and it still beats tuned gradient boosting on real
ones.

Two consequences, both teachable:

- **The prior is a design artifact, not data.** Someone chose it. Change it and you change the
  model's inductive bias directly - which is why `Real-TabPFN` (continued pre-training on real
  data) and `Drift-Resilient TabPFN` (a prior that includes temporal shift) are the natural
  follow-ups.
- **The prior may not wash out.** Melnychuk et al. (2026) show PFN-based average-treatment-effect
  estimators carry **prior-induced confounding bias that is not asymptotically overwritten by
  data**. For a course that teaches consistency and asymptotics, that is a sharp, concrete
  result.

## The architecture, and why it is not a normal transformer

From the Nature paper (Hollmann et al., 2025), as summarised in the instructor's
`mech_interp/research_ideas/tabpfn_mech_interp.md`:

- Each **cell** of the table gets its own representation. Not each row, not each token - each
  cell.
- Each layer applies, in order:
  1. **attention over features** - a cell attends to the other features in its own row;
  2. **attention over samples** - a cell attends to the same feature down its own column;
  3. an MLP sublayer.
  Each with residual connections and normalisation.
- **Random feature embeddings** are added before layer 1, to break the symmetry between
  identically distributed features.
- Training-sample representations are **not** influenced by test samples.

So the residual stream is indexed by **(row, column)**, not by position in a 1-D token stream.
There are two structurally distinct attention mechanisms rather than one.

Two things follow that are worth a slide each:

- **The invariances are architectural.** Permuting rows or columns of a table should not change
  the prediction, and here that is enforced by construction rather than learned. Compare this to
  a CNN's translation equivariance (ch6) - the same design idea, a different symmetry.
- **Most interpretability tooling assumes a 1-D token stream** and therefore does not transfer
  cleanly. Biloš et al. (2026) trace the permutation invariances to specific positional
  parameters, and find those can be removed without hurting accuracy.

## Why it is fast

There are no gradient steps at inference. The cost is one forward pass over a context holding
the whole training set, so the practical limit is context length, not epochs. That reframes the
usual scaling question: for a PFN, "bigger dataset" means "longer context", which is why so much
of the literature (`scaling` category in the manifest: sketching, distillation, TuneTables,
chunking) is about getting more rows into the context rather than about training longer.

## Terms

- **PFN** - prior-data fitted network. The general idea.
- **TabPFN** - the specific model line from Prior Labs for tabular data.
- **ICL** - in-context learning. Already taught in `llm_training`; no need to re-explain.

Search all three names. The instructor's notes record that searching only "TabPFN" missed a
~25-paper theory literature filed under "prior-data fitted networks", and missed two 2026
mechanistic papers filed under "tabular foundation models". Product name, architecture name and
concept name each retrieve a different slice.
