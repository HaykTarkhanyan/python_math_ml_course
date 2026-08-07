# What it actually learned, and the argument about it

Distilled from `mech_interp/research_ideas/tabpfn_mech_interp.md`, which grounds each claim in
one of the 39 collected papers. This is the richest teaching material in the whole topic,
because the field is **openly disagreeing in print** and the disagreement is decidable.

## The live dispute: is it doing Bayesian inference?

This is a genuine, unresolved, three-way argument, and it is the best thing a lecture can be
built around. The course already teaches MLE vs MAP vs Bayesian (`math/23_stat_mle_map.qmd`), so
students have exactly the vocabulary needed.

| Position | Source | Claim |
|---|---|---|
| **Bayesian** | Müller, Hollmann & Hutter, *Bayes' Power for Explaining ICL Generalizations* (2024/10) - **written by the TabPFN authors** | The Bayesian reading, not MLE, is what explains in-context generalisation. |
| **Frequentist** | Nagler, *Statistical Foundations of PFNs* (ICML 2023) | A **purely frequentist** reading works: PFNs are pre-tuned but *untrained* predictors. This explains behaviour the Bayesian story struggles with, including why accuracy keeps improving on datasets **larger than any seen in pre-training**. |
| **Neither cleanly** | Melnychuk et al. (2026/03) | For causal effect estimation the prior induces **confounding bias that data does not asymptotically wash out**. |

Nagler's version has a sharp, checkable structure worth putting on a slide: variance vanishes
when per-sample sensitivity is low, bias vanishes only with **localisation** around the test
point, and **the architecture guarantees only the first**. Neither side has measured its
quantity inside a real model, which is precisely why the instructor's research notes call this
the strongest open item.

**Teaching value:** here is a model in Nature, deployed commercially, whose own authors and a
leading theorist disagree about what it is doing. That is a far more honest picture of ML than
"here is the method, here is the accuracy table".

## What the interpretability work has established

Six studies, with what each did and did not show:

- **Biloš et al., *A Mechanistic Study of Tabular Foundation Models*** (2026/05). Compares
  TabPFN v2, TabICLv2 and Mitra across 49 datasets. The headline: the architectures
  **converge in accuracy but implement qualitatively different readouts** - an attention-weighted
  vote over context labels in one, a class-conditional mean in another - each confirmed
  *causally*, not just correlationally. They then engineer perturbations against each readout
  and reproduce the predicted failure. Same score, different algorithm, different way to break
  it.
- **Balef et al., *Is One Layer Enough?*** (2026/05). Layerwise study across six tabular
  in-context models. Finds substantial **depthwise redundancy** and builds a looped single-layer
  model at **20% of the parameters** with comparable performance. Echoes ch10's finding that
  more depth is not automatically more model.
- **Gupta et al., *Where Computation Lives Inside TabPFN*** (2026/06). Causal patching and
  ablation on feature-wise heads: one head is **2-5x more causally necessary** at its peak
  layer. Contrastive steering **fails to transfer across samples**. Only two datasets, and only
  feature-wise attention.
- **Swelam et al., *Does TabPFN Understand Causal Structures?*** (2025/11). A learned adapter
  decodes causal adjacency matrices from frozen embeddings; the causal information concentrates
  in middle layers. Shows the information is **decodable**, not that the model **uses** it - a
  distinction ch05 already teaches.
- **Zheng et al., *From Tables to Signals*** (2025/11). Spectral adaptivity: effective capacity
  adapts to the **in-context sample count**, not to training epochs.
- **McCarter, *What exactly has TabPFN learned to do?*** (2025/02). Black-box probing of
  inductive biases, memorably characterised as "brilliant or baffling".

## Why this model is an unusually good interpretability target

The instructor's review argues mechanistic interpretability's binding constraint is
**evaluation**: the field cannot reliably distinguish a correct explanation from a merely
compelling one. Heap et al. found automated interpretability metrics do not separate trained
transformers from randomly initialised ones; Hase et al. found that localising a fact does not
tell you where to edit it.

TabPFN lifts that constraint, for four reasons:

1. **The training distribution is a known generative process** - somebody wrote the prior down.
2. **The target algorithm is specified** - approximate the posterior predictive.
3. **Exact ground truth is computable at small scale** - the true posterior predictive can be
   computed or MCMC-approximated on small synthetic problems.
4. **The prior is an experimental variable** - nanoTabPFN pre-trains in one minute, so you can
   intervene on the training distribution itself.

Point 4 is the one with no analogue in frontier LLM work. It is the *Toy Models of
Superposition* setup - controlled synthetic data, known ground truth - except the model is
simultaneously state of the art.

For this course that connects directly to **ch8**, which already teaches sparse autoencoders and
interpretability and has a homework on reading an RNN's mind.
