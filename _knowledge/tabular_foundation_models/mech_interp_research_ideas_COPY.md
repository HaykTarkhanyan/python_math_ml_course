# Mechanistic interpretability on TabPFN: research directions

Written 2026-08-06, revised the same day after a second and third literature pass.
Grounded in the 39 papers in `papers/tabpfn/` and the field map in
`lit_review/mechanistic_interpretability_review.md`.

Ideas are grouped by **ambitiousness** (A/B/C) and tagged with **compute tier** (T0-T4)
calibrated to this machine: 16 GB RAM, Intel Iris Xe integrated graphics, no CUDA, plus
Colab CLI access in WSL (T4/L4/A100/H100).

> **Revision note, two passes.** The first draft claimed several things were untouched.
>
> *Second pass* found two 2026 papers missed because the sweep keyed on "TabPFN" while they
> say "Tabular Foundation Models": Biloš et al. (2605.21288) and Balef et al.
> (2605.06510). Between them they substantially cover what were ideas A2, A4 and B5, and
> partly anticipate C1.
>
> *Third pass* searched the underlying concept, **"prior-data fitted networks"**, and found a
> ~25-paper theory literature that had been entirely absent - including Nagler's *Statistical
> Foundations of PFNs* and the TabPFN authors' own *Bayes' Power for Explaining ICL
> Generalizations*. These do not close an idea; they turn B2 from a speculative question into
> an active three-way dispute that mechanistic interpretability can adjudicate, which is now
> the strongest item in this document.
>
> Product name, architecture name and concept name each retrieve a different slice of a
> literature. **Search all three.** Superseded entries are struck through and rewritten rather
> than deleted, so the record of what was wrong stays visible.

---

## 1. Why TabPFN is an unusually good target

The literature review concluded that mechanistic interpretability's binding constraint is
**evaluation**: the field cannot reliably tell a correct explanation from a compelling one.
Heap et al. (2025) found automated interpretability metrics do not distinguish trained
transformers from randomly initialised ones. Hase et al. (2023) found localising a fact does
not tell you where to edit it. Both are failures of *verification*, not of generation.

TabPFN is the one practically important model where that constraint can be lifted:

1. **The training distribution is a known generative process.** Pre-training uses synthetic
   datasets sampled from an explicit prior over structural causal models and Bayesian neural
   networks. You know what generated the data, because someone wrote it down.
2. **The target algorithm is specified.** The objective is to approximate the posterior
   predictive distribution. "Is the model doing Bayesian inference?" is a precise question
   with a checkable answer, unlike "what algorithm is GPT-2 running?"
3. **Exact ground truth is computable at small scale.** On small synthetic problems the true
   posterior predictive can be computed exactly or MCMC-approximated.
4. **The prior is an experimental variable.** nanoTabPFN (Pfefferle et al., 2025) pre-trains
   in **one minute on a single GPU**, reported as 160,000x faster than TabPFN v2 pre-training.

Property 4 matters most. It gives **interventional control over the training distribution**,
impossible for any frontier LLM. This is the *Toy Models of Superposition* setup - controlled
synthetic data, known ground truth - except the model is also genuinely state of the art.

### Architecture

From the Nature paper (Hollmann et al., 2025): each **cell** gets its own representation. Each
layer applies **attention over features** (a cell attends to other features in its row), then
**attention over samples** (a cell attends to the same feature down its column), then an MLP
sublayer, each with residual and normalisation. Random feature embeddings are added before
layer 1 to break symmetry between identically-distributed features. Training-sample
representations are not influenced by test samples.

The residual stream is indexed by (row, column), not token position, and there are two
structurally distinct attention mechanisms. Most of the MI toolkit assumes a 1-D token stream.
That is a problem and an opportunity: the row/column factorisation gives components a semantic
type before you start, which is the interpretive foothold LLM work lacks.

---

## 2. What has already been done

Read this before starting anything below. As of the second pass, more is taken than the first
draft assumed.

| Paper | What it did | What it did NOT do |
|---|---|---|
| **Biloš et al., *A Mechanistic Study of Tabular Foundation Models*** (2605.21288) | Compares TabPFN v2, TabICLv2, Mitra over a 49-dataset benchmark. Finds architectures **converge in accuracy but realise qualitatively distinct similarity-based readouts** - attention-weighted vote over context labels vs class-conditional mean - each confirmed causally. Traces permutation invariances to specific positional parameters whose removal preserves accuracy. Engineers perturbations against each readout that reproduce predicted failure modes | Does not vary the training prior. Does not explain *why* a similarity readout achieves near-Bayes accuracy. No dictionary learning |
| **Balef et al., *Is One Layer Enough?*** (2605.06510) | First large-scale layerwise study across **6** tabular ICL models. Distinct inference stages, latent dynamics differing from LLMs, substantial **depthwise redundancy**. Builds a looped single-layer model at **20% of parameters** with comparable performance. Code released | Descriptive at the layer level; no head-level or feature-level circuit. Does not connect redundancy to the prior |
| Gupta et al., *Where Computation Lives Inside TabPFN* (2606.12917) | First causal analysis of TabPFN specifically. Patching, ablation, attention entropy on **feature-wise** heads. One head 2-5x more causally necessary at peak layer. Contrastive steering **fails to transfer across samples** | Feature-wise heads only, **two** datasets, no sample-wise attention, no SAEs |
| Gupta et al., *Through The Looking Glass* (2601.08181) | Probing for regression coefficients, intermediate values, final answer across layers | Correlational only |
| Swelam et al., *Does TabPFN Understand Causal Structures?* (2511.07236) | Learned adapter decodes causal adjacency matrices from frozen embeddings; causal info concentrated in mid layers | Decodability, not use |
| McCarter, *What exactly has TabPFN learned to do?* (2502.08978) | Black-box inductive biases, "brilliant or baffling" | No internals |
| Zheng et al., *From Tables to Signals* (2511.18278) | Spectral Adaptivity: capacity adapts to in-context sample count, not training epochs | No mechanism |
| Ye et al., *A Closer Look at TabPFN v2* (2502.17361) | Infers attribute relationships from randomised attribute tokens | Reports, does not explain |
| **Nagler, *Statistical Foundations of PFNs*** (2305.11097, ICML'23) | Theory. A **purely frequentist** reading of PFNs as pre-tuned but untrained predictors explains their behaviour, including why accuracy improves on datasets larger than any seen in pre-training. Variance vanishes with low per-sample sensitivity; bias vanishes only with localisation around the test feature, and **the architecture ensures only the former** | No mechanistic evidence. The sensitivity and localisation claims are never measured inside a real model |
| **Müller, Hollmann & Hutter, *Bayes' Power for Explaining ICL Generalizations*** (2410.01565) | Argues the **Bayesian** reading, not MLE, explains ICL generalisation. Written by the TabPFN authors | Also no mechanistic evidence; directly in tension with Nagler |
| Melnychuk et al., *Frequentist Consistency of PFNs for Causal Inference* (2603.12037) | PFN-based ATE estimators show **prior-induced confounding bias: the prior is not asymptotically overwritten by data**; proposes a one-step posterior correction | Estimator-level, not mechanism-level |

**Frontier in four sentences.** Tabular foundation models implement *similarity-based
readouts*, not Bayesian inference, and different architectures implement *different* ones
despite matching accuracy. Their depth is substantially redundant, to the point that a looped
single layer at 20% of parameters is competitive. The theory literature is actively split on
whether a PFN is best understood as Bayesian or frequentist, and **neither side has produced
mechanistic evidence**. Nobody has varied the training prior, and nobody has applied dictionary
learning to a tabular foundation model.

---

## 3. Compute tiers

| Tier | Hardware | Realistic budget | Ask first? |
|---|---|---|---|
| **T0** | This laptop, CPU | Minutes. TabPFN v2 inference on n<=1000, d<=20 is seconds per fit | No |
| **T1** | This laptop, CPU | Hours to overnight. Sweeps of hundreds of fits, activation caching | Mention it |
| **T2** | One Colab T4 session | 2-4 h. SAE training on cached activations, nanoTabPFN pre-training, GPU inference at scale | Yes - quota |
| **T3** | Colab L4/A100, several sessions | Large SAE suites, hundreds of nanoTabPFN pre-trains | Yes |
| **T4** | Out of reach here | Full TabPFN v2/2.5/3 pre-training from scratch | Use nanoTabPFN instead |

TabPFN is small and its activations are low-dimensional, so **caching every activation for a
whole experiment fits on disk** and most analysis is then a pure CPU job. Per `CLAUDE.md`,
anything at T2+ or any parallel multi-core sweep gets flagged before running.

---

## 4. Tier A - grounded extensions

### A1. Map the sample-wise attention heads [T0-T1]
The one TabPFN-specific causal study (2606.12917) covered feature-wise heads. **Sample-wise
attention is where in-context learning must live** - it is the only path by which a test cell
sees training rows. Run patching, ablation and entropy over sample-wise heads on more than two
datasets.

*Risk:* obvious follow-up for an active group. Check arXiv first. Still the cheapest way to
build the tooling everything below needs.

### A2. ~~Retrieval vs kernel vs Bayes~~ → **Where the similarity readout breaks** [T0-T1]
*Largely answered.* Biloš et al. established the readouts are similarity-based and identified
which family does which. The live question is now narrower and better: **construct datasets
where the identified readout must fail** - where an attention-weighted vote over context
labels gives a different answer from the exact posterior - and check whether TabPFN fails
there as predicted. Biloš engineered hub and rank attacks; the untested version is failure
predicted from the *prior*, not from the readout. This is the cheapest entry into §5's B2.

### A3. Diagnose the steering failure [T0-T1]
2606.12917 reports contrastive steering fails to transfer across samples, and attributes this
to ICL encoding task structure in-context. That is a discussion-section hypothesis, not a
result. Test whether steering transfers *within* a fixed in-context set but not across sets.
If so, TabPFN's features are **context-relative**: the direction encoding a concept is defined
relative to the current in-context distribution, so there is no fixed "refusal direction"
analogue (contrast Arditi et al., 2024). A negative result with a mechanism.

### A4. ~~Layer-wise phase diagram~~ → **Head-level structure inside the redundant depth** [T1]
*Largely done at the layer level* by 2605.06510, which found depthwise redundancy and built a
looped single-layer model at 20% of parameters. What that leaves open is **head-level**: if
depth is redundant, are the same heads recomputing the same function each layer, or are
different heads converging on the same output? Those imply different things about what
redundancy means, and only the second is genuinely wasteful. Their code is released, so this
starts from a working baseline rather than a reimplementation.

### A5. How is feature identity recovered from statistics alone? [T0-T1]
Ye et al. found attribute relationships are inferred from randomised attribute tokens. The
Nature paper says random feature embeddings exist only to break symmetry. So the model
reconstructs "what this column means" from distributional evidence during the forward pass.
Biloš et al. traced *permutation invariances* to specific positional parameters, which is a
strong lead on where to look. Well-posed, cheap, and a known-interesting answer.

---

## 5. Tier B - new mechanism claims

### B1. Sparse autoencoders on TabPFN cell representations [T2]
**Still untouched.** Dictionary learning has been applied to language and, through 2026, to
vision and vision-language models, but not to a tabular foundation model. Open questions that
are genuinely new rather than transplanted:

- Is there superposition in a model whose inputs are continuous values, with no discrete
  vocabulary? The superposition argument (Elhage et al., 2022) counts features against
  dimensions; what *is* a feature when there are no tokens?
- What do latents turn out to be? Plausible: "this column is categorical", "this row is an
  outlier", "these two columns interact", "the target is monotone in this feature". If SAE
  latents correspond to **statistical properties of the dataset**, that is a qualitatively
  different kind of feature from anything in the LLM SAE literature, and it would be the first
  evidence that the SAE framework generalises past token-based models in a non-trivial way.
- Cells are indexed by (row, column) and there are two attention types. One dictionary, or one
  per stream? The answer is itself a result.

Cheap: small model, low-dimensional activations, no Gemma-Scope-scale infrastructure needed.
Best novelty-to-compute ratio here.

### B2. Adjudicate the Bayes-vs-frequentist dispute mechanistically [T1-T2] - the strongest item here
There is a **live three-way disagreement** about what a PFN computes, and mechanistic
interpretability is the tool that could settle it. The positions:

1. **Bayesian.** Müller, Hollmann and Hutter (2410.01565) - the TabPFN authors - argue the
   Bayesian reading, not the maximum-likelihood one, is what explains in-context learning
   generalisation.
2. **Frequentist.** Nagler (2305.11097, ICML 2023) shows "a purely frequentistic
   interpretation of PFNs as pre-tuned, but untrained predictors explains their behavior",
   and explains why accuracy improves on datasets *larger* than any seen in pre-training,
   which the naive Bayesian story does not predict.
3. **Neither, mechanically.** Biloš et al. (2605.21288) find the implemented readout is a
   similarity-weighted vote over context labels, confirmed by causal intervention.

**Nagler hands you a directly testable mechanistic claim.** His analysis says a predictor's
variance vanishes if its sensitivity to individual training samples vanishes, and its bias
vanishes only if it is **appropriately localised around the test feature** - and that the
transformer architecture used in current PFNs *ensures only the former*. That is a statement
about what the architecture can and cannot do, phrased in terms of sensitivity and
localisation, both of which are measurable directly from attention patterns and patching
experiments.

So the experiment is well posed: measure sample-sensitivity and test-feature localisation
inside TabPFN, and check whether the pattern matches Nagler's prediction. If it does, the
frequentist account gains mechanistic support and the Bayesian framing is a motivating story
rather than a description. If it does not, Nagler's theory needs revising. Either way the
result is publishable, and the question is currently being argued in theory papers with no
mechanistic evidence on either side.

Supporting evidence that the prior does not wash out: the causal-inference PFN paper
(2603.12037) finds "prior-induced confounding bias: the prior is not asymptotically overwritten
by data", which is exactly what you would expect if the model is a pre-tuned predictor rather
than a Bayesian updater.

### B3. Recover a circuit for a known structural causal model [T2]
Generate data from a specific known SCM and find the circuit TabPFN uses. Because **you know
the true causal graph**, you can check whether the recovered circuit corresponds to it. This
is *Interpretability in the Wild* with an answer key - the thing that line of work has always
lacked. Biloš et al.'s "state a falsifiable surrogate rule, test it causally, transplant it to
other backbones" is a ready-made protocol to reuse.

### B4. Prior ablation with nanoTabPFN [T2-T3] - strongest design here
**Untouched, and Biloš et al. made it more valuable.** They showed different architectures
learn different readouts. That immediately raises: **does the prior or the architecture
determine the readout?** Their comparison confounds the two, because each model family comes
with both its own architecture and its own pre-training distribution.

nanoTabPFN lets you hold architecture fixed and vary the prior: remove causal structure,
remove feature interactions, change the SCM family, vary noise. Then ask whether a given
readout appears only when the prior contains the corresponding structure. At roughly one minute
per pre-training run, hundreds of priors is an afternoon on a T4.

This is a controlled experiment on the training distribution, it is the natural next question
after Biloš et al., and nanoTabPFN is the only way to run it. If one idea here gets done, this.

### B5. ~~Universality~~ → **Within-family universality** [T2]
*Answered negatively across families.* Biloš et al. found distinct readouts across
architectures despite converging accuracy - evidence **against** universality, which is itself
notable given it was a founding claim of the circuits programme (Olah et al., 2020).

Still open: universality *within* the TabPFN line. v1, v2, 2.5, 3 and nanoTabPFN share a
lineage and closely related priors at very different scales. Do the same head roles and SAE
features recur as the model grows? "Universality holds within a training-prior family but not
across architectures" would be a clean, quotable formulation, and it is exactly the shape of
claim B4 can test causally.

### B6. Is the decodable causal information actually used? [T2]
**Untouched.** Swelam et al. showed causal structure is decodable from mid-layer embeddings by
a learned adapter. **Decodable is not used** - precisely the trap Hase et al. (2023) documented
for factual localisation. Ablate or corrupt the mid-layer causal representation and test
whether predictions change as the causal hypothesis predicts. Sharper now: if the readout is a
similarity-weighted vote (Biloš), it is not obvious that explicit causal structure plays any
functional role at all, and showing that would qualify a prominent claim.

### B7. Do engineered failure modes transfer to real data? [T1-T2]
Biloš et al. built hub and rank attacks against the inferred readouts and reproduced predicted
failures. Those were constructed adversarially. The applied question: **do naturally occurring
tables exhibit the same structure?** If real datasets with high hubness systematically degrade
TabPFN, that converts a mechanistic finding into a deployment warning, and it is the kind of
result practitioners actually act on. Cheap: run the attack diagnostics over OpenML tables and
correlate with error.

---

## 6. Tier C - field-level bets

### C1. A ground-truth benchmark for mechanistic explanations [T3] - highest value, now sharper
Build synthetic tasks where **the correct mechanistic explanation is known by construction**,
because you specified both the data-generating process and the training prior. Score MI methods -
SAEs, activation patching, attribution patching, probing, automated interpretability - on
whether they recover it.

*Partly anticipated:* Biloš et al. already validate mechanisms by engineering perturbations
that reproduce predicted failures. That is validation against **predicted behaviour**. What is
still missing, and is the stronger form, is validation against a **known mechanism**: with
nanoTabPFN you control the prior, so for simple enough priors you know what the model *should*
have learned, and you can ask whether each method recovers it. Behaviour-prediction can be
satisfied by a surrogate that is not the mechanism; ground truth cannot.

This attacks the review's central finding directly. Heap et al. (2025) showed current automated
interpretability metrics cannot distinguish trained from random transformers, which means the
field scores explanations with instruments never themselves validated. Nothing validates an
instrument except ground truth, and this is where ground truth exists.

### C2. Does in-context learning here implement gradient descent? [T2-T3]
A standing literature argues transformer ICL implements implicit gradient descent or
preconditioned GD (Ahn et al., NeurIPS 2023; looped-transformer work, ICML 2024), mostly on
architectures chosen for analytic convenience. TabPFN is the cleanest realistic test case: ICL
is its only mode and it is small enough to analyse exhaustively.

Sharpened by both new papers. Biloš et al. say the readout is similarity-based, which does not
obviously look like GD; Balef et al. report **iterative refinement with overlapping
computation across depth**, which does. Those sit in tension, and resolving it is a real
contribution. The looped single-layer model is an unusually direct probe: if a loop of one
layer suffices, whatever iteration is happening is simple enough to characterise fully.

### C3. Mechanistic distillation of TabPFN [T2-T3]
Balef et al. proved the payoff path exists: layer-level insight yielded 20% of
parameters at comparable performance. The TabPFN-specific version starts one level deeper -
from head roles and SAE features rather than layer statistics - and asks how far it compresses.
This is the most fundable framing here, because it converts interpretability into a number
somebody cares about, and because a compression result is a **falsifiable test of
understanding**: if the mechanism you identified is real, removing everything else should not
hurt.

### C4. Design a prior for interpretability, then verify [T3]
Invert the usual approach: instead of interpreting a model you were given, design the training
prior so the resulting model is interpretable **by construction**, then check mechanistically
whether it worked. Related in spirit to Softmax Linear Units (Elhage et al., 2022) and
Transformer Programs (Friedman et al., 2023), but acting on the data distribution rather than
the architecture - the lever TabPFN uniquely exposes. nanoTabPFN makes the loop short enough to
search the space.

### C5. Transfer validated methods back to LLMs [T3]
If a mechanism is established with ground truth in TabPFN and the same analysis finds a
corresponding mechanism in LLM in-context learning, you have bootstrapped from a domain where
verification is possible into one where it is not. Speculative; disanalogies may dominate. This
is the payoff that would justify the whole programme, and the one most likely to fail.

---

## 7. The matrix

| | **T0-T1** laptop | **T2** one T4 session | **T3** L4/A100, multi-session |
|---|---|---|---|
| **A** grounded | A1 sample-wise heads · A2 where the readout breaks · A3 steering diagnosis · A5 feature identity | A4 head-level redundancy | |
| **B** new claims | B2 **why similarity ≈ Bayes** · B7 failure modes on real data | B1 **SAEs on cells** · B3 known-SCM circuit · B5 within-family universality · B6 decodable vs used | B4 **prior ablation** |
| **C** field-level | | C2 ICL as gradient descent | C1 **ground-truth benchmark** · C3 mechanistic distillation · C4 interpretable-by-design prior · C5 transfer to LLMs |

Bold: highest value for effort.

---

## 8. Practical notes

**Tooling does not exist for this.** TransformerLens, nnsight and the rest assume a 1-D token
stream with a single attention type. TabPFN has two attention types over a 2-D cell grid, so
hooks have to be written against the TabPFN codebase directly. Budget real time for this; it is
the main reason the field has one causal paper rather than ten. Balef et al. released
code (`github.com/amirbalef/is_one_layer_enough`), which is the closest thing to a starting
point and worth reading before writing anything.

**Patching semantics differ.** TabPFN's attention is not causally masked in the autoregressive
sense, and training-sample representations do not depend on test samples. "Corrupt the input and
patch a clean activation" needs re-deriving: what counts as a minimal pair when the unit of
input is a whole table? Getting this definition right is a genuine contribution on its own, and
getting it wrong invalidates everything downstream.

**Ground truth is only as good as the prior implementation.** Every claim in §1 rests on the
released pre-training prior matching what the papers describe. Verify that against the code
before building on it.

**Read first, in this order.** Nature paper (architecture) → Nagler 2305.11097 and Müller
2410.01565 (what it is *supposed* to compute, and the dispute) → Biloš 2605.21288 (what it
actually computes) → Balef 2605.06510 (how that is distributed over depth) → Gupta
2606.12917 (head-level method) → nanoTabPFN 2511.03634 (the experimental lever).

---

## 9. Honest assessment

**Crowded, and moving fast.** Between the first and second draft of this document, two papers
appeared that closed three ideas. Three groups are visibly active: Gupta et al. on TabPFN
internals, Biloš et al. on cross-architecture mechanism, Rezaei Balef/Eggensperger on layerwise
dynamics. Assume roughly a three-month scoop window on anything in Tier A.

**Likely to disappoint.** A3 starts from a method that already failed once here - frame it as
explaining a negative result, not making steering work. C5 reads well and usually dies on
disanalogies. A1 is probably someone's current project.

**Underrated.** B4 and B1. Both untouched, both cheap, and B4 answers the question Biloš et
al.'s result raises but their design cannot settle. B6 and B7 are underrated differently: both
are debunking-shaped, less fun to write, and exactly what the review found the field short of.

**The real reason to do this.** Not that TabPFN is important, though it is. It is that
mechanistic interpretability has no way to check its own answers, and this is the one place it
could. That belongs in the introduction of any paper from this list, not the discussion.

**Confidence.** Claims about what is and is not done rest on the 34 papers in `papers/tabpfn/`.
The first draft got this wrong by searching for a product name instead of a concept, and two
relevant papers were sitting one query away. Treat "nobody has done X" here as "nobody in this
corpus", and run the search yourself - by concept - before committing to a project.

---

## 10. Concrete first week

1. **Day 1-2, T0.** TabPFN v2 on CPU on small synthetic tables. Cache activations for both
   attention types. Reproduce one number from 2606.12917 or 2605.06510. If you cannot reproduce
   a published number, everything downstream is unanchored.
2. **Day 2-3, T0.** Read Biloš et al. properly and reimplement their surrogate-rule test for
   TabPFN v2 alone. This gives you a working causal-intervention harness and a validated
   baseline mechanism to build on, rather than starting from a blank page.
3. **Day 4, T0-T1.** Build the A2 datasets where the similarity readout must diverge from the
   exact posterior. Reusable by B2, B3 and C1.
4. **Day 5-6, T1.** Run A1 patching over sample-wise heads. Cheapest path to a real finding;
   de-risks tooling for all of Tier B.
5. **Day 7, T2** (ask first). Stand up nanoTabPFN, do one modified-prior pre-training run end
   to end. That loop is the gate on B4, the best idea here.

The order is deliberate: each step builds infrastructure the next needs, and the expensive
commitment comes last, after the cheap steps have shown whether the tooling holds.
