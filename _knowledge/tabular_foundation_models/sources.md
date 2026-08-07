# Sources

Three confidence levels, kept separate on purpose.

## A. Local, title-verified (highest confidence)

`~/OneDrive/Desktop/mech_interp/papers/tabpfn/` - 39 PDFs, 155 MB, retrieved 2026-08-06. Every
PDF was title-matched against the registry before saving; `papers_manifest.json` (copied into
this folder) carries title, authors, date, DOI, arXiv id and match ratio for each.

Categories: analysis 11, extension 9, core 6, scaling 6, theory 5, comparison 2.

Load-bearing entries for a chapter:

| Paper | arXiv | Used for |
|---|---|---|
| Accurate predictions on small data with a tabular foundation model (Nature) | - | The v2 result and the architecture |
| Transformers Can Do Bayesian Inference | 2112.10510 | The founding PFN idea |
| TabPFN (v1) | 2207.01848 | Origin |
| nanoTabPFN | 2511.03634 | **The practical** |
| TabPFN-2.5 | 2511.08667 | Current-generation claims |
| TabPFN-3 Technical Report | 2605.13986 | Latest in the local collection |
| Statistical Foundations of PFNs (Nagler) | 2305.11097 | The frequentist side of the dispute |
| Bayes' Power for Explaining ICL Generalizations | 2410.01565 | The Bayesian side, by the TabPFN authors |
| Frequentist Consistency of PFNs for Causal Inference | 2603.12037 | Prior-induced bias that does not wash out |
| A Mechanistic Study of Tabular Foundation Models | 2605.21288 | Same accuracy, different readouts |
| Is One Layer Enough? | 2605.06510 | Depthwise redundancy, 20%-parameter looped model |
| Where Computation Lives Inside TabPFN | 2606.12917 | Causal head localisation |
| Does TabPFN Understand Causal Structures? | 2511.07236 | Decodable vs used |
| What exactly has TabPFN learned to do? | 2502.08978 | Black-box inductive biases |
| Realistic Evaluation of TabPFN v2 in Open Environments | 2505.16226 | **The counterweight paper** |
| TabICL | 2502.05564 | Large-data competitor |
| TabDPT | 2410.18164 | Real-data pre-training competitor |
| From Tables to Time | 2501.02945 | ch08 connection |
| Drift-Resilient TabPFN | 2411.10634 | Shift, via the prior |
| Interpretable Machine Learning for TabPFN | 2403.10923 | ch05 connection |
| PFNs Scale When Treated as Weak Learners | 2503.01256 | ch04 boosting connection |

Also local: `mech_interp/research_ideas/tabpfn_mech_interp.md` - the synthesis these notes lean
on most; `mech_interp/lit_review/` - the mechanistic interpretability review, 53/53 references
verified.

### nanoTabPFN, quoted exactly (read from the PDF, 2026-08-07)

> "restricted to a small data setting it achieves a performance comparable to traditional
> machine learning baselines within one minute of pre-training on a single GPU (160 000x faster
> than TabPFN v2 pretraining)"

Figure 4 caption: "Within 60 seconds of pretraining on one **consumer** GPU, nanoTabPFN achieves
average ROC AUC on a **subset of subsampled** datasets from TabArena comparable to traditional
machine learning baselines."

**Read the qualifiers carefully before designing a lab.** The 60-second run reaches
*traditional ML baselines* on *subsampled small* datasets. It does **not** reach TabPFN v2. A
practical must promise "comparable to a random forest", never "state of the art".

Two more facts from the same paper:

- Code: `https://github.com/automl/nanoTabPFN` (AutoML group, not Prior Labs - so the TabPFN-2.5
  licence restriction probably does not apply, but **confirm the actual licence** before
  building on it).
- **It does not support categorical features or missing values.** The user must preprocess
  beforehand. Any dataset chosen for a practical has to be numeric and complete, or come with
  preprocessing already written.

## B. Web-checked 2026-08-07 (current, but single-pass)

- [Prior-Labs/tabpfn_2_5 on Hugging Face](https://huggingface.co/Prior-Labs/tabpfn_2_5) and its
  [README](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/README.md) - the
  `tabpfn-2.5-license-v1.1` terms, and the 50,000-sample / 2,000-feature limits.
- [TabPFN on GitHub](https://github.com/PriorLabs/tabpfn) - the implementation.
- [TabPFN-2.5 on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-chfhncrdzlb3s) -
  evidence of the commercial channel.
- [TabICLv2](https://arxiv.org/pdf/2602.11139) - "better, faster, scalable, and open".
- [TabPrep: Closing the Feature Engineering Gap](https://arxiv.org/pdf/2606.02384) - disputes
  that feature engineering is absorbed.
- [LimiX-2M](https://arxiv.org/pdf/2606.04485) - another tabular foundation model.
- [The state of Tabular Foundation Models (2026)](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models) -
  field overview; secondary source, treat as orientation not evidence.

## C. Unverified - do not put on a slide without checking

- **TabPFN-3-Plus**, reported by a secondary source as available since June 2026. No paper in
  the local collection, no primary source read.
- ~~nanoTabPFN's one-minute / 160,000x figure.~~ **VERIFIED 2026-08-07** against the PDF - see
  the exact quote in section A below.
- ~~The Nature paper's exact headline numbers.~~ **VERIFIED 2026-08-08**, read from
  `nature2025_*.pdf` (Nature vol 637, 9 Jan 2025, p.319). Quotes now in
  `ml/ch14_tabular_fm/TABULAR_FM_CHAPTER_PLAN.md`. Key facts, all verbatim from the paper:
  wins on datasets **up to 10,000 samples**; **2.8 s vs an ensemble of the strongest baselines
  tuned for 4 h**; compute **O(n^2 + m^2)**, memory **O(n . m)**; **two-way attention** with a
  separate representation per **cell**, giving invariance to the order of both samples and
  features; separately, runs on up to **50 million cells (5M rows x 10 features) on one H100**
  at "less than 1,000 bytes per cell".
  **One correction this produced:** the planning note previously described attention as
  quadratic in the number of cells, i.e. O((n.m)^2). That is wrong - the factorisation is
  exactly what avoids it. Fixed in the plan.

## What was deliberately not copied

The 155 MB of PDFs stay in `mech_interp/papers/tabpfn/`. This repo's `.git` is already 1.9 GB.
`papers_manifest.json` here has every `pdf_url`, and `mech_interp/scripts/download_tabpfn_papers.py`
rebuilds the collection from it.
