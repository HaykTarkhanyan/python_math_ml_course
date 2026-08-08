# Sources

All gathered **2026-08-08**. Status key:

- **[read]** - I fetched the page or paper and read the content the claim comes from.
- **[read, abstract]** - only the abstract/landing page was readable; body not checked.
- **[read, secondary]** - I read a summary page *about* the paper, not the paper.
- **[search]** - the claim comes from a search-result synthesis. Reliable for dates and headline
  numbers, less so for fine detail.
- **[unverified]** - could not confirm in a primary source, or actively refuted.

---

## The primary paper

| Source | Status | Good for |
|---|---|---|
| [I-JEPA, arXiv 2301.08243v3 (CVPR 2023)](https://arxiv.org/pdf/2301.08243) | **[read, full text]** | Downloaded and read end to end with `pdftotext`. **Every number in `02_the_mechanics.md` comes from here**: Table 1 (linear ImageNet), Table 2 (1% ImageNet), Table 6 (masking ablation, 54.2 / 15.5 / 20.2 / 17.6), Table 7 (target space, 66.9 vs 40.7), Tables 8-10 (masking sweeps), masking hyperparameters (4 targets at scale (0.15, 0.2), aspect (0.75, 1.5); context (0.85, 1.0) minus overlap; 0.25 avg context ratio), predictor width 384 and depths 6/12/16, EMA momentum 0.996 to 1.0, AdamW / batch 2048 / lr 1e-4 to 1e-3 to 1e-6, no `[CLS]` token, ViT-H/14 on 16 A100s in <72h |
| [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | [search] | Official code and checkpoints |

## The model line

| Source | Status | Good for |
|---|---|---|
| [V-JEPA 2, arXiv 2506.09985](https://arxiv.org/abs/2506.09985) | [read, abstract] | >1M hours video; SSv2 77.3; EK100 39.7 recall@5; PerceptionTest 84.0; TempCompass 76.9 at 8B; <62h DROID for V-JEPA 2-AC; zero-shot Franka deployment in two labs |
| [Meta blog: V-JEPA 2 world model and benchmarks](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | [read] | 1.2B parameters; 1M hours video + 1M images; two-stage recipe; 65-80% pick-and-place with novel objects zero-shot; the three new benchmarks (IntPhys 2, MVPBench, CausalVQA) and the human-vs-model gap |
| [V-JEPA 2.1, arXiv 2603.14482 (Mar 2026)](https://arxiv.org/html/2603.14482v1) | [read] | The dense-feature admission ("noisy", "fragmented local spatial structure"); NYUv2 RMSE 0.642 to 0.350; ADE20K mIoU 24.4 to 47.8; SSv2 77.3 to 76.9; vs DINOv3 ViT-7B 0.307/0.309 depth and 7.71/5.68 mAP Ego4D; releases ViT-g 1B, ViT-G 2B, distilled ViT-B 80M / ViT-L 300M |
| [VL-JEPA, arXiv 2512.10942 (Dec 2025, rev Feb 2026)](https://arxiv.org/abs/2512.10942) | [read, abstract] | ~1.6B params; predicts continuous text embeddings instead of generating tokens; 50% fewer trainable params; 2.85x fewer decode ops; author list including LeCun and Pascale Fung |
| [TechTalks: VL-JEPA](https://bdtechtalks.com/2026/01/03/meta-vl-jepa-vision-language-model/) | [read] | 65.7% WorldPrediction-WM vs GPT-4o 53.3 and Gemini-2.0 55.6; 1.6B vs 7B/13B baselines; 46.4 vs 44.6 zero-shot video classification; stated weakness on appearance-centric tasks |
| [LeJEPA overview, alphaXiv (arXiv 2511.08544, 11 Nov 2025)](https://www.alphaxiv.org/overview/2511.08544v1) | **[read, secondary]** | The isotropic-Gaussian optimality claim; SIGReg via random 1-D projections and the Epps-Pulley statistic, O(N); removes stop-grad / EMA / predictor / registers; `L = (1-lambda)L_pred + lambda*SIGReg`; 50+ architectures, 8 families, 10+ datasets, up to 1.8B; ViT-H/14 79% linear in 100 epochs; beats DINOv2/v3 on in-domain pretraining. **The paper itself was not read** |
| [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa) | [search] | Reference implementation |
| [LLM-JEPA, arXiv 2509.14252](https://arxiv.org/abs/2509.14252) | [search] | `[PRED]` token, tied-weight predictor; gains on NL-RX / GSM8K / Spider across Llama-3 and Gemma-2; needs paired multi-view data; ~3x training cost |
| [awesome-jepa](https://github.com/AbdelStark/awesome-jepa) | [read] | The full modality catalogue in `03_the_model_line.md`. A community list, so treat individual entries as pointers rather than verified facts |

## V-JEPA (2024) - the weakest-sourced part of these notes

| Source | Status | Good for |
|---|---|---|
| [V-JEPA, OpenReview](https://openreview.net/forum?id=WFYbBOEOtv) | [search] | 82.1 Kinetics-400 and 71.2 SSv2 under **frozen** evaluation, reported as +4 and +10 over the previous best. **Not read directly** - the paper shipped under the title *Revisiting Feature Prediction for Learning Visual Representations from Video*, and I did not confirm which numbers belong to which version. Verify before putting either number on a slide |

## Theory and critique

| Source | Status | Good for |
|---|---|---|
| [Apple ML: How JEPA Avoids Noisy Features - The Implicit Bias of Deep Linear Self Distillation Networks](https://machinelearning.apple.com/research/implicit-bias) | [read] | The one real theory result here: in a deep linear setting, latent prediction is implicitly biased toward **high-influence features** (large regression coefficients), which is why it filters out unpredictable fine detail where MAE does not |
| [SALT: Rethinking JEPA - Compute-Efficient Video SSL with Frozen Teachers, arXiv 2509.24317](https://arxiv.org/abs/2509.24317) | [read, abstract] | A **frozen** teacher beats EMA at matched FLOPs; dominates V-JEPA's accuracy-FLOPs Pareto frontier; student quality robust to teacher quality; spend compute on the student. No numbers in the abstract |
| [Temporal vs Spatial: DINOv3 vs V-JEPA2, arXiv 2509.21595](https://arxiv.org/abs/2509.21595) | [read, abstract] | Neither dominates. DINOv3 silhouette 0.31 vs 0.21 and 6.16x class separation; V-JEPA2 variance 0.094 vs 0.288. Conclusion is task-dependent |
| [Connecting JEPA with Contrastive SSL, arXiv 2410.19560](https://arxiv.org/html/2410.19560v1) | [search] | Argues the families are closer than advertised; identifies EMA as insufficient against complete collapse and I-JEPA's failure to learn the patch-representation mean |
| [Sora and V-JEPA Have Not Learned The Complete Real World Model, arXiv 2407.10311](https://arxiv.org/pdf/2407.10311) | [search] | Philosophical objection via Kant's productive imagination. Not an empirical result; do not present as one |
| [Rohit Bandaru: Deep Dive into Yann LeCun's JEPA](https://rohitbandaru.github.io/blog/JEPA-Deep-Dive/) | [read] | Best-explained secondary source found. Energy-based framing, the two anti-collapse families, the "what is possible vs what will happen" phrasing, the ~3-second video horizon limitation. A personal blog, not peer reviewed |

## The position paper and the world-model framing

| Source | Status | Good for |
|---|---|---|
| LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) | **[search]** | The six modules (perception, world model, cost, actor, short-term memory, configurator), Mode 1 vs Mode 2, H-JEPA. **The paper itself was not read for these notes** - the module list is corroborated across several secondary sources but should be checked against the original before it goes on a slide |
| [Introduction to Latent Variable Energy-Based Models, arXiv 2306.02572](https://arxiv.org/abs/2306.02572) | [search] | The formal EBM companion to the position paper. Not read |
| [Wikipedia: World model (AI)](https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)) | [search] | Corroboration only for the definitional split |

## AMI Labs and the departure

| Source | Status | Good for |
|---|---|---|
| [TechCrunch: AMI Labs raises $1.03B (9 Mar 2026)](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/) | [read] | $1.03B at $3.5B pre-money; investors; Paris HQ plus NY / Montreal / Singapore; LeCun Chairman, Alexandre LeBrun CEO, Saining Xie CSO, Pascale Fung CRIO, Michael Rabbat VP World Models, Laurent Solly COO; no revenue plans; first partner Nabla; intends to publish and open-source |
| [CNBC: LeCun leaving Meta (19 Nov 2025)](https://www.cnbc.com/2025/11/19/meta-chief-ai-scientist-yann-lecun-is-leaving-the-company-.html) | [search] | Announced 19 Nov 2025, departing at end of 2025, after 12 years as Chief AI Scientist. Corroborated by NY1, US News, Bloomberg and Washington Times on the same date |

## The tutorial

| Source | Status | Good for |
|---|---|---|
| [Tutorial on Joint Embedding Predictive Architectures (JEPA): Foundations, Applications, and Future Directions](https://openreview.net/pdf?id=Zr4PUe0ZNl) - Monemi, Chinipardaz, Rasti, Bennis, Latva-aho | **[read, full text]** | Retrieved with Playwright (see note below) and read end to end. Source of: the **unified two-term loss** `L = E[d(pred, target)] + lambda*R` and the three-family taxonomy named as teacher-student / non-parametric estimators / moment-matching (`01_the_idea.md`); the world-model lineage I-JEPA to IWM to Seq-JEPA/PLDM to V-JEPA 2-AC/LeWM (`04`); the **goal-conditioned planning limitation** (`04`); the reconstruction limitation, the C-JEPA and StoP-JEPA criticisms of I-JEPA, and latent-rollout error accumulation (`05`); LeWM's single-GPU / 48x-faster-planning / six-to-one-hyperparameter numbers |

**Provenance and caveats.** `curl` returned a Cloudflare challenge page; the Playwright browser
cleared OpenReview's Turnstile check, after which the PDF was fetched from inside the page session
and decoded locally. Two things to keep in mind when using it:

- **The authors are a wireless-communications group** (University of Oulu, plus Jundi-Shapur), not
  JEPA researchers. The tutorial has a 6G/semantic-communication slant, and roughly a fifth of it
  is about wireless applications irrelevant to this course.
- **It is a preprint.** The DOI placeholder (`10.1145/XXXXXXX`) indicates a submission still in
  review. Treat it as **secondary literature** - excellent for taxonomy, structure and framing,
  which is what it was used for here. Numbers taken from it about other people's models were not
  independently verified and are flagged as such where they appear.

## The Welch Labs series

| Source | Status | Good for |
|---|---|---|
| [Yann LeCun's $1B Bet Against LLMs, Part 1](https://youtu.be/kYkIdXwW2AE) (Welch Labs, 2 May 2026, 37:24) | **[read, full transcript]** | The blurry-prediction argument, the **counting argument** (50,257 vs ~10^(15 million)), the **bouncing ball**, the **dashcam leaves** quote, Siamese networks at Bell Labs, representation collapse, and the **Barlow Twins origin story** (Horace Barlow 1961, Stephane Deny, cross-correlation to identity). Also the DINOv3 scoreboard: AlexNet 59.3 / Barlow Twins 73.2 / ViT 88.6 / DINOv3 88.4 |
| [Part 2](https://youtu.be/v_jDvpEGTIg) (Welch Labs, 30 May 2026, 40:57) | **[read, full transcript]** | The **alternative-stack** framing, V-JEPA vs CLIP, the **mushroom** argument for VL-JEPA, the VL-JEPA learning-curve experiment (35% vs 20% at 5M samples), LeCun's two-pronged **critique of VLA**, the **LeWorldModel PushT walkthrough** with decoded rollouts and visible drift, **CEM planning** step by step, and **hierarchical planning** (5 to 15 steps) |

Local copies: `ml/ch16_jepa/_reference_welchlabs_lecun_p1/` and `..._p2/` - transcripts, 43 stills,
timestamped beat maps in each README. Videos are git-ignored.

**Status of this source.** Higher than a typical explainer channel: the series **interviewed LeCun
directly** and credits Stephane Deny (Barlow Twins), David Fan and Nicolas Ballas (V-JEPA), Delong
Chen / Mustafa Shukor / Theo Moutakanni (VL-JEPA), Wancong Zhang, and **Randall Balestriero**
(LeJEPA) as having discussed the work with them. **LeCun's quotes are primary**; the surrounding
narration is good secondary explanation. It is still a channel with a book and a poster to sell,
and part 1 in particular presents his side sympathetically - it contains none of IntPhys 2, the
V-JEPA 2 dense-feature failure, or the DINOv3-beats-JEPA-on-static-images point.

**Primary-source pointers it handed over** (Balestriero's recommended reading, plus an on-screen
credit). Not yet read; IDs recorded so they can be:

- **LeWorldModel** - arXiv 2603.19312
- **Hierarchical Planning with Latent World Models**, Zhang et al. - arXiv 2604.03208
- SSL and spectral embedding - arXiv 2205.11508

## Actively refuted

| Claim | Status | What is actually true |
|---|---|---|
| "I-JEPA achieves 72.4% semi-supervised on 1% ImageNet vs MAE's 59.8%" - from a search-result synthesis | **[unverified, refuted]** | Table 2 of the paper `[read]`: I-JEPA ViT-H/14 (300 ep) **73.3**, MAE ViT-H/14 (1600 ep) **71.5**. The claimed 12.6-point gap is 1.8 points. The genuine result is the 5x epoch difference |
| "I-JEPA beats DINO and iBOT" | **[unverified]** | Table 1 `[read]`: I-JEPA ViT-H/14 **79.3**, DINO ViT-B/8 **80.1**, iBOT ViT-L/16 **81.0**. Only the 448px I-JEPA (81.1) matches. The paper says it "decreases the gap" |

## Single-source, treat with care

| Claim | Where it came from |
|---|---|
| V-JEPA 2-AC plans in ~16 seconds per action | One search synthesis, in a comparison against Cosmos. Not confirmed in the paper or blog. **Verify before quoting** |
| Human scores of 85-95% on IntPhys 2 / MVPBench / CausalVQA | Meta's blog `[read]`, but as a band rather than per-benchmark figures. The qualitative claim - humans near-ceiling, models near chance - is stated plainly by Meta; the exact percentages are not |
