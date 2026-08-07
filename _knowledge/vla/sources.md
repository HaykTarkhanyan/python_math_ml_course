# Sources

All gathered by web search on **2026-08-07**. Nothing here comes from model training data; every
date, size and number was checked against a live page.

Status key:

- **[read]** - I fetched the page or paper and read the content the claim comes from.
- **[search]** - the claim comes from a search-result synthesis of this page, not a direct read.
  Treated as reliable for dates and headline numbers, less so for fine detail.
- **[unverified]** - could not confirm in a primary source. Flagged in the notes as an estimate.

---

## Surveys and field overviews

| Source | Status | Good for |
|---|---|---|
| [State of VLA Research at ICLR 2026 (Moritz Reuss)](https://mbreuss.github.io/blog_post_iclr_26_vla.html) | [read] | The best single snapshot of the field. 164 submissions vs 1 at ICLR 2024; benchmark saturation numbers; the open-vs-closed capability gap; the definition of VLA used throughout these notes. A personal blog by an active researcher, not peer reviewed |
| [VLA in Robotics: A Survey of Datasets, Benchmarks, and Data Engines (arXiv 2604.23001, Apr 2026)](https://arxiv.org/html/2604.23001v1) | [read] | Data source taxonomy, benchmark comparison table, data-engine catalogue (MimicGen, ROSIE, ALOHA/GELLO costs), CALVIN 0.08% five-step figure, THE COLOSSEUM compounded-perturbation finding |
| [VLA Safety: Threats, Challenges, Evaluations, and Mechanisms (arXiv 2604.23775, Apr 2026)](https://arxiv.org/pdf/2604.23775) | [read] | Threat taxonomy skeleton (patches / prompt injection / backdoors) and defense categories. Light on numbers |
| [A Survey on VLA Models: An Action Tokenization Perspective (arXiv 2507.01925)](https://arxiv.org/abs/2507.01925) | [read, abstract only] | Eight-way taxonomy of action token types (language, code, affordance, trajectory, goal state, latent, raw action, reasoning). Its claimed OXE-vs-LLM token ratio was **not** confirmed - see unverified section |
| [Wikipedia: Vision-language-action model](https://en.wikipedia.org/wiki/Vision-language-action_model) | [read] | Cross-check on dates, sizes and the discrete-vs-continuous tradeoff framing. Used only to corroborate, never as the sole source for a number |

## Models

| Source | Status | Good for |
|---|---|---|
| [RT-1 (arXiv 2212.06817)](https://arxiv.org/abs/2212.06817) | [search] | Dec 2022; 35M params, 3 Hz, 130k episodes, 13 robots, 17 months, 700+ instructions, 97% seen / 76% new |
| [RT-2 (arXiv 2307.15818)](https://arxiv.org/abs/2307.15818) | [search] | Jul 28 2023; PaLI-X 55B and PaLM-E 12B variants; ~3x on emergent skills |
| [Open X-Embodiment (arXiv 2310.08864)](https://arxiv.org/abs/2310.08864) | [search] | Oct 2023; 1M+ trajectories, 22 embodiments, 60 datasets, 34 labs, 21 institutions, 527 skills; RT-1-X ~50% over baselines, RT-2-X ~3x emergent |
| [Octo](https://octo-models.github.io/) | [search] | May 2024; 27M/93M, 800k OXE trajectories, diffusion head, fully open |
| [OpenVLA (arXiv 2406.09246)](https://arxiv.org/abs/2406.09246) | [search] | Jun 2024; 7B, DINOv2 + SigLIP + Llama-2, 970k episodes, 64 A100s x 15 days, +16.5 pts over RT-2-X, MIT license |
| [OpenVLA-OFT (arXiv 2502.19645)](https://arxiv.org/abs/2502.19645) | [search] | Feb 2025; parallel decoding + chunking + L1 regression, 26x faster, 97.1% LIBERO |
| [pi-0 (arXiv 2410.24164)](https://arxiv.org/html/2410.24164v4) | [read] | PaliGemma 3B + 300M action expert = 3.3B; H=50; up to 50 Hz; 10 flow steps; ~10,000 h training data, 903M timesteps proprietary across 68 tasks / 7 configs, plus OXE + Bridge v2 + DROID |
| [pi-0.5 blog](https://www.pi.website/blog/pi05) | [read] | Apr 22 2025; hierarchical discrete-subtask then flow-matching; ~100 homes; 94% OOD follow and success vs 83% ID; the web-data ablation. **Company blog, not peer reviewed** |
| [pi\*0.6 / RECAP blog](https://www.pi.website/blog/pistar06) | [read] | Nov 17 2025; 5B backbone; espresso throughput ~15 to >30 per hour; failures cut 2x+; the 18-hour / 50-laundry / 59-box demos. **Company blog** |
| [FAST (arXiv 2501.09747)](https://arxiv.org/abs/2501.09747) | [search] | Jan 2025; DCT + BPE action tokenizer, ~10x compression, 5x faster training, FAST+ trained on 1M trajectories |
| [Real-Time Chunking (arXiv 2506.07339)](https://arxiv.org/abs/2506.07339) | [search] | NeurIPS 2025; freeze-and-inpaint asynchronous chunk execution, training-free, works on pi-0.5 |
| [openpi repo](https://github.com/Physical-Intelligence/openpi) | [read] | Apache-2.0; pi-0, pi-0-FAST, pi-0.5 checkpoints; >8 GB inference, >22.5 GB LoRA, >70 GB full fine-tune; no pi-0.6 released |
| [Figure Helix](https://www.figure.ai/news/helix) | [search] | Feb 2025; 7B S2 at 7-9 Hz, 80M S1 at 200 Hz, single latent-vector interface, ~500 h teleop. **Company blog, no benchmark numbers exist publicly** |
| [GR00T N1 (arXiv 2503.14734)](https://arxiv.org/abs/2503.14734) | [search] | Mar 2025; 2.2B total / 1.34B Eagle-2 VLM; diffusion transformer; Apache-2.0; 63.9 ms per 16-action chunk on L40 |
| [NVIDIA Isaac GR00T reference humanoid](https://nvidianews.nvidia.com/news/nvidia-open-humanoid-robot-reference-design) | [search] | Jun 1 2026; open humanoid reference design (Unitree H2 Plus + Sharpa hands + Jetson Thor); partner labs |
| [SmolVLA](https://huggingface.co/blog/smolvla) | [search] | Jun 3 2025; 450M, flow matching, 10M frames from 487 LeRobot community datasets, async inference ~30% faster |
| [Gemini Robotics 1.0 announcement](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/) | [search] | Mar 12 2025; Gemini 2.0 based; ALOHA 2 primary training platform; -ER split |
| [Gemini Robotics 1.5 blog](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) | [read] | Sep 25 2025; thinking-before-acting; Motion Transfer; ALOHA 2 / Apollo / Franka; ER 1.5 SOTA on 15 embodied-reasoning benchmarks, aggregate >60; ER 1.5 in the Gemini API, VLA to partners only |
| [Gemini Robotics 1.5 paper (arXiv 2510.03342)](https://arxiv.org/abs/2510.03342) | [search] | The technical report behind the above |
| [Gemini Robotics 2 blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) | [read] | Jul 30 2026; three models; whole-body humanoid control; 22-DoF hands; measured bars: general manipulation 45.7-76.3%, multi-finger 32-92% (bulb-screwing 36%), gripper 74.2-89.6%; robots listed; not open. **Company blog** |
| [Dream-VLA (arXiv 2512.22615)](https://arxiv.org/abs/2512.22615) | [search] | HKU + Huawei, Dec 27 2025 (rev Jan 2026); 7B diffusion-LLM backbone; 1 diffusion step, 27x speedup; LIBERO 97.2%, SimplerEnv-Bridge 71.4%, Fractal 60.5%; Apache-2.0 |
| [LingBot-VLA 2.0 (MarkTechPost)](https://www.marktechpost.com/2026/07/08/lingbot-vla-2-0/) | [read] | Jul 8 2026; Robbyant / Ant Group; 6B on Qwen3-VL-4B; Apache-2.0; 55-dim canonical action space; ~50k h robot + 10k h human video; GM-100 numbers. **Secondary source - the technical report was not read** |

## Data

| Source | Status | Good for |
|---|---|---|
| [DROID (arXiv 2403.12945)](https://arxiv.org/abs/2403.12945) | [search] | 76k trajectories / 350 h, 564 scenes, 86 tasks, 50 collectors, 13 institutions, 12 months, 1,417 viewpoints, identical Franka + Zed + Quest hardware |
| [AgiBot World Colosseo (arXiv 2503.06669)](https://arxiv.org/abs/2503.06669) | [search] | Mar 2025; 1,003,672 trajectories (~43.8 TB), 217 tasks, 100 robots, 4,000 m2 facility, five domains; GO-1 >60% on complex tasks, +32 pts over RDT |
| [EgoScale (arXiv 2602.16710)](https://arxiv.org/abs/2602.16710) | [search] | Feb 2026, NVIDIA; 20,854 h egocentric human video; log-linear scaling law between human-action loss and data scale; +54% success on a 22-DoF hand |
| [ACT / ALOHA (arXiv 2304.13705)](https://arxiv.org/abs/2304.13705) | [search] | Action chunking and temporal ensembling; the compounding-error argument; low-cost bimanual teleop |
| [Grounding Sim-to-Real Generalization (arXiv 2603.22876)](https://arxiv.org/html/2603.22876v2) | [read] | Jun 2026; >10,000 real trials; spatial randomization beats appearance; frame-wise beats episode-wise by 3-13 pts; RL raises real success 5.6% to 33.4%, with DR to 42.8% |
| [Bessemer: Robotics and physical AI](https://www.bvp.com/atlas/bessemer-predicts-robotics-and-physical-ai) | [search] | The "robot data is orders of magnitude scarcer than internet text" framing. **VC analysis, not research** |
| [Robotics scaling law blog (gogoduck912)](https://gogoduck912.github.io/blog/robotics-scaling-law/) | [search] | Source of the ~300,000 h robot / ~1B h video / ~300T tokens comparison. **Independent blog estimate - see unverified** |

## Benchmarks and evaluation methodology

| Source | Status | Good for |
|---|---|---|
| [LIBERO (arXiv 2306.03310)](https://arxiv.org/abs/2306.03310) | [search] | The original benchmark; four suites; authors; intended as a lifelong-learning benchmark |
| [CALVIN (arXiv 2112.03227)](https://arxiv.org/abs/2112.03227) | [search] | 34 tasks, ABC/ABCD splits, five-chained-instruction metric |
| [SIMPLER (arXiv 2405.05941)](https://arxiv.org/abs/2405.05941) | [search] | Visual Matching vs Variant Aggregation; the sim-real ranking correlation claim |
| [PhAIL (arXiv 2605.29710)](https://arxiv.org/pdf/2605.29710) | [read] | May 2026; real-robot benchmark plus distributional methodology (RMST, P-P plots, McNemar, Mantel-Haenszel, bootstrap); critique of trial counts and missing intervals |
| [SureSim (arXiv 2510.04354)](https://arxiv.org/abs/2510.04354) | [search] | Oct 2025; imperfect simulation + small real testing with non-asymptotic confidence intervals |

## Failure modes and robustness

| Source | Status | Good for |
|---|---|---|
| [LIBERO-PRO (arXiv 2510.03827)](https://arxiv.org/html/2510.03827) | [read] | The 90%+ to **0.0%** collapse; the `fdsgfdsgsd` / `xxx` nonsense-instruction experiment; the 0.2-unit position cliff; OpenVLA / pi-0 / pi-0.5 |
| [LIBERO-Plus (arXiv 2510.13626)](https://arxiv.org/abs/2510.13626) | [read] | Seven perturbation dimensions; 95% to below 30% under camera/initial-state perturbation; "models tend to ignore language instructions completely"; positional bias |
| [LIBERO-Para (arXiv 2603.28301)](https://arxiv.org/html/2603.28301v1) | [read] | Mar 30 2026; PRIDE metric (keyword + structural similarity); 22.8-51.9 pt drops across 7 configs of 4 families; 80-96% of failures are planning-level |
| [DAERT (arXiv 2604.05595)](https://arxiv.org/abs/2604.05595v1) | [read] | Apr 7 2026; RL-based diversity-aware adversarial instruction search; **93.33% to 5.85%** on pi-0 and OpenVLA |
| [Q-DIG (arXiv 2603.12510)](https://arxiv.org/html/2603.12510v3) | [read] | Apr 2026; quality-diversity prompt generation; OpenVLA-OFT, pi-0.5, GR00T N1.6; 97.2% archive coverage; adversarial fine-tuning 76.9% to 82.1%, ~15% on GR00T |
| [Eva-VLA (arXiv 2509.18953)](https://arxiv.org/html/2509.18953v1) | [read] | Sep 2025; CMA-ES over object rotation, lighting, patches; OpenVLA >60% failure across all types, 97.8% on long-horizon; **random** perturbations 33.2-55.7% failure; OpenVLA-OFT 4.7% clean to 67.6% |
| [Limited Linguistic Diversity in Embodied AI Datasets (arXiv 2601.03136)](https://arxiv.org/pdf/2601.03136) | [read] | Template reuse and vocabulary poverty in robot instruction data - the root cause of linguistic fragility. **Qualitative summary only; the specific per-dataset statistics were not extracted** |
| [When Vision Overrides Language (arXiv 2602.17659)](https://arxiv.org/pdf/2602.17659) | [read] | Counterfactual vision-language conflict; models follow visual priors over explicit instructions. **The exact per-model numbers were not extracted from the PDF** |
| [When Does Language Matter? (arXiv 2606.11906)](https://arxiv.org/abs/2606.11906) | [read] | LIBERO translated into 10 languages; **30-50% success drop** on non-English; step-wise language sensitivity; inference-time intervention |
| [VLM4VLA (arXiv 2601.03309)](https://arxiv.org/abs/2601.03309) | [read] | Jan 6 2026 (rev May 2026); the visual module, not language, is the bottleneck; general VLM capability poorly predicts control; VLM init still beats scratch |
| [CoRL 2026 Memory for Robot Foundation Models workshop](https://corl2026-memory.github.io/) | [search] | The shallow-memory critique and the "robots repeat failed policies" observation |

## Safety and red-teaming

| Source | Status | Good for |
|---|---|---|
| [RedVLA (arXiv 2604.22591)](https://arxiv.org/html/2604.22591v1) | [read] | Apr 24 2026; two-stage physical red teaming; three-level cost taxonomy; four hazard categories; six models; ASR 92.7% avg / 95.5% peak on pi-0.5 / 64.9% floor on OpenVLA; capability-vulnerability coupling (+20.6 benign, +25.6 ASR); "Success + Unsafe" characterization; SimpleVLA-Guard (0.94/0.89 PRC-AUC, 59.5% ASR reduction, 4-10 pt benign cost) |
| [SafeVLA (arXiv 2503.03480)](https://arxiv.org/abs/2503.03480) | [search] | NeurIPS 2025; constrained-MDP safety alignment; Safety-CHORES benchmark; 83.58% cost reduction with +3.85% success; the "refuses to describe but still does it" framing |
| [VLA-Forget (arXiv 2604.03956)](https://arxiv.org/pdf/2604.03956) | [search] | Apr 2026; unlearning for embodied foundation models. Mentioned only in passing |

## Commercial context

| Source | Status | Good for |
|---|---|---|
| [Humanoid robotics 2026 commercial reality (VaaSBlock)](https://www.vaasblock.com/news/humanoid-robotics-figure-tesla-optimus-commercial-reality-2026/) | [search] | Figure's 11-month BMW Spartanburg pilot; Figure/OpenAI split Feb 2025; 1X NEO positioning; Unitree UnifoLM-VLA-0 open-sourced Jan 2026. **Industry blog - treat deployment claims as approximate** |

---

## Could not verify

Listed explicitly so nobody quotes them as facts.

1. **"~300,000 hours of robot manipulation data exists in total."** From an independent blog and
   echoed by VC analysis. No primary measurement found. The order of magnitude is uncontested but
   the number is an estimate. Present it as such.
2. **"OXE contains about 1/200,000 of the tokens in an LLM corpus."** Attributed in search results
   to the action-tokenization survey (arXiv 2507.01925). I read only that paper's abstract and
   could not confirm the claim. **Not used in the notes.**
3. **"Zero-shot cross-embodiment transfer is 80-100% between arms with similar grippers, 68-96%
   for vision policies."** Appeared in a search synthesis without a clearly attributable paper.
   Used in `03` with hedging; do not put it on a slide as a hard number.
4. **Negative cross-embodiment transfer on Unitree H1 / G1.** Traced to
   [arXiv 2602.18025 (Cross-Embodiment Offline RL for Heterogeneous Robot Datasets)](https://arxiv.org/pdf/2602.18025)
   via search synthesis; the paper itself was not read. The qualitative claim (negative transfer
   when suboptimal data meets high embodiment diversity) is reported by more than one source, so I
   kept it, but the specific robots should be double-checked before lecturing on it.
5. **"Modal per-condition n is 10-20; none of 13 standard-practice papers report confidence
   intervals or paired tests."** This appears in the PhAIL project materials. I read the PhAIL PDF
   summary but did not see that exact table. High confidence in the direction, medium in the
   exact counts.
6. **GR00T N1.6.** Referenced as an evaluated model in the Q-DIG paper. I did not find an
   independent NVIDIA announcement for a version numbered 1.6, only N1 and N1.5. Possibly a
   naming variant. Check before citing.
7. **"Most VLAs degrade past five-minute task lengths"** and **"VLAs fail on categorically new
   tools or contact dynamics."** Both from a commercial model-guide site
   (roboticscenter.ai), not research. Directionally consistent with the CALVIN and generalization
   literature, but they are opinion, and are used in `05` only as framing.
8. **pi-0.5's 94% OOD figures, Helix's 200 Hz claims, Gemini Robotics 2's bar charts, pi\*0.6's
   throughput numbers.** All are real published numbers, and all are **company-reported on
   self-selected tasks with unpublished protocols**. They are not falsified; they are
   unreproducible. Every one of them is labelled in the notes.
