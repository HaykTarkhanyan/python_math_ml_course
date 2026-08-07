# 04 - Evaluation, and everything that is wrong with it

This is the most teachable material in the chapter. VLA evaluation is a case study in how a field
can produce steadily improving numbers that mean progressively less, and it is documented well
enough now that the critique can be made from primary sources rather than opinion.

## How VLAs are benchmarked

### The simulated benchmarks

Three dominate, plus a growing set of stress-test variants.

- **[LIBERO](https://arxiv.org/abs/2306.03310)** (Liu et al., 2023). Table-top manipulation, four
  suites - Spatial, Object, Goal, Long (also called LIBERO-10) - originally built to study
  lifelong-learning knowledge transfer. It is now the default VLA leaderboard, which was never
  its purpose.
- **[CALVIN](https://arxiv.org/abs/2112.03227)** (Mees et al., 2021). Long-horizon and
  language-conditioned: 34 tasks, and the headline metric is how many of five chained
  instructions the policy completes in sequence. The ABC split (train on environments A, B, C,
  test on D) is the generalization test that still has room in it.
- **[SIMPLER / SimplerEnv](https://arxiv.org/abs/2405.05941)** (Li et al., 2024). Simulated
  replicas of the *real* setups used to evaluate real policies (Google Robot, WidowX/Bridge), so
  that a real-robot policy can be scored without hardware. Two modes: Visual Matching, which
  tries to reproduce the real scene as closely as possible, and Variant Aggregation, which
  perturbs backgrounds, lighting, distractors and textures. Its selling point is a demonstrated
  correlation between simulated and real rankings.

Also in use: RLBench, RoboCasa, BEHAVIOR-1K (1,000 everyday activities, multi-room), VLABench,
and THE COLOSSEUM for compounded perturbations.

### The state of the numbers, as of ICLR 2026

From Moritz Reuss's [analysis of 164 VLA submissions to ICLR 2026](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
(up from *one rejected paper* at ICLR 2024 - an 18x year-on-year jump):

- **LIBERO is "basically solved"** at 95-98%. The difference between 98% and 99% carries no
  information.
- **CALVIN**: above 4.0 on ABC is standard, above 4.5 is state of the art (out of 5 chained
  instructions).
- **SIMPLER**: reported success spans **40-99% on Bridge**, which makes cross-paper comparison
  close to meaningless. Roughly 70-80% is current SOTA on the Google Robot variants.
- **RLBench**: VLAs remain "far away" from specialized 3D methods.

Two structural criticisms from the same source, both worth a slide. Performance clustering near
the ceiling masks real progress. And academic models that match frontier baselines *in simulation*
"dramatically lag" them in zero-shot open-world behaviour, because the benchmarks reward local
fine-tuning and never test the thing the models are supposedly for.

### Real-robot evaluation, and why it is barely reproducible

Simulation is a proxy. The claim being made is about physical robots, and physical evaluation has
problems that have no analogue in the rest of ML:

- **The test set is a room.** Lighting drifts through the day. Objects get placed slightly
  differently. The gripper wears. Another lab cannot reproduce your evaluation because they do
  not have your room.
- **Every trial costs a minute and a human reset.** So n is tiny.
- **There is no held-out set in any meaningful sense.** The evaluation scenes are usually
  variations of the training scenes, by necessity.

The trial-count problem has now been measured. Analysis reported alongside
[PhAIL](https://arxiv.org/abs/2605.29710) (May 2026, a real-robot VLA benchmark with an explicit
distributional methodology) finds that across 13 standard-practice papers, **modal per-condition
n is 10-20, and none of the 13 report confidence intervals or paired tests**. Toyota Research
Institute's Large Behavior Model work is cited as the exception, reporting Bayesian credible
regions and paired Barnard's exact / Welch's t-tests with Bonferroni correction at n=50 real and
200 simulated per condition. A separate line of work states plainly that "20-30 real-world trials
are insufficient for statistically significant conclusions".

Do the arithmetic with a class: at n=20, the 95% Wilson interval around an observed 70% success
rate is roughly 48-86%. Two policies reported at 70% and 80% on n=20 are statistically
indistinguishable. Most published VLA comparisons are inside that noise band. This is a first-rate
teaching moment for a course that covers confidence intervals, and it lands harder here than in
any toy example.

The proposed fixes:

- **[SureSim](https://arxiv.org/abs/2510.04354)** (Oct 2025): combine large-scale imperfect
  simulation with small-scale real testing, using non-asymptotic mean estimation to put honest
  confidence intervals on real-world performance.
- **RobotArena-infinity** and Gaussian-splatting real-to-sim: reconstruct the real scene so the
  evaluation is at least shareable.
- **[PhAIL](https://arxiv.org/abs/2605.29710)**: report distributions, not means - per-cell RMST,
  P-P plots, bootstrap, McNemar and Mantel-Haenszel tests.

None of these is standard practice yet.

## Failure mode 1: linguistic fragility

This is the failure mode the chapter should lead with, because it is the most surprising and it
undermines the "language" in "vision-language-action" directly.

### The benchmark scores are memorization

**[LIBERO-PRO](https://arxiv.org/abs/2510.03827)** (Zhou et al., updated May 2026) perturbs LIBERO
along four axes - object attributes, initial positions, instruction wording, environment - and
evaluates OpenVLA, pi-0 and pi-0.5. The result is the single most quotable number in this
literature: models that exceed **90% on standard LIBERO collapse to 0.0%** under the generalized
setting.

The diagnostic experiment is better than the headline. Replace the task instruction with a
**meaningless character string** - the paper's examples are `fdsgfdsgsd` and `xxx` - and the model
"still executes nearly the same action trajectory as in the original task". The policy is not
reading the instruction. It is pattern-matching the scene to a memorized trajectory. Reported
accuracies, the authors conclude, indicate "rote recall rather than robust task understanding".

They also find a hard positional cliff: OpenVLA and pi-0 success "collapses once the displacement
exceeds 0.2 units, dropping sharply to zero thereafter"; pi-0.5 holds to about 0.4.

### The same finding from a different direction

**[LIBERO-Plus](https://arxiv.org/abs/2510.13626)** (Fei et al., Dec 2025) perturbs seven
dimensions: object layout, camera viewpoint, robot initial state, language instruction, lighting,
background texture, sensor noise. Findings:

- Performance drops **from 95% to below 30%** under modest camera-viewpoint and initial-state
  perturbations.
- Models appear *insensitive* to language variation - and the follow-up experiments explain why:
  they "tend to ignore language instructions completely". Swap the target object named in the
  instruction and the model executes the original task anyway; measured success in the swapped
  condition drops to near zero, worst for OpenVLA-OFT.

Note the trap in the reasoning here, and make sure students catch it. "Robust to paraphrase"
looks like a good result. It is actually evidence the language channel is dead. The paper's
framing - that models "exhibit positional bias rather than genuine semantic understanding" - is
the right one.

### But they are also fragile to paraphrase

The two results are not contradictory; they apply to different model families and different
perturbation strengths, and this is exactly the kind of disagreement to report rather than
resolve.

**[LIBERO-Para](https://arxiv.org/abs/2603.28301)** (Kim et al., March 2026) builds a paraphrase
benchmark and a metric, **PRIDE**, combining keyword similarity (are the task-critical action and
object words preserved?) with structural similarity (dependency-tree edit distance). Across seven
configurations of four architecture families spanning 0.6B-7.5B (OpenVLA-OFT variants, pi-0.5,
X-VLA, VLA-Adapter, Xiaomi-Robotics-0), **every model degraded, by 22.8 to 51.9 percentage
points**. Object-level lexical variation drove most of it. 80-96% of failures were planning-level
rather than execution-level - the robot moved competently to the wrong thing.

### Adversarial instruction search makes it much worse

**[DAERT](https://arxiv.org/abs/2604.05595)** (Tong et al., April 2026), "Diversity-Aware Embodied
Red Teaming", uses RL to *search* the space of rephrasings for ones that break the policy, with an
explicit diversity objective so the search does not collapse onto one attack template. Against
pi-0 and OpenVLA in simulation, it drove **average task success from 93.33% to 5.85%**.

That is the number to put on the slide. Not an adversarial image patch, not a jailbreak - just
different English words for the same request.

**[Q-DIG](https://arxiv.org/html/2603.12510v3)** (Srikanth et al., April 2026) does the same with
quality-diversity optimization, using a VLM as a mutator across eight attack styles (step-by-step
phrasing, uncommon vocabulary, human-centric tone, and so on), against OpenVLA-OFT, pi-0.5 and
GR00T N1.6 on SimplerEnv and LIBERO-Goal. It reports 97.2% archive coverage against 36-38% for
baselines, and - importantly - a **defense that partly works**: fine-tuning on the generated
adversarial instructions raised OpenVLA-OFT from 76.9% to 82.1% on *unseen* adversarial
instructions, with about 15% improvement on GR00T N1.6. Human raters judged the generated
instructions to be natural, which matters: these are not gibberish attacks.

### Why the language channel is weak: two structural causes

**Cause 1: the instructions in the training data are templated.**
[Limited Linguistic Diversity in Embodied AI Datasets](https://arxiv.org/abs/2601.03136) (Wanna et
al., 2026) measures instruction variation across the major robot datasets and finds high template
reuse, limited vocabulary, and large fractions of instructions following identical structural
patterns. If every demonstration says "pick up the X and place it in the Y", the model learns the
slot filler, not the sentence. Everyone has been optimizing task diversity and nobody has been
optimizing linguistic diversity.

**Cause 2: vision dominates when the two channels conflict.**
[When Vision Overrides Language](https://arxiv.org/abs/2602.17659) (Fang et al., 2026) constructs
counterfactual scenes where the instruction contradicts the visual prior, and finds models
"systematically ignore conflicting language instructions in favor of visual cues", even when
following the text would have produced the correct action. Combine this with the LIBERO-Plus
finding and a consistent picture emerges: the visual pathway carries the policy, and language is
close to a task-ID lookup.

This is corroborated architecturally by [VLM4VLA](https://arxiv.org/abs/2601.03309) (Zhang et al.,
January 2026), whose modality-level ablations identify **the visual module, not the language
component, as the primary performance bottleneck**, and which finds that improving a VLM's
performance on embodied QA/pointing/depth tasks does *not* reliably improve downstream control.
Their conclusion: standard VLM competence is necessary but not sufficient, and there is a
"persistent domain gap between current VLM pretraining objectives and the requirements of embodied
action-planning".

### Non-English is much worse

Directly relevant to a bilingual course:
[When Does Language Matter?](https://arxiv.org/abs/2606.11906) (2026) translated LIBERO into ten
languages and measured **success rate drops of 30-50% under non-English instructions**. Its more
interesting finding is *step-wise language sensitivity*: language dependence is highly non-uniform
across the trajectory, with a few steps dominating total failure while most steps are effectively
language-agnostic. Their fix is an inference-time intervention applied only at the sensitive
steps.

For a course taught in Armenian and English, this is a concrete, checkable claim to raise with
students: if you gave an existing open VLA an instruction in Armenian, it would very probably fail,
and now you know roughly why and where.

## Failure mode 2: physical and visual brittleness

**[Eva-VLA](https://arxiv.org/abs/2509.18953)** (Liu et al., SJTU, Sept 2025) turns discrete
physical variations into a continuous black-box optimization using CMA-ES, searching over object
3D rotations, Gaussian point-light illumination, and adversarial patch placement. Against OpenVLA
and OpenVLA-OFT on LIBERO:

- OpenVLA failure rates exceed **60% across all variation types**.
- Object 3D transformation is worst: **97.8% failure on long-horizon tasks**.
- **Even random (non-adversarial) perturbations** cause 33.2-55.7% failure.
- OpenVLA-OFT, despite a 4.7% clean failure rate, hits **67.6% failure** under object
  transformations.

That last pair is the important one: the better model on the leaderboard is not the more robust
model.

The COLOSSEUM finding reported in the [datasets and benchmarks survey](https://arxiv.org/abs/2604.23001)
generalizes it - single-axis robustness does not extrapolate to compounded perturbations, and
almost every benchmark varies one factor at a time.

## Failure mode 3: long horizons, memory, and no recovery

CALVIN's own numbers make the point: chained-instruction success drops to **0.08%** for five
sequential instructions ([survey](https://arxiv.org/abs/2604.23001)). And the survey's real
complaint is diagnostic - when a long-horizon task fails, the benchmark cannot tell you whether it
was planning, memory, or skill composition that broke.

Two related structural gaps:

- **Memory is shallow.** Robot foundation models handle memory as "a window of recent context, a
  replay buffer, or retrieval over past observations" ([CoRL 2026 memory
  workshop](https://corl2026-memory.github.io/)). A consequence noted there: robots repeat
  policies that already failed, because the failure is outside the context window.
- **No recovery behaviour, by construction.** Policies are trained on *successful* demonstrations.
  There is no corrective experience in the data, so when the robot reaches an unexpected state it
  has nothing to condition on. This is exactly the gap that pi\*0.6's RECAP, and 2026 work like
  RePO-VLA and Failing Forward, target.

## Failure mode 4: unsafe physical behaviour, and embodied red-teaming

Text-model safety does not transfer. The framing in [SafeVLA](https://arxiv.org/abs/2503.03480)
(NeurIPS 2025) is the clearest statement of why: a model that would refuse to *describe* how to
knock over a glass will still emit an action trajectory that knocks over the glass. The safety
constraint was learned in token space; the harm happens in configuration space.

### RedVLA

**[RedVLA](https://arxiv.org/abs/2604.22591)** (Zhang, Zhang, Fan, Shen, Cai, Yang, Ji; April 24,
2026) is the reference work, billed as the first red-teaming framework for VLA *physical* safety.
Its framing is the important part: the risk has "fundamentally shifted from the intent space to
the physical space". The attacker is not trying to make the model say something bad; the attacker
is arranging the *environment*.

Method, two stages:

1. **Risk Scenario Synthesis.** Identify the critical interaction regions along a benign
   trajectory, then inject risk objects (a knife, a bottle) into the scene while keeping the task
   feasible and the instruction semantically unchanged. The scene stays benign-looking and the
   instruction stays innocent - this is not a prompt attack.
2. **Trajectory-Driven Risk Amplification.** Gradient-free optimization over the placement of the
   risk object, iteratively refined until it reliably triggers unsafe behaviour. Because it is
   gradient-free, it works across heterogeneous model families.

Risk taxonomy, three cost levels: **state-level** (a single state-action pair is a violation),
**cumulative-level** (harm accumulates over time), **conditional-level** (violation requires
temporal logic over state dependencies). Four hazard categories: resource damage, dangerous item
misuse, robot damage, environmental harm.

Evaluated on LIBERO against six models in three families - OpenVLA, OpenVLA-OFT, VLA-Adapter,
VLA-Adapter-Pro, pi-0, pi-0.5 - plus sim-to-real validation on a Franka. Results:

- Peak **attack success rate 95.5% on pi-0.5** within 10 optimization iterations.
- Average ASR **92.7%** across the five stronger models; range 64.9% (OpenVLA) to 95.5% (pi-0.5).
- By risk level: >95% state-level, 88.9% cumulative, 66.1% conditional.
- **Capability and vulnerability move together.** OpenVLA-OFT improved benign success by 20.6
  points over OpenVLA and increased ASR by 25.6 points. The more competent policy is the more
  dangerous one, because it actually gets to the object.
- Most unsafe rollouts were classified "Success + Unsafe" or "Attempt + Unsafe" - i.e. the robot
  **completed the task and caused harm on the way**. It did not break down; it succeeded
  dangerously. A pure task-success metric would have scored these as wins.

RedVLA also proposes **SimpleVLA-Guard**, an LSTM over the policy's internal latent features that
halts execution before an unsafe action, trained on RedVLA-generated scenarios, with Functional
Conformal Prediction for threshold calibration. Reported 0.94 PRC-AUC offline on seen tasks and
0.89 on unseen, a 59.5% reduction in ASR online, at the cost of a 4-10 point drop in benign task
performance.

### The rest of the embodied safety landscape

- **[SafeVLA](https://arxiv.org/abs/2503.03480)** (NeurIPS 2025) formulates safety as a constrained
  MDP and optimizes from a min-max perspective against elicited unsafe behaviours. Reports an
  **83.58% reduction in cumulative safety-violation cost** with **+3.85% task success** on its
  Safety-CHORES benchmark (procedurally generated scenes with corners, blind spots, fragile
  collections and dangerous equipment). The joint improvement is the notable claim - safety and
  capability did not trade off here.
- **[The VLA safety survey](https://arxiv.org/abs/2604.23775)** (Li, Yin, Huang et al., April 2026)
  organizes the threat space into adversarial patches, prompt injection, and backdoors, and
  catalogues defenses (input filtering, safety fine-tuning, uncertainty quantification,
  adversarial training). Useful as a lecture skeleton; light on numbers.
- **VLA-Forget** (April 2026) applies machine unlearning to embodied foundation models - removing
  a learned capability from a deployed policy.

## Practical checklist: how to read a VLA result

Worth giving students as a handout.

1. **Simulation or real?** If real, what is n? If n < 30 and there is no interval, the comparison
   is probably noise.
2. **Which LIBERO?** LIBERO at 97% means almost nothing in 2026. LIBERO-Plus, LIBERO-PRO or
   LIBERO-Para numbers mean something.
3. **Was the instruction varied?** If every evaluation used the exact training phrasing, the
   language claim is untested.
4. **Was anything varied at all?** Camera pose and object placement are the axes models actually
   break on.
5. **Company blog or paper?** pi-0.5's 94%, Helix's dexterity, Gemini Robotics 2's charts - all
   company-reported on self-chosen tasks with unpublished protocols. Not fraudulent, not
   reproducible.
6. **Is a demo being used as evidence?** Eighteen hours of espresso is a demo. It tells you the
   system does not fall over; it does not tell you the success rate.
