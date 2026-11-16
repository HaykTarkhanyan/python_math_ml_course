# Week 08 study guide: World Models

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Week 2's transition function $P(s_{t+1} \mid s_t, a_t)$ (a world model is a learned one), week 4's cross-entropy method, week 5's actor-critic and pathwise gradients, week 6's VAE and flow matching, and week 7's video tokens. This lecture reuses almost everything so far. The first 13 minutes are project logistics; the lecture proper starts at [13:00](https://youtu.be/cTTmUZlOF2s?t=780).

### Extra comments

**A world model is the $P$ that week 4 assumed.** Value iteration needed the dynamics model; model-free RL threw it away and paid in samples. Learning $P$ from data brings planning back: imagine many futures, pick the best, act once in the real world. Policies answer "what should I do?"; world models answer "what happens if I do this?". ([15:00](https://youtu.be/cTTmUZlOF2s?t=900), [slides 11](slides.md#slide-11) and [12](slides.md#slide-12))

**"Action-conditioned" is the defining word.** A model that generates video from a text prompt cannot answer "what if I push left instead of right?", so by the lecture's definition it is not a world model, however realistic its videos look. Keep this in mind when papers and companies call any video generator a world model. ([20:01](https://youtu.be/cTTmUZlOF2s?t=1201), [slide 14](slides.md#slide-14))

**L2 blur is the same averaging problem for the third time.** Behaviour cloning with squared error averages modes (week 3), a VAE decoder with squared error averages reconstructions (week 6), and a video predictor with squared error averages possible futures into a blur (slide 17). The fix is always the same in spirit: model the distribution (latent variables, diffusion, flow) instead of predicting its mean. Finn et al.'s other trick, predicting how pixels move rather than their values, is why their model generalised to objects it had never seen. ([24:02](https://youtu.be/cTTmUZlOF2s?t=1442), [slides 15](slides.md#slide-15) and [17](slides.md#slide-17))

**Visual MPC is week 4's CEM with a learned simulator.** Sample action sequences, roll each out in the video model, score the predicted frames against a goal, refit the sampling distribution to the best ones, execute only the first action and replan. Replanning every step is what keeps accumulated prediction error in check, and it is also why this is slow. ([22:03](https://youtu.be/cTTmUZlOF2s?t=1323), [slide 17](slides.md#slide-17))

**Imagination drift is exposure bias.** The memory RNN in Ha and Schmidhuber's model was trained on encodings of real frames and is then fed its own sampled predictions: week 7's teacher-forcing mismatch and week 3's compounding error, once more. Their paper has a neat related trick: raising the sampling temperature of the dream made it harder and more uncertain, which stopped the controller from exploiting the model's mistakes (in the VizDoom task, low-temperature dreams contained exploits that did not exist in the real game). Agents exploiting world-model errors is the downside listed for Dreamer on slide 29 too. ([33:00](https://youtu.be/cTTmUZlOF2s?t=1980), [slide 26](slides.md#slide-26))

**The controller was tiny on purpose.** In the original World Models, C is a single linear layer with under a thousand parameters, trained with CMA-ES, an evolution strategy that is a close relative of CEM. Putting the capacity into V and M and keeping C small meant a gradient-free optimiser could still train it. ([27:00](https://youtu.be/cTTmUZlOF2s?t=1620), [slide 23](slides.md#slide-23))

**The RSSM's prior and posterior are week 3's latent plans again.** The posterior $q(z_t \mid h_t, o_t)$ sees the observation, the prior $p(z_t \mid h_t)$ does not, and a KL term trains the prior to predict what the posterior will say. In imagination there are no observations, so you sample from the prior, exactly as the play-data model sampled plans from its prior at test time. The deterministic path $h_t$ is never sampled, which is what keeps long imagined rollouts from wandering off. ([35:01](https://youtu.be/cTTmUZlOF2s?t=2101), [slide 27](slides.md#slide-27))

**Dreamer turns planning into a forward pass.** PlaNet ran CEM in latent space at every step. Dreamer trains an actor-critic inside the frozen world model, backpropagating through the learned dynamics (week 5's pathwise gradient, since the model is differentiable), so acting needs one network call. DayDreamer used this to teach a real quadruped to walk in about an hour of real interaction. ([38:00](https://youtu.be/cTTmUZlOF2s?t=2280), [40:00](https://youtu.be/cTTmUZlOF2s?t=2400), [slide 29](slides.md#slide-29))

**Why DreamerV3's symlog matters.** Rewards differ by orders of magnitude across tasks (Atari scores in the thousands, robot rewards near 1). $\text{symlog}(x) = \text{sign}(x)\ln(1 + |x|)$ squashes large values and leaves small ones nearly unchanged, so one set of hyperparameters works across domains without per-task reward scaling. That robustness is what let one configuration reach diamonds in Minecraft. ([42:02](https://youtu.be/cTTmUZlOF2s?t=2522), [slide 32](slides.md#slide-32))

**Untangling the compression numbers on slide 36.** The transcriber note flags that the figures do not agree. The likely source: a tokenizer named "8x8x8" (Cosmos CV8x8x8) downsamples time, height and width by 8 each, so 512 times fewer positions than raw pixels, not 512 times fewer than ViT patch tokens. Always check whether a compression factor is per side, per area, against pixels or against patches before comparing models. The 51,200-token figure itself is right: $256$ tokens per frame $\times$ 200 frames. ([47:02](https://youtu.be/cTTmUZlOF2s?t=2822), [slides 35](slides.md#slide-35) and [36](slides.md#slide-36))

**Inverse dynamics models are an old, strong trick.** An IDM predicts $a_t$ from $(o_t, o_{t+1})$: given what happened, which action caused it? That is much easier than choosing the action (no planning, no intent), so it needs little labelled data. OpenAI's VPT trained an IDM on a small labelled Minecraft set and used it to label about 70,000 hours of online video. Video-action models reuse the idea: let the video model imagine the future and let a small IDM-like head read actions off it. ([54:00](https://youtu.be/cTTmUZlOF2s?t=3240), [slide 37](slides.md#slide-37))

**What the mimic-video ablation actually shows.** Given features from the true future video, the action decoder succeeds every time; given the model's own predicted video, success depends on video quality. So the bottleneck is video prediction, not the action head, and better video models should mean better policies without more robot data. Running the video model only partway through denoising and reading its features is the speed trick that makes this usable for control. ([01:00:01](https://youtu.be/cTTmUZlOF2s?t=3601), [slides 42](slides.md#slide-42) and [43](slides.md#slide-43))

**JEPA's collapse problem in one line.** If you only ask "predict the next embedding", the encoder can map every frame to the same vector and score a perfect loss. The three fixes on slide 51 all stop that: a slowly updated (EMA) target encoder, a frozen pre-trained encoder (DINO-WM), or a regulariser that forces embeddings to spread out like a Gaussian. ([01:10:01](https://youtu.be/cTTmUZlOF2s?t=4201), [slide 51](slides.md#slide-51))

### Going further

- Ha and Schmidhuber, [World Models](https://arxiv.org/abs/1803.10122). The paper was published with an interactive web version; the arXiv abstract links to it.
- Hafner et al., [Dream to Control](https://arxiv.org/abs/1912.01603) (Dreamer), [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (DreamerV3) and [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527) (Dreamer 4).
- Wu et al., [DayDreamer](https://arxiv.org/abs/2206.14176), world models on real robots.
- Baker et al., [Video PreTraining (VPT)](https://arxiv.org/abs/2206.11795), the inverse-dynamics-labelling idea at scale.
- Assran et al., [V-JEPA 2](https://arxiv.org/abs/2506.09985), and Zhou et al., [DINO-WM](https://arxiv.org/abs/2411.04983), for latent world models without pixel decoding.

## Lecture quiz

### Q1

What does a world model compute, as opposed to a policy?

- A) $\pi(a_t \mid s_t, g)$: which action to take for a goal.
- B) Only the reward of the current state.
- C) The goal the user wants.
- D) $p(s_{t+1} \mid s_t, a_t)$: what happens next if a given action is taken.

<details><summary>Answer</summary>

**D.** A policy maps state and goal to an action; a world model maps state and action to the next state. Having the second lets you try actions in imagination before committing in the real world. ([slide 11](slides.md#slide-11), [15:00](https://youtu.be/cTTmUZlOF2s?t=900))

</details>

### Q2

By the lecture's definition, why is a text-to-video generator not strictly a world model?

- A) It is too slow.
- B) It is not conditioned on actions, so you cannot ask what would happen under different actions.
- C) It uses diffusion.
- D) It has no memory of past frames.

<details><summary>Answer</summary>

**B.** The definition is $P(s_{t+1} \mid s_t, a_t, h_t)$. A model conditioned only on text cannot be used to compare candidate actions, which is the whole point for planning. ([slide 14](slides.md#slide-14), [20:01](https://youtu.be/cTTmUZlOF2s?t=1201))

</details>

### Q3

Why were the video predictions in Finn et al. (2016) blurry?

- A) The cameras had low resolution.
- B) The model had too few parameters.
- C) With an L2 loss, the best prediction under uncertainty is the average of the possible futures, which looks blurred.
- D) The flow field was wrong.

<details><summary>Answer</summary>

**C.** Squared error rewards predicting the mean. When the future is uncertain, the mean of several sharp futures is a blurry one. It is the same averaging as mode-averaging in behaviour cloning. ([slide 17](slides.md#slide-17), [24:02](https://youtu.be/cTTmUZlOF2s?t=1442))

</details>

### Q4

How does Visual MPC choose an action?

- A) It samples many action sequences, predicts the video for each with the learned model, scores the predictions against the goal, refits the sampling distribution (CEM), executes the best first action and replans.
- B) A trained actor network outputs it.
- C) DQN picks it.
- D) It picks a random action.

<details><summary>Answer</summary>

**A.** It is planning with a learned simulator and a gradient-free optimiser. Replanning at every step limits accumulated prediction error but makes it computationally expensive, one of the cons on slide 17. ([slide 17](slides.md#slide-17), [22:03](https://youtu.be/cTTmUZlOF2s?t=1323))

</details>

### Q5

Why do imagined rollouts in the original World Models drift away from reality?

- A) The VAE is too small.
- B) CMA-ES is used for the controller.
- C) The controller is linear.
- D) The RNN was trained on encodings of real frames but in imagination receives its own sampled predictions, so errors compound.

<details><summary>Answer</summary>

**D.** It is exposure bias, the same training-versus-deployment mismatch as teacher forcing and behaviour cloning. The RSSM's learned prior and deterministic memory path are the fix. ([slide 26](slides.md#slide-26), [33:00](https://youtu.be/cTTmUZlOF2s?t=1980))

</details>

### Q6

What is the RSSM's prior $p(z_t \mid h_t)$ for?

- A) Reconstructing the images.
- B) Producing a plausible latent state when there is no observation, as in imagination; a KL term trains it to agree with the posterior that does see observations.
- C) Encoding the reward.
- D) Speeding up training.

<details><summary>Answer</summary>

**B.** Without a learned prior there is no principled way to form a belief without an observation, which was the second problem on slide 26. ([slide 27](slides.md#slide-27), [34:01](https://youtu.be/cTTmUZlOF2s?t=2041))

</details>

### Q7

How does Dreamer differ from PlaNet?

- A) PlaNet plans with CEM at every step; Dreamer trains an actor-critic in imagination by backpropagating through the frozen world model, so acting takes one forward pass.
- B) Dreamer has no world model.
- C) PlaNet trains its policy with RL.
- D) There is no difference.

<details><summary>Answer</summary>

**A.** Both share the RSSM. Dreamer amortises planning into a learned policy, which is fast at test time but can exploit the model's errors (the con on slide 29). ([slide 29](slides.md#slide-29), [38:00](https://youtu.be/cTTmUZlOF2s?t=2280))

</details>

### Q8

A 10-second clip at 20 frames per second has $256 \times 256$ frames, each cut into $16 \times 16$ patches. How many tokens is that with per-frame ViT tokenisation?

- A) 256
- B) 5,120
- C) 51,200
- D) 200

<details><summary>Answer</summary>

**C.** $(256/16)^2 = 256$ tokens per frame, times 200 frames. A (256) is one frame, D (200) the number of frames. That is why video models need aggressive spatial and temporal compression. ([slide 35](slides.md#slide-35), [47:02](https://youtu.be/cTTmUZlOF2s?t=2822))

</details>

### Q9

In a video-action model like mimic-video, what mainly has to be learned from robot data?

- A) The whole video model, from scratch.
- B) A lightweight action decoder that reads actions off the video model's features; the video backbone comes pre-trained on internet video (optionally fine-tuned on robot video).
- C) Nothing; it works zero-shot.
- D) A vision-language model.

<details><summary>Answer</summary>

**B.** The video model already knows how things move; robot data only has to teach control. That division is where the claimed 10x sample efficiency over VLAs comes from. ([slides 40](slides.md#slide-40) and [41](slides.md#slide-41), [58:02](https://youtu.be/cTTmUZlOF2s?t=3482))

</details>

### Q10

A JEPA world model predicts the next frame's embedding instead of its pixels. What can go wrong, and what is one fix?

- A) Nothing can go wrong without a decoder.
- B) A pixel decoder fixes everything.
- C) More data fixes it automatically.
- D) The encoder can map every frame to the same vector and get zero loss; fixes include a slowly updated (EMA) target encoder, a frozen pre-trained encoder, or a regulariser that forces spread-out embeddings.

<details><summary>Answer</summary>

**D.** Collapse is the central problem of every self-supervised method that does not reconstruct its input. Slide 51 lists the three standard remedies and their trade-offs. ([slide 51](slides.md#slide-51), [01:10:01](https://youtu.be/cTTmUZlOF2s?t=4201))

</details>

## Guest talk

### Context

Scott Reed is a founding member of NVIDIA's GEAR lab (Generalist Embodied Agent Research, led by Jim Fan and Yuke Zhu; the captions write "Yuka"). Before that he spent eight years at Google DeepMind on generative models and generalist agents and led Gato, one transformer trained on text, images, Atari and robot data. His PhD was at Michigan with Honglak Lee. The talk compares three choices of "backbone" for robot policies: a vision-language model (VLA), a video model (world action model), or training from scratch.

### Extra comments

**What "backbone" means here.** The network that turns camera images, robot sensor readings and the instruction into one representation that actions are predicted from. The choice decides not just inference speed but what pre-training knowledge the policy can absorb. ([01:00](https://youtu.be/fqkp_wkov6M?t=60), [02:00](https://youtu.be/fqkp_wkov6M?t=120))

**Why $\pi$ models use discrete and continuous actions at once.** Training the VLM only on discrete tokens (FAST) lets it be co-trained with ordinary language tasks, while sending flow-matching gradients straight into the VLM tends to damage its language abilities (catastrophic forgetting). The separate flow-matching action expert then produces the actual continuous actions, with no risk of a garbled token decoding into a wild arm motion, which he raises as a safety point. ([06:01](https://youtu.be/fqkp_wkov6M?t=361), [07:02](https://youtu.be/fqkp_wkov6M?t=422))

**Read the TRI large-behavior-model paper for its method.** He recommends it less for the model than for the evaluation: blind head-to-head comparisons (evaluators do not know which policy is which), run on the same robot at the same time, with statistical tests. That is the level of care real-robot claims need; many papers do not reach it. ([09:02](https://youtu.be/fqkp_wkov6M?t=542))

**"Data islands."** Pooling data from many robots is the hope behind cross-embodiment training, but a model can learn each robot's data as a separate cluster with little transfer between them, like the human and robot clusters in Danfei Xu's week 3 talk. Reed sees little evidence yet of real zero-shot transfer to new embodiments in today's VLAs, and argues video prediction suffers less from this because everything shares one output space: pixels. ([12:02](https://youtu.be/fqkp_wkov6M?t=722), [21:00](https://youtu.be/fqkp_wkov6M?t=1260))

**World action models, three flavours.** DreamZero jointly denoises video and actions with a causal DiT and a KV cache; Cosmos Policy also predicts future robot state and value, tiled into the image latent so that everything looks like video to the model; ByteDance's GR-1 and GR-2 are autoregressive and interleave text, state and images, closest to Gato. His analogy: like reasoning in LLMs, much of the benefit of predicting video may come from doing it during pre-training, which is why the "Flash" variant that barely denoises the video at inference still works. ([13:00](https://youtu.be/fqkp_wkov6M?t=780), [15:02](https://youtu.be/fqkp_wkov6M?t=902))

**The failure analysis is the key result.** DreamZero's failures mostly traced back to bad predicted video: the robot did what the imagined video showed. That matches the mimic-video ablation in the lecture, and it means improving the video model (possibly with human video, no robot needed) should improve the policy. The 14B model was much better than the 2B one. ([19:01](https://youtu.be/fqkp_wkov6M?t=1141), [20:00](https://youtu.be/fqkp_wkov6M?t=1200))

**Their costs.** Sampling is expensive, latents built to reconstruct pretty video may encode far more than control needs, and context length is still short. Domain-specific tokenisers (as in Minecraft world models) compress far more; whether that works for open-world robotics is an open research question. ([21:00](https://youtu.be/fqkp_wkov6M?t=1260), [22:00](https://youtu.be/fqkp_wkov6M?t=1320))

**From scratch is back, thanks to human data.** UMI-style companies (he names Generalist and Sunday, week 6's guest) collect very large amounts of data from sensorised humans, and egocentric camera data scales further. In the study he cites (its name is garbled in the captions), a large amount of head-camera video plus only about 50 hours with wrist and finger tracking gave strong results, with success rising steadily as human data grew. ([24:02](https://youtu.be/fqkp_wkov6M?t=1442), [25:01](https://youtu.be/fqkp_wkov6M?t=1501))

**Two Q&A answers worth remembering.** On simulators: neural simulators may take over some contact-rich manipulation, but hand-written physics engines are so much cheaper to run that they will stay for whole-body control. On force and touch: as a language model must implicitly model the world to predict text well, a good enough video model must carry features correlated with forces; a small amount of sensorised data may be enough to draw them out. ([31:02](https://youtu.be/fqkp_wkov6M?t=1862), [33:01](https://youtu.be/fqkp_wkov6M?t=1981))

### Going further

- Reed et al., [A Generalist Agent](https://arxiv.org/abs/2205.06175) (Gato).
- TRI LBM Team, [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://arxiv.org/abs/2507.05331), for how to evaluate robot policies properly.
- Wu et al., [Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation](https://arxiv.org/abs/2312.13139) (GR-1).
- NVIDIA, [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575), the video backbone family behind Cosmos Policy.

## Guest talk quiz

### Q1

What does Reed mean by the "backbone" of a robot policy?

- A) The robot's mechanical spine.
- B) The action decoder alone.
- C) The network that encodes camera, robot-sensor and language inputs into a shared representation used for predicting actions; it decides what pre-training knowledge transfers.
- D) The training dataset.

<details><summary>Answer</summary>

**C.** His three options are VLM-based, video-model-based, or trained from scratch, and the choice shapes both inference cost and transfer from pre-training. ([02:00](https://youtu.be/fqkp_wkov6M?t=120))

</details>

### Q2

Why do the $\pi$ models predict discrete action tokens and also have a continuous flow-matching action expert?

- A) Discrete tokens let the VLM be co-trained with language tasks without flow-matching gradients damaging it; the flow expert produces the real continuous actions and avoids token-decoding errors that could cause dangerous motions.
- B) Only for speed.
- C) To support more robot types.
- D) The robot hardware requires both.

<details><summary>Answer</summary>

**A.** He describes two problems: catastrophic forgetting when continuous-action gradients flow into the VLM, and the safety risk of a mis-decoded discrete action. The two heads address one each. ([06:01](https://youtu.be/fqkp_wkov6M?t=361), [08:00](https://youtu.be/fqkp_wkov6M?t=480))

</details>

### Q3

What are "data islands"?

- A) Robots placed far apart in a warehouse.
- B) Storage limits on robot datasets.
- C) Datasets owned by different labs.
- D) Data from different embodiments learned as separate clusters with little transfer between them, so pooling cross-embodiment data helps less than hoped.

<details><summary>Answer</summary>

**D.** It is his main reservation about cross-embodiment VLAs, and one reason he finds video prediction appealing: all embodiments share the same output space. ([12:02](https://youtu.be/fqkp_wkov6M?t=722))

</details>

### Q4

What did the failure analysis of DreamZero find?

- A) Most failures came from the action head.
- B) Most failures came from errors in the predicted video; the robot followed what the imagined video showed, so better video models should give better policies.
- C) Failures were random.
- D) Most failures came from misread language instructions.

<details><summary>Answer</summary>

**B.** That puts the bottleneck in video prediction, which can improve with more (including human) video, not only with more robot data. ([20:00](https://youtu.be/fqkp_wkov6M?t=1200))

</details>

### Q5

Will learned world models make physics simulators redundant, according to Reed?

- A) Not soon: neural simulators may help for contact-rich manipulation and deformable objects, but for whole-body control hand-written simulators are far cheaper to run.
- B) Yes, already.
- C) Physics simulators are better than neural ones for deformable objects.
- D) Only RL will be needed, with no simulator at all.

<details><summary>Answer</summary>

**A.** He expects both to coexist for a long time. C is the reverse of his view: deformables and contact are where he thinks neural simulation may shine. ([31:02](https://youtu.be/fqkp_wkov6M?t=1862), [32:02](https://youtu.be/fqkp_wkov6M?t=1922))

</details>
