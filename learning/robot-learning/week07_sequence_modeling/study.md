# Week 07 study guide: Sequence Modeling & Transformers

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Matrix products and the softmax function, cross-entropy loss, and week 3's autoregressive discretisation and action chunking. If you have never implemented attention, do it once in NumPy before or after this lecture: it is about ten lines, and it makes every later week easier.

### Extra comments

**Two reasons a policy needs history.** A single frame does not contain the state (how fast is the pedestrian moving, did I already open the drawer?): week 2's state-versus-observation point. And deciding every step independently gives jittery motion. Sequence models address the first; action chunking addresses the second. ([01:00](https://youtu.be/imSTfMJjp7M?t=60), [slide 2](slides.md#slide-2))

**Check the attention numbers yourself.** Softmax depends only on differences between scores: weight ratios are $e^{\Delta s}$. "spoon" (3.1) against "up" (2.8): $e^{0.3} \approx 1.35$, and indeed $38.9\% / 28.8\% \approx 1.35$. "spoon" against "The" (0.4): $e^{2.7} \approx 15$. Adding the same constant to every score changes nothing, which is also why the causal mask can use $-\infty$: $e^{-\infty} = 0$. ([08:00](https://youtu.be/imSTfMJjp7M?t=480), [slide 7](slides.md#slide-7))

**Why divide by $\sqrt{d_k}$.** If query and key entries are roughly independent with unit variance, their dot product has variance $d_k$. With $d_k = 64$ the scores spread over tens of units, the softmax becomes nearly one-hot, and its gradients nearly vanish. Dividing by $\sqrt{d_k}$ puts the scores back at unit scale. ([08:00](https://youtu.be/imSTfMJjp7M?t=480), [slide 7](slides.md#slide-7))

**Attention is a set operation, so order must be added.** Permute the input tokens and the outputs are permuted the same way; nothing else changes. Absolute encodings tag each token with its position; relative ones bias each score by the distance between the two tokens. RoPE rotates queries and keys by angles proportional to their positions, so their dot product depends only on the offset, which is why it extends better to longer sequences than learned absolute positions. ([09:02](https://youtu.be/imSTfMJjp7M?t=542), [slides 8](slides.md#slide-8) and [14](slides.md#slide-14))

**Teacher forcing is behaviour cloning on tokens.** The model always conditions on the true previous tokens, never its own. At generation time it conditions on its own outputs, and one early mistake shifts every later input: week 3's compounding error, called "exposure bias" in NLP. The same fixes have analogues: scheduled sampling is a DAgger cousin, and RL fine-tuning trains on the model's own outputs. ([13:01](https://youtu.be/imSTfMJjp7M?t=781), [slide 10](slides.md#slide-10))

**Scaling-law arithmetic you can do in your head.** Training compute is $C \approx 6ND$ ($N$ parameters, $D$ tokens: about 2 FLOPs per parameter per token forward, 4 backward). Chinchilla-optimal is about 20 tokens per parameter, so a 7B model "wants" about 140B tokens, and GPT-3's 175B parameters wanted about 3.5T tokens (the slide's figure) but got 300B. Llama 3's 8B model was trained on about 15T tokens, roughly 1,900 tokens per parameter: wasteful for training, but a smaller model is cheaper to serve to millions of users, which is the right plot's "inference-optimal" point. ([24:00](https://youtu.be/imSTfMJjp7M?t=1441), [slides 17](slides.md#slide-17) and [18](slides.md#slide-18))

**Image token counts are a latency budget.** A $224 \times 224$ image cut into $16 \times 16$ patches gives $14 \times 14 = 196$ tokens; with 14-pixel patches, as in the SigLIP encoder many VLAs use, it is $16 \times 16 = 256$. Three cameras is then about 768 tokens per step before any language or history, and attention cost grows with the square of the sequence length. This is why robot policies use small images and few history frames. ([28:01](https://youtu.be/imSTfMJjp7M?t=1681), [slide 20](slides.md#slide-20))

**CLIP's loss is classification within the batch.** For each image, the matching caption must win a softmax over all $N$ captions in the batch, and vice versa. Big batches mean more negatives and a harder task (CLIP used batches of 32,768). Zero-shot classification falls out for free: embed "a photo of a {label}" for each label and pick the closest. ([31:01](https://youtu.be/imSTfMJjp7M?t=1861), [slide 22](slides.md#slide-22))

**Flamingo's $\tanh(\alpha)$ gate, with $\alpha = 0$ at the start, is week 6's adaLN-Zero idea again.** New layers inserted into a pre-trained network start as the identity, so the network behaves exactly like the original language model on day one and learns to use the new input gradually. "Start new modules as the identity" shows up all over modern architectures. ([35:01](https://youtu.be/imSTfMJjp7M?t=2101), [slide 24](slides.md#slide-24))

**Action tokens, in practice.** Normalise each action dimension using its 1st and 99th percentiles (so one outlier does not stretch the bins), cut $[-1, 1]$ into 256 bins, and map each bin to a token. OpenVLA and the PaLM-E version of RT-2 overwrite the 256 least-used tokens in the language model's vocabulary; the PaLI-X version of RT-2 reused the tokens for the numbers 0 to 255. ([41:00](https://youtu.be/imSTfMJjp7M?t=2460), [slide 29](slides.md#slide-29))

**Why naive binning fails at high control rates.** At 20 Hz, consecutive actions are almost identical, so each token is almost predictable from the previous one: copying it gets a low loss. The model is trained mostly on trivial predictions and learns little about the informative changes. It is week 3's copycat problem, now inside the tokenizer. ([47:02](https://youtu.be/imSTfMJjp7M?t=2822), [slide 32](slides.md#slide-32))

**FAST, in plain terms.** A smooth trajectory looks like a few slow waves, so its discrete cosine transform puts almost all the information into a few low-frequency coefficients. Round the coefficients (most become zero), list them low-frequency first across dimensions, and BPE-compress the result. The only lossy step is the rounding, whose scale sets the precision. A chunk becomes a handful of dense tokens, each carrying real information. ([50:01](https://youtu.be/imSTfMJjp7M?t=3001), [slide 33](slides.md#slide-33))

**The temporal-ensemble weights point the "wrong" way on purpose.** With $w_i = e^{-m i}$ and $i = 0$ the oldest prediction, older predictions count slightly more. With $m = 0.01$ the difference is small, so the main effect is plain averaging of overlapping chunks, which smooths the motion; ACT's paper defines it the same way (see the transcriber note on slide 30). ([45:00](https://youtu.be/imSTfMJjp7M?t=2700), [slide 30](slides.md#slide-30))

### Going further

- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), then Karpathy's [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) (2 hours).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), and Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (ViT).
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla).
- Brohan et al., [RT-2](https://arxiv.org/abs/2307.15818), and Pertsch et al., [FAST](https://arxiv.org/abs/2501.09747), for actions as tokens.

## Lecture quiz

### Q1

Why does a purely reactive policy $\pi(a_t \mid o_t)$ struggle on many robot tasks?

- A) It has too many parameters.
- B) It cannot output continuous actions.
- C) One frame does not capture the full state (velocities, what happened earlier), and deciding each step independently tends to give jerky motion.
- D) It can only be trained with RL.

<details><summary>Answer</summary>

**C.** Those are slide 2's two points: memory and smoothness. Sequence models address the first, action chunking the second. Reactive policies can output continuous actions and can be trained by imitation (B and D are wrong). ([slide 2](slides.md#slide-2), [01:00](https://youtu.be/imSTfMJjp7M?t=60))

</details>

### Q2

In the attention demo, "spoon" scores 3.1 and "up" scores 2.8. After the softmax, how much more weight does "spoon" get than "up"?

- A) $e^{0.3} \approx 1.35$ times as much
- B) $3.1 / 2.8 \approx 1.11$ times as much
- C) 0.3 more
- D) The same weight

<details><summary>Answer</summary>

**A.** Softmax weights are proportional to $e^{\text{score}}$, so their ratio is $e^{3.1 - 2.8}$. The slide's 38.9% and 28.8% agree: $38.9 / 28.8 \approx 1.35$. ([slide 7](slides.md#slide-7), [08:00](https://youtu.be/imSTfMJjp7M?t=480))

</details>

### Q3

Why are attention scores divided by $\sqrt{d_k}$?

- A) To make attention causal.
- B) Dot products of $d_k$-dimensional vectors grow with $d_k$ (variance about $d_k$); unscaled, the softmax saturates and its gradients vanish.
- C) To make the weights sum to one.
- D) To save memory.

<details><summary>Answer</summary>

**B.** The softmax already makes the weights sum to one (C), and causality comes from the mask (A). The scaling only keeps the scores at a range where the softmax still has useful gradients. ([slide 7](slides.md#slide-7))

</details>

### Q4

Why does a transformer need positional encodings?

- A) To speed up training.
- B) To compress the sequence.
- C) To mask out future tokens.
- D) Attention treats its input as a set: permuting the tokens only permutes the outputs, so without position information word order is invisible.

<details><summary>Answer</summary>

**D.** "The robot picked up the spoon" and "the spoon picked up the robot" would look the same to attention. Masking (C) is a separate mechanism that controls which tokens may be attended to. ([slide 8](slides.md#slide-8), [09:02](https://youtu.be/imSTfMJjp7M?t=542))

</details>

### Q5

Teacher forcing trains each prediction on the true previous tokens. Which week 3 idea is this, and what is the matching problem?

- A) DAgger; no problem remains.
- B) Behaviour cloning; at generation time the model conditions on its own possibly wrong outputs, so errors compound (exposure bias).
- C) Reward shaping; rewards become sparse.
- D) The cross-entropy method; it finds only one mode.

<details><summary>Answer</summary>

**B.** Training always sees expert prefixes, deployment sees the model's own prefixes: the training and test input distributions differ, exactly as in behaviour cloning. DAgger (A) is the fix, not the description. ([slide 10](slides.md#slide-10), [13:01](https://youtu.be/imSTfMJjp7M?t=781))

</details>

### Q6

Using the Chinchilla rule of about 20 tokens per parameter, roughly how many training tokens is compute-optimal for a 7-billion-parameter model?

- A) About 140 billion
- B) About 7 billion
- C) About 1.4 trillion
- D) About 20 billion

<details><summary>Answer</summary>

**A.** $20 \times 7 \times 10^9 = 1.4 \times 10^{11}$. C is the figure for a 70B model. Models meant to be served widely are often trained far beyond this point, because a smaller, over-trained model is cheaper at inference. ([slide 17](slides.md#slide-17), [25:00](https://youtu.be/imSTfMJjp7M?t=1500))

</details>

### Q7

A ViT cuts a $224 \times 224$ image into $16 \times 16$ patches. How many patch tokens does that give?

- A) 224
- B) 16
- C) 768
- D) 196

<details><summary>Answer</summary>

**D.** $224 / 16 = 14$ patches per side, $14^2 = 196$. 768 (C) is the number of values in one patch, $16 \times 16 \times 3$. ([slide 20](slides.md#slide-20), [29:00](https://youtu.be/imSTfMJjp7M?t=1740))

</details>

### Q8

In CLIP's $N \times N$ similarity matrix for a batch, what does the loss push up?

- A) The similarity of each image with itself.
- B) The similarity between different captions.
- C) The diagonal entries, the matching image-caption pairs, relative to every mismatched pair in the same row and column.
- D) The temperature.

<details><summary>Answer</summary>

**C.** Each row is a softmax classification "which caption belongs to this image?", and each column the reverse. The other captions in the batch serve as negatives. ([slide 22](slides.md#slide-22), [31:01](https://youtu.be/imSTfMJjp7M?t=1861))

</details>

### Q9

Why does naive per-timestep binning of actions get worse as the control frequency rises?

- A) Consecutive actions barely change, so each token adds little information and can be predicted by copying the previous one; training is dominated by trivial predictions.
- B) More bins are needed at higher frequency.
- C) Transformers cannot process long sequences.
- D) Robots cannot be controlled at 20 Hz.

<details><summary>Answer</summary>

**A.** The slide's plot peaks at 5 Hz and collapses by 20 Hz. The fix is compression, so that each token carries more information. Robots run at 20 Hz and faster all the time (D). ([slide 32](slides.md#slide-32), [47:02](https://youtu.be/imSTfMJjp7M?t=2822))

</details>

### Q10

What property of robot actions does the discrete cosine transform in FAST exploit?

- A) Actions are already discrete.
- B) Smooth trajectories concentrate their information in a few low-frequency coefficients, so after rounding most coefficients are zero and a chunk compresses into few tokens.
- C) The DCT is learned from robot data.
- D) Nothing: BPE alone handles continuous values.

<details><summary>Answer</summary>

**B.** This is the same reason JPEG works on images. The DCT is a fixed transform, not learned (C), and slide 33 notes that BPE alone is not suited to continuous data (D). ([slide 33](slides.md#slide-33), [50:01](https://youtu.be/imSTfMJjp7M?t=3001))

</details>

## Guest talk

### Context

Ted Xiao is a founding member of technical staff at Project Prometheus; before that he was a staff research scientist and tech lead on Google DeepMind's robotics and Gemini multimodal teams. He was part of many of the papers in this course (QT-Opt, RT-1, RT-2, Open X-Embodiment), so the talk is a first-hand history of a decade of robot learning at Google, in three eras, with his take on what comes next.

### Extra comments

**Era 1, existence proofs (about 2016 to 2021).** The bet: robot hardware was close to ready (the 2007 teleoperated PR1 from week 1 makes the same point), and intelligence was the bottleneck, so put a dozen arms in a room and see if learning methods from games work on real grasping. QT-Opt (week 4) came out of these arm farms. The era ended with the limits of real-world RL becoming clear and a turn to large imitation datasets (BC-Z, MT-Opt, learning from play). ([02:00](https://youtu.be/VS7Ulaugevg?t=120), [04:03](https://youtu.be/VS7Ulaugevg?t=243))

**Era 2, foundation models.** RT-1 took the design rules that made language models work (everything is a token, decoder transformer, a strong pre-trained image encoder, discretised actions) and trained a 35M-parameter policy running at 3 Hz on hundreds of tasks. Language models then entered in three ways: as planners (SayCan), as data augmenters (DIAL relabelled episodes with new instructions using a VLM), and finally as the backbone itself (RT-2). ([13:02](https://youtu.be/VS7Ulaugevg?t=782), [15:00](https://youtu.be/VS7Ulaugevg?t=900))

**SayCan is week 4's Q-function, used as an affordance.** The language model scores how useful each skill is for the instruction; a value function trained on the robot's skills scores how likely each skill is to succeed from the current state; the product picks the next step. The value function is what stops the LLM from proposing "pick up the apple" when no apple is in reach. ([16:00](https://youtu.be/VS7Ulaugevg?t=960))

**Why RT-2 generalised.** It was co-fine-tuned on robot data mixed about 50/50 with web data, and kept answering web questions while learning to output action tokens. Concepts that were never in the robot data (new objects, written text, simple reasoning) transferred to control. That co-training recipe, rather than the tokenisation itself, is the main lesson. ([18:00](https://youtu.be/VS7Ulaugevg?t=1080), [19:01](https://youtu.be/VS7Ulaugevg?t=1141))

**Era 3, scaling.** Gemini Robotics put Gemini 2.0 behind an action model trained mainly on ALOHA bimanual data, then adapted to other bodies, including a humanoid (Apptronik's Apollo; the captions garble the name). Gemini Robotics 1.5 added "thinking" (reasoning in language before acting) and motion transfer: one policy trained on several bodies at once with a shared action space, controlling all of them without per-robot fine-tuning. ([22:01](https://youtu.be/VS7Ulaugevg?t=1321), [23:00](https://youtu.be/VS7Ulaugevg?t=1380))

**A number to reconcile.** He describes Open X-Embodiment as more than 30 labs; week 1's paper abstract says 21 institutions. The collaboration and the dataset kept growing after the first paper version, so both can be right at different times. ([19:01](https://youtu.be/VS7Ulaugevg?t=1141))

**His open questions, worth keeping.** The data pyramid (cheap internet, simulation and human video at the bottom, a little high-quality teleoperation on top) is likely, but the timeline decides strategy: if learning from cheap data takes ten years, teleoperation stays king meanwhile. Video world models bring priors about time, space and physics that text does not, but much of the reasoning in today's best image and video generators still runs through text. And locomotion has been solved mostly by sim-to-real RL while manipulation runs on imitation; nobody yet knows how to unite them. ([30:01](https://youtu.be/VS7Ulaugevg?t=1801), [33:00](https://youtu.be/VS7Ulaugevg?t=1980))

### Going further

- Brohan et al., [RT-1](https://arxiv.org/abs/2212.06817) and [RT-2](https://arxiv.org/abs/2307.15818).
- Ahn et al., [Do As I Can, Not As I Say](https://arxiv.org/abs/2204.01691) (SayCan), and Xiao et al., [Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models](https://arxiv.org/abs/2211.11736) (DIAL).
- Gemini Robotics Team, [Gemini Robotics](https://arxiv.org/abs/2503.20020) and [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342).

## Guest talk quiz

### Q1

What was the bet behind Google Brain's early "arm farms"?

- A) Testing how durable robot hardware is.
- B) Producing marketing videos.
- C) Proving that simulation was already accurate enough.
- D) Hardware was nearly ready and intelligence was the bottleneck; with many robots collecting real experience in parallel, learning methods that worked in games might work for real manipulation.

<details><summary>Answer</summary>

**D.** The DQN and AlphaGo results showed what learning could do in digital worlds; the question was whether scale in real data could do the same for robots. QT-Opt was the main result. ([02:00](https://youtu.be/VS7Ulaugevg?t=120), [04:03](https://youtu.be/VS7Ulaugevg?t=243))

</details>

### Q2

How did SayCan choose the next step of a long task?

- A) The language model alone chose it.
- B) It combined the language model's score for each candidate skill with a value function estimating how likely the robot is to succeed at that skill from the current state.
- C) It chose a random skill.
- D) A human chose it.

<details><summary>Answer</summary>

**B.** The language model knows what would be useful; the value function knows what is feasible right now. Their product gives plans that are both. ([16:00](https://youtu.be/VS7Ulaugevg?t=960))

</details>

### Q3

What is the key idea of RT-2?

- A) Treat actions as text tokens and co-fine-tune a large pre-trained VLM on robot data mixed with web data, so web knowledge transfers to control.
- B) Train a transformer from scratch on robot data only.
- C) Learn the policy with online RL.
- D) Use a separate symbolic planner.

<details><summary>Answer</summary>

**A.** Keeping web data in the mix is what preserved the VLM's knowledge, and is why RT-2 handled objects and concepts never seen in its robot data. ([18:00](https://youtu.be/VS7Ulaugevg?t=1080))

</details>

### Q4

What does "motion transfer" in Gemini Robotics 1.5 do?

- A) Transfers human motion-capture data to the robot.
- B) Retargets internet videos onto robots.
- C) Trains one policy on several robot bodies at once with a shared action space, so it can control all of them without per-robot fine-tuning.
- D) Speeds up the robot's motions.

<details><summary>Answer</summary>

**C.** The earlier Gemini Robotics model needed post-training to move to a new body; training on the bodies together removes that step. ([23:00](https://youtu.be/VS7Ulaugevg?t=1380), [24:01](https://youtu.be/VS7Ulaugevg?t=1441))

</details>

### Q5

What does Xiao say about the future role of high-quality teleoperation data?

- A) It will disappear within a year.
- B) It depends on timing: if learning from cheap, diverse, imperfect data matures soon, teleoperation shrinks to a small fine-tuning layer; if that research takes years, teleoperation remains the best bet.
- C) Only teleoperation data will ever matter.
- D) Simulation will replace all real data.

<details><summary>Answer</summary>

**B.** He expects the data pyramid to win eventually, with teleoperation as "the cherry on top", but says nobody knows the timeline, and the timeline decides what is the right investment today. ([30:01](https://youtu.be/VS7Ulaugevg?t=1801), [31:00](https://youtu.be/VS7Ulaugevg?t=1860))

</details>
