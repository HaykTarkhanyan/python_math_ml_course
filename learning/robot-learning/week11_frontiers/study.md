# Week 11 study guide: Frontier & Open Problems

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

This lecture is a map of open problems, so it leans on everything before it: imitation versus RL (weeks 3 to 5), backbones and data recipes (weeks 7 to 9, and the guest talks by Xiao, Reed and Vuong), and reasoning (week 10). The slides call it "Lecture 12"; it is week 11. The last third is advice on doing research.

### Extra comments

**How to read a robot demo.** The opening slides poke at "robotics is solved" posts. Questions worth asking of any video: Is it autonomous or teleoperated (week 1's PR1)? Is it sped up (look for "4x" in the corner)? Was this scene, these objects, this task in the training data? How many takes did it need, and what is the success rate over many trials? Generalisation comes in levels (one object, one room, one building, the world, slide 4), and most demos sit at the first two. ([02:00](https://youtu.be/eL4lcy1KNzE?t=120), [05:02](https://youtu.be/eL4lcy1KNzE?t=302), [slides 1](slides.md#slide-1) to [4](slides.md#slide-4))

**The scalability-alignment trade-off, in numbers.** Teleoperation gives data in exactly the robot's body and sensors, but in the thousands of hours. Data wearables (UMI-style grippers, gloves) reach about $10^5$ hours, and egocentric human video about $10^7$, while matching the robot less and less. Each step to the left gains roughly two orders of magnitude of data and loses some of its direct usefulness; Danfei Xu (week 3) and Cheng Chi (week 6) are both working on narrowing that gap. ([15:01](https://youtu.be/eL4lcy1KNzE?t=901), [slide 11](slides.md#slide-11))

**Why dexterity and generalisation pull apart.** Dexterity needs fast control loops and senses beyond vision (touch, force); generalisation comes from large foundation models, which are slow, and from mobility, which gives the robot varied data. A big model and a 50 Hz control loop are hard to combine, which is the motivation for the speed tricks in weeks 7, 9 and 10 (chunking, FAST, dropping reasoning at inference). ([17:01](https://youtu.be/eL4lcy1KNzE?t=1021), [slide 12](slides.md#slide-12))

**The long tail of sensing.** Touch data is scarce; data with several senses recorded at once (camera, depth, touch, force) barely exists; and every robot's sensors differ. FuSe's answer is to use language as the common ground: fine-tune a pre-trained generalist with losses that tie touch and audio to language, so instructions like "pick the object that feels squishy" become possible without huge touch datasets. The lecturer's 2019 video of a robot crushing a mug is the motivation in one clip. ([20:01](https://youtu.be/eL4lcy1KNzE?t=1201), [slides 15](slides.md#slide-15) and [17](slides.md#slide-17))

**Introspection is the problem under several weeks' worth of slides.** "When should the human take over?" (week 3, slide 14), "is this instruction ambiguous?" (week 3, slide 16), "should I think harder?" (week 10) and "should I stop and ask?" (slide 19) all need the model to know when it does not know. A network's own confidence is not a reliable signal out of distribution: it can be confidently wrong, and the ways a situation can be new (objects, scene, body, instruction, task) are many. ([27:01](https://youtu.be/eL4lcy1KNzE?t=1621), [slides 19](slides.md#slide-19) and [20](slides.md#slide-20))

**What "stitching" means.** Imitation reproduces paths that appear in the data. Value-based offline RL can do better: if one trajectory passes through a state and another trajectory continues from that same state to the goal, the learned values connect the two, and the policy can combine pieces into a path no single demonstration contained, or take a shortcut the data only hints at. That is the 1 to 3 arrow on slide 22, and the reason the field keeps returning to RL despite its difficulty. ([29:01](https://youtu.be/eL4lcy1KNzE?t=1741), [slides 21](slides.md#slide-21) and [22](slides.md#slide-22))

**Why autonomous data is hard to use.** A robot's own rollouts are cheap and show its real failure modes, but they visit states demonstrations never do and look different in time (jerkier, longer, different rates). Mixing them naively with expert data teaches the policy the jerky, suboptimal behaviour too. $\pi_{0.7}$'s metadata conditioning in week 9 is one practical answer: label each episode's quality and condition on it. ([32:00](https://youtu.be/eL4lcy1KNzE?t=1920), [slide 24](slides.md#slide-24))

**Lifelong learning versus forgetting.** Today's deployed policies are frozen and slowly degrade as the world drifts away from their training data (the dashed line on slide 25). Fine-tuning on new data improves the new tasks but risks catastrophic forgetting of old ones, the same worry Reed raised about fine-tuning VLMs into VLAs. In-context adaptation (a correction in words, a demonstration in the prompt) would avoid retraining altogether, if it can be made reliable. ([34:02](https://youtu.be/eL4lcy1KNzE?t=2042), [slides 25](slides.md#slide-25) and [26](slides.md#slide-26))

**Two first steps from the lecturer's own work.** SOAR lets robots improve without humans: a VLM proposes tasks that fit the scene, the robot tries them, and another VLM judges success, giving a self-labelling data loop (week 4's QT-Opt image-subtraction reward, generalised). PALO adapts to a new task by searching over language decompositions of the task with a VLM, needing about 5 demonstrations instead of more than 100 for fine-tuning in parameter space. ([42:00](https://youtu.be/eL4lcy1KNzE?t=2520), [slides 30](slides.md#slide-30) and [31](slides.md#slide-31))

**The research advice, condensed.** Most research is incremental and most ideas never become papers, so pick problems you are excited about. Be your own "reviewer 2": name the most likely reason the idea fails before investing months. Problem-driven research is safer for a PhD than method-driven. Debug by starting with something that must work (overfit a tiny dataset first). Simple ideas win because they scale. (The slide dates ALVINN to 1986; it appeared at NIPS 1988, see the transcriber note on slide 35.) ([44:00](https://youtu.be/eL4lcy1KNzE?t=2640), [52:00](https://youtu.be/eL4lcy1KNzE?t=3120), [slides 35](slides.md#slide-35) to [40](slides.md#slide-40))

### Going further

- Levine et al., [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643), for stitching and why offline RL is hard.
- Zhou et al., [Autonomous Improvement of Instruction Following Skills via Foundation Models](https://arxiv.org/abs/2407.20635) (SOAR).
- Jones et al., [Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding](https://arxiv.org/abs/2501.04693) (FuSe).
- PALO (Myers, Zheng, Mees et al., CoRL 2024) is cited on slide 31; search arXiv for "Policy Adaptation via Language Optimization".

## Lecture quiz

### Q1

What trade-off does the plot of data scalability against hardware alignment show?

- A) Teleoperation is the most scalable data source.
- B) Egocentric human video matches the robot best.
- C) The more scalable a data source (egocentric video, about $10^7$ hours), the less it matches the robot's body and sensors; teleoperation matches best but reaches only about $10^3$ hours.
- D) There is no trade-off.

<details><summary>Answer</summary>

**C.** The three sources fall on one line: each step toward more data is a step away from the robot. Wearables sit in between. ([slide 11](slides.md#slide-11), [15:01](https://youtu.be/eL4lcy1KNzE?t=901))

</details>

### Q2

Why do dexterity and generalisation require different ingredients?

- A) Dexterity needs fast control and senses such as touch and force, while generalisation leans on large foundation models (which are slow) and on mobility for varied data.
- B) Dexterity needs less data than generalisation.
- C) Generalisation requires touch sensing.
- D) They do not; one model solves both.

<details><summary>Answer</summary>

**A.** Slide 12's Venn diagram: high-frequency control and multimodal sensing on one side, foundation models and mobility on the other. The goal is the overlap. ([slide 12](slides.md#slide-12), [17:01](https://youtu.be/eL4lcy1KNzE?t=1021))

</details>

### Q3

Which of these is NOT one of the three problems on the "long tail of robot sensing" slide?

- A) Scarcity: modalities like touch are not available at scale.
- B) Pairing: data with several senses recorded together barely exists.
- C) Heterogeneity: sensors differ across robots.
- D) Tactile signals cannot be stored digitally.

<details><summary>Answer</summary>

**D.** Touch can be recorded fine; there is just little of it, rarely paired with other senses, and every robot's sensors differ. ([slide 15](slides.md#slide-15), [22:00](https://youtu.be/eL4lcy1KNzE?t=1320))

</details>

### Q4

On the "when and how to reason" slide, what should a robot do when a task is truly out of distribution (a new goal or a failure it cannot handle)?

- A) Think harder with more reasoning tokens.
- B) Escalate: pause and ask for help or defer.
- C) Keep running the reactive policy.
- D) Act randomly to explore.

<details><summary>Answer</summary>

**B.** More thinking helps near the training distribution; when the situation is unknown, the safest move is to stop and ask. That only works if the robot can tell which case it is in, which is the introspection problem. ([slide 19](slides.md#slide-19), [26:02](https://youtu.be/eL4lcy1KNzE?t=1562))

</details>

### Q5

Why is introspection (knowing what it does not know) an open problem for robot policies?

- A) A policy can be confidently wrong out of distribution, so its own confidence is not a reliable signal, and situations can be new along many axes (objects, scenes, bodies, instructions, tasks).
- B) Robots have too little compute.
- C) Language models cannot express uncertainty.
- D) It is solved already.

<details><summary>Answer</summary>

**A.** The lecture points out that this breaks a basic assumption of current machine learning models; it is the hidden requirement behind deciding when to intervene, ask, or think longer. ([slide 20](slides.md#slide-20), [27:01](https://youtu.be/eL4lcy1KNzE?t=1621))

</details>

### Q6

The dataset contains only the path 1 to 2 to 3. What can offline RL do that imitation cannot?

- A) Copy the path more precisely.
- B) Nothing different.
- C) Learn a better policy than the data shows, for example going from 1 to 3 directly, because value estimates reveal what is better; imitation is bounded by the demonstrated behaviour.
- D) Learn without any data.

<details><summary>Answer</summary>

**C.** Imitation reproduces the demonstrations; RL optimises a reward and can stitch pieces of suboptimal trajectories into a better one. ([slide 22](slides.md#slide-22), [30:01](https://youtu.be/eL4lcy1KNzE?t=1801))

</details>

### Q7

What goes wrong if you naively mix expert demonstrations and a robot's autonomous rollouts for imitation?

- A) Nothing; more data is always better.
- B) The two differ in which states they visit and in their timing (frequency, length, smoothness), so the policy also imitates jerky, suboptimal behaviour.
- C) There is too much data to store.
- D) Autonomous rollouts contain no images.

<details><summary>Answer</summary>

**B.** Slide 24's two mismatches. They call for weighting, alignment or conditioning (for example on episode quality) rather than plain pooling. ([slide 24](slides.md#slide-24), [33:00](https://youtu.be/eL4lcy1KNzE?t=1980))

</details>

### Q8

In the autonomous-improvement system on slide 30 (SOAR), what do the foundation models do?

- A) They control the robot directly.
- B) They collect human demonstrations.
- C) They replace the robot policy.
- D) They propose tasks that fit the scene and judge whether attempts succeeded, so robots collect and label their own data without humans.

<details><summary>Answer</summary>

**D.** Task proposal and reward detection are the two jobs that normally need a person. Automating them closes the data flywheel loop on slide 23. ([slide 30](slides.md#slide-30), [42:00](https://youtu.be/eL4lcy1KNzE?t=2520))

</details>

### Q9

Why does the lecture advise starting a project with something that should definitely work?

- A) A known-good starting point (for example overfitting a tiny dataset) separates bugs from bad ideas; then you make the problem harder step by step.
- B) It gets papers published faster.
- C) Reviewers prefer simple experiments.
- D) It saves GPU hours.

<details><summary>Answer</summary>

**A.** If your setup cannot even overfit, the problem is in the code, not the idea. Debugging is otherwise guesswork. ([slide 40](slides.md#slide-40), [52:00](https://youtu.be/eL4lcy1KNzE?t=3120))

</details>

## Guest talk

### Context

Lucas Beyer studied mechanical engineering and did his PhD in robot perception and computer vision at RWTH Aachen, then joined Google Brain in Zurich, where he co-created the Vision Transformer, SigLIP and PaliGemma (the VLM inside $\pi_0$ and many other VLAs); he is now at Meta. He calls it "the least robotics lecture" of the course: the story of how vision went from barely working to a reliable recipe, which is the part that comes before a VLM becomes a VLA.

### Extra comments

**How to measure "general" vision.** His team's benchmark, VTAB, samples many diverse tasks, gives a few examples of each, and scores how well a representation adapts. The goal he describes is what people do easily: see five examples of a new category and recognise the next one. ([07:01](https://youtu.be/0XB7fNS_ONg?t=421), [08:00](https://youtu.be/0XB7fNS_ONg?t=480))

**Self-supervised pre-training disappointed.** Across many self-supervised methods, none was consistently best, and performance on the pre-training task barely predicted downstream performance; only making models bigger reliably helped. Adding just a few labelled examples into pre-training changed that, which pushed the team to large-scale supervised pre-training. He says robotics has the same problem: a pre-training objective that does not predict downstream success. ([10:00](https://youtu.be/0XB7fNS_ONg?t=600), [12:02](https://youtu.be/0XB7fNS_ONg?t=722))

**Why the field took so long to scale.** Training ten times more data with the old recipe gave worse results, and people concluded a million images were enough. The fix was scaling model size, data and training time together; scale only one and you see almost nothing. Two practical lessons: a curve that looks flat for 8 GPU-weeks can climb steadily over 8 GPU-months, and hyperparameters (his weight-decay example) can rank runs differently early and late, so killing runs early picks the wrong one. ([17:01](https://youtu.be/0XB7fNS_ONg?t=1021), [19:01](https://youtu.be/0XB7fNS_ONg?t=1141))

**Classes limit what you can learn; captions do not.** Classic datasets start from a list of class names, so they only cover what someone thought of (his anecdote about how a famous dataset's 80 categories were chosen makes the point). Web images come with free text, so CLIP-style training can learn any concept that appears in captions. SigLIP simplifies CLIP's loss: instead of a softmax over the whole batch (week 7), each image-text pair gets its own yes/no sigmoid. ([21:00](https://youtu.be/0XB7fNS_ONg?t=1260), [24:00](https://youtu.be/0XB7fNS_ONg?t=1440))

**Filtering to English throws away the world.** Filtering web data toward English and ImageNet-like images raises ImageNet scores but loses landmarks, objects and customs from most of the planet (his toilet thought experiment). His team's "No Filter" study showed the cost and argued for training on the unfiltered data. ([25:00](https://youtu.be/0XB7fNS_ONg?t=1500), [28:01](https://youtu.be/0XB7fNS_ONg?t=1681))

**Why contrastive models are "bags of words", and the fix.** To match "a cat sitting to the left of a dog" against other captions in a batch, it is usually enough to recognise "cat" and "dog"; learning "left of" would need a near-identical counterexample in the same batch, which random web data almost never provides. Training to write the caption word by word forces the model to get "left" right, which is why SigLIP 2 adds a captioning loss. ([30:01](https://youtu.be/0XB7fNS_ONg?t=1801), [31:01](https://youtu.be/0XB7fNS_ONg?t=1861))

**His one lesson: look at the data.** A benchmark meant to test word order asked models to choose between "the horse is eating the grass" and "the grass is eating the horse"; a captioner that never saw the image scored very well, because the text alone gives the answer. He says broken benchmarks like this reappear about every five years. Check what a benchmark actually tests before trusting its numbers, for LIBERO or any robot benchmark too. ([32:01](https://youtu.be/0XB7fNS_ONg?t=1921), [33:01](https://youtu.be/0XB7fNS_ONg?t=1981))

**PaliGemma's mid-training is a skills curriculum.** A SigLIP image encoder feeds image tokens into a Gemma language model, and the mid-training mixture is chosen to teach skills rather than be useful by itself: captioning, reading text in images, object questions, detection written as text (coordinates binned into 1024 values, like week 7's action bins), and segmentation masks as VQ-VAE tokens (week 6). Resolution grows during mid-training because high-resolution pre-training is too expensive. In the Q&A he stresses not freezing the image encoder, which lets the model recover from CLIP-style weaknesses. ([36:00](https://youtu.be/0XB7fNS_ONg?t=2160), [53:03](https://youtu.be/0XB7fNS_ONg?t=3183))

**"Perception is solved" means the recipe is.** A paper claimed VLMs are "blind" because they fail tasks like counting line intersections. He generated 128 examples, fine-tuned PaliGemma, and got about 95% accuracy: the model could see; it just had not been told what the task was. His recipe: collect a couple hundred examples in an afternoon, fine-tune a pre-trained VLM (tune the learning rate), RL-tune if needed, then distil into a smaller model if latency matters. What remains open is beyond perception: reasoning and planning, the subject of week 10. ([42:01](https://youtu.be/0XB7fNS_ONg?t=2521), [44:01](https://youtu.be/0XB7fNS_ONg?t=2641), [01:02:00](https://youtu.be/0XB7fNS_ONg?t=3720))

**Practical Q&A answers.** Data augmentation is worth at most about ten times more data, after expensive tuning, so get real data if you can. Remove near-duplicates of your evaluation data from the pre-training set (about 10 million per billion images in their experience) or your results are not measuring generalisation. And distillation is how a huge, well-trained model becomes one that runs on a robot. ([50:01](https://youtu.be/0XB7fNS_ONg?t=3001), [51:00](https://youtu.be/0XB7fNS_ONg?t=3060))

### Going further

- Zhai et al., [A Large-scale Study of Representation Learning with the Visual Task Adaptation Benchmark](https://arxiv.org/abs/1910.04867) (VTAB), and Kolesnikov et al., [Big Transfer (BiT)](https://arxiv.org/abs/1912.11370).
- Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) (SigLIP), and Tschannen et al., [Image Captioners Are Scalable Vision Learners Too](https://arxiv.org/abs/2306.07915).
- Yuksekgonul et al., [When and why vision-language models behave like bags-of-words](https://arxiv.org/abs/2210.01936), and Pouget et al., [No Filter](https://arxiv.org/abs/2405.13777).
- Beyer et al., [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726), whose fine-tuning study he recommends for choosing hyperparameters.
- Steiner et al., [How to train your ViT?](https://arxiv.org/abs/2106.10270), for the augmentation-versus-data trade-off.

## Guest talk quiz

### Q1

What did Beyer's team find when comparing many self-supervised pre-training methods?

- A) One method was clearly the best.
- B) None was consistently best, and pre-training-task performance barely predicted downstream performance; only making models larger reliably helped.
- C) Self-supervised beat supervised pre-training.
- D) Larger models did worse.

<details><summary>Answer</summary>

**B.** The lack of correlation between pre-training and downstream performance is the part he says robotics shares. ([10:00](https://youtu.be/0XB7fNS_ONg?t=600), [11:01](https://youtu.be/0XB7fNS_ONg?t=661))

</details>

### Q2

Training on ten times more data with the old recipe gave worse results. What was missing?

- A) More GPUs.
- B) Stronger data augmentation.
- C) Better labels.
- D) Scaling the model size and the training length along with the data: scaling all three together pays off, scaling one shows almost nothing.

<details><summary>Answer</summary>

**D.** Bigger data needs a bigger model and much longer training to show its benefit, which is why many people concluded too early that more data did not help. ([17:01](https://youtu.be/0XB7fNS_ONg?t=1021), [18:00](https://youtu.be/0XB7fNS_ONg?t=1080))

</details>

### Q3

Why do CLIP-style models struggle with relations like "a cat to the left of a dog"?

- A) The contrastive task can usually be solved by recognising "cat" and "dog" alone; learning "left of" would need a near-identical counterexample in the same batch, which is rare. Training to generate captions word by word forces the relation to be learned.
- B) The images are too small.
- C) They have no text encoder.
- D) Because of the sigmoid loss.

<details><summary>Answer</summary>

**A.** The training task offers a shortcut, and models take it. Captioning removes the shortcut, which is why later encoders such as SigLIP 2 add a captioning loss. ([31:01](https://youtu.be/0XB7fNS_ONg?t=1861), [32:01](https://youtu.be/0XB7fNS_ONg?t=1921))

</details>

### Q4

What is the lesson of the "horse eating grass" benchmark?

- A) Captioning pre-training fails on relations.
- B) VLMs are blind.
- C) Look at a benchmark's data before trusting it: a captioner that never sees the image scored very well, because the text alone gives the answer away.
- D) Use bigger batches.

<details><summary>Answer</summary>

**C.** He calls it the one thing to take away from the talk: broken benchmarks recur, and only looking at the examples catches them. ([33:01](https://youtu.be/0XB7fNS_ONg?t=1981))

</details>

### Q5

Why does Beyer argue against filtering web pre-training data to English text and ImageNet-like images?

- A) Filtering is too slow.
- B) It raises scores on ImageNet-like benchmarks but throws away concepts from most of the world (landmarks, everyday objects), hurting how well the model works globally.
- C) For legal reasons.
- D) English captions are noisier.

<details><summary>Answer</summary>

**B.** The filtered model mislabels landmarks outside North America in his examples, and a benchmark built for global coverage gets worse as ImageNet scores rise. ([27:01](https://youtu.be/0XB7fNS_ONg?t=1621), [29:01](https://youtu.be/0XB7fNS_ONg?t=1741))

</details>

### Q6

What does Beyer recommend when a VLM cannot do a new perception task out of the box?

- A) Keep rephrasing the prompt until it works.
- B) Train a model from scratch.
- C) Use RL only.
- D) Collect a couple hundred examples, fine-tune a pre-trained VLM (tuning the learning rate), add RL tuning if needed, and distil into a smaller model if latency matters.

<details><summary>Answer</summary>

**D.** His intersection-counting example went from "VLMs are blind" to about 95% accuracy with 128 examples: the skill was there, the task was not understood. ([44:01](https://youtu.be/0XB7fNS_ONg?t=2641), [45:01](https://youtu.be/0XB7fNS_ONg?t=2701))

</details>
