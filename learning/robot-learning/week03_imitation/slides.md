# Lecture 3: Imitation Learning - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 02.03.2026
- Source: `slides.pdf` in this folder (64 frames grabbed from the lecture recording, 1280x720, no text layer), from github.com/idanL1212000/RobotLerning-2026-ETH-Zurich `lectures/week03_slides.pdf`
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://youtu.be/Ef4R5s1LqoQ (transcript: `transcript_lecture.md`; guest talk: `transcript_guest_danfei_xu.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 3: Imitation Learning

Oier Mees - ETH zurich, Microsoft - 02.03.2026

## Slide 1: Feedback -> Actions (frame 2)

- Moved from 3 papers to 2 papers per lecture
- Extra paper slots on Thursdays in IFW A 32.1
- HW 2 released, due on March 12th
- We can see when you use AI in autograder

## Title [shown again] (frame 3)

[Same title slide as frame 1.]

## Slide 2: Recap: Mapping Observations to Actions (frame 4)

[Figure: observation $o_t$ (snowy road through a car windscreen) -> convolutional network -> action distribution $a_t$ (Left 0.0, Right 0.0, Straight 0.8, Backward 0.2), with an arrow from $a_t$ back to $o_t$.]

- $\pi_\theta(a_t \mid o_t)$ - Policy - Partially Observable
- $\pi_\theta(a_t \mid s_t)$ - Policy - Fully Observable
- $a_t$ - action: the decision taken by the agent at time $t$
- $o_t$ - observation: what the agent observes at time $t$
- $s_t$ - state: the "state" of the world at time $t$
- $\tau$ trajectory: sequence of state and actions $(s_1, a_1, \dots, s_T, a_T)$
- $r(s, a)$ reward: how good is $s, a$?

## Slide 3: Imitation Learning (frame 5)

- **Data**: Given trajectories collected by an "expert" "demonstrations" $\mathcal{D} = \{(s_1, a_1, \dots, s_t)\}$ [Figure: a Dataset cylinder holding $(s_1, a_1, \dots, s_t)$]
- **Goal**: Learn a policy $\pi_\theta$ that "imitates" the expert's behavior

[Figure: road image -> **Policy** -> the same road image with a forked arrow (choose a direction).]

## Slide 4: Behavioral Cloning (frame 6)

1. Given expert demonstrations: $\mathcal{D} = \{(s_1, a_1, \dots, s_t)\}$
2. For deterministic policy, regress to expert's actions: $\min_\theta \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{s}, \mathbf{a}) \in \mathcal{D}} \|\mathbf{a} - \hat{\mathbf{a}}\|^2$ where $\hat{\mathbf{a}} = \pi_\theta(\mathbf{s})$
3. Deploy policy $\pi_\theta$ on robot

## Slide 10 [shown early]: Learning to Fly in Swiss Forests (frame 7)

- Same camera trick

[Figure: video frame of a quadcopter flying along a forest trail, with the title card "Quadcopter navigation in the forest - Trail following under the tree canopy".]

[Note: this frame shows slide 10 between slides 4 and 5; the lecturer skipped ahead briefly. Slide 10 appears again in order at frame 16.]

## Slide 5: What can go wrong? (frames 8-10)

**Supervised Learning**

[Figure: an ellipse $p(x)$ containing sample points $x_1, x_2, x_3$.]

Inputs are independently, identically distributed (**i.i.d.**) and **independent of predicted labels**!

**Behavioral Cloning**

[Figure: the expert's states $s_1, s_2, s_3, s_4$ along one path; the policy's predicted actions $\hat{a}_1, \hat{a}_2, \hat{a}_3$ each deviate slightly, so the visited states drift further from the expert path at every step.]

Small error in predicted action can lead to **drift away** from training data distribution!

$$p_{expert} \ne p_\pi(s)$$

- $p_{expert}$: States visited by expert
- $p_\pi(s)$: States visited by the policy

[Frame 9 is the earlier build showing only the expert path $s_1 \to s_4$; frames 8 and 10 show the full slide.]

## Slide 6: Does it work? (frames 11-13)

[Figure: embedded NVIDIA video of a self-driving car learning to steer: a parking lot lined with traffic cones (subtitle "It began its driving lessons in a parking lot"), then a road narrowed by cones and barriers in a construction zone (subtitle "even leaving the road to stay safe.").]

Reference: *End to End Learning for Self-Driving Cars*, Bojarski, et. al., 2016

[Note: slide 7 does not appear in the captured frames; the numbering jumps from 6 to 8.]

## Slide 8: Why did it work? (frame 14)

[Figure: top view of a two-lane road with a car drifted against the edge line, captioned:] What should the policy do here? Expert never "visited this state"!

## Slide 9: Why did it work? Data Augmentation! (frame 15)

- Add "fake" data that illustrates corrections with side-facing cameras

$$\mathcal{D}_{\text{aug}} = \{(\mathbf{o}_{\text{center}}, \mathbf{a}), (\mathbf{o}_{left}, \mathbf{a} + \delta), (\mathbf{o}_{right}, \mathbf{a} - \delta)\}$$

[Figure: top view of a car with three cameras on its front: the left-facing camera is paired with "Action Label: Steer Right", the centre camera with "Action Label: Go Straight", the right-facing camera with "Action Label: Steer Left".]

## Slide 10: Learning to Fly in Swiss Forests (frames 16-22)

- Same camera trick

[Figure: embedded explainer video playing across seven frames. Title card "Quadcopter navigation in the forest - Trail following under the tree canopy" (same as frame 7). "Main challenge: perceiving a real-world trail from images is extremely difficult". Six forest photos sorted into "trail heading left", "trail straight ahead", "trail heading right", asking "which direction is the trail heading to?". "How we solved the problem": trail image -> neural network -> steering wheel with left / straight / right arrows. "How the classifier works": an input image passing through convolutional feature maps. "Training the classifier": a cartoon hiker wearing a head-mounted camera.]

## Slide 11: Upper Bound of Behavioral Cloning (frames 23-24)

[Figure: the drifting-trajectory diagram from slide 5, with a mistake probability $\epsilon$ marked at each step and horizon $T$ at the end; a photo of a tightrope walker between two towers.]

- $T$: time horizon
- $\epsilon$: probability of $\pi_\theta$ making a mistake at any steps

$$c(\mathbf{s}, \mathbf{a}) = \begin{cases} 0 & \text{if } \mathbf{a} = \pi^\star(\mathbf{s}) \\ 1 & \text{otherwise} \end{cases}$$

$$P(\mathbf{a} \ne \pi^\star(\mathbf{s}) \mid \mathbf{s}) \le \epsilon, \quad \forall \mathbf{s} \in \mathcal{D}_{\text{train}}$$

[Second build (frame 24) adds:]

$$\mathbb{E}\left[\sum_t c(s_t, a_t)\right] \le \epsilon T + (1 - \epsilon)(\epsilon(T - 1) + \cdots) \approx \mathbf{O}(\epsilon T^2)$$

Distribution shift causes the error to grow **quadratically!**

More analysis: "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" by Ross et al., 2011.

## Slide 12: Addressing Compounding Error (frame 25)

Can we make?

$$p_{expert} = p_\pi(s)$$

- $p_{expert}$: States visited by expert
- $p_\pi(s)$: States visited by the policy

## Slide 13: Aggregating Corrective Behavior (frame 26)

**Dagger: Dataset Aggregation**

1. Roll out policy $\pi_\theta$ and collect states: $(s'_1, \hat{a}_1, \dots, s'_t)$
2. Query/label expert action at visited states $\mathbf{a}^* \sim \pi_{\text{expert}}(\cdot \mid \mathbf{s}')$
3. **Aggregate** corrections with existing data $\mathcal{D} \leftarrow \mathcal{D} \cup \{(\mathbf{s}', \mathbf{a}^*)\}$
4. Update policy $\theta \leftarrow \arg\min_\theta \mathcal{L}(\pi_\theta, \mathcal{D})$

[Figure: the drifting-trajectory diagram, now with blue arrows from each state the policy visited pointing back toward the expert path (the expert's corrective labels).]

- ✅ Algorithm will converge $p_\pi(s) = p_{expert}$
- ✅ Achieves $O(\epsilon T)$ instead of BC $O(\epsilon T^2)$
- ❌ Hindsight labeling & expert querying difficult

Paradox BC works better if the data has more mistakes and recoveries!

## Slide 14: Aggregating Online Interventions (frame 27)

**Human Gated Dagger**

1. Roll out policy $\pi_\theta$ and collect states: $(s'_1, \hat{a}_1, \dots, s'_t)$
2. Expert intervenes at time $t$ when policy makes mistake
3. Expert provides partial demo $(\mathbf{s}'_t, \mathbf{a}^*_t, \dots, \mathbf{s}'_T)$
4. **Aggregate** new demos $\mathcal{D} \leftarrow \mathcal{D} \cup \{(\mathbf{s}'_i, \mathbf{a}^*_i)\}; i \ge t$
5. Update policy $\pi_\theta \leftarrow \min_\theta \mathcal{L}(\pi_\theta, \mathcal{D})$

[Figure: the drifting-trajectory diagram where, after two policy steps, a blue expert arrow takes over and steers to $s_4$.]

- ✅ Lets the human take control!
- ❌ How to detect when an intervention is needed?

[Note: step 5 should read $\arg\min_\theta$, as in step 4 of slide 13.]

## Slide 15: Case Study: Language Ambiguities (frame 28)

[Figure: three panels of a two-armed PR2 robot at a table of household objects. Panel 1 dialogue: "Fetch the round yellow thing" / "Do you mean the lemon in the middle?" / "Yes", with boxes around the candidate yellow objects. Panel 2: "Place it left of the object on the bottom", with a heat map over the table showing where to place. Panel 3: the robot placing the object.]

Reference: *Composing Pick-and-Place Tasks By Grounding Language*, **Mees** and Burgard, ISER 2020

## Slide 16: Case Study: Language Ambiguities [method] (frames 29-31)

"Pick up the blue stuff"

[Figure: camera image of a table with a black bowl, a blue mug, a blue bowl, a red cup, a can in a bowl and a toy car; the blue mug and black bowl have blue boxes, the blue bowl a green box. A robot icon asks: "Do you mean this blue bowl on the table?"]

1. Train referential expression comprehension model with ranking loss

$$\mathcal{L}_{\text{rank}} = \sum_i \Big[\lambda_1 \max\big(0, m_1 + S(o_i \mid r_j) - S(o_i \mid r_i)\big) + \lambda_2 \max\big(0, m_1 + S(o_k \mid r_i) - S(o_i \mid r_i)\big)\Big]$$

2. Train referential expression generation model

$$\mathcal{L}_{\text{gen}} = -\sum_i \log P(r_i \mid v_i)$$

$$\mathcal{L}_{\text{mmi}} = \sum_i \big[\lambda_3 \max\big(0, m_2 + \log P(r_i \mid v_k) - \log P(r_i \mid v_i)\big)\big]$$

3. If at test time there are multiple objects within margin $m$, generate descriptions and ask user

[The three numbered steps appear one per build.]

## Slide 17: Case Study: Language Ambiguities [demo] (frames 32-38)

[Figure: embedded video (2x speed) playing across seven frames: the PR2 robot at two small tables of household objects follows spoken commands shown as subtitles: "Fetch the red thing" (picks up a red bowl), "Place it behind the colored dish" (places it), then for an ambiguous request the robot asks "I am not sure, do you mean the middle banana?", picks the banana, and gets "Place it on top of the bottom object".]

Reference: *Composing Pick-and-Place Tasks By Grounding Language*, **Mees** and Burgard, ISER 2020

## Slide 18: Why might we still fail to mimick the expert? (frame 39)

**Non-Markovian behavior**

- $\pi_\theta(a_t \mid o_t)$ - arrow: action depends only on current observation
- $\pi_\theta(a_t \mid o_1, \dots, o_t)$ - arrow: Human behavior might be affected by past observations, emotions, privileged information etc.

[Note: slide 19 does not appear in the captured frames; the numbering jumps from 18 to 20.]

## Slide 20: Adding History to Policy (frame 40)

- Encode with some sequence to sequence architecture

[Figure: a video frame of a robot arm at a desk with drawers, captioned "1. 'Open the drawer'" (2x) -> **Seq2Seq Model** -> robot arm icon.]

❌ Adding history to the policy **doesn't always make things better**!

## Slide 21: Causal Confusion (frames 41-42)

- Policy might infer **spurious correlations**!

[Figure: video frame of the robot arm at the drawer, "1. 'Open the drawer'" (2x).]

1. Data: every time the robot opens the drawer, the gripper grasps the handle with 10 N
2. Model learns: if history shows a gripper force spike of 10 N, pull to open
3. Test time: gripper slips and only reads 2 N, the pull command is never triggered

[Second build (frame 42) adds:] ❌ Policy thinks history of gripper sensor causes open drawer instead of the visual state!

More analysis: "Causal Confusion in Imitation Learning" by de Haan et al., 2019.

## Slide 22: Why might we still fail to mimick the expert? [multimodality] (frame 43)

**Multimodal behavior**

- **Stochasticity:** Many (**infinite**) ways to close a drawer with a 7-DoF robot!
- **Expert Inconsistency**: Humans operators might use different '**modes**' across trials

[Figure: three photos of the robot arm closing the same drawer with clearly different gripper poses.]

## Slide 23: Multimodal Behavior (frame 44)

- Deterministic MSE BC policy will fail due to "**averaging**" of **modes**!

[Figure: two photos of a snowy slope with a lone tree. Left: ski tracks split and pass the tree on the left and on the right. Right: a skier who took the average of the two tracks and went straight into the tree.]

## Slide 24: Mixture of Gaussian Distributions (frame 45)

- More expressive than a single gaussian
- Need to predefine number of gaussians

[Figure: left, two Gaussian components (red and blue) and their bimodal sum (black outline), labelled $\mu_1, \mu_2, \Sigma_1, \Sigma_2$; right, a bar chart of the mixture weights $w_1 = 0.65$ and $w_2 = 0.35$.]

$$\pi(\mathbf{a} \mid \mathbf{o}) = \sum_i w_i \mathcal{N}(\mu_i, \Sigma_i)$$

## Slide 25: Autoregressive Discretization (frame 46)

- Discretization great for representing multimodal distributions
- Impractical to scale to higher dimensions (exponential)
- Solution: **per-dimension discretization with sequence model**

[Figure: the ski-slope image as input tokens -> **Autoregressive Transformer** -> output tokens $a_{t,0}, a_{t,1}, a_{t,2}$, one per action dimension.]

$$p(\mathbf{a_t} \mid \mathbf{s_t}) = p(a_{t,0}, a_{t,1}, a_{t,2} \mid \mathbf{s_t}) = p(a_{t,2} \mid \mathbf{s_t}, a_{t,0}, a_{t,1})\, p(a_{t,1} \mid \mathbf{s_t}, a_{t,0})\, p(a_{t,0} \mid \mathbf{s_t})$$

## Slide 26: Diffusion (frame 47)

- Models complex distributions over continuous variables
- We can replace images with robot actions, more on W6

[Figure: a cat photo $x_0$ getting noisier left to right until it is pure noise $x_T$. Top arrow, **Forward Process**: $\mathbf{x_{i+1}} = \mathbf{x_i} + \text{noise}$. Bottom arrow, right to left, **Generative Backward Process**.]

learn $f(x_i) = x_{i-1}$ - in practice $f(\mathbf{x_i}) = \text{noise}$, $\mathbf{x_{i-1}} = \mathbf{x_i} - f(\mathbf{x_i})$

## Slide 27: Latent Variable Models (frame 48)

- Output still gaussian, but receives additional input
- Can represent "any" distribution
- Popular: conditional VAEs

$$\pi(\mathbf{a} \mid \mathbf{o}, z) = f_\theta(\mathbf{o}, z)$$

- [arrow at $f_\theta$] Conditional Decoder

[Figure: the ski slope with two tracks around the tree, the left track labelled $z_{\text{left}} \sim \mathcal{N}(0, \mathbf{I})$ and the right $z_{\text{right}} \sim \mathcal{N}(0, \mathbf{I})$: the latent picks the mode.]

## Slide 28: Learning Multiple Tasks (frame 49)

Latent Variable Models are different from task conditioning

| "one task" | "multiple tasks" |
|---|---|
| $\pi_\theta(a_t \mid s_t)$ | $\pi_\theta(a_t \mid s_t, \text{task id})$ |
| Single task Behavior Cloning (Pomerleau 1991) | Task conditioned Behavior Cloning (Rahmatizadeh 2018) |

[Figure: a robot tidying toy blocks in a living room.]

## Slide 29: Learning Multiple Tasks [success?] (frames 50-51)

"Move the sliding door to the left" - $\pi_\theta(a_t \mid s_t, \text{task id})$

[Figure: embedded video of a robot arm at a desk with a sliding door, a drawer and buttons, moving the sliding door.]

Did it succeed? Where is the threshold, 50%?

## Slide 30: How to Scale Control to "Any" Tasks? (frame 52)

**Tasks** are often **continuous**, not discrete

| "one task" | "multiple tasks" | "any tasks" |
|---|---|---|
| $\pi_\theta(a_t \mid s_t)$ | $\pi_\theta(a_t \mid s_t, \text{task id})$ | $\pi_\theta(a_t \mid s_t, s_g)$ |
| Single task Behavior Cloning (Pomerleau 1991) | Task conditioned Behavior Cloning (Rahmatizadeh 2018) | Goal state conditioned Behavior Cloning (Lynch 2019) |

[Figure: under "any tasks", a "current" and a "goal" image of a simulated desk, with a latent space in between where a dotted arrow goes from $s_t$ to $s_g$.]

## Slide 31: [untitled: transition question] (frame 53)

**Let's put everything together!**

**How can we learn any task with a latent variable model that handles multimodality?**

## Slide 32: Case Study: Play Data (frame 54)

- No upfront tasks
- Scalable, reset-free data collection
- Rich and highly multimodal behaviors

[Figure: video of a person teleoperating a robot arm at a desk full of objects ("play").]

[Note: slide 33 does not appear in the captured frames; the numbering jumps from 32 to 34.]

## Slide 34: Case Study: Learning Policies Play Data (frames 55-56)

[Figure: first builds of the architecture diagram completed on slides 36-37: a **Sequence** of observation boxes taken from a play video; its last frame becomes the **Goal state**, which a **Goal encoder** turns into a goal feature vector; at the bottom, the initial state box next to the goal feature.]

[Note: slide 35 does not appear in the captured frames; the numbering jumps from 34 to 36.]

## Slide 36: Case Study: Learning Policies from Play Data [plan prior and posterior] (frame 57)

[Figure: architecture diagram.]

- **Sequence** (whole play window) -> **Posterior (Plan Recognition)** -> a Gaussian over latent plans
- **Goal state** -> **Goal encoder** -> **Goal features**
- **Initial state** + goal features -> **Prior (Plan Proposal)** -> a Gaussian over latent plans
- **KL loss** between the posterior and the prior Gaussians

## Slide 37: Case Study: Learning Policies from Play Data [action decoder] (frame 58)

[Figure: the slide 36 diagram completed: arrows show the goal state is the last frame of the sequence and the initial state the first. A latent plan is sampled from the posterior ("Sample latent plan"); the **Action decoder** takes the latent plan, the goal features and the **Current state** and outputs an **Action likelihood**.]

## Slide 38: Case Study: Learning Policies from Play Data [language goals] (frame 59)

[Figure: an "Unstructured data" cylinder that is almost all **Visual**, with a thin **Language** slice labelled 1%. Two encoders feed one **Latent Goal**: a **Goal Image encoder** (a goal image of the robot desk) and a **Language encoder** (the instruction "Push the pink block to the right"). The latent goal, a **Latent Plan** and the **Current state** (a camera image) go into the **Action decoder**, which outputs an **Action likelihood**.]

## Slide 39: The Math Behind Goal-Conditioned Latent Variable Models (frame 60)

$p_\theta(\tau, z)$ - arrows: $\tau$ = trajectories; $z$ = latent plans -> when z continuous, marginalization becomes intractable

Optimize surrogate objective; variational lower bound of marginal log-likelihood

$$\log p_\theta(\tau) \ge \underbrace{-KL\big(q_\phi(z \mid \tau) \,\|\, p_\theta(z)\big)}_{\text{Regularization}} + \underbrace{\mathbb{E}_{q_\phi(z \mid \tau)}\left[\log p_\theta(\tau \mid z)\right]}_{\text{Reconstruction}}$$

- **Regularization**: are latent plan $z$ close to our prior distribution over $\tau$?
- **Reconstruction**: how well does latent plan $z$ explain the expert behavior $\tau$?

## Slide 40: The Math Behind Goal-Conditioned Latent Variable Models [conditioned] (frame 61)

[Figure: small copy of the slide 37 architecture diagram.]

$$\log p_\theta(\tau \mid c) \ge -KL\big(q_\phi(z \mid \tau, c) \,\|\, p_\theta(z \mid c)\big) + \mathbb{E}_{q_\phi(z \mid \tau, c)}\left[\log p_\theta(\tau \mid z, c)\right]$$

$$c \leftarrow (s_c, s_g)$$

- [arrows] $q_\phi(z \mid \tau, c)$ = **Posterior - Plan Recognition**; $p_\theta(z \mid c)$ = **Prior - Plan Proposal**; $p_\theta(\tau \mid z, c)$ = **Action Decoder**

## Slide 41: My PhD: Natural Language + Learning from Play (frame 62)

[Figure: a thumbnail above each paper.]

- *CALVIN: A Benchmark for Language-conditioned Policy Learning for Long-horizon Robot Manipulation Tasks*, **Mees** et. al., RA-L 2022 - **Best Paper Award RA-L 2022**
- *What Matters in Language Conditioned Imitation Learning over Unstructured Data*, **Mees, et al.** RA-L 2022.
- *Grounding Language with Visual Affordances over Unstructured Data*, **Mees, et al.** ICRA 2023. - **Finalist Best Paper Award ICRA 2023**
- *Affordance Learning from Play for Sample-Efficient Policy Learning*, Borja\*, **Mees\*** et. al., ICRA 2022
- *Latent Plans for Task Agnostic Offline Reinforcement Learning*, Rosete\*, **Mees\*** et. al., CoRL 2022

## Slide 42: Conclusion: Behavioral Cloning Failures (frame 63)

| Distribution Shift | Non-Markovian Behavior | Multimodal Behavior |
|---|---|---|
| Dagger | Privileged Observation | Mixture of Gaussians |
| Human Gated Dagger -> Detect when to intervene | History & Causal Confusion | Autoregressive Discretization |
| | | Diffusion |
| | | Latent Variable Models |

## End (frame 64)

Thank you for your attention
