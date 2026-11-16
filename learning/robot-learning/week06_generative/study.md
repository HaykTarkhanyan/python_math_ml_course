# Week 06 study guide: Generative Models

- Written by Claude from this week's slides (`slides.md`), the transcripts and general background. It is not course material: check anything that matters against the lecture.
- Quiz answers are folded under "Answer". In the course browser the quizzes are clickable.
- `slide N` links point into `slides.md`; time links open the video at that moment.

## Lecture

### Before you watch

Week 3's multimodality problem (the skier and the tree) is the motivation, and week 5's reparameterisation trick comes back. You need Gaussians, the KL divergence, and the fact that the sum of independent Gaussians is Gaussian with the variances added. That last fact is the whole derivation on slide 23.

### Extra comments

**Three questions, three model families.** Slide 3 sorts the lecture neatly. A VAE answers "how do I encode data so that I can sample codes?", diffusion answers "how do I turn noise into data step by step?", and flow matching answers "what velocity field carries noise to data?". In robotics, the "data" is an action or a chunk of actions, conditioned on the observation. ([03:01](https://youtu.be/qd6Ldsuu46I?t=181), [slide 3](slides.md#slide-3))

**Why an autoencoder cannot generate.** Nothing tells the encoder where to put codes, so they end up in scattered clusters with empty space between them. A random code usually lands in a gap the decoder never saw. The VAE's KL term pulls every encoding toward $\mathcal{N}(0, I)$, which packs the clusters into one region you know how to sample from. Turn the KL weight up and the latent becomes easy to sample but reconstructions blur; turn it down and you are back to an autoencoder (that dial is the "$\beta$-VAE"). ([06:01](https://youtu.be/qd6Ldsuu46I?t=361), [slides 7](slides.md#slide-7) and [8](slides.md#slide-8))

**The two VAE failure modes, in robot terms.** Posterior collapse: a strong decoder (for example one that also sees the observation history) can predict actions well without using $z$, so $z$ is ignored and you have a unimodal policy again. Prior mismatch: if the "go left" and "go right" encodings sit in two blobs, a sample from $\mathcal{N}(0, I)$ can land between them and decode to "straight into the tree". ACT, the conditional VAE behind ALOHA (week 3), sidesteps the second at test time by simply setting $z$ to the prior's mean. ([14:05](https://youtu.be/qd6Ldsuu46I?t=845), [slide 12](slides.md#slide-12))

**The straight-through estimator, demystified.** $z_q = z_e + \text{sg}[z_q - z_e]$ is numerically just $z_q$ in the forward pass. In the backward pass, the stop-gradient term counts as a constant, so the gradient with respect to $z_e$ is exactly the gradient with respect to $z_q$. It is a deliberate lie that works because $z_q$ is the nearest code, so the two are close. The codebook loss moves codes toward encoder outputs; the commitment loss moves encoder outputs toward codes; each uses a stop-gradient so that the two do not chase each other. ([17:01](https://youtu.be/qd6Ldsuu46I?t=1021), [slides 14](slides.md#slide-14) and [15](slides.md#slide-15))

**FSQ and other discrete tokenisers.** Codebook collapse (most codes never used) is common enough that Finite Scalar Quantization replaces the learned codebook: project to a few dimensions and round each to a handful of levels. There is nothing to collapse, and the "codebook" is the grid. Discrete codes matter in robotics because a transformer can then predict actions or video as tokens; FAST (Pertsch et al., 2025) is a widely used action tokeniser that uses a frequency transform plus byte-pair encoding instead of a VQ-VAE. ([22:00](https://youtu.be/qd6Ldsuu46I?t=1320), [slide 17](slides.md#slide-17))

**Why the forward step uses $\sqrt{1 - \beta}$.** If $x_i$ has variance 1, then $x_{i+1} = \sqrt{1 - \beta}\, x_i + \sqrt{\beta}\, \epsilon$ has variance $(1 - \beta) + \beta = 1$. The signal shrinks while noise grows, but the total scale stays fixed, so after enough steps you land on exactly $\mathcal{N}(0, I)$, the distribution you will sample from. That is why actions and images are normalised to roughly unit scale before diffusion training. ([26:01](https://youtu.be/qd6Ldsuu46I?t=1561), [slide 20](slides.md#slide-20))

**Predicting noise is predicting the score.** The network's noise estimate relates to the gradient of the log-density of the noisy data: $\epsilon_\theta(x_i, i) \approx -\sqrt{1 - \bar\alpha_i}\, \nabla_{x} \log p(x_i)$. Removing predicted noise is a step uphill on probability, which is what slide 24's "score step plus Langevin noise" means. Noise is also a convenient target: at every step it is unit-scale Gaussian, so the regression problem looks the same at every noise level. ([27:02](https://youtu.be/qd6Ldsuu46I?t=1622), [slides 21](slides.md#slide-21) and [24](slides.md#slide-24))

**DDIM is a sampler, not a new model.** The same network, trained with the DDPM loss, can be sampled deterministically while skipping most steps. Diffusion Policy trained with 100 steps and sampled with about 10 DDIM steps, which brought one action chunk down to roughly a tenth of a second on a desktop GPU. Every denoising step is a full network call, so step count is control latency. ([34:00](https://youtu.be/qd6Ldsuu46I?t=2040), [slide 25](slides.md#slide-25))

**Classifier-free guidance, by the value of $w$.** $w = 0$ gives the unconditional model, $w = 1$ the plain conditional one, and $w > 1$ extrapolates past the conditional prediction, away from the unconditional one: samples follow the prompt more closely and vary less. One model does both because the condition is dropped (replaced by $\emptyset$) for a random fraction of training examples. ([37:00](https://youtu.be/qd6Ldsuu46I?t=2220), [slide 27](slides.md#slide-27))

**Why Diffusion Policy predicts chunks.** It predicts a sequence of future actions (16 steps in the paper's default) and executes only the first few (8) before predicting again. Inside one chunk the actions come from one sample, so the robot commits to one mode (left of the tree or right) instead of flickering between modes from one step to the next. Re-planning every few steps keeps it reactive. The observation is encoded once and injected by FiLM, so the $K$ denoising steps only rerun the small action network. ([40:01](https://youtu.be/qd6Ldsuu46I?t=2401), [slides 28](slides.md#slide-28) and [29](slides.md#slide-29))

**adaLN-Zero in one sentence.** Each transformer block's scale and shift come from the conditioning (diffusion step and observation embedding), and the final gate starts at zero, so at initialisation every block is the identity and the network grows into its depth instead of being unstable from the start. It comes from the image DiT paper (Peebles and Xie, 2022); slide 32 is the lecturer's robot version. ([43:00](https://youtu.be/qd6Ldsuu46I?t=2580), [slide 32](slides.md#slide-32))

**Flow matching, and a convention warning.** Draw a straight line from a noise sample to a data sample; the velocity along it is the constant $x_0 - \epsilon$; train a network to predict that velocity at random points on random lines. Sampling integrates the learned velocity with a few Euler steps. The averaged velocity field is not straight everywhere (lines cross), which is what rectified flow's "reflow" retraining fixes. Papers and codebases disagree on whether $t = 0$ is noise or data (the slide puts noise at $t = 0$; some implementations put it at $t = 1$, which flips the sign of the velocity). Check the convention before copying code. $\pi_0$ predicts chunks of 50 actions with 10 integration steps. ([44:01](https://youtu.be/qd6Ldsuu46I?t=2641), [slides 33](slides.md#slide-33) to [35](slides.md#slide-35))

### Going further

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), section 3 for the simplified loss, and Song et al., [DDIM](https://arxiv.org/abs/2010.02502).
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), and Liu et al., [Rectified Flow](https://arxiv.org/abs/2209.03003).
- Chi et al., [Diffusion Policy](https://arxiv.org/abs/2303.04137), the case study; read it before the guest talk.
- Dasari et al., [The Ingredients for Robotic Diffusion Transformers](https://arxiv.org/abs/2410.10088), the lecturer's DiT-block policy.
- Mentzer et al., [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505), and Pertsch et al., [FAST](https://arxiv.org/abs/2501.09747), for discrete codes and action tokens.

## Lecture quiz

### Q1

Why can you not generate new data by feeding random codes to a plain autoencoder's decoder?

- A) The decoder was never trained.
- B) Nothing constrains where the encoder puts codes, so random codes usually land in gaps the decoder never saw.
- C) The encoder is deterministic, so the decoder cannot run without it.
- D) The bottleneck is too large.

<details><summary>Answer</summary>

**B.** The only pressure on an autoencoder's latent space is reconstruction, which leaves codes in scattered clusters. A VAE adds a distribution you can sample from. The decoder runs fine on any input (C is wrong); it just produces garbage for codes it never saw. ([slide 7](slides.md#slide-7), [06:01](https://youtu.be/qd6Ldsuu46I?t=361))

</details>

### Q2

What does the KL term in the VAE objective do?

- A) It makes reconstructions sharper.
- B) It makes the encoder deterministic.
- C) It enlarges the bottleneck.
- D) It pulls each encoding distribution $q_\phi(z \mid x)$ toward the prior $\mathcal{N}(0, I)$, so the latent space is continuous and can be sampled.

<details><summary>Answer</summary>

**D.** That is the "regularisation" term on slide 8. If anything it makes reconstructions blurrier (A), since it fights against encoding every detail. ([slide 8](slides.md#slide-8), [07:01](https://youtu.be/qd6Ldsuu46I?t=421))

</details>

### Q3

In a VQ-VAE, the nearest-codebook lookup is not differentiable. What does the straight-through estimator do in the backward pass?

- A) Copies the gradient with respect to $z_q$ onto $z_e$, as if quantisation were the identity.
- B) Sets the encoder's gradient to zero.
- C) Sends the gradient only to the codebook.
- D) Differentiates the argmin exactly.

<details><summary>Answer</summary>

**A.** $z_q = z_e + \text{sg}[z_q - z_e]$ has the value of $z_q$ and the gradient of $z_e$. It is an approximation that works because the chosen code is the closest one. The codebook itself is trained by its own loss term, not by this gradient (C). ([slide 14](slides.md#slide-14), [18:00](https://youtu.be/qd6Ldsuu46I?t=1080))

</details>

### Q4

If $x_i$ has variance 1, what is the variance of $x_{i+1} = \sqrt{1 - \beta}\, x_i + \sqrt{\beta}\, \epsilon$, with $\epsilon \sim \mathcal{N}(0, 1)$ independent of $x_i$?

- A) $1 - \beta$
- B) $1 + \beta$
- C) 1
- D) $\beta$

<details><summary>Answer</summary>

**C.** Variances of independent terms add: $(1 - \beta) \cdot 1 + \beta \cdot 1 = 1$. The forward process is variance-preserving, so it ends exactly at the unit Gaussian you sample from at generation time. ([slide 20](slides.md#slide-20), [26:01](https://youtu.be/qd6Ldsuu46I?t=1561))

</details>

### Q5

Using $x_i = \sqrt{\bar\alpha_i}\, x_0 + \sqrt{1 - \bar\alpha_i}\, \bar\epsilon$ with $\bar\alpha_i = 0.25$, what is $x_i$?

- A) $0.5\, x_0 + 0.87\, \bar\epsilon$ (approximately)
- B) $0.25\, x_0 + 0.75\, \bar\epsilon$
- C) $0.75\, x_0 + 0.25\, \bar\epsilon$
- D) $0.5\, x_0 + 0.5\, \bar\epsilon$

<details><summary>Answer</summary>

**A.** $\sqrt{0.25} = 0.5$ and $\sqrt{0.75} \approx 0.87$. The coefficients are square roots, and their squares add to 1 (the variance-preserving property). B uses the variances instead of the standard deviations. ([slide 23](slides.md#slide-23), [32:00](https://youtu.be/qd6Ldsuu46I?t=1920))

</details>

### Q6

Why can DDPM training pick a random noise level $i$ for each example instead of running the forward chain step by step?

- A) It makes Adam converge faster.
- B) To avoid adding noise.
- C) Because the network does not depend on $i$.
- D) The closed form lets you jump from $x_0$ to any $x_i$ in one step, so each training example costs one noise draw and one network call.

<details><summary>Answer</summary>

**D.** That is the "teleporting" formula: the product $\bar\alpha_i$ collects all the steps. The network does depend on $i$ (C is wrong): the noise level is one of its inputs. ([slide 23](slides.md#slide-23), [31:00](https://youtu.be/qd6Ldsuu46I?t=1860))

</details>

### Q7

How does DDIM differ from DDPM?

- A) It uses a different training objective.
- B) It is a different sampler for the same trained model: deterministic, and able to skip steps (for example 20 instead of 1000).
- C) It needs a separate classifier.
- D) It adds more noise at every step.

<details><summary>Answer</summary>

**B.** The training is unchanged; only sampling changes. DDIM drops the noise term (D is backwards), which makes the path deterministic and lets it jump between distant noise levels. ([slide 25](slides.md#slide-25), [35:01](https://youtu.be/qd6Ldsuu46I?t=2101))

</details>

### Q8

In classifier-free guidance, what does a guidance scale $w > 1$ do?

- A) $w > 1$ gives the unconditional model.
- B) It ignores the condition.
- C) It extrapolates past the conditional prediction, away from the unconditional one: samples follow the condition more closely and vary less.
- D) It requires training a separate classifier.

<details><summary>Answer</summary>

**C.** $w = 0$ is unconditional and $w = 1$ is the plain conditional model. Above 1 you trade diversity for fidelity. No classifier is needed (D), which is the point of the method. ([slide 27](slides.md#slide-27), [38:00](https://youtu.be/qd6Ldsuu46I?t=2280))

</details>

### Q9

What is the regression target in flow matching with the straight path $x_t = (1 - t)\epsilon + t x_0$?

- A) The noise $\epsilon$.
- B) The clean sample $x_0$.
- C) The log-density.
- D) The velocity $x_0 - \epsilon$, which is constant along the path.

<details><summary>Answer</summary>

**D.** Differentiating the path with respect to $t$ gives $x_0 - \epsilon$. Diffusion predicts $\epsilon$ (A); with a linear schedule the two are equivalent reparameterisations, as the bottom of slide 33 says. ([slide 33](slides.md#slide-33), [44:01](https://youtu.be/qd6Ldsuu46I?t=2641))

</details>

### Q10

Diffusion Policy predicts a sequence of future actions but executes only the first few before predicting again. Why?

- A) Within one sampled chunk the actions are consistent with one mode, so the robot does not flicker between modes; predicting again every few steps keeps it reactive.
- B) To save GPU memory.
- C) Diffusion models cannot predict a single action.
- D) To need fewer demonstrations.

<details><summary>Answer</summary>

**A.** Sampling every step independently from a multimodal policy can alternate between "left" and "right" and end up in the middle anyway. A chunk is one coherent sample; the receding horizon keeps reacting to new observations. ([slide 29](slides.md#slide-29), [40:01](https://youtu.be/qd6Ldsuu46I?t=2401))

</details>

## Guest talk

### Context

Cheng Chi wrote Diffusion Policy (this lecture's case study) and the Universal Manipulation Interface (UMI), a hand-held gripper with a camera for collecting demonstrations without a robot. He is now CTO and co-founder of Sunday Robotics, with Tony Zhao (ALOHA and ACT). The captions get his PhD advisor wrong: he did his PhD with Shuran Song, first at Columbia, then at Stanford (the captions say "Peter Allen" and "Curran's lab"). The talk is deliberately not about algorithms: it covers what he had to learn after the PhD, about hardware, manufacturing and data operations.

### Extra comments

**"There was never actually end-to-end."** Even a pixels-to-actions policy sits between drivers, calibration, filtering and low-level control (PID, sometimes SLAM): the week 2 stack. Algorithms dominate courses because they are software, easy to download and easy to grade. In his experience, hardware and data differ more between teams than algorithms do. ([03:00](https://youtu.be/tvFvIEOBKfM?t=180), [04:01](https://youtu.be/tvFvIEOBKfM?t=241))

**Software analogies for hardware.** The bill of materials (BOM) is the `requirements.txt`; procurement is `pip install`; supply-chain management is resolving the whole dependency tree, which he compares to uv, rather than only the top layer. Retail (Amazon) is a physical cache that hides months of lead time until you need a large quantity or a changed part. The buy-versus-build decision depends on that supply chain, and newcomers to hardware tend to build too much themselves. ([06:01](https://youtu.be/tvFvIEOBKfM?t=361), [09:00](https://youtu.be/tvFvIEOBKfM?t=540))

**No two units are identical, so track them.** Every glove gets a serial number and a QR code, each component is logged in a database, and a calibration routine records each joint sensor's range; out-of-range readings flag faulty units. The ML payoff: that record follows every recorded episode into the training data loader, so if one batch of gloves turns out bad, researchers can exclude its data with a config change. Knowing where each training example came from (data provenance) is worth building early. ([12:01](https://youtu.be/tvFvIEOBKfM?t=721), [16:01](https://youtu.be/tvFvIEOBKfM?t=961))

**Why simplicity matters so much in hardware.** He says each added component multiplies the chances of failure. A quick illustration (mine, not from the talk): if each of 50 parts works 99% of the time, all 50 work only $0.99^{50} \approx 60\%$ of the time. The glove went through 100 to 150 CAD iterations to stay minimal. ([28:03](https://youtu.be/tvFvIEOBKfM?t=1683), [29:01](https://youtu.be/tvFvIEOBKfM?t=1741))

**Data collection is an operation, and operations are about people.** The data team grew from about 20 collectors recruited one by one, to about 200 run by an operations lead with spreadsheets and Slack, to thousands under a head of data who came from Tesla's Autopilot labelling. His lesson: clear communication, trust and respect matter more than tooling; motivated collectors even fixed technical problems, like sticking notes on blank white walls so the phone-based visual tracking had features to lock onto. ([18:00](https://youtu.be/tvFvIEOBKfM?t=1080), [20:01](https://youtu.be/tvFvIEOBKfM?t=1201))

**Week 5's importance sampling, in a data operation.** In the Q&A he notes that the distribution you want to train on is not the distribution of the real world, so you instruct collectors in detail to over-sample what you care about. Same idea as reweighting samples from one distribution to estimate another. ([31:01](https://youtu.be/tvFvIEOBKfM?t=1861))

**Why wheels.** The first home robot has a wheeled base because it is passively stable: cut the power and it does not fall on anyone. A startup can serve a niche (single-floor homes) first, and he is cautious about legged robots in homes until there is agreement on what "safe" means for them. ([26:02](https://youtu.be/tvFvIEOBKfM?t=1562))

### Going further

- Chi et al., [Diffusion Policy](https://arxiv.org/abs/2303.04137) and [Universal Manipulation Interface](https://arxiv.org/abs/2402.10329), his two PhD projects.
- The courses he names as formative: Lynch's [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) and (the captions are garbled here) most likely Russ Tedrake's MIT course notes, of which [Robotic Manipulation](https://manipulation.csail.mit.edu/) is the one closest to this class.

## Guest talk quiz

### Q1

What does Chi mean when he says robotics was "never actually end-to-end"?

- A) Robots do not use neural networks.
- B) Policies always need language input.
- C) There is always a software layer (drivers, calibration, low-level control, sometimes SLAM) between the sensors, the policy and the actuators.
- D) Training data is never collected end to end.

<details><summary>Answer</summary>

**C.** The learned policy is one layer of a stack, and the classical robotics around it still has to work. ([03:00](https://youtu.be/tvFvIEOBKfM?t=180))

</details>

### Q2

In his software analogy, what are the hardware versions of `requirements.txt` and `pip install`?

- A) The bill of materials (BOM) and procurement, with supply-chain management handling the whole tree of suppliers.
- B) CAD files and 3D printing.
- C) Firmware and drivers.
- D) Quality assurance and shipping.

<details><summary>Answer</summary>

**A.** The BOM lists every part, its quantity, source and price; procurement gets the parts. Supply-chain management is like a resolver for the full dependency tree, not just the first layer. ([06:01](https://youtu.be/tvFvIEOBKfM?t=361), [10:00](https://youtu.be/tvFvIEOBKfM?t=600))

</details>

### Q3

Why does Sunday give every glove a serial number and record its components and calibration?

- A) For marketing.
- B) For legal compliance.
- C) For shipping labels.
- D) Units are never identical; tracking each glove through to the training data loader lets them catch faulty units and filter out the data those units recorded.

<details><summary>Answer</summary>

**D.** Hardware quality directly becomes data quality. Linking every episode to the device that recorded it turns a hardware fault into a one-line data-loader change. ([12:01](https://youtu.be/tvFvIEOBKfM?t=721), [16:01](https://youtu.be/tvFvIEOBKfM?t=961))

</details>

### Q4

What is his main lesson about scaling data collection to thousands of people?

- A) Build a better user interface.
- B) It is an operation, and operations are about people: clear communication, trust and motivation matter more than software.
- C) Automate collection with AI-generated instructions.
- D) Pay collectors as little as possible.

<details><summary>Answer</summary>

**B.** He started out expecting to solve it with code and instructions, and found that a Slack channel and a spreadsheet suffice for about 200 people if the people side is right, while no software fixes it if it is not. (He does say collectors are paid, so D is wrong.) ([20:01](https://youtu.be/tvFvIEOBKfM?t=1201), [31:01](https://youtu.be/tvFvIEOBKfM?t=1861))

</details>

### Q5

Why does Sunday's first home robot use wheels instead of legs?

- A) Wheels use cheaper motors.
- B) Wheels are faster.
- C) A wheeled base is passively stable: if the power is cut it does not fall, which matters for safety in a home; single-floor homes are a big enough first market.
- D) Legs cannot carry arms.

<details><summary>Answer</summary>

**C.** Safety without power is hard to guarantee for a bipedal robot, and a startup can start with a niche rather than the whole humanoid market. ([26:02](https://youtu.be/tvFvIEOBKfM?t=1562))

</details>
