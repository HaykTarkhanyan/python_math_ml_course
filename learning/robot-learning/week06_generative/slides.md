# Lecture 6: Generative Models - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 23.03.2026
- Source: `slides.pdf` in this folder (38 frames grabbed from the lecture recording, 1280x720, no text layer), from github.com/idanL1212000/RobotLerning-2026-ETH-Zurich `lectures/week06_slides.pdf`
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://youtu.be/qd6Ldsuu46I (transcript: `transcript_lecture.md`; guest talk: `transcript_guest_cheng_chi.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 6: Generative Models

Oier Mees - ETH zurich, Microsoft - 23.03.2026

## Slide 1: Recap: Why Generative Modeling for Robotics? (frame 2)

**Stochasticity:** Many (**infinite**) ways to close a drawer with a 7-DoF robot!

**Expert Inconsistency**: Humans operators might use different '**modes**' across trials

[Figure: three photos of a robot arm closing the same drawer with clearly different gripper poses.]

## Slide 2: Recap: Multimodal Behavior (frame 3)

- Deterministic MSE BC policy will fail due to "**averaging**" of **modes**!
- **Goal**: learn the **full distribution**!

[Figure: two photos of a snowy slope with a lone tree. Left: ski tracks split and pass the tree on both sides. Right: a skier who took the average of the two tracks and went straight into the tree.]

## Slide 3: Generative Modeling (frame 4)

- **Goal**: Learn complex, **full distributions**

$$p_\theta(x) \rightarrow \text{in robotics } \pi_\theta(a_t \mid s_t)$$

| How to Encode a Distribution | How to Sample from a Distribution | How to Transport through a Distribution |
|---|---|---|
| VAEs | Diffusion Models | Flow Models |

## Slide 4: Generative Modeling [noise to data] (frame 5)

- A generative model converts samples from an initial distribution $p(z)$ to a data distribution $p_{data}(x)$

[Figure: a square of random colour noise -> **Generative Model** (trapezoid) -> a photo of a cat.]

## Slide 5: Latent Variable Models (frame 6)

- Output still gaussian, but receives additional input
- Can represent "any" distribution
- Popular: conditional VAEs

$$\pi(\mathbf{a} \mid \mathbf{o}, z) = f_\theta(\mathbf{o}, z)$$

- [arrow at $f_\theta$] Conditional Decoder

[Figure: the ski slope with two tracks around the tree, the left track labelled $z_{\text{left}} \sim \mathcal{N}(0, \mathbf{I})$ and the right $z_{\text{right}} \sim \mathcal{N}(0, \mathbf{I})$.]

## Slide 6: Autoencoder (frame 7)

- Compress input $x$ into a compact latent representation $z$, then reconstruct $x'$

[Figure: graph $x \to z \to x'$ above a network diagram: **Encoder** -> **$z$ Latent** -> **Decoder**, with $z = f_\phi(x)$ under the encoder and $x' = g_\theta(z)$ under the decoder.]

$$\mathcal{L} = \underbrace{\left\| x - g_\theta\big(f_\phi(x)\big) \right\|^2}_{\text{Reconstruction}}$$

## Slide 7: Issues with Autoencoders (frame 8)

- Latent space has no structure
- Bottleneck (size) is the only regularizer
- $z$ is deterministic, can't sample new data!

[Figure: a 2D latent space ($z_1$, $z_2$ axes) with three separate clusters labelled class A, class B and class C; the space between them is marked "? empty ? can't sample here".]

## Slide 8: Variational Autoencoders (VAEs) (frame 9)

- Constrain $z$ to follow a **distribution**, often $p(z) = \mathcal{N}(0, I)$
  - Forces latent space to be continuous, structured & sampleable

[Figure: **Encoder** $q_\phi(z \mid x)$ outputs $\mu$ and $\sigma^2$ -> **Sample** -> **$z$ Latent** -> **Decoder** $p_\theta(x \mid z)$.]

$$\log p_\theta(x) \ge \underbrace{-KL\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{Regularization}} + \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right]}_{\text{Reconstruction}}$$

- **Regularization**: are latents $z$ close to our prior $p(z)$?
- **Reconstruction**: how well does latent $z$ explain $x$?

## Slide 9: Reparameterization Trick (frame 10)

- Problem: Sampling is non-differentiable

[Figure: two pipelines. Top, "✗ Naive sampling - gradients cannot flow": Encoder (outputs $\mu, \sigma^2$) -> Sample z ($z \sim \mathcal{N}(\mu, \sigma^2)$, dashed box) -> Decoder (outputs $x'$) -> Loss $\mathcal{L}$, with the backward arrow marked "✗ gradient blocked here" at the sampling box. Bottom, "✓ Reparameterised - gradients flow through $\mu$ and $\sigma$": Encoder (outputs $\mu, \sigma^2$) -> $\mu$ and $\sigma$ ($= \sqrt{\sigma^2}$); $\sigma$ is multiplied ($\odot$) by $\epsilon$ ($\sim \mathcal{N}(0, I)$, not learned) and added to $\mu$: $z = \mu + \sigma \odot \epsilon$ -> Decoder -> Loss, with "✓ gradients flow to $\mu$ and $\sigma$".]

$$z = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)$$

## Slide 10: Derivation (frames 11-12)

$$\nabla_\phi \mathcal{L} = \nabla_\phi \int q_\phi(z \mid x) f(z)\, dz$$

**Option 1) Score Function Estimator - REINFORCE**

$$\nabla_\phi \mathcal{L} = \mathbb{E}_{z \sim q_\phi(z \mid x)}\left[f(z) \nabla_\phi \log q_\phi(z \mid x)\right]$$

- [arrow] $\nabla_\phi$ acts on $\log q_\phi$ only; High variance, as $\frac{\partial z}{\partial \phi} = 0$

[Option 2 is added in the second build (frame 12):]

**Option 2) Reparameterization**

| Step | Comment |
|---|---|
| $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$ | Reparam. $z \to \epsilon$, $p(\epsilon)$ independent of $\phi$ |
| $\nabla_\phi \mathcal{L} = \nabla_\phi \int p(\epsilon) f\big(\mu_\phi(x) + \sigma_\phi(x) \odot \epsilon\big)\, d\epsilon$ | Leibniz/linearity, move grad. inside integral |
| $\nabla_\phi \mathcal{L} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\left[\nabla_\phi f\big(\mu_\phi(x) + \sigma_\phi(x) \odot \epsilon\big)\right]$ | $\nabla_\phi$ acts on f directly -> low variance $\frac{\partial z}{\partial \phi} \ne 0$ |
| $\nabla_\phi \mathcal{L} \approx \frac{\partial f}{\partial z} \cdot \frac{\partial z}{\partial \phi}$, where $\frac{\partial z}{\partial \mu_\phi} = 1$, $\frac{\partial z}{\partial \sigma_\phi} = \epsilon$ | Chain Rule |

## Slide 11: Variational Autoencoders (VAEs) [latent space] (frame 13)

- Latent space is continuous & structured
- Generative! Any z ~ p(z) decodes meaningfully
- Smooth! Nearby z 's produce similar outputs

[Figure: "VAE latent space": the class A, B and C clusters now sit inside one dashed circle labelled $p(z) = \mathcal{N}(0, I)$, and a point "sample $z \sim p(z)$" drawn from the middle of the circle.]

## Slide 12: Issues with Variational Autoencoders (frame 14)

- **Posterior Collapse**: powerful decoder can learn to model $p(x)$ directly, while ignoring $z$
- **Prior Mismatch**: sampling $z \sim \mathcal{N}(0, I)$ at test time can land in regions the decoder was never trained on

Test time: $z \sim \mathcal{N}(0, \mathbf{I}) \rightarrow$ lands in $z_{\text{left}}$ (✓), $z_{\text{right}}$ (✓), or gap (✗)

[Figure: the ski slope with the two tracks around the tree.]

Training time: $z_{\text{left}} \sim q_\phi(z \mid x_{\text{left}}) = \mathcal{N}(\mu_{\text{left}}, \sigma^2_{\text{left}})$ and $z_{\text{right}} \sim q_\phi(z \mid x_{\text{right}}) = \mathcal{N}(\mu_{\text{right}}, \sigma^2_{\text{right}})$

## Slide 13: Vector Quantized VAE (VQ-VAE) (frame 15)

- Replace the continuous Gaussian latent $z$ with a **discrete codebook** of $K$ learned vectors
- Force to "commit" to a discrete code

[Figure: the VQ-VAE architecture from the paper. Encoder: an image of a dog -> CNN -> $z_e(x)$ (feature grid). Each feature vector is replaced by its nearest entry of the "Embedding Space" codebook $e_1, e_2, e_3, \dots, e_K$, giving a grid of code indices $q(z \mid x)$ (e.g. 1, 3, 2, 53) and the quantized grid $z_q(x)$. Decoder: CNN -> reconstructed image $p(x \mid z_q)$. A red arrow $\nabla_z L$ shows the gradient copied from the decoder input back to the encoder output. Right inset: in embedding space, $z_e(x)$ is pulled toward its nearest code $e_2$, and $z_q(x) \sim q(z \mid x)$.]

Reference: *Neural Discrete Representation Learning*, van den Ord, Vinyals, et al., (2017)

## Slide 14: VQ-VAE (frame 16)

- What's the nearest codebook vector $\mathcal{C} = \{e_1, e_2, \dots, e_K\}$?

$$z_q(x) = e_k \quad \text{where} \quad k = \arg\min_j \left\| z_e(x) - e_j \right\|_2$$

- [arrow at the argmin] **Not Differentiable!** updating $z_e(x)$ will still point to same neighbor

**Straight-Through Estimator**: copy the gradients from the decoder input ($z_q$) directly to the encoder output ($z_e$), bypassing the argmin

$$\frac{\partial z_q}{\partial z_e} \approx \mathbf{I} \quad \Rightarrow \quad \nabla_{z_e} \mathcal{L} \approx \nabla_{z_q} \mathcal{L}$$

$$z_q = z_e + \text{sg}[z_q - z_e]$$

$$\frac{\partial z}{\partial z_e} = \frac{\partial z_e}{\partial z_e} + \frac{\partial\, \text{sg}[z_q - z_e]}{\partial z_e} = 1 + 0 = 1$$

[Note: the reference author is van den Oord; the slide spells it "van den Ord".]

## Slide 15: VQ-VAE: Training (frame 17)

**Full Loss**

$$\mathcal{L} = \underbrace{\left\| x - D(z_q) \right\|_2^2}_{\text{Reconstruction}} + \underbrace{\left\| \text{sg}(z_e) - e_k \right\|_2^2}_{\text{Codebook}} + \underbrace{\beta \left\| z_e - \text{sg}(e_k) \right\|_2^2}_{\text{Commitment}}$$

- **Reconstruction**: Did the decoder recover x from the quantised code? Trains encoder & decoder jointly via STE
- **Codebook**: Moves the codebook vectors $e_k$ toward the encoder outputs to learn a representative vocabulary of the data
- **Commitment**: Forces the encoder to stay close to its chosen codebook entry, preventing the latent space from growing arbitrarily

Training gives us the codebook $\mathcal{C}$ and decoder $D$, but **how do we sample new data at test time?**

## Slide 16: VQ-VAE: How to Generate New Data? (frame 18)

1. Training Stage: $x \xrightarrow{\text{encoder}} z_e \xrightarrow{\arg\min} e_k \in \mathcal{C} = \{e_1, e_2, \dots, e_K\}$
2. Collect code indices per training sample $\mathcal{D} = \{k^{(1)}, k^{(2)}, \dots, k^{(N)}\}$
3. Learn a categorical prior over the sequence of codes: $p(k_1, \dots, k_n) = \prod_{i=1}^{n} p(k_i \mid k_1, \dots, k_{i-1})$ - i.e. autoregressive transformer...
4. Inference time:
   - I) Sample Index: $k_i \sim p(k_i \mid k_1, \dots, k_{i-1})$
   - II) Lookup Vector: $e_{k_i} = \text{Lookup}(\mathcal{C}, k_i)$
   - III) Decode: $\hat{x} = D(e_{k_1}, \dots, e_{k_n})$

## Slide 17: VQ-VAE [latent spaces compared] (frame 19)

[Figure: three 2D latent-space panels.]

| Autoencoder | VAE | VQ-VAE |
|---|---|---|
| scattered · gaps · no structure | compact · continuous · sampleable | discrete · explicit · no gaps |
| [clusters A, B, C far apart, the space between them marked "gap ✗"] | [clusters inside a dashed $\mathcal{N}(0, I)$ circle, with a "sample" point in the middle] | [four codebook vectors $e_1 \dots e_4$ drawn as diamonds, each with its encoder outputs snapped around it] |
| no structure -> gaps | structured but continuous | discrete -> no gaps possible |

- ✅ Fixes posterior collapse
- ✅ No prior mismatch
- ✅ Discrete and controllable, good compression
- ❌ Codebook collapse -> Finite Scalar Quantization (FSQ)
- ❌ Need to determine $K$ in advance
- ❌ Reconstruction still uses MSE, mean-seeking in outputs remains

[Note: slide 18 does not appear in the captured frames; the numbering jumps from 17 to 19.]

## Slide 19: Diffusion (frame 20)

- VQ-VAEs still have mean-seeking in outputs due to MSE
- Diffusion learns complex distributions over continuous variables

[Figure: a cat photo $x_0$ getting noisier left to right until it is pure noise $x_T$; top arrow **Forward Process**, bottom arrow (right to left) **Generative Backward Process**.]

## Slide 20: Diffusion: Forward Process (frame 21)

- Iteratively add gaussian noise

$$x_{i+1} = \underbrace{\sqrt{1 - \beta_{i+1}}}_{\text{Signal Decay}}\, x_i + \underbrace{\sqrt{\beta_{i+1}}}_{\text{Noise Scale}}\, \epsilon_i \qquad \epsilon_i \sim \mathcal{N}(0, \mathbf{I})$$

- $\beta_i$: noise schedule
- [arrow at $\epsilon_i$] Reparametrization Trick!

[Figure: the cat image getting noisier left to right.]

## Slide 21: Diffusion: Backward Process (frame 22)

- Learn $p(x_{i-1} \mid x_i)$, which is intractable (infinite clean images could have produced the noisy version)
- **Solution:** instead of predicting the previous image, predict the added noise $x_{i-1} \approx x_i - \epsilon_\theta(x_i, i)$!

[Figure: pure noise $x_T$ denoised left to right into the cat image $x_0$.]

## Slide 22: Diffusion: Objective (frame 23)

- Same ELBO as VAE, but $T$ denoising steps instead of one, and q is fixed

$$\log p_\theta(x_0) \ge \underbrace{-D_{KL}\big(q(x_T \mid x_0) \,\|\, p(x_T)\big)}_{\text{Prior Regularization}} + \sum_{i=2}^{T} \underbrace{-D_{KL}\big(q(x_{i-1} \mid x_i, x_0) \,\|\, p_\theta(x_{i-1} \mid x_i)\big)}_{\text{Denoising Matching}} + \underbrace{\log p_\theta(x_0 \mid x_1)}_{\text{Reconstruction}}$$

- KL between two Gaussians has closed form

$$D_{KL} \propto \left\| \tilde{\mu}_i - \mu_\theta(x_i, i) \right\|^2 \propto \left\| \epsilon - \epsilon_\theta(x_i, i) \right\|^2$$

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{i \sim U(1, T),\, \epsilon \sim \mathcal{N}(0, \mathbf{I})}\left[\left\| \epsilon - \epsilon_\theta(x_i, i) \right\|^2\right]$$

Reference: *Denoising Diffusion Probabilistic Models*, Ho et al., (2020)

## Slide 23: Diffusion: Efficient Training (frame 24)

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{i \sim U(1, T),\, \epsilon \sim \mathcal{N}(0, \mathbf{I})}\left[\left\| \epsilon - \epsilon_\theta(x_i, i) \right\|^2\right]$$

- [arrow at $i \sim U(1, T)$] Requires "teleporting" instead of iterative forward process O(T)
- **Signal Retention** for algebraic convenience: $\alpha_i = 1 - \beta_i$

| Step | |
|---|---|
| Forward Process | $x_{i+1} = \sqrt{1 - \beta_{i+1}}\, x_i + \sqrt{\beta_{i+1}}\, \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \mathbf{I})$ |
| Expand | $x_1 = \sqrt{\alpha_1}\, x_0 + \sqrt{1 - \alpha_1}\, \epsilon_0$, $x_2 = \sqrt{\alpha_2}\, x_1 + \sqrt{1 - \alpha_2}\, \epsilon_1$ |
| Substitute | $x_2 = \sqrt{\alpha_2}\big(\sqrt{\alpha_1}\, x_0 + \sqrt{1 - \alpha_1}\, \epsilon_0\big) + \sqrt{1 - \alpha_2}\, \epsilon_1 = \sqrt{\alpha_1 \alpha_2}\, x_0 + \sqrt{\alpha_2 (1 - \alpha_1)}\, \epsilon_0 + \sqrt{1 - \alpha_2}\, \epsilon_1$ |
| [noise variances add up] | $\alpha_2 (1 - \alpha_1) + (1 - \alpha_2) = 1 - \alpha_1 \alpha_2$ |
| Gaussian Sum | $\mathcal{N}(0, \sigma_a^2 I) + \mathcal{N}(0, \sigma_b^2 I) = \mathcal{N}\big(0, (\sigma_a^2 + \sigma_b^2) I\big)$ -> $x_2 = \sqrt{\alpha_1 \alpha_2}\, x_0 + \sqrt{1 - \alpha_1 \alpha_2}\, \bar{\epsilon}$ |
| Generalizing | Signal Coefficient $= \sqrt{\alpha_i \alpha_{i-1} \dots \alpha_1} = \sqrt{\prod_{j=1}^{i} \alpha_j} = \sqrt{\bar{\alpha}_i}$ |

$$\boxed{x_i = \sqrt{\bar{\alpha}_i}\, x_0 + \sqrt{1 - \bar{\alpha}_i}\, \bar{\epsilon}}$$

Allows "teleporting" from $x_0$ to any noisy $x_i$ in a single step $O(1)$!

## Slide 24: Diffusion: DDPM Sampling (frame 25)

1. Sample Gaussian Noise $x_T \sim \mathcal{N}(0, I)$
2. For $i = T, \dots, 1$:
   - I. Sample $z \sim \mathcal{N}(0, I)$ if $i > 1$, else $z = 0$
   - II. Denoise $x_{i-1} = \underbrace{\frac{1}{\sqrt{\alpha_i}}\left(x_i - \frac{1 - \alpha_i}{\sqrt{1 - \bar{\alpha}_i}}\, \epsilon_\theta(x_i, i)\right)}_{\text{Predicted Mean}} + \underbrace{\sigma_i z}_{\text{Stochastic (Langevin) Term}}$
3. Return $x_0$

- [arrow at step 2] **Very slow! Image generation T=1000!**
- [arrow at $\sigma_i$] Typically $\sigma_i = \sqrt{\beta_i}$
- **Stochastic (Langevin) Term**: We add noise during denoising to increase sample diversity and avoid collapsing into a mean distribution
- DDPM is a discrete version of a Stochastic Differential Equation (SDE)
- DDPM denoising = score step ($\nabla \log(p)$) towards higher density + Langevin noise

## Slide 25: Diffusion: DDIM Sampling (frame 26)

Reference: *Denoising Diffusion Implicit Models*, Song et al., (2020)

1. Sample Gaussian Noise $x_T \sim \mathcal{N}(0, I)$
2. Define subset of timesteps $\tau = \{t_1, t_2, \dots, t_K\}$ (only 20 steps instead of 1000)
3. For $i = K, K-1, \dots, 1$: (where $x_i$, $\bar{\alpha}_i$ denote $x_{t_i}$ and $\bar{\alpha}_{t_i}$)
   - I. Denoise $\hat{x}_0 = \frac{x_i - \sqrt{1 - \bar{\alpha}_i}\, \epsilon_\theta(x_i, i)}{\sqrt{\bar{\alpha}_i}}$
   - II. Deterministic Re-Noise $x_{i-1} = \sqrt{\bar{\alpha}_{i-1}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{i-1}}\, \epsilon_\theta(x_i, i)$
4. Return $x_0$

- [arrow at step I] Inverse of "teleportation" formula, can jump to any $x_{i-1} \in \tau$ directly
- [arrow at $\bar{\alpha}_{i-1}$] $\alpha_{i-1}$ from $\tau$, not necessarily the previous timestep, this enables skipping timesteps
- [arrow at the end of step II] No noise term!

- Decouples number of denoising iterations in training & inference
- DDPM is markovian, DDIM defines a non-Markovian forward process that results in the exact same marginal distributions as DDPM
- DDIM discretizes the probability flow ODE

## Slide 26: [untitled: conditional generation] (frame 27)

**Everything covered so far is unconditional generation from noise!**

**What about conditional generation?**

[Figure: a noise square and the prompt "A photorealistic image of an astronaut riding a horse" -> **Generative Model** $\epsilon_\theta(x_t, t, \mathbf{c})$ (the prompt enters as $\mathbf{c}$) -> an image of an astronaut riding a horse in a desert.]

## Slide 27: Classifier-Free Guidance (CFG) (frame 28)

- **Problems**: Model might ignore conditioning, classifier guidance required training separate classifier $p(\mathbf{c} \mid x_t)$
- **Solution**: train a single model for conditional & unconditional generation by randomly dropping the condition during training

$$\hat{\epsilon} = \underbrace{\epsilon_\theta(x_t, t, \emptyset)}_{\text{Unconditional}} + w \underbrace{\big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\big)}_{\text{Guidance=Conditional-Unconditional}}$$

- [arrow at $w$] **Guidance Scale** condition fidelity vs diversity trade-off

Reference: *Classifier-Free Diffusion Guidance*, Ho & Salimans, (2022)

## Slide 28: Case Study: Diffusion Policy (frame 29)

- **Key**: Denoises robot action sequences instead of images
- DDPM training, DDIM inference, injects observations into noise prediction network (no future state pred required)

$$\mathcal{L} = \text{MSE}\big(\epsilon^k, \epsilon_\theta(O_t, A_t^0 + \epsilon^k, k)\big) \qquad A_t^{k-1} = \alpha\big(A_t^k - \gamma\, \epsilon_\theta(O_t, A_t^k, k) + \mathcal{N}(0, \sigma^2 I)\big)$$

[Figure: a robot arm pushing a red T-shaped block on a table, with many coloured dots showing sampled action trajectories.]

Reference: *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, Cheng Chi et al., (2023)

## Slide 29: Case Study: Diffusion Policy [architecture] (frame 30)

- $O_t$ computed **once** before $K$ denoising steps
- 1D temporal U-Net with $O_t$ conditioned via Feature-wise Linear Modulation (FiLM)
- Handles multimodality in demonstrations!

[Figure: the paper's overview figure. Input: image observation sequence plus robot pose $\mathbf{O}_t$. Output: action sequence, shown as the denoising steps $A_t^K$, $A_t^3$, $A_t^0$ converging from scattered dots to one trajectory. (a) Diffusion Policy General Formulation: observations $o_{t-2}, o_{t-1}, o_t$ go into Diffusion Policy $\varepsilon_\theta(\mathbf{O}, \mathbf{A}, k)$, which outputs an action sequence $a_t \dots a_{t+3}$ over a prediction horizon $T_p$; the window then slides to $\mathbf{O}_{t+4}$ / $\mathbf{A}_{t+4}$. (b) CNN-based: Conv1D blocks on the action embedding, each modulated by FiLM ($a \cdot x + b$ from linear layers on $\mathbf{O}_t$), repeated $\times K$, outputting $\nabla E(\mathbf{A}_t)$. (c) Transformer-based: action embeddings cross-attend to observation embeddings, repeated $\times K$.]

Reference: *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, Cheng Chi et al., (2023)

[Note: slides 30 and 31 do not appear in the captured frames; the numbering jumps from 29 to 32.]

## Slide 32: Scaling Diffusion in Robotics: DiT Block Policy (frames 31-33)

- Replaces cross-attention with adaptive Layer-Norm
- Scales with Model & Data size
- Stable Training
- Long horizon bimanual dexterous tasks (1500+ timesteps)

[Figure: left, an embedded video of the two-armed sushi-cutting workstation playing across three frames (the arms pick up a red knife and cut the roll). Right, the "DiT-Block Policy" diagram: the goal "Cut Sushi" (encoded with DistilBERT) and three camera views (global, left wrist, right wrist), each through a ResNet + FiLM, feed a **Self Attention Encoder** (repeated N x); a **Diffusion Decoder** (repeated N x) takes the noisy actions $x^K$ and step $K$ and outputs $\epsilon$. An inset shows the adaLN-Zero block: Layer Norm -> Scale, Shift ($\gamma_1, \beta_1$) -> Multi-Head Self-Attention -> Scale ($\alpha_1$), then Layer Norm -> Scale, Shift ($\gamma_2, \beta_2$) -> MLP -> Scale ($\alpha_2$), with the scale and shift values produced by an MLP from the mean encoder output and $K$.]

Reference: *The Ingredients for Robotic Diffusion Transformers*, Dasari, **Mees** et al., ICRA, 2025

[Note: week 2 cites the same paper as ICRA 2024; ICRA 2025, as here, is correct.]

## Slide 33: Flow Matching (frame 34)

Reference: *Flow Matching for Generative Modeling*, Lipman et al., (2023)

- Learn a velocity field $v_\theta(x_t, t)$ that transports samples from noise $\epsilon \sim \mathcal{N}(0, I)$ to data $x_0 \sim p(x_0)$ along continuous paths

$$x_t = (1 - t)\epsilon + t x_0, \quad \epsilon \sim \mathcal{N}(0, I), \quad t \in [0, 1]$$

- [brace under $(1 - t)\epsilon + t x_0$] Straight line from noise to $x_0$, no schedule

**Training:**

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\left\| \underbrace{v_\theta(x_t, t)}_{\text{Predicted velocity}} - \underbrace{(x_0 - \epsilon)}_{\text{Ground-truth velocity}} \right\|^2\right]$$

- [brace under $(x_0 - \epsilon)$] Ground-truth velocity from noise to $x_0$ is constant, independent of $t$

**Inference:**

$$x_{t + \Delta t} = x_t + v_\theta(x_t, t) \Delta t$$

Integrate learned velocity with Euler ODE

$$\underbrace{\mathbb{E}\left[\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2\right]}_{\text{Diffusion, predict noise}} \;\overset{\text{With linear schedule}}{\Longleftrightarrow}\; \underbrace{\mathbb{E}\left[\left\| v_\theta(x_t, t) - (x_0 - \epsilon) \right\|^2\right]}_{\text{Flow Matching, predict velocity}}$$

## Slide 34: Rectified Flow (frame 35)

- Retrains Flow Matching on noise-data pairs generated by its own flow, untangling trajectory crossings & straightening paths

[Figure: two animated sampling panels with "Sampling Duration" progress bars. **Flow Matching - Curved Paths Flow Slow**: sample trajectories (orange) bend and cross on their way from noise to the data points (blue). **Rectified Flow - Straight Paths Flow Fast**: the trajectories are nearly straight lines into the same data points.]

Source: Alec Helbling

## Slide 35: Case Study: $\pi_0$ (frame 36)

- Finetunes a VLM to produce actions via Flow Matching
- Samples flow-time $\tau$ from a shifted Beta distribution to focus on noisier (harder action prediction) parts of flow

[Figure: the $\pi_0$ overview. Training data: the "$\pi$ dataset" (robot photos), "Internet pre-training" and "OXE". Three camera images each pass through a ViT, and together with the instruction "fold shirt" go into the **pre-trained VLM** (SigLIP (400M) + Gemma (2.6B)). An **action expert** (300M) takes the robot state $q_t$ and noise and outputs an action chunk $a_t, a_{t+1}, \dots, a_{t+H}$. Target robots: 14 DoF Bimanual Manipulators, 18 DoF Mobile Manipulators, 7 and 8 DoF Single Arm Manipulators.]

Reference: *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, Physical Intelligence (2024)

## Slide 36: Conclusion (frame 37)

| Latent Variable Models | Diffusion Models | Flow Models |
|---|---|---|
| **AE**: compressed representation via reconstruction | **DDPM**: Iteratively denoise Gaussian noise to generate samples | **Flow Matching**: Learns velocity field to transport noise |
| **VAE**: constrain latent to follow a distribution via reparametrization trick | **DDIM**: Faster deterministic sampling via non-Markovian diffusion process | **Rectified Flow**: Straighten ODE trajectories via iterative reflow for faster, fewer-step sampling |
| **VQ-VAE**: Discretize latent space using learned codebook vectors | **CFG**: Steer generation by blending conditional & unconditional scores | |

$$\underbrace{\text{Diffusion (DDPM)}}_{\text{Stochastic SDE}} \xrightarrow{\sigma = 0} \underbrace{\text{DDIM}}_{\text{Deterministic ODE}} \overset{\text{Linear Schedule}}{\longleftrightarrow} \underbrace{\text{Flow Matching}}_{\text{same ODE, simpler Euler solver}}$$

## End (frame 38)

Thank you for your attention
