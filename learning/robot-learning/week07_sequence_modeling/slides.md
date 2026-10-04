# Lecture 7: Sequence Modeling & Transformers - slide text

- Lecture: Robot Learning: From Fundamentals to Foundation Models, ETH Zurich, Spring 2026, Oier Mees, 30.03.2026
- Source: `slides.pdf` in this folder (59 frames grabbed from the lecture recording, 1280x720, no text layer), from github.com/idanL1212000/RobotLerning-2026-ETH-Zurich `lectures/week07_slides.pdf`
- Transcribed from the frame images by Claude (vision), not by an OCR engine. Text is verbatim; equations are LaTeX; pictures are described in `[Figure: ...]`. Frames that are animation builds of the same slide are merged into one section, with the frame numbers listed. Everything in square brackets is the transcriber's own wording, not slide text: `[Figure: ...]` descriptions, `[Note: ...]` remarks, build and arrow annotations, and heading parts that are not on the slide (e.g. `[untitled: ...]`, `[continued]`).
- Video: https://youtu.be/imSTfMJjp7M (transcript: `transcript_lecture.md`; guest talk: `transcript_guest_ted_xiao.md`)

---

## Title (frame 1)

**Robot Learning: From Fundamentals to Foundation Models**

Lecture 7: Sequence Modeling & Transformers

Oier Mees - ETH zurich, Microsoft - 30.03.2026

[Note: slide 1 does not appear in the captured frames; the numbering starts at 2.]

## Slide 2: Recap: Limitations of Reactive Policies (frame 2)

[Figure: observation $o_t$ (snowy road through a windscreen) -> **Policy** $\pi_\theta(a_t \mid o_t)$ -> action $a_t$ (the road with a forked arrow), with an arrow from $a_t$ back to $o_t$.]

**One Snapshot Can't Model Full State**

- **Memory**: what did I do 10 seconds ago? How fast are pedestrians moving?
- **Action Smoothness**: single step model might produce jerky movements

## Slide 3: Robotics as a Sequence Modeling Problem (frame 3)

- Robots usually operate in a POMDP, which require reasoning over history $\tau = (o_0, a_0, o_1, a_1, \dots, o_T)$

[Figure: token rows labelled **Image** (blue) and **Action** (purple) above a tabletop camera image and an "ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$" robot-arm picture.]

$$p_\pi(\tau) = \rho(s_0) \prod_{t=0}^{T-1} \pi(a_t \mid o_t)\, P(o_{t+1} \mid o_t, a_t)$$

- [arrow at $\pi(a_t \mid o_t)$] Trajectory is a sequence, but the policy is single-step and reactive

## Slide 4: Autoregressive Models (frame 4)

- Any joint distribution can be written as a product of conditionals (chain rule of probability)

$$p_\theta(x) = \prod_{i=1}^{T} p_\theta(x_i \mid x_{1:i-1})$$

- [arrow at the conditional] How do we learn each conditional?

## Slide 5: Modeling Sequences with RNNs (frame 5)

- SOTA until 2016 in NLP, process sequences sequentially
- Limitations:
  - Learn long-range dependencies, exploding/vanishing gradients
  - Parallelize training due to recurrence
  - Retain information, fixed-size $h_t$ state compresses all history

$$h_t = f(h_{t-1}, x_t)$$

[Figure: an interactive "RNN" panel, "O(n) - Steps to reach first word: 6": hidden states $h_0 \dots h_6$ over the words "The robot picked up the spoon carefully"; "To reach 'The', information flows through 6 hidden states. For length n: O(n)."]

References: *Long Short-Term Memory*, Hochreiter & Schmidhuber (1997); *Sequence to Sequence Learning with Neural Networks*, Sutskever et al., (2014)

## Slide 6: Transformer (frame 6)

- Replace recurrence with attention, giving direct $O(1)$, equal access to the entire history

[Figure: interactive panel "Transformer vs RNN: path length to attend past words" over the tokens "The robot picked up the spoon (carefully)". Transformer side, "O(1) - Any two tokens interact in a single step, regardless of distance": "'spoon' reaches 'The' in 1 step - same cost regardless of distance". RNN side, "O(n) - Steps to reach first word: 5": "To reach 'The', information flows through 5 hidden states. For length n: O(n)."]

Reference: *Attention Is All You Need*, Vaswani et al., (2017)

## Slide 7: Transformer: Attention Mechanism (frames 7-26)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- Compare current hidden state (query) to all past hidden states (keys)
- Construct attention distribution to figure out what parts of the history are relevant via a softmax

[Figure: an interactive step-through on the sentence "The robot picked up the spoon", with steps 1 - embed, 2 - project Q,K,V, 3 - raw scores, 4 - softmax, 5 - weighted sum. Step 2: "Q = eWQ, K = eWK, V = eWV. Every token creates three vectors via learned weight matrices: Query (what am I looking for?), Key (what do I advertise to others?), Value (what I will contribute if attended to)." The query is "spoon"; the other words get keys. Step 3: "score(q, $k_i$) = q · $k_i$ / √d. The query vector for 'spoon' is dot-producted with every Key vector. A high score means that key matches what the query is looking for. These are raw unnormalized scores." Raw scores (unnormalized - does not sum to 1): The 0.4, robot 2.2, picked 1.7, up 2.8, the 0.9, spoon 3.1. Step 4: "$\alpha_i = \exp(\text{score}_i) / \sum_j \exp(\text{score}_j)$. Softmax normalizes the raw scores into probabilities summing to 1. High scores become dominant weights; low scores are suppressed. These are the final attention probabilities $\alpha_i$." Softmax weights (sum = 1.00): The 2.6%, robot 15.8%, picked 9.6%, up 28.8%, the 4.3%, spoon 38.9%. Step 5: "output = $\sum_i \alpha_i \cdot v_i$. The output is a weighted blend of all Value vectors, weighted by $\alpha_i$. 'spoon' now carries a blended representation absorbing context from the entire sequence it attended to." A bar shows output("spoon") as a mix of v(The) ... v(spoon) in those proportions.]

[The live demo cycles through steps 2-5 repeatedly across frames 7-26; all twenty frames show one of the states above.]

## Slide 8: Positional Encodings (frame 27)

- Attention has no inherent notion of order
- Inject order information to tokens

| Absolute Positional Encoding | Relative Positional Encoding |
|---|---|
| A fixed vector PE is added to each token embedding before the transformer sees it. | A bias b(i-j) based on the distance between tokens is added to the attention score, not the embedding. |

[Figure: two worked examples on "The robot picked up the spoon". Absolute: (1) token embeddings + (2) fixed position vectors PE(0) ... PE(5) = (3) inputs to transformer e+PE(0) ... e+PE(5); a red box "PE(6)? PE(7)? never seen during training". Relative: (1) context tokens (keys), query = "spoon"; offset from query; (2) relative bias per token b(-5) ... b(-1), added to attention score; (3) biased attention scores s+b(-5) ... s+b(-1); a green box "s+b(-6), s+b(-7): offsets -6, -7 work fine ✓".]

## Slide 9: Original Transformer Architecture (frame 28)

[Figure: the encoder-decoder transformer. **Encoder** (x N layers): Source embedding ("The robot picked...") + PE -> Multi-head self-attention (bidirectional - sees all tokens) -> Add & Norm -> Feed-forward network (two linear layers + ReLU) -> Add & Norm -> encoder output (K, V). **Decoder** (x N layers): Target embedding ("Il robot ha..." shifted right) + PE -> Masked MH self-attention (causal - past output only) -> Add & Norm -> Multi-head cross-attention (Q: decoder, K,V: encoder) -> Add & Norm -> Feed-forward network -> Add & Norm -> Linear + softmax (next token probability) -> output prediction. A causal attention-mask matrix (query position 0-9 against key position 0-9) is filled on and below the diagonal: "can attend (score kept)"; above it "masked ($-\infty \to 0$ after softmax)".]

- **Cross-attention**: Same as self-attention, but Q comes from the decoder, K and V come from the encoder
- **Encoder - Bidirectional self-attention**: Every token attends to every other token, past and future
- **Decoder - Causal self-attention**: Each token only attends to itself and past tokens

## Slide 10: Training (frame 29)

[Figure: teacher-forcing example. Input (ground truth, shifted right): ⟨s⟩, The, robot, picked, up -> **Decoder-only transformer + causal mask** -> predicted distribution (one step ahead) $\hat{p}(x_1) \dots \hat{p}(x_5)$ with p = 0.72, 0.61, 0.85, 0.54, 0.68 -> target (ground truth): The, robot, picked, up, spoon. Per-token loss $-\log p_\theta(x_t \mid x_1 \dots x_{t-1})$: The 0.329, robot 0.494, picked 0.163, up 0.616, spoon 0.386; average 0.397 = L.]

**Teacher Forcing**: maximize the likelihood of the next correct token $x_t$ given the true preceding tokens $x_{1:t-1}$

**Cross-Entropy Loss**

$$L = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

- ✅ 1x forward pass computes all T losses with Transformers
- ❌ T forward passes with RNNs

## Slide 11: Language Tokenization (frame 30)

- **Characters**: a=0, b=1, c=2,...
  - Small vocabulary
  - Large number of tokens
- **Words**: cat=0, car=1, dog=2, ...
  - Large vocabulary
  - Small number of tokens
  - What about new words?

**Ideally we would like something in between**

## Slide 12: Byte-Pair Encoding (frame 31)

- Tokenize based on groupings of characters, prioritizing by frequency
- Generalizes to novel combinations of characters

```text
function BPE (strings C, number of merges k)
  V <- all unique characters in C                      # initial vocabulary is characters
  for i = 1 to k do                                     # repeat k times
    t_L, t_R <- most frequent pair of adjacent tokens in C
    t_NEW <- t_L + t_R                                  # concatenate into new token
    V <- V + t_NEW                                      # add to vocabulary
    replace each (t_L, t_R) in C with t_NEW             # update corpus
  return vocab V
```

Reference: *Neural Machine Translation of Rare Words with Subword Units*, Sennrich et al., (2015)

## Slide 13: BPE Example (frames 32-36)

- BPE compression factor scales with corpus size

[Figure: an interactive BPE demo on the input sentence "The robots melted the robot's unbearably cheesy fondue", stepping through 0 - start, 1 - "th", 2 - "the", 3 - "ro", 4 - "rob", 5 - "robo", 6 - "robot", 7 - result, 8 - summary. Step 0: "every character is its own token. '_' marks word boundaries. The sentence has 54 tokens - one per character including spaces", with the top pair frequencies ('h'+'e' 3x, 'r'+'o' 3x, 't'+'h' 2x, ...). Step 6: merge robo + t -> robot (2x in corpus): "The full word is now a single token. Both 'robots' and 'robot's' share this token - BPE naturally handles morphological variants"; vocabulary learned: th, the, ro, rob, robo, robot; token count 44 (-10 from start). Step 8, compression summary: characters (start) 54 tokens, after BPE merges 34 tokens; 37% fewer tokens; 0.63x compression ratio; 7 merge rules learned; key tokens: "the", "robot" -> 1 token, "unbearably" -> 4 tokens, "fondue" -> 4 tokens. "Real LLMs use 50k-100k vocab with millions of merges - common words become 1 token, rare domain words split into subwords."]

## Slide 14: Large Language Models (frame 37)

- Decoder-only architectures -> simpler to scale
- Extend BPE to work on byte-level -> no unknown tokens
- Techniques for scaling:
  - Rotary Position Embeddings (RoPE)
    - Rotates $Q$ and $K$ vectors by an angle proportional to their position, so that $Q \cdot K$ depends only on the relative distance between tokens
  - FlashAttention:
    - Attention with $O(n)$ memory instead of $O(n^2)$ by tiling the computation in fast SRAM, never materializing the full attention matrix

-> RoPE + FlashAttention make training on longer sequences practical

## Slide 15: Scaling LLMs (frame 38)

[Figure: three circles sized by model size.]

| GPT-1 (2018) | GPT-2 (2019) | GPT-3 (2020) |
|---|---|---|
| 117M params, 1B tokens | 1.5B params, 10B tokens | 175B params, 300B tokens |
| Pretraining + fine-tuning | Zero-shot generalization | In-context learning emerges |

## Slide 16: The Bitter Lesson - Sutton (2019) (frame 39)

- Methods that scale with compute ultimately beat hand-engineered solutions
  -> In-context learning is an emergent property of scale

[Figure: a portrait of Richard Sutton, and the GPT-3 paper plot of Accuracy (%) against Number of Examples in Context (K), from zero-shot through one-shot to few-shot, for 175B, 13B and 1.3B parameter models, each with a "Natural Language Prompt" (solid) and "No Prompt" (dashed) curve. The 175B model climbs to about 65% accuracy; 13B reaches about 25%; 1.3B stays below about 5%.]

Reference: *Language Models are Few-Shot Learners*, Brown et al., OpenAI (2020)

## Slide 17: Scaling Laws (frame 40)

- Given compute budget, scale data & model size equally
  - Optimal model size $N^* \propto C^{0.5}$, data $D^* \propto C^{0.5}$, compute $C \approx 6 \times N \times D$
  - ~20 tokens per parameter for compute-optimality

[Figure: three log-scale plots against compute ($\log_{10}$ FLOPs, $10^{17}$ to $10^{24}$). (1) "training loss vs compute": one curve per model size (75M, 125M, 250M, 500M, 1B, 2.5B, 5B, 10B), each flattening out, with their lower "envelope" highlighted. (2) "optimal model size $N^*$ vs compute": a dashed "optimal scaling frontier" line with GPT-1, GPT-2 and GPT-3 plotted near it (GPT-3 above it). (3) "optimal training tokens $D^*$ vs compute": the same frontier, with GPT-3 well below it (too few tokens for its size).]

Reference: *Training Compute-Optimal Large Language Models*, Hoffmann et al. (2022)

## Slide 18: Scaling Laws: Extrapolating Performance (frame 41)

- Fit power law on cheap runs -> extrapolate to predict optimal N and final loss before committing compute

[Figure: two plots. Left, "isoFLOP curves - fit on small runs, predict large model performance": validation loss against model size N for compute budgets $C = 10^{19}$, $10^{21}$, $10^{23}$, each a U-shape with a marked minimum; a "power law fit ($N^*$ locus)" through the minima is extended ("extrapolation") into a "never trained" region, with "Prediction at $C = 10^{25}$: $N^* \approx$ 282B params, loss $\approx$ 1.85". Right, "tokens vs model size - compute-optimal to inference-optimal": training tokens D against model size N with the "Chinchilla frontier ($D^* = 20 \times N^*$)". GPT-1/2/3 lie below it, "undertrained" (GPT-3: "300B tokens, needs ~3.5T"); Chinchilla and LLaMA 1-2 sit on it, "compute-optimal"; LLaMA 3 (70B) sits far above it, "inference-optimal: deliberately over-trained". The top right is shaded "extrapolation region".]

## Slide 19: [untitled: transition question] (frame 42)

**How can we extend LLMs & Transformers to Vision?**

## Slide 20: Image Tokenization (frame 43)

- Split image to patches, flatten & add positional encodings

[Figure: ViT patch pipeline. A 224x224 image is split into 14x14 = 196 patches (16x16 px each); one 16x16 patch = 768 values (16x16x3 RGB) is flattened (768-d), passed through a linear projection W (768 -> 768 learned weights) to a patch embedding (d = 768 dims), plus a positional embedding that encodes patch position, giving a patch token (1 of 196 tokens) that goes into the transformer, "same as for text tokens". Bottom: "x 196 patches -> sequence of 197 tokens (196 patch + 1 [CLS])": [CLS], $p_1, p_2, \dots, p_{196}$.]

Reference: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, Dosovitskiy, Beyer et al. (2020)

## Slide 21: [untitled: transition question] (frame 44)

**How does the Transformer know which text tokens correspond to which visual tokens?**

## Slide 22: CLIP: Two-Tower Contrastive Alignment (frame 45)

[Figure: top, images 1 ... N go through an **image encoder** (ViT) and an **image proj.** $W_i$ to image embeddings; captions ("a dog running", "green forest", "city at night", ...) go through a **text encoder** (transformer) and a **text proj.** $W_t$ to text embeddings; matching pairs are compared by cosine similarity. Bottom left, the CLIP paper figure: a text encoder (caption "Pepper the aussie pup") and an image encoder fill an N x N matrix of products $I_i \cdot T_j$, with the matching pairs on the diagonal highlighted.]

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N} \sum_{i=1}^{N}\left[\underbrace{\log \frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N} e^{s_{ij}/\tau}}}_{\text{Image-to-Text Loss}} + \underbrace{\log \frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N} e^{s_{ji}/\tau}}}_{\text{Text-to-Image Loss}}\right]$$

- [arrow at $\tau$] Learnable temperature, entropy of the resulting probability distribution

Reference: *Learning Transferable Visual Models From Natural Language Supervision*, Radford et al. (2021)

## Slide 23: Llava: Early-Fusion (frame 46)

- Prepends image tokens to the text sequence, LLM processes everything in single unified self-attention pass

[Figure: image input -> **CLIP vision encoder** (always frozen; patch embeddings) -> **MLP connector** (projection layer; trainable) -> image tokens $i_1, i_2, i_3, \dots$ (196), prepended to the text tokens $w_1 \dots w_4$ from the text prompt ("describe this"): a "unified token sequence into LLM". **LLM (LLaMA)**: self-attention over all tokens jointly; standard autoregressive generation; no new attention mechanisms -> text output "a cat sitting on a mat".]

Reference: *Visual Instruction Tuning*, Liu et al. (2023)

## Slide 24: Flamingo: Late-Fusion via Gated Cross-Attention (frame 47)

- Vision is injected at every layer via cross-attention, the LM never sees image tokens directly

[Figure: image -> **vision encoder** (frozen) -> **Perceiver resampler** (compresses to fixed-size K, V) -> visual features (K and V for x-attn). Text tokens only ($w_1 \dots w_5$) flow up a stack that alternates "frozen self-attention · Chinchilla LM" with "gated cross-attention: Q from text · K,V from vision, $x \leftarrow x + \tanh(\alpha) \cdot \text{x-attn}(Q, K, V)$", ending in text output "a cat sitting on a mat". Side notes: "$\alpha$ learned, init = 0: $\tanh(0) = 0 \to x \leftarrow x$, LM fully preserved"; "trained: cross-attention layers, Perceiver resampler, tanh gate $\alpha$ (one per layer)".]

Reference: *Flamingo: a Visual Language Model for Few-Shot Learning*, Alayrac et al. (2022)

## Slide 25: Native Multimodal Models (frame 48)

- Instead of adding vision to an existing LLM, train all modalities jointly from scratch

[Figure: "any combination of inputs": image -> vision enc., audio -> audio enc., video -> video enc., text -> tokenizer ("each modality is optional"), merged into an "interleaved token sequence (any order, any mix)". Examples: image caption task (img img img txt txt txt -> "describe this image"); audio transcription (aud aud aud aud txt -> "transcribe this"); video QA (vid txt vid txt vid txt, freely interleaved); multimodal reasoning (img txt aud img txt aud, any mix, any order). All go into a **unified multimodal transformer** ("trained from scratch - every weight has always seen every modality") -> text (+ image) output.]

## Slide 26: VLM Taxonomy (frame 49)

| early fusion: pre-trained LLM + vision adapter | late fusion: pre-trained LLM + cross-attention | natively multimodal: joint training from scratch |
|---|---|---|
| LLaVA · 2023 - CLIP ViT + MLP + LLaMA | Flamingo · 2022 - gated cross-attn + Chinchilla LM | Gemini 1 · 2023 - dense transformer · from scratch |
| PaliGemma 2 · 2024 - SigLIP + linear proj. + Gemma 2 | GPT-4V · 2023 - undisclosed · likely cross-attn | Chameleon · 2024 - VQ-VAE: images as discrete tokens in shared vocabulary |
| InternVL 2.5 · 2024 - InternViT-6B + MLP + LLM | LLaMA 3.2 Vision · 2024 - cross-attention + LLaMA 3 | GPT-4o · 2024 - natively multimodal · undisclosed |
| DeepSeek-VL2 · Dec 2024 - SigLIP + MLP + DeepSeekMoE | IDEFICS 2 · 2024 - open Flamingo reproduction | Gemini 3 · 2025 - sparse MoE · 1M ctx · Deep Think |
| Gemma 3 · Mar 2025 - SigLIP + linear proj. + Gemma 3 | NVLM-X · NVIDIA 2024 - cross-attn + InternViT-6B | LLaMA 4 · 2025 - MoE · 400B / 17B · 1M ctx |
| Phi-4-multimodal · Feb 2025 - SigLIP-2 + MoE-LoRA + Phi-4 | | GPT-5 · Aug 2025 - native MM · unified routing |
| Qwen2.5-VL · 2025 - dyn-res ViT + MLP + Qwen2.5 | | Qwen3.5 · Feb 2026 - joint pretraining · text + vision |
| Qwen3-VL · 2025 - DeepStack + MRoPE + Qwen3 | | |
| Mistral Large 3 · 2025 - sparse MoE 675B / 41B + vision | | |

## Slide 27: [untitled: transition question] (frame 50)

**Transformer → LLMs → VLMs → Native Multimodal models**

**Can we extend it to robot actions?**

## Slide 28: Robotics as Multimodal Sequence Modeling (frame 51)

[Figure: three groups of tokens, **Language** (green), **Image** (blue), **Action** (purple), above their sources: the instruction "Pick up the spoon", the tabletop camera image, and "ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$" with a robot arm. Same figure as week 1, slide 33.]

## Slide 29: Robot Action Tokenization for VLAs (frame 52)

**Per-Dimension, Per-Timestep Binning**

**Quantile** bounds prevent outliers from expanding the discretization range and wasting bin resolution

1. step 1 - quantile normalization per dimension [Figure: histograms of one action dimension for robot A (small gripper), with $Q_1 = -0.02$ and $Q_{99} = +0.04$, and robot B (large arm), with $Q_1 = -1.50$ and $Q_{99} = +2.30$; values outside the quantiles are clipped, and both map to "normalised [-1, +1] - identical for every robot".]
2. step 2 - divide [-1, +1] uniformly into 256 bins · bin width = 2/256 ≈ 0.0078 [bin 0 ... 256 equal bins ... bin 255]
3. step 3 - N-dim action at timestep t -> N discrete tokens [Figure: action $a_t \in \mathbb{R}^7$ -> tokens 183, 107, 240, 018, 155, 072, 201 for $a_1 \dots a_7$.]
4. step 4 - inject action tokens into pre-trained language model vocabulary · train with next-token prediction [existing vocabulary (kept) | 256 slots -> action tokens [0...255]]

References: *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, Brohan et al. (2023); *OpenVLA: An Open-Source Vision-Language-Action Model*, Kim et al. (2024)

## Slide 30: Action Chunking (frame 53)

- Predict k future actions for smoother behavior $\pi(a_{t:t+k} \mid o_t)$

[Figure: two timelines. "action chunking - predict K actions at once, execute all K before re-querying the model": query 1 returns $a_0 \dots a_3$ for $t \dots t+3$ (chunk 1 - execute all 4), then re-query, and query 2 returns $a_4 \dots a_7$ (chunk 2 - execute all 4). "action chunking + temporal ensemble - re-query every step · $w_i = \exp(-m \cdot i)$ · i = 0 is oldest prediction": queries at $t, t+1, t+2, t+3$ each predict 4 actions, so 4 overlapping predictions exist for step $t+3$, combined as $\hat{a}_3 = \sum w_i \cdot a_i / \sum w_i$. A plot of the weight $w_i = e^{-0.01 i}$ against i (age of prediction) from 0 (oldest, $w_0 = 1.0$, highest weight) to 50 (newest, $w_{50} \approx 0.61$, lowest weight).]

Reference: *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, Zhao et al. (2023)

[Note: by the slide's own definition (i = 0 is the oldest prediction) the oldest prediction gets the highest weight; the plot labels agree. This matches the ACT paper, but reads the opposite way from the usual intuition that newer predictions should count more.]

[Note: slide 31 does not appear in the captured frames; the numbering jumps from 30 to 32.]

## Slide 32: Naïve Action Tokenization (frame 54)

As frequency increases:

- Information per token decreases
- Correlation between token increases

-> Solution: **compression**

[Figure: plot of task success score (-1 to 5.5) against data control frequency (2, 5, 10, 20 Hz) for "Naïve Binning Tokenization": about 1.1 at 2 Hz, a peak of about 3.6 at 5 Hz, then 0.5 at 10 Hz and about -0.1 at 20 Hz, with error bars. Right: three tokens entering an **Autoregressive Transformer**, with an arrow showing the next token predicted from the previous ones.]

## Slide 33: Effective Action Tokenization (frame 55)

- How to compress continuous robot data?
  - Byte-Pair Encoding (BPE) not suited for continuous data
  - VQ learned compression complex to train
- Key idea: **Discrete Cosine Transform** (DCT) ≈ JPEG compression

[Figure: five-step pipeline. (1) Normalized action chunk (first 2 dimensions displayed) -> Discrete Cosine Transform (DCT) -> (2) Frequency components -> Quantize -> (3) Sparse frequency matrix (each dim = 1 row), mostly zeros, e.g. rows 124 12 -3 0 0 0 12 / -86 0 0 0 0 0 0 / 344 3 1 0 0 1 5 / ... -> Flatten -> (4) Low-frequency components first: 124, -86, 344, -45, 178, 12, 0, 3, 0, 15, ... -> Byte Pair Encoding (BPE) -> (5) Compressed action tokens: 978, 233, 19, 1022, 1.]

## Slide 34: FAST: Efficient Action Tokenizer for VLAs (frame 56)

- Scales to higher frequencies and trains 5x faster

[Figure: left, score against data control frequency (2-20 Hz): FAST rises from about 3 at 2 Hz to about 7 at 20 Hz, while Naïve Binning Tokenization peaks at about 3.5 at 5 Hz and falls to about 0 at 20 Hz. Right, evaluation score against train iteration (0-700k): "$\pi_0$ + FAST (ours)" reaches about 79 by 100k iterations; $\pi_0$ reaches about 76 only at 700k, marked "5x faster VLA Training".]

Reference: *FAST: Efficient Action Tokenization for Vision-Language-Action Models*, Pertsch\*, Stachowicz\*, Ichter, Driess, Nair, Vuong, **Mees** et al. RSS 2025. **Finalist Best Conference Paper Award**

## Slide 35: FAST: Efficient Action Tokenizer for VLAs [results] (frame 57)

[Two arrows from the title:]

| Generality @ Berkeley, Stanford, UW | Dexterity @ Physical Intelligence |
|---|---|
| [Figure: a 3x3 grid of autonomous robot-arm rollouts in kitchens (4x speed)] | [Figure: dexterous tasks: a robot arm over a table of dishes, laundry folding, a bussing station (8x speed)] |

Reference: *FAST: Efficient Action Tokenization for Vision-Language-Action Models*, Pertsch\*, Stachowicz\*, Ichter, Driess, Nair, Vuong, **Mees** et al. RSS 2025. **Finalist Best Conference Paper Award**

## Slide 36: Conclusion (frame 58)

- Robotics is a sequence modeling problem
- Transformers as a scalable architecture → LLMs
- Different architectures for multimodal inputs → VLMs, Native Multimodal models
- Naïve robot tokenization breaks at high frequency control
- Compression-based techniques like FAST popular

## End (frame 59)

Thank you for your attention
