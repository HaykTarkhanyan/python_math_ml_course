# 11 Training LLMs — Compute & memory requirements

> Auto-extracted from `9compute.pdf` with `pdftotext` (beamer deck). Slide titles (ALL-CAPS in the source) are marked as `##`. Inline `�` marks a glyph pdftotext could not map (often ×, a Greek letter, or a bullet); diagrams are NOT captured — see the PDF for figures.

Deep Learning for NLP
Training LLMs Memory and compute requirements
Learning goals Learn about different contributions to compute requirements Learn how model size components influence memory requirements


## Number Of Parameters: Notation
In this slide set, we use P for the number of parameters. (Unfortunately, we use N for the number of parameters in other slide sets.)


## Compute Requirements


## Compute Requirements
Basic equation: Cost to train a transformer (decoder) model: C   T = 6PD
Source: Quentin et al., 2023


## Compute Requirements C   T = 6Pd
where:  is throughput of hardware: (No. GPUs) x (FLOPs/GPU) T is the time spent training the model, in seconds P is the number of parameters in the model D is the dataset size (in tokens) C: No. of floating-point operations to train the model: C = Cforward + Cbackward Cforward  2PD; Cbackward  4PD 2PD: 2 comes from the multiply-accumulate operation used in matrix multiplication 4PD: backward pass is approximately twice the compute of the forward pass In the backward pass at each layer, gradients have to be calculated for the weights at that layer and for the previous layers' outputs.


## Compute Units
C (actually, C per time) can be measured in different units:
FLOPs = FLOP-seconds which is [Floating Point Ops / Second] We also use multiples: GFLOP-seconds, TFLOP-seconds etc. Other multiples like PFLOP-days are used in papers 1 PFLOP-day = 1015 � 24 � 3600 FLOP-seconds Actual FLOPs are always lower than the advertised theoretical FLOPs
GPU-hours GPU model is also required, since they have different compute capacities


## Parameters Vs Dataset

Model performance depends on number of parameters P, but also on number of training tokens D

One proposed optimal tradeoff between P and D is: D = 20P

This was posited for Chinchilla models no longer seen as valid. See next lecture

, Hoffmann et al., 2022

You need a training set of several 100s of billions of tokens at least (e.g., LLM does not become fluent otherwise).

Two typical ways of determining P (the size of the model) in 2026:

Based on available compute budget: how much data and how many parameters are optimal? Based on a desired inference time performance (e.g., generate 100 tokens per second): how much data and how many parameters are optimal?


## Memory Requirements


## Memory Requirements
Common questions: How big is this model in bytes? Will it fit/train on my GPUs?
Model size components: Model parameters Optimizer states Gradients Activations


## Number Representations
fp32: single precision floating point number as defined by IEEE 754 standard, takes 32 bits or 4 bytes fp16: half precision float number as defined by , IEEE 754-2008 occupying 16 bits or 2 bytes bf16 or brain floating point 16, developed by Google Brain project, occupying 16 bits or 2 bytes
fewer bits for the "significand" than fp16  bf16 is less precise larger dynamic range than fp16 (8 exponent bits instead of 5)  less likely to suffer from underflow or overflow INT8: integer (e.g., from -128 to 127), occupying 8 bits or 1 byte


## Fp32/Fp16 ( ) (Maxime Labonne)

You can represent every float like this: (-1)S � 2e-bias � 1.m


## Int8 Quantization
It's harder (though not impossible) to fit meaningful floats into 8 bits. Alternative: use integer quantization: INT8.

Real_number = stored_integer * scaling_factor

Absmax quantization (symmetric):

Xquant = round

127 max |X|

X

Zero-point quantization (assymmetric):

scale

=

255 max (X)-min(X)

zeropoint = -round(scale � min(X)) - 128

Xquant = round(scale � X + zeropoint)


## Absmax Int8 (Symmetric)

Link: Maarten Grootendorst


## Zero-Point Int8 (Asymmetric)

Link: Maarten Grootendorst


## Zero-Point Quantization

We assume: min(X)  0  max(X)

scale

=

255 max (X)-min(X)

zeropoint = -round(scale � min(X)) - 128

Xquant = round(scale � X + zeropoint)

Xquant (0) = zeropoint

Xquant = round(scale � X + zeropoint) = round(scale � X + -round(scale � min(X) - 128))

round(scale � (X - min(X))) - 128

=

round (255

X-min(X) max (X)-min(X)

)

-

128

Xquant (max(X)) = 255 - 128 = 127

Xquant (min(X)) = 0 - 128 = -128


## Active Area Of Research

Link: Nvidia paper on NVFP4


## Model Parameters
Parameter size depends on chosen representation: fp32: Memmodel = 4 bytes/param � Nparams fp16 or bf16: Memmodel = 2 bytes/param � Nparams INT8: Memmodel = 1 byte/param � Nparams
It is common to use mixed representations: fp32 + fp16 fp32 + bf16


## Optimizer States
AdamW: MemAdamW = 8 bytes/param � Nparams Momentum: 4 bytes/param Variance: 4 bytes/param
bitsandbytes (8-bit optimizer): Memoptimizer = 2 bytes/param � Nparams Momentum: 1 byte/param Variance: 1 byte/param


## Gradients
They are usually stored in the same datatype as the model parameters. Their memory overhead contribution is:
fp32: Memgrad = 4 bytes/param � Nparams fp16 or bf16: Memgrad = 2 bytes/param � Nparams INT8: Memgrad = 1 byte/param � Nparams


## Activations
GPUs are bottlenecked by memory, not FLOPs Save GPU memory by recomputing activations of certain layers Various schemes for how exactly this is implemented.
Total memory when training without activations: Memtraining = Memparams + Memopt + Memgrad Total memory when training with activations: Memtraining = Memparams + Memopt + Memgrad + Memactiv
