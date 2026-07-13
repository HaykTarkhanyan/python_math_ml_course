# 11 Training LLMs — Reducing compute (fast)

> Auto-extracted from `9fast.pdf` with `pdftotext` (beamer deck). Slide titles (ALL-CAPS in the source) are marked as `##`. Inline `�` marks a glyph pdftotext could not map (often ×, a Greek letter, or a bullet). Diagrams are not auto-captured, but the **key figures now have hand-written visual notes added inline** (marked `> **Figure**`, from viewing the rendered slides); minor diagrams still need the PDF.

Deep Learning for NLP
Training LLMs How to reduce compute and memory
Learning goals Learn about different techniques to reduce compute and memory Learn about distributed training with data/tensor parallelism Learn about FlashAttention


## Distributed Training
Training LLMs on several GPUs is faster than on one, but we need to figure out how to distribute. Avoiding OOM (out of memory) issues Data parallelism: split the data on different model replicas Tensor parallellism: split model parameters accross GPUs


## Data Parallelism (1) (Animated Gif!)

> **Figure:** "Distributed Data Parallelism" — 4 GPUs (GPU 0–3), each holding a *full replica* of the model (Layer 0/1/2). The Global Batch is split into Micro Batch 0–3, one per GPU; after local forward/backward, gradients are averaged across GPUs via all-reduce ("Grad AR"). Legend: Idle / Forward / Backward / Grad AR. Source: Nvidia.

Source: Nvidia


## Data Parallelism (2)

Data Splitting Dataset divided into smaller chunks. Each chunk assigned to a different GPU. Each node processes a different subset of the data in parallel.
Model Replication Each GPU has a replica of the neural network model These replicas are trained independently on their respective data subsets
Gradient Aggregation (All Reduce) Gradients are computed locally on each node and then averaged across nodes.
Parameter Synchronization Model parameters are updated synchronously across nodes. This ensures that all model replicas remain consistent with each other after each update step.


## Tensor Parallelism (1)

> **Figure:** A single matmul `X · A = Y` (inputs × weights = outputs) is shown to be *equivalent to* splitting the weight matrix `A` into column shards `A1, A2, A3` — each on a different GPU — producing output shards `Y1, Y2, Y3` that concatenate into `Y`. I.e. column-parallel tensor sharding of one weight matrix across GPUs. Source: Nvidia.

Source: Nvidia


## Tensor Parallelism (2)

Model Partitioning: The model's layers or tensors are split across multiple devices Different parts of the model are assigned to different devices, enabling them to work on separate portions of the computations simultaneously
Forward and Backward Passes: During the forward pass, each device processes its portion of the tensors with intermediate results passed between devices In the backward pass, gradients are computed in the reverse order, again with necessary data transfers between devices
Parameter Updates: Parameter updates can be performed independently on each device for the parameters they own After each update step, the devices synchronize to ensure consistency across the distributed model

FlashAttention
Fast and Memory-Efficient Exact Attention with IO-Awareness
Memory-efficient Reducing from O(n2) to O(n) Question: Why quadratic?
IO aware Reducing memory load/store operations
Therefore: much faster than the regular implementation The bigger the attention matrices, the more you gain (until O(n) no longer fits).
Exact Same as "vanilla attention", not an approximation


## Gpu Memory Hierarchy

Source: Dao, 2022 �


## Computing Considerations
GPU computations are extremely fast, but data transfer is slow. Typical GPU computations are elementwise operations with a huge number of elements, resulting in high memory access  GPU has to wait for data.  Transformer operations are memory-bound. IO aware means reducing memory load/store operations


## Flash Attention
Tiling or chunking of matrices into blocks Operation fusion
For example: first matrix multiplication (query-key), softmax, second matrix multiplication (weight-value) This minimizes redundant swapping in and out of data General idea: you avoid that the large intermediate tensors (e.g., pre-softmax) are written to HBM and then read back into SRAM memory


## Sram Vs Hbm / Fused Kernel

> **Figure:** Left — GPU memory pyramid by bandwidth/size: **SRAM** 19 TB/s (20 MB, on-chip) ▸ **HBM** 1.5 TB/s (40 GB, GPU RAM) ▸ **Main memory / CPU DRAM** 12.8 GB/s (>1 TB). Right — "Attention on GPT-2" runtime bar: standard PyTorch attention ≈15 ms split across Matmul/Dropout/Softmax/Mask/Matmul (each a separate HBM round-trip) vs **FlashAttention** as a single **fused kernel** ≈4 ms. Point: attention is memory-bound, so fusing to keep data in SRAM is the win. Source: Dao, 2022.

Source: Dao, 2022 �


## Tiling

> **Figure:** Matrix multiply `A × B = C` drawn as grids: a highlighted *row-band* of `A` times a *column-band* of `B` produces one small *tile* of `C`. FlashAttention computes attention block-by-block in tiles small enough to fit in SRAM, so the full n×n attention matrix is never materialized in HBM.


## Operation Fusion

> **Figure:** Two Memory↔Compute lanes. *Unfused* (left): data is shuffled back and forth between Memory and Compute for every operation (many load/store round-trips). *Fused* (right): one load, the whole chain of ops runs on-chip, then one store — far fewer memory transfers. Source: horace.io/brrr_intro.

Source: https://horace.io/brrr_intro.html


## Limitations And Prospects
FlashAttention requires implementing "custom" attention in CUDA A new CUDA kernel for each new attention implementation CUDA is lower-level than PyTorch Implementation not transferable accross GPUs
Towards IO-Aware Deep Learning Extending beyond attention DeepSeek


## Teaching Evaluation
teaching evaluation: QAEQ7
