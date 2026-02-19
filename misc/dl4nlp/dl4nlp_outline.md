Topic order is randomish, also the list is not exhaustive - we'll adapt based on the audience and time constraints.


Pre-Transformer NLP
- RNNs
- N-grams
- Word Embeddings: CBOW, Skip-gram (Word2Vec)

Transformers
- Why LSTM is not enough?
- Self-Attention
- Transformer Architecture
- Positional Encoding (Sinusoidal, RoPE, ALiBi)
- Encoder-Decoder


Tokenization:
- Word / Subword / Character
- BPE
- FastText

Decoding:
- Greedy Search
- Beam Search
- Sampling (Top-k, Top-p)
- Temperature

Evaluation:
- Perplexity
- BLEU
- ROUGE
- Benchmarks

Early notable models 
- GPT
- BERT
- T5


Pre-training:
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Causal Language Modeling (CLM)
- Contrastive Learning

Fine-tuning:
- Supervised Fine-tuning
- Reinforcement Learning from Human Feedback (RLHF)
- Reward Models
- Instruction Tuning
- DPO (Direct Preference Optimization)
- GRPO (DeepSeek)
- PEFT (Parameter-Efficient Fine-Tuning)
- LoRA (Low-Rank Adaptation)
- Adapter Tuning

  
Prompting
- Zero-shot Learning
- Few-shot Learning
- Chain-of-Thought Prompting

Hallucination & Grounding
- What is Hallucination?
- Why it Happens

Mixture of Experts (MoE)
- Sparse vs Dense Models
- Gating Mechanism

Inference Optimization
- KV-Cache
- Quantization (4-bit, 8-bit)
- Distillation

RAG (Retrieval-Augmented Generation)
- Motivation: Grounding & Up-to-date Knowledge
- Architecture: Retriever + Generator
- Embeddings & Vector Databases
- Chunking Strategies

Scaling Laws
- Model Size
- Data Size
- Compute
- Chinchilla
