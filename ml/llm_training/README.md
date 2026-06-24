# LLM Training

Reference materials on training large language models: end-to-end playbooks plus the
key papers behind tokenization, pretraining, fine-tuning, and preference/RL alignment.

Downloaded **2026-06-23**. Files are numbered `01`-`10` in reading order.

- [`materials/`](materials/) - the originals (PDF papers, HTML playbook snapshots).
- [`materials_md/`](materials_md/) - plain-text / markdown versions of the same files, for fast reading, search, and grep. See [`materials_md/README.md`](materials_md/README.md).

## Contents (reading order)

| # | Title | Topic | Original | Source URL |
|---|---|---|---|---|
| 01 | The Smol Training Playbook: Secrets to Building World-Class LLMs (HuggingFaceTB) | End-to-end playbook | `blogs/01_smol_training_playbook.html` | https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook |
| 02 | The Ultra-Scale Playbook: Training LLMs on GPU Clusters (nanotron) | Scaling / distributed training | `blogs/02_ultrascale_playbook.html` | https://huggingface.co/spaces/nanotron/ultrascale-playbook |
| 03 | The Llama 3 Herd of Models | Full LLM system report | `papers/03_llama3_2407.21783.pdf` | https://arxiv.org/abs/2407.21783 |
| 04 | BPE-Dropout: Simple and Effective Subword Regularization | Tokenization (BPE) | `papers/04_bpe_dropout_1910.13267.pdf` | https://arxiv.org/abs/1910.13267 |
| 05 | Don't Stop Pretraining: Adapt LMs to Domains and Tasks | Continual / domain-adaptive pretraining | `papers/05_dont_stop_pretraining_2004.10964.pdf` | https://arxiv.org/abs/2004.10964 |
| 06 | LoRA: Low-Rank Adaptation of Large Language Models | Parameter-efficient fine-tuning | `papers/06_lora_2106.09685.pdf` | https://arxiv.org/abs/2106.09685 |
| 07 | LIMA: Less Is More for Alignment | Instruction tuning / data quality | `papers/07_lima_2305.11206.pdf` | https://arxiv.org/abs/2305.11206 |
| 08 | Direct Preference Optimization (DPO) | Preference alignment | `papers/08_dpo_2305.18290.pdf` | https://arxiv.org/abs/2305.18290 |
| 09 | DeepSeekMath (introduces GRPO) | RL alignment / reasoning | `papers/09_grpo_deepseekmath_2402.03300.pdf` | https://arxiv.org/abs/2402.03300 |
| 10 | Qwen3 Technical Report | Full LLM system report | `papers/10_qwen3_2505.09388.pdf` | https://arxiv.org/abs/2505.09388 |

## Suggested reading order

1. **Playbooks first** for the big picture - Smol Training Playbook (01: what to do and why), then Ultra-Scale Playbook (02: how to scale across GPUs).
2. **Tokenization:** BPE-Dropout (04).
3. **Pretraining & adaptation:** Don't Stop Pretraining (05).
4. **Fine-tuning:** LoRA (06, efficient), LIMA (07, data quality over quantity).
5. **Alignment:** DPO (08, offline preference), then GRPO/DeepSeekMath (09, RL).
6. **Put it together:** the Llama 3 (03) and Qwen3 (10) reports show all of the above at production scale.

---

*Papers are arXiv PDFs (verified `%PDF` headers). Playbooks are HTML page snapshots. If a file is ever lost, re-download from the source URL above.*
