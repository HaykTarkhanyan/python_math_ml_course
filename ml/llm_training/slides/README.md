# LLM-training paper decks

Beamer lecture decks built from papers in [`../materials/papers/`](../materials/papers/).
Each deck is a self-contained lecture on one paper, following the `ml/` slide conventions
(`ml/SLIDE_STYLE.md`): `default`/`dove` theme, 16:9, Armenian-flag palette, section
transition slides, Python-generated figures, a cold-open hook and a `Next:` recap box.

The eighteen decks span the LLM-training pipeline end to end -- **scaling, tokenization &
pretraining** (9, 5, 6, 13), **attention internals & position** (15, 16), **fine-tuning &
alignment** (10, 1-4, 12, 18), **efficient training at scale** (17), **reasoning** (14, 11),
and two **model reports** (7, 8) that tie it together. Decks 1-4 cross-reference each other in
order; the later decks point back to the foundations they build on.

| # | Deck | Paper | One line |
|---|---|---|---|
| 1 | [`01_grpo/01_grpo.pdf`](01_grpo/) | DeepSeekMath (GRPO), [2402.03300](https://arxiv.org/abs/2402.03300) | Critic-free RL: replace PPO's value net with a group-relative baseline |
| 2 | [`02_dpo/02_dpo.pdf`](02_dpo/) | DPO, [2305.18290](https://arxiv.org/abs/2305.18290) | Skip the reward model and RL: preference alignment as one classification loss |
| 3 | [`03_lora/03_lora.pdf`](03_lora/) | LoRA, [2106.09685](https://arxiv.org/abs/2106.09685) | Fine-tune giant models by training two tiny low-rank matrices |
| 4 | [`04_lima/04_lima.pdf`](04_lima/) | LIMA, [2305.11206](https://arxiv.org/abs/2305.11206) | 1,000 curated examples, no RLHF, competitive with RLHF products |
| 5 | [`05_bpe_dropout/05_bpe_dropout.pdf`](05_bpe_dropout/) | BPE-Dropout, [1910.13267](https://arxiv.org/abs/1910.13267) | Stochastic subword segmentation from plain BPE as regularization |
| 6 | [`06_dont_stop_pretraining/06_dont_stop_pretraining.pdf`](06_dont_stop_pretraining/) | Don't Stop Pretraining, [2004.10964](https://arxiv.org/abs/2004.10964) | Cheap domain- & task-adaptive continued pretraining (DAPT/TAPT) |
| 7 | [`07_llama3/07_llama3.pdf`](07_llama3/) | Llama 3, [2407.21783](https://arxiv.org/abs/2407.21783) | A simple, well-executed recipe at scale beats architectural cleverness |
| 8 | [`08_qwen3/08_qwen3.pdf`](08_qwen3/) | Qwen3, [2505.09388](https://arxiv.org/abs/2505.09388) | Unified thinking/non-thinking modes with a thinking-budget dial |
| 9 | [`09_scaling_laws/09_scaling_laws.pdf`](09_scaling_laws/) | Kaplan [2001.08361](https://arxiv.org/abs/2001.08361) + Chinchilla [2203.15556](https://arxiv.org/abs/2203.15556) | Compute-optimal: scale params and data equally (~20 tokens/param) |
| 10 | [`10_instructgpt/10_instructgpt.pdf`](10_instructgpt/) | InstructGPT, [2203.02155](https://arxiv.org/abs/2203.02155) | The RLHF pipeline (SFT -> reward model -> PPO) that GRPO/DPO later simplify |
| 11 | [`11_deepseek_r1/11_deepseek_r1.pdf`](11_deepseek_r1/) | DeepSeek-R1, [2501.12948](https://arxiv.org/abs/2501.12948) | Reasoning emerges from pure RL; distill it into small models |
| 12 | [`12_qlora/12_qlora.pdf`](12_qlora/) | QLoRA, [2305.14314](https://arxiv.org/abs/2305.14314) | 4-bit NF4 base + 16-bit adapters: finetune 65B on one GPU, no quality drop |
| 13 | [`13_moe/13_moe.pdf`](13_moe/) | Switch [2101.03961](https://arxiv.org/abs/2101.03961) + Mixtral [2401.04088](https://arxiv.org/abs/2401.04088) | Route each token to a few experts: huge params, constant compute per token |
| 14 | [`14_chain_of_thought/14_chain_of_thought.pdf`](14_chain_of_thought/) | CoT [2201.11903](https://arxiv.org/abs/2201.11903) + Let's Verify [2305.20050](https://arxiv.org/abs/2305.20050) | Reason step by step, and reward the steps (process supervision) |
| 15 | [`15_flash_attention/15_flash_attention.pdf`](15_flash_attention/) | FlashAttention, [2205.14135](https://arxiv.org/abs/2205.14135) | IO-aware exact attention: tile it, never write the N x N matrix to HBM |
| 16 | [`16_rope/16_rope.pdf`](16_rope/) | RoFormer/RoPE, [2104.09864](https://arxiv.org/abs/2104.09864) | Encode position by rotating q,k so the score depends only on relative distance |
| 17 | [`17_deepseek_v3/17_deepseek_v3.pdf`](17_deepseek_v3/) | DeepSeek-V3, [2412.19437](https://arxiv.org/abs/2412.19437) | 671B MoE trained cheaply: aux-loss-free balancing, FP8 training, MTP (R1's base) |
| 18 | [`18_flan/18_flan.pdf`](18_flan/) | FLAN, [2109.01652](https://arxiv.org/abs/2109.01652) | Instruction tuning at scale: many tasks-as-instructions to zero-shot generalization |

## Layout (per deck)

```
NN_topic/
  NN_topic.tex     # the deck; \input{../../../preamble}
  NN_topic.pdf     # compiled output (checked in)
  fig/*.pdf        # figures, Python-generated (matplotlib)
  py_src/make_figures.py   # regenerates every figure in fig/
  code/            # (DPO only) source listings shown on slides
```

## Build

Figures (needs `matplotlib` + `numpy`; use the repo `ma` venv):

```bash
python3 NN_topic/py_src/make_figures.py     # writes NN_topic/fig/*.pdf
```

Deck (run twice so the Outline / `\tableofcontents` resolves), then clean:

```bash
cd NN_topic
pdflatex -interaction=nonstopmode -halt-on-error NN_topic.tex
pdflatex -interaction=nonstopmode -halt-on-error NN_topic.tex
python3 ../../../../clean_latex.py .        # strip .aux/.log/.nav/.toc/... (keeps the PDF)
```

Or use the repo `/compile-deck` skill, which runs the full compile -> verify -> clean loop.

## Figure fidelity

Benchmark/ablation numbers hardcoded in the figure scripts are taken verbatim from each
paper's tables. Where a paper reports only a **trend** and not per-point values (GRPO
Maj@K/Pass@K and method-ablation ordering; DPO's reward-KL frontier; LIMA's data-quantity
plateau), the figure is drawn to reproduce the reported *shape* and is labelled as
schematic on the slide. Every other chart uses real reported numbers.
