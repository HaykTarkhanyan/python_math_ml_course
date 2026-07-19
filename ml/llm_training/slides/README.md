# LLM-training paper decks

Beamer lecture decks built from papers in [`../materials/papers/`](../materials/papers/).
Each deck is a self-contained lecture on one paper, following the `ml/` slide conventions
(`ml/SLIDE_STYLE.md`): `default`/`dove` theme, 16:9, Armenian-flag palette, section
transition slides, Python-generated figures, a cold-open hook and a `Next:` recap box.

The four decks form a short **LLM-training / alignment** mini-series and cross-reference
each other in order:

| # | Deck | Paper | One line |
|---|---|---|---|
| 1 | [`01_grpo/01_grpo.pdf`](01_grpo/) | DeepSeekMath (GRPO), [2402.03300](https://arxiv.org/abs/2402.03300) | Critic-free RL: replace PPO's value net with a group-relative baseline |
| 2 | [`02_dpo/02_dpo.pdf`](02_dpo/) | DPO, [2305.18290](https://arxiv.org/abs/2305.18290) | Skip the reward model and RL: preference alignment as one classification loss |
| 3 | [`03_lora/03_lora.pdf`](03_lora/) | LoRA, [2106.09685](https://arxiv.org/abs/2106.09685) | Fine-tune giant models by training two tiny low-rank matrices |
| 4 | [`04_lima/04_lima.pdf`](04_lima/) | LIMA, [2305.11206](https://arxiv.org/abs/2305.11206) | 1,000 curated examples, no RLHF, competitive with RLHF products |

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
