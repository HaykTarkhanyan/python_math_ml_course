# Text Embeddings

Reading direction on text embedding models: from sentence-level encoders through contrastive
pre-training to LLM-based and multilingual embedders.

Downloaded **2026-08-01**. Numbered in the order the direction was handed over.

- [`papers/`](papers/) - the original arXiv PDFs.
- [`papers/llm_readable/`](papers/llm_readable/) - `pdftotext` versions for fast reading, search
  and grep. Each file carries `=== PAGE n ===` markers matching the PDF page numbers.

## Contents

| # | Title | Topic | arXiv | Pages |
|---|---|---|---|---|
| 01 | MTEB: Massive Text Embedding Benchmark | Evaluation | [2210.07316](https://arxiv.org/abs/2210.07316) | 24 |
| 02 | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Sentence encoders | [1908.10084](https://arxiv.org/abs/1908.10084) | 11 |
| 03 | SimCSE: Simple Contrastive Learning of Sentence Embeddings | Contrastive learning | [2104.08821](https://arxiv.org/abs/2104.08821) | 17 |
| 04 | Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5) | Contrastive pre-training | [2212.03533](https://arxiv.org/abs/2212.03533) | 17 |
| 05 | Multilingual E5 Text Embeddings: A Technical Report (mE5) | Multilingual | [2402.05672](https://arxiv.org/abs/2402.05672) | 6 |
| 06 | Improving Text Embeddings with Large Language Models (E5-Mistral) | LLM-based embedders | [2401.00368](https://arxiv.org/abs/2401.00368) | 20 |
| 07 | Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models | LLM-based embedders | [2506.05176](https://arxiv.org/abs/2506.05176) | 14 |
| 08 | Less is More: Adapting Text Embeddings for Low-Resource Languages with Small Scale Noisy Synthetic Data (ATE) | Low-resource adaptation | [2603.22290](https://arxiv.org/abs/2603.22290) | 9 |

118 pages total.

## Two links needed resolving

The handover list gave a non-paper URL for two entries:

- **MTEB** was given as <https://huggingface.co/mteb>, the HuggingFace organisation page (leaderboard
  and datasets, no paper). Replaced with the MTEB paper, arXiv 2210.07316.
- **ATE** was given as <https://metric-ai-lab.github.io/less-is-more-embeddings/>, the project page.
  The paper behind it is arXiv 2603.22290, *Less is More: Adapting Text Embeddings for Low-Resource
  Languages with Small Scale Noisy Synthetic Data* (Navasardyan, Bughdaryan, Minasyan, Davtyan).

Both arXiv IDs were checked against arxiv.org before download, and every PDF was verified to carry a
`%PDF` header and to match its expected title after extraction.

## Fidelity caveat

The `.txt` versions are text only - figures, plots, and multi-column tables are lost or garbled.
For anything precise (a number in a table, an equation, a figure), open the PDF at the page number
given by the nearest `=== PAGE n ===` marker.

## Regenerate

Re-download from the arXiv links above, then re-extract with the `ma` venv:

```bash
./ma/Scripts/python.exe extract_embedding_papers.py
```
