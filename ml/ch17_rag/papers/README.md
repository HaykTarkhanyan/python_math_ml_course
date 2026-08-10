# RAG source papers

Instructor-side grounding for the RAG chapter (L41, L42). **These are for building and
fact-checking the decks, not a student reading list** - the slides teach the ideas and cite
sparingly.

Downloaded **2026-08-10**. Numbered in teaching order: lexical first, then dense, then the
pipeline that combines them, then how to make it fast, then how to measure it.

- [`../papers/`](.) - the original PDFs.
- [`llm_readable/`](llm_readable/) - `pdftotext` versions for fast reading, search and grep.
  Each file carries `=== PAGE n ===` markers matching the PDF page numbers.

## Contents

| # | Title | Topic | Source | Pages |
|---|---|---|---|---|
| 01 | The Probabilistic Relevance Framework: BM25 and Beyond (Robertson & Zaragoza, 2009) | Lexical retrieval | [staff.city.ac.uk](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) | 59 |
| 02 | Dense Passage Retrieval for Open-Domain QA (Karpukhin et al., 2020) | Dense retrieval, bi-encoders | [2004.04906](https://arxiv.org/abs/2004.04906) | 13 |
| 03 | ColBERT: Late Interaction over BERT (Khattab & Zaharia, 2020) | Late interaction, reranking | [2004.12832](https://arxiv.org/abs/2004.12832) | 10 |
| 04 | Efficient and Robust ANN Search using HNSW graphs (Malkov & Yashunin, 2016) | Vector indexes | [1603.09320](https://arxiv.org/abs/1603.09320) | 13 |
| 05 | Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020) | The RAG pipeline | [2005.11401](https://arxiv.org/abs/2005.11401) | 19 |
| 06 | Ragas: Automated Evaluation of RAG (Es et al., 2023) | Evaluation | [2309.15217](https://arxiv.org/abs/2309.15217) | 8 |

122 pages total.

## Why BM25 is not an arXiv link

BM25 has no single arXiv paper. The canonical modern reference is Robertson & Zaragoza's
*Foundations and Trends in Information Retrieval* survey (Vol 3, No 4, 2009, DOI
10.1561/1500000019), which derives BM25 from the probabilistic relevance framework. The PDF
above is the copy hosted on Robertson's own City, University of London staff page.

The classic BM25 scoring function is Equation (3.15), on PDF page 30. The saturation and
length-normalisation components are (3.13) and (3.12) on pages 29-30. Parameter guidance
(`0.5 < b < 0.8`, `1.2 < k1 < 2`) is in Section 3.5.

## Related material already in the repo

- [`ml/text_embedding/`](../../text_embedding/) - 8 papers on the embedder side (MTEB, SBERT,
  SimCSE, E5, mE5, E5-Mistral, Qwen3-Embedding, ATE). Covers most of what L41's dense-retrieval
  and Armenian sections need; **read that folder alongside this one.**
- [`ml/llm_training/`](../../llm_training/) - the generation side of the stack.
- [`ml/ch9_attention/`](../../ch9_attention/) - transformer background the encoders assume.

## Fidelity caveat

The `.txt` versions are text only - figures, plots and multi-column tables are lost or garbled.
The BM25 equations in particular come through mangled. For anything precise (a number in a
table, an equation, a figure), open the PDF at the page number given by the nearest
`=== PAGE n ===` marker.

## Regenerate

Re-download from the links above, then re-extract with the `ma` venv:

```bash
./ma/Scripts/python.exe ml/ch17_rag/extract_rag_papers.py
```

The extract script verifies page counts and asserts each PDF's expected title appears on page 1,
so a bad download fails loudly rather than silently producing garbage.
