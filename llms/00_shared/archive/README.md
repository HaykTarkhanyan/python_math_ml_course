# archive/ - superseded LLM-chapter scripts and figures

Moved here on 2026-10-28 with the `ml/14_llms` -> `llms/` move: no current deck uses any of these (checked by
matching every `\includegraphics` in the 14 decks). Kept as a record, not expected to run: the scripts compute
their repo root as `parents[3]`, which is one level off from here.

| File | What it was |
|---|---|
| `py_src/l25_l26_figs.py` | July figures for the old L25/L26 decks (permutation proof, positional encoding, causal mask, quadratic cost, three architectures). Ported into `04_transformer_block/py_src/block_figs.py` and `05_transformers_in_the_world/py_src/head_figs.py` at slot size. |
| `py_src/l25_real_attention.py` | Real attention of `intfloat/multilingual-e5-small` on the 3Blue1Brown sentence for old L25; replaced by GPT-2's heads in `block_figs.py`. |
| `py_src/bert_attention.py` | DistilBERT attention heatmap for the July L24 attention deck; cut in the 2026-10-05 restructure. |
| `py_src/attn_intuition.py` | softmax-scaling and illustrative-pattern figures for July L24; replaced by `03_attention/py_src/attention_figs.py`. |
| `fig/l25_*.pdf`, `fig/l26_*.pdf` | outputs of the two L25/L26 scripts above. |
| `fig/bert_*.pdf`, `fig/attn_illustrative.pdf`, `fig/softmax_scaling.pdf`, `fig/dot_product_geometry.pdf`, `fig/worked_scores.pdf` | outputs of the July L24 scripts. |
