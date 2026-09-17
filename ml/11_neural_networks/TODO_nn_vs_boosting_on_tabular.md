# TODO: show NN vs gradient boosting on tabular data

**Reminder for the neural-networks chapter (ch5).** When teaching neural nets, explicitly contrast
them with gradient boosting (XGBoost / LightGBM / CatBoost) on **tabular** data. Do not let
students walk away thinking "neural nets are always the upgrade."

## The point to make
- On typical **tabular** data, gradient-boosted trees usually **match or beat** deep nets, while
  training faster, tuning more easily, and being more robust to uninformative features and to
  feature scaling.
- Neural nets win on **unstructured** data: images, audio, long text, sequences (and very large
  datasets where representation learning pays off).
- Rule of thumb: tabular -> reach for boosting first; unstructured / very large -> neural nets.

## How to show it (one frame, ideally with a figure)
- A small benchmark bar chart: XGBoost / LightGBM vs an MLP (optionally a tabular-DL model such as
  FT-Transformer or TabNet) on a few tabular datasets - boosting on top or tied, at a fraction of
  the cost. Could reuse the synthetic `make_classification` setup from
  `ml/ch3_trees/py_src/make_xgb_figures.py`, or a couple of OpenML datasets.
- Pair it with the "why": trees handle irregular / non-smooth target functions and uninformative
  features better; MLPs are biased toward smooth functions and need more care on raw tabular input.

## Callback
- L12 (advanced boosting) closes with "next: models for data where trees struggle - images, text,
  sequences." This frame is the other half of that sentence: nets are for non-tabular data, not a
  blanket upgrade over boosting.

## References (verify exact cite when building the slide)
- Grinsztajn, Oyallon, Varoquaux (2022), "Why do tree-based models still outperform deep learning
  on tabular data?", NeurIPS 2022 Datasets & Benchmarks (arXiv:2207.08815). The canonical result:
  trees stay state-of-the-art on medium tabular data even after tuning, and not just because of
  categorical features.
- Shwartz-Ziv & Armon (2022), "Tabular Data: Deep Learning is Not All You Need", Information Fusion.
