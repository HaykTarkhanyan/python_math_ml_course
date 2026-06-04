# Data Stack — Reference (python_libs/ 01-07)

Covers: API integration (LLM), NumPy, Pandas, visualization, applied data analysis.

## Notebooks

- [01_openai_api_timestamp_generator.ipynb](../../python_libs/01_openai_api_timestamp_generator.ipynb) — **OpenAI API** — Text generation, image generation, audio (Whisper, TTS), **structured output** (JSON schema / Pydantic), conversation history management. Project: generate YouTube chapter timestamps from a transcript.
- [02_numpy.ipynb](../../python_libs/02_numpy.ipynb) — **NumPy** — Array creation (from list, zeros/ones, empty, identity, random, ranges), 1D and 2D indexing + slicing, reshape, attributes (`shape`, `dtype`, `ndim`), math ops, trig/log/exp, aggregations (sum/mean/min/max/argmin/argmax), stats, **linear algebra** (`dot`, `matmul`, `inv`, `det`, `eig`, `svd`), rounding, concat (`hstack`/`vstack`/`concatenate`), split, add/remove elements, sort, shuffle, rotate.
- [03_pandas_1.ipynb](../../python_libs/03_pandas_1.ipynb) — **Pandas 1** — `Series` (creation, indexing, ops), CSV format, **DataFrame** (creation, head/tail, indexing by column / row, `loc` / `iloc`, row + col deletion, string ops, new col creation, new row addition, `describe`), filtering (single + multi-condition), **groupby**, **pivot table**, renaming, replacing, saving.
- [04_pandas_2.ipynb](../../python_libs/04_pandas_2.ipynb) — **Pandas 2** — `merge` (inner/left/right/outer), suffixes, **Excel** (multiple sheets), **HTML** read, `melt` (wide → long), **datetime** indexing and operations.
- [05_noble_people_analysis.ipynb](../../python_libs/05_noble_people_analysis.ipynb) — **Noble people analysis** — Capstone-style EDA on Wikipedia "noble persons" data, suicide patterns analysis, pivot tables.
- [06_data_viz.ipynb](../../python_libs/06_data_viz.ipynb) — **Data viz (Matplotlib)** — Good examples, colors / styles / markers / opacity, `fig` / `axes` API, subplots, **histogram**, **bar**, **box plot**, examples of bad charts, fancy plots, **facets**, decorations, **wordcloud** (with mask).
- [07_kargin_project.ipynb](../../python_libs/07_kargin_project.ipynb) — **Kargin project** — Applied analysis of Armenian Kargin TV show data: missing values, location analysis, actor counts, fuzzy search (`fuzzywuzzy`), YouTube API integration.

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| L01 design matrix `X` in Python | libs 02 (NumPy 2D arrays, reshape) |
| L01 GD step in code | libs 02 (vectorized ops, `dot`) |
| OLS closed form `(X^T X)^{-1} X^T y` | libs 02 (`inv`, `dot`) → links to math L05 (SVD) |
| Pandas EDA in all labs | libs 03-04 |
| Train/test split mechanics | libs 03 (indexing) + math/stat L1 |
| `groupby` for stratified CV | libs 03 |
| Feature engineering with `merge` (joining external tables) | libs 04 |
| Time-series feature engineering | libs 04 (datetime) |
| Plotting fitted line on data | libs 06 (Matplotlib basics) |
| Plotting risk surface | libs 06 (contour / surface plots) + libs 02 (meshgrid) |
| Confusion matrix heatmap | libs 06 |
| ROC / PR curves | libs 06 |
| Reading scraped datasets (HTML tables) | libs 04 |
| OpenAI API for LLM labs | libs 01 |
| Structured output (Pydantic + LLM) | libs 01 + libs 13 |
| Whisper-based transcript labs (audio ML) | libs 01 (audio section) |

## Notes

- Libs 02 (NumPy) is the **most-cited Python lib** in ML labs. Almost every ML notebook starts with `import numpy as np`.
- Libs 03 + 04 (Pandas) are the second-most cited. Linked from every EDA lab.
- Libs 06 visualization patterns (`fig, axes`) are reused throughout ML — when teaching plot building, point at the relevant section here.
- Libs 07 (Kargin project) is a worked example of a capstone-style EDA — use as a reference when students ask "what should my final notebook look like?".
- The Kargin notebook includes a `fuzzywuzzy` fuzzy-matching pass — useful prerequisite for any NLP / entity-resolution lab.
- Libs 01 OpenAI API + structured output ties into the "Modern LLM Engineering" section of the ML concepts checklist (currently planned for Week 28).
