# Python Index — Reference Map for Older Materials

Generated: 2026-05-29
Purpose: When the ML course needs to reference, link back to, or build on Python material already covered in `python/` or `python_libs/`, look here first. Companion to `ml/math_index/`.

## How to use this

- If a future ML lecture, lab, or homework needs to assume / cite Python content, read the relevant section file, find the matching notebook, and link to it from the ML decks/qmd.
- "Use when teaching" headers in each section file suggest which ML lecture topics naturally pair with that Python material.
- Don't re-teach Python in ML decks. Link back to the right notebook.

## Folder map

| Section | File | Coverage | Notebooks |
|---|---|---|---|
| Python fundamentals | [01_fundamentals.md](01_fundamentals.md) | Syntax, control flow, functions, files | `python/01`-`09` |
| OOP + tooling + capstone | [02_oop_and_tooling.md](02_oop_and_tooling.md) | Git/PEP8, exceptions, decorators, classes, inheritance, capstone | `python/10`-`18` + `python/mini_projects/` |
| Data stack | [03_data_stack.md](03_data_stack.md) | OpenAI API, NumPy, Pandas, viz, capstone analysis | `python_libs/01`-`07` |
| Engineering practices | [04_engineering.md](04_engineering.md) | Logging, CLIs, testing, scraping, parallelization | `python_libs/08`-`11` + `clis/`, `testing/`, `scraping/` |
| Backend + modern coding | [05_backend_and_modern.md](05_backend_and_modern.md) | SQL, Pydantic, FastAPI, Supabase, vibe coding, clean code | `python_libs/12`-`18` + `apis/`, `dbs/` |

## Where each artifact type lives

- **Teaching notebooks (`python/`):** 19 numbered notebooks (`00_template.ipynb` through `18_youtube_translator.ipynb`). Bilingual (Armenian section headers: `📚 Նյութը` = material, `🛠️ Գործնական` = practical, `🏡 Տնային` = homework).
- **Library notebooks (`python_libs/`):** 19 numbered notebooks covering the data + engineering stack.
- **Mini-projects (`python/mini_projects/`):**
  - `approximate_pi/` — Monte Carlo Pi estimation
  - `telegram_data/` — Telegram chat data analysis
  - `text_corpus_classes/` — text corpus analysis with OOP (`main.ipynb`)
  - `youtube_playlist/` — playlist analytics (`playlist_analytics.ipynb`)
- **Standalone `.py` files (`python/py_files/`):** `files_helper.py`, `files_main.py`, `streamlit_app.py` — referenced from notebooks for the files / streamlit modules.
- **Sub-projects under `python_libs/`:**
  - `apis/` — FastAPI + Flask shawarma example apps (`fastapi_shawarma.py`, `flask_shawarma.py`, `shaurma_example.py`, `shurma_app.py`, `requirements.txt`).
  - `clis/` — CLI library comparison: `08_02_argparse.py`, `08_02_click.py`, `08_02_fire.py`, `08_02_typer.py`, `commands.sh`.
  - `dbs/` — SQL sample data: `customers.csv`, `orders.csv`, `products.csv`, `generate_sample_data.py`.
  - `scraping/` — Math/Mech staff + YSU project: `utils.py`, `test_utils.py`, `for_lecture/`, `htmls_per_person/`, `htmls_ysu/`, `ysu_project/`.
  - `testing/` — Pytest examples: `calculator.py` + `test_calculator.py`, `elections.py` + `test_elections.py`.
  - `unittest/` — unittest examples.
- **Plan + status docs:**
  - `python/00_25_nov_plan.md` — Python module plan
  - `00_template.ipynb` (both directories) — starting template for new notebooks.

## Conventions used in these files

- Each notebook entry: `[NN file.ipynb](path) — Title — one-line topic list`.
- Bilingual notebooks: topics listed in English even though slide content is Armenian.
- Cross-references back to math modules where Python content uses math concepts (e.g., NumPy linalg → math 01-03).

## Cross-reference to ML course

Most ML lectures depend on this Python content. Examples:

| ML lecture / topic | Python prerequisite |
|---|---|
| Any ML lab (sklearn, fit/predict) | python 14-16 (OOP — sklearn estimator pattern) |
| L01 linear regression lab (NumPy arrays, design matrix) | libs 02 (NumPy) |
| L01-L04 EDA labs | libs 03-04 (Pandas), libs 06 (viz) |
| HW01 GD from scratch | python 07-08 (functions), libs 02 (NumPy) |
| Project EDA notebooks | libs 03-04, libs 06 |
| MLOps / FastAPI model serving | libs 15 (FastAPI), libs 16 (Supabase) |
| Testing ML pipelines | libs 09 (pytest), libs 13 (Pydantic for config) |
| Scraping training data | libs 10-11 (scraping) |
| Logging during training | libs 08 (logging) |
| Clean architecture for ML projects | libs 18 (clean code), libs 17 (vibe coding for prompting) |
| OpenAI / LLM labs | libs 01 (OpenAI API) |
| SQL for data warehousing | libs 12 (SQL) |

## What's complete vs partial

Per `python-course.md` memory and the directory contents:
- All 19 python/ notebooks are present; the course is described as "18-module Python course: fundamentals → OOP → capstone".
- All 19 python_libs/ notebooks are present, plus supporting subfolders.
- Some notebooks may still be in active development (size doesn't always reflect completion); check head/tail when in doubt.
