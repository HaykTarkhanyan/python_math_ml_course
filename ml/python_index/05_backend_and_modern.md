# Backend + Modern Coding — Reference (python_libs/ 12-18 + apis/, dbs/)

Covers: SQL, Pydantic, FastAPI, Supabase, vibe coding, clean code / architecture. The "deploy your ML model" half of the course.

## Notebooks

- [12_sql.ipynb](../../python_libs/12_sql.ipynb) — **SQL** — `SELECT` basics, `WHERE`, `JOIN` (inner / left / right / full), `GROUP BY` + aggregates, ordering, ranking. Run against the sample DB in `python_libs/dbs/`.
- [13_pydantic.ipynb](../../python_libs/13_pydantic.ipynb) — **Pydantic** — Models, field defaults, optional fields, **custom validators**, nested models, model_config, model_dump / model_validate.
- [14_misc_libraries.ipynb](../../python_libs/14_misc_libraries.ipynb) — **Misc Python libraries** — Useful tools not big enough for their own notebook (`tqdm`, `rich`, `loguru`, `pathlib`, etc.). Treat as a "Python power tools" reference.
- [15_fast_api.ipynb](../../python_libs/15_fast_api.ipynb) — **FastAPI** — Route definitions, request/response models with Pydantic, path + query params, body, dependencies, async, OpenAPI docs auto-generation. Companion app: `python_libs/apis/fastapi_shawarma.py`.
- [16_dbs_supabase.ipynb](../../python_libs/16_dbs_supabase.ipynb) — **Databases + Supabase** — Connecting to Postgres (Supabase), CRUD, auth, real-time channels (mentioned).
- [17_vibe_coding.ipynb](../../python_libs/17_vibe_coding.ipynb) — **Vibe coding** — AI-assisted development workflows, prompting patterns, using Claude / Copilot / Cursor effectively for Python work.
- [18_clean_code_architecture.ipynb](../../python_libs/18_clean_code_architecture.ipynb) — **Clean code + architecture** — SOLID-flavored principles, separation of concerns, module structure, when to refactor.

## Sub-folders (companion source)

### `python_libs/apis/` — API server examples (pairs with libs 15)

- `fastapi_shawarma.py` — FastAPI shawarma menu / order API
- `flask_shawarma.py` — Flask version for comparison
- `shaurma_example.py`, `shurma_app.py` — additional variants
- `requirements.txt` — needed packages
- `README.md` — how to run each
- `example.py` — minimal sample

### `python_libs/dbs/` — SQL sample data (pairs with libs 12)

- `customers.csv` — sample customers table
- `orders.csv` — sample orders table (joins to customers)
- `products.csv` — sample products table
- `generate_sample_data.py` — script that produced the CSVs
- `README.md` — schema description

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| MLOps W28-30: serving a trained model behind an API | libs 15 (FastAPI), apis/ examples |
| Pydantic for training config validation | libs 13 |
| Pydantic for LLM structured output (OpenAI tools) | libs 13 + libs 01 |
| Reading training data from SQL warehouse | libs 12 |
| Feature stores backed by Postgres | libs 12 + libs 16 (Supabase) |
| Storing model predictions + metadata in a DB | libs 16 |
| Logging predictions + user feedback for active learning | libs 16 |
| Clean architecture for ML pipelines (`data/` `models/` `training/` `serving/`) | libs 18 |
| Using AI tools to scaffold model code | libs 17 |
| `tqdm` progress bars during training | libs 14 |
| `pathlib` for cross-platform paths in ML scripts | libs 14 |
| `loguru` vs stdlib `logging` | libs 14 (mentions) vs libs 08 (stdlib) |

## Notes

- Libs 13 (Pydantic) is the bridge to FastAPI — Pydantic models double as request/response schemas in libs 15.
- Libs 12 (SQL) uses the sample DB in `python_libs/dbs/`. When teaching joins or aggregations in ML context (e.g., joining clicks + impressions), use this sample data first before pointing students at real datasets.
- Libs 16 (Supabase) is the practical hosted-Postgres + auth option for student projects. Good for capstones that need persistence without infrastructure overhead.
- Libs 17 (vibe coding) sets expectations for AI-assisted workflows — refer here when discussing Copilot / Cursor / Claude with students rather than re-explaining.
- Libs 18 (clean code / architecture) is the closest thing the course has to a software engineering capstone — useful pre-read before students structure their ML final projects.
- For ML model serving labs: start with `python_libs/apis/fastapi_shawarma.py` as a baseline. Students replace the menu/orders endpoint with their `predict` endpoint.
