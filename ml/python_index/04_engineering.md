# Engineering Practices — Reference (python_libs/ 08-11 + subprojects)

Covers: production-grade Python — logging, CLIs, testing, debugging, scraping, parallelization. Directly applicable to ML pipelines.

## Notebooks

- [08_logging__clis.ipynb](../../python_libs/08_logging__clis.ipynb) — **Logging + CLIs** — First logger + levels (DEBUG / INFO / WARNING / ERROR / CRITICAL), formatters, file + stream handlers, **CLI frameworks**: argparse, click, fire, typer (each with own demo file in `python_libs/clis/`).
- [09_testing__debugging.ipynb](../../python_libs/09_testing__debugging.ipynb) — **Testing + debugging** — pytest (test discovery, assertions, fixtures, parametrize), debugging (breakpoint, pdb), companion files in `python_libs/testing/`.
- [10_scraping__parallelization.ipynb](../../python_libs/10_scraping__parallelization.ipynb) — **Scraping + parallelization** — `requests`, `BeautifulSoup`, HTML parsing, **multiprocessing** and **threading** (when to use each), pool patterns.
- [11_ysu_scraping.ipynb](../../python_libs/11_ysu_scraping.ipynb) — **YSU scraping project** — Applied scraping of Yerevan State University staff data, person-by-person HTML parsing, year-of-birth extraction.

## Sub-folders (companion source for above notebooks)

### `python_libs/clis/` — CLI library comparison

- `08_02_argparse.py` — argparse demo
- `08_02_click.py` — click demo (executable)
- `08_02_fire.py` — fire demo
- `08_02_typer.py` — typer demo
- `commands.sh` — shell commands to run each
- `test_dir/` — sample data for CLI tests

### `python_libs/testing/` — pytest examples

- `calculator.py` + `test_calculator.py` — basic unit tests
- `elections.py` + `test_elections.py` — more realistic case, fixtures, teardown
- `teardown_elections.csv`, `teardown_elections_1.csv` — sample data
- `commands.sh` — pytest invocations

### `python_libs/unittest/` — older unittest examples (deprecated in favor of pytest)

### `python_libs/scraping/` — YSU + Math/Mech scraping

- `utils.py` + `test_utils.py` — scraping helpers + tests (pairs with libs 09 testing patterns)
- `for_lecture/` — lecture-time examples
- `htmls_per_person/` — saved HTML per person (cached pages)
- `htmls_ysu/` — saved YSU pages
- `math_mech_staff.csv`, `math_mech_staff.html` — Math/Mech faculty scrape outputs
- `sample_page.html` — small example
- `ysu_project/` — full YSU project (matches notebook 11)
- `in_class.ipynb` — live-coded scraping notebook

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| Logging training runs (per-epoch loss, metrics) | libs 08 (logging) |
| Saving training logs to file (per global CLAUDE.md convention) | libs 08 |
| CLI for `train.py` / `predict.py` | libs 08 (CLI frameworks) |
| Unit tests for preprocessing functions | libs 09 (pytest) |
| Tests for custom estimator (`fit`, `predict` behavior) | libs 09 |
| Debugging `nan` losses, exploding gradients | libs 09 (breakpoint, pdb) |
| Web scraping datasets for ML | libs 10-11 |
| Parallel data preprocessing on multicore | libs 10 (multiprocessing) |
| When **not** to use multiprocessing on this laptop | libs 10 + global CLAUDE.md (16 GB / Iris Xe) |
| Caching scraped HTML (avoid re-scraping during dev) | libs 10-11 (htmls_per_person pattern) |
| Reproducibility: log seed, log config, log results | libs 08 (logging) + libs 13 (Pydantic for config — see libs index 05) |

## Notes

- Libs 08 (logging) — the global CLAUDE.md mandates `logging` over `print` and a `logs/` directory for any script. The notebook is the canonical reference for that convention.
- Libs 09 (pytest) — pairs with the `pytest` config in the project (`python_libs/testing/`). When porting ML training code, follow this pattern: code in `*.py`, tests next to it as `test_*.py`.
- Libs 10 (parallelization) — Important caveat per global CLAUDE.md: this 16 GB Iris Xe laptop locks up under multi-core. Teach `multiprocessing` but **discourage** running heavy parallel work locally.
- Libs 11 (YSU scraping) — has fully cached HTML; safe to re-run for demos without re-hitting the YSU server.
- The `scraping/utils.py` + `test_utils.py` pattern in libs 09 — has its own paired test — is a small canonical example for teaching "library code + tests" structure.
