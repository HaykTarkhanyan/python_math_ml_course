# Python OOP + Tooling + Capstone — Reference (python/ 10-18)

Covers: dev tooling, exceptions, recursion, decorators, full OOP path, capstone project.

## Notebooks

- [10_git_conda_pep8.ipynb](../../python/10_git_conda_pep8.ipynb) — **Git, conda, PEP8** — Shell from notebooks, **Git + GitHub** workflow basics, conda environment management, **PEP8** style conventions.
- [11_exception_handling.ipynb](../../python/11_exception_handling.ipynb) — **Exceptions** — `try` / `except` (single + multi-except), exception type granularity, `else`, `finally`, `raise`, `assert`, defensive vs EAFP approaches.
- [12_streamlit_recursion.ipynb](../../python/12_streamlit_recursion.ipynb) — **Streamlit + recursion** — Streamlit intro (companion: `python/py_files/streamlit_app.py`), recursion (factorial, Fibonacci).
- [13_decorators.ipynb](../../python/13_decorators.ipynb) — **Decorators** — First-class functions recap, decorators (no args, with args, returning values), examples: run-twice, function slow-down (`time.sleep`), debugger decorator, real-world use cases referenced.
- [14_classes.ipynb](../../python/14_classes.ipynb) — **Classes (intro)** — List comparison motivation, `__init__`, `self`, instance methods, class methods, class variables, instance vs class scope, naming conventions (camelCase for classes).
- [15_inheritance_polymorphism.ipynb](../../python/15_inheritance_polymorphism.ipynb) — **Inheritance + polymorphism** — Single inheritance, method override, **magic methods** (`__str__`, `__repr__`, `__eq__`, `__add__`/`__sub__`/`__mul__`/`__truediv__`/`__floordiv__`/`__mod__`/`__pow__`, `__bool__`).
- [16_encapsulation_abstraction.ipynb](../../python/16_encapsulation_abstraction.ipynb) — **Encapsulation + abstraction** — Public / protected (`_x`) / private (`__x`), `@property` getters/setters, `abc.ABC` for abstract methods, abstract variables.
- [17_dataclass_iterator_generator_context_manager.ipynb](../../python/17_dataclass_iterator_generator_context_manager.ipynb) — **Dataclass, iterator, generator, context manager** — `@dataclass` (defaults, frozen), iterator protocol (`__iter__`, `__next__`), generators (`yield`, generator expressions), context managers (`__enter__`, `__exit__`, `contextlib`), timer example.
- [18_youtube_translator.ipynb](../../python/18_youtube_translator.ipynb) — **Capstone: YouTube Translator** — End-to-end project: YouTube audio download → transcription → DeepL translation. OOP-organized, multi-class architecture.

## Mini-projects (`python/mini_projects/`)

- `approximate_pi/` — Monte Carlo estimate of π (random point sampling inside unit circle).
- `telegram_data/` — analyze Telegram export data.
- `text_corpus_classes/main.ipynb` — text corpus analysis using classes (load data, compute metrics, search themes).
- `youtube_playlist/playlist_analytics.ipynb` — YouTube playlist analytics.

## Standalone `.py` files (`python/py_files/`)

- `files_helper.py` + `files_main.py` — referenced from notebook 09 file I/O section.
- `streamlit_app.py` — referenced from notebook 12 Streamlit intro.

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| Git for ML projects (commit, branch, PR) | python 10 |
| sklearn Estimator API (`fit` / `predict` / `transform`) | python 14 (class basics) + 15 (inheritance) |
| Custom transformer in sklearn Pipeline | python 14-16 (full OOP) |
| Polymorphism: many estimators share interface | python 15 |
| Abstract base for an Estimator | python 16 (`abc.ABC`) |
| `@dataclass` for ML configs | python 17 |
| Iterator pattern for batched data loading | python 17 (iterator) |
| Generators for streaming large datasets | python 17 (generator) |
| `with` block for model checkpoint files / tensorboard writer | python 17 (context manager) |
| Decorators for timing / caching / logging in training loops | python 13 |
| Exception handling in production inference | python 11 |
| Streamlit demo for ML model | python 12 |
| Capstone end-to-end project pattern | python 18 |

## Notes

- The OOP track (14-17) is the *direct* prerequisite for understanding sklearn's design and any custom estimator work in ML.
- Notebook 13 (decorators) is heavily referenced later — timing decorators show up in libs 08 (logging), debugger decorators preview pytest fixtures.
- Notebook 18 (YouTube Translator) is a working multi-file project pattern — point students to it when they ask "how do I structure my final ML project?".
- Notebook 17's context manager section pairs naturally with PyTorch's `torch.no_grad()` and similar ML idioms.
