# Python Fundamentals — Reference (python/ 01-09)

Covers: syntax, control flow, data structures, functions, files, modules.
All notebooks are bilingual (Armenian section headings, mixed Armenian + English explanations).

## Notebooks

- [01_Intro.ipynb](../../python/01_Intro.ipynb) — **Intro** — `print`, variables and naming (with reserved-name notes), arithmetic operators, order of operations, short-form assignment (`+=`, `*=`), tuple unpacking (`a, b = b, a`).
- [02_conditions.ipynb](../../python/02_conditions.ipynb) — **Conditions** — Comparison operators, booleans (truthy/falsy, `bool()`), `if` / `elif` / `else`, nested ifs, `pass`, why `elif` over chained `if`, Python 3.10 `match` statement.
- [03_Str_Range_List_some_funcs.ipynb](../../python/03_Str_Range_List_some_funcs.ipynb) — **Strings, range, list, common funcs** — String creation + escape, concatenation, multiplication, indexing + negative indices, immutability, list creation, list arithmetic, common functions on lists and numbers, `math` module intro.
- [04_loops.ipynb](../../python/04_loops.ipynb) — **Loops** — `for` over lists/strings/range, accumulators, `break`, `continue`, `while`, `while-else`, mini-games (tic-tac-toe, rock-paper-scissors).
- [05_Lst_str_methods_one_line_if_for.ipynb](../../python/05_Lst_str_methods_one_line_if_for.ipynb) — **List/str methods + one-liners** — String methods (upper/lower, startswith/endswith, split/join, count/find/replace, strip), list methods (sort/reverse/insert/remove/index), **list comprehensions** (squares, matrix, flattening), one-line if-for patterns.
- [06_tuple_set_dictionary.ipynb](../../python/06_tuple_set_dictionary.ipynb) — **Tuples, sets, dicts** — Tuple creation + immutability + advantages, set creation + operations (union, intersection, difference, symmetric difference), dict basics + `keys`/`values`/`items`/`get`/`update`, fruit-counting example.
- [07_Functions_1.ipynb](../../python/07_Functions_1.ipynb) — **Functions 1** — Motivation, function definition, no-arg vs single-arg vs multi-arg, return values, default arguments, `*args`, `**kwargs`, argument order rules, `zip`. Closes with a TODO for recursion (covered in notebook 12).
- [08_Functions_2.ipynb](../../python/08_Functions_2.ipynb) — **Functions 2** — Scope rules (LEGB), `lambda` (single + multi-arg), returning multiple values, functions as arguments / first-class objects, `map` / `filter` / `reduce` from `functools`.
- [09_files_packages_terminal.ipynb](../../python/09_files_packages_terminal.ipynb) — **Files, packages, terminal** — Calling shell from notebooks (`!`), modules/packages, `time` and `datetime`, file I/O (read/write/append, `with` context manager), **JSON** (load/dump).

## Use when teaching (ML hooks)

| ML concept | Pull from |
|---|---|
| Loading CSV data in any lab | python 09 (file I/O), libs 03 (Pandas read_csv) |
| Implementing GD by hand (loop + accumulator) | python 04 (loops), python 07 (functions) |
| Vectorization comparison vs Python loop | python 04 (slow loop baseline) → libs 02 (NumPy fast) |
| `lambda x: x[0]` patterns in sorting / sklearn | python 08 |
| Comprehensions for feature engineering | python 05 |
| JSON serialization of model configs | python 09 (JSON section) |
| Dict of hyperparameters | python 06 |
| `*args` / `**kwargs` in custom estimator wrappers | python 07 |
| `map`/`filter`/`reduce` vs sklearn `Pipeline` | python 08 |

## Notes

- Notebooks 03-08 are the longest and densest — they're the heart of "you can now write Python".
- Notebook 04 mini-games (tic-tac-toe, rock-paper-scissors) are good predict-first material — students implement them as homework.
- Notebook 07 ends with a `# RECURSIA TODO` marker pointing to notebook 12 — recursion is covered there alongside Streamlit.
- Naming: `03_Str_Range_List_some_funcs` and `05_Lst_str_methods_one_line_if_for` are awkward filenames; left as-is. When linking from ML decks, use clean display titles.
