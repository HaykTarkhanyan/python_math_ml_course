"""Author the Python-libraries understanding-test question bank and emit a CSV.

Library/tooling content only (python_libs/ 01-18): the data stack (NumPy, Pandas,
viz), engineering practices (logging, testing, parallelism, scraping), and backend
+ modern coding (SQL, Pydantic, FastAPI, pathlib). Pure-language content lives in
the separate python/tests/ bank. English-only. Each question is one readable
q(...) block; validation fails loudly before any CSV is written.

Run: python build_csv.py
"""

import csv
import logging
import sys
from pathlib import Path

# --- logging (console + logs/build_csv.log) ---------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "build_csv.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("build_csv")

OUT = Path(__file__).parent / "python_libs_understanding_test.csv"

FIELDS = [
    "id", "area", "module", "difficulty", "template",
    "question",
    "optA", "optB", "optC", "optD",
    "correct", "points",
    "feedback_correct", "feedback_wrong",
]

AREAS = {"data_stack", "engineering", "backend_modern"}
AREA_ORDER = ["data_stack", "engineering", "backend_modern"]
TEMPLATES = {"misconception", "predict", "flaw", "match", "compare"}
LETTERS = ["A", "B", "C", "D"]


def q(qid, area, module, difficulty, template, question,
      options, correct, fb_ok, fb_no, points=1):
    """Build one question row. options = list of 4 strings (A,B,C,D order)."""
    if len(options) != 4:
        raise ValueError(f"{qid}: need exactly 4 options, got {len(options)}")
    row = {
        "id": qid, "area": area, "module": module, "difficulty": difficulty,
        "template": template, "question": question,
        "correct": correct, "points": points,
        "feedback_correct": fb_ok, "feedback_wrong": fb_no,
    }
    for letter, opt in zip(LETTERS, options):
        row[f"opt{letter}"] = opt
    return row


# ---------------------------------------------------------------------------
# QUESTIONS
# ---------------------------------------------------------------------------
QUESTIONS = [

    # === Data stack: NumPy / Pandas / viz (python_libs/ 01-07) ===
    q("PY-D-01", "data_stack", "libs02", 3, "predict",
      "a = np.array([1, 2, 3, 4]); b = a[1:3]; b[0] = 99. What is a afterward?",
      ["[1, 2, 3, 4] (slicing makes a copy)",
       "[1, 99, 3, 4] (a NumPy slice is a view into the same data)",
       "an error",
       "[99, 2, 3, 4]"],
      "B",
      "Unlike Python lists, a NumPy slice returns a view that shares memory with the original. Writing to the view writes to a. Use a[1:3].copy() for an independent array.",
      "NumPy slicing does not copy (A) - that is list behavior. No error (C). The edited element is index 1 of a, giving [1,99,3,4] (B), not D."),

    q("PY-D-02", "data_stack", "libs02", 2, "predict",
      "a has shape (3, 1) and b has shape (1, 4). What is the shape of a + b?",
      ["an error - the shapes do not match",
       "(3, 4) via broadcasting",
       "(3, 1)",
       "(1, 4)"],
      "B",
      "Broadcasting stretches size-1 dimensions to match: (3,1) and (1,4) combine to (3,4). Dimensions are compatible when they are equal or one of them is 1.",
      "Broadcastable shapes do not error (A). The result expands both, not keeping either original (C, D). It becomes (3,4) (B)."),

    q("PY-D-03", "data_stack", "libs02", 2, "misconception",
      "Why is a vectorized NumPy operation usually far faster than an equivalent Python for-loop?",
      ["NumPy uses more CPU cores by default",
       "the loop runs in optimized compiled C over contiguous typed memory, avoiding per-element Python overhead",
       "NumPy caches the result",
       "Python loops are artificially slowed down"],
      "B",
      "Vectorized ops push the loop into compiled C over a typed, contiguous array, skipping the per-iteration interpreter overhead of a Python loop. Same work, far less overhead.",
      "It is not multicore by default (A) or caching (C). Python loops are not throttled (D); they are just interpreted per element (B)."),

    q("PY-D-04", "data_stack", "libs02", 2, "misconception",
      "For a 2D array M, what does M.sum(axis=0) compute?",
      ["the sum of every element",
       "it collapses axis 0 (down the rows), giving one value per column",
       "one value per row",
       "the sum along the diagonal"],
      "B",
      "axis=0 means 'collapse the row axis', so you sum down each column and get one number per column. axis=1 collapses columns, giving one per row.",
      "That is sum() with no axis (A). axis=0 gives a per-column result, not per-row (C). It is not a diagonal sum (D)."),

    q("PY-D-05", "data_stack", "libs03", 2, "match",
      "What is the difference between df.loc[...] and df.iloc[...]?",
      ["they are aliases",
       "loc selects by label (index/column names); iloc selects by integer position",
       "loc is by position; iloc is by label",
       "loc is for rows only; iloc is for columns only"],
      "B",
      "loc is label-based (uses index and column names, end-inclusive); iloc is integer-position-based (end-exclusive). Mixing them up is a top Pandas bug.",
      "They are not aliases (A) and C swaps them. Both can address rows and columns (D). loc=label, iloc=position (B)."),

    q("PY-D-06", "data_stack", "libs03", 3, "flaw",
      "A student writes df[df.a > 0]['b'] = 5 and is surprised df is unchanged (with a warning). What is the flaw?",
      ["the comparison df.a > 0 is wrong",
       "chained indexing may assign to a temporary copy, so the write never reaches df; use df.loc[df.a > 0, 'b'] = 5",
       "you cannot assign to a DataFrame at all",
       "column 'b' must be numeric"],
      "B",
      "Chained indexing (two [] in a row) can return a copy, so the assignment lands on that throwaway, not df - hence SettingWithCopyWarning. Do it in one step with .loc.",
      "The filter is fine (A). DataFrames are assignable (C) and dtype is not the issue (D). The bug is chained indexing hitting a copy (B)."),

    q("PY-D-07", "data_stack", "libs03", 2, "match",
      "What does df.groupby('city')['sales'].mean() do conceptually?",
      ["sorts the rows by city",
       "splits rows by city, computes the mean of sales within each group, then combines (split-apply-combine)",
       "drops the city column",
       "returns the single overall mean of sales"],
      "B",
      "groupby follows split-apply-combine: split rows into groups by the key, apply the aggregation (mean) per group, and combine into one result per city.",
      "It groups, not merely sorts (A). It does not drop columns (C) or give one overall mean (D); it is per-group (B)."),

    q("PY-D-08", "data_stack", "libs03", 2, "misconception",
      "In NumPy/Pandas, what is the result of np.nan == np.nan?",
      ["True",
       "False - NaN is not equal to anything, including itself",
       "it raises an error",
       "NaN"],
      "B",
      "By IEEE-754, NaN does not equal anything, even itself, so np.nan == np.nan is False. To detect NaN use np.isnan(x) or pd.isna(x), never ==.",
      "NaN never equals itself (A). It returns a bool, not an error (C) or NaN (D). Use isna/isnan to test (B)."),

    q("PY-D-09", "data_stack", "libs02", 2, "predict",
      "a = np.array([1, 2, 3]); b = np.array([1, 0, 3]). What does a == b give?",
      ["a single True or False",
       "an element-wise boolean array: array([True, False, True])",
       "an error",
       "the number of matching elements"],
      "B",
      "Array comparisons are element-wise and return a boolean array. To reduce to one answer use (a == b).all() or .any(); putting the array directly in an if raises 'ambiguous truth value'.",
      "It does not collapse to one bool (A) or a count (D). It does not error here (C); it returns the per-element mask (B)."),

    # === Engineering practices (python_libs/ 08-11) ===
    q("PY-E-01", "engineering", "libs08", 1, "match",
      "Why use the logging module with levels (DEBUG/INFO/WARNING/ERROR) instead of print?",
      ["print is broken in scripts",
       "levels let you filter and route messages by severity and toggle detail without deleting code",
       "logging is always faster than print",
       "print cannot display text"],
      "B",
      "Logging adds severity levels, formatting, and handlers (file + console) so you can control verbosity centrally and keep diagnostics in production - things print cannot do cleanly.",
      "print is not broken (A) and can show text (D). Logging is not chosen for raw speed (C); it is for leveled, routable, configurable output (B)."),

    q("PY-E-02", "engineering", "libs10", 3, "compare",
      "For a CPU-bound task (heavy number crunching) in CPython, which usually gives real speedup, and why?",
      ["threading, because threads always run in parallel",
       "multiprocessing, because the GIL stops threads from running Python bytecode truly in parallel",
       "neither can ever speed anything up",
       "threading, because it uses less memory"],
      "B",
      "CPython's GIL lets only one thread execute Python bytecode at a time, so threads do not speed up CPU-bound work. Separate processes sidestep the GIL. Threads still help for I/O-bound work.",
      "Threads do not run Python bytecode in parallel under the GIL (A). Processes can help CPU-bound work (C). Memory is not the deciding factor (D)."),

    q("PY-E-03", "engineering", "libs09", 2, "match",
      "What is a pytest fixture for?",
      ["asserting that a test passes",
       "providing reusable setup (and teardown) - e.g. a temp file or sample data - injected into tests",
       "skipping a test",
       "measuring code coverage"],
      "B",
      "Fixtures supply prepared state (data, connections, temp dirs) to tests and can clean up afterward, keeping tests DRY and isolated. A test requests one by naming it as an argument.",
      "Asserting uses assert (A). Skipping uses markers (C). Coverage is a separate tool (D). Fixtures are reusable setup/teardown (B)."),

    q("PY-E-04", "engineering", "libs09", 1, "misconception",
      "In pytest, how do you check an expected result inside a test?",
      ["return the value from the test function",
       "use a plain assert statement; pytest reports a helpful failure if it is False",
       "print the value and read it by eye",
       "wrap everything in try/except"],
      "B",
      "pytest uses plain assert and rewrites it to show a useful message on failure. A test passes if no assertion fails and no exception escapes; return values are ignored.",
      "Return values are ignored (A). Printing is not a check (C). You do not need try/except for normal assertions (D); just assert (B)."),

    q("PY-E-05", "engineering", "libs10", 2, "flaw",
      "You fetch a page with requests + BeautifulSoup, but the data visible in the browser is missing from what you parsed. The most likely reason?",
      ["BeautifulSoup is broken",
       "the content is rendered by JavaScript after load, which requests does not execute",
       "the site has no HTML at all",
       "you must always use multiprocessing to scrape"],
      "B",
      "requests fetches the raw HTML response; it does not run JavaScript. Content injected by JS after load is not in that HTML. Use a browser driver (Selenium/Playwright) or a backing API.",
      "BeautifulSoup parses what it is given (A). The page has HTML, just not the JS-built parts (C). Multiprocessing is unrelated (D). JS rendering is the cause (B)."),

    q("PY-E-06", "engineering", "libs10", 2, "misconception",
      "Adding multiprocessing to a small, fast task sometimes makes it slower. Why?",
      ["multiprocessing is always slower",
       "spawning processes and serializing data between them has overhead that can exceed the work saved",
       "the GIL blocks separate processes",
       "small tasks cannot be parallelized at all"],
      "B",
      "Process startup and inter-process data serialization (pickling) cost time. For small or quick tasks that overhead can dwarf the gains, so a plain loop wins. Parallelism pays off when the work per item is large.",
      "It is not always slower (A) - it helps big CPU-bound jobs. The GIL does not block separate processes (C). Small tasks can be parallelized, just often not worth it (D)."),

    # === Backend + modern coding (python_libs/ 12-18) ===
    q("PY-B-01", "backend_modern", "libs12", 2, "predict",
      "Some customers have no orders. SELECT ... FROM customers LEFT JOIN orders ... vs INNER JOIN: what is the key difference?",
      ["no difference",
       "LEFT JOIN keeps all customers (NULLs where no order); INNER JOIN keeps only customers that have orders",
       "INNER JOIN keeps all customers; LEFT JOIN drops the unmatched",
       "LEFT JOIN only works on numeric columns"],
      "B",
      "LEFT JOIN returns every left-table row, filling NULLs where there is no match. INNER JOIN returns only rows matched on both sides, dropping order-less customers.",
      "There is a real difference (A) and C reverses it. Join type does not depend on column type (D). LEFT keeps unmatched left rows (B)."),

    q("PY-B-02", "backend_modern", "libs12", 2, "misconception",
      "In SQL, what is the difference between WHERE and HAVING?",
      ["they are the same",
       "WHERE filters rows before grouping; HAVING filters groups after GROUP BY and can use aggregates",
       "HAVING runs before WHERE",
       "WHERE can use aggregate functions like COUNT(*)"],
      "B",
      "WHERE filters individual rows before aggregation; HAVING filters grouped results after GROUP BY and can reference aggregates like COUNT(*). Order: WHERE, then GROUP BY, then HAVING.",
      "They differ (A) and HAVING comes after WHERE, not before (C). WHERE cannot use aggregates (D) - that is what HAVING is for (B)."),

    q("PY-B-03", "backend_modern", "libs13", 2, "misconception",
      "You define a Pydantic model with field age: int and pass age='30' (a string). What happens by default?",
      ["it raises immediately because the type is wrong",
       "Pydantic validates and coerces '30' to the int 30",
       "it stores the string '30' unchanged",
       "the field is set to None"],
      "B",
      "Pydantic validates input against the declared types and coerces compatible values, so '30' becomes 30. Genuinely invalid input (e.g. 'abc') raises a ValidationError.",
      "Coercible input does not immediately raise (A). It does not keep the raw string (C) or null it (D); it converts to int (B)."),

    q("PY-B-04", "backend_modern", "libs15", 2, "match",
      "In FastAPI, what do you get by declaring a request body as a Pydantic model?",
      ["nothing automatic; you parse the JSON yourself",
       "automatic request parsing, type validation, and OpenAPI docs generated from the model",
       "the endpoint becomes synchronous only",
       "the model replaces the database"],
      "B",
      "FastAPI uses the Pydantic model to parse and validate the incoming JSON and to auto-generate the OpenAPI/Swagger schema. An invalid body gets a clear 422 automatically.",
      "You do not hand-parse (A). It does not force sync (C) or act as a database (D). It gives validation plus docs for free (B)."),

    q("PY-B-05", "backend_modern", "libs14", 1, "match",
      "Why prefer pathlib (Path('data') / 'file.csv') over manual string concatenation for file paths?",
      ["it is only shorter to type",
       "it builds correct paths across operating systems and offers methods like .exists(), .suffix, .parent",
       "it makes files load faster",
       "strings cannot represent paths"],
      "B",
      "pathlib handles OS-specific separators correctly and gives path-aware methods (exists, suffix, parent, glob), avoiding fragile concatenation like 'a' + '/' + 'b'.",
      "It is more than brevity (A). It does not affect load speed (C). Strings can hold paths but are error-prone across OSes (D); pathlib is safer (B)."),

    q("PY-B-06", "backend_modern", "libs12", 3, "misconception",
      "A column has some NULLs. What does COUNT(col) return, and how does col = NULL behave?",
      ["COUNT(col) counts all rows; col = NULL finds the NULLs",
       "COUNT(col) skips NULLs; and col = NULL never matches - you must use col IS NULL",
       "COUNT(col) errors on NULLs",
       "NULL equals NULL, so col = NULL works fine"],
      "B",
      "COUNT(col) counts only non-NULL values (use COUNT(*) for all rows). NULL means 'unknown', so any = comparison yields unknown, never true - you must use IS NULL / IS NOT NULL.",
      "COUNT(col) excludes NULLs, so A is wrong, and it does not error (C). NULL = NULL is not true (D); test with IS NULL (B)."),

    # === Gap-fill: Data stack ===
    q("PY-D-10", "data_stack", "libs06", 2, "match",
      "Why use fig, ax = plt.subplots() and ax.plot(...) instead of plt.plot(...)?",
      ["it is required; plt.plot does not exist",
       "the object-oriented fig/ax API gives explicit control over multiple subplots and is clearer for complex figures",
       "ax.plot is faster",
       "plt.plot cannot draw lines"],
      "B",
      "The fig/ax (object-oriented) API makes each Axes explicit, which scales cleanly to subplots and fine control. plt.* works on the 'current' axes implicitly - fine for quick one-off plots.",
      "plt.plot exists (A) and can draw lines (D). It is about clarity/control, not speed (C). The OO API is preferred for non-trivial figures (B)."),

    q("PY-D-11", "data_stack", "libs04", 2, "predict",
      "pd.merge(left, right, on='id') with no 'how' argument - which rows are kept?",
      ["all rows from both tables",
       "only ids present in BOTH tables (inner join is the default)",
       "all rows from left only",
       "it raises without an explicit how"],
      "B",
      "merge defaults to how='inner', keeping only keys found in both frames. Use how='left'/'right'/'outer' to keep unmatched rows (filled with NaN).",
      "It does not keep everything (A - that is outer) or left-only (C - that is left). It does not require how (D). Default is inner (B)."),

    q("PY-D-12", "data_stack", "libs03", 2, "misconception",
      "Your DataFrame has NaNs, yet df.mean() returns numbers. Why, and what do dropna/fillna do?",
      ["mean errors on NaN, so the numbers are wrong",
       "pandas skips NaN in aggregations by default; dropna removes missing rows/cols, fillna replaces them with a value",
       "NaNs are counted as zero in the mean",
       "you must always dropna before any operation"],
      "B",
      "Most pandas aggregations skip NaN by default (so mean ignores them, not treats them as 0). dropna drops missing entries; fillna substitutes (0, a mean, forward-fill, etc.). Choose deliberately.",
      "mean does not error (A) or treat NaN as 0 (C) - that would change the result. You do not always need dropna first (D); aggregations handle NaN (B)."),

    q("PY-D-13", "data_stack", "libs03", 2, "compare",
      "For an elementwise numeric transform on a big DataFrame column, which is usually preferred?",
      ["df['x'].apply(a python function), because apply is always fastest",
       "a vectorized op like df['x'] * 2 or np.log(df['x']), which is much faster than per-row apply",
       "a Python for-loop over the rows",
       "df.iterrows()"],
      "B",
      "Vectorized column operations run in optimized C over the whole array. apply (and especially iterrows/for-loops) call Python per element/row and are far slower. Reach for vectorization first.",
      "apply is not the fastest (A); it is per-element Python. Explicit loops (C) and iterrows (D) are the slowest. Vectorize (B)."),

    q("PY-D-14", "data_stack", "libs02", 2, "misconception",
      "a = np.array([1, 2, 3]) has dtype int64. You do a[0] = 3.7. What is a[0]?",
      ["3.7",
       "3 - the float is truncated to fit the integer dtype of the array",
       "it raises a type error",
       "the whole array becomes float"],
      "B",
      "A NumPy array has a fixed dtype. Assigning 3.7 into an int array truncates to 3; the array does not change dtype per element. Create a float array if you need decimals.",
      "It does not store 3.7 (A) or convert the array (D). It does not error (C); it silently truncates to 3 (B) - a real gotcha."),

    q("PY-D-15", "data_stack", "libs02", 3, "predict",
      "a is a NumPy array. b = a[[0, 2]] (fancy indexing with a list); b[0] = 99. Does a change?",
      ["yes, like a slice it is a view",
       "no - fancy indexing (with a list/array of indices) returns a copy, not a view",
       "it raises an error",
       "only if a is 2D"],
      "B",
      "Basic slicing (a[0:2]) returns a view; fancy indexing (a[[0,2]] or boolean masks) returns a copy. So writing to b does not affect a here. View-vs-copy depends on the index type.",
      "Fancy indexing copies, unlike a slice (A). No error (C), and it does not depend on dimensionality (D). a is unchanged (B)."),

    q("PY-D-16", "data_stack", "libs02", 1, "misconception",
      "Why call np.random.seed(509) (or use a Generator with a fixed seed) before generating random numbers?",
      ["to make the numbers more random",
       "to make results reproducible - the same seed yields the same 'random' sequence every run",
       "it makes generation faster",
       "it is required or NumPy errors"],
      "B",
      "Seeding fixes the pseudo-random sequence so runs are reproducible (key for experiments and debugging). It does not improve randomness or speed; without it NumPy just picks an unpredictable seed.",
      "It does not increase randomness (A) or speed (C), and it is optional (D). Its purpose is reproducibility (B)."),

    # === Gap-fill: Engineering practices ===
    q("PY-E-07", "engineering", "libs09", 2, "match",
      "What does @pytest.mark.parametrize do?",
      ["runs a test in parallel",
       "runs the same test function once per set of input/expected values, reporting each case separately",
       "marks a test to be skipped",
       "measures the parameters' memory use"],
      "B",
      "parametrize feeds multiple (input, expected) cases into one test function, each reported as its own pass/fail. It replaces copy-pasting near-identical tests.",
      "It is not about parallelism (A), skipping (C), or memory (D). It multiplies one test over many cases (B)."),

    q("PY-E-08", "engineering", "libs09", 2, "match",
      "How do you assert that a function raises ValueError on bad input, in pytest?",
      ["wrap it in try/except and pass if it crashes",
       "use 'with pytest.raises(ValueError):' around the call",
       "assert that the function returns an error",
       "you cannot test for exceptions"],
      "B",
      "with pytest.raises(ValueError): asserts the block raises that exception (and fails the test if it does not). Cleaner and more precise than a manual try/except.",
      "Manual try/except is clumsy and easy to get wrong (A). Functions raise, not return, exceptions (C). Exceptions are testable (D)."),

    q("PY-E-09", "engineering", "libs10", 2, "predict",
      "For a task that mostly waits on network/disk (I/O-bound), does threading help in CPython despite the GIL?",
      ["no, the GIL blocks everything",
       "yes - threads release the GIL while waiting on I/O, so they overlap the waiting time effectively",
       "only multiprocessing ever helps",
       "threading only helps CPU-bound work"],
      "B",
      "The GIL is released during blocking I/O, so threads can overlap their waits - great for I/O-bound work (many requests, file reads). The GIL only blocks parallel CPU-bound Python execution.",
      "The GIL does not block during I/O waits (A). Processes are not the only option (C). Threading helps I/O-bound, not CPU-bound, work (D - reversed)."),

    q("PY-E-10", "engineering", "libs08", 2, "misconception",
      "With a fresh logger at default settings, log.info('hi') shows nothing. Why?",
      ["info messages are broken",
       "the default level is WARNING, so INFO and DEBUG are filtered out until you lower the level",
       "you must restart Python",
       "logging requires a file to write to"],
      "B",
      "The default level is WARNING, so INFO/DEBUG are suppressed until you configure a lower level (e.g. logging.basicConfig(level=logging.INFO)). Nothing is broken; it is just filtered.",
      "INFO works once enabled (A). No restart needed (C) and no file required (D). The default WARNING threshold hides INFO (B)."),

    q("PY-E-11", "engineering", "libs08", 1, "match",
      "What problem do argparse / click / typer solve?",
      ["they speed up the program",
       "they parse command-line arguments into typed values, with help text and validation, instead of manually reading sys.argv",
       "they replace functions",
       "they are only for web servers"],
      "B",
      "CLI libraries turn command-line flags/args into structured, validated, documented inputs (--name, --count) so you do not hand-parse sys.argv. typer/click also add types and auto-generated help.",
      "They are about argument parsing, not speed (A), not replacing functions (C), and not web-only (D)."),

    q("PY-E-12", "engineering", "libs09", 2, "match",
      "In a unit test, why 'mock' an external API call?",
      ["to make the real call faster",
       "to replace the slow/unreliable external dependency with a controlled fake, so the test is fast, deterministic, and offline",
       "because real calls are illegal in tests",
       "mocking tests the external API itself"],
      "B",
      "Mocking substitutes a fake for an external dependency (network, DB, clock), so the unit test runs fast and deterministically and tests your code, not the third party. You assert how the mock was called.",
      "It does not speed up the real call (A) - it avoids it. Real calls are not illegal, just undesirable in unit tests (C). It tests your code, not the API (D)."),

    # === Gap-fill: Backend + modern coding ===
    q("PY-B-07", "backend_modern", "libs12", 2, "misconception",
      "In standard SQL, 'SELECT city, name FROM t GROUP BY city' is usually an error. Why?",
      ["you cannot GROUP BY a text column",
       "every selected column must be in GROUP BY or wrapped in an aggregate; 'name' is neither",
       "GROUP BY must come before SELECT",
       "you can only group by one column"],
      "B",
      "When you group, each output row represents a group, so non-grouped columns must be aggregated (MAX, COUNT, ...) - otherwise which 'name' would it show? Add name to GROUP BY or aggregate it.",
      "You can group by text (A). SELECT is written first regardless (C). You can group by multiple columns (D). The rule is grouped-or-aggregated (B)."),

    q("PY-B-08", "backend_modern", "libs12", 3, "match",
      "You want each row plus a rank within its category, WITHOUT collapsing rows. Which SQL tool fits?",
      ["GROUP BY category",
       "a window function: RANK() OVER (PARTITION BY category ORDER BY ...)",
       "WHERE category = ...",
       "DISTINCT"],
      "B",
      "Window functions (OVER / PARTITION BY) compute across a group of rows while keeping every row - perfect for ranks, running totals, moving averages. GROUP BY would collapse the rows.",
      "GROUP BY collapses rows (A). WHERE filters (C); DISTINCT dedups (D). To keep rows and add a per-group rank, use a window function (B)."),

    q("PY-B-09", "backend_modern", "libs13", 2, "match",
      "Beyond type coercion, what does a Pydantic field validator let you do?",
      ["nothing extra",
       "enforce custom rules (e.g. value must be positive, email must contain @) and raise on violation",
       "speed up the model",
       "connect to a database"],
      "B",
      "Validators run custom logic per field (or across fields), letting you enforce domain rules beyond type checks and raise a ValidationError when they fail. They express business constraints.",
      "They add real capability (A), not speed (C) or DB access (D). They enforce custom rules (B)."),

    q("PY-B-10", "backend_modern", "libs15", 2, "misconception",
      "When should a FastAPI endpoint be 'async def' rather than 'def'?",
      ["always - async is strictly better",
       "when it awaits non-blocking I/O (async DB/HTTP clients); a plain def is fine and runs in a threadpool for blocking work",
       "never - FastAPI does not support async",
       "only for GET requests"],
      "B",
      "Use async def when you await truly non-blocking I/O. If you call blocking libraries, a normal def (which FastAPI runs in a threadpool) avoids blocking the event loop. async is not automatically faster.",
      "async is not always better (A) - blocking code in async def stalls the loop. FastAPI fully supports async (C). It is unrelated to the HTTP method (D)."),

    q("PY-B-11", "backend_modern", "libs12", 2, "misconception",
      "Why build queries with parameters (e.g. execute('... WHERE id = ?', (uid,))) instead of f-string concatenation?",
      ["it is shorter",
       "it prevents SQL injection - user input is sent as data, not executable SQL",
       "parameters make queries run faster only",
       "f-strings do not work with SQL"],
      "B",
      "Parameterized queries pass user input as data the driver escapes, so a malicious value cannot alter the query. Formatting user input straight into SQL is the classic injection vulnerability.",
      "It is about security, not brevity (A) or speed alone (C). f-strings 'work' but are dangerous here (D). Parameterization prevents injection (B)."),

    q("PY-B-12", "backend_modern", "libs18", 2, "misconception",
      "What does the Single Responsibility Principle say a module or class should have?",
      ["only one method",
       "one reason to change - it should do one well-defined job, not mix unrelated concerns",
       "no dependencies at all",
       "exactly one attribute"],
      "B",
      "SRP: a unit should have one reason to change, i.e. one cohesive responsibility. Mixing parsing + I/O + business logic in one class makes it fragile; split concerns so each can change independently.",
      "It is not about a literal single method (A) or attribute (D), nor zero dependencies (C). It is one reason to change / one responsibility (B)."),

]


def validate(rows):
    """Fail loudly on any structural problem before writing the CSV."""
    problems = []
    seen = set()
    for r in rows:
        rid = r.get("id", "<no-id>")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        if r["area"] not in AREAS:
            problems.append(f"{rid}: unknown area '{r['area']}'")
        if r["template"] not in TEMPLATES:
            problems.append(f"{rid}: unknown template '{r['template']}'")
        if r["correct"] not in LETTERS:
            problems.append(f"{rid}: correct must be A-D, got '{r['correct']}'")
        if int(r["difficulty"]) not in (1, 2, 3):
            problems.append(f"{rid}: difficulty must be 1-3")
        if int(r["points"]) < 1:
            problems.append(f"{rid}: points must be >= 1")
        for f in FIELDS:
            if f not in r:
                problems.append(f"{rid}: missing field '{f}'")
            elif str(r[f]).strip() == "":
                problems.append(f"{rid}: empty field '{f}'")
    if problems:
        for p in problems:
            log.error(p)
        raise ValueError(f"{len(problems)} validation problem(s); CSV not written")


def main():
    QUESTIONS.sort(key=lambda r: AREA_ORDER.index(r["area"]))
    validate(QUESTIONS)

    by_area = {}
    by_template = {}
    for r in QUESTIONS:
        by_area[r["area"]] = by_area.get(r["area"], 0) + 1
        by_template[r["template"]] = by_template.get(r["template"], 0) + 1

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(QUESTIONS)

    log.info(f"Wrote {len(QUESTIONS)} questions to {OUT}")
    log.info(f"By area: {dict(sorted(by_area.items()))}")
    log.info(f"By template: {dict(sorted(by_template.items()))}")


if __name__ == "__main__":
    sys.exit(main())
