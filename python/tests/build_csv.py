"""Author the pure-Python understanding-test question bank and emit a validated CSV.

Pure language only (python/ 01-18): fundamentals + OOP/tooling. Library content
(NumPy, Pandas, FastAPI, etc.) lives in the separate python_libs/tests/ bank.
English-only sibling of math/tests/build_csv.py. Each question is one readable
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

OUT = Path(__file__).parent / "python_understanding_test.csv"

FIELDS = [
    "id", "area", "module", "difficulty", "template",
    "question",
    "optA", "optB", "optC", "optD",
    "correct", "points",
    "feedback_correct", "feedback_wrong",
]

AREAS = {"fundamentals", "oop_tooling"}
AREA_ORDER = ["fundamentals", "oop_tooling"]
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

    # === Fundamentals (python/ 01-09) ===
    q("PY-F-01", "fundamentals", "07", 3, "misconception",
      "def add(x, items=[]): items.append(x); return items. You call add(1) then add(2). What comes back?",
      ["[1] then [2] (a fresh list each call)",
       "[1] then [1, 2] (the default list persists across calls)",
       "an error on the second call",
       "[2] then [1, 2]"],
      "B",
      "Default arguments are evaluated once, at definition time. The same list object is reused, so appends accumulate. Use items=None and create a new list inside the function.",
      "A assumes a fresh default per call, but the default is built once at def time (B). It does not error (C), and the order is [1] then [1,2], not D."),

    q("PY-F-02", "fundamentals", "02", 2, "misconception",
      "What is the difference between == and is in Python?",
      ["they are interchangeable",
       "== compares values; is compares object identity (same object in memory)",
       "is compares values; == compares identity",
       "is only works on numbers"],
      "B",
      "== asks 'are the values equal?'; is asks 'are these the exact same object?'. Use == for value checks and reserve is for None and other singletons.",
      "They are not interchangeable (A) and C has them backwards. is works on any object (D). Small-int/string caching can make is seem to work by accident - do not rely on it."),

    q("PY-F-03", "fundamentals", "03", 2, "predict",
      "a = [1, 2, 3]; b = a; b.append(4). What is a now?",
      ["[1, 2, 3] (b is a separate copy)",
       "[1, 2, 3, 4] (a and b refer to the same list)",
       "an error",
       "[4]"],
      "B",
      "Assignment binds another name to the same object; it does not copy. b and a are one list, so the append shows in both. Use a.copy() or list(a) for a real copy.",
      "b = a does not copy (A). No error (C). Both names point to one list, so a becomes [1,2,3,4] (B), not D."),

    q("PY-F-04", "fundamentals", "06", 1, "misconception",
      "Which of these is truthy (evaluates to True in an if)?",
      ["[] (empty list)",
       "0",
       "[0] (a list containing a zero)",
       "'' (empty string)"],
      "C",
      "Empty containers, 0, '', and None are all falsy. A non-empty list like [0] is truthy - it has one element, even though that element is 0.",
      "Empty list (A), 0 (B), and empty string (D) are all falsy. Only the non-empty [0] is truthy - emptiness, not content, decides."),

    q("PY-F-05", "fundamentals", "03", 1, "misconception",
      "s = 'hello'; s.upper(). After this line, what is s?",
      ["'HELLO' (upper modifies s in place)",
       "'hello' (strings are immutable; upper returns a new string)",
       "an error",
       "None"],
      "B",
      "Strings are immutable. s.upper() returns a NEW string and leaves s unchanged. To keep the result, write s = s.upper().",
      "Methods like upper() cannot modify a string in place (A). It returns a value, not None (D), and does not error (C)."),

    q("PY-F-06", "fundamentals", "08", 2, "predict",
      "In Python 3, what does map(str, [1, 2, 3]) return?",
      ["the list ['1', '2', '3']",
       "a lazy map iterator that yields the values once when consumed",
       "a tuple",
       "it prints the values immediately"],
      "B",
      "In Python 3, map (and filter) return lazy iterators, not lists. Wrap in list(...) to materialize, and remember they are exhausted after one pass.",
      "It is not a list (A) - that was Python 2 - nor a tuple (C). Nothing is printed (D); it computes only when iterated."),

    q("PY-F-07", "fundamentals", "02", 2, "misconception",
      "What does the expression 0 or 'hi' evaluate to?",
      ["True",
       "'hi' (or returns the first truthy operand, not a boolean)",
       "False",
       "0"],
      "B",
      "and/or return one of their operands, not a bool. 'or' returns the first truthy value: 0 is falsy, so the result is 'hi'. This is what powers the 'x or default' idiom.",
      "They return operands, not booleans (A, C). It skips the falsy 0 and returns 'hi', not 0 (D)."),

    q("PY-F-08", "fundamentals", "01", 1, "compare",
      "In Python 3, what are the result and type of 7 / 2 versus 7 // 2?",
      ["both give 3 (int)",
       "7/2 = 3.5 (float), 7//2 = 3 (int, floor division)",
       "7/2 = 3 (int), 7//2 = 3.5 (float)",
       "both give 3.5 (float)"],
      "B",
      "/ is true division and always returns a float (3.5). // is floor division, returning the floored value (3), as an int here.",
      "In Python 3, / never truncates to int (A), and C is reversed. // floors rather than giving a float (D)."),

    q("PY-F-09", "fundamentals", "06", 3, "predict",
      "a = [[1, 2], [3, 4]]; b = a.copy(); b[0].append(99). What is a[0]?",
      ["[1, 2] (the copy is fully independent)",
       "[1, 2, 99] (the copy is shallow; inner lists are shared)",
       "an error",
       "[99]"],
      "B",
      "list.copy() is a shallow copy: the outer list is new, but the inner lists are the same objects. Mutating an inner list shows in both. Use copy.deepcopy for full independence.",
      "A shallow copy does not copy nested objects (A). No error (C). The inner list is shared, so a[0] becomes [1,2,99] (B)."),

    q("PY-F-10", "fundamentals", "06", 2, "misconception",
      "Which of these can NOT be used as a dictionary key?",
      ["a string",
       "a tuple of numbers",
       "a list",
       "an integer"],
      "C",
      "Dict keys must be hashable, which in practice means immutable. Lists are mutable and unhashable, so they cannot be keys. Strings, numbers, and tuples of hashables can.",
      "Strings (A), tuples of numbers (B), and ints (D) are all hashable, valid keys. Only the mutable list (C) is unhashable."),

    # === OOP + tooling (python/ 10-18) ===
    q("PY-O-01", "oop_tooling", "14", 3, "predict",
      "A class has a class-level attribute tags = [] shared by all instances. One instance does self.tags.append('x'). What do other instances see?",
      ["only that instance has 'x'; others stay []",
       "all instances see ['x'], because the list is shared at the class level",
       "it raises an error",
       "a new per-instance list is created automatically"],
      "B",
      "A mutable class attribute is shared by every instance until one rebinds it. Appending mutates the single shared list, so all instances see it. Initialize mutable state in __init__ instead.",
      "It is not per-instance (A, D) - class attributes are shared. No error (C). The shared list now holds 'x' for everyone (B)."),

    q("PY-O-02", "oop_tooling", "15", 2, "match",
      "What is the intended difference between __str__ and __repr__?",
      ["they are identical",
       "__str__ is a readable form for users; __repr__ is an unambiguous form for developers/debugging",
       "__repr__ is for users; __str__ is for developers",
       "__str__ is required and __repr__ is forbidden"],
      "B",
      "__str__ targets end users (readable); __repr__ targets developers and should be unambiguous (ideally eval-able). If __str__ is missing, Python falls back to __repr__.",
      "They serve different audiences (A) and C swaps them. Both are allowed; __repr__ is the sensible fallback (D)."),

    q("PY-O-03", "oop_tooling", "16", 2, "misconception",
      "What does the @property decorator let you do?",
      ["make a method run twice",
       "access a method like an attribute (obj.x, not obj.x()), enabling computed or validated attributes",
       "make an attribute private",
       "turn a function into a class"],
      "B",
      "@property exposes a method as if it were a plain attribute, so you can compute or validate on access without changing the calling syntax. Add a setter to control assignment.",
      "It is not about running twice (A) or privacy (C - that is name mangling). It does not create a class (D). It is attribute-style access to a method (B)."),

    q("PY-O-04", "oop_tooling", "15", 2, "predict",
      "Class B(A) overrides method foo and does not call super(). When you call foo() on a B instance, which runs?",
      ["A.foo only",
       "B.foo only (the override replaces it unless you call super)",
       "both, A.foo then B.foo automatically",
       "it raises because of the conflict"],
      "B",
      "An override replaces the parent method for that class. The parent's version runs only if you explicitly call super().foo(); Python does not chain them automatically.",
      "The override takes precedence, so A.foo alone is wrong (A). Nothing runs both automatically (C). Overriding is normal, not an error (D)."),

    q("PY-O-05", "oop_tooling", "16", 2, "misconception",
      "What does a leading double underscore (e.g. self.__secret) actually do?",
      ["makes the attribute truly private and inaccessible",
       "triggers name mangling to _ClassName__secret, discouraging but not preventing outside access",
       "deletes the attribute after use",
       "makes it a class variable"],
      "B",
      "A double underscore triggers name mangling, mainly to avoid name clashes in subclasses. It is a convention/speed bump, not real privacy - you can still reach _ClassName__secret.",
      "Python has no truly private attributes (A). It is not deleted (C) or made class-level (D); it is renamed to discourage access (B)."),

    q("PY-O-06", "oop_tooling", "17", 2, "predict",
      "g = (x for x in range(3)). You call list(g), then list(g) again. What does the second call return?",
      ["[0, 1, 2] again",
       "[] because a generator is exhausted after one full pass",
       "an error",
       "[0, 1, 2, 0, 1, 2]"],
      "B",
      "Generators are one-shot iterators. After the first list(g) consumes it, it is exhausted and yields nothing, so the second call returns []. Rebuild the generator to iterate again.",
      "Generators do not restart (A, D). It is not an error (C); an exhausted generator simply yields nothing (B)."),

    q("PY-O-07", "oop_tooling", "13", 2, "misconception",
      "What does a decorator (@my_decorator above a function) do?",
      ["runs the function immediately at definition time",
       "wraps the function, replacing it with a new function that usually calls the original",
       "makes the function private",
       "documents the function only, with no runtime effect"],
      "B",
      "A decorator takes the function and returns a (usually wrapping) function bound to the same name, adding behavior - logging, timing, caching - around the original call.",
      "It does not call the function at def time (A). It is not about privacy (C) and is not a no-op (D); it actively replaces the function (B)."),

    q("PY-O-08", "oop_tooling", "11", 2, "misconception",
      "In try/except/finally, when does the finally block run?",
      ["only if no exception was raised",
       "only if an exception was raised",
       "always - whether or not an exception occurred, even if the try block returns",
       "only if you call it manually"],
      "C",
      "finally always runs - on success, on a handled or unhandled exception, even when the try block returns. It is for cleanup that must happen no matter what.",
      "It is not conditional on success (A) or failure (B), and it runs automatically, not manually (D). 'Always' is the point (C)."),

    q("PY-O-09", "oop_tooling", "17", 2, "match",
      "What does a with block (context manager) guarantee?",
      ["the code runs faster",
       "the resource's cleanup (__exit__) runs even if an exception occurs inside the block",
       "the block runs in parallel",
       "exceptions inside are silently ignored"],
      "B",
      "A context manager runs __enter__ on entry and guarantees __exit__ on exit, even if the body raises. That is why 'with open(...)' reliably closes the file.",
      "It is not about speed (A) or parallelism (C). It does not swallow exceptions by default (D); it guarantees cleanup (B)."),

    # === Gap-fill: Fundamentals ===
    q("PY-F-11", "fundamentals", "05", 2, "predict",
      "What does [x*x for x in range(4)] produce?",
      ["a lazy generator object",
       "[0, 1, 4, 9] (a list, built eagerly)",
       "{0, 1, 4, 9} (a set)",
       "it modifies range in place"],
      "B",
      "A list comprehension eagerly builds and returns a new list: [0,1,4,9]. Use (...) for a lazy generator or {...} for a set/dict comprehension.",
      "Square brackets build a list now, not a lazy generator (A - that needs ()), and not a set (C - needs {}). It does not mutate range (D)."),

    q("PY-F-12", "fundamentals", "06", 1, "misconception",
      "What is set([1, 2, 2, 3, 3, 3])?",
      ["[1, 2, 3] (a list)",
       "{1, 2, 3} (a set with duplicates removed, unordered)",
       "{1, 2, 2, 3, 3, 3}",
       "an error because of the duplicates"],
      "B",
      "A set stores unique, unordered elements, so duplicates collapse to {1, 2, 3}. Sets are ideal for membership tests and deduplication.",
      "It returns a set, not a list (A). Duplicates are removed, so not C. Duplicate input is fine, no error (D)."),

    q("PY-F-13", "fundamentals", "07", 2, "misconception",
      "In def f(*args, **kwargs):, what are args and kwargs?",
      ["both are lists",
       "args is a tuple of positional arguments; kwargs is a dict of keyword arguments",
       "args is a dict; kwargs is a tuple",
       "they are reserved keywords you cannot rename"],
      "B",
      "*args collects extra positional args into a tuple; **kwargs collects extra keyword args into a dict. The names are convention - the * and ** are what matter.",
      "args is a tuple, not a list (A), and C swaps them. The names are conventional, not reserved (D)."),

    q("PY-F-14", "fundamentals", "05", 2, "predict",
      "For s = 'abcde', what is s[::-1]?",
      ["'abcde'",
       "'edcba' (a reversed copy via step -1)",
       "an error",
       "'e'"],
      "B",
      "The slice [start:stop:step] with step -1 walks backward, producing a reversed copy: 'edcba'. Slicing never mutates; it returns a new sequence.",
      "Step -1 reverses, so not unchanged (A). It is valid syntax, no error (C). It is the whole reversed string, not one char (D)."),

    q("PY-F-15", "fundamentals", "06", 3, "predict",
      "t = ([1, 2], 3); t[0].append(99). What happens?",
      ["an error - tuples are immutable",
       "it works: t becomes ([1, 2, 99], 3) - the tuple is fixed but its list element is mutable",
       "t becomes ([1, 2], 3, 99)",
       "it silently does nothing"],
      "B",
      "A tuple's immutability means you cannot reassign its slots, but if a slot holds a mutable object (a list), you can still mutate that object. So t[0].append works.",
      "Mutating the contained list is allowed (A); only t[0] = ... would fail. The append goes into the inner list, not the tuple (C), and it does take effect (D)."),

    q("PY-F-16", "fundamentals", "07", 2, "predict",
      "def add_item(lst): lst.append(1). You call add_item(my) on your list my = []. What is my afterward?",
      ["[] (the function received a copy)",
       "[1] (lists are passed by reference; the function mutated your list)",
       "an error",
       "None"],
      "B",
      "Python passes object references. The function received the same list object, so appending inside changes your list. To avoid this, copy inside or pass lst[:].",
      "It is not a copy (A). No error (C). The function returns None, but my itself becomes [1] (D confuses the return value with the list)."),

    q("PY-F-17", "fundamentals", "06", 2, "predict",
      "d = {'a': 1}. What is the difference between d['b'] and d.get('b')?",
      ["both return None",
       "d['b'] raises KeyError; d.get('b') returns None (or a default you pass)",
       "both raise KeyError",
       "both return 0"],
      "B",
      "Indexing a missing key raises KeyError; .get returns None, or .get('b', default) returns your fallback. Use .get when a missing key is expected.",
      "d['b'] does not return None (A) - it raises. .get does not raise (C). Neither defaults to 0 unless asked (D)."),

    q("PY-F-18", "fundamentals", "08", 2, "match",
      "To sort a list of (name, age) tuples by age, what do you pass to sorted?",
      ["sorted(data, reverse=True)",
       "sorted(data, key=lambda t: t[1])",
       "sorted(data, key=t[1])",
       "sorted(data.age)"],
      "B",
      "key takes a function applied to each element to produce the sort value. lambda t: t[1] extracts the age, and the list is ordered by it.",
      "reverse alone does not pick a field (A). key needs a function, not a bare expression (C). Tuples have no .age attribute (D)."),

    q("PY-F-19", "fundamentals", "05", 1, "match",
      "You want both the index and the value while looping over a list. What is the idiomatic tool?",
      ["range(len(lst)) and index manually",
       "for i, v in enumerate(lst)",
       "zip(lst, lst)",
       "lst.items()"],
      "B",
      "enumerate(lst) yields (index, value) pairs, cleaner than indexing with range(len(...)). zip pairs multiple iterables; .items() is for dicts, not lists.",
      "range(len(...)) works but is not idiomatic (A). zip(lst, lst) pairs the list with itself (C). Lists have no .items() (D)."),

    q("PY-F-20", "fundamentals", "06", 2, "compare",
      "For many 'x in collection' lookups on large data, which is faster - a set or a list - and why?",
      ["list, because it preserves order",
       "set, because membership is O(1) average (hashing) vs O(n) scan for a list",
       "they are identical",
       "list, because sets cannot do 'in'"],
      "B",
      "Sets (and dict keys) use hashing for average O(1) membership; a list must scan element by element, O(n). For many lookups, convert to a set.",
      "Order does not help lookup speed (A). They are not identical for large n (C). Lists do support 'in', just slowly (D)."),

    q("PY-F-21", "fundamentals", "08", 3, "predict",
      "funcs = [lambda: i for i in range(3)]. What does [f() for f in funcs] return?",
      ["[0, 1, 2]",
       "[2, 2, 2] - the lambdas capture the variable i, which is 2 after the loop ends",
       "an error",
       "[0, 0, 0]"],
      "B",
      "Closures capture the variable, not its value at creation time. By the time the lambdas run, i is 2, so all return 2. Fix with a default arg: lambda i=i: i.",
      "They do not capture per-iteration values (A, D). No error (C). All three see the final i = 2 (B)."),

    q("PY-F-22", "fundamentals", "03", 2, "predict",
      "In Python 3, does range(1000000) build a list of a million ints?",
      ["yes, it creates the full list immediately",
       "no, range is a lazy sequence that computes values on demand, using O(1) memory",
       "it raises a MemoryError",
       "it returns a generator that is exhausted after one pass"],
      "B",
      "range is a lazy, reusable sequence: it stores start/stop/step and computes values as needed, so it uses constant memory regardless of size. Wrap in list() only if you need all values.",
      "It does not build a list (A) or run out of memory (C). Unlike a generator, range is reusable, not one-shot (D)."),

    # === Gap-fill: OOP + tooling ===
    q("PY-O-10", "oop_tooling", "17", 2, "match",
      "What does @dataclass give you for a class?",
      ["it makes the class abstract",
       "it auto-generates __init__, __repr__, and __eq__ from the declared fields",
       "it makes all fields private",
       "it prevents inheritance"],
      "B",
      "@dataclass synthesizes boilerplate - __init__, __repr__, __eq__ and more - from typed field declarations, so you write less plumbing. frozen=True makes instances immutable.",
      "It is not about abstractness (A), privacy (C), or blocking inheritance (D). It generates the boilerplate methods (B)."),

    q("PY-O-11", "oop_tooling", "17", 2, "misconception",
      "What is the difference between an iterable and an iterator?",
      ["they are the same",
       "an iterable can produce an iterator (has __iter__); an iterator yields values one at a time (has __next__) and is consumed as it goes",
       "an iterator can be reused infinitely; an iterable cannot be looped",
       "only lists are iterable"],
      "B",
      "An iterable (list, str, dict) hands out an iterator via __iter__. The iterator does the stepping via __next__ and is exhausted as consumed. A list is iterable but is not its own iterator.",
      "They differ (A) and C is backwards (iterators exhaust). Many types are iterable, not just lists (D)."),

    q("PY-O-12", "oop_tooling", "12", 2, "misconception",
      "What is the most common cause of a RecursionError (infinite recursion)?",
      ["using recursion at all",
       "a missing or unreachable base case, so the function never stops calling itself",
       "returning a value too early",
       "Python does not support recursion"],
      "B",
      "Every recursion needs a base case that stops the descent. Without one (or if it is never reached), calls stack until Python hits its recursion limit and raises RecursionError.",
      "Recursion itself is fine (A). Returning early ends recursion safely (C). Python supports recursion, with a depth limit (D)."),

    q("PY-O-13", "oop_tooling", "11", 2, "misconception",
      "Why is 'except Exception: pass' (a bare catch-all that ignores the error) usually bad practice?",
      ["it is a syntax error",
       "it silently swallows all errors, including bugs you did not anticipate, hiding real problems",
       "it slows the program down",
       "you can only catch one exception ever"],
      "B",
      "Catching everything and ignoring it hides genuine bugs (typos, logic errors) along with the one you expected. Catch the specific exception you can handle, and let the rest surface.",
      "It is valid syntax (A) and not a speed issue (C). You can catch multiple types (D). The problem is masking real errors (B)."),

    q("PY-O-14", "oop_tooling", "16", 2, "match",
      "What does inheriting from abc.ABC with an @abstractmethod do?",
      ["nothing at runtime",
       "it forbids instantiating the base class and forces subclasses to implement the abstract methods",
       "it makes the class faster",
       "it automatically implements the methods for you"],
      "B",
      "An abstract base class cannot be instantiated directly, and any concrete subclass must override every @abstractmethod or it too stays abstract. This enforces an interface contract.",
      "It has real runtime effect (A). It does not implement methods for you (D) or affect speed (C); it enforces the contract (B)."),

    q("PY-O-15", "oop_tooling", "14", 2, "misconception",
      "What is the difference between @classmethod and @staticmethod?",
      ["they are the same",
       "a classmethod receives the class (cls) as its first arg; a staticmethod receives neither self nor cls",
       "a staticmethod receives self; a classmethod receives nothing",
       "both receive self"],
      "B",
      "@classmethod gets cls (good for alternative constructors); @staticmethod gets no implicit first argument (just a function grouped under the class). Neither gets an instance.",
      "They differ (A) and C is backwards. A regular method gets self; these do not (D). classmethod=cls, staticmethod=nothing (B)."),

    q("PY-O-16", "oop_tooling", "15", 2, "match",
      "You define __len__ on your class. What does that enable?",
      ["nothing; __len__ is ignored",
       "calling len(obj) on instances returns whatever __len__ returns",
       "it makes the object iterable",
       "it defines addition with +"],
      "B",
      "Dunder methods hook into built-ins: __len__ backs len(obj), __eq__ powers ==, __add__ powers +, __iter__ makes it iterable. They let your objects behave like built-ins.",
      "It is not ignored (A). Iteration is __iter__ (C); addition is __add__ (D). __len__ specifically backs len() (B)."),

    q("PY-O-17", "oop_tooling", "15", 3, "predict",
      "In diamond inheritance D(B, C) where both B and C inherit from A, what decides which method runs when D does not override it?",
      ["a random choice",
       "the Method Resolution Order (MRO) - a deterministic linearization, visible via D.__mro__",
       "always A's version",
       "it raises because of ambiguity"],
      "B",
      "Python resolves attributes along the MRO, a deterministic C3 linearization (left-to-right, depth-considered). super() follows this chain, not just the literal parent. Check ClassName.__mro__.",
      "It is not random (A) or always the grandparent (C), and it does not error (D). The MRO decides (B)."),

    q("PY-O-18", "oop_tooling", "16", 2, "predict",
      "A class exposes value via @property with a setter that raises ValueError for negatives. What happens on obj.value = -5?",
      ["it silently stores -5",
       "the setter runs and raises ValueError, rejecting the assignment",
       "it bypasses the setter",
       "properties cannot validate"],
      "B",
      "Assigning to a property routes through its setter, so the validation runs and can reject bad input by raising. This is the Pythonic way to guard attribute writes while keeping obj.value syntax.",
      "The setter is not bypassed (A, C). Properties are exactly how you validate attribute access (D). The assignment raises (B)."),

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
