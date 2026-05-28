"""Generate python_topics.csv: study tracker for the Python notebooks (01-18).

Matches the structure of optimization.csv: same headers, utf-8-sig encoding
so Google Sheets / Excel render correctly. Re-runnable - overwrites the file.

Run: uv run python misc/solo/make_python_csv.py
"""

import csv
import logging
from pathlib import Path


Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/make_python_csv.log"),
    ],
)
log = logging.getLogger(__name__)


OUT = Path("misc/solo/python_topics.csv")

HEADERS = ["Section", "Item", "Link", "Note", "Done", "Question / comment"]

BASE = "https://hayktarkhanyan.github.io/python_math_ml_course/python/"
UV_VIDEO = "https://www.youtube.com/watch?v=AMdG7IjgSPM"

SEC_FUND = "Python fundamentals"
SEC_TOOL = "Tooling and exception handling"
SEC_APPS = "Streamlit, recursion, decorators"
SEC_OOP = "Object-oriented programming"
SEC_PROJ = "Capstone project"

# (section, item, link, note)
ROWS = [
    # fundamentals (01-09)
    (SEC_FUND, "01 Intro", BASE + "01_Intro.html", ""),
    (SEC_FUND, "02 Conditions", BASE + "02_Conditions.html", ""),
    (SEC_FUND, "03 Strings, ranges, lists, some functions", BASE + "03_Str_Range_List_some_funcs.html", ""),
    (SEC_FUND, "04 Loops", BASE + "04_loops.html", ""),
    (SEC_FUND, "05 List/str methods, one-line if/for", BASE + "05_Lst_str_methods_one_line_if_for.html", ""),
    (SEC_FUND, "06 Tuple, set, dictionary", BASE + "06_tuple_set_dictionary.html", ""),
    (SEC_FUND, "07 Functions (part 1)", BASE + "07_Functions_1.html", ""),
    (SEC_FUND, "08 Functions (part 2)", BASE + "08_Functions_2.html", ""),
    (SEC_FUND, "09 Files, packages, terminal", BASE + "09_Files_Packages_Terminal.html", ""),

    # tooling + exception handling (10-11)
    (SEC_TOOL, "10 Git, conda, PEP8", BASE + "10_git_conda_pep8.html",
     "Quite important from a setup perspective. The notebook covers git and conda."),
    (SEC_TOOL, "uv (modern pip/conda alternative)", UV_VIDEO,
     "Fast modern Python package/project manager (drop-in replacement for pip/conda). Worth knowing - example video linked."),
    (SEC_TOOL, "Setup check", "",
     "Make sure you have VS Code, Python, and git installed on your PC."),
    (SEC_TOOL, "11 Exception handling", BASE + "11_exception_handling.html",
     "Important. No need to fully memorize when finally vs else trigger - understand the idea."),

    # streamlit, recursion, decorators (12-13)
    (SEC_APPS, "12 Streamlit, recursion", BASE + "12_streamlit_recursion.html",
     "Streamlit is good to know - we'll use it a lot. Recursion is also an important concept."),
    (SEC_APPS, "13 Decorators", BASE + "13_decorators.html",
     "More advanced, but very good to know."),

    # OOP (14-17)
    (SEC_OOP, "14 Classes", BASE + "14_Classes.html", "Mandatory."),
    (SEC_OOP, "15 Inheritance, polymorphism", BASE + "15_inheritance_polymorphism.html",
     "Inheritance will be used a lot. Polymorphism is less crucial, but good to at least watch the lecture video."),
    (SEC_OOP, "16 Encapsulation, abstraction", BASE + "16_encapsulation_abstraction.html", ""),
    (SEC_OOP, "17 Dataclass, iterator, generator, context manager", BASE + "17_dataclass_iterator_generator_context_manager.html",
     "Bonus - we don't need this depth. Just watch enough to know what dataclasses and generators are."),

    # capstone project (18)
    (SEC_PROJ, "18 YouTube translator (project)", BASE + "18_youtube_translator.html",
     "Capstone project. Ping me when you reach this stage - we'll plan the project together, then review your work."),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        for section, item, link, note in ROWS:
            w.writerow([section, item, link, note, "", ""])
    log.info(f"Wrote {OUT.resolve()} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
