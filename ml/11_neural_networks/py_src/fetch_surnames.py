"""Data prep for the Armenian Name Inventor practical (47_name_inventor_solution.ipynb).

Fetches every page title from the Armenian Wikipedia category
"Կատեգորիա:Հայկական ազգանուններ" (Armenian surnames), cleans it, and writes
ml/11_neural_networks/data/surnames_hy.txt (one lowercased surname per line, sorted),
so the notebook runs from a pinned file with no network dependency.

Cleaning:
  - keep only titles made purely of Armenian letters (drops disambiguated titles like
    "X (ազգանուն)", Latin/Cyrillic strays, multi-word titles);
  - lowercase (so the model vocabulary is not doubled by capitals);
  - dedupe + sort (deterministic output).

Run with the project venv:
    ./ma/Scripts/python.exe ml/11_neural_networks/py_src/fetch_surnames.py
"""

import json
import logging
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
CH_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
DATA_DIR = CH_DIR / "data"
LOGS_DIR = REPO_ROOT / "logs"

API = "https://hy.wikipedia.org/w/api.php"
CATEGORY = "Կատեգորիա:Հայկական ազգանուններ"
OUT_FILE = DATA_DIR / "surnames_hy.txt"
USER_AGENT = "python-math-ml-course data prep (github.com/HaykTarkhanyan/python_math_ml_course)"

# Armenian letters: uppercase U+0531-U+0556, lowercase U+0561-U+0587 (incl. the և ligature).
ARMENIAN_ONLY = re.compile(r"^[Ա-Ֆա-և]+$")


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("fetch_surnames")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(LOGS_DIR / "fetch_surnames.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def fetch_category_titles(log: logging.Logger) -> list[str]:
    titles: list[str] = []
    cont = ""
    while True:
        url = (
            f"{API}?action=query&list=categorymembers"
            f"&cmtitle={urllib.parse.quote(CATEGORY)}"
            f"&cmlimit=500&cmtype=page&format=json{cont}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        batch = [m["title"] for m in data["query"]["categorymembers"]]
        titles.extend(batch)
        log.info(f"fetched batch of {len(batch)} (total {len(titles)})")
        if "continue" in data:
            cont = "&cmcontinue=" + urllib.parse.quote(data["continue"]["cmcontinue"])
        else:
            break
    if not titles:
        raise RuntimeError(f"category {CATEGORY!r} returned zero pages - API change or typo?")
    return titles


def clean(titles: list[str], log: logging.Logger) -> list[str]:
    kept, dropped = [], []
    for t in titles:
        if ARMENIAN_ONLY.match(t):
            kept.append(t.lower())
        else:
            dropped.append(t)
    log.info(f"kept {len(kept)} / {len(titles)}; dropped {len(dropped)} non-pure-Armenian titles")
    if dropped:
        log.info(f"dropped examples: {dropped[:10]}")
    names = sorted(set(kept))
    log.info(f"after dedupe: {len(names)} names")
    yan = sum(n.endswith("յան") for n in names)
    lengths = [len(n) for n in names]
    log.info(f"-յան endings: {yan}/{len(names)} ({yan / len(names):.1%})")
    log.info(f"length min/mean/max: {min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}")
    return names


def main() -> None:
    log = setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    titles = fetch_category_titles(log)
    names = clean(titles, log)
    OUT_FILE.write_text("\n".join(names) + "\n", encoding="utf-8")
    log.info(f"wrote {len(names)} surnames to {OUT_FILE}")


if __name__ == "__main__":
    main()
