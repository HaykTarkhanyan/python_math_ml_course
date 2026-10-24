"""Fetch and clean the Armenian text for the seq2seq project's Task B (spaces), once.

Writes ml/13_rnns/data/hy_sentences.tsv (split, article id, sentence) and hy_sentences_meta.json
(article titles, fetch date, counts). Students never download anything: the notebook reads the
shipped file.

Source: Armenian Wikipedia through the MediaWiki API. Two kinds of articles:
  - FUNNY: a hand-picked list (food, Kargin Haghordum, Vardavar...). These go to the TEST split,
    so the sentences the notebook shows off were never trained on (instructor, 2026-10-24: "pick
    funny cases").
  - RANDOM: random main-namespace articles, long ones only (most short ones are bot stubs whose
    sentences repeat a template). 10% of them also go to TEST.
Cleaning: lowercase; Armenian in-word marks removed; a sentence is kept only if every word is
made of Armenian letters (no digits, Latin, hyphens), it has 4+ words, and it was not seen before.

Run (network; measured 2026-10-24: 1181 s, the API answered HTTP 429 about once a minute and
the script waited ~45 s each time, as told):
    ./ma/Scripts/python.exe ml/13_rnns/py_src/build_seq2seq_data.py
"""
import json
import logging
import random
import re
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tqdm import tqdm

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
OUT_DIR = CH / "data"
OUT_TSV, OUT_META = OUT_DIR / "hy_sentences.tsv", OUT_DIR / "hy_sentences_meta.json"
LOGS_DIR = REPO_ROOT / "logs"

API = "https://hy.wikipedia.org/w/api.php"
UA = "python-math-ml-course/seq2seq-data (github.com/HaykTarkhanyan/python_math_ml_course)"
SEED = 509
TARGET_BYTES = 900_000          # stop collecting random articles at ~0.9 MB of sentences
MIN_ARTICLE_BYTES = 8_000       # wikitext size; shorter random articles are mostly bot stubs
MAX_SENT_PER_ARTICLE = 80       # so one huge article cannot dominate
MIN_FUNNY = 8                   # fail loudly if fewer funny articles exist
TEST_SHARE = 0.10
PAUSE, RETRIES = 1.0, 5         # seconds between calls; the first run at 0.15 s got HTTP 429

FUNNY = ["Կարգին հաղորդում", "Խաշ", "Լավաշ", "Դոլմա", "Խորոված", "Գաթա", "Ժենգյալով հաց",
         "Հարիսա", "Մածուն", "Թան", "Սուջուխ", "Բաստուրմա", "Ղափամա", "Վարդավառ", "Էշ",
         "Կատու", "Ծիրան", "Մանթի", "Թթի օղի", "Ճուտ"]

ARM_LETTERS = set(chr(c) for c in range(0x0561, 0x0588))   # ա ... ֆ and the ligature և
IN_WORD_MARKS = re.compile("[՛՜՞­]")   # ՛ ՜ ՞ and the soft hyphen
SENT_SPLIT = re.compile(r"[։:.!?;…\n]+")               # ։ is the Armenian full stop
STRIP = ",«»()[]\"'`-–—՝՚"                          # ՝ Armenian comma, ՚ apostrophe


def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("build_seq2seq_data")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "build_seq2seq_data.log",
                                                           encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def api(**params):
    """One API call, ~1 per second. On 429/503 wait as told (Retry-After) and retry, logged;
    after RETRIES failures the error is raised."""
    params |= {"format": "json", "formatversion": 2}
    req = Request(f"{API}?{urlencode(params)}", headers={"User-Agent": UA})
    for attempt in range(RETRIES + 1):
        try:
            with urlopen(req, timeout=30) as r:
                data = json.load(r)
            break
        except HTTPError as e:
            if e.code not in (429, 503) or attempt == RETRIES:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
            logging.getLogger("build_seq2seq_data").warning(
                f"HTTP {e.code}, retry {attempt + 1}/{RETRIES} in {wait} s")
            time.sleep(wait)
    if "error" in data:
        raise RuntimeError(f"MediaWiki API error: {data['error']}")
    time.sleep(PAUSE)
    return data


def lengths(titles):
    """Wikitext size per requested title, keyed by the page it lands on after normalisation and
    redirects. Returns ({page title: size}, [requested titles that do not exist])."""
    out, missing = {}, []
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        q = api(action="query", prop="info", redirects=1, titles="|".join(chunk))["query"]
        hop = {r["from"]: r["to"] for r in q.get("normalized", []) + q.get("redirects", [])}
        sizes = {p["title"]: p["length"] for p in q["pages"] if not p.get("missing")}
        for t in chunk:
            while t in hop:
                t = hop[t]
            if t in sizes:
                out[t] = sizes[t]
            else:
                missing.append(t)
    return out, missing


def extract(title):
    d = api(action="query", prop="extracts", explaintext=1, redirects=1, titles=title)
    page = d["query"]["pages"][0]
    if page.get("missing") or "extract" not in page:
        raise RuntimeError(f"no extract for {title!r}")
    return page["extract"]


def clean_sentences(text):
    text = IN_WORD_MARKS.sub("", text.lower())
    for sent in SENT_SPLIT.split(text):
        words = [w.strip(STRIP) for w in sent.split()]
        if len(words) < 4 or not all(w and set(w) <= ARM_LETTERS for w in words):
            continue
        yield " ".join(words)


def main():
    log = setup_logging()
    t0 = time.perf_counter()
    rng = random.Random(SEED)
    seen, rows, articles = set(), [], []

    def add_article(title, split):
        n = 0
        for s in clean_sentences(extract(title)):
            if s in seen:
                continue
            seen.add(s)
            rows.append((split, len(articles), s))
            n += 1
            if n == MAX_SENT_PER_ARTICLE:
                break
        articles.append({"title": title, "split": split, "sentences": n})
        return n

    funny, missing = lengths(FUNNY)
    log.info(f"funny articles found: {len(funny)} of {len(FUNNY)}; missing: {missing}")
    if len(funny) < MIN_FUNNY:
        raise RuntimeError(f"only {len(funny)} funny articles exist, need {MIN_FUNNY}")
    for title in tqdm(sorted(funny), desc="funny"):
        add_article(title, "test")

    size = lambda: sum(len(s.encode("utf-8")) + 1 for _, _, s in rows)
    bar = tqdm(total=TARGET_BYTES, desc="random", unit="B", unit_scale=True)
    while size() < TARGET_BYTES:
        batch = [p["title"] for p in api(action="query", list="random", rnnamespace=0,
                                         rnlimit=100)["query"]["random"]]
        long_ones = [t for t, n in lengths(batch)[0].items() if n >= MIN_ARTICLE_BYTES]
        for title in long_ones:
            if title in funny or any(a["title"] == title for a in articles):
                continue
            before = size()
            add_article(title, "test" if rng.random() < TEST_SHARE else "train")
            bar.update(size() - before)
            if size() >= TARGET_BYTES:
                break
    bar.close()

    OUT_DIR.mkdir(exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8", newline="\n") as fh:
        for split, aid, s in rows:
            fh.write(f"{split}\t{aid}\t{s}\n")
    meta = {"fetched": date.today().isoformat(), "source": API, "seed": SEED,
            "n_sentences": len(rows),
            "n_train": sum(r[0] == "train" for r in rows), "n_test": sum(r[0] == "test" for r in rows),
            "bytes": OUT_TSV.stat().st_size, "articles": articles}
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"wrote {OUT_TSV.name}: {len(rows)} sentences ({meta['n_train']} train, "
             f"{meta['n_test']} test) from {len(articles)} articles, {meta['bytes']:,} bytes, "
             f"in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
