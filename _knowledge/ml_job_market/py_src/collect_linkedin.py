"""Collect ML / AI / data job postings from LinkedIn's public (logged-out) job search: Armenia, and a
worldwide sample for comparison.

Run (repo root):  ./ma/Scripts/python.exe -u _knowledge/ml_job_market/py_src/collect_linkedin.py
Takes ~12-18 min (a guess: ~120 search pages + ~250 postings at one request per ~2.5 s, more if
LinkedIn answers 429 and the backoff kicks in).

Location is a numeric geoId, not a name: location=Armenia returns jobs in Armenia, Quindio
(Colombia). 103030111 = Armenia, 92000000 = Worldwide (both checked 2026-10-28).
Search results are fuzzy, so only cards whose title matches a role class (common.ROLES) get their
full text fetched; the worldwide sample keeps core ML/AI/DS titles only.
Output: data/linkedin_<date>.json. Contact details are redacted.
"""
import json
import re
import time
from datetime import date

from bs4 import BeautifulSoup
from tqdm import tqdm

from common import DATA, Fetcher, get_logger, redact, role_of

SEARCH = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
POSTING = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{}"
MARKETS = {
    "armenia": {"geoId": "103030111", "max_pages": 20,
                "queries": ["machine learning", "data scientist", "AI engineer", "LLM", "generative AI",
                            "computer vision", "NLP", "deep learning", "MLOps", "data analyst"]},
    "global": {"geoId": "92000000", "max_pages": 5,
               "queries": ["machine learning engineer", "AI engineer", "data scientist", "LLM engineer"]},
}
log = get_logger("collect_linkedin")


def text_of(node):
    return node.get_text(" ", strip=True) if node else ""


def parse_cards(html):
    soup = BeautifulSoup(html, "html.parser")
    cards = []
    for c in soup.select("[data-entity-urn^='urn:li:jobPosting:']"):
        jid = c["data-entity-urn"].rsplit(":", 1)[1]
        link = c.select_one("a.base-card__full-link")
        t = c.select_one("time")
        cards.append({"id": jid,
                      "title": text_of(c.select_one(".base-search-card__title")),
                      "company": text_of(c.select_one(".base-search-card__subtitle")),
                      "location": text_of(c.select_one(".job-search-card__location")),
                      "posted": t.get("datetime") if t else None,
                      "url": link["href"].split("?")[0] if link else f"https://www.linkedin.com/jobs/view/{jid}"})
    return cards


def parse_posting(html):
    soup = BeautifulSoup(html, "html.parser")
    desc = soup.select_one(".show-more-less-html__markup")
    criteria = {text_of(li.select_one(".description__job-criteria-subheader")):
                text_of(li.select_one(".description__job-criteria-text"))
                for li in soup.select(".description__job-criteria-item")}
    if desc is None:
        raise RuntimeError("no description block in the posting HTML")
    return redact(desc.get_text("\n", strip=True)), criteria


def main():
    t0 = time.time()
    f = Fetcher(log, pause=2.5)
    cards = {}                                   # (market, id) -> card
    for market, cfg in MARKETS.items():
        for q in cfg["queries"]:
            before, stale, pages = len(cards), 0, 0
            for page in range(cfg["max_pages"]):
                r = f.get(SEARCH, params={"keywords": q, "geoId": cfg["geoId"], "start": page * 10})
                got = parse_cards(r.text)
                pages += 1
                new = 0
                for c in got:
                    key = (market, c["id"])
                    if key not in cards:
                        cards[key] = {**c, "market": market, "query": q}
                        new += 1
                # A page of already-seen cards is not the end: other queries found them first.
                stale = stale + 1 if not new else 0
                if not got or stale >= 2:
                    break
            log.info(f"{market} / '{q}': {pages} pages, {len(cards) - before} new cards ({len(cards)} total)")

    not_am = [c for (m, _), c in cards.items() if m == "armenia" and "Armenia" not in c["location"]]
    if not_am:
        log.warning(f"{len(not_am)} 'armenia' cards have a location without 'Armenia', e.g. {not_am[0]['location']!r}")

    keep = []
    for (market, _), c in cards.items():
        role, core = role_of(c["title"])
        if (market == "armenia" and role) or (market == "global" and core):
            keep.append({**c, "role_from_title": role})
    log.info(f"{len(keep)} of {len(cards)} cards kept for full text")

    # Postings already fetched today are reused, so a re-run only downloads what is new.
    path = DATA / f"linkedin_{date.today().isoformat()}.json"
    cached = {}
    if path.exists():
        cached = {(j["market"], j["id"]): j for j in json.loads(path.read_text(encoding="utf-8"))["jobs"]}
        log.info(f"reusing {len(cached)} postings already in {path.name}")

    out, failures = [], []
    for c in tqdm(keep, desc="postings", mininterval=5):
        if (c["market"], c["id"]) in cached:
            out.append({**cached[(c["market"], c["id"])], "query": c["query"], "role_from_title": c["role_from_title"]})
            continue
        try:
            desc, criteria = parse_posting(f.get(POSTING.format(c["id"])).text)
        except Exception as e:                   # recorded and counted, never dropped quietly
            log.error(f"failed posting {c['id']}: {e}")
            failures.append({"id": c["id"], "market": c["market"], "error": str(e)})
            continue
        out.append({"source": "linkedin", **c, "description": desc,
                    "level": criteria.get("Seniority level", ""), "job_type": criteria.get("Employment type", ""),
                    "function": criteria.get("Job function", ""), "industries": criteria.get("Industries", "")})

    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"collected": date.today().isoformat(), "markets": MARKETS,
                                "cards_seen": len(cards), "kept_for_text": len(keep),
                                "jobs": out, "failures": failures}, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"wrote {len(out)} postings ({len(failures)} failures) to {path} in {time.time() - t0:.0f}s, {f.calls} requests")
    if failures:
        log.error(f"{len(failures)} postings failed - see 'failures' in the output")


if __name__ == "__main__":
    main()
