"""Collect open jobs from staff.am that could be ML / AI / data roles, with their full text.

Run (repo root):  ./ma/.venv/Scripts/python.exe -u _knowledge/ml_job_market/py_src/collect_staffam.py
Takes ~5-8 min (a guess: ~27 list pages + ~150-250 job pages at one request per ~1.5 s).

staff.am's keyword search is fuzzy ("machine learning" returns 131 mostly unrelated jobs), so this
walks every listing page, keeps a job if its title matches a role class (common.ROLES) or it sits in
an IT / data category, then fetches each kept job's page and stores its text. Deciding what counts
as an ML job happens later, in analyze.py.
Output: data/staffam_<date>.json. Contact details are redacted.
"""
import json
import re
import time
from datetime import date

from tqdm import tqdm

from common import DATA, Fetcher, get_logger, html_to_text, redact, role_of

BASE = "https://staff.am/en/jobs"
IT_CATEGORIES = {"software-development", "other-it", "analytics", "quality-assurance", "it-support"}
log = get_logger("collect_staffam")


def next_data(f, url):
    r = f.get(url)
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    if not m:
        raise RuntimeError(f"no __NEXT_DATA__ on {url}")
    return json.loads(m.group(1))["props"]["pageProps"]


def tr(d, *langs):
    """First non-empty translation, and which language it was."""
    for lang in langs or ("en", "am", "ru"):
        if d and d.get(lang):
            return d[lang], lang
    return "", None


def main():
    t0 = time.time()
    f = Fetcher(log, pause=1.5)
    listings, page, total = {}, 1, None
    while True:
        pp = next_data(f, f"{BASE}?page={page}")
        total = total or pp["totalCount"]
        jobs = [j for j in pp["jobs"] if "id" in j]          # one promo card per page has no id
        new = [j for j in jobs if j["id"] not in listings]
        for j in new:
            listings[j["id"]] = j
        log.info(f"list page {page}: {len(jobs)} jobs, {len(new)} new, {len(listings)}/{total} so far")
        if not new:
            break
        page += 1
    if abs(len(listings) - total) > 0.02 * total:
        log.error(f"collected {len(listings)} listings but staff.am reports {total}")
        raise SystemExit(1)

    keep = []
    for j in listings.values():
        title = tr(j["title"])[0]
        role, _ = role_of(title)
        if role or j["category"]["code"] in IT_CATEGORIES:
            keep.append(j)
    log.info(f"{len(keep)} of {len(listings)} listings kept for full text (title match or IT/data category)")

    out, failures = [], []
    for j in tqdm(keep, desc="job pages", mininterval=5):
        # Each language has its own slug, and some en slugs 404 (data-scientist-360 did, its am slug works).
        urls = [f"{BASE}/{j['category']['code']}/{s}" for s in dict.fromkeys(j["slug"].get(k) for k in ("en", "am", "ru")) if s]
        job, errors = None, []
        for url in urls:
            try:
                cand = next_data(f, url)["job"]
            except Exception as e:
                errors.append(f"{url}: {e}")
                continue
            if cand["id"] != j["id"]:
                errors.append(f"{url}: page is job {cand['id']}, not {j['id']}")
                continue
            job = cand
            break
        if job is None:                                          # recorded and counted, never dropped quietly
            log.error(f"failed job {j['id']}: {errors}")
            failures.append({"id": j["id"], "urls": urls, "errors": errors})
            continue
        parts = {}
        lang_used = None
        for field in ("description", "responsibilities", "required_qualifications", "additional_information"):
            text, lang = tr(job.get(field))
            parts[field] = redact(html_to_text(text))
            lang_used = lang_used or lang
        out.append({
            "source": "staff.am", "id": job["id"], "url": url,
            "title": tr(job["title"])[0],
            "company": tr(job["companiesStruct"]["title"])[0].strip(),
            "category": job["category"]["code"],
            "city": tr((job.get("job_city") or {}).get("title"))[0],
            "remote": bool(job.get("is_remote")),
            "level": tr((job.get("job_candidate_level") or {}).get("title"))[0],
            "job_type": tr((job.get("job_type") or {}).get("title"))[0],
            "salary_from": job.get("salary_from"), "salary_to": job.get("salary_to"),
            "salary_currency": job.get("salary_currency"),
            "posted": (j.get("activated_at") or {}).get("staffam"), "deadline": job.get("deadline"),
            "skills": job.get("skills"),                        # kept as staff.am sends it
            "lang": lang_used, **parts,
        })

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"staffam_{date.today().isoformat()}.json"
    path.write_text(json.dumps({"collected": date.today().isoformat(), "total_listings": len(listings),
                                "kept_for_text": len(keep), "jobs": out, "failures": failures},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"wrote {len(out)} jobs ({len(failures)} failures) to {path} in {time.time() - t0:.0f}s, {f.calls} requests")
    if failures:
        log.error(f"{len(failures)} job pages failed - see 'failures' in the output")


if __name__ == "__main__":
    main()
