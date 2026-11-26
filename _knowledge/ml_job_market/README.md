# ML job market - what employers ask for

Open ML / AI / data postings in Armenia (staff.am, LinkedIn) and a worldwide LinkedIn sample, tagged by
skill, so course projects can be designed around what is actually hired for. First snapshot 2026-10-28.

**Read:** `ML_JOB_MARKET_<date>.html` (standalone, open in a browser). Commentary for the course:
`notes/implications_<date>.html` (also section 3 of the report).

## Layout

| Path | What |
|---|---|
| `data/staffam_<date>.json` | staff.am: every listing walked, full text of ML/AI/data or IT-category jobs. Contact details redacted |
| `data/linkedin_<date>.json` | LinkedIn public search: cards for the queries, full text of title-matched postings. Contact details redacted |
| `results/skills_<date>.json` | per posting: role, seniority, years asked, skills found; per group: counts. The artifact of record for the report |
| `ML_JOB_MARKET_<date>.html` | report built from the results file only |
| `notes/implications_<date>.html` | hand-written: what the snapshot means for the course (pulled into the report) |
| `notes/published_reports.html` | published numbers for context (Lightcast / Stanford AI Index, AGBU, ...) with how much to trust each |
| `py_src/` | `collect_staffam.py`, `collect_linkedin.py`, `analyze.py`, `report.py`, `common.py` (shared helpers, role patterns) |

## Re-run (repo root, `ma` venv)

```bash
./ma/.venv/Scripts/python.exe -u _knowledge/ml_job_market/py_src/collect_staffam.py    # 5.4 min (2026-10-28, 177 requests)
./ma/.venv/Scripts/python.exe -u _knowledge/ml_job_market/py_src/collect_linkedin.py   # 16 min fresh (327 requests); 5 min re-run same day
./ma/.venv/Scripts/python.exe _knowledge/ml_job_market/py_src/analyze.py               # ~15 s
./ma/.venv/Scripts/python.exe _knowledge/ml_job_market/py_src/report.py                # ~3 s
```

The LinkedIn collector reuses postings already in today's file, so a same-day re-run only downloads new ones.

A new date gives new files next to the old ones, so snapshots can be compared later. Write
`notes/implications_<date>.html` for a new snapshot, or the report shows a red "no commentary" line.

## Gotchas

- **staff.am's keyword search is fuzzy** ("machine learning" returned 131 jobs, mostly unrelated), so the
  collector walks all listing pages (`/en/jobs?page=N`, 50 per page) and filters itself. Its data is in the
  page's `__NEXT_DATA__` JSON, in `en` / `am` / `ru`.
- **A posting's `lang` is the first language field the employer filled in, not the language of its text**: some
  type Armenian into the English field. In the 2026-10-28 file, 11 of 148 differ from a letter count (by letters:
  116 English, 31 Armenian, 1 Russian). Count letters when the language matters (the job-extraction practical in
  `llms/02_decoding/` does).
- **LinkedIn needs the country id, not the name**: `location=Armenia` returns Armenia, Quindio (Colombia).
  `geoId=103030111` is Armenia, `92000000` is Worldwide. Public endpoints: `jobs-guest/jobs/api/seeMoreJobPostings/search`
  (10 cards per `start` step) and `jobs-guest/jobs/api/jobPosting/<id>`.
- Skill matching is a regex dictionary in `analyze.py` (presence per posting). It over-counts a skill named in
  passing and misses synonyms; read the shares as rough.
- **One employer can set the numbers**: EPAM had 35 of 74 Armenian ML/AI/DS postings on 2026-10-28, with shared
  template text (AWS / Azure / GCP read ~60% of postings but 26-30% of companies). The report leads with share of companies.
- **Search words bias the worldwide sample**: postings found by "AI engineer" / "LLM engineer" were 97% LLM jobs vs 42%
  for "machine learning engineer" / "data scientist". The worldwide comparison group uses the neutral searches only;
  `query_check` in the results file keeps the split. Each LinkedIn search after the first one stops after 2 pages
  with nothing new (a page of already-seen cards is not the end of the results).
