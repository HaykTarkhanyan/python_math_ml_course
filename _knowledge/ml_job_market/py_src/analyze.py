"""Tag every collected posting with skills, a role and a seniority bucket, and count them per group.

Run (repo root):  ./ma/Scripts/python.exe _knowledge/ml_job_market/py_src/analyze.py [YYYY-MM-DD]
Takes ~15 s (measured 13 s on 2026-10-28, ~300 postings). Reads data/staffam_<date>.json and data/linkedin_<date>.json (latest date if none given),
writes results/skills_<date>.json. report.py builds the HTML from that file only.

Skills are matched with the regex dictionary below (presence per posting, not counts), on the title
plus the full text. A hand-made dictionary instead of a skills library on purpose: the point is the
course's own categories (RAG, agents, evals, MCP, credit scoring ...), which generic skill
taxonomies do not track.
"""
import json
import re
import sys
import time
from collections import Counter
from statistics import median

from common import DATA, RESULTS, ROLE_LABEL, get_logger, role_of

log = get_logger("analyze")

# (name, category, regex). Case-insensitive unless the pattern starts with (?-i).
SKILLS = [
    # languages
    ("Python", "Languages", r"\bpython\b"),
    ("SQL", "Languages", r"\bsql\b|postgres|mysql|t-sql|pl/sql"),
    ("R", "Languages", r"(?-i:\bR\b\s*(?:,|/|or\b|and\b)|(?:,|/|\bor|\band)\s*\bR\b(?![&\w])|\bR (?:language|programming)|\bRStudio\b)"),
    ("C++", "Languages", r"c\+\+"),
    ("Java / Scala", "Languages", r"\bjava\b(?!script)|\bscala\b"),
    ("JavaScript / TypeScript", "Languages", r"javascript|typescript|node\.?js|\breact\b"),
    # math and statistics
    ("Statistics", "Math & statistics", r"statistic|статистик|վիճակագր"),
    ("Probability", "Math & statistics", r"probabilit|bayesian"),
    ("Linear algebra / calculus", "Math & statistics", r"linear algebra|calculus"),
    ("A/B testing, experiments", "Math & statistics", r"a/b test|ab test|experiment(al)? design|hypothesis test|causal inference"),
    ("Optimization", "Math & statistics", r"\boptimi[sz]ation algorithm|convex|operations research|linear programming"),
    # classic ML
    ("scikit-learn", "Classic ML", r"scikit|sklearn"),
    ("Gradient boosting", "Classic ML", r"xgboost|lightgbm|catboost|gradient boost"),
    ("Time series / forecasting", "Classic ML", r"time[- ]series|forecast"),
    ("Recommender systems", "Classic ML", r"recommend(er|ation) (system|engine)|recsys|personali[sz]ation"),
    ("Fraud / anomaly detection", "Classic ML", r"anomaly detection|fraud|outlier detection"),
    ("Credit scoring / risk models", "Classic ML", r"credit scor|scorecard|risk model|credit risk|\bpd model|probability of default"),
    ("Feature engineering", "Classic ML", r"feature engineering|feature selection"),
    # deep learning
    ("PyTorch", "Deep learning", r"pytorch|\btorch\b"),
    ("TensorFlow / Keras", "Deep learning", r"tensorflow|\bkeras\b"),
    ("Deep learning", "Deep learning", r"deep learning|neural network"),
    ("Transformers / Hugging Face", "Deep learning", r"transformers?\b(?! station)|hugging ?face|\bbert\b"),
    ("Computer vision", "Deep learning", r"computer vision|object detection|image (classification|segmentation|processing)|opencv|\byolo\b|\bocr\b"),
    ("NLP", "Deep learning", r"\bnlp\b|natural language"),
    ("Speech / voice", "Deep learning", r"speech[- ]to[- ]text|\basr\b|text[- ]to[- ]speech|\btts\b|speech (recognition|synthesis)|\bwhisper\b|voice (ai|agent|assistant)|\baudio\b"),
    ("Multimodal / VLM", "Deep learning", r"multi-?modal|vision[- ]language|\bvlms?\b"),
    # LLM / GenAI
    ("LLMs", "LLM & GenAI", r"\bllms?\b|large language model|generative ai|\bgenai\b|gen ai"),
    ("RAG", "LLM & GenAI", r"\brag\b|retrieval[- ]augmented"),
    ("Vector databases", "LLM & GenAI", r"vector (db|database|store|search|index)|pinecone|weaviate|qdrant|milvus|chroma|pgvector|\bfaiss\b"),
    ("Embeddings", "LLM & GenAI", r"embedding"),
    ("Prompt engineering", "LLM & GenAI", r"\bprompt(s|ing)?\b"),                # not "promptly"
    ("Fine-tuning (LoRA, SFT, RLHF)", "LLM & GenAI", r"fine[- ]?tun|\blora\b|qlora|\bpeft\b|\bsft\b|\brlhf\b|\bdpo\b"),
    ("Agents / agentic", "LLM & GenAI", r"agentic|\bai agents?\b|\bllm agents?\b|multi-?agent|autonomous agents?|\bagent (framework|workflow|system|orchestration)s?"),
    ("LangChain / LangGraph", "LLM & GenAI", r"langchain|langgraph"),
    ("LlamaIndex", "LLM & GenAI", r"llama[- ]?index"),
    ("Other agent frameworks", "LLM & GenAI", r"crew ?ai|autogen|semantic kernel|agents sdk|smolagents|pydantic ?ai|\bdspy\b|google adk"),
    ("Tool / function calling", "LLM & GenAI", r"tool[- ]?(calling|use)|function[- ]calling"),
    ("MCP", "LLM & GenAI", r"\bmcp\b|model context protocol"),
    ("LLM provider APIs", "LLM & GenAI", r"openai|\bgpt-?[345o]|chatgpt|anthropic|\bclaude\b|\bgemini\b|mistral|\bllama\b|bedrock|azure openai"),
    ("LLM evals / observability", "LLM & GenAI", r"\bevals?\b|llm[- ]as[- ]a?[- ]?judge|ragas|deepeval|langsmith|langfuse|promptfoo|arize|evaluation (pipeline|framework|harness)s?|(llm|rag|genai) evaluation"),
    ("Guardrails / safety", "LLM & GenAI", r"guardrail|hallucination|prompt injection|red[- ]team|responsible ai|ai safety"),
    ("LLM serving / inference", "LLM & GenAI", r"\bvllm\b|\btgi\b|triton|tensorrt|\bonnx\b|llama\.cpp|ollama|model serving|inference (optimi[sz]ation|server|engine)s?"),
    ("Quantization", "LLM & GenAI", r"quantiz"),
    ("Knowledge graphs", "LLM & GenAI", r"knowledge graph|graphrag|neo4j|cypher"),
    # MLOps and infrastructure
    ("Docker", "MLOps & infra", r"docker|containeri[sz]"),
    ("Kubernetes", "MLOps & infra", r"kubernetes|\bk8s\b|\bhelm\b"),
    ("Experiment tracking (MLflow, W&B)", "MLOps & infra", r"mlflow|weights (&|and) biases|\bwandb\b|experiment tracking"),
    ("Orchestration (Airflow etc.)", "MLOps & infra", r"airflow|prefect|dagster|kubeflow|\bargo\b"),
    ("CI/CD", "MLOps & infra", r"ci/cd|\bcicd\b|github actions|gitlab ci|jenkins"),
    ("Monitoring / drift", "MLOps & infra", r"model monitoring|\bdrift\b|observability|prometheus|grafana"),
    ("MLOps (named)", "MLOps & infra", r"mlops|ml ops|llmops"),
    ("Spark / Databricks", "MLOps & infra", r"\bspark\b|pyspark|databricks|hadoop"),
    ("Kafka / streaming", "MLOps & infra", r"kafka|stream processing|real-time data"),
    ("AWS", "MLOps & infra", r"\baws\b|amazon web services|sagemaker|\bbedrock\b"),
    ("GCP", "MLOps & infra", r"\bgcp\b|google cloud|vertex ai|bigquery"),
    ("Azure", "MLOps & infra", r"\bazure\b"),
    ("Linux / shell", "MLOps & infra", r"\blinux\b|\bbash\b|shell script"),
    ("Git", "MLOps & infra", r"\bgit\b|github|gitlab|bitbucket"),
    ("APIs (FastAPI, REST)", "MLOps & infra", r"fastapi|\bflask\b|django|rest(ful)? api|\bapis?\b|microservice|grpc"),
    ("NoSQL / caches", "MLOps & infra", r"mongodb|redis|cassandra|nosql|dynamodb|elasticsearch|opensearch"),
    ("Data warehouses / dbt", "MLOps & infra", r"snowflake|redshift|clickhouse|data warehouse|\bdbt\b|bigquery"),
    # data analysis
    ("pandas / NumPy", "Data analysis", r"pandas|numpy|scipy|polars"),
    ("BI tools (Power BI, Tableau)", "Data analysis", r"power ?bi|tableau|looker|superset|metabase|qlik"),
    ("Excel", "Data analysis", r"\bexcel\b"),
    ("ETL / data pipelines", "Data analysis", r"\betl\b|\belt\b|data pipeline"),
    ("Visualization", "Data analysis", r"matplotlib|seaborn|plotly|visuali[sz]ation|dashboard"),
    # other requirements
    ("English", "Other requirements", r"english|անգլերեն|английск"),
    ("Russian", "Other requirements", r"russian|ռուսերեն|русск"),
    ("Armenian", "Other requirements", r"armenian|հայերեն|армянск"),
    ("Degree mentioned", "Other requirements", r"\bmaster'?s\b|\bmsc\b|m\.sc|\bphd\b|ph\.d|bachelor|\bbsc\b|\bdegree\b|магистр|бакалавр|մագիստր|բակալավր"),
    ("PhD mentioned", "Other requirements", r"\bphd\b|ph\.d|doctorate|доктор|кандидат наук"),
    ("Research / publications", "Other requirements", r"publication|neurips|\bicml\b|\biclr\b|\bcvpr\b|research papers?|state[- ]of[- ]the[- ]art"),
]
SKILL_RES = []
for name, cat, pat in SKILLS:
    SKILL_RES.append((name, cat, re.compile(pat) if pat.startswith("(?-i") else re.compile(pat, re.I)))

ML_SIGNAL = re.compile(r"machine learning|deep learning|\bllms?\b|pytorch|tensorflow|scikit|computer vision|\bnlp\b|"
                       r"generative ai|\brag\b|neural network|машинн\w* обучени|մեքենայական ուսուցում", re.I)
YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:[-–]\s*\d{1,2}\s*\+?\s*)?(?:years?|yrs|лет|года?|տարվա|տարի)", re.I)
EXPERIENCE = re.compile(r"experience|опыт|փորձ", re.I)
ENTRY_T = re.compile(r"\b(intern|internship|junior|jr\.?|trainee|entry|graduate|student)\b", re.I)
SENIOR_T = re.compile(r"\b(senior|sr\.?|lead|principal|staff|head|director|manager|architect|chief)\b", re.I)


def latest(prefix, day):
    files = sorted(DATA.glob(f"{prefix}_{day or '*'}.json"))
    if not files:
        raise SystemExit(f"no data/{prefix}_{day or '*'}.json - run the collector first")
    return files[-1]


def years_min(text):
    found = []
    for m in YEARS.finditer(text):
        window = text[max(0, m.start() - 60): m.end() + 60]
        if EXPERIENCE.search(window) and 0 < int(m.group(1)) <= 20:
            found.append(int(m.group(1)))
    return min(found) if found else None


def seniority(title, level, yrs):
    if ENTRY_T.search(title):
        return "entry"
    if SENIOR_T.search(title):
        return "senior"
    lv = (level or "").lower()
    if any(w in lv for w in ("intern", "entry", "junior")):
        return "entry"
    if any(w in lv for w in ("senior", "director", "executive", "lead")):
        return "senior"
    if yrs is not None:
        return "entry" if yrs <= 1 else ("mid" if yrs <= 4 else "senior")
    if any(w in lv for w in ("associate", "mid")):
        return "mid"
    return "unstated"


def to_postings(staff, linkedin):
    rows = []
    for j in staff["jobs"]:
        text = "\n".join(j[k] for k in ("description", "responsibilities", "required_qualifications", "additional_information") if j.get(k))
        rows.append({"source": "staff.am", "market": "armenia", "id": j["id"], "title": j["title"], "company": j["company"],
                     "location": j["city"] or "Armenia", "url": j["url"], "posted": (j["posted"] or "")[:10],
                     "level": j["level"], "text": text, "query": None})
    for j in linkedin["jobs"]:
        if j["market"] == "armenia" and "Armenia" not in j["location"]:
            log.warning(f"dropping LinkedIn 'armenia' posting located in {j['location']!r}: {j['title']}")
            continue
        rows.append({"source": "linkedin", "market": j["market"], "id": j["id"], "title": j["title"], "company": j["company"],
                     "location": j["location"], "url": j["url"], "posted": j["posted"] or "", "level": j["level"],
                     "text": j["description"], "query": j["query"]})
    return rows


# Searches that select for LLM work. Postings they found first would inflate the LLM shares, so the
# results carry a per-origin split (query_check) to show how much the search wording moves the numbers.
LLM_QUERIES = {"AI engineer", "LLM", "generative AI", "LLM engineer"}
CHECK_SKILLS = ["LLMs", "RAG", "Agents / agentic", "LLM provider APIs", "Fine-tuning (LoRA, SFT, RLHF)", "Python", "PyTorch", "SQL"]


def origin(p):
    if p["source"] == "staff.am":
        return "staff.am, every listing (no search words)"
    return "LinkedIn, LLM / AI searches" if p["query"] in LLM_QUERIES else "LinkedIn, neutral searches"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def main():
    t0 = time.time()
    day = sys.argv[1] if len(sys.argv) > 1 else None
    sf, lf = latest("staffam", day), latest("linkedin", day)
    staff, linkedin = (json.loads(p.read_text(encoding="utf-8")) for p in (sf, lf))
    rows = to_postings(staff, linkedin)

    seen, postings, dupes = {}, [], 0
    for r in rows:
        key = (r["market"], norm(r["company"]), norm(r["title"]))
        if key in seen:                                    # same job on both sites (or listed twice)
            seen[key].setdefault("also_on", []).append(r["url"])
            dupes += 1
            continue
        role, core = role_of(r["title"])
        ml_text = len(set(m.group(0).lower() for m in ML_SIGNAL.finditer(r["text"])))
        if role is None:
            if ml_text < 2:
                continue                                   # an IT job with no ML in it
            role = "other"
        group = (("AM" if r["market"] == "armenia" else "GL") + "_" +
                 ("core" if core else "analyst" if role in ("analyst", "data_eng") else "swe_ai" if role == "swe_ai" else "other"))
        # Worldwide postings found by an LLM-selecting search are 97% LLM jobs (2026-10-28): they would make the
        # comparison group about the search words, so they get their own group, used only for "what LLM jobs ask".
        if group == "GL_core" and r["query"] in LLM_QUERIES:
            group = "GL_llm"
        hay = r["title"] + "\n" + r["text"]
        skills = [name for name, _, rx in SKILL_RES if rx.search(hay)]
        yrs = years_min(r["text"])
        p = {k: r[k] for k in ("source", "market", "id", "title", "company", "location", "url", "posted", "query")}
        p.update(role=role, role_label=ROLE_LABEL[role], group=group, seniority=seniority(r["title"], r["level"], yrs),
                 years_min=yrs, skills=skills, ml_terms_in_text=ml_text)
        seen[key] = p
        postings.append(p)

    groups = {}
    for g in sorted({p["group"] for p in postings}):
        ps = [p for p in postings if p["group"] == g]
        yrs = [p["years_min"] for p in ps if p["years_min"] is not None]
        # Company-level shares too: one employer's templated postings (EPAM had ~35 of 76) should not set the numbers.
        companies = {p["company"] for p in ps}
        groups[g] = {"n": len(ps), "n_companies": len(companies),
                     "skills_companies": {name: len({p["company"] for p in ps if name in p["skills"]}) for name, _, _ in SKILL_RES},
                     "roles": dict(Counter(p["role_label"] for p in ps).most_common()),
                     "seniority": dict(Counter(p["seniority"] for p in ps)),
                     "years_min_median": median(yrs) if yrs else None, "years_stated": len(yrs),
                     "skills": {name: sum(name in p["skills"] for p in ps) for name, _, _ in SKILL_RES},
                     "companies": dict(Counter(p["company"] for p in ps).most_common())}
        log.info(f"{g}: {len(ps)} postings")

    query_check = []
    for g in ("AM_core", "GL"):
        ps = [p for p in postings if p["group"] == g or (g == "GL" and p["group"] in ("GL_core", "GL_llm"))]
        for o in sorted({origin(p) for p in ps}):
            sub = [p for p in ps if origin(p) == o]
            query_check.append({"group": g, "origin": o, "n": len(sub), "companies": len({p["company"] for p in sub}),
                                "shares": {s: round(100 * sum(s in p["skills"] for p in sub) / len(sub), 1) for s in CHECK_SKILLS}})
            log.info(f"{g} / {o}: n={len(sub)}, LLMs {query_check[-1]['shares']['LLMs']}%")

    RESULTS.mkdir(parents=True, exist_ok=True)
    snap = staff["collected"]
    out = RESULTS / f"skills_{snap}.json"
    out.write_text(json.dumps({
        "snapshot": snap,
        "inputs": {"staffam_file": sf.name, "staffam_listings": staff["total_listings"], "staffam_fetched": len(staff["jobs"]),
                   "staffam_failures": len(staff["failures"]), "linkedin_file": lf.name, "linkedin_cards": linkedin["cards_seen"],
                   "linkedin_fetched": len(linkedin["jobs"]), "linkedin_failures": len(linkedin["failures"]),
                   "linkedin_queries": {m: c["queries"] for m, c in linkedin["markets"].items()}, "duplicates_merged": dupes},
        "skills_meta": [{"name": n, "category": c} for n, c, _ in SKILLS],
        "groups": groups, "query_check": query_check, "postings": postings}, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"wrote {out} ({len(postings)} postings, {dupes} duplicates merged) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
