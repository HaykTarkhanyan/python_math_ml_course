"""What "logprobs do not line up with the reply" looks like, for the job-extraction practical (section 4).

Sends the notebook's endpoint test (same system prompt, schema and test posting) once more to every DeepSeek V4.1
Flash endpoint that failed it with "logprobs do not line up" in results/job_extraction_endpoints.json, and saves the
reply next to its logprob entries: results/job_extraction_logprob_mismatch.json. The notebook only reads that file.

Run (repo root, ~10 s, ~$0.0002):
    ./ma/Scripts/python.exe llms/02_decoding/py_src/non_essential/probe_logprob_mismatch.py
"""
import json
import logging
import os
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

CH = Path(__file__).resolve().parents[2]
REPO_ROOT = CH.parents[1]
OUT = CH / "results" / "job_extraction_logprob_mismatch.json"
MODEL = "deepseek/deepseek-v4.1-flash"

# the notebook's endpoint test, copied from build_job_extraction_nb.py
SALARY = {"type": "object", "additionalProperties": False, "required": ["min", "max", "currency"],
          "properties": {"min": {"type": ["number", "null"]}, "max": {"type": ["number", "null"]},
                         "currency": {"type": ["string", "null"]}}}
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["seniority", "min_years_experience", "skills", "salary", "work_mode"],
          "properties": {"seniority": {"type": "string", "enum": ["junior", "mid", "senior", "lead", "not_stated"]},
                         "min_years_experience": {"type": ["integer", "null"]},
                         "skills": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                         "salary": {"anyOf": [{"type": "null"}, SALARY]},
                         "work_mode": {"type": "string", "enum": ["on_site", "hybrid", "remote", "not_stated"]}}}
SYSTEM = ("You extract facts from a job posting. Use only what the text says; if it does not say, use not_stated "
          "or null. seniority: junior, mid, senior, lead (team lead, head, director, C level) or not_stated. "
          "min_years_experience: the smallest number of years of experience asked for, or null. "
          "skills: up to 10 skills or tools the posting asks for. salary: only if the text states pay. "
          "work_mode: on_site, hybrid, remote or not_stated.")
TEST = "Senior Python developer, fully remote, at least 5 years of experience. Salary 3000-4000 USD."


def main():
    (REPO_ROOT / "logs").mkdir(exist_ok=True)
    log = logging.getLogger("probe_logprob_mismatch")
    for h in (logging.StreamHandler(), logging.FileHandler(REPO_ROOT / "logs" / "probe_logprob_mismatch.log",
                                                           encoding="utf-8")):
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    t0 = time.time()

    load_dotenv(REPO_ROOT / ".env")                          # OPENROUTER_API_KEY, never printed
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=120)
    saved = json.loads((CH / "results" / "job_extraction_endpoints.json").read_text(encoding="utf-8"))
    failed = [t["endpoint"] for t in saved["tried"]
              if t["model"] == "DeepSeek V4.1 Flash" and t["test"].startswith("logprobs do not")]
    if not failed:
        raise SystemExit("no endpoint failed with 'logprobs do not line up' in the saved endpoint test")

    cases, cost = [], 0.0
    for tag in failed:
        r = client.chat.completions.create(
            model=MODEL, max_tokens=200, temperature=0, logprobs=True, top_logprobs=5,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": TEST}],
            response_format={"type": "json_schema", "json_schema": {"name": "job_posting", "strict": True, "schema": SCHEMA}},
            extra_body={"provider": {"order": [tag], "allow_fallbacks": False, "require_parameters": True},
                        "reasoning": {"enabled": False}})
        u = r.model_dump().get("usage") or {}
        cost += u.get("cost") or 0
        lp = r.choices[0].logprobs.content if r.choices[0].logprobs and r.choices[0].logprobs.content else []
        pieces = [bytes(t.bytes).decode("utf-8", "replace") if t.bytes is not None else t.token for t in lp]
        cases.append({"endpoint": tag, "completion_tokens": u.get("completion_tokens"),
                      "content": r.choices[0].message.content or "", "pieces": pieces})
        log.info(f"{tag}: {u.get('completion_tokens')} completion tokens, {len(pieces)} logprob entries")

    OUT.write_text(json.dumps({"date": date.today().isoformat(), "model": MODEL, "test": TEST, "cases": cases,
                               "cost": cost}, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"wrote {OUT.name}: {len(cases)} endpoints, ${cost:.5f}, {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
