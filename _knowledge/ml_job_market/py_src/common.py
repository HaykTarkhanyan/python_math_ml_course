"""Shared helpers for the job-market collectors and the analysis: HTTP with backoff, text clean-up,
redaction of contact details, role classes from a job title, and logging setup.
"""
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]          # repo root
HERE = Path(__file__).resolve().parents[1]          # _knowledge/ml_job_market
DATA = HERE / "data"
RESULTS = HERE / "results"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9"}


def get_logger(name):
    (ROOT / "logs").mkdir(exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        for h in (logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / f"{name}.log", encoding="utf-8")):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


class Fetcher:
    """requests.Session with a fixed pause between calls and backoff on 429 / 5xx.
    Raises after the last retry: the caller decides whether that kills the run."""

    def __init__(self, log, pause=1.5, retries=4):
        self.s = requests.Session()
        self.s.headers.update(UA)
        self.log, self.pause, self.retries = log, pause, retries
        self.calls = 0

    def get(self, url, **kw):
        for attempt in range(self.retries + 1):
            time.sleep(self.pause)
            self.calls += 1
            r = self.s.get(url, timeout=30, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504, 999) and attempt < self.retries:
                wait = 20 * 2 ** attempt
                self.log.warning(f"HTTP {r.status_code} on {url} - waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            raise RuntimeError(f"HTTP {r.status_code} on {url}")
        raise RuntimeError(f"gave up on {url}")


def html_to_text(html):
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text("\n", strip=True)


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?:\+?374|\b0)[\s()-]*\d{2}[\s()-]*\d{2,3}[\s-]*\d{2,3}[\s-]*\d{0,3}")


def redact(text):
    """Drop e-mail addresses and Armenian phone numbers: the repo is public."""
    return PHONE.sub("[phone]", EMAIL.sub("[email]", text or ""))


# Role classes from the title, checked in this order. "core" = an ML / AI / data-science role.
ROLES = [
    ("genai",    True,  r"\b(llm|genai|gen ai|generative|agentic|prompt engineer|rag)\b"),
    ("cv",       True,  r"computer vision|\bcv engineer|vision (ml|ai|engineer)|perception"),
    ("nlp",      True,  r"\bnlp\b|natural language|\bspeech\b|\basr\b|\bvoice ai\b|conversational ai"),
    ("mlops",    True,  r"mlops|ml ops|ml platform|ml infrastructure|machine learning (platform|infrastructure)"),
    ("research", True,  r"research (scientist|engineer)|applied scientist|\bml researcher|ai researcher"),
    ("ml_eng",   True,  r"machine learning|\bml\b|deep learning|\bml/ai\b|\bai/ml\b"),
    ("ds",       True,  r"data scien"),
    # annotation / evaluation gigs: "AI Trainer", "AI Training Contributor - Armenian", "AI Product Evaluator"
    ("ai_data",  False, r"ai trainer|ai training|data training|\bai (product )?evaluator|annotat|data label|\brater\b"),
    # a software role that uses or integrates AI: ".NET Engineer with AI tools", "AI-first Full Stack"
    ("swe_ai",   False, r"^(?=.*\b(engineer|developer|programmer|lead)\b)(?=.*(\bai\b[- ](native|first|enabled|proficient|assisted|tools)"
                        r"|with ai|ai integration|(full[- ]?stack|fullstack|\.net|\bjava\b|node|frontend|backend|software|python team).*\bai\b))"),
    ("analyst",  False, r"data analy|\bbi\b|business intelligence|analytics engineer|product analyst|data analytics|reporting analyst"),
    ("data_eng", False, r"data engineer|\betl\b|data platform"),
    ("ai_eng",   True,  r"\b(ai|artificial intelligence)\b.*\b(engineer|developer|scientist|architect|lead|researcher)\b"
                        r"|\b(engineer|developer|architect|lead)\b.*\b(ai|artificial intelligence)\b"),
    ("ai_other", False, r"\b(ai|artificial intelligence)\b"),
]
ROLE_RES = [(name, core, re.compile(p, re.I)) for name, core, p in ROLES]
ROLE_LABEL = {"genai": "LLM / GenAI engineer", "cv": "Computer vision", "nlp": "NLP / speech",
              "mlops": "MLOps / ML platform", "research": "Research scientist / engineer",
              "ml_eng": "ML engineer", "ds": "Data scientist", "ai_eng": "AI engineer (general)",
              "ai_data": "AI data work (training, evaluation)", "swe_ai": "Software engineer + AI",
              "ai_other": "Other AI-titled role", "analyst": "Data / BI analyst", "data_eng": "Data engineer",
              "other": "Other title, ML in text"}


def role_of(title):
    """(role, is_core) for a job title, or (None, False) when nothing matches."""
    for name, core, rx in ROLE_RES:
        if rx.search(title or ""):
            return name, core
    return None, False
