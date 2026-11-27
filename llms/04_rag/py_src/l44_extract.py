"""Paid step: let a real LLM extract entities and relationships from every L44 chunk.

The deck's teaching graph (l44_data.TRIPLES) is written by hand. This script checks what a real
extractor produces from the same 38 chunks, so the "graph walk" result on the slides is
measured on an LLM-built graph, not on the hand-written one. GraphRAG-style prompt, one call
per chunk, JSON out. No resolution here: l44_llm_graph.py builds and analyses the graph.

Writes: results/l44_extraction_<model>.json (raw replies, kept as the record) and appends one
line per call to results/l44_calls.jsonl (the cost ledger).
Cost (prices checked on OpenRouter 2026-11-27): gpt-4o-mini about $0.01, gpt-6.1-terra about
$0.12 for all 38 chunks. Time: ~1-3 min per model.

Run:  ./ma/.venv/Scripts/python.exe llms/04_rag/py_src/l44_extract.py openai/gpt-4o-mini
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from l44_data import CORPUS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = HERE.parent / "results"
LEDGER = RESULTS / "l44_calls.jsonl"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l44_extract.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# provider pinned so reruns are comparable; same settings as sketch_finder's config
MODELS = {
    "openai/gpt-4o-mini": {"provider": "openai", "temperature": True, "reasoning": False},
    "openai/gpt-6.1-terra": {"provider": "openai", "temperature": False, "reasoning": True},
}

PROMPT = """You extract a knowledge graph from one chunk of a cheese factory's document archive.

Return JSON with two lists:
  "entities": [{"name": ..., "type": ..., "description": ...}]
  "relationships": [{"source": ..., "target": ..., "relation": ..., "description": ...}]

Entity types: recall, incident, supplier, equipment, product, material, test, other.
Use the names as they appear in the text. Every relationship's source and target must be the
name of an entity in your list. Extract only what this chunk states.

Chunk:
\"\"\"{chunk}\"\"\""""


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        raise SystemExit(f"usage: l44_extract.py MODEL, one of {list(MODELS)}")
    model = sys.argv[1]
    settings = MODELS[model]
    load_dotenv(REPO / ".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"], max_retries=3, timeout=120)
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"l44_extraction_{model.split('/')[-1]}.json"
    records, total, t0 = [], 0.0, time.time()
    for i, chunk in enumerate(tqdm(CORPUS, desc=model)):
        extra = {"provider": {"order": [settings["provider"]], "allow_fallbacks": False,
                              "require_parameters": True},
                 "usage": {"include": True}}
        if settings["reasoning"]:
            extra["reasoning"] = {"effort": "low"}
        req = {"model": model, "max_tokens": 4000, "extra_body": extra,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": PROMPT.replace("{chunk}", chunk)}]}
        if settings["temperature"]:
            req["temperature"] = 0
        resp = client.chat.completions.create(**req)
        if not resp.choices:
            raise RuntimeError(f"chunk {i}: provider error {getattr(resp, 'error', None)}")
        cost = getattr(resp.usage, "cost", None)
        if cost is None:
            raise RuntimeError(f"chunk {i}: OpenRouter returned no cost")
        total += cost
        text = resp.choices[0].message.content
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            log.error("chunk %d: reply is not JSON: %r", i, text[:200])
            raise RuntimeError(f"chunk {i}: invalid JSON") from e
        for key in ("entities", "relationships"):
            if not isinstance(parsed.get(key), list):
                raise RuntimeError(f"chunk {i}: reply has no list {key!r}: {text[:200]}")
        records.append({"chunk_index": i, "chunk": chunk, "reply": parsed,
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens, "cost": cost})
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": "l44_extract",
                                "model": model, "provider": settings["provider"],
                                "prompt_tokens": resp.usage.prompt_tokens,
                                "completion_tokens": resp.usage.completion_tokens,
                                "cost": cost}) + "\n")
    out_path.write_text(json.dumps({"model": model, "prompt": PROMPT, "total_cost": total,
                                    "records": records}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log.info("%s: %d chunks, $%.4f, %.0f s -> %s", model, len(records), total,
             time.time() - t0, out_path)


if __name__ == "__main__":
    main()
