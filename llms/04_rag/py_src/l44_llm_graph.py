"""Build the graph a real LLM extracted (l44_extract.py output) and measure what it answers.

No API calls: it reads results/l44_extraction_*.json. For each extractor model and two levels
of entity resolution it measures, for the recall question:

  * walk-h       - supplier-typed nodes within h hops of the recall-typed nodes (h = 2, 3, 4)
  * local search - GraphRAG-style: the recall nodes and their direct neighbours, then the
                   chunks those entities were extracted from go into the prompt. Counts the
                   right and wrong suppliers named in those chunks, and the chunks used.

Resolution levels:
  raw       - names lowercased and stripped, nothing else
  id-merge  - plus one rule: a name that contains an ID code (INC-031, RC-2019-02, PRS-220)
              becomes that code, so "the PRS-220 press" and "PRS-220" are one node

Writes: logs/l44_llm_graph.log, results/l44_llm_graph.json
Run (~2 s):  ./ma/.venv/Scripts/python.exe llms/04_rag/py_src/l44_llm_graph.py
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from l44_data import CORPUS, RECALL_ANSWER, RECALL_DISTRACTORS

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOGS / "l44_llm_graph.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

CODE = re.compile(r"\b([A-Z]{2,3}-\d{2,4}(?:-\d{2})?)\b", re.IGNORECASE)


def norm_raw(name):
    return " ".join(name.lower().strip().split())


def norm_id(name):
    m = CODE.search(name)
    return m.group(1).lower() if m else norm_raw(name)


def build(records, norm):
    g = nx.Graph()
    types = defaultdict(Counter)
    sources = defaultdict(set)
    surface = defaultdict(set)
    for r in records:
        i = r["chunk_index"]
        ents = r["reply"]["entities"]
        for e in ents:
            n = norm(str(e.get("name", "")))
            if not n:
                continue
            g.add_node(n)
            types[n][str(e.get("type", "other")).lower()] += 1
            sources[n].add(i)
            surface[n].add(str(e["name"]))
        for rel in r["reply"]["relationships"]:
            a, b = norm(str(rel.get("source", ""))), norm(str(rel.get("target", "")))
            if a and b and a != b:
                g.add_edge(a, b)
                sources[a].add(i)
                sources[b].add(i)
    node_type = {n: types[n].most_common(1)[0][0] if types[n] else "other" for n in g.nodes}
    return g, node_type, sources, surface


def supplier_hits(names):
    low = [n.lower() for n in names]
    right = sorted(s for s in RECALL_ANSWER if any(s.lower() in n for n in low))
    wrong = sorted(s for s in RECALL_DISTRACTORS if any(s.lower() in n for n in low))
    return right, wrong


def analyse(records, level, norm):
    g, node_type, sources, surface = build(records, norm)
    recalls = sorted(n for n in g.nodes if node_type[n] == "recall")
    out = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "recall_nodes": recalls}
    # name variants of the same machine / incident
    variants = {}
    for code in ("prs-220", "prs-400", "inc-031", "rc-2019-02"):
        names = sorted({s for n in g.nodes if code in n.replace(" ", "-") for s in surface[n]})
        nodes = sorted(n for n in g.nodes if code in n.replace(" ", "-"))
        variants[code] = {"surface_names": names, "nodes": nodes}
    out["variants"] = variants
    # walk-h
    for h in (2, 3, 4):
        reach = set()
        for rc in recalls:
            reach |= set(nx.single_source_shortest_path_length(g, rc, cutoff=h))
        sup = [n for n in reach if node_type[n] == "supplier"]
        right, wrong = supplier_hits(sup)
        out[f"walk_{h}"] = {"right": right, "wrong": wrong}
    # local search: recall nodes + neighbours -> their source chunks
    hood = set(recalls)
    for rc in recalls:
        hood |= set(g.neighbors(rc))
    chunks = sorted({i for n in hood for i in sources[n]})
    text = " ".join(CORPUS[i] for i in chunks)
    right = sorted(s for s in RECALL_ANSWER if s in text)
    wrong = sorted(s for s in RECALL_DISTRACTORS if s in text)
    out["local_search"] = {"neighbourhood": sorted(hood), "chunks": chunks,
                           "n_chunks": len(chunks), "right": right, "wrong": wrong}
    log.info("  [%s] %d nodes, %d edges; recall nodes %s", level, out["nodes"], out["edges"],
             recalls)
    for code, v in variants.items():
        log.info("    %-10s %d node(s) %s from names %s", code, len(v["nodes"]), v["nodes"],
                 v["surface_names"])
    for h in (2, 3, 4):
        w = out[f"walk_{h}"]
        log.info("    walk-%d: right %d/4 %s | wrong %d %s", h, len(w["right"]), w["right"],
                 len(w["wrong"]), w["wrong"])
    ls = out["local_search"]
    log.info("    local search: %d chunks %s | right %d/4 %s | wrong %d %s", ls["n_chunks"],
             ls["chunks"], len(ls["right"]), ls["right"], len(ls["wrong"]), ls["wrong"])
    return out


def main():
    files = sorted(RESULTS.glob("l44_extraction_*.json"))
    if not files:
        raise FileNotFoundError("no results/l44_extraction_*.json - run l44_extract.py first")
    out = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        log.info("=" * 78)
        log.info("%s (%d chunks, extraction cost $%.4f)", d["model"], len(d["records"]),
                 d["total_cost"])
        if len(d["records"]) != len(CORPUS):
            raise ValueError(f"{f.name}: {len(d['records'])} records for {len(CORPUS)} chunks")
        out[d["model"]] = {"extraction_cost": d["total_cost"],
                           "raw": analyse(d["records"], "raw", norm_raw),
                           "id-merge": analyse(d["records"], "id-merge", norm_id)}
    path = RESULTS / "l44_llm_graph.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
