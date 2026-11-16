"""Build the standalone HTML report from results/skills_<date>.json (nothing is re-collected).

Run (repo root):  ./ma/Scripts/python.exe _knowledge/ml_job_market/py_src/report.py [YYYY-MM-DD]
Takes ~3 s. Writes ML_JOB_MARKET_<date>.html in _knowledge/ml_job_market/ with plotly.js inlined
once (~5 MB), so the file can be sent as is. The commentary comes from notes/implications_<date>.html.
"""
import html
import json
import sys
import time
from statistics import median

import plotly.graph_objects as go

from common import HERE, RESULTS, get_logger

log = get_logger("report")
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
SECTIONS = [("summary", "Summary"), ("how", "How this was collected, and its limits"),
            ("implications", "What it means for the course"), ("who", "Who is hiring in Armenia"),
            ("skills", "Skills: Armenia vs worldwide"), ("genai", "LLM and GenAI skills in detail"),
            ("seniority", "Seniority and experience"), ("analyst", "Data analyst roles in Armenia"),
            ("am_list", "Every Armenian posting"), ("gl_list", "The worldwide sample"),
            ("published", "Published reports, for context")]
_first_fig = [True]


def fig_html(fig):
    inc = "inline" if _first_fig[0] else False
    _first_fig[0] = False
    return fig.to_html(full_html=False, include_plotlyjs=inc, config={"displaylogo": False, "responsive": True})


def hbar(cats, series, title, pct=True):
    fig = go.Figure()
    for name, vals, color in series:
        fig.add_bar(y=cats, x=vals, name=name, orientation="h", marker_color=color, cliponaxis=False,
                    text=[(f"{v:.0f}%" if pct else f"{v:g}") for v in vals], textposition="outside",
                    hovertemplate="%{y}: %{x:.1f}" + ("%" if pct else "") + f"<extra>{html.escape(name)}</extra>")
    mx = max(max(v) for _, v, _ in series) or 1
    fig.update_layout(title=title, barmode="group", template="plotly_white", font=dict(size=13),
                      height=90 + len(cats) * (14 + 13 * len(series)),
                      margin=dict(l=10, r=60, t=60, b=30), legend=dict(orientation="h", y=1.0, x=0, yanchor="bottom"),
                      xaxis=dict(range=[0, mx * 1.18], ticksuffix="%" if pct else ""), yaxis=dict(autorange="reversed"))
    return fig_html(fig)


def share(g, name):
    """Share of postings that mention the skill."""
    return 100 * g["skills"][name] / g["n"] if g["n"] else 0.0


def cshare(g, name):
    """Share of companies with at least one posting that mentions the skill: one employer counts once."""
    return 100 * g["skills_companies"][name] / g["n_companies"] if g["n_companies"] else 0.0


def diverging(vals):
    """Background colours: green above the median, red below, white at it; alpha capped at 0.4."""
    if not vals:
        return []
    mid = median(vals)
    span = max(abs(v - mid) for v in vals) or 1
    out = []
    for v in vals:
        a = min(0.4, 0.4 * abs(v - mid) / span)
        out.append(f"rgba(27,127,59,{a:.2f})" if v > mid else f"rgba(217,0,18,{a:.2f})" if v < mid else "transparent")
    return out


def postings_table(ps, meta_cat):
    rows = []
    for p in sorted(ps, key=lambda p: (p["role_label"], p["company"])):
        sk = [s for s in p["skills"] if meta_cat[s] != "Other requirements"]
        rows.append(f"<tr><td><a href='{html.escape(p['url'])}'>{html.escape(p['title'])}</a></td>"
                    f"<td>{html.escape(p['company'])}</td><td>{html.escape(p['role_label'])}</td>"
                    f"<td>{p['seniority']}</td><td>{'' if p['years_min'] is None else p['years_min']}</td>"
                    f"<td>{p['source']}</td><td>{html.escape(', '.join(sk))}</td></tr>")
    return ("<table><colgroup><col style='width:22%'><col style='width:13%'><col style='width:12%'><col style='width:7%'>"
            "<col style='width:5%'><col style='width:7%'><col style='width:34%'></colgroup>"
            "<tr><th>Title</th><th>Company</th><th>Role</th><th>Level</th><th>Yrs</th><th>Source</th><th>Skills found</th></tr>"
            + "".join(rows) + "</table>")


def section(sid, title, body):
    return f"<details open id='{sid}'><summary><h2>{title}</h2></summary>{body}</details>"


def main():
    t0 = time.time()
    day = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(RESULTS.glob(f"skills_{day or '*'}.json"))
    if not files:
        raise SystemExit("no results/skills_*.json - run analyze.py first")
    R = json.loads(files[-1].read_text(encoding="utf-8"))
    snap, G, inp = R["snapshot"], R["groups"], R["inputs"]
    meta_cat = {m["name"]: m["category"] for m in R["skills_meta"]}
    cats = list(dict.fromkeys(m["category"] for m in R["skills_meta"]))
    am, gl = G.get("AM_core"), G.get("GL_core")
    if not am or not gl:
        raise SystemExit(f"missing a core group in {files[-1].name}: {list(G)}")
    P = R["postings"]
    S = {}

    # ---- summary
    n_of = lambda g: G.get(g, {"n": 0})["n"]
    top_comp = max(am["companies"], key=am["companies"].get)
    top_am = sorted((s for s in meta_cat if meta_cat[s] != "Other requirements"), key=lambda s: -cshare(am, s))[:15]
    cards = [(am["n"], f"ML / AI / DS postings open in Armenia, from {am['n_companies']} companies"),
             (n_of("AM_swe_ai"), "Armenian software roles that ask for AI"),
             (n_of("AM_analyst"), "Armenian data analyst / engineer postings"),
             (n_of("AM_other"), "other Armenian AI roles (AI data work, AI-titled non-engineering)"),
             (gl["n"], f"worldwide comparison sample (neutral searches), from {gl['n_companies']} companies")]
    S["summary"] = ("<div class='cards'>" + "".join(f"<div class='card'><b>{n}</b><span>{t}</span></div>" for n, t in cards) + "</div>"
                    + f"<p><b>{html.escape(top_comp)}</b> alone has {am['companies'][top_comp]} of the {am['n']} Armenian ML/AI/DS postings, "
                    "many from the same templates. So skills are shown two ways: share of <i>postings</i>, and share of <i>companies</i> "
                    "(a company counts once if any of its postings asks for the skill). The company share is the fairer one.</p>"
                    + "<p>Fifteen skills asked by the most Armenian employers (technical skills only):</p>"
                    + "<table><colgroup><col style='width:34%'><col style='width:17%'><col style='width:17%'><col style='width:17%'><col style='width:15%'></colgroup>"
                    + "<tr><th>Skill</th><th>Armenia, companies</th><th>Armenia, postings</th><th>Worldwide, companies</th><th>Category</th></tr>"
                    + "".join(f"<tr><td>{s}</td><td>{cshare(am, s):.0f}%</td><td>{share(am, s):.0f}%</td><td>{cshare(gl, s):.0f}%</td>"
                              f"<td>{meta_cat[s]}</td></tr>" for s in top_am)
                    + "</table>")

    # ---- how
    q = inp["linkedin_queries"]
    S["how"] = f"""<p>Snapshot of <b>{snap}</b>. Open postings only, so this is what is being hired for this month, not a trend.</p>
<ul>
<li><b>staff.am</b>: all {inp['staffam_listings']} open listings walked page by page (its keyword search is fuzzy); {inp['staffam_fetched']} with an
  ML/AI/data title or in an IT/data category had their full text fetched ({inp['staffam_failures']} failed).</li>
<li><b>LinkedIn</b>, public logged-out search: {inp['linkedin_cards']} result cards for the queries
  Armenia: {', '.join(q['armenia'])}; worldwide: {', '.join(q['global'])}. {inp['linkedin_fetched']} postings with a matching title were
  fetched in full ({inp['linkedin_failures']} failed). The Armenia searches use LinkedIn's country id: the place name
  "Armenia" returns jobs in Armenia, Colombia.</li>
<li>{inp['duplicates_merged']} postings that appeared twice (same company and title in the same market) were merged.</li>
<li>A posting counts as ML/AI/DS when its <b>title</b> matches a role pattern; other titles count only if the text names
  at least two ML terms. Skills are regex matches on title + text (presence, not frequency); the dictionary is in
  <code>py_src/analyze.py</code>.</li>
</ul>
<p class='note'><b>Limits.</b> The worldwide sample is LinkedIn's top results for four queries, so it leans to large companies and
the US. Regex tagging misses synonyms and can catch a word used in passing (e.g. "Docker" in a list of nice-to-haves counts
the same as a hard requirement). Small groups move a lot: one Armenian posting is {100 / am['n']:.1f}% of the Armenian group.</p>"""
    qc = R["query_check"]
    cols = list(qc[0]["shares"])
    S["how"] += ("<h3>Do the search words drive the numbers?</h3><p>Some searches select for LLM work (Armenia: AI engineer, LLM, "
                 "generative AI; worldwide: AI engineer, LLM engineer). Each posting is credited to the first search that found it; "
                 "the shares below split each market by where its postings came from.</p>"
                 "<table><tr><th>Market</th><th>Found by</th><th>Postings</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
                 + "".join(f"<tr><td>{'Armenia' if r['group'] == 'AM_core' else 'Worldwide'}</td><td>{r['origin']}</td><td>{r['n']}</td>"
                           + "".join(f"<td>{r['shares'][c]:.0f}%</td>" for c in cols) + "</tr>" for r in qc)
                 + "</table><p class='note'>Worldwide, the LLM-selecting searches return almost only LLM jobs, so the worldwide comparison "
                 "group in this report uses the neutral searches only (machine learning engineer, data scientist); the LLM-search postings "
                 "are used only in the LLM section. In Armenia those searches added almost no ML postings that the neutral ones had not "
                 "already found, so the Armenian group keeps everything.</p>")

    # ---- implications (hand-written per snapshot)
    imp = HERE / "notes" / f"implications_{snap}.html"
    S["implications"] = (imp.read_text(encoding="utf-8") if imp.exists() else
                         f"<p class='bad'>No commentary written for this snapshot yet ({imp.name} is missing).</p>")
    if not imp.exists():
        log.error(f"{imp} is missing - the report says so in red")

    # ---- who
    roles = list(dict.fromkeys(list(am["roles"]) + list(gl["roles"])))
    S["who"] = (hbar(roles, [("Armenia", [100 * am["roles"].get(r, 0) / am["n"] for r in roles], RED),
                             ("Worldwide sample", [100 * gl["roles"].get(r, 0) / gl["n"] for r in roles], BLUE)],
                     "Role mix (share of ML/AI/DS postings)")
                + "<p class='note'>The worldwide role mix mostly reflects which four queries were run; read the Armenian bars.</p>")
    comp = {}
    for p in P:
        if p["group"] in ("AM_core", "AM_swe_ai", "AM_other"):
            comp[p["company"]] = comp.get(p["company"], 0) + 1
    topc = sorted(comp, key=lambda c: -comp[c])[:25]
    S["who"] += hbar(topc, [("open ML/AI postings", [comp[c] for c in topc], RED)],
                     "Companies with the most open ML/AI postings in Armenia (ML/AI/DS + software-with-AI + other AI roles)", pct=False)

    # ---- skills
    ranked = sorted((s for s in meta_cat if meta_cat[s] != "Other requirements"), key=lambda s: -max(cshare(am, s), cshare(gl, s)))[:35]
    S["skills"] = hbar(ranked, [("Armenia, share of companies", [cshare(am, s) for s in ranked], RED),
                                ("Armenia, share of postings", [share(am, s) for s in ranked], ORANGE),
                                ("Worldwide sample, share of companies", [cshare(gl, s) for s in ranked], BLUE)],
                       "35 most-asked skills in ML/AI/DS postings")
    S["skills"] += "<p class='note'>Last column of each table: Armenia's company share minus the worldwide company share.</p>"
    for c in cats:
        names = [s for s in meta_cat if meta_cat[s] == c]
        diffs = [cshare(am, s) - cshare(gl, s) for s in names]
        cols = diverging(diffs)
        S["skills"] += (f"<h3>{c}</h3><table><colgroup><col style='width:34%'><col style='width:15%'><col style='width:15%'><col style='width:15%'><col style='width:21%'></colgroup>"
                        "<tr><th>Skill</th><th>Armenia, companies</th><th>Armenia, postings</th><th>Worldwide, companies</th><th>Armenia minus worldwide</th></tr>"
                        + "".join(f"<tr><td>{s}</td><td>{cshare(am, s):.0f}%</td><td>{share(am, s):.0f}%</td><td>{cshare(gl, s):.0f}%</td>"
                                  f"<td style='background:{col}'>{d:+.0f} pts</td></tr>" for s, d, col in zip(names, diffs, cols))
                        + "</table>")

    # ---- genai
    gen = [s for s in meta_cat if meta_cat[s] == "LLM & GenAI"]
    llm_am = [p for p in P if p["group"] == "AM_core" and "LLMs" in p["skills"]]
    llm_gl_neutral = [p for p in P if p["group"] == "GL_core" and "LLMs" in p["skills"]]
    llm_gl = [p for p in P if p["group"] in ("GL_core", "GL_llm") and "LLMs" in p["skills"]]
    sub = lambda ps, s: 100 * sum(s in p["skills"] for p in ps) / len(ps) if ps else 0.0
    S["genai"] = (f"<p>{len(llm_am)} of {am['n']} Armenian ML/AI/DS postings ({100 * len(llm_am) / am['n']:.0f}%) mention LLMs or GenAI; "
                  f"in the worldwide neutral sample {len(llm_gl_neutral)} of {gl['n']} ({100 * len(llm_gl_neutral) / gl['n']:.0f}%). "
                  f"Below, among LLM postings only (worldwide: all {len(llm_gl)} LLM postings, including those the LLM searches found):</p>"
                  + hbar(gen, [("Armenia", [sub(llm_am, s) for s in gen], RED), ("Worldwide, all LLM postings", [sub(llm_gl, s) for s in gen], BLUE)],
                         "What LLM postings ask for (share of postings that mention LLMs)"))

    # ---- seniority
    lv = ["entry", "mid", "senior", "unstated"]
    S["seniority"] = hbar(lv, [("Armenia", [100 * am["seniority"].get(x, 0) / am["n"] for x in lv], RED),
                               ("Worldwide sample", [100 * gl["seniority"].get(x, 0) / gl["n"] for x in lv], BLUE)],
                          "Seniority (from the title first, then the posting's level field, then years asked)")
    S["seniority"] += (f"<p>Median minimum years of experience asked, where stated: Armenia <b>{am['years_min_median']}</b> "
                       f"({am['years_stated']} of {am['n']} postings state it), worldwide <b>{gl['years_min_median']}</b> "
                       f"({gl['years_stated']} of {gl['n']}).</p>")

    # ---- analysts
    an = G.get("AM_analyst")
    if an and an["n"]:
        tops = sorted((s for s in meta_cat if meta_cat[s] != "Other requirements"), key=lambda s: -share(an, s))[:20]
        S["analyst"] = (f"<p>{an['n']} postings. Not ML roles, but the most common data job a graduate of this course can get in Armenia.</p>"
                        + hbar(tops, [("Armenian analyst / data engineer postings", [share(an, s) for s in tops], ORANGE)],
                               "Top 20 skills in Armenian data analyst / engineer postings"))
    else:
        S["analyst"] = "<p>No analyst postings in this snapshot.</p>"

    # ---- lists
    S["am_list"] = (postings_table([p for p in P if p["market"] == "armenia" and p["group"] != "AM_analyst"], meta_cat)
                    + "<h3>Data analyst / engineer postings</h3>" + postings_table([p for p in P if p["group"] == "AM_analyst"], meta_cat))
    S["gl_list"] = postings_table([p for p in P if p["market"] == "global"], meta_cat)
    S["published"] = (HERE / "notes" / "published_reports.html").read_text(encoding="utf-8")

    toc = "".join(f"<li><a href='#{sid}'>{t}</a></li>" for sid, t in SECTIONS)
    body = "".join(section(sid, f"{i + 1}. {t}", S[sid]) for i, (sid, t) in enumerate(SECTIONS))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ML job market {snap}</title>
<style>
body{{margin:0;background:#fff;color:#1d232b;font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:24px 24px 80px}} h1{{font-size:27px;margin:0 0 4px}}
.meta,.note{{color:#5b6470}} .note{{font-size:14px}} .bad{{color:#D90012;font-weight:700}}
nav{{border:1px solid #d9dde3;border-radius:8px;padding:10px 18px;margin:16px 0 22px}} nav ol{{columns:2;margin:4px 0}}
nav a{{color:#0033A0;text-decoration:none}}
details{{border:1px solid #d9dde3;border-radius:8px;margin:0 0 16px;padding:0 18px}} details[open]{{padding-bottom:14px}}
summary{{cursor:pointer;padding:12px 0}} summary h2{{display:inline;font-size:20px;margin:0}} h3{{font-size:16px;margin:18px 0 6px}}
table{{border-collapse:collapse;width:100%;table-layout:fixed;margin:8px 0 12px}}
th,td{{border:1px solid #d9dde3;padding:5px 8px;text-align:left;vertical-align:top;font-size:13.5px;overflow-wrap:anywhere}}
th{{background:#f5f6f8}} a{{color:#0033A0}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 14px}} .card{{border:1px solid #d9dde3;border-radius:8px;padding:10px 14px;min-width:200px}}
.card b{{display:block;font-size:26px;color:#D90012}} .card span{{font-size:13px;color:#5b6470}}
code{{background:#f5f6f8;padding:0 4px;border-radius:3px}}
</style></head><body><main>
<h1>ML and AI jobs: Armenia and a worldwide sample</h1>
<p class="meta">Snapshot {snap} from staff.am and LinkedIn. Built from <code>results/{files[-1].name}</code> by
<code>py_src/report.py</code>; raw postings in <code>data/</code>.</p>
<nav><b>Contents</b><ol>{toc}</ol></nav>{body}</main></body></html>"""
    out = HERE / f"ML_JOB_MARKET_{snap}.html"
    out.write_text(page, encoding="utf-8")
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
