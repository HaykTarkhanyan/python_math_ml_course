"""Compile out/*.json into a single self-contained HTML report (out/report.html).

The notebook is the experiment and writes the JSONs; this script only reads them,
so the report can be rebuilt any time without rerunning anything. Opens the
report in the browser when done (pass --no-open to suppress).
"""

import argparse
import html
import json
import logging
import sys
from pathlib import Path

import plotly.graph_objects as go

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("logs/build_report.log", encoding="utf-8")],
)
log = logging.getLogger("build_report")

OUT = Path("out")
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

BASE_LAYOUT = dict(
    template="plotly_white", font=dict(size=13),
    margin=dict(l=10, r=90, t=40, b=40),  # right padding so bar-end labels fit
    height=380,
)


def load(name: str, required: bool = True):
    p = OUT / name
    if not p.exists():
        if required:
            log.error(f"missing required result file: {p}")
            raise FileNotFoundError(p)
        log.warning(f"optional result file missing: {p} - section will be skipped")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def esc(s) -> str:
    return html.escape(str(s))


def hbar(labels, values, color, title, xtitle, textfmt="{:.2f}"):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=color,
        text=[textfmt.format(v) for v in values], textposition="outside",
    ))
    fig.update_layout(title=title, xaxis_title=xtitle,
                      yaxis=dict(autorange="reversed"), **BASE_LAYOUT)
    fig.update_xaxes(range=[0, max(values) * 1.18])
    return fig


def table(headers, rows, widths):
    cols = "".join(f'<col style="width:{w}">' for w in widths)
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td{' ' + c[1] if isinstance(c, tuple) else ''}>"
                         f"{c[0] if isinstance(c, tuple) else esc(c)}</td>"
                         for c in row) + "</tr>"
        for row in rows)
    return (f'<table><colgroup>{cols}</colgroup>'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


def heat_cell(v, lo, med, hi):
    """Diverging green->white->red background, median pivot, alpha capped at 0.4."""
    if v <= med:
        a = 0.4 * (med - v) / max(med - lo, 1e-9)
        bg = f"rgba(46,139,87,{a:.2f})"
    else:
        a = 0.4 * (v - med) / max(hi - med, 1e-9)
        bg = f"rgba(217,0,18,{a:.2f})"
    return (str(v), f'style="background:{bg};text-align:center"')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    census = load("vocab_census.json")
    eff = load("corpus_efficiency.json")
    merges = load("merge_order.json")
    boom = load("token_explosion.json")
    gallery = load("word_gallery.json")
    meta = load("meta.json")
    glitch = load("glitch_candidates.json", required=False)
    live = load("live_probe_results.json", required=False)
    mcand = load("model_candidates.json", required=False)
    live_own = load("live_probe_per_model.json", required=False)
    fun = load("fun_probe_results.json", required=False)
    vis = load("visible_probe_results.json", required=False)
    comedy = load("comedy_probe_results.json", required=False)
    gemma = load("gemma_glitch.json", required=False)
    small = load("live_probe_per_model_small.json", required=False)
    local = load("local_base_probe.json", required=False)

    names = list(eff)
    labels = [eff[n]["label"] for n in names]
    figs = []

    # --- efficiency ---
    order = sorted(names, key=lambda n: -eff[n]["hy"]["chars_per_token"])
    figs.append(hbar([eff[n]["label"] for n in order],
                     [eff[n]["hy"]["chars_per_token"] for n in order], BLUE,
                     "Armenian characters per token (higher = cheaper)",
                     "chars / token, hy-wiki sample"))
    order_p = sorted(names, key=lambda n: eff[n]["armenian_premium"])
    fig = hbar([eff[n]["label"] for n in order_p],
               [eff[n]["armenian_premium"] for n in order_p], RED,
               "The Armenian premium vs English (lower = fairer)",
               "(tokens/char hy) / (tokens/char en)", textfmt="x{:.2f}")
    fig.add_vline(x=1.0, line_dash="dash", line_color="#555")
    figs.append(fig)

    # --- census ---
    order_c = sorted(names, key=lambda n: -census[n]["pct_armenian"])
    figs.append(hbar([census[n]["label"] for n in order_c],
                     [census[n]["pct_armenian"] for n in order_c], ORANGE,
                     "Share of vocabulary containing Armenian characters",
                     "% of vocab", textfmt="{:.2f}%"))

    census_rows = [[census[n]["label"], f"{census[n]['vocab_size']:,}",
                    f"{census[n]['n_armenian']:,}", census[n]["n_pure_armenian"],
                    census[n]["longest"][0]["surface"] if census[n]["longest"] else "-",
                    census[n]["longest"][0]["n_chars"] if census[n]["longest"] else 0]
                   for n in order_c]

    # --- merges ---
    merge_rows = []
    for n in names:
        m = merges.get(n, {})
        fa = m.get("first_armenian") or []
        if fa:
            merge_rows.append([eff[n]["label"], m["kind"], f"{fa[0]['rank']:,}",
                               f"{fa[0]['pct_through']}%", fa[0]["result"]])
        elif m.get("kind") == "sentencepiece scores":
            top = m["top_armenian"][0]
            merge_rows.append([eff[n]["label"], m["kind"], "-", "-",
                               f"top piece: {top['piece']} (score {top['score']})"])
        elif m.get("kind") == "vocab order":
            merge_rows.append([eff[n]["label"], m["kind"],
                               f"{m['first_armenian_index']:,}", "-",
                               m["first_armenian_token"]])
        else:
            merge_rows.append([eff[n]["label"], m.get("kind", "?"), "-", "-",
                               "NO ARMENIAN AT ALL"])

    # --- explosion ---
    boom_rows = []
    for n in sorted(names, key=lambda n: -boom[n]["pct_over_1"]):
        b = boom[n]
        w = b["worst"][0] if b["worst"] else None
        boom_rows.append([b["label"], f"{b['pct_over_1']}%",
                          b["mean_tokens_per_word"], b["n_unk_words"],
                          f"{w['word']} ({w['n_tokens']} tok / {w['n_chars']} ch)"
                          if w else "-"])
    worst_detail = []
    for n in names:
        for w in boom[n]["worst"][:3]:
            if w["ratio"] > 1:
                worst_detail.append([boom[n]["label"], w["word"], w["n_chars"],
                                     w["n_tokens"], w["ratio"],
                                     " | ".join(w["pieces"])])

    # --- gallery heatmap table ---
    words = gallery["words"]
    counts = [[gallery["tokenizers"][w][n]["n"] for n in names] for w in words]
    flat = sorted(v for row in counts for v in row)
    lo, hi = flat[0], flat[-1]
    med = flat[len(flat) // 2]
    gal_rows = [[w] + [heat_cell(c, lo, med, hi) for c in row]
                for w, row in zip(words, counts)]

    # --- glitch ---
    glitch_html = "<p><b>Glitch section skipped</b> - glitch_candidates.json not found (run the notebook without ATH_SKIP_WEIGHTS).</p>"
    if glitch:
        h = glitch["norm_hist"]
        centers = [(a + b) / 2 for a, b in zip(h["bin_edges"], h["bin_edges"][1:])]
        fig = go.Figure()
        fig.add_bar(x=centers, y=h["all"], name="all 250k tokens", marker_color=BLUE)
        fig.add_bar(x=centers, y=h["armenian"], name="Armenian tokens", marker_color=RED)
        fig.update_layout(title="XLM-R embedding L2 norms (log count)",
                          xaxis_title="embedding norm", yaxis_type="log",
                          barmode="overlay", **BASE_LAYOUT)
        fig.update_traces(opacity=0.75)
        figs.append(fig)

        pm = glitch.get("pca_map")
        ec = glitch.get("embedding_clusters")
        if pm:
            hover = [f"{t}<br>norm {n} · corpus count {c}"
                     for t, n, c in zip(pm["token"], pm["norm"], pm["count"])]
            groups = [
                ("Armenian tokens", [i for i in range(len(hover))
                                     if not pm["is_candidate"][i] and not pm["is_top"][i]],
                 dict(color=BLUE, size=5, opacity=0.35), "markers"),
                ("15 most frequent", [i for i in range(len(hover)) if pm["is_top"][i]],
                 dict(color=ORANGE, size=10), "markers"),
                ("glitch candidates", [i for i in range(len(hover)) if pm["is_candidate"][i]],
                 dict(color=RED, size=12, symbol="x"), "markers+text"),
            ]
            fig = go.Figure()
            for gname, idx, marker, mode in groups:
                fig.add_scatter(
                    x=[pm["x"][i] for i in idx], y=[pm["y"][i] for i in idx],
                    mode=mode, marker=marker, name=gname,
                    text=[pm["token"][i] for i in idx] if "text" in mode else None,
                    textposition="top center", textfont=dict(size=10),
                    hovertext=[hover[i] for i in idx], hoverinfo="text")
            fig.update_layout(title="XLM-R Armenian token embeddings - PCA map "
                                    "(hover for tokens)",
                              **{**BASE_LAYOUT, "height": 650})
            figs.append(fig)

        clus_html = ""
        if ec:
            crows = [[c["cluster"], c["size"], c["mean_norm"], c["median_count"],
                      c["n_candidates"], "  ".join(c["sample"])]
                     for c in sorted(ec["clusters"], key=lambda x: x["mean_norm"])]
            clus_html = f"""
        <h3>Embedding-space clustering (k-means, k={ec['k']})</h3>
        <p>The SolidGoldMagikarp method: cluster the token embeddings and see where
        the corpus-null candidates land. Mean cosine distance to the global
        embedding centroid: candidates <b>{ec['centroid_dist_candidates']}</b> vs
        other Armenian tokens <b>{ec['centroid_dist_rest']}</b>. Clusters sorted by
        mean embedding norm - the glitch zone is on top.</p>
        {table(['cluster', 'size', 'mean norm', 'median corpus count', 'candidates',
                'tokens nearest center'], crows,
               ['9%', '9%', '12%', '18%', '12%', '40%'])}
        """

        if gemma:
            gp = gemma["pca_map"]
            ghover = [f"{t}<br>norm {n} · count {c}"
                      for t, n, c in zip(gp["token"], gp["norm"], gp["count"])]
            idx_n = [i for i in range(len(ghover)) if gp["is_null"][i]]
            idx_r = [i for i in range(len(ghover)) if not gp["is_null"][i]]
            fig = go.Figure()
            fig.add_scatter(x=[gp["x"][i] for i in idx_r],
                            y=[gp["y"][i] for i in idx_r], mode="markers",
                            marker=dict(color=BLUE, size=7, opacity=0.5),
                            name="Armenian tokens",
                            hovertext=[ghover[i] for i in idx_r], hoverinfo="text")
            fig.add_scatter(x=[gp["x"][i] for i in idx_n],
                            y=[gp["y"][i] for i in idx_n], mode="markers+text",
                            marker=dict(color=RED, size=13, symbol="x"),
                            name="corpus-null (incl. live-glitch)",
                            text=[gp["token"][i] for i in idx_n],
                            textposition="top center", textfont=dict(size=10),
                            hovertext=[ghover[i] for i in idx_n], hoverinfo="text")
            fig.update_layout(title="Gemma 2 (9B) Armenian token embeddings - "
                                    "PCA map (hover for tokens)",
                              **{**BASE_LAYOUT, "height": 550})
            figs.append(fig)

        gemma_html = ""
        if gemma:
            gn_rows = [[r["internal"], r["count"], r["norm"]]
                       for r in gemma["armenian_by_norm"][:15]]
            gb_rows = [[b["internal"], b["id"], b["norm"], b["count"]]
                       for b in gemma["global_bottom30"][:15]]
            gemma_html = f"""
        <h3>Bonus deep dive: Gemma 2 embedding norms (fetched by byte range)</h3>
        <p>Same analysis for the tokenizer that failed the live repeat test.
        The embedding matrix (256k x 3584, bf16) was fetched from the 18 GB
        <code>{esc(gemma['repo'])}</code> repo via HTTP Range requests on the
        safetensors tensor offsets - 1.8 GB instead of 18. Median norm: all
        tokens {gemma['norm_median_all']}, Armenian {gemma['norm_median_armenian']}.
        {esc(gemma['note'])}.</p>
        <h4>Lowest-norm Armenian tokens in Gemma 2</h4>
        {table(['token', 'corpus count', 'emb norm'], gn_rows, ['50%', '25%', '25%'])}
        <h4>Gemma's global SolidGoldMagikarp zone (any script)</h4>
        {table(['token', 'id', 'emb norm', 'corpus count'], gb_rows,
               ['40%', '20%', '20%', '20%'])}
        """

        cand_rows = [[c["internal"], c["id"], c["norm"]]
                     for c in glitch["armenian_candidates"]]
        probe_rows = [[p.get("kind", "?"), p["word"], p.get("p_copy", "-"),
                       p.get("rank", "-"), p.get("model_says", p.get("note", ""))]
                      for p in glitch["repeat_probe"]]
        bottom_rows = [[b["internal"], b["id"], b["norm"], b["corpus_count"]]
                       for b in glitch["global_bottom30"][:15]]
        glitch_html = f"""
        <p>{glitch['n_corpus_null']:,} of {glitch['n_armenian_tokens']:,} Armenian
        tokens in XLM-R's vocab never occur once in {glitch['corpus_chars']:,}
        characters of Armenian Wikipedia. Cross-checked with embedding norms
        (undertrained tokens keep tiny norms) and a repeat-after-me MLM probe.</p>
        <h3>Armenian glitch candidates (corpus-null, lowest embedding norm)</h3>
        {table(['token', 'id', 'emb norm'], cand_rows, ['40%', '30%', '30%'])}
        <h3>Repeat-after-me probe (mask the 4th repetition)</h3>
        {table(['kind', 'word', 'P(copy)', 'rank', 'model answered'], probe_rows,
               ['12%', '25%', '15%', '13%', '35%'])}
        <h3>Global lowest-norm tokens (any script) - the SolidGoldMagikarp zone</h3>
        {table(['token', 'id', 'emb norm', 'corpus count'], bottom_rows,
               ['40%', '20%', '20%', '20%'])}
        {clus_html}
        {gemma_html}
        """

    # --- live probe (optional) ---
    def prompt_rows():
        """The exact templates, imported from the probe scripts so the report
        can never drift from what actually ran."""
        from comedy_probe import PROMPTS as COMEDY_P
        from fun_probe import PROMPTS as FUN_P
        from glitch_live_probe import TASKS as LIVE_T
        from visible_probe import prompts_for
        rows = []
        for battery, prompts in [
                ("live probe (XLM-R words + own candidates)", LIVE_T),
                ("fun probe", FUN_P),
                ("visible probe - word targets", prompts_for("ԲԱՌ")),
                ("visible probe - symbol targets", prompts_for("֏")),
                ("comedy probe", COMEDY_P)]:
            for name, t in prompts.items():
                rows.append([battery, name, (f"<code>{esc(t)}</code>", "")])
        return rows

    templates_html = f"""
<details><summary><b>All prompt templates</b> (exact strings sent, temperature 0
everywhere; <code>{{w}}</code> = the probed token; visible-probe rows shown
instantiated for ԲԱՌ / ֏)</summary>
{table(['battery', 'prompt name', 'template'], prompt_rows(),
       ['24%', '14%', '62%'])}
</details>
"""

    live_html = ""
    if live:
        srows = []
        for tname, s in live["summary"].items():
            srows.append([s["model"],
                          s.get("repeat_control", "-"), s.get("repeat_glitch", "-"),
                          s.get("repeat_long", "-"),
                          s.get("spell_control", "-"), s.get("spell_glitch", "-"),
                          s.get("spell_long", "-"),
                          f"${s['cost_usd']:.4f}"])
        fails = [e for e in live["ledger"] if not e["ok"]]
        frows = [[e["model"].split("/")[1], e["task"], e["word"],
                  e["output"][:90] or "(empty)"] for e in fails[:24]]
        more = f"<p class='meta'>... plus {len(fails) - 24} more failures in the JSON.</p>" \
               if len(fails) > 24 else ""
        live_html = f"""
<h2>6 · Live model probe (OpenRouter)</h2>
<p>Served models asked to <i>repeat</i> and <i>spell</i> each word, temperature 0
(n ok / n asked). These words are the <b>XLM-R</b> candidates + controls - a
cross-vocab test. Total spend: <b>${live['total_cost_usd']:.4f}</b>. Ledger:
<code>out/live_probe_results.json</code>.</p>
{templates_html}
{table(['model', 'repeat ctrl', 'repeat glitch', 'repeat long',
        'spell ctrl', 'spell glitch', 'spell long', 'cost'],
       srows, ['26%', '11%', '11%', '11%', '11%', '11%', '11%', '8%'])}
<h3>Failures</h3>
{table(['model', 'task', 'word', 'model output'], frows,
       ['18%', '10%', '22%', '50%'])}
{more}
"""

    if mcand:
        mrows = []
        for tname, m in mcand.items():
            tops = ", ".join(f"{c['surface']} ({c['count']})"
                             for c in m["candidates"][:6] if c["surface"])
            mrows.append([m["label"], f"{m['n_armenian']:,}",
                          f"{m['n_corpus_null']:,}", tops])
        live_html += f"""
<h3>Each model's OWN vocab candidates</h3>
<p>The XLM-R candidates only diagnose XLM-R - so here each served model's own
vocabulary is mined for Armenian tokens that are absent or rarest in the same
20M-char corpus (count in parentheses). No embedding norms for these models
(weights gated or multi-GB), so this is corpus evidence only.</p>
{table(['tokenizer', 'Armenian tokens', 'corpus-null', 'rarest tokens (count)'],
       mrows, ['24%', '14%', '12%', '50%'])}
"""

    if live_own:
        orows = [[s["model"], s.get("repeat_own", "-"), s.get("spell_own", "-"),
                  f"${s['cost_usd']:.4f}"]
                 for s in live_own["summary"].values()]
        ofails = [e for e in live_own["ledger"] if not e["ok"]]
        ofrows = [[e["model"].split("/")[1], e["task"], e["word"],
                   e["output"][:90] or "(empty)"] for e in ofails[:40]]
        live_html += f"""
<h3>Probing each model with its own candidates</h3>
<p>The true SolidGoldMagikarp protocol: every probed string is a single token
in that model's OWN vocabulary. Spend: <b>${live_own['total_cost_usd']:.4f}</b>.
Ledger: <code>out/live_probe_per_model.json</code>.</p>
{table(['model', 'repeat own-candidates', 'spell own-candidates', 'cost'],
       orows, ['34%', '25%', '25%', '16%'])}
<h4>Failures</h4>
{table(['model', 'task', 'word', 'model output'], ofrows,
       ['18%', '10%', '22%', '50%'])}
"""

    if fun:
        frows = [[e["model"].split("/")[1], e["word"], e["prompt"],
                  e["output"][:200] or "(empty)"] for e in fun["ledger"]]
        live_html += f"""
<h3>Interrogating the confirmed glitch tokens</h3>
<p>The SolidGoldMagikarp battery on the tokens that failed the repeat test:
definitions, identity checks, stories, counts - and the quoted-vs-bare contrast
(quotes change the token boundary, so a model that cannot repeat the bare token
often can repeat the quoted one: the glitch lives in the token, not the string).
Spend: <b>${fun['total_cost_usd']:.4f}</b>. Ledger:
<code>out/fun_probe_results.json</code>.</p>
{table(['model', 'token', 'prompt', 'model output'], frows,
       ['14%', '12%', '12%', '62%'])}
"""

    if vis:
        vrows = [[e["model"].split("/")[1], e["word"], e["prompt"],
                  ("✓" if e["ok"] else "✗"), e["output"][:180] or "(empty)"]
                 for e in vis["ledger"]]
        n_ok = sum(e["ok"] for e in vis["ledger"])
        live_html += f"""
<h3>Visible-input probe: the model definitely sees the word</h3>
<p>Every prompt spells the target through guaranteed-visible tokens (single
letters, fragments, or an ASCII codepoint) and demands the word in the OUTPUT,
still at temperature 0. Input blindness is bypassed - any ✗ left is the model
failing to <i>emit</i> the string it just assembled. Result:
<b>{n_ok}/{len(vis['ledger'])}</b> outputs contain the target. Spend:
<b>${vis['total_cost_usd']:.4f}</b>. Ledger:
<code>out/visible_probe_results.json</code>.</p>
{table(['model', 'target', 'prompt', 'ok', 'model output'], vrows,
       ['14%', '12%', '13%', '5%', '56%'])}
"""

    if small and live_own:
        srows2 = []
        for tname, s in small["summary"].items():
            big = live_own["summary"].get(tname, {})
            srows2.append([tname, big.get("model", "-"),
                           big.get("repeat_own", "-"), big.get("spell_own", "-"),
                           s["model"], s.get("repeat_own", "-"),
                           s.get("spell_own", "-")])
        sfails = [e for e in small["ledger"] if not e["ok"] and e["task"] == "repeat"]
        sfrows = [[e["model"].split("/")[1], e["word"],
                   e["output"][:90] or "(empty)"] for e in sfails]
        live_html += f"""
<h3>Smaller siblings, same vocab</h3>
<p>The own-candidate battery rerun on smaller models sharing the tokenizer -
the hypothesis: less capacity to paper over a dead embedding. Spend:
<b>${small['total_cost_usd']:.4f}</b>. Ledger:
<code>out/live_probe_per_model_small.json</code>.</p>
{table(['tokenizer', 'big model', 'repeat', 'spell',
        'small model', 'repeat', 'spell'], srows2,
       ['10%', '22%', '11%', '11%', '24%', '11%', '11%'])}
<h4>Small-model repeat failures</h4>
{table(['model', 'word', 'output'], sfrows, ['24%', '24%', '52%'])}
"""

    if local:
        lrows = [[r["word"], r["corpus_count"] if r["corpus_count"] is not None
                  else "control", r["n_tokens"], r["p_copy_first_token"],
                  r["copy_rank"], ("✓" if r["echo_ok"] else "✗"),
                  r["raw_continuation"][:120] or "(empty)"]
                 for r in local["results"]]
        live_html += f"""
<h3>Local base model: {esc(local['model'])} on CPU</h3>
<p>The purest protocol, free of charge: the raw candidate token as the ENTIRE
input, greedy continuation (no chat template, no instructions) - plus a few-shot
echo test with the exact copy probability of the first target token. Base
models are where classic glitch behavior lives. Ledger:
<code>out/local_base_probe.json</code>.</p>
{table(['string', 'corpus count', 'tokens', 'P(copy)', 'copy rank', 'echo',
        'raw greedy continuation'], lrows,
       ['10%', '10%', '7%', '10%', '10%', '6%', '47%'])}
"""

    if comedy:
        crows2 = [[e["model"].split("/")[1], f"{e['word']} ({e['corpus_count']})",
                   e["prompt"], e["output"][:200] or "(empty)"]
                  for e in comedy["ledger"]]
        live_html += f"""
<h3>The comedy set: bare rare tokens, evasion-proof questions</h3>
<p>All probeable candidates of every model (corpus-null AND rare - the rare ones
are where the confident nonsense lives), asked bare-token questions that cannot
be dodged. Token's corpus count in parentheses. Temperature 0 throughout.
Spend: <b>${comedy['total_cost_usd']:.4f}</b>. Ledger:
<code>out/comedy_probe_results.json</code>.</p>
{table(['model', 'token (count)', 'prompt', 'model output'], crows2,
       ['14%', '15%', '11%', '60%'])}
"""

    # --- assemble ---
    fig_html = [f.to_html(full_html=False,
                          include_plotlyjs="inline" if i == 0 else False,
                          config={"displaylogo": False})
                for i, f in enumerate(figs)]

    hy = eff[names[0]]["hy"]
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Armenian Glitch Token Hunt</title>
<style>
 body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 1100px;
        margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
 h1 {{ border-bottom: 3px solid {ORANGE}; padding-bottom: 8px; }}
 h2 {{ margin-top: 40px; color: {BLUE}; }}
 table {{ border-collapse: collapse; width: 100%; table-layout: fixed;
         margin: 12px 0; font-size: 14px; }}
 th, td {{ border: 1px solid #ddd; padding: 6px 8px; overflow-wrap: break-word;
          text-align: left; vertical-align: top; }}
 th {{ background: #f5f5f5; }}
 .meta {{ color: #666; font-size: 13px; }}
</style></head><body>
<h1>Armenian Glitch Token Hunt</h1>
<p class="meta">run {esc(meta['run_at'])} · hy corpus {hy['n_chars']:,} chars
({meta['n_hy_articles']:,} wiki articles) · en baseline
{eff[names[0]]['en']['n_chars']:,} chars · 8 tokenizers ·
report rebuilt from out/*.json by build_report.py</p>

<h2>1 · Corpus efficiency</h2>
<p>Same Armenian Wikipedia sample through every tokenizer. The premium chart is
the exchange rate: how many times more tokens per character Armenian costs than
English through the same tokenizer.</p>
{fig_html[0]}{fig_html[1]}

<h2>2 · Vocab census</h2>
{fig_html[2]}
{table(['tokenizer', 'vocab', 'Armenian tokens', 'pure Armenian',
        'longest Armenian token', 'chars'],
       census_rows, ['26%', '12%', '14%', '13%', '25%', '10%'])}

<h2>3 · Merge archaeology</h2>
<p>For BPE tokenizers: the rank of the first Armenian merge - how far down the
training corpus's priority list Armenian sat. XLM-R (no merges) shows its
top-scored SentencePiece piece, mBERT its first Armenian vocab index.</p>
{table(['tokenizer', 'evidence', 'first Armenian rank', '% through vocab',
        'string'], merge_rows, ['26%', '20%', '16%', '14%', '24%'])}

<h2>4 · Token explosion</h2>
<p>Armenian words that cost more tokens than they have letters (encoded bare;
an unmerged Armenian letter is 2 UTF-8 bytes = up to 2 tokens).</p>
{table(['tokenizer', 'words with ratio > 1', 'mean tokens/word', 'invisible (unk) words',
        'worst word'],
       boom_rows, ['24%', '17%', '15%', '15%', '29%'])}
<h3>Worst offenders in detail</h3>
{table(['tokenizer', 'word', 'chars', 'tokens', 'ratio', 'pieces'],
       worst_detail, ['20%', '18%', '8%', '8%', '8%', '38%'])}

<h2>5 · Glitch hunt (XLM-R deep dive)</h2>
{glitch_html}
{''.join(fig_html[3:])}

{live_html}

<h2>7 · Word gallery</h2>
<p>Token counts for everyday Armenian words (green = few tokens, red = many).</p>
{table(['word'] + [eff[n]['label'].split(' (')[0] for n in names], gal_rows,
       ['19%'] + [f'{81 / len(names):.0f}%'] * len(names))}

<p class="meta">Rebuild: <code>python build_report.py</code> · experiment:
<code>armenian_token_hunt.ipynb</code> · playground:
<code>python tokenizer_app.py</code></p>
</body></html>"""

    out_path = OUT / "report.html"
    out_path.write_text(doc, encoding="utf-8")
    log.info(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")

    if not args.no_open:
        import os
        os.startfile(out_path.resolve())  # noqa - Windows-only by design
        log.info("opened in browser")


if __name__ == "__main__":
    sys.exit(main())
