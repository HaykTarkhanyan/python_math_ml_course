"""Figures for LLM-2's long-context frames (ml/14_llms/LLM2_embeddings_position.tex) drawn from
PUBLISHED numbers, so they need no model weights (embeddings_position_figs.py needs GPT-2's).

  emb_lost_middle.pdf  Liu et al. (TACL 2024), "Lost in the Middle", Table 6 (20 documents,
                       GPT-3.5-Turbo 0613) and Table 1 (closed-book, oracle). Values checked
                       against the paper on 2026-10-27 (ar5iv 2307.03172).

Writes ml/14_llms/results/long_context_figs.json (the plotted values and their source).
Run: ./ma/Scripts/python.exe ml/14_llms/py_src/long_context_figs.py   (~3 s)
"""
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG, RESULTS, LOGS = CH / "fig", CH / "results", ROOT / "logs"

W = 5.5
FS, FS_SMALL = 8, 7
ARMRED, ARMBLUE, ARMORANGE, GREY = "#D90012", "#0033A0", "#F2A800", "#777777"

# Liu et al., Table 6: answer document at index 0, 4, 9, 14, 19 of 20 (shown 1-based)
LOST = {"source": "Liu et al. (2024), Lost in the Middle, TACL; Table 6 and Table 1",
        "model": "gpt-3.5-turbo-0613", "documents": 20,
        "position": [1, 5, 10, 15, 20], "accuracy": [75.8, 57.2, 53.8, 55.4, 63.2],
        "closed_book": 56.1, "oracle": 88.3}


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("long_context_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "long_context_figs.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def fig_lost_middle(log):
    d = LOST
    # the slide's claim: two middle positions score below having no documents at all
    below = [p for p, a in zip(d["position"], d["accuracy"]) if a < d["closed_book"]]
    assert below == [10, 15], below
    fig, ax = plt.subplots(figsize=(W * 0.62, 1.95))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.22)
    ax.plot(d["position"], d["accuracy"], color=ARMBLUE, marker="o", ms=4, lw=1.6)
    for x, y in zip(d["position"], d["accuracy"]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=FS_SMALL, color=ARMBLUE)
    ax.axhline(d["oracle"], color=GREY, ls="--", lw=1)
    ax.text(20.4, d["oracle"] + 1.0, f"only the right document: {d['oracle']:.1f}", ha="right",
            va="bottom", fontsize=FS_SMALL, color=GREY)
    ax.axhline(d["closed_book"], color=ARMRED, ls="--", lw=1)
    ax.text(0.4, 46.5, f"no documents at all (dashed): {d['closed_book']:.1f}", ha="left",
            va="bottom", fontsize=FS_SMALL, color=ARMRED)
    ax.set_xticks(d["position"])
    ax.set_xlim(0, 21)
    ax.set_ylim(45, 95)
    ax.set_xlabel("position of the answer among 20 documents", fontsize=FS_SMALL)
    ax.set_ylabel("accuracy, %", fontsize=FS_SMALL)
    ax.set_title("GPT-3.5-Turbo, 20 documents per question", fontsize=FS)
    ax.tick_params(labelsize=FS_SMALL, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out = FIG / "emb_lost_middle.pdf"
    fig.savefig(out, metadata={"CreationDate": None})
    plt.close(fig)
    log.info(f"saved {out.name}; positions below closed-book: {below}")
    return {"lost_in_the_middle": d, "below_closed_book": below}


def main():
    log = setup_logging()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    R = {"date": time.strftime("%Y-%m-%d")}
    R.update(fig_lost_middle(log))
    (RESULTS / "long_context_figs.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
    log.info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
