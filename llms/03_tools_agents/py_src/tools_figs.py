"""Figures for the tools practical's session 2 deck (llms/03_tools_agents/06_mcp_skills.tex).

mcp_roles.pdf          - who talks to whom with MCP: the model, your app (the client), MCP servers
orchestrator.pdf       - one agent vs an orchestrator with workers, call by call (the notebook's part on it)

Both are schematic (boxes and arrows), drawn at slot size with 7-10 pt text.
Run (repo root, ~3 s):  ./ma/Scripts/python.exe llms/03_tools_agents/py_src/tools_figs.py
"""
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

CH = Path(__file__).resolve().parents[1]
FIG = CH / "fig"
LOGS = Path(__file__).resolve().parents[3] / "logs"
BLUE, RED, ORANGE, GREY = "#0033A0", "#D90012", "#F2A800", "#666666"
log = logging.getLogger("tools_figs")


def box(ax, x, y, w, h, text, color, fs=8, bold=False, fill_alpha=0.10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=color, ec=color, alpha=fill_alpha, lw=0))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc="none", ec=color, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="black", weight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text="", color=GREY, fs=7, off=(0, 0.05), style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, color=color, lw=1.1,
                                 linestyle=ls, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], text, ha="center", va="center", fontsize=fs,
                color=color)


def mcp_roles():
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    box(ax, 1.2, 2.0, 2.4, 1.2, "the model\n(via the API)", BLUE, fs=8, bold=True)
    box(ax, 6.1, 2.0, 3.4, 1.6, "your app\n= the MCP client\n(the loop)", RED, fs=8, bold=True)
    box(ax, 10.9, 3.3, 3.9, 0.75, "server: yerevan-tools", ORANGE, fs=7.5)
    box(ax, 10.9, 2.0, 3.9, 0.75, "server: a browser", ORANGE, fs=7.5)
    box(ax, 10.9, 0.7, 3.9, 0.75, "server: someone else's", ORANGE, fs=7.5)
    arrow(ax, 4.3, 2.4, 2.5, 2.4, "schemas +\nmessages", fs=7, off=(0, 0.42))
    arrow(ax, 2.5, 1.6, 4.3, 1.6, "tool_calls", fs=7, off=(0, -0.3))
    for y in (3.3, 2.0, 0.7):
        arrow(ax, 7.8, 2.0 + (y - 2.0) * 0.35, 8.9, y, style="<|-|>")
    ax.text(8.3, 3.95, "MCP: tools/list, tools/call", fontsize=7, color=GREY, ha="center")
    ax.text(1.3, 0.75, "never speaks MCP", fontsize=7, color=BLUE, ha="center", style="italic")
    fig.savefig(FIG / "mcp_roles.pdf", bbox_inches="tight")
    plt.close(fig)


def orchestrator():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.5), gridspec_kw={"width_ratios": [1, 1.55]})
    for ax in (a1, a2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
    # one agent
    a1.set_title("one agent: 2 calls", fontsize=9, color=BLUE)
    box(a1, 5, 5.0, 8.6, 0.8, "call 1: asks for 3 tools\nat once", BLUE, fs=7.5)
    box(a1, 5, 3.4, 8.6, 0.9, "your code runs\nweather, search, weather", GREY, fs=7.5)
    box(a1, 5, 1.8, 8.6, 0.8, "call 2: the answer", BLUE, fs=7.5)
    arrow(a1, 5, 4.6, 5, 3.85)
    arrow(a1, 5, 2.95, 5, 2.2)
    a1.text(5, 0.7, "one conversation holds everything", fontsize=7.5, color=GREY, ha="center", style="italic")
    # orchestrator + workers
    a2.set_title("orchestrator + 3 workers: 8 calls", fontsize=9, color=RED)
    box(a2, 5, 5.2, 9.6, 0.7, "call 1: orchestrator writes the plan (3 subtasks)", RED, fs=7.5)
    for i, x in enumerate((1.7, 5.0, 8.3)):
        box(a2, x, 3.3, 3.0, 1.5, f"worker {i + 1}\ncall: a tool\ncall: an answer", ORANGE, fs=7.5)
        arrow(a2, 5 + (x - 5) * 0.4, 4.85, x, 4.05)
        arrow(a2, x, 2.55, 5 + (x - 5) * 0.4, 1.75)
    box(a2, 5, 1.35, 9.6, 0.7, "call 8: orchestrator merges the findings", RED, fs=7.5)
    a2.text(5, 0.35, "each worker starts from an empty conversation", fontsize=7, color=GREY, ha="center",
            style="italic")
    fig.tight_layout()
    fig.savefig(FIG / "orchestrator.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    LOGS.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    for h in (logging.StreamHandler(), logging.FileHandler(LOGS / "tools_figs.log", encoding="utf-8")):
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    t0 = time.time()
    mcp_roles()
    orchestrator()
    log.info(f"wrote mcp_roles.pdf, orchestrator.pdf to {FIG} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
