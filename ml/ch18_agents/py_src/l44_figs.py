"""Figures for L44 (agents and tool use).

  01  ILLUSTRATIVE - the five-step trace: the model decides, your code does.
  02  ILLUSTRATIVE - the four approaches as a ladder over three questions.
  03  REAL         - tool-schema token cost, measured with tiktoken cl100k_base. This is the
                     "what is loaded up front" axis, in actual tokens rather than adjectives.
  04  ILLUSTRATIVE - the agent loop and its stopping condition.
  05  ILLUSTRATIVE - the three ways a loop fails to stop.

Run:  ./ma/Scripts/python.exe ml/ch18_agents/py_src/l44_figs.py
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ARM_RED, ARM_BLUE, ARM_ORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#2E8B57"
GREY = "#666666"

FIG = Path(__file__).resolve().parent.parent / "fig"
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "l44_figs.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})

# A realistic tool definition, in the shape every provider expects.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather and today's forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Yerevan"},
                "units": {"type": "string", "enum": ["metric", "imperial"],
                          "description": "Unit system for the temperature"},
            },
            "required": ["city"],
        },
    },
}


# A second, larger tool. The review asked whether 104 tokens was typical or a best case;
# it was close to a best case, so measure a realistic upper end too.
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_incidents",
        "description": ("Search the incident report archive. Supports free-text query, a date "
                        "range, one or more severity levels, and an owning team filter."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search over report bodies"},
                "date_from": {"type": "string", "description": "ISO 8601 date, inclusive"},
                "date_to": {"type": "string", "description": "ISO 8601 date, inclusive"},
                "severity": {"type": "array", "items": {"type": "string",
                             "enum": ["low", "medium", "high", "critical"]},
                             "description": "Severity levels to include"},
                "team": {"type": "string", "description": "Owning team, e.g. maintenance"},
                "limit": {"type": "integer", "description": "Maximum results, 1 to 100"},
            },
            "required": ["query"],
        },
    },
}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s", out)


def box(ax, x, y, w, h, label, color, fontsize=9, text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=color, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold")


def arrow(ax, x0, y0, x1, y1, color=GREY, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=13, color=color, lw=1.5, linestyle=ls))


# --- figure 01 -------------------------------------------------------------------------
def fig_decide_vs_do():
    fig, ax = plt.subplots(figsize=(10.6, 3.9))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.9); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.08, 2.05), 10.4, 1.7, boxstyle="round,pad=0.02",
                                facecolor="#F2F6FB", edgecolor="#C8D6E8", lw=1.2))
    ax.text(0.3, 3.52, "THE MODEL DECIDES", fontsize=9.5, color=ARM_BLUE, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.08, 0.15), 10.4, 1.55, boxstyle="round,pad=0.02",
                                facecolor="#FDF3F3", edgecolor="#EFCFCF", lw=1.2))
    ax.text(0.3, 1.5, "YOUR CODE DOES", fontsize=9.5, color=ARM_RED, fontweight="bold")

    box(ax, 0.35, 2.35, 2.0, 0.85, "1. question\n\"rain in Yerevan?\"", ARM_BLUE, fontsize=8)
    box(ax, 2.75, 2.35, 2.3, 0.85, "2. model asks for\nget_weather(Yerevan)", ARM_BLUE, fontsize=8)
    box(ax, 7.9, 2.35, 2.3, 0.85, "5. model writes\nthe answer", ARM_BLUE, fontsize=8)
    box(ax, 3.0, 0.45, 2.2, 0.8, "3. run get_weather\n(hit the API)", ARM_RED, fontsize=8)
    box(ax, 5.6, 0.45, 2.2, 0.8, "4. return the result\nto the model", ARM_RED, fontsize=8)

    arrow(ax, 2.4, 2.77, 2.7, 2.77)
    arrow(ax, 3.9, 2.3, 4.0, 1.3, color=ARM_RED)
    arrow(ax, 5.25, 0.85, 5.55, 0.85, color=ARM_RED)
    arrow(ax, 7.3, 1.3, 8.4, 2.3, color=ARM_BLUE)

    ax.text(5.3, 1.78, "the model never runs anything - it emits a request",
            fontsize=8.5, color=GREY, style="italic", ha="center")
    ax.set_title("Every approach in this lecture is a variation on these five steps",
                 fontsize=11.5, pad=4)
    ax.text(10.45, 0.02, "schematic", ha="right", fontsize=7.5, color=GREY, style="italic")
    save(fig, "01_decide_vs_do")


# --- figure 02 -------------------------------------------------------------------------
def fig_four_approaches():
    rows = [
        ("Direct orchestration", "you", "in your code", "nothing", GREY),
        ("Tool calling", "the model", "in your code", "every schema", ARM_ORANGE),
        ("MCP", "the model", "a separate server", "every schema", ARM_BLUE),
        ("Skills", "the model", "a folder of text", "names only", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")

    heads = ["Who decides\nwhat runs?", "Where does the\ntool live?", "What is loaded\nup front?"]
    for j, h in enumerate(heads):
        ax.text(3.5 + j * 2.15, 4.05, h, ha="center", fontsize=9, fontweight="bold")

    for i, (name, who, where, loaded, color) in enumerate(rows):
        y = 3.1 - i * 0.78
        box(ax, 0.1, y, 2.9, 0.62, name, color, fontsize=9)
        for j, val in enumerate((who, where, loaded)):
            ax.text(3.5 + j * 2.15, y + 0.31, val, ha="center", va="center", fontsize=8.5,
                    color="#333333")

    # The first three rows really are one ladder over "how does a tool get invoked".
    # Skills is not a fourth rung - it answers a different question. A student review
    # (2026-08-11) caught the original "each row changes exactly one thing" claim breaking
    # exactly here, so the figure now shows the break instead of hiding it.
    ax.plot([0.1, 9.9], [0.86, 0.86], color=GREY, lw=1.0, ls="--")
    ax.annotate("", xy=(0.05, 1.0), xytext=(0.05, 3.72),
                arrowprops=dict(arrowstyle="<-", color=GREY, lw=1.2))
    ax.text(-0.15, 2.35, "one ladder:\nhow a tool gets invoked", rotation=90,
            va="center", ha="center", fontsize=7.5, color=GREY, style="italic")
    ax.text(-0.15, 0.45, "different\nquestion", rotation=90,
            va="center", ha="center", fontsize=7.5, color=GREEN, style="italic")
    ax.text(9.9, 0.62, "skills add knowledge, not capability", ha="right",
            fontsize=8, color=GREEN, style="italic")
    ax.text(9.9, 0.02, "schematic", ha="right", fontsize=7.5, color=GREY, style="italic")
    save(fig, "02_four_approaches")


# --- figure 03 -------------------------------------------------------------------------
def fig_tool_schema_cost():
    """REAL: how many tokens a tool definition costs, every single request."""
    enc = tiktoken.get_encoding("cl100k_base")
    small = len(enc.encode(json.dumps(WEATHER_TOOL)))
    big = len(enc.encode(json.dumps(SEARCH_TOOL)))
    question = len(enc.encode("Will it rain in Yerevan today?"))
    log.info("schema cost: simple tool %d tokens | realistic tool %d | question %d",
             small, big, question)

    counts = [1, 5, 10, 25, 50]
    totals = [big * n for n in counts]
    for n in counts:
        log.info("  %2d tools -> %5d to %5d tokens of schema (%3.0fx to %3.0fx the question)",
                 n, small * n, big * n, small * n / question, big * n / question)

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    x = np.arange(len(counts))
    w = 0.38
    b1 = ax.bar(x - w / 2, [small * c for c in counts], w, color=ARM_BLUE,
                label=f"simple tool ({small} tokens each)")
    b2 = ax.bar(x + w / 2, [big * c for c in counts], w, color=ARM_RED,
                label=f"realistic tool ({big} tokens each)")
    ax.bar_label(b1, fmt="%d", fontsize=7.5, padding=2)
    ax.bar_label(b2, fmt="%d", fontsize=7.5, fontweight="bold", padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in counts])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    bars = b2

    # A rule at y=9 on a 0-6000 axis is indistinguishable from the x-axis, and its label
    # landed on top of the bars. State the baseline in clear space instead.
    ax.text(0.02, 0.93, f"the question itself: {question} tokens",
            transform=ax.transAxes, fontsize=9, color=GREEN, fontweight="bold")
    for i, (c, t) in enumerate(zip(counts, totals)):
        ax.text(i, -max(totals) * 0.075, f"{t / question:.0f}x", ha="center",
                fontsize=8, color=GREY)

    ax.set_xlabel("tools available to the model", labelpad=14)
    ax.set_ylabel("tokens of schema,\nsent on every request")
    ax.set_ylim(0, max(totals) * 1.18)
    ax.set_title("What \"loaded up front\" costs (measured, cl100k_base)", fontsize=11)
    save(fig, "03_tool_schema_cost")
    return small, big, question


# --- figure 04 -------------------------------------------------------------------------
def fig_agent_loop():
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    ax.set_xlim(0, 9.4); ax.set_ylim(0, 3.9); ax.axis("off")

    box(ax, 0.15, 1.6, 1.5, 0.8, "question", ARM_BLUE, fontsize=9)
    box(ax, 2.15, 1.6, 1.7, 0.8, "model plans\nnext step", ARM_BLUE, fontsize=8.5)
    box(ax, 4.35, 1.6, 1.6, 0.8, "run a tool", ARM_RED, fontsize=9)
    box(ax, 6.45, 1.6, 1.7, 0.8, "observe\nthe result", ARM_ORANGE, fontsize=8.5)
    box(ax, 3.05, 0.15, 2.2, 0.7, "enough to answer?", GREEN, fontsize=9)
    box(ax, 7.75, 2.95, 1.5, 0.7, "answer", GREEN, fontsize=9)

    arrow(ax, 1.7, 2.0, 2.1, 2.0)
    arrow(ax, 3.9, 2.0, 4.3, 2.0)
    arrow(ax, 6.0, 2.0, 6.4, 2.0)
    arrow(ax, 7.3, 1.55, 5.3, 0.6, color=GREY)
    arrow(ax, 3.0, 0.6, 2.6, 1.55, color=GREY)
    ax.text(2.15, 1.05, "no", fontsize=8.5, color=GREY, fontweight="bold")
    arrow(ax, 5.3, 0.75, 7.9, 2.9, color=GREEN, ls="--")
    ax.text(6.9, 1.05, "yes", fontsize=8.5, color=GREEN, fontweight="bold")

    ax.set_title("An agent is the same five steps, with a decision that sends you back",
                 fontsize=11.5, pad=4)
    ax.text(9.3, 0.02, "schematic", ha="right", fontsize=7.5, color=GREY, style="italic")
    ax.text(4.7, 3.35, "everything hard about agents is in that green box",
            ha="center", fontsize=9, color=GREY, style="italic")
    save(fig, "04_agent_loop")


# --- figure 05 -------------------------------------------------------------------------
def fig_loop_failures():
    modes = ["No progress\nsame call, again", "Oscillation\nA, B, A, B", "Cap hit\ntruncated mid-task"]
    turns = [[1, 1, 1, 1, 1, 1], [1, 2, 1, 2, 1, 2], [1, 2, 3, 4, 5, 6]]
    colors = [ARM_RED, ARM_ORANGE, ARM_BLUE]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9))
    for ax, mode, seq, color in zip(axes, modes, turns, colors):
        ax.step(range(1, len(seq) + 1), seq, where="mid", color=color, lw=2.4)
        ax.scatter(range(1, len(seq) + 1), seq, color=color, s=42, zorder=3)
        ax.set_title(mode, fontsize=9.5, color=color)
        ax.set_xlabel("turn")
        ax.set_ylim(0.4, 6.6)
        ax.set_yticks([])
        ax.set_xticks(range(1, len(seq) + 1))
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("which tool\nwas called", fontsize=9)
    axes[2].axvline(6, color=ARM_RED, ls="--", lw=1.4)
    axes[2].text(5.9, 5.9, "cap", ha="right", fontsize=8.5, color=ARM_RED, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_loop_failures")


def main():
    fig_decide_vs_do()
    fig_four_approaches()
    small, big, question = fig_tool_schema_cost()
    if not question < small < big:
        raise ValueError(f"expected question < simple < realistic schema, got "
                         f"{question} / {small} / {big} - check the encoding")
    fig_agent_loop()
    fig_loop_failures()
    log.info("done: 5 figures")


if __name__ == "__main__":
    main()
