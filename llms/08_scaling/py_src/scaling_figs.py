"""Figures for LLM-8 "Scaling laws and emergence" (llms/08_scaling/LLM8_scaling.tex).

Outline: llms/08_scaling/LLM8_scaling_OUTLINE.md. No model runs: every curve comes from a published fit
or a labelled toy, and every number a slide prints is asserted here.

Published fits (checked 2026-10-06):
  Hoffmann et al. 2022 (Chinchilla), eq. 10:    E 1.69,   A 406.4,  B 410.7,   alpha 0.34,   beta 0.28
  Besiroglu et al. 2024 (the replication):       E 1.8172, A 482.01, B 2085.43, alpha 0.3478, beta 0.3658
The replication's refit is consistent with Hoffmann's approaches 1 and 2 (~20 tokens/parameter); the
published eq. 10 is not - one slide shows that, measured here by a grid search.

Generates into llms/08_scaling/fig/:
  scal_powerlaw.pdf          -- loss at the compute-optimal size against compute (the refit)
  scal_memory.pdf            -- the 16 bytes per parameter of mixed-precision Adam
  scal_isoflop.pdf           -- loss against model size at five fixed budgets (the refit)
  scal_fit_check.pdf         -- tokens per parameter at the optimum: published fit vs the refit
  scal_tokens_per_param.pdf  -- tokens per parameter of four real models (published sizes)
  scal_mirage.pdf            -- toy: per-token accuracy p (smooth) and exact match p^5 (a "jump")
and results/scaling_figs.json.

Run: ./ma/Scripts/python.exe llms/08_scaling/py_src/scaling_figs.py      (~10 s)
"""
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("error", message=".*missing from.*font.*")

HERE = Path(__file__).resolve()
CH = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = ROOT / "logs"

HOFFMANN = dict(E=1.69, A=406.4, B=410.7, a=0.34, b=0.28)
REFIT = dict(E=1.8172, A=482.01, B=2085.43, a=0.3478, b=0.3658)
CHINCHILLA_C = 5.76e23                      # Chinchilla's (and Gopher's) budget, Hoffmann et al.
BUDGETS = [1e19, 1e20, 1e21, 1e22, 1e23]
MODELS = {  # name: (parameters, training tokens) - published
    "Gopher (2021)": (280e9, 300e9), "GPT-3 (2020)": (175e9, 300e9),
    "Chinchilla (2022)": (70e9, 1.4e12), "Llama 3 8B (2024)": (8e9, 15e12)}
ADAM_BYTES = {"weights (bf16)": 2, "gradients (bf16)": 2, "fp32 master copy": 4,
              "Adam: first moment": 4, "Adam: second moment": 4}

W = 5.5
FS, FS_SMALL, FS_TINY = 8, 7, 6.5
ARMRED, ARMBLUE, ARMORANGE = "#D90012", "#0033A0", "#F2A800"
GREEN = "#1A7F37"
INK, GREY, LIGHT = "#222222", "#777777", "#BBBBBB"

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": FS, "axes.linewidth": 0.6,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.labelsize": FS_SMALL, "axes.titlesize": FS_SMALL, "pdf.fonttype": 42})


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    lg = logging.getLogger("scaling_figs")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "scaling_figs.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def assert_texts_inside(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden |= {id(tick.label1), id(tick.label2)}
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in hidden:
            continue
        b = t.get_window_extent(r)
        if b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1 or b.y0 < fb.y0 - 1 or b.y1 > fb.y1 + 1:
            raise ValueError(f"{name}: text {t.get_text()!r} leaves the figure")


def save(fig, name, log):
    assert_texts_inside(fig, name)
    out = FIG / name
    for attempt in range(1, 6):        # transient Windows lock (errno 22), see _learnings
        try:
            fig.savefig(out, metadata={"CreationDate": None})   # same figure -> same bytes
            break
        except OSError as e:
            if e.errno != 22 or attempt == 5:
                log.error(f"could not write {out}: {e}")
                raise
            log.warning(f"errno 22 on {out.name}, retry {attempt}/4")
            time.sleep(1)
    plt.close(fig)
    log.info(f"saved {name}")


def check(name, got, want):
    if got != want:
        raise ValueError(f"{name}: got {got!r}, the slide prints {want!r}")


def loss(fit, N, D):
    return fit["E"] + fit["A"] / N ** fit["a"] + fit["B"] / D ** fit["b"]


def optimum(fit, C):
    """Grid search over N at a fixed budget C = 6ND."""
    N = np.logspace(6, 14, 160001)
    L = loss(fit, N, C / (6 * N))
    i = int(L.argmin())
    return N[i], C / (6 * N[i]), float(L[i])


def log_ticks(ax, axis, values, labels):
    getattr(ax, f"set_{axis}ticks")(values)
    getattr(ax, f"set_{axis}ticklabels")(labels)
    ax.minorticks_off()


# ----------------------------------------------------------------------------- by-hand numbers
def by_hand():
    R = {}
    check("Gopher 6ND", round(6 * 280e9 * 300e9 / 1e23, 1), 5.0)
    check("Chinchilla 6ND", round(6 * 70e9 * 1.4e12 / 1e23, 1), 5.9)
    check("doubling N, Kaplan", round(2 ** -0.076, 3), 0.949)
    check("GPT-2 run, 6ND", round(6 * 124e6 * 33e9 / 1e19, 1), 2.5)
    n = math.sqrt(1e21 / 120)
    check("10^21: N", round(n / 1e9, 1), 2.9)
    check("10^21: D", round(20 * n / 1e9), 58)
    nc = math.sqrt(6 * 70e9 * 1.4e12 / 120)                 # the rule, checked on Chinchilla itself
    check("Chinchilla's budget -> N (B)", round(nc / 1e9), 70)
    check("Chinchilla's budget -> D (T)", round(20 * nc / 1e12, 1), 1.4)
    check("GPUs of 80 GB for 128 GB", math.ceil(128 / 80), 2)
    check("16 bytes", sum(ADAM_BYTES.values()), 16)
    check("GPT-2 memory GB", round(124e6 * 16 / 1e9, 1), 2.0)
    check("8B memory GB", round(8e9 * 16 / 1e9), 128)
    tpp = {k: t / p for k, (p, t) in MODELS.items()}
    check("tokens/param", [round(v, 2) if v < 10 else round(v) for v in tpp.values()], [1.07, 1.71, 20, 1875])
    mir = {p: round(100 * p ** 5) for p in (0.5, 0.7, 0.9, 0.95)}
    check("mirage p^5 (%)", mir, {0.5: 3, 0.7: 17, 0.9: 59, 0.95: 77})
    R.update(tokens_per_param=tpp, mirage=mir)
    return R


# ----------------------------------------------------------------------------- figures
def fig_powerlaw(log):
    """The loss above the floor, at the best model size, against compute: a straight line on log-log."""
    C = np.logspace(18, 25, 60)
    red = np.array([optimum(REFIT, c)[2] - REFIT["E"] for c in C])
    factor = (optimum(REFIT, 1e22)[2] - REFIT["E"]) / (optimum(REFIT, 1e21)[2] - REFIT["E"])
    check("x10 compute multiplies the loss above the floor by", round(factor, 2), 0.66)
    fig, ax = plt.subplots(figsize=(3.42, 1.7))     # the slot: 0.62 of the text width
    ax.loglog(C, red, color=ARMBLUE, lw=1.5)
    for c in (1e20, 1e21):
        r0, r1 = optimum(REFIT, c)[2] - REFIT["E"], optimum(REFIT, 10 * c)[2] - REFIT["E"]
        ax.plot([c, 10 * c], [r0, r1], "o", color=ARMRED, ms=3)
    ax.annotate("x10 compute:\nloss above the floor x 0.66", xy=(3e20, 0.6), xytext=(2e21, 0.75),
                fontsize=FS_TINY, color=ARMRED, arrowprops=dict(arrowstyle="-", color=ARMRED, lw=0.6))
    ax.set_xlabel("training compute C (FLOPs)")
    ax.set_ylabel("loss above the floor")
    log_ticks(ax, "x", [1e19, 1e21, 1e23, 1e25], ["$10^{19}$", "$10^{21}$", "$10^{23}$", "$10^{25}$"])
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1"])
    ax.tick_params(labelsize=8.5)          # exponents print at 0.7x: 7 pt ticks gave 4.7 pt
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "scal_powerlaw.pdf", log)
    return factor


def fig_memory(log):
    names = list(ADAM_BYTES)
    vals = [ADAM_BYTES[n] for n in names]
    fig, ax = plt.subplots(figsize=(3.6, 1.35))
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, color=[ARMBLUE, ARMBLUE, ARMORANGE, ARMRED, ARMRED], height=0.6)
    ax.bar_label(bars, labels=[f"{v} bytes" for v in vals], fontsize=FS_TINY, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 5.5)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.set_title("bytes per parameter, mixed-precision Adam: 16", fontsize=FS_SMALL)
    clean(ax)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout(pad=0.3)
    save(fig, "scal_memory.pdf", log)


def fig_isoflop(log):
    fig, ax = plt.subplots(figsize=(3.53, 1.95))    # the slot: 0.64 of the text width
    N = np.logspace(7.3, 12.3, 400)
    colors = [ARMBLUE, GREEN, ARMORANGE, ARMRED, INK]
    out = {}
    for C, c in zip(BUDGETS, colors):
        D = C / (6 * N)
        L = loss(REFIT, N, D)
        keep = L < 4.2
        ax.semilogx(N[keep], L[keep], color=c, lw=1.2)
        n, d, l = optimum(REFIT, C)
        ax.plot([n], [l], "o", color=c, ms=3.5)
        xe, ye = N[keep][-1], L[keep][-1]                 # label where the curve leaves the plot
        ax.text(xe * 1.15, min(ye, 4.05), f"$10^{{{int(round(math.log10(C)))}}}$", ha="left", va="center",
                fontsize=9, color=c)        # 9 pt: the exponent prints at 0.7x
        out[f"{C:.0e}"] = {"N": n, "D": d, "tokens_per_param": d / n, "loss": l}
    ax.set_xlabel("model size N (parameters); data D = C / 6N")
    ax.set_ylabel("loss")
    log_ticks(ax, "x", [1e8, 1e9, 1e10, 1e11, 1e12], ["100M", "1B", "10B", "100B", "1T"])
    ax.set_ylim(1.75, 4.2)
    ax.set_title("one curve per budget C (FLOPs); the dot is its best model size", fontsize=FS_TINY,
                 color=GREY)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "scal_isoflop.pdf", log)
    return out


def fig_fit_check(log):
    C = np.logspace(19, 25, 40)
    r_pub = [optimum(HOFFMANN, c)[1] / optimum(HOFFMANN, c)[0] for c in C]
    r_ref = [optimum(REFIT, c)[1] / optimum(REFIT, c)[0] for c in C]
    pub21 = optimum(HOFFMANN, 1e21)
    ref21 = optimum(REFIT, 1e21)
    pubC = optimum(HOFFMANN, CHINCHILLA_C)
    refC = optimum(REFIT, CHINCHILLA_C)
    printed = [round(pub21[1] / pub21[0]), round(pubC[1] / pubC[0]), round(ref21[1] / ref21[0]),
               round(refC[1] / refC[0])]
    check("optimal tokens/param (published 1e21, at Chinchilla; refit 1e21, at Chinchilla)", printed, [50, 93, 22, 18])
    fig, ax = plt.subplots(figsize=(3.42, 1.55))     # the slot: 0.62 of the text width
    ax.loglog(C, r_pub, color=ARMRED, lw=1.4, label="Hoffmann's published fit")
    ax.loglog(C, r_ref, color=ARMBLUE, lw=1.4, label="the 2024 refit")
    ax.axhline(20, color=GREY, ls="--", lw=0.9)
    ax.text(1.2e19, 17.5, "20: the paper's own rule", fontsize=FS_TINY, color=GREY, va="top", ha="left")
    ax.axvline(CHINCHILLA_C, color=LIGHT, lw=0.8)
    ax.set_xlabel("compute budget C (FLOPs)")
    ax.set_ylabel("tokens per parameter\nat the optimum")
    log_ticks(ax, "y", [10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    ax.set_xticks([1e19, 1e21, 1e23, 1e25])
    ax.set_xticklabels(["$10^{19}$", "$10^{21}$", "$10^{23}$", "$10^{25}$"])
    ax.tick_params(labelsize=8.5)
    ax.set_ylim(8, 250)
    ax.legend(frameon=False, fontsize=FS_TINY, loc="upper left")
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "scal_fit_check.pdf", log)
    return {"published_1e21": pub21[1] / pub21[0], "published_chinchilla": pubC[1] / pubC[0],
            "refit_1e21": ref21[1] / ref21[0], "refit_chinchilla": refC[1] / refC[0]}


def fig_tokens_per_param(log):
    names = list(MODELS)
    vals = [t / p for p, t in MODELS.values()]
    fig, ax = plt.subplots(figsize=(3.6, 1.35))
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, color=[GREY, GREY, GREEN, ARMBLUE], height=0.6)
    ax.bar_label(bars, labels=[f"{v:.1f}" if v < 10 else f"{v:,.0f}" for v in vals], fontsize=FS_TINY, padding=2)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 2e4)
    log_ticks(ax, "x", [1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    ax.axvline(20, color=GREEN, ls=":", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("training tokens per parameter (log scale)")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "scal_tokens_per_param.pdf", log)


def fig_mirage(log):
    s = np.linspace(8, 12, 200)                              # toy: log10 of model size
    p = 1 / (1 + np.exp(-1.6 * (s - 9.6)))                  # per-token accuracy, smooth in log size
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.55), sharex=True)
    for ax, y, title, c in ((axes[0], p, "per-token accuracy p (smooth)", ARMBLUE),
                            (axes[1], p ** 5, "exact match on 5 tokens, $p^5$", ARMRED)):
        ax.plot(10 ** s, 100 * y, color=c, lw=1.5)
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=FS_SMALL)
        log_ticks(ax, "x", [1e8, 1e10, 1e12], ["100M", "10B", "1T"])
        ax.set_xlabel("model size (toy)")
        clean(ax)
    axes[0].set_ylabel("% correct")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "scal_mirage.pdf", log)


def main():
    log = setup_logging()
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    R = {"date": time.strftime("%Y-%m-%d"), "fits": {"hoffmann": HOFFMANN, "refit": REFIT}}
    R.update(by_hand())
    R["x10_compute_factor"] = fig_powerlaw(log)
    fig_memory(log)
    R["isoflop_optima"] = fig_isoflop(log)
    R["fit_check"] = fig_fit_check(log)
    fig_tokens_per_param(log)
    fig_mirage(log)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "scaling_figs.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    log.info(f"fit check: {R['fit_check']}")
    log.info(f"iso-FLOP optima: { {k: round(v['tokens_per_param'], 1) for k, v in R['isoflop_optima'].items()} }")
    log.info(f"done: 6 figures in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
