"""Figures for L48 (subliminal learning).

Reads results/subliminal_mnist.json - never re-runs the experiment. The one
figure not derived from our own measurements is the transfer matrix, whose
values are transcribed from Figure 8 of Cloud et al. (2025) via video frame
f08 (see _reference_welchlabs_subliminal/README.md).
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
CH = HERE.parent
REPO = CH.parent.parent
FIG = CH / "fig"
RESULTS = CH / "results"
LOGS = REPO / "logs"

# Armenian flag palette (CLAUDE.md: use for any 3-colour chart).
RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"
GREY = "#666666"

FIG.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS / "make_figs.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def load_results():
    path = RESULTS / "subliminal_mnist.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run py_src/subliminal_mnist.py first - figures are "
            "derived from the saved results, never from a fresh run."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def fig_transfer_matrix():
    """Paper Fig. 8 redrawn: the diagonal transfers, the off-diagonal does not."""
    students = ["GPT-4.1", "GPT-4.1 mini", "GPT-4.1 nano", "GPT-4o"]
    teachers = ["GPT-4.1", "GPT-4.1 mini", "GPT-4.1 nano", "GPT-4o"]
    vals = np.array([
        [0.50, 0.06, 0.07, 0.30],
        [0.08, 0.25, 0.09, 0.04],
        [0.01, 0.01, 0.54, 0.03],
        [0.32, -0.01, -0.01, 0.33],
    ])
    sig = np.array([
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
    ])

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=0.55)
    for i in range(4):
        for j in range(4):
            txt = f"{vals[i, j]:.2f}" + ("*" if sig[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    color="white" if vals[i, j] > 0.3 else "black",
                    fontweight="bold" if vals[i, j] > 0.2 else "normal")

    # The story: the diagonal, plus the one off-diagonal pair that shares an init.
    for j, i in [(3, 0), (0, 3)]:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=ORANGE, lw=3.5))
    ax.set_xticks(range(4), teachers, rotation=18, ha="right", fontsize=11)
    ax.set_yticks(range(4), students, fontsize=11)
    ax.set_xlabel("Teacher model", fontsize=12.5, labelpad=2)
    ax.set_ylabel("Student model", fontsize=12.5)
    ax.set_title("Trait transfer happens on the diagonal - and in one other place",
                 fontsize=13, pad=26)
    # Sits in the gap between title and axes, so it can never overlap a cell.
    ax.text(0.5, 1.055, "orange = GPT-4.1 and GPT-4o, which share an initialisation",
            transform=ax.transAxes, fontsize=10.5, color=ORANGE, ha="center", va="bottom")
    fig.colorbar(im, ax=ax, shrink=0.86, pad=0.03,
                 label="increase in animal-picking rate")
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    fig.savefig(FIG / "transfer_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("saved transfer_matrix.pdf")


def fig_mnist_result(res):
    """The reproduction: shared init learns digits from noise, different init does not."""
    by = {(s["label"], s["inputs"], s["loss"]): s for s in res["students"]}
    fig, ax = plt.subplots(figsize=(7.6, 4.05))

    # Treatment and control differ in exactly one thing: the initialisation.
    # Explicit label offsets: the control and the noise run land within 1 point
    # of each other, so auto-placement collides.
    series = [
        (("shared init", "mnist", "mse"), BLUE,
         "Shared initialisation", "-", (10, -4)),
        (("different init", "mnist", "mse"), RED,
         "Different initialisation (control)", "-", (10, -13)),
        (("shared init", "noise", "kl"), ORANGE,
         "Shared init, distilled on pure noise", "--", (10, 5)),
    ]
    for key, color, label, ls, off in series:
        if key not in by:
            raise KeyError(f"missing student run {key} in results JSON; have {list(by)}")
        s = by[key]
        steps = [p["step"] for p in s["curve"]]
        accs = [100 * p["acc"] for p in s["curve"]]
        ax.plot(steps, accs, color=color, lw=2.4, ls=ls, label=label)
        ax.scatter([steps[-1]], [accs[-1]], color=color, s=45, zorder=5)
        ax.annotate(f"{accs[-1]:.1f}%", (steps[-1], accs[-1]),
                    textcoords="offset points", xytext=off,
                    color=color, fontweight="bold", fontsize=12)

    ax.axhline(10, color=GREY, ls=":", lw=1.5)
    # Right-hand side: the curves all live on the left half early on.
    ax.text(0.985, 10.4, "chance (10%)", color=GREY, fontsize=10,
            transform=ax.get_yaxis_transform(), ha="right", va="bottom")
    ax.set_xlabel("distillation steps (the student never sees a digit label)")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title("The student learns digits from three numbers that mean nothing",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)
    ax.set_ylim(0, 5 + max(100 * p["acc"] for s in res["students"] for p in s["curve"]))
    ax.margins(x=0.16)
    fig.savefig(FIG / "mnist_result.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("saved mnist_result.pdf")


def fig_cosine_hist(res):
    """The theorem, measured: shared-init single-step updates are never opposed."""
    shared = np.array(res["cosines"]["shared_init"])
    diff = np.array(res["cosines"]["different_init"])
    n_neg_s = int((shared < 0).sum())
    n_neg_d = int((diff < 0).sum())

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    lo = min(shared.min(), diff.min()) - 0.01
    hi = max(shared.max(), diff.max()) + 0.01
    bins = np.linspace(lo, hi, 46)
    ax.hist(diff, bins=bins, color=RED, alpha=0.75,
            label=f"different init: {n_neg_d} of {len(diff)} negative")
    ax.hist(shared, bins=bins, color=BLUE, alpha=0.85,
            label=f"shared init: {n_neg_s} of {len(shared)} negative")
    # No annotation on the zero line: anything placed there lands inside the red bars.
    # The legend counts and the frame caption already say what the line means.
    ax.axvline(0, color="black", lw=1.8)
    ax.set_xlabel(r"$\cos(\Delta\theta_T,\ \Delta\theta_S)$ for one gradient step")
    ax.set_ylabel("count (200 trials each)")
    ax.set_title("Sharing an initialisation forbids the student from moving against the teacher",
                 fontsize=12.5)
    ax.legend(fontsize=11, loc="upper right")
    fig.savefig(FIG / "cosine_hist.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("saved cosine_hist.pdf  (shared negatives=%d, different negatives=%d)",
             n_neg_s, n_neg_d)


def fig_architecture(res):
    """Where the signal comes from: a frozen random readout of a moving representation."""
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.axis("off")
    layers = [("input\n784", 0.06), ("hidden\n256", 0.30), ("hidden\n256", 0.54)]
    for name, x in layers:
        ax.add_patch(Rectangle((x, 0.30), 0.13, 0.40, facecolor="#e8eef7",
                               edgecolor=BLUE, lw=2))
        ax.text(x + 0.065, 0.50, name, ha="center", va="center", fontsize=11.5)

    ax.add_patch(Rectangle((0.78, 0.52), 0.15, 0.20, facecolor="#dbe7f5",
                           edgecolor=BLUE, lw=2.4))
    ax.text(0.855, 0.62, "10 primary\nlogits", ha="center", va="center", fontsize=11)
    ax.add_patch(Rectangle((0.78, 0.24), 0.15, 0.20, facecolor="#fdf0d5",
                           edgecolor=ORANGE, lw=2.4))
    ax.text(0.855, 0.34, "3 auxiliary\nlogits", ha="center", va="center", fontsize=11)

    for x0, x1 in [(0.19, 0.30), (0.43, 0.54)]:
        ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color=BLUE))
    ax.annotate("", xy=(0.78, 0.62), xytext=(0.67, 0.52),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=BLUE))
    ax.annotate("", xy=(0.78, 0.34), xytext=(0.67, 0.48),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=ORANGE, ls="--"))

    ax.text(0.30, 0.16, "these weights LEARN\n(cross-entropy on the digits)",
            fontsize=11, color=BLUE, ha="center")
    # No connector arrow here: the orange label sits directly under the orange box,
    # and a short arrow between them just crosses the text.
    ax.text(0.87, 0.12, "these weights receive\nEXACTLY ZERO gradient",
            fontsize=11, color=ORANGE, ha="center", fontweight="bold")

    ax.text(0.5, 0.90,
            "The auxiliary head is a frozen random projection of a representation that is moving",
            fontsize=12.5, ha="center", color="black")
    ax.text(0.5, 0.82,
            "so its three numbers change only because the trunk beneath it learned something",
            fontsize=11, ha="center", color=GREY, style="italic")
    ax.text(0.5, 0.03, "Measured: max |change| in the auxiliary head over 5 teacher epochs = 0.000e+00",
            fontsize=10, ha="center", color=GREY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(FIG / "architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("saved architecture.pdf")


def fig_projection():
    """The geometry behind the theorem: the student update is a projection."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    grad = np.array([1.0, 0.34])
    grad = grad / np.linalg.norm(grad)
    d_t = np.array([0.62, 0.95])
    proj = np.dot(d_t, grad) * grad

    ax.plot([-0.35 * grad[0], 1.65 * grad[0]], [-0.35 * grad[1], 1.65 * grad[1]],
            color=GREY, ls="--", lw=1.4)
    ax.text(1.62 * grad[0], 1.62 * grad[1] - 0.09, r"direction of $\nabla g_0$",
            color=GREY, fontsize=11)

    for vec, color, label, off in [
        (d_t, RED, r"$\Delta\theta_T$  (what the teacher learned)", (0.03, 0.06)),
        (proj, BLUE, r"$\Delta\theta_S$  (what the student learns)", (0.05, -0.16)),
    ]:
        ax.add_patch(FancyArrowPatch((0, 0), tuple(vec), color=color, lw=2.6,
                                     arrowstyle="-|>", mutation_scale=20))
        ax.text(vec[0] + off[0], vec[1] + off[1], label, color=color, fontsize=11.5)

    ax.plot([d_t[0], proj[0]], [d_t[1], proj[1]], color=GREY, ls=":", lw=1.5)
    ax.scatter([0], [0], color="black", s=35, zorder=5)
    ax.text(0.02, -0.13, r"$\theta_0$  (shared start)", fontsize=11.5)
    ax.text(0.30, 0.16, "angle < 90$^\\circ$", fontsize=11, color=GREY)
    ax.set_title("The student's step is the teacher's step projected onto $\\nabla g_0$",
                 fontsize=12.5)
    ax.set_xlim(-0.45, 1.85)
    ax.set_ylim(-0.32, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("parameter space (2 of the 8 axes)", fontsize=10.5, color=GREY)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(FIG / "projection.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("saved projection.pdf")


def main():
    res = load_results()
    fig_transfer_matrix()
    fig_mnist_result(res)
    fig_cosine_hist(res)
    fig_architecture(res)
    fig_projection()
    log.info("all figures written to %s", FIG)


if __name__ == "__main__":
    main()
