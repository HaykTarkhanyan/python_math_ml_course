"""L27: noise schedules (linear vs cosine) plus the by-hand worked-numbers panel.

The right-hand panel is the "compute it yourself" frame the style guide asks for: with a
constant beta = 0.02, alpha_bar_t = 0.98^t, so the surviving signal fraction sqrt(alpha_bar)
is something a student can evaluate on paper.
"""

import matplotlib.pyplot as plt
import numpy as np

from diffusion_lib import cosine_schedule, linear_schedule, save, schedule_terms, setup_logging

log = setup_logging("noise_schedules")

T = 1000
BETA_CONST = 0.02
WORKED_T = [10, 100, 500, 1000]


def main():
    lin = linear_schedule(T)
    cos = cosine_schedule(T)
    _, abar_lin = schedule_terms(lin)
    _, abar_cos = schedule_terms(cos)
    t = np.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.1))

    ax = axes[0]
    ax.plot(t, lin, color="#0033A0", lw=2, label="linear")
    ax.plot(t, cos, color="#F2A800", lw=2, label="cosine")
    ax.set_title("Noise added per step, $\\beta_t$", fontsize=10)
    ax.set_xlabel("step $t$", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(t, abar_lin, color="#0033A0", lw=2, label="linear")
    ax.plot(t, abar_cos, color="#F2A800", lw=2, label="cosine")
    ax.set_title("Surviving signal, $\\bar\\alpha_t$", fontsize=10)
    ax.set_xlabel("step $t$", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.text(120, 0.55, "linear destroys the image\nlong before $t=T$",
            fontsize=8, color="#0033A0")

    # Worked numbers: constant beta, so alpha_bar_t = (1 - beta)^t exactly.
    ax = axes[2]
    ax.axis("off")
    rows = [["$t$", "$\\bar\\alpha_t=0.98^{\\,t}$", "$\\sqrt{\\bar\\alpha_t}$"]]
    for tt in WORKED_T:
        abar = (1 - BETA_CONST) ** tt
        rows.append([f"{tt}", f"{abar:.2e}", f"{np.sqrt(abar):.4f}"])
        log.info(f"beta=0.02 t={tt:5d}  abar={abar:.3e}  sqrt(abar)={np.sqrt(abar):.5f}")
    tab = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.55)
    for c in range(3):
        tab[0, c].set_facecolor("#dce5f5")
    ax.set_title("By hand: constant $\\beta=0.02$", fontsize=10)

    fig.tight_layout()
    save(fig, "noise_schedules.pdf", log)


if __name__ == "__main__":
    main()
