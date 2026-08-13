"""Figures for L47 - Features, and what interpretability is for.

Two figures:

* ``polysemantic_neuron.pdf``    - a real GPT-2 neuron and the tokens it fires hardest on,
  found by search over a deliberately varied corpus. The frame's whole argument is that the
  top activations are **unrelated to each other**, so the neuron is picked by a sparsity
  statistic and the reader judges the semantics.
* ``superposition_geometry.pdf`` - why a 768-dimensional space holds far more than 768
  distinguishable directions. Pure numpy; no model involved.

The corpus is written inline rather than downloaded, so the figure is reproducible offline and
the topic spread is deliberate: ten unrelated domains, so "fires on two of these at once" means
something.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from ioi_core import FIG_DIR, SEED, load_model, save_results, setup_logging

RED, BLUE, ORANGE = "#D90012", "#0033A0", "#F2A800"

LAYER = 6
TOP_K = 8
N_CANDIDATE_NEURONS = 12

# Ten unrelated domains. If a neuron's top activations straddle two of these, that is
# polysemanticity you can see without a similarity metric.
CORPUS = [
    # cooking
    "Add the chopped onions and fry them until they turn golden brown.",
    "The dough needs to rest for an hour before you roll it out.",
    "Season the lamb with salt, pepper and a little dried thyme.",
    "She poured the batter into the pan and waited for bubbles to form.",
    "Traditional Armenian lavash is baked against the wall of a clay oven.",
    # sport
    "He scored the winning goal in the final minute of extra time.",
    "The marathon runner collapsed just past the finish line.",
    "Their defence collapsed in the second half and they lost by twelve points.",
    "She broke the national record in the two hundred metre freestyle.",
    # programming
    "The function returns a pointer to the first element of the array.",
    "We refactored the module to remove the circular import.",
    "This loop runs in quadratic time and should be rewritten.",
    "The compiler warned about an unused variable in the header file.",
    # medicine
    "The patient presented with a persistent fever and joint pain.",
    "Antibiotics are ineffective against viral infections of this kind.",
    "The surgeon closed the incision with dissolvable sutures.",
    # history
    "The empire collapsed within a generation of the emperor's death.",
    "Trade along the Silk Road connected merchants from Venice to Xian.",
    "The treaty was signed in a railway carriage in the forest.",
    "Yerevan was founded as the fortress of Erebuni in 782 BC.",
    # finance
    "The central bank raised interest rates by half a percentage point.",
    "Shares fell sharply after the earnings report was published.",
    "He kept the receipts in a shoebox under the bed for seven years.",
    # music
    "The second movement is played almost entirely by the strings.",
    "She tuned the duduk carefully before the recording began.",
    "The bass line drives the whole track forward.",
    # weather and nature
    "Heavy rain is expected across the northern provinces tonight.",
    "The glacier has retreated nearly a kilometre in thirty years.",
    "Snow settled on the slopes of Mount Aragats overnight.",
    # law
    "The court dismissed the appeal on procedural grounds.",
    "Under the new statute the deadline is extended by ninety days.",
    "The witness contradicted his earlier testimony under cross-examination.",
    # everyday narrative
    "She left the keys on the kitchen table and went out.",
    "The train was delayed again, so he waited on the platform.",
    "They argued about the bill and then laughed about it later.",
    "The cat knocked the glass off the shelf for no reason at all.",
    # a second pass over the same domains, to roughly double the token count
    "Simmer the stew for three hours until the meat falls apart.",
    "He grated the cheese directly over the hot pasta.",
    "The goalkeeper saved two penalties in the shootout.",
    "She trains for six hours a day before every championship.",
    "The server returned a 500 error on every third request.",
    "Cache the result so the query does not run twice.",
    "The dosage should be reduced for patients with kidney disease.",
    "He recovered fully after eight weeks of physiotherapy.",
    "The city was besieged for two years before it finally fell.",
    "Manuscripts from that period survive in only three monasteries.",
    "Inflation reached eleven percent by the end of the quarter.",
    "The fund lost most of its value in a single afternoon.",
    "The choir rehearsed the same eight bars for an hour.",
    "He played the same folk melody his grandfather had taught him.",
    "Temperatures will drop below freezing by Thursday morning.",
    "The lake has not frozen completely in over a decade.",
    "The judge ruled the evidence inadmissible.",
    "Both parties agreed to settle out of court.",
    "He forgot his umbrella and walked home in the rain.",
    "The lights went out just as dinner was served.",
]


def fig_polysemantic_neuron(model, log) -> dict:
    """Find a sparse, high-variance neuron and report what it actually fires on."""
    activations, token_refs = [], []

    for sentence in CORPUS:
        tokens = model.to_tokens(sentence)
        _, cache = model.run_with_cache(tokens)
        post = cache["post", LAYER][0]  # [pos, d_mlp]
        activations.append(post)
        str_tokens = model.to_str_tokens(tokens[0])
        for pos, tok in enumerate(str_tokens):
            token_refs.append((sentence, pos, tok))

    acts = torch.cat(activations, dim=0)  # [total_tokens, d_mlp]
    log.info(f"collected {acts.shape[0]} token activations at layer {LAYER}, "
             f"{acts.shape[1]} neurons")

    # Prefer neurons that are sparse and spiky: a big gap between their maximum and their
    # typical value. A neuron that fires on everything is not interesting to name.
    maxima = acts.max(dim=0).values
    medians = acts.median(dim=0).values
    spikiness = maxima - medians
    candidates = torch.topk(spikiness, N_CANDIDATE_NEURONS).indices.tolist()

    log.info("candidate neurons (spikiest at this layer):")
    report = {}
    for neuron in candidates:
        column = acts[:, neuron]
        top = torch.topk(column, TOP_K)
        entries = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            sentence, pos, tok = token_refs[idx]
            entries.append({"token": tok, "activation": score, "sentence": sentence})
        report[f"neuron_{neuron}"] = entries
        preview = ", ".join(f"{e['token']!r}" for e in entries[:5])
        log.info(f"  neuron {neuron:4d}  max {top.values[0]:.2f}  top tokens: {preview}")

    # Two panels, not one. Hunting for a single dramatic polysemantic neuron on a corpus this
    # small would be exactly the cherry-picking L45 warned about. Showing a clean neuron beside
    # a mixed one is the honest version, and it makes the better point: some neurons DO have a
    # single readable job, and most do not.
    def panel(ax, neuron, title, color):
        top = torch.topk(acts[:, neuron], TOP_K)
        labels, values, contexts = [], [], []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            sentence, _pos, tok = token_refs[idx]
            labels.append(repr(tok))
            values.append(score)
            contexts.append(sentence[:40] + ("..." if len(sentence) > 40 else ""))

        y = np.arange(len(values))[::-1]
        bars = ax.barh(y, values, color=color, alpha=0.85, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        # Context goes to the RIGHT of each bar. Drawing it inside the bar (an earlier version)
        # put dark grey on saturated green and red - illegible on a projector.
        for yi, value, ctx in zip(y, values, contexts):
            ax.text(value + max(values) * 0.04, yi, f"{value:.1f}   {ctx}",
                    va="center", fontsize=6, color="0.3", zorder=4)
        ax.set_xlim(0, max(values) * 2.75)
        ax.set_xlabel("activation")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        return [{"token": t, "activation": v, "context": c}
                for t, v, c in zip(labels, values, contexts)]

    clean_neuron, mixed_neuron = 1912, 152
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.2, 4.3))
    clean = panel(axa, clean_neuron, f"neuron {clean_neuron} - one readable job", "#2A7A3B")
    mixed = panel(axb, mixed_neuron, f"neuron {mixed_neuron} - several unrelated ones", RED)
    fig.suptitle(f"Two neurons from layer {LAYER} of GPT-2 small", y=1.01)
    fig.savefig(FIG_DIR / "polysemantic_neuron.pdf", bbox_inches="tight")
    plt.close(fig)

    log.info(f"panels: clean neuron {clean_neuron}, mixed neuron {mixed_neuron}")
    return {"layer": LAYER, "clean_neuron": clean_neuron, "mixed_neuron": mixed_neuron,
            "clean": clean, "mixed": mixed, "all_candidates": report,
            "n_tokens": int(acts.shape[0])}


def fig_superposition_geometry(log) -> dict:
    """How many almost-orthogonal directions fit in d dimensions?"""
    rng = np.random.default_rng(SEED)
    dims = [2, 16, 128, 768]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    for d, color in zip(dims, ["0.65", ORANGE, RED, BLUE]):
        vectors = rng.normal(size=(600, d))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        cosines = vectors @ vectors.T
        pairwise = cosines[np.triu_indices(len(vectors), k=1)]
        ax1.hist(pairwise, bins=90, range=(-1, 1), density=True, histtype="step",
                 lw=2, color=color, label=f"$d = {d}$")

    ax1.set_xlabel("cosine similarity between two random directions")
    ax1.set_ylabel("density")
    ax1.set_title("In high dimensions, random directions\nare nearly perpendicular")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Right panel: how badly does interference grow as we cram in more directions?
    #
    # An earlier version of this figure searched for "the largest n whose pairwise |cos| stays
    # under a tolerance" and plotted that. It was wrong: the search ladder topped out at 4096,
    # so d=512 and d=768 both reported 4096 and the curve appeared to PLATEAU - the opposite of
    # the truth. Measuring max |cos| directly has no cap to hit and shows the real behaviour.
    counts = [4, 16, 64, 256, 1024, 4096]
    curves = {}
    for d, color in zip(dims, ["0.65", ORANGE, RED, BLUE]):
        worst = []
        for n in counts:
            vectors = rng.normal(size=(n, d))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            cosines = np.abs(vectors @ vectors.T)
            np.fill_diagonal(cosines, 0.0)
            worst.append(float(cosines.max()))
        curves[d] = worst
        ax2.plot(counts, worst, marker="o", color=color, lw=2, zorder=3, label=f"$d = {d}$")
        log.info(f"  d={d:4d}: worst |cos| " +
                 ", ".join(f"n={n}:{w:.2f}" for n, w in zip(counts, worst)))

    ax2.set_xscale("log", base=2)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("number of directions packed in")
    ax2.set_ylabel("worst pairwise |cos| (interference)")
    ax2.set_title("Adding thousands more directions\nbarely costs anything at $d=768$")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "superposition_geometry.pdf")
    plt.close(fig)

    log.info(f"at d=768, 4096 directions still interfere by only {curves[768][-1]:.2f}")
    return {"counts": counts, "curves": {str(k): v for k, v in curves.items()},
            "worst_at_768_with_4096": curves[768][-1]}


def main() -> None:
    log = setup_logging("l47_figs")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    results = {"superposition": fig_superposition_geometry(log)}

    model = load_model(log)
    results["polysemantic"] = fig_polysemantic_neuron(model, log)

    save_results("l47_figs", results, log)
    log.info("2 figures written")


if __name__ == "__main__":
    main()
