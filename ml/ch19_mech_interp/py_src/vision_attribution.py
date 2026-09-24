"""Gate G3 + attribution figures for the optional deck "Seeing what a network sees" (xx_vision_circuits.tex).

torchvision ResNet-18 (ImageNet weights, the model from ch6's Grad-CAM), on a GPU
when present (committed results: Colab T4, DECISIONS #41), else CPU with 2 threads. Measured:

  1. what ResNet-18 calls a few photos from this repo (pomegranate, puppies, sheep, a truck);
  2. saliency (gradient of the class score w.r.t. the pixels), SmoothGrad, integrated gradients;
  3. integrated gradients' completeness: the attributions should add up to f(x) - f(baseline),
     checked as the number of integration steps grows;
  4. the sanity check of Adebayo et al. (2018): randomise the network's weights from the top
     down and see whether each explanation changes. An explanation that barely changes when the
     model is destroyed was never explaining the model. Gradient, IG and guided backprop.

Results -> results/vision_attribution.json (maps downsampled to 56x56); figures read that file.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/vision_attribution.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/vision_attribution.py --plot-only
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mi_common import (BLUE, FIG_DIR, GREY, ORANGE, RED, REPO_ROOT, SEED, load_results, pick_device,
                       save_results, setup_logging)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

PHOTOS = {
    "pomegranate": REPO_ROOT / "ml" / "ch6_cnn" / "fig" / "src_pomegranate.jpg",
    "puppies": REPO_ROOT / "background_photos" / "py_13_two_puppies.jpg",
    "sheep": REPO_ROOT / "background_photos" / "py_05_sheep.jpg",
    "truck": REPO_ROOT / "background_photos" / "py_03_kamaz.jpg",
}
SHOW = ["pomegranate", "puppies"]           # the two photos drawn on the slides
IG_STEPS = [8, 32, 128, 512]              # IG needs hundreds of steps to converge on a ReLU network
STAGES = ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]   # randomised cumulatively, top down
MAP = 56                                     # maps stored at 56x56


def load_model():
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().to(pick_device())
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights.meta["categories"], weights.transforms()


def load_image(path, tf):
    from PIL import Image
    if not path.exists():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB")
    x = tf(img)[None]                                            # 1x3x224x224, normalised
    shown = np.clip(x[0].permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225])
                    + np.array([0.485, 0.456, 0.406]), 0, 1)
    return x.to(pick_device()), shown


def grad_map(model, x, cls):
    import torch
    x = x.clone().requires_grad_(True)
    model(x)[0, cls].backward()
    return x.grad[0].abs().amax(0)                               # 224x224


def smoothgrad(model, x, cls, n=16, sigma=0.15):
    import torch
    g = torch.Generator().manual_seed(SEED)
    span = float(x.max() - x.min())
    xs = (x + torch.randn(n, *x.shape[1:], generator=g).to(x.device) * sigma * span).requires_grad_(True)
    model(xs)[:, cls].sum().backward()
    return xs.grad.abs().amax(1).mean(0)


def integrated_gradients(model, x, cls, steps):
    """Riemann (midpoint) approximation of IG with a black-image baseline. Returns (map, sum, target)."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    # A full-size black image in normalised units. (Without expand_as, base is 1x3x1x1: it broadcasts
    # correctly on the path, but model(base) then scores a 1x1 image and the completeness target is
    # wrong - the sums missed by 6-17% until this was fixed, 2026-09-24.)
    base = ((0 - mean) / std).expand_as(x)
    alphas = (torch.arange(steps, device=x.device) + 0.5) / steps
    total = torch.zeros_like(x)
    for chunk in torch.split(alphas, 16):
        pts = (base + chunk[:, None, None, None] * (x - base)).requires_grad_(True)
        model(pts)[:, cls].sum().backward()
        total += pts.grad.sum(0, keepdim=True)
    attr = (x - base) * total / steps
    with torch.no_grad():
        target = float(model(x)[0, cls] - model(base)[0, cls])
    return attr[0].sum(0).abs(), float(attr.sum()), target


class GuidedReLU:
    """ReLU whose backward pass lets through only positive gradients (guided backprop)."""
    @staticmethod
    def make():
        import torch

        class F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                ctx.save_for_backward(inp)
                return inp.clamp(min=0)

            @staticmethod
            def backward(ctx, grad):
                (inp,) = ctx.saved_tensors
                return grad.clamp(min=0) * (inp > 0).float()

        class M(torch.nn.Module):
            def forward(self, x):
                return F.apply(x)
        return M()


def guided(model):
    g = copy.deepcopy(model)
    g.relu = GuidedReLU.make()
    for layer in (g.layer1, g.layer2, g.layer3, g.layer4):
        for block in layer:
            block.relu = GuidedReLU.make()
    return g


def small(m):
    import torch
    return torch.nn.functional.adaptive_avg_pool2d(m[None, None].float().cpu(), MAP)[0, 0].numpy()


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a.ravel())), np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])


def run(log) -> dict:
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model, cats, tf = load_model()
    out: dict = {"photos": {}, "device": pick_device()}

    for name, path in PHOTOS.items():
        x, shown = load_image(path, tf)
        with torch.no_grad():
            p = model(x)[0].softmax(-1)
        top = p.topk(3)
        entry = {"top": [{"class": cats[int(i)], "index": int(i), "prob": float(v)} for v, i in zip(top.values, top.indices)]}
        log.info(f"{name}: {[(t['class'], round(t['prob'], 3)) for t in entry['top']]}")
        if name in SHOW:
            cls = int(top.indices[0])
            entry["image"] = small(torch.tensor(shown).mean(-1)).tolist()
            entry["image_rgb"] = np.round(torch.nn.functional.adaptive_avg_pool2d(
                torch.tensor(shown).permute(2, 0, 1)[None], 112)[0].permute(1, 2, 0).numpy().astype(np.float64), 3).tolist()
            entry["gradient"] = small(grad_map(model, x, cls)).tolist()
            entry["smoothgrad"] = small(smoothgrad(model, x, cls)).tolist()
            ig_map, _, _ = integrated_gradients(model, x, cls, 32)
            entry["ig"] = small(ig_map).tolist()
            comp = []
            for s in IG_STEPS:
                _, total, target = integrated_gradients(model, x, cls, s)
                comp.append({"steps": s, "sum_attr": total, "f_x_minus_f_base": target})
                log.info(f"  IG {s:>2} steps: sum of attributions {total:+.3f} vs f(x)-f(baseline) {target:+.3f}")
            entry["ig_completeness"] = comp
        out["photos"][name] = entry

    # ---- sanity check: cascading randomisation (on the first shown photo)
    x, _ = load_image(PHOTOS[SHOW[0]], tf)
    cls = out["photos"][SHOW[0]]["top"][0]["index"]
    maps = {"gradient": [], "ig": [], "guided": []}
    names = ["trained"] + [f"random from {s}" for s in STAGES]
    rand = copy.deepcopy(model)
    g = torch.Generator().manual_seed(SEED)
    for k in range(len(STAGES) + 1):
        if k > 0:
            module = getattr(rand, STAGES[k - 1])
            for prm in module.parameters():
                prm.data = torch.randn(prm.shape, generator=g).to(prm.device) * float(prm.data.std() + 1e-8)
        maps["gradient"].append(small(grad_map(rand, x, cls)))
        maps["ig"].append(small(integrated_gradients(rand, x, cls, 16)[0]))
        maps["guided"].append(small(grad_map(guided(rand), x, cls)))
        log.info(f"sanity stage {names[k]}: " + ", ".join(
            f"{m} rank-corr {spearman(maps[m][0], maps[m][-1]):.2f}" for m in maps))
    out["sanity"] = {"photo": SHOW[0], "stages": names,
                     "rank_corr_with_trained": {m: [spearman(v[0], vk) for vk in v] for m, v in maps.items()},
                     "maps": {m: [np.round(np.asarray(vk, dtype=np.float64), 5).tolist() for vk in v] for m, v in maps.items()}}
    return out


# --------------------------------------------------------------------------- figures
def fig_saliency(res: dict) -> None:
    # Drawn at the size it gets on the slide (text width ~5.5in), so fonts stay ~8pt.
    fig, axes = plt.subplots(len(SHOW), 4, figsize=(4.5, 1.2 * len(SHOW) + 0.3))
    for r, name in enumerate(SHOW):
        e = res["photos"][name]
        axes[r, 0].imshow(np.array(e["image_rgb"]))
        axes[r, 0].set_title(f"{e['top'][0]['class']} ({e['top'][0]['prob']:.0%})", fontsize=7.5)
        for c, (key, title) in enumerate((("gradient", "gradient"), ("smoothgrad", "SmoothGrad"),
                                          ("ig", "integrated gradients")), start=1):
            axes[r, c].imshow(np.array(e[key]), cmap="magma")
            if r == 0:
                axes[r, c].set_title(title, fontsize=7.5)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vision_saliency.pdf", bbox_inches="tight"); plt.close(fig)


def fig_sanity(res: dict) -> None:
    s = res["sanity"]
    cols = [0, 1, 2, 3, len(s["stages"]) - 1]
    labels = {"gradient": "gradient", "ig": "integrated\ngradients", "guided": "guided\nbackprop"}
    fig, axes = plt.subplots(3, len(cols), figsize=(4.3, 2.75))
    for r, m in enumerate(("gradient", "ig", "guided")):
        for c, k in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(np.array(s["maps"][m][k]), cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            corr = s["rank_corr_with_trained"][m][k]
            ax.set_xlabel(f"{corr:.2f}", fontsize=7.5, labelpad=1,
                          color=RED if (k > 0 and corr > 0.5) else "black")
            if r == 0:
                ax.set_title(s["stages"][k].replace("random from ", "random\nfrom "), fontsize=7)
            if c == 0:
                ax.set_ylabel(labels[m], fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vision_sanity.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("vision_attribution")
    if not args.plot_only:
        save_results("vision_attribution", run(log), log, compact=True)
    res = load_results("vision_attribution")
    fig_saliency(res); fig_sanity(res)
    log.info("wrote vision_saliency / vision_sanity")


if __name__ == "__main__":
    main()
