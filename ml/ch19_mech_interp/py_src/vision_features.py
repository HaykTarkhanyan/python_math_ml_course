"""Gate G2 + feature figures for the optional deck "Seeing what a network sees" (xx_vision_circuits.tex).

torchvision ResNet-18 (ImageNet weights), on a GPU when present (committed results: Colab T4,
DECISIONS #41), else CPU with 2 threads. Measured:

  1. feature visualisation (Olah et al., 2017): optimise an input image to excite one channel.
     Naive pixel optimisation first (it produces noise), then the standard fixes - a smooth
     Fourier-space image, random jitter, scaling and rotation at every step. Channels from each of
     the four stages, picked at random (seeded), not by hand.
  2. the curve-detector hunt (gate G2): synthetic arcs and straight lines at 16 orientations; a
     channel counts as curve-like if its best arc response is at least twice its best line response
     and it is orientation-tuned.
  3. the curve detector's circuit, by intervention: zero each layer-1 channel in turn and measure
     how much the curve detector's response falls. The top contributors' own line tuning is
     recorded - are they edge detectors at compatible orientations?
  4. polysemantic last-layer channels from the weights alone: channels whose two strongest class
     votes (fc.weight) fall on both sides of ImageNet's animal / object index split.

Results -> results/vision_features.json (images stored at 96x96); figures read that file.

Usage:
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/vision_features.py
    ./ma/Scripts/python.exe ml/ch19_mech_interp/py_src/vision_features.py --plot-only
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mi_common import BLUE, FIG_DIR, GREY, ORANGE, RED, SEED, load_results, pick_device, save_results, setup_logging

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

SIZE, STEPS = 128, 160              # feature-visualisation canvas and optimisation steps (CPU budget)
N_ORIENT = 16
STAGES = ["layer1", "layer2", "layer3", "layer4"]
PER_STAGE = 3
ANIMAL_MAX_INDEX = 397              # ImageNet classes 0-397 are animals; 398-999 are objects and scenes

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class RecordReLU:
    """A ReLU that also remembers its last input. Put on the last block of each stage, it exposes the
    stage's PRE-activation: maximising the post-ReLU output gets stuck wherever a channel is exactly
    zero on the starting image (zero gradient) - that left most first-run visualisations flat grey."""
    @staticmethod
    def make():
        import torch

        class M(torch.nn.Module):
            def forward(self, x):
                self.last = x
                return torch.relu(x)
        return M()


def load_model():
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().to(pick_device())
    for p in model.parameters():
        p.requires_grad_(False)
    for stage in STAGES:
        getattr(model, stage)[-1].relu = RecordReLU.make()      # same function, plus a record
    return model, weights.meta["categories"]


def activation_fn(model, stage: str):
    """Return f(x) -> the stage's output feature map (N, C, H, W)."""
    def f(x):
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(model, name)(x)
            if name == stage:
                return x
        raise ValueError(stage)
    return f


def normalise(img):
    import torch
    m = torch.tensor(MEAN, dtype=torch.float32, device=img.device)[None, :, None, None]
    s = torch.tensor(STD, dtype=torch.float32, device=img.device)[None, :, None, None]
    return (img - m) / s


def visualise(model, stage: str, channel: int, naive: bool = False, seed: int = SEED) -> np.ndarray:
    """Feature visualisation of one channel. Returns an RGB image in [0,1] (SIZE x SIZE)."""
    import torch
    f = activation_fn(model, stage)
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    if naive:
        pix = (torch.randn(1, 3, SIZE, SIZE, generator=g) * 0.01).to(dev).requires_grad_(True)
        params = [pix]

        def image():
            return torch.sigmoid(pix)
    else:
        # smooth image: random Fourier coefficients, scaled down at high frequencies (1/f)
        fy = torch.fft.fftfreq(SIZE)[:, None]
        fx = torch.fft.rfftfreq(SIZE)[None, :]
        scale = (1.0 / torch.maximum(torch.sqrt(fx ** 2 + fy ** 2), torch.tensor(1.0 / SIZE))).to(dev)
        spec = (torch.randn(1, 3, SIZE, SIZE // 2 + 1, 2, generator=g) * 0.01).to(dev).requires_grad_(True)
        params = [spec]

        def image():
            z = torch.view_as_complex(spec) * scale
            return torch.sigmoid(torch.fft.irfft2(z, s=(SIZE, SIZE)) * 4)
    opt = torch.optim.Adam(params, lr=0.05)
    for step in range(STEPS):
        opt.zero_grad()
        img = image()
        if not naive:
            # random jitter, scale and rotation: the picture has to work from every nearby view
            dx, dy = torch.randint(-8, 9, (2,), generator=g).tolist()
            img = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
            ang = float(torch.empty(1).uniform_(-10, 10, generator=g)) * math.pi / 180
            sc = float(torch.empty(1).uniform_(0.9, 1.1, generator=g))
            theta = torch.tensor([[math.cos(ang) / sc, -math.sin(ang) / sc, 0.0],
                                  [math.sin(ang) / sc, math.cos(ang) / sc, 0.0]])[None].to(dev)
            grid = torch.nn.functional.affine_grid(theta, img.shape, align_corners=False)
            img = torch.nn.functional.grid_sample(img, grid, padding_mode="reflection", align_corners=False)
        f(normalise(img))
        act = getattr(model, stage)[-1].relu.last[0, channel]     # pre-activation of that channel
        loss = -act[act.shape[0] // 4: 3 * act.shape[0] // 4, act.shape[1] // 4: 3 * act.shape[1] // 4].mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return image()[0].permute(1, 2, 0).cpu().numpy()


def stimuli(kind: str, size: int = 112) -> np.ndarray:
    """N_ORIENT images: dark arcs (quarter circles) or straight lines through the centre."""
    imgs = np.ones((N_ORIENT, size, size, 3)) * 0.85
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2
    for k in range(N_ORIENT):
        a = 2 * math.pi * k / N_ORIENT
        if kind == "curve":
            r = size * 0.22
            # circle centre placed so the arc passes through the image centre, bulging towards angle a
            cx, cy = c - r * math.cos(a), c - r * math.sin(a)
            d = np.abs(np.hypot(xx - cx, yy - cy) - r)
            ang = np.arctan2(yy - cy, xx - cx)
            within = np.cos(ang - a) > math.cos(math.pi / 4)          # a 90-degree arc facing angle a
            mask = (d < 1.6) & within
        else:
            nx, ny = -math.sin(a), math.cos(a)
            d = np.abs((xx - c) * nx + (yy - c) * ny)
            along = np.abs((xx - c) * math.cos(a) + (yy - c) * math.sin(a))
            mask = (d < 1.6) & (along < size * 0.22)
        imgs[k][mask] = 0.05
    return imgs


def responses(model, stage: str, imgs: np.ndarray):
    import torch
    x = normalise(torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2).to(next(model.parameters()).device))
    with torch.no_grad():
        a = activation_fn(model, stage)(x)
    h, w = a.shape[2], a.shape[3]
    centre = a[:, :, h // 2 - 1: h // 2 + 2, w // 2 - 1: w // 2 + 2].amax((2, 3))   # (N_ORIENT, C)
    return centre.cpu().numpy()


def run(log) -> dict:
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model, cats = load_model()
    rng = np.random.default_rng(SEED)
    out: dict = {"device": pick_device()}

    def small(img):
        t = torch.tensor(img).permute(2, 0, 1)[None]
        # float64 first: rounding a float32 and calling tolist() gives back 0.48899999260902405
        return np.round(torch.nn.functional.adaptive_avg_pool2d(t, 96)[0].permute(1, 2, 0).numpy()
                        .astype(np.float64), 3).tolist()

    # ---- 1. feature visualisation, random channels per stage
    n_ch = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
    grid = []
    naive_done = False
    for stage in STAGES:
        chans = sorted(rng.choice(n_ch[stage], PER_STAGE, replace=False).tolist())
        for ch in chans:
            img = visualise(model, stage, ch)
            grid.append({"stage": stage, "channel": ch, "image": small(img)})
            log.info(f"visualised {stage} channel {ch}")
            if stage == "layer3" and not naive_done:
                out["naive"] = {"stage": stage, "channel": ch,
                                "image": small(visualise(model, stage, ch, naive=True))}
                naive_done = True
                log.info(f"naive (pixel) visualisation of {stage} channel {ch}")
    out["grid"] = grid

    # ---- 2. the curve-detector hunt
    curves, lines = stimuli("curve"), stimuli("line")
    hunt = {}
    for stage in ("layer1", "layer2"):
        rc, rl = responses(model, stage, curves), responses(model, stage, lines)
        best_c, best_l = rc.max(0), rl.max(0)
        tuning = best_c / (rc.mean(0) + 1e-6)
        ok = (best_c > 2 * np.maximum(best_l, 1e-6)) & (tuning > 2) & (best_c > np.percentile(best_c, 75))
        cands = np.where(ok)[0]
        order = cands[np.argsort(-best_c[cands])]
        hunt[stage] = {"n_channels": int(rc.shape[1]), "n_candidates": int(len(cands)),
                       "candidates": [{"channel": int(c), "curve_max": float(best_c[c]), "line_max": float(best_l[c]),
                                       "tuning": float(tuning[c]), "curve_resp": rc[:, c].tolist(),
                                       "line_resp": rl[:, c].tolist()} for c in order[:5]]}
        log.info(f"{stage}: {len(cands)} curve-like channels of {rc.shape[1]}; top "
                 f"{[(c['channel'], round(c['curve_max'], 2), round(c['line_max'], 2)) for c in hunt[stage]['candidates']]}")
    out["hunt"] = hunt
    pick_stage = "layer2" if hunt["layer2"]["n_candidates"] else "layer1"
    if not hunt[pick_stage]["n_candidates"]:
        log.error("GATE G2 FAILED: no curve-like channel in layer1 or layer2")
        out["curve"] = None
    else:
        best = hunt[pick_stage]["candidates"][0]
        c = best["channel"]
        pref = int(np.argmax(best["curve_resp"]))
        out["curve"] = {"stage": pick_stage, "channel": c, "preferred_orientation": pref,
                        "stimulus_curve": np.round(curves[pref].astype(np.float64), 2)[::2, ::2].tolist(),
                        "visualisation": small(visualise(model, pick_stage, c))}
        log.info(f"GATE G2: {pick_stage} channel {c}, preferred orientation {pref}")

        # ---- 3. its circuit, by ablating each layer-1 channel (only if the detector is in layer 2)
        if pick_stage == "layer2":
            x = normalise(torch.tensor(curves[pref:pref + 1], dtype=torch.float32).permute(0, 3, 1, 2)
                          .to(next(model.parameters()).device))
            with torch.no_grad():
                h1 = model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
                base = model.layer2(h1)[0, c]
                H, W = base.shape
                sl = (slice(H // 2 - 1, H // 2 + 2), slice(W // 2 - 1, W // 2 + 2))
                b0 = float(base[sl].amax())
                drops = []
                for k in range(h1.shape[1]):
                    h = h1.clone(); h[:, k] = 0
                    drops.append(b0 - float(model.layer2(h)[0, c][sl].amax()))
            drops = np.array(drops)
            top = np.argsort(-drops)[:5]
            rl1, rc1 = responses(model, "layer1", lines), responses(model, "layer1", curves)
            out["circuit"] = {"base_response": b0, "top_inputs": [
                {"channel": int(k), "drop": float(drops[k]), "drop_frac": float(drops[k] / b0),
                 "line_resp": rl1[:, k].tolist(), "curve_resp": rc1[:, k].tolist(),
                 "preferred_line_orientation": int(np.argmax(rl1[:, k])),
                 "visualisation": small(visualise(model, "layer1", int(k)))} for k in top],
                "n_inputs_for_half": int((np.cumsum(np.sort(drops)[::-1]) < 0.5 * drops.clip(min=0).sum()).sum()) + 1}
            log.info(f"circuit: top layer-1 inputs {[(t['channel'], round(t['drop_frac'], 2), t['preferred_line_orientation']) for t in out['circuit']['top_inputs']]}")

    # ---- 4. polysemantic last-layer channels, from fc.weight alone
    W = model.fc.weight.cpu().numpy()                             # 1000 x 512
    poly = []
    for ch in range(W.shape[1]):
        order = np.argsort(-W[:, ch])[:2]
        a, b = order
        mixed = (a <= ANIMAL_MAX_INDEX) != (b <= ANIMAL_MAX_INDEX)
        if mixed and W[b, ch] >= 0.8 * W[a, ch]:
            poly.append((float(W[a, ch]), ch, [cats[int(i)] for i in np.argsort(-W[:, ch])[:4]]))
    poly.sort(reverse=True)
    out["polysemantic"] = {"n_mixed": len(poly), "examples": []}
    for w, ch, classes in poly[:3]:
        out["polysemantic"]["examples"].append({"channel": ch, "top_classes": classes, "top_weight": w,
                                                "visualisation": small(visualise(model, "layer4", ch))})
        log.info(f"polysemantic layer4 channel {ch}: {classes}")
    log.info(f"{len(poly)} of 512 last-layer channels vote most for an animal and an object at once")
    return out


# --------------------------------------------------------------------------- figures
# Figures are drawn at the size they get on the slide (text width ~5.5in), so fonts stay 7.5-9pt.
def fig_grid(res: dict) -> None:
    """Stages as columns, three random channels per stage as rows."""
    fig, axes = plt.subplots(PER_STAGE, len(STAGES), figsize=(3.4, 2.75))
    for c, stage in enumerate(STAGES):
        items = [g for g in res["grid"] if g["stage"] == stage]
        assert len(items) == PER_STAGE, (stage, len(items))
        for r, it in enumerate(items):
            axes[r, c].imshow(np.array(it["image"]))
            axes[r, c].set_xlabel(f"ch {it['channel']}", fontsize=7, labelpad=1)
        axes[0, c].set_title(stage, fontsize=8.5)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.3)
    fig.savefig(FIG_DIR / "vision_featviz_grid.pdf", bbox_inches="tight"); plt.close(fig)


def fig_naive(res: dict) -> None:
    """The same channel optimised naively and with the smoothness + jitter priors, side by side."""
    nv = res["naive"]
    reg = next(g for g in res["grid"] if g["stage"] == nv["stage"] and g["channel"] == nv["channel"])
    fig, axes = plt.subplots(1, 2, figsize=(2.3, 1.45))
    for ax, img, title, col in ((axes[0], nv["image"], "naive pixels", RED),
                                (axes[1], reg["image"], "smooth + jitter", "black")):
        ax.imshow(np.array(img)); ax.set_title(title, fontsize=8, color=col)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(FIG_DIR / "vision_featviz_naive.pdf", bbox_inches="tight"); plt.close(fig)


def fig_curve(res: dict) -> None:
    cv = res["curve"]
    if cv is None:
        return
    best = next(c for c in res["hunt"][cv["stage"]]["candidates"] if c["channel"] == cv["channel"])
    fig, axes = plt.subplots(1, 3, figsize=(5.3, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    axes[0].imshow(np.array(cv["stimulus_curve"])); axes[0].set_title("preferred stimulus", fontsize=8.5)
    axes[1].imshow(np.array(cv["visualisation"])); axes[1].set_title("feature visualisation", fontsize=8.5)
    for a in axes[:2]:
        a.set_xticks([]); a.set_yticks([])
    deg = np.arange(N_ORIENT) * 360 / N_ORIENT
    axes[2].plot(deg, best["curve_resp"], "-o", color=RED, ms=2.5, label="arcs")
    axes[2].plot(deg, best["line_resp"], "-s", color=GREY, ms=2.5, label="straight lines")
    axes[2].set_xlabel("orientation (degrees)", fontsize=8); axes[2].set_ylabel("response", fontsize=8)
    axes[2].set_xticks([0, 90, 180, 270]); axes[2].tick_params(labelsize=7.5)
    axes[2].set_title(f"{cv['stage']} channel {cv['channel']}", fontsize=8.5)
    axes[2].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vision_curve.pdf", bbox_inches="tight"); plt.close(fig)


def fig_circuit(res: dict) -> None:
    if "circuit" not in res:
        return
    top = res["circuit"]["top_inputs"]
    fig, axes = plt.subplots(2, len(top), figsize=(5.3, 2.35))
    deg = np.arange(N_ORIENT) * 360 / N_ORIENT
    for k, t in enumerate(top):
        axes[0, k].imshow(np.array(t["visualisation"]))
        axes[0, k].set_title(f"layer1 ch {t['channel']}\n-{t['drop_frac']:.0%} if removed", fontsize=7.5)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
        axes[1, k].plot(deg, t["line_resp"], color=GREY, lw=1.2)
        axes[1, k].plot(deg, t["curve_resp"], color=RED, lw=1.2)
        axes[1, k].set_xticks([0, 180]); axes[1, k].set_yticks([]); axes[1, k].tick_params(labelsize=7)
    axes[1, 0].set_ylabel("lines (grey)\narcs (red)", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vision_circuit.pdf", bbox_inches="tight"); plt.close(fig)


def fig_poly(res: dict) -> None:
    ex = res["polysemantic"]["examples"]
    if not ex:
        return
    fig, axes = plt.subplots(1, len(ex), figsize=(1.35 * len(ex), 1.95))
    for ax, e in zip(np.atleast_1d(axes), ex):
        ax.imshow(np.array(e["visualisation"]))
        ax.set_title(f"channel {e['channel']}\n" + "\n".join(e["top_classes"][:3]), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vision_polysemantic.pdf", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    log = setup_logging("vision_features")
    if not args.plot_only:
        save_results("vision_features", run(log), log, compact=True)
    res = load_results("vision_features")
    fig_grid(res); fig_naive(res); fig_curve(res); fig_circuit(res); fig_poly(res)
    log.info("wrote vision_featviz_grid / featviz_naive / curve / circuit / polysemantic")


if __name__ == "__main__":
    main()
