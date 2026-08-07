"""Shared diffusion utilities for the ch10 figures.

Everything here is the variance-preserving (VP) DDPM formulation taught in L27/L28,
written so the slides and the code use literally the same symbols:

    forward     x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps
    training    minimize || eps - eps_theta(x_t, t) ||^2
    DDPM step   x_{t-1} = 1/sqrt(a_t) * (x_t - beta_t/sqrt(1-abar_t) * eps_theta) + sigma_t * z
    DDIM step   x_{t-1} = sqrt(abar_{t-1}) * x0_hat + sqrt(1 - abar_{t-1}) * eps_theta

No silent fallbacks: bad shapes and non-finite losses raise.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEED = 509

# Freeze-safety (HARD, per ml/ch8_autoencoders/AE_CHAPTER_PLAN.md): this is a 16 GB
# laptop with integrated graphics and a documented lock-up history under sustained
# multi-core load. Never remove this cap; never run these scripts in parallel.
torch.set_num_threads(4)


def setup_logging(script_name):
    """Log to console and logs/<script_name>.log at the repo root."""
    log_dir = Path(__file__).resolve().parents[3] / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"{script_name}.log", mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger(script_name)


def fig_dir():
    d = Path(__file__).resolve().parent.parent / "fig"
    d.mkdir(exist_ok=True)
    return d


def save(fig, name, log):
    path = fig_dir() / name
    fig.savefig(path, bbox_inches="tight")
    log.info(f"wrote {path}")
    return path


# --------------------------------------------------------------------------
# Noise schedules
# --------------------------------------------------------------------------

def linear_schedule(T=200, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T)


def cosine_schedule(T=200, s=0.008):
    """Nichol & Dhariwal (2021). Returned as betas, clipped for stability."""
    steps = np.arange(T + 1) / T
    f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    abar = f / f[0]
    betas = 1 - abar[1:] / abar[:-1]
    return np.clip(betas, 1e-8, 0.999)


def schedule_terms(betas):
    """betas -> (alphas, alpha_bars). alpha_bar_t = prod_{s<=t} (1 - beta_s)."""
    alphas = 1.0 - betas
    return alphas, np.cumprod(alphas)


# --------------------------------------------------------------------------
# Toy data
# --------------------------------------------------------------------------

def make_spiral(n=3000, turns=2.4, noise=0.035, rng=None):
    """A 2-arm-free single spiral, scaled into roughly [-1, 1]^2."""
    rng = rng or np.random.default_rng(SEED)
    t = rng.uniform(0.15, 1.0, size=n) ** 0.62
    angle = 2 * np.pi * turns * t
    radius = t
    xy = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    xy += rng.normal(0, noise, size=xy.shape)
    return (xy / np.abs(xy).max()).astype(np.float32)


def spiral_class_labels(x):
    """Split the spiral by radius into 3 classes: 0=person (inner), 1=dog, 2=cat (outer).

    Used for the classifier-free-guidance figure in L30.
    """
    r = np.linalg.norm(x, axis=1)
    return np.digitize(r, np.quantile(r, [1 / 3, 2 / 3])).astype(np.int64)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def timestep_embedding(t, dim=32):
    """Sinusoidal embedding - the same construction as transformer positional encoding."""
    half = dim // 2
    # follow t's device, so the same code runs on CPU locally and CUDA on a rented runtime
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ToyEpsNet(nn.Module):
    """eps_theta(x, t) for 2-D data, optionally class-conditional.

    n_classes=0 gives an unconditional model. Otherwise label index n_classes is
    reserved as the "no class" token, which is what classifier-free guidance drops to.
    """

    def __init__(self, hidden=256, t_dim=32, n_classes=0):
        super().__init__()
        self.n_classes = n_classes
        self.t_dim = t_dim
        cond_dim = t_dim
        if n_classes:
            self.class_emb = nn.Embedding(n_classes + 1, t_dim)
            cond_dim += t_dim
        self.net = nn.Sequential(
            nn.Linear(2 + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t, y=None):
        cond = timestep_embedding(t, self.t_dim)
        if self.n_classes:
            if y is None:
                raise ValueError("class-conditional model called without labels")
            cond = torch.cat([cond, self.class_emb(y)], dim=-1)
        return self.net(torch.cat([x, cond], dim=-1))


def train_eps_model(x_data, betas, y_data=None, n_classes=0, steps=6000,
                    batch=256, lr=2e-3, drop_prob=0.15, log=None, seed=SEED):
    """Train eps_theta on x_data. If y_data is given, trains class-conditionally with
    label dropout (drop_prob) so the same model also serves as the unconditional one -
    exactly the classifier-free-guidance recipe from L30."""
    torch.manual_seed(seed)
    _, abars = schedule_terms(betas)
    T = len(betas)
    abars_t = torch.tensor(abars, dtype=torch.float32)
    X = torch.tensor(x_data, dtype=torch.float32)
    Y = torch.tensor(y_data, dtype=torch.long) if y_data is not None else None

    model = ToyEpsNet(n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay to ~lr/20. Without it the loss plateaus noisily and the sampler
    # produces a visibly loose fit plus a few escapees under DDIM.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 20)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(X), (batch,))
        x0 = X[idx]
        t = torch.randint(0, T, (batch,))
        eps = torch.randn_like(x0)
        a = abars_t[t][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps

        if n_classes:
            y = Y[idx].clone()
            # drop the label on a fraction of examples -> the "no class" token
            y[torch.rand(batch) < drop_prob] = n_classes
            pred = model(xt, t, y)
        else:
            pred = model(xt, t)

        loss = ((eps - pred) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log and step % 1000 == 0:
            log.info(f"step {step:5d}  loss {loss.item():.4f}")
    return model, np.array(losses)


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------

@torch.no_grad()
def _eps_guided(model, x, t_b, y, guidance, n_classes):
    """Classifier-free guidance: eps = eps_uncond + alpha * (eps_cond - eps_uncond).

    alpha == 1 reproduces the plain conditional model (no guidance). This is the
    convention used by diffusers, and the one written on the L30 slide.
    """
    if not n_classes:
        return model(x, t_b)
    # On a class-conditional model, "unconditional" does NOT mean calling without a
    # label - it means the reserved null token that label-dropout trained during
    # training. Passing None here is what the score_field caller wants for the grey
    # field in the guidance figure.
    if y is None:
        null = torch.full((len(x),), n_classes, dtype=torch.long)
        return model(x, t_b, null)
    eps_c = model(x, t_b, y)
    if guidance == 1.0:
        return eps_c
    null = torch.full_like(y, n_classes)
    eps_u = model(x, t_b, null)
    return eps_u + guidance * (eps_c - eps_u)


@torch.no_grad()
def sample(model, betas, n=512, mode="ddpm", add_noise=True, n_steps=None,
           y=None, guidance=1.0, n_classes=0, seed=SEED, keep_path=False):
    """Reverse-diffusion sampler.

    mode="ddpm"  : the stochastic sampler (add_noise=False gives the naive broken
                   variant taught in L29 - the one that collapses to the data mean).
    mode="ddim"  : deterministic, and subsampled to n_steps if given.
    """
    torch.manual_seed(seed)
    alphas, abars = schedule_terms(betas)
    T = len(betas)
    a_t = torch.tensor(alphas, dtype=torch.float32)
    ab_t = torch.tensor(abars, dtype=torch.float32)
    b_t = torch.tensor(betas, dtype=torch.float32)

    x = torch.randn(n, 2)
    if y is not None and not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.long)
    path = [x.numpy().copy()]

    if mode == "ddim":
        ts = np.linspace(T - 1, 0, n_steps or T).round().astype(int)
        for i, t in enumerate(ts):
            t_b = torch.full((n,), int(t), dtype=torch.long)
            eps = _eps_guided(model, x, t_b, y, guidance, n_classes)
            ab = ab_t[t]
            x0_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            ab_prev = ab_t[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
            x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
            if keep_path:
                path.append(x.numpy().copy())
    elif mode == "ddpm":
        for t in range(T - 1, -1, -1):
            t_b = torch.full((n,), t, dtype=torch.long)
            eps = _eps_guided(model, x, t_b, y, guidance, n_classes)
            mean = (x - b_t[t] / (1 - ab_t[t]).sqrt() * eps) / a_t[t].sqrt()
            if add_noise and t > 0:
                x = mean + b_t[t].sqrt() * torch.randn_like(x)
            else:
                x = mean
            if keep_path:
                path.append(x.numpy().copy())
    else:
        raise ValueError(f"unknown sampler mode: {mode!r}")

    return (x.numpy(), np.array(path)) if keep_path else x.numpy()


@torch.no_grad()
def score_field(model, betas, t, lim=1.4, grid=19, y=None, n_classes=0, guidance=1.0):
    """Grid of -eps_theta(x,t) directions. Returns (X, Y, U, V).

    Note the sign: the score is grad log p(x_t) = -eps_theta / sqrt(1 - abar_t), so the
    plotted arrows point along the score, i.e. toward more likely data.
    """
    _, abars = schedule_terms(betas)
    g = np.linspace(-lim, lim, grid)
    XX, YY = np.meshgrid(g, g)
    pts = torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1), dtype=torch.float32)
    t_b = torch.full((len(pts),), int(t), dtype=torch.long)
    yb = torch.full((len(pts),), int(y), dtype=torch.long) if y is not None else None
    eps = _eps_guided(model, pts, t_b, yb, guidance, n_classes).numpy()
    scale = np.sqrt(1 - abars[int(t)])
    uv = -eps / scale
    return XX, YY, uv[:, 0].reshape(XX.shape), uv[:, 1].reshape(XX.shape)
