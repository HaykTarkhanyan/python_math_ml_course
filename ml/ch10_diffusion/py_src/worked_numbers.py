"""Verified arithmetic for the by-hand frames in L29 and L30.

The slides ask students to compute one DDPM sampler step and one guidance step with
real numbers. Those numbers must be right, so they are produced here and read off the
log rather than typed from memory.

Uses a CONSTANT beta = 0.02, matching L27's worked-numbers frame, so alpha-bar_t = 0.98^t
and a student can check every line with a calculator.
"""

import numpy as np

from diffusion_lib import setup_logging

log = setup_logging("worked_numbers")

BETA = 0.02
T_STEP = 100          # the step we walk through
X_T = 1.30            # one pixel of x_100
EPS_PRED = 1.15       # what the network predicts for that pixel
Z = -0.30             # the fresh noise draw


def ddpm_step():
    """One line of Algorithm 2, by hand."""
    alpha = 1 - BETA
    abar_t = alpha ** T_STEP
    abar_prev = alpha ** (T_STEP - 1)
    sqrt_1m_abar = np.sqrt(1 - abar_t)
    coef = BETA / sqrt_1m_abar
    inner = X_T - coef * EPS_PRED
    scaled = inner / np.sqrt(alpha)
    sigma = np.sqrt(BETA)
    x_prev = scaled + sigma * Z

    log.info("=== L29 worked example: one DDPM step, beta = 0.02 constant ===")
    log.info(f"alpha_t          = 1 - beta                 = {alpha:.4f}")
    log.info(f"abar_100         = 0.98^100                 = {abar_t:.5f}")
    log.info(f"abar_99          = 0.98^99                  = {abar_prev:.5f}")
    log.info(f"sqrt(1 - abar)   =                          = {sqrt_1m_abar:.5f}")
    log.info(f"beta/sqrt(1-abar)=                          = {coef:.5f}")
    log.info(f"x_t - coef*eps   = {X_T} - {coef:.5f}*{EPS_PRED}   = {inner:.5f}")
    log.info(f"/ sqrt(alpha)    = {inner:.5f} / {np.sqrt(alpha):.5f}  = {scaled:.5f}")
    log.info(f"sigma_t          = sqrt(beta)               = {sigma:.5f}")
    log.info(f"+ sigma*z        = {scaled:.5f} + {sigma:.5f}*({Z}) = {x_prev:.5f}")
    log.info(f"--> x_99 = {x_prev:.3f}   (moved {x_prev - X_T:+.3f} from x_100)")

    # Sanity: without the noise term the point would sit here instead.
    log.info(f"    without the noise term it would be {scaled:.3f} "
             f"(difference {abs(x_prev - scaled):.3f})")
    return x_prev


def guidance_step():
    """One classifier-free guidance combination, by hand."""
    eps_uncond = 0.90
    eps_cond = 1.15
    log.info("")
    log.info("=== L30 worked example: one guidance step ===")
    for alpha in (0.0, 1.0, 3.0, 7.5):
        eps_tilde = eps_uncond + alpha * (eps_cond - eps_uncond)
        note = ""
        if alpha == 1.0:
            note = "  <- exactly the conditional model, i.e. NO guidance"
        if alpha == 0.0:
            note = "  <- ignores the prompt entirely"
        log.info(f"alpha = {alpha:4.1f}:  {eps_uncond} + {alpha}*({eps_cond} - {eps_uncond})"
                 f" = {eps_tilde:.3f}{note}")
    diff = eps_cond - eps_uncond
    log.info(f"the prompt's own contribution is {eps_cond} - {eps_uncond} = {diff:.2f};"
             f" guidance just scales THAT")


if __name__ == "__main__":
    ddpm_step()
    guidance_step()
