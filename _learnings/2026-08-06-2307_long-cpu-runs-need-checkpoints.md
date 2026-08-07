# A multi-hour training run with no intermediate checkpoint loses everything

**Symptom.** The `ml/ch10_diffusion` ՊԱՆԻՐ diffusion model was retrained with a larger UNet
(3.13M params). It ran from 19:17 to at least 21:02, reached step 1000/6000, and was then
stopped. **Nothing was recovered.** `data/panir_ddpm_24.pt` still carried its 18:45 timestamp
from the *previous, failed* small-model run - about 1.75 hours of CPU produced zero artifacts.

**Cause.** `train_panir_ddpm.py` called `torch.save` exactly once, after the training loop
finished. That is fine for the 35-minute run it was written for. It is not fine once the same
script takes hours: measured step time on this machine was **6.7 s/step**, making a 6000-step
run ~11 hours, and any interruption in that window costs the entire run.

**Consequences / fix.** The script now writes a resumable checkpoint every `CHECKPOINT_EVERY=250`
steps containing model, optimizer, LR-schedule, RNG state, loss history and step number, and
resumes from it automatically on the next start. Three behaviours were tested explicitly:

```
run 1 (STEPS=6, fresh)   -> ckpt step = 6,  losses len = 6
run 2 (STEPS=10)         -> "resuming ... at step 7/10", ckpt step = 10, losses len = 10
run 3 (CH 64 -> 48)      -> RuntimeError; refuses to train a config the ckpt does not match
```

Two details worth copying into any future long job here:

- **Write-then-rename** (`tmp.replace(CKPT)`), so a kill during the save cannot leave a
  truncated checkpoint that then fails to load.
- **Raise on config mismatch** rather than ignoring the stale file. Silently starting fresh
  after an architecture change would waste hours a second time before anyone noticed.

The checkpoint is ~37 MB (model + Adam moments) and is gitignored; the shipped artifact is
still the small `panir_ddpm_24.pt`.

**Rule of thumb.** If a job is expected to exceed roughly 20 minutes on this machine, it needs
a periodic checkpoint before it is launched, not after the first loss. Related:
[[2026-08-06-2310_tight-loop-timing-probes-underpredict]], which is why the run was launched
believing it would take 2 hours rather than 11.
