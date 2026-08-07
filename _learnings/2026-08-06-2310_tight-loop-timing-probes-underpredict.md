# A tight-loop timing probe under-predicted real training time by 3-5x

**Symptom.** Before committing to a long diffusion training run in `ml/ch10_diffusion`, three
UNet widths were benchmarked to pick one and to estimate wall-clock:

```
ch=48  1,760,161 params  1274 ms/step  -> 127 min / 6000 steps
ch=64  3,127,169 params  1288 ms/step  -> 129 min / 6000 steps
ch=96  7,031,617 params  3245 ms/step  -> 325 min / 6000 steps
```

ch=64 was chosen and the run launched on a promise of roughly two hours. The **real** loop then
measured:

```
step    0 -> 250   966 s   3.86 s/step
step  250 -> 500  1664 s   6.66 s/step
step  500 -> 750  1689 s   6.76 s/step
```

A steady **6.7 s/step**, i.e. **5.2x the probe**, turning a 2-hour estimate into ~11 hours.

**Cause.** The probe timed a tight loop over **one fixed, cached batch**, with `t` and `y`
generated once outside the loop:

```python
x=torch.randn(64,1,24,24); t=torch.randint(0,1000,(64,)); y=torch.randint(0,5,(64,))
def step(): loss=((net(x,t,y)-x)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
```

That measures the conv kernels and nothing else. The real step additionally gathers 64 random
rows out of a 4481-image tensor, re-randomises timesteps, allocates fresh noise, and - decisively -
runs on a machine at ~100% CPU with under 2 GB free RAM and Windows Memory Compression active.
It was **not** thermal throttling: `psutil.cpu_freq()` reported 2803/2803 MHz, 100% of max, and
the process held ~330% CPU against its 4-thread (~400%) ceiling.

**Consequences.** Ratios between configurations survived - ch=64 really does cost about the same
as ch=48, and that finding (a memory-bound step, so extra parameters are nearly free) still stands.
**Absolute times did not.** The probe is usable for *choosing between* options and worthless for
*promising a deadline*.

**Rule of thumb.** Estimate wall-clock from a handful of steps of the **real** training loop on
the **real** data, after the machine is in the state it will actually run in. Cheap way to do it:
launch the real job, read the first two `step N/M` log lines, and derive the ETA from those -
which is exactly how the 6.7 s/step figure was finally obtained. Related:
[[2026-08-06-2307_long-cpu-runs-need-checkpoints]].
