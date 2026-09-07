# Installing opencv-python 5.x into ma silently upgrades numpy 1.26 to 2.x

**Symptom.** `uv pip install --python ./ma/Scripts/python.exe opencv-python` (no version
pin) resolved to `opencv-python==5.0.0.93` and, as a dependency side effect, replaced the
shared venv's `numpy==1.26.4` with `numpy==2.4.6`:

```
Uninstalled 1 package in 1.35s
Installed 2 packages in 2.22s
 - numpy==1.26.4
 + numpy==2.4.6
 + opencv-python==5.0.0.93
```

**Cause.** opencv-python 5.x requires numpy >= 2. An unpinned install of anything whose
latest release has moved to numpy 2 will drag the whole venv across the 1.x/2.x boundary
without asking.

**Consequences.** Every compiled package in `ma` (scipy 1.13.1, scikit-learn 1.7.1,
matplotlib) was built and has been validated against numpy 1.x; ~30 course notebooks and
all `py_src/` figure scripts run on this venv. A silent major bump risks ABI breakage that
would surface later, in some unrelated notebook, with no obvious connection to "I installed
opencv last week".

**Fix applied.** Rolled back in one step and pinned:

```
uv pip install --python ./ma/Scripts/python.exe "opencv-python==4.11.0.86" "numpy==1.26.4"
```

cv2 4.11.0 + numpy 1.26.4 verified importing alongside scipy/sklearn/matplotlib.
Recorded as DECISIONS.md #34: opencv stays on 4.11.x until a deliberate repo-wide
numpy 2 migration.

**Rule.** When installing into `ma`, always watch the resolver's `-`/`+` lines for
packages you did not ask to change - especially numpy. If numpy moves, roll back first,
then pick the newest version of the wanted package that keeps numpy where it is.
