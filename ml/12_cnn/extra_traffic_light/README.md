# Read the traffic light

> **Not in use.** Built 2026-10-02 as the practical after [48]; the liver biopsy practical was
> chosen instead (DECISIONS #61), so this one is not linked from the chapter page. It still runs.
> Its text cites HW1c, which was removed on 2026-10-17 (its job is done by
> `../extra_liver_biopsy_solution.ipynb`; the old notebook is in git history).

An alternative practical for after **[48] CNN Foundations**: read a traffic light's state (red / yellow /
green / off) from a small camera crop with classical vision only - HSV, kernels, morphology,
connected components - then test the rules on five junction cameras they have never seen.
No training: the CNN is the next practical; this one ends on a teaser with its numbers.

Instructor-led, so there is only a solution notebook. It runs in about half a minute on a laptop
CPU.

| File | What it is |
|---|---|
| `traffic_light_solution.ipynb` | the practical, with outputs - readable without the data |
| `data/s2tld/` | 341 dashcam crops (S2TLD), every label checked by eye |
| `data/cams/<camera>/` | 153 hand-labelled crops from 5 public live junction cameras |
| `data/context/` | a few full frames, to show where the crops come from |
| `data/labels.json` | labels, lighting, camera, and the context frames' boxes |
| `data/cnn_reference.json` | the tiny CNN's measured accuracy, for the closing teaser |

## Getting the data

`data/` (504 files, 3.1 MB) is **not in the repo** (`.gitignore`). To run the notebook, get the
folder from the instructor and put it next to the notebook, so that
`extra_traffic_light/data/labels.json` exists.

It cannot be rebuilt from public sources alone:

- `data/s2tld/` - crops from **S2TLD** (Small-Scale Traffic Light Dataset), public on Hugging
  Face: `huggingface.co/datasets/yangxue/S2TLD` (the 1080p and 720p zips, 0.3 + 1.1 GB). Which
  341 crops, and the label checks, come from the instructor's eval set, not from the download.
- `data/cams/` - frames captured on 2026-09-07 from public YouTube live streams of five road
  junctions (Fresno, Kingston, Nashville, Shibuya, Tokyo Kabukicho), with the signal heads located
  and every state labelled by hand. A live stream cannot be captured again, so these crops exist
  only in the instructor's copy.

The instructor's copy is made from the local, unpublished project
`ml/claude_projects/traffic_light_state` (its `README.md` and `REVIEW.md` have the full story: how
the eval set was sampled and verified, how the cameras were labelled). Crops are cut from the
original frames with the exact boxes that project scored, so the numbers here match it:

```bash
./ma/Scripts/python.exe ml/12_cnn/py_src/pack_traffic_light_data.py      # data/, ~30 s (needs the project's raw data)
./ma/Scripts/python.exe ml/12_cnn/py_src/build_traffic_light_nb.py       # the notebook, executed, ~1 min
```
