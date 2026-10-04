# Robot Learning: From Fundamentals to Foundation Models (ETH Zurich, Spring 2026)

Personal study copy of the course materials, in machine-readable form.

- Course page: https://cvg.ethz.ch/lectures/Robot-Learning/
- Lecturer: Oier Mees (https://www.oiermees.com), ETH 263-5911-00L
- Official homework repo: https://github.com/mees-robot-learning-course/ethz-course-2026
- Slide frames: https://github.com/idanL1212000/RobotLerning-2026-ETH-Zurich (`lectures/`)
- Fetched 2026-10-04

## Browser

**Online: https://hayktarkhanyan.github.io/python_math_ml_course/robot-learning/**

One page per week: summary and key ideas, the lecture and guest talk with a clickable contents list (YouTube chapters, or estimated from the slides for weeks 1, 3, 8, 9, 11) and a transcript that follows the video, slide text with the original frames, transcriber notes, and a personal notes box. Search covers all transcripts and slides.

- Your notes are kept only in your browser's local storage: not in the repo, not online. Use **Export notes** to download them.
- Rebuild after changing any transcript, `slides.md` or `scripts/browser_content.json` (~1 min): `./ma/Scripts/python.exe learning/robot-learning/scripts/build_browser.py`, then commit `browser/data.js` (and `browser/frames/` if slides changed). Pushing to `main` redeploys via `.github/workflows/publish.yml`.
- Locally: open `browser/index.html`; for the embedded player run `python -m http.server 8765 -d learning/robot-learning/browser` and open http://localhost:8765 (YouTube refuses `file://` pages).
- Estimated contents lists were checked on weeks 9 and 11 against the videos: the right slide is on screen at the estimated minute 76-81% of the time, within one slide 93-98%.

## Layout

```
weekNN_topic/
  transcript_lecture.md            lecture transcript (timestamps link into the video)
  transcript_guest_<name>.md       guest talk transcript (weeks 2-11)
  slides.pdf                       image-only slide frames (gitignored, ~35 MB total)
  slides.md                        slide text with LaTeX math (all 11 weeks)
homework/                          assignment READMEs hw1-hw4 from the official repo
browser/                           the course browser (index.html, app.js, style.css hand-written;
                                   data.js and frames/ generated, 800px)
scripts/fetch_transcripts.py       rebuilds every transcript (see its docstring)
scripts/build_browser.py           rebuilds browser/data.js and browser/frames/
scripts/browser_content.json       per-week summaries, key ideas, papers, homework links
_raw/                              yt-dlp caption tracks + info JSON, the 720p week 9-11
                                   lecture videos, full-size slide frames (all gitignored)
```

## Schedule

| Week | Date | Topic | Lecture video | Guest talk |
|---|---|---|---|---|
| [01](week01_intro/) | Feb 16 | Introduction to Robot Learning | [X0k14u6pSxw](https://www.youtube.com/watch?v=X0k14u6pSxw) | - |
| [02](week02_control_mdp/) | Feb 23 | Robot Control & MDPs | [5-Bb84eTTqQ](https://www.youtube.com/watch?v=5-Bb84eTTqQ) | Abhishek Gupta (UW) [aG8NPTPhwkE](https://youtu.be/aG8NPTPhwkE) |
| [03](week03_imitation/) | Mar 02 | Imitation Learning | [Ef4R5s1LqoQ](https://youtu.be/Ef4R5s1LqoQ) | Danfei Xu (Georgia Tech) [qvTP6T5oq1w](https://youtu.be/qvTP6T5oq1w) |
| [04](week04_rl_1/) | Mar 09 | Reinforcement Learning I | [90raNpc11tQ](https://youtu.be/90raNpc11tQ) | Aviral Kumar (CMU, Google DeepMind) [fHHLmTu9sFk](https://youtu.be/fHHLmTu9sFk) |
| [05](week05_rl_2/) | Mar 16 | Reinforcement Learning II | [AdTGz8YnnlE](https://youtu.be/AdTGz8YnnlE) | Andrew Wagenmaker (UC Berkeley) [CPmTpXA5azw](https://youtu.be/CPmTpXA5azw) |
| [06](week06_generative/) | Mar 23 | Generative Models | [qd6Ldsuu46I](https://youtu.be/qd6Ldsuu46I) | Cheng Chi (Sunday Robotics) [tvFvIEOBKfM](https://youtu.be/tvFvIEOBKfM) |
| [07](week07_sequence_modeling/) | Mar 30 | Sequence Modeling & Transformers | [imSTfMJjp7M](https://youtu.be/imSTfMJjp7M) | Ted Xiao (Prometheus) [VS7Ulaugevg](https://youtu.be/VS7Ulaugevg) |
| [08](week08_world_models/) | Apr 13 | World Models | [cTTmUZlOF2s](https://youtu.be/cTTmUZlOF2s) | Scott Reed (NVIDIA GEAR) [fqkp_wkov6M](https://www.youtube.com/watch?v=fqkp_wkov6M) |
| [09](week09_generalist_policies/) | Apr 27 | Generalist Robot Policies | [dtofzDY9zuo](https://youtu.be/dtofzDY9zuo) | Quan Vuong (Physical Intelligence) [pzolgvyWEFY](https://youtu.be/pzolgvyWEFY) |
| [10](week10_reasoning/) | May 04 | Embodied Reasoning & Test-time Scaling | [CxhrjQuGEuE](https://youtu.be/CxhrjQuGEuE) | Archit Sharma (Google DeepMind) [oBEkY6NeE_o](https://youtu.be/oBEkY6NeE_o) |
| [11](week11_frontiers/) | May 11 | Frontier & Open Problems | [eL4lcy1KNzE](https://youtu.be/eL4lcy1KNzE) | Lucas Beyer (Meta) [0XB7fNS_ONg](https://youtu.be/0XB7fNS_ONg) |
| 12 | May 18 | Guest lecture: Dieter Fox (UW, AI2) | not uploaded | - |

The paper-discussion readings for each week are listed on the course page.

## Homework

| File | Topic | Due |
|---|---|---|
| [hw1_pytorch_tutorial.md](homework/hw1_pytorch_tutorial.md) | PyTorch & NumPy tutorial | March 5 |
| [hw2_robot_control_mdps.md](homework/hw2_robot_control_mdps.md) (+ [installation guide](homework/hw2_robot_control_mdps_Installation_Guide.md)) | Robot control & MDPs | March 12 |
| [hw3_imitation_learning.md](homework/hw3_imitation_learning.md) | Imitation learning | March 26 |
| [hw4_reinforcement_learning.md](homework/hw4_reinforcement_learning.md) | Reinforcement learning | April 16 |

Images and relative links inside these READMEs point into the original repo and do not resolve here.

## Where the content comes from, and its limits

- **Transcripts** are YouTube's auto-generated English captions (the uploads have no manual subtitles), with the rolling duplicate lines removed and one paragraph per minute. Names and jargon contain speech-recognition errors. 21 videos, ~160k words. Rebuild with `python learning/robot-learning/scripts/fetch_transcripts.py` (`--rebuild` re-renders from `_raw/` without network).
- **Slides**: the official PDFs on the course page are AES-256 encrypted with a user password (poppler: "Incorrect password"), so the source here is a student's frame grabs from the lecture recordings. They are images with no text layer and include animation builds as separate frames. Weeks 1-8 come from that repo at 1280x720. Its weeks 9-11 were only 640x360, so those three were re-extracted on 2026-10-04 from the 720p YouTube recordings with the repo's own `lectures/extract_slides.py` (default settings), giving 1280x720 frames; their frame numbering differs from the repo's PDFs.
- **slides.md** (all 11 weeks) was transcribed from the frame images by Claude: text verbatim, equations in LaTeX, figures described in brackets, animation builds and embedded-video frames merged, `[Note: ...]` for transcriber remarks (slide errors, missing slide numbers). Anything in square brackets, headings included, is the transcriber's wording; everything else is on the slide. Every file's equations compile (checked with pandoc + xelatex), and every frame is covered exactly once. TA e-mail addresses on week 1's staff slide are left out.
