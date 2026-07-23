---
name: youtube-reference
description: >
  Use whenever the user shares a video link (youtu.be / youtube.com or similar),
  typically phrased "check this URL", "watch this video", "check <link>", or
  "watch <link>" - or any intent to get its transcript, screenshot it, summarize
  it, or borrow visuals into slides. Runs the yt-dlp + ffmpeg pipeline (metadata,
  cleaned timestamped transcript, 720p download, frame grabs) into a repo
  _reference_<slug>/ folder, then REVIEWS the transcript and key frames, SUGGESTS
  concrete repo-grounded ways to use the video, and INTERVIEWS the user on what and
  how to incorporate before changing any files. Trigger on a bare "check/watch
  <URL>" even without the words yt-dlp, transcript, or screenshot.
allowed-tools:
  - Read
  - Bash
  - Edit
  - Write
  - Glob
  - Grep
  - AskUserQuestion
user-invocable: true
---

# YouTube reference pipeline

Turn a video link into usable reference material, then help the user decide what to do
with it. The golden rule: **fetch and review first, then suggest and interview, then
incorporate - never auto-incorporate.** The user shares a link to explore it; what
ends up in the slides (or a figure, an example, a summary) is their call, made after
they have seen what the video offers.

Tools assumed present: `yt-dlp`, `ffmpeg`, `python` (not `python3`). The `ma` venv is
only needed if you post-process with pandas/numpy - this pipeline does not.

## Workflow

1. **Fetch.** *First check the video isn't already mined* - grep the repo for the video id
   (`grep -rl <id> ml/`). A deck may already carry its transcript (`research/`) and frames
   (`fig/borrowed/`), in which case reuse them and skip the ~100 MB re-download; only re-fetch for
   a genuinely new purpose. Otherwise run `yt_fetch.sh` -> metadata, `transcript.txt`, 720p
   `video.mp4`, `.gitignore`, into `ml/<chapter>/_reference_<slug>/` (see storage convention
   below). **For full-bleed showcase slides, fetch 1080p** (`bestvideo[height<=1080]+...`) - 720p
   looks soft blown up to full screen.
2. **Review.** Read `transcript.txt` to map the structure. Grab a first set of frames at
   the key visual beats with `grab_frames.sh`, and Read the handful that matter to
   confirm content. Skip sponsor/ad reads and the outro.
3. **Suggest.** Report back: what the video is, its structure, and - the part that makes
   this skill worth it - **2-4 concrete, repo-grounded suggestions for how to use it.**
   Not generic ("you could make slides") but specific: which frames map to which existing
   deck or frame or topic, and whether each point is better as a full-bleed still, a
   redrawn house-style figure, a worked example, or a summary note. Ground every
   suggestion in a real target in this repo - a deck under `ml/`, a chapter `.qmd`, a
   `py_src/` figure - and in the instructor's pedagogy (predict-first, worked numbers,
   trees-vs-nets honesty, etc.). Give a recommendation, not just a menu.
4. **Interview.** Ask what and how they want to incorporate **before touching any deck or
   figure.** Cover: the goal (slides / figure / example / summary / just understanding);
   if slides, *which deck* (confirm the live target - this repo has parallel NN decks:
   legacy `L14/L15` house-cut vs the active `dl_*` LMU-embed set - ask if unsure), which
   frames, full-bleed still vs house-style figure-with-caption, and where in the deck;
   caption/attribution preference; how many frames. Use `AskUserQuestion` when a few
   discrete choices will settle it; otherwise just ask.
5. **Incorporate.** Do what they asked, following the relevant skill (`slide-style` for
   decks, `make-figures` for redrawn figures). Copy chosen frames into the deck's sibling
   `fig/`. See "Embedding frames into a deck" below for the mechanics.
6. **Verify.** Compile twice, eyeball the render, clean aux files, record the change in
   the deck's `% Provenance:` block, open the PDF. Report placements (page numbers) so
   the user can adjust.

Steps 1-3 are cheap and safe to run proactively on a bare "check this URL". Steps 5-6
change repo files - gate them behind the step-4 interview.

## The fast path (steps 1-2)

Two bundled scripts do the deterministic work. Run them from the skill's `scripts/`:

```bash
SK=".claude/skills/youtube-reference/scripts"
DEST="ml/<chapter>/_reference_<slug>"       # sibling of the deck it supports
bash "$SK/yt_fetch.sh" "<URL>" "$DEST"      # metadata + transcript + 720p + .gitignore
```

Then **read `$DEST/transcript.txt`**, pick timestamps, and grab frames:

```bash
bash "$SK/grab_frames.sh" "$DEST/video.mp4" "$DEST/frames" \
  00:00:35 00:01:45 00:03:52 00:11:40 00:24:35
```

`grab_frames.sh` writes `frames/fNN_HH-MM-SS.jpg`. Read the ones that matter, then move
to step 3 (suggest).

### Curating many candidates: tile them into one contact sheet
When you grab 20+ candidates, Reading each one burns context. Tile them into a single image and
Read that **once**. Order by timestamp, renumber to a clean sequence, and print a
position->timestamp map so every tile is identifiable:

```bash
F="$DEST/frames"; SEQ="$CLAUDE_JOB_DIR/tmp/seq"; mkdir -p "$SEQ"
i=1; for f in $(ls "$F"/*.jpg | sed -E 's/.*_([0-9-]+)\.jpg/\1\t&/' | sort | cut -f2); do
  cp "$f" "$SEQ/seq_$(printf '%03d' $i).jpg"; echo "$i=$(basename "$f")"; i=$((i+1)); done
ffmpeg -nostdin -loglevel error -y -i "$SEQ/seq_%03d.jpg" \
  -vf "scale=480:270,tile=6x6" -frames:v 1 "$CLAUDE_JOB_DIR/tmp/sheet.png"
```

Read `sheet.png`, pick winners by tile position, then Read only those 2-3 at full res to confirm
they caught a **settled** state - animated explainers (3b1b, etc.) hold each beat, so grab a
second or two *after* the narration lands, not mid-transition/scroll. The same tile trick verifies
the finished deck: `pdftoppm -png -r 32 deck.pdf p` then tile the pages for a whole-deck overflow
glance before rendering the two or three you want to inspect closely at `-r 110`.

## Pipeline steps (what the scripts do, so you can vary them)

1. **Metadata** - confirm what the link is before downloading:
   `yt-dlp --print "%(title)s | %(duration>%H:%M:%S)s | %(uploader)s | %(upload_date)s | %(view_count)s views" URL`
2. **Transcript** - `yt-dlp --write-auto-subs --write-subs --sub-langs "en.*" --skip-download`
   writes rolling-duplicate VTT; `clean_vtt.py` dedupes it into `transcript.txt` with a
   timestamp every minute. Read this first - it is how you choose frame timestamps.
3. **Download** - `yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" --merge-output-format mp4`.
   720p is enough to see visuals and keeps the file ~50-150 MB instead of GBs.
4. **Frames** - `ffmpeg -ss <ts> -i <video> -frames:v 1 -q:v 3 out.jpg`. Put `-ss`
   **before** `-i` for fast seeking.

### Choosing timestamps
Read the transcript and target the moments a slide or summary would actually use.
**Skip sponsor/ad reads and the outro** - dead weight. For a dense animated explainer,
sample ~every 5-10 s where visuals change fast, and pick single beats elsewhere. Name
topic-clustered pulls `s1_<topic>_HH-MM-SS.jpg` (e.g. `s1_relu_00-11-22.jpg`) so related
frames sort together. Alternative for slide-heavy videos: scene-detection -
`ffmpeg -i video.mp4 -vf "select='gt(scene,0.3)',showinfo" -vsync vfr frames/scene_%03d.jpg` -
which captures each distinct visual state but can over/under-produce; transcript-driven
fixed timestamps are usually more predictable.

## Repo storage convention

Everything lands in `ml/<chapter>/_reference_<slug>/` (the `_reference_` prefix matches
how LMU source is vendored here). Layout:

```
_reference_<slug>/
  transcript.txt   # cleaned, timestamped  (committed)
  frames/          # fNN_HH-MM-SS.jpg       (committed, small)
  README.md        # what it is, beat map, re-fetch command (committed)
  meta.txt         # one-line metadata      (committed)
  video.mp4        # 720p download          (GIT-IGNORED, large)
  video.en*.vtt    # raw subs               (git-ignored)
```

`yt_fetch.sh` writes a `.gitignore` excluding `video.mp4` and `*.vtt` - keep it. A ~90 MB
video does not belong in git history; the README carries the yt-dlp line to re-fetch it.
**Never** commit the video unless the user explicitly asks.

Write `README.md` yourself after grabbing frames - the script cannot summarize content.
Include: source URL + uploader + date + duration, a one-line topic, the re-fetch command,
a list of frame sets and what each illustrates, and a timestamped "key beats" map. See
`ml/ch5_neural_networks/_reference_welchlabs/README.md` for a worked example.

## Gotchas (learned the hard way)

- **`/tmp` is a trap on this machine.** Git Bash `/tmp` does not map to a Windows path the
  Read tool can open. Render PNGs / write files you need to *view* into a Windows-visible
  dir (the repo folder, or `$CLAUDE_JOB_DIR/tmp`) and Read them with the `C:\...` path.
- **The Bash tool times out at 2 min.** ~30 sequential ffmpeg grabs will blow it. Batch to
  ~15-20 per call, or use `run_in_background: true`. (The calls usually finish even when
  the *wrapper* reports a timeout - check with `ls` before re-running.)
- **VTT auto-subs are full of rolling duplicate lines** - always run `clean_vtt.py`, never
  read the raw `.vtt`.
- **Translated subtitle langs (e.g. `en-tr`) can throw HTTP 429.** Harmless - the real
  English track still downloads. `yt_fetch.sh` tolerates this.
- **`WARNING: ... no impersonate target is available`** from yt-dlp is harmless noise.
- **Reading 40+ frames into context is wasteful.** Extract generously to disk (cheap, it
  is a deliverable), but only Read the handful you need - or tile many into one contact sheet
  (see "Curating many candidates" above) and Read that once.

## Embedding frames into a deck (step 5 mechanics)

Only after the interview. This is a slide-deck edit, so `slide-style` and
`ml/SLIDE_STYLE.md` govern it (attribution line on third-party figures, provenance block,
etc.). The parts specific to video stills:

1. Copy chosen frames from `_reference_<slug>/frames/` into the deck's sibling `fig/` with
   descriptive names: `cp frames/f12_....jpg fig/welchlabs_deep_vs_wide.jpg`.
2. Add a full-bleed macro **matching the deck's aspect ratio** (video stills are 16:9):
   - **16:9 deck** (`aspectratio=169`) - image fills exactly, no bars:
     ```latex
     \newcommand{\wlslide}[1]{%
       {\usebackgroundtemplate{\includegraphics[width=\paperwidth,height=\paperheight]{#1}}%
       \begin{frame}[plain]
         \begin{tikzpicture}[remember picture, overlay]
           \node[anchor=south west, font=\tiny, text=white!70]
             at ([shift={(0.28cm,0.18cm)}]current page.south west) {Source (Year)};
         \end{tikzpicture}
       \end{frame}}%
     }
     ```
   - **4:3 deck** (`aspectratio=43`, e.g. the `dl_*` LMU-embed decks) - letterbox on black
     so the 16:9 still is **not** distorted:
     ```latex
     \newcommand{\wlslide}[1]{%
       {\setbeamercolor{background canvas}{bg=black}%
       \begin{frame}[plain]
         \begin{tikzpicture}[remember picture, overlay]
           \node at (current page.center) {\includegraphics[width=\paperwidth]{#1}};
           \node[anchor=south west, font=\tiny, text=white!70]
             at ([shift={(0.28cm,0.18cm)}]current page.south west) {Source (Year)};
         \end{tikzpicture}
       \end{frame}}%
     }
     ```
3. Call `\wlslide{fig/name.jpg}` between frames at the matching teaching moment. Compile
   twice (needed for `remember picture`), verify the render, then clean with
   `./ma/Scripts/python.exe clean_latex.py <dir>`.
