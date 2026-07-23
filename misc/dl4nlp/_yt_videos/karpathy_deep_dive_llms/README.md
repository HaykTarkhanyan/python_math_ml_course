# Deep Dive into LLMs like ChatGPT — Andrej Karpathy

Reference material for the `dl4nlp` course. Karpathy's ~3.5h general-audience walk
through the full LLM pipeline: pretraining -> post-training -> reinforcement learning.

- **Source:** https://www.youtube.com/watch?v=7xTGNNLPyMI
- **Uploader:** Andrej Karpathy · **Published:** 2025-02-05 · **Duration:** 03:31:23 · 8.3M views
- **What it is:** end-to-end mental model for how ChatGPT-style models are built and why
  they behave the way they do (hallucinations, "tokens to think", spelling failures, RLHF).

## Files

| File | Committed? | What |
|------|-----------|------|
| `transcript.txt` | yes | cleaned, deduped, timestamp every minute (41k words) |
| `meta.txt` | yes | one-line metadata |
| `README.md` | yes | this file |
| `frames/` | yes (when grabbed) | `fNN_HH-MM-SS.jpg` stills |
| `video.mp4` | **no (git-ignored)** | 1080p H.264 download (~700 MB) |
| `video.en*.vtt` | no (git-ignored) | raw subtitle tracks |

## Re-fetch the video (git-ignored, not in history)

```bash
yt-dlp -f "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]" \
  --merge-output-format mp4 -o "video.%(ext)s" \
  "https://www.youtube.com/watch?v=7xTGNNLPyMI"
```

## Key beats (official chapter markers)

| Timestamp | Chapter |
|-----------|---------|
| 00:00:00 | introduction |
| 00:01:00 | pretraining data (internet) — FineWeb, Common Crawl, filtering |
| 00:07:47 | tokenization — bytes -> BPE, vocabulary vs sequence-length tradeoff |
| 00:14:27 | neural network I/O |
| 00:20:11 | neural network internals |
| 00:26:01 | inference |
| 00:31:09 | GPT-2: training and inference |
| 00:42:52 | Llama 3.1 base model inference |
| 00:59:23 | pretraining -> post-training |
| 01:01:06 | post-training data (conversations) — SFT, assistant behavior |
| 01:20:32 | hallucinations, tool use, knowledge / working memory |
| 01:41:46 | knowledge of self |
| 01:46:56 | models need tokens to think |
| 02:01:11 | tokenization revisited — why models struggle with spelling |
| 02:04:53 | jagged intelligence |
| 02:07:28 | supervised finetuning -> reinforcement learning |
| 02:14:42 | reinforcement learning |
| 02:27:47 | DeepSeek-R1 |
| 02:42:07 | AlphaGo |
| 02:48:26 | reinforcement learning from human feedback (RLHF) |
| 03:09:39 | preview of things to come |
| 03:15:15 | keeping track of LLMs |
| 03:18:34 | where to find LLMs |
| 03:21:46 | grand summary |

## Frame sets

_(none grabbed yet — add here when frames are extracted, e.g. `s1_tokenization_00-07-47.jpg`)_
