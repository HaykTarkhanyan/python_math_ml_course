# yt-dlp now needs a JavaScript runtime, or every download 403s

**Symptom.** Fetching a YouTube video with the repo's `youtube-reference` pipeline died on the
download step, while every *other* step worked fine:

```
$ yt-dlp -f "137+bestaudio/..." --merge-output-format mp4 -o "$DEST/video.%(ext)s" "https://youtu.be/iv-5mZ_9CPY"
WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by
default; to use another runtime add --js-runtimes RUNTIME[:PATH] to your command/config.
[info] iv-5mZ_9CPY: Downloading 1 format(s): 137+251
ERROR: unable to download video data: HTTP Error 403: Forbidden
```

Metadata printed fine. The description printed fine. **Subtitles downloaded fine.** `yt-dlp -F`
listed every format including 1080p. Only the media bytes 403'd.

**Cause.** YouTube signs media URLs with a token computed by obfuscated JavaScript shipped with the
player page. yt-dlp needs a JS engine to run it. Metadata and subtitle endpoints are not signed
that way, which is why they kept working - and why this reads like rate-limiting or geo-blocking
when it is neither. yt-dlp only enables **deno** by default, and deno is not installed here.

**Fix.** Node v20.20.0 and bun 1.3.11 are both already on PATH. Point yt-dlp at one:

```bash
yt-dlp --js-runtimes node -f "bestvideo[height<=1080]+bestaudio" --merge-output-format mp4 \
  -o "video.%(ext)s" "<URL>"
```

Downloaded 408 MB / 1920x1080 / h264 on the retry, no other change. Verified with:

```
$ ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height -of csv=p=0 video.mp4
h264,1920,1080
```

**Consequences.**

- **`.claude/skills/youtube-reference/scripts/yt_fetch.sh` is currently broken** for any new video.
  It omits `--js-runtimes node` on its download line. It is also hard-coded to 720p, so it needs a
  quality argument anyway when a deck wants full-bleed stills.
- The misleading part is the **partial success**. `transcript.txt` and `meta.txt` land normally, so
  the folder looks healthy until you check for `video.mp4`. Do not infer the download worked from
  the script exiting without an obvious failure - `ls` the mp4, or `ffprobe` it.
- Environment-dependent, not video-dependent: it will hit every YouTube fetch on this machine until
  either deno is installed or the flag is added to the script.

**Disproven along the way.** The first guess was rate limiting, because the subtitle fetch *did*
emit a real `HTTP Error 429: Too Many Requests` for the translated `en-ar` / `en-id` tracks. That
429 is unrelated noise - the English track still downloads, and the 403 reproduced immediately on
retry rather than easing off the way throttling would.
