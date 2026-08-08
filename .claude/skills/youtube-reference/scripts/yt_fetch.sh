#!/usr/bin/env bash
# Fetch a video's metadata, transcript, and 720p download into a repo reference folder.
#
# Usage: bash yt_fetch.sh <URL> <DEST_DIR>
#   e.g. bash yt_fetch.sh "https://youtu.be/XXXX" ml/ch5_neural_networks/_reference_welchlabs
#
# Produces in DEST: meta.txt, description.txt (chapters + links mentioned in the video),
# transcript.txt, video.mp4 (git-ignored), video.en*.vtt (git-ignored), a .gitignore,
# and an empty frames/ dir. Write README.md yourself after.
set -uo pipefail

URL="${1:?usage: bash yt_fetch.sh <URL> <DEST_DIR> [HEIGHT]}"
DEST="${2:?usage: bash yt_fetch.sh <URL> <DEST_DIR> [HEIGHT]}"
# Max video height. 720 keeps files ~50-150 MB and is enough to read visuals; pass 1080
# when the frames are destined for full-bleed showcase slides (720 looks soft blown up).
HEIGHT="${3:-720}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DEST/frames"

echo "== metadata =="
yt-dlp --no-warnings --print \
  "%(title)s | %(duration>%H:%M:%S)s | %(uploader)s | %(upload_date)s | %(view_count)s views" \
  "$URL" | tee "$DEST/meta.txt"

# Description carries the chapter timestamps and any links the author mentions
# (colab, repos, papers). Always save it - it is small and committed.
echo "== description (chapters + links) =="
yt-dlp --no-warnings --skip-download --print "%(description)s" "$URL" > "$DEST/description.txt" 2>/dev/null \
  && echo "description: $(wc -l < "$DEST/description.txt") lines -> description.txt" \
  || echo "WARNING: description not fetched"

echo "== subtitles =="
# en.* also pulls translated tracks (en-tr) that sometimes 429; tolerate it, the real
# English track still lands. || true so one failing lang does not abort the script.
yt-dlp --write-auto-subs --write-subs --sub-langs "en.*" --skip-download \
  -o "$DEST/video" "$URL" 2>&1 | tail -3 || true
VTT="$(ls "$DEST"/video.en.vtt "$DEST"/video.en-orig.vtt 2>/dev/null | head -1 || true)"
if [ -n "${VTT:-}" ]; then
  python "$HERE/clean_vtt.py" "$VTT" "$DEST/transcript.txt"
  echo "transcript words: $(wc -w < "$DEST/transcript.txt")"
else
  echo "WARNING: no English subtitles found - transcript.txt not written"
fi

echo "== download ${HEIGHT}p =="
# --js-runtimes node is REQUIRED: YouTube signs media URLs with obfuscated JS, and yt-dlp
# only enables deno by default, which is not installed here. Without it the media bytes
# 403 while metadata and subtitles still succeed - so the folder looks healthy but has no
# video. See _learnings/2026-08-03-1250_yt-dlp-needs-a-js-runtime.md.
yt-dlp --js-runtimes node \
  -f "bestvideo[height<=$HEIGHT]+bestaudio/best[height<=$HEIGHT]" \
  --merge-output-format mp4 -o "$DEST/video.%(ext)s" "$URL" 2>&1 | tail -2

# Partial success is the trap here - assert the mp4 actually landed rather than trusting
# a clean exit.
if [ -s "$DEST/video.mp4" ]; then
  echo "video.mp4: $(du -h "$DEST/video.mp4" | cut -f1)"
else
  echo "ERROR: $DEST/video.mp4 missing or empty - download failed" >&2
  exit 1
fi

# keep the large/raw assets out of git history
cat > "$DEST/.gitignore" <<'EOF'
# Large/raw assets - kept locally, not committed
video.mp4
*.vtt
EOF

echo "== done =="
ls -la "$DEST"
