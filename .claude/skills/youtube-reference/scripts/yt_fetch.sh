#!/usr/bin/env bash
# Fetch a video's metadata, transcript, and 720p download into a repo reference folder.
#
# Usage: bash yt_fetch.sh <URL> <DEST_DIR>
#   e.g. bash yt_fetch.sh "https://youtu.be/XXXX" ml/ch5_neural_networks/_reference_welchlabs
#
# Produces in DEST: meta.txt, transcript.txt, video.mp4 (git-ignored), video.en*.vtt
# (git-ignored), a .gitignore, and an empty frames/ dir. Write README.md yourself after.
set -uo pipefail

URL="${1:?usage: bash yt_fetch.sh <URL> <DEST_DIR>}"
DEST="${2:?usage: bash yt_fetch.sh <URL> <DEST_DIR>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DEST/frames"

echo "== metadata =="
yt-dlp --no-warnings --print \
  "%(title)s | %(duration>%H:%M:%S)s | %(uploader)s | %(upload_date)s | %(view_count)s views" \
  "$URL" | tee "$DEST/meta.txt"

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

echo "== download 720p =="
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
  --merge-output-format mp4 -o "$DEST/video.%(ext)s" "$URL" 2>&1 | tail -2

# keep the large/raw assets out of git history
cat > "$DEST/.gitignore" <<'EOF'
# Large/raw assets - kept locally, not committed
video.mp4
*.vtt
EOF

echo "== done =="
ls -la "$DEST"
