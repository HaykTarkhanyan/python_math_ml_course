#!/usr/bin/env bash
# Extract single frames from a downloaded video at given timestamps.
#
# Usage: bash grab_frames.sh <VIDEO> <OUT_DIR> <HH:MM:SS> [HH:MM:SS ...]
#   e.g. bash grab_frames.sh video.mp4 frames 00:00:35 00:01:45 00:11:40
#
# Writes OUT_DIR/fNN_HH-MM-SS.jpg (NN = 1-based order). -ss before -i = fast seek.
# Note: the Bash tool times out at ~2 min; keep batches to ~15-20 timestamps, or call
# this with run_in_background. Frames usually finish even if the wrapper reports timeout -
# check with `ls` before re-running.
set -uo pipefail

V="${1:?usage: bash grab_frames.sh <VIDEO> <OUT_DIR> <TS>...}"; shift
OUT="${1:?usage: bash grab_frames.sh <VIDEO> <OUT_DIR> <TS>...}"; shift
mkdir -p "$OUT"

i=0
for t in "$@"; do
  i=$((i + 1))
  label=$(printf "%02d" "$i")
  safe=$(echo "$t" | tr ':' '-')
  ffmpeg -nostdin -loglevel error -ss "$t" -i "$V" -frames:v 1 -q:v 3 \
    "$OUT/f${label}_${safe}.jpg" -y
done
echo "extracted $i frame(s) to $OUT"
