#!/bin/bash
# Render poster.html at every platform size with headless Edge, at 2x pixel density.
# Run from anywhere: bash misc/social_posts/2026-10_llm_launch/render.sh   (~1 min, ~10 s per image)
# Needs internet for the Google Fonts (Noto Sans Armenian, JetBrains Mono).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd -W)"   # Windows-style path, Edge wants it
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
# Own throwaway profile: with the default one, headless Edge hangs while a normal Edge window is open.
PROFILE="$(cygpath -w "$(mktemp -d)")"

# platform  layout     width  height
SIZES="
linkedin  portrait   1080 1350
linkedin  square     1080 1080
linkedin  landscape  1200  627
facebook  portrait   1080 1350
facebook  square     1080 1080
facebook  story      1080 1920
"

echo "$SIZES" | while read -r platform layout w h; do
  [ -z "$platform" ] && continue
  mkdir -p "$DIR/out/$platform"
  out="$DIR/out/$platform/llm_${platform}_${layout}_${w}x${h}@2x.png"
  log="$("$EDGE" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
    --user-data-dir="$PROFILE" --window-size="$w,$h" --virtual-time-budget=10000 \
    --screenshot="$out" "file:///$DIR/poster.html?s=$layout" 2>&1)"
  grep -q "bytes written" <<< "$log" || { echo "FAILED: $platform $layout" >&2; echo "$log" >&2; exit 1; }
  echo "ok  $platform/$(basename "$out")"
done
