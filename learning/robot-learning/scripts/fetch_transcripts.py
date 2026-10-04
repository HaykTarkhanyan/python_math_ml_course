"""Fetch YouTube transcripts for the ETH Robot Learning 2026 course videos.

Run from the repo root (~7 min for all 21 videos, ~20 s each + a pause between):

    python learning/robot-learning/scripts/fetch_transcripts.py

For each video it saves the raw caption track and info JSON to `_raw/<id>/`
(gitignored), then writes a cleaned, timestamped Markdown transcript into the
week folder. Already-fetched videos are skipped; delete `_raw/<id>/` to refetch.
Pass --rebuild to regenerate every Markdown file from `_raw/` without network.

Captions are YouTube's auto-generated English track (the course uploads have no
manual subtitles), so expect ASR errors in names and jargon.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # learning/robot-learning
REPO = ROOT.parents[1]
RAW = ROOT / "_raw"

# (week folder, output file stem, speaker, YouTube id) - from the course page,
# cross-checked against the channel's upload list on 2026-10-04.
VIDEOS = [
    ("week01_intro", "transcript_lecture", "Oier Mees", "X0k14u6pSxw"),
    ("week02_control_mdp", "transcript_lecture", "Oier Mees", "5-Bb84eTTqQ"),
    ("week02_control_mdp", "transcript_guest_abhishek_gupta", "Abhishek Gupta", "aG8NPTPhwkE"),
    ("week03_imitation", "transcript_lecture", "Oier Mees", "Ef4R5s1LqoQ"),
    ("week03_imitation", "transcript_guest_danfei_xu", "Danfei Xu", "qvTP6T5oq1w"),
    ("week04_rl_1", "transcript_lecture", "Oier Mees", "90raNpc11tQ"),
    ("week04_rl_1", "transcript_guest_aviral_kumar", "Aviral Kumar", "fHHLmTu9sFk"),
    ("week05_rl_2", "transcript_lecture", "Oier Mees", "AdTGz8YnnlE"),
    ("week05_rl_2", "transcript_guest_andrew_wagenmaker", "Andrew Wagenmaker", "CPmTpXA5azw"),
    ("week06_generative", "transcript_lecture", "Oier Mees", "qd6Ldsuu46I"),
    ("week06_generative", "transcript_guest_cheng_chi", "Cheng Chi", "tvFvIEOBKfM"),
    ("week07_sequence_modeling", "transcript_lecture", "Oier Mees", "imSTfMJjp7M"),
    ("week07_sequence_modeling", "transcript_guest_ted_xiao", "Ted Xiao", "VS7Ulaugevg"),
    ("week08_world_models", "transcript_lecture", "Oier Mees", "cTTmUZlOF2s"),
    ("week08_world_models", "transcript_guest_scott_reed", "Scott Reed", "fqkp_wkov6M"),
    ("week09_generalist_policies", "transcript_lecture", "Oier Mees", "dtofzDY9zuo"),
    ("week09_generalist_policies", "transcript_guest_quan_vuong", "Quan Vuong", "pzolgvyWEFY"),
    ("week10_reasoning", "transcript_lecture", "Oier Mees", "CxhrjQuGEuE"),
    ("week10_reasoning", "transcript_guest_archit_sharma", "Archit Sharma", "oBEkY6NeE_o"),
    ("week11_frontiers", "transcript_lecture", "Oier Mees", "eL4lcy1KNzE"),
    ("week11_frontiers", "transcript_guest_lucas_beyer", "Lucas Beyer", "0XB7fNS_ONg"),
]

PAUSE_S = 5  # between videos, to stay clear of YouTube's per-IP rate limit

TS_RE = re.compile(r"(\d\d):(\d\d):(\d\d)\.\d\d\d --> ")
TAG_RE = re.compile(r"<[^>]+>")

(REPO / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPO / "logs" / "fetch_transcripts.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def fetch(video_id: str) -> Path:
    """Download the info JSON and English auto-captions into _raw/<id>/."""
    dest = RAW / video_id
    if (dest / "v.info.json").exists() and (dest / "v.en.vtt").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "--no-warnings", "--skip-download", "--write-info-json",
        "--write-auto-subs", "--write-subs", "--sub-langs", "en",
        "-o", str(dest / "v.%(ext)s"), f"https://www.youtube.com/watch?v={video_id}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0 or not (dest / "v.en.vtt").exists():
        raise RuntimeError(f"yt-dlp failed for {video_id} (exit {res.returncode}): {res.stderr[-800:]}")
    return dest


def parse_vtt(path: Path) -> list[tuple[int, str]]:
    """Rolling auto-captions -> (start_second, line) with repeated lines dropped."""
    cues, cur = [], None
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = TS_RE.match(ln)
        if m:
            h, mi, s = map(int, m.groups())
            cur = h * 3600 + mi * 60 + s
            continue
        s = ln.strip()
        if not s or s == "WEBVTT" or s.startswith(("Kind:", "Language:")):
            continue
        txt = TAG_RE.sub("", s).strip()
        if txt and (not cues or cues[-1][1] != txt):
            cues.append((cur, txt))
    return cues


def hms(sec: int) -> str:
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def render(week: str, stem: str, speaker: str, video_id: str) -> int:
    """Write the Markdown transcript; return its word count."""
    raw = RAW / video_id
    info = json.loads((raw / "v.info.json").read_text(encoding="utf-8"))
    cues = parse_vtt(raw / "v.en.vtt")
    if not cues:
        raise RuntimeError(f"{video_id}: caption file parsed to zero lines")

    # one paragraph per minute, each opening with a timestamp that links into the video
    paras, buf, start, minute = [], [], None, None
    for sec, txt in cues:
        if minute is not None and sec // 60 != minute:
            paras.append((start, " ".join(buf)))
            buf, start = [], None
        if start is None:
            start = sec
        buf.append(txt)
        minute = sec // 60
    paras.append((start, " ".join(buf)))

    up = info["upload_date"]
    lines = [
        f"# {info['title']}",
        "",
        f"- Speaker: {speaker}",
        f"- Video: https://www.youtube.com/watch?v={video_id}",
        f"- Duration: {info['duration_string']}, uploaded {up[:4]}-{up[4:6]}-{up[6:]}",
        "- Source: YouTube auto-generated English captions, cleaned (rolling duplicates removed),"
        " one paragraph per minute. Expect speech-recognition errors in names and jargon.",
        "",
    ]
    if info.get("chapters"):
        lines += ["## Chapters", ""]
        lines += [f"- [{hms(int(c['start_time']))}](https://youtu.be/{video_id}?t={int(c['start_time'])}) {c['title']}"
                  for c in info["chapters"]]
        lines.append("")
    lines += ["## Video description", "", info.get("description", "").strip(), "", "## Transcript", ""]
    lines += [f"[{hms(s)}](https://youtu.be/{video_id}?t={s}) {t}\n" for s, t in paras]

    out = ROOT / week / f"{stem}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return sum(len(t.split()) for _, t in paras)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-render Markdown from _raw/ only, no network")
    args = ap.parse_args()

    t0, failed = time.time(), []
    for i, (week, stem, speaker, vid) in enumerate(VIDEOS, 1):
        try:
            fetched = False
            if not args.rebuild and not (RAW / vid / "v.en.vtt").exists():
                fetch(vid)
                fetched = True
            words = render(week, stem, speaker, vid)
            elapsed = time.time() - t0
            eta = elapsed / i * (len(VIDEOS) - i)
            log.info(f"{i}/{len(VIDEOS)} {week}/{stem}.md: {words} words, {elapsed:.0f}s elapsed, ~{eta:.0f}s left")
            if fetched and i < len(VIDEOS):
                time.sleep(PAUSE_S)
        except Exception:
            log.exception(f"{i}/{len(VIDEOS)} FAILED {week}/{stem} ({vid})")
            failed.append(vid)

    log.info(f"done in {time.time() - t0:.0f}s")
    if failed:
        log.error(f"{len(failed)} video(s) failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
