"""Clean a YouTube auto-sub .vtt into a readable, timestamped transcript.

Auto-generated VTT is full of rolling duplicate lines and inline <c> timing tags.
This strips the tags, drops consecutive duplicate text, and emits one paragraph per
minute with a leading [HH:MM:SS] stamp.

Usage: python clean_vtt.py <input.vtt> <output.txt>
"""
import re
import sys

if len(sys.argv) != 3:
    sys.exit("usage: python clean_vtt.py <input.vtt> <output.txt>")

src, out = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").read().splitlines()

ts_re = re.compile(r"(\d\d:\d\d:\d\d)\.\d\d\d --> ")
tag_re = re.compile(r"<[^>]+>")

cues = []
cur_ts = None
for ln in lines:
    m = ts_re.match(ln)
    if m:
        cur_ts = m.group(1)
        continue
    s = ln.strip()
    if not s or s == "WEBVTT" or s.startswith("Kind:") or s.startswith("Language:"):
        continue
    txt = tag_re.sub("", ln).strip()
    if txt:
        cues.append((cur_ts, txt))

# drop a line only when its text repeats the last emitted line
clean = []
last = None
for ts, txt in cues:
    if txt != last:
        clean.append((ts, txt))
        last = txt

# collapse into one paragraph per minute, each with a leading timestamp
with open(out, "w", encoding="utf-8") as f:
    block_ts = None
    buf = []
    last_hm = None

    def flush():
        if buf:
            f.write(f"[{block_ts}] " + " ".join(buf) + "\n\n")

    for ts, txt in clean:
        if block_ts is None:
            block_ts = ts
        if last_hm is not None and ts[:5] != last_hm:  # new minute -> new paragraph
            flush()
            buf = []
            block_ts = ts
        buf.append(txt)
        last_hm = ts[:5]
    flush()

print(f"clean lines: {len(clean)}")
