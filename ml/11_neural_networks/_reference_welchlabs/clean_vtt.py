import re, sys
src = sys.argv[1]
out = sys.argv[2]
lines = open(src, encoding="utf-8").read().splitlines()
ts_re = re.compile(r'(\d\d:\d\d:\d\d)\.\d\d\d --> ')
tag_re = re.compile(r'<[^>]+>')
cues = []
cur_ts = None
for ln in lines:
    m = ts_re.match(ln)
    if m:
        cur_ts = m.group(1)
        continue
    if not ln.strip() or ln.strip() in ("WEBVTT",) or ln.startswith("Kind:") or ln.startswith("Language:"):
        continue
    txt = tag_re.sub("", ln).strip()
    if not txt:
        continue
    cues.append((cur_ts, txt))
# dedupe: keep a line only when text changes from last emitted
clean = []
last = None
for ts, txt in cues:
    if txt == last:
        continue
    clean.append((ts, txt))
    last = txt
# collapse into ~ every-15s paragraphs with a leading timestamp
with open(out, "w", encoding="utf-8") as f:
    block_ts = None
    buf = []
    def flush():
        if buf:
            f.write(f"[{block_ts}] " + " ".join(buf) + "\n\n")
    last_h = None
    for ts, txt in clean:
        mm = ts[:5]  # HH:MM
        if block_ts is None:
            block_ts = ts
        # new paragraph every new minute
        if last_h is not None and ts[:5] != last_h:
            flush(); buf=[]; block_ts=ts
        buf.append(txt)
        last_h = ts[:5]
    flush()
print("clean lines:", len(clean))
