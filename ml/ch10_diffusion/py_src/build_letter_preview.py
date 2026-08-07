"""Student-facing preview of the ՊԱՆԻՐ letter set.

Shows the same eight samples per class at native-cropped, 32, 24, 20 and 16 px, so the
identity of each letter and the cost of each downsample are visible on one screen, then
spells ՊԱՆԻՐ from real samples at each resolution.

The chapter trains at 24x24 (see DECISIONS.md); the other columns are kept so the choice
is arguable rather than asserted.

Reads:  data/mashtots_raw/  (run extract_mashtots.py first)
Writes: mashtots_letters.html  (self-contained, ~0.2 MB)
"""

import base64
import io
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# This script logs Armenian letters; the Windows console defaults to cp1252 and mangles them.
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8")

CHAPTER = Path(__file__).resolve().parent.parent
RAW = CHAPTER / "data" / "mashtots_raw" / "Train" / "Train"
OUT = CHAPTER / "mashtots_letters.html"

LOGS = CHAPTER.parent.parent / "logs"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "build_letter_preview.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# Folder id -> letter, in ՊԱՆԻՐ order. Ids are the competition's own labels.
LETTERS = [(26, "Պ", "PEH"), (0, "Ա", "AYB"), (21, "Ն", "NOW"), (10, "Ի", "INI"), (32, "Ր", "REH")]
SIZES = [None, 32, 24, 20, 16]   # None = native, cropped only
TRAIN_SIZE = 24                  # highlighted as the chosen one
N_SHOW = 8
THRESH = 40
SEED = 509


def crop_to_ink(a, pad_frac=0.12):
    """Tight-crop to the glyph, then pad to a square with a small margin.

    The glyph occupies only ~34-40 px of the 64 px frame, so resizing the raw frame spends
    most of its pixels on empty margin and dissolves 1-2 px strokes. Cropping first roughly
    doubles the surviving ink at 16-24 px.
    """
    ys, xs = np.where(a > THRESH)
    if len(ys) == 0:
        return a
    top, bot, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = bot - top + 1, right - left + 1
    side = int(max(h, w) * (1 + 2 * pad_frac))
    out = np.zeros((side, side), dtype=a.dtype)
    y0, x0 = side // 2 - h // 2, side // 2 - w // 2
    out[y0:y0 + h, x0:x0 + w] = a[top:bot + 1, left:right + 1]
    return out


def prep(a, n):
    c = crop_to_ink(a)
    if n is not None:
        c = np.asarray(Image.fromarray(c).resize((n, n), Image.LANCZOS))
    c = c.astype(np.float32)
    if c.max() > 0:
        c *= 255.0 / c.max()          # thin antialiased strokes never reach full white
    return c.astype(np.uint8)


def uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
body{background:#111;color:#eee;font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:28px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#888;margin-bottom:26px;max-width:780px}
.cls{margin:0 0 30px;border:1px solid #2a2a2a;border-radius:8px;padding:14px 16px;background:#161616}
.hdr{display:flex;align-items:baseline;gap:14px;margin-bottom:10px}
.glyph{font-family:Sylfaen,serif;font-size:40px;color:#7fb2ff;line-height:1}
.meta{color:#888;font-size:13px} b{color:#ddd}
table{border-collapse:collapse} td{padding:3px}
td.lbl{color:#888;font-size:12px;text-align:right;padding-right:10px;white-space:nowrap}
tr.pick td.lbl{color:#7fb2ff;font-weight:600}
tr.pick img{outline:1px solid #7fb2ff33}
img{image-rendering:pixelated;display:block;border-radius:2px}
.word{display:flex;align-items:center;gap:2px;margin:4px 0 14px}
.word img{height:88px;image-rendering:pixelated}
.wlbl{color:#888;font-size:12px;width:74px}
.word.pick .wlbl{color:#7fb2ff;font-weight:600}
"""


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"missing extracted data: {RAW} - run extract_mashtots.py first")
    rng = np.random.default_rng(SEED)

    picks = {}
    for cid, _, _ in LETTERS:
        files = sorted((RAW / str(cid)).glob("*.png"))
        if not files:
            raise RuntimeError(f"no PNGs for class {cid} in {RAW}")
        picks[cid] = [np.asarray(Image.open(f).convert("L")) for f in rng.choice(files, N_SHOW, replace=False)]
        log.info("class %-2d  %d of %d images", cid, N_SHOW, len(files))

    blocks = []
    for cid, arm, name in LETTERS:
        rows = []
        for n in SIZES:
            label = "native (cropped)" if n is None else f"{n}x{n}"
            cls = ' class="pick"' if n == TRAIN_SIZE else ""
            tds = "".join(f'<td><img src="{uri(prep(a, n))}" style="width:88px;height:88px"></td>' for a in picks[cid])
            rows.append(f'<tr{cls}><td class="lbl">{label}</td>{tds}</tr>')
        blocks.append(
            f'<div class="cls"><div class="hdr"><span class="glyph">{arm}</span>'
            f'<span class="meta">folder <b>{cid}</b> &middot; <b>{arm}</b> ({name})</span></div>'
            f'<table>{"".join(rows)}</table></div>'
        )

    words = []
    for n in SIZES:
        label = "native" if n is None else f"{n}x{n}"
        cls = " pick" if n == TRAIN_SIZE else ""
        imgs = "".join(f'<img src="{uri(prep(picks[c][0], n))}">' for c, _, _ in LETTERS)
        words.append(f'<div class="word{cls}"><span class="wlbl">{label}</span>{imgs}</div>')

    html = (
        '<!doctype html><meta charset="utf-8"><title>ՊԱՆԻՐ - Mashtots letter set</title>'
        f"<style>{CSS}</style>"
        "<h1>ՊԱՆԻՐ - the five letters, at five resolutions</h1>"
        '<div class="sub">Handwritten Armenian capitals from the Mashtots dataset. Each row shows the '
        "same eight samples, so you can follow one glyph down the column as it loses pixels. "
        f"The chapter trains at <b>{TRAIN_SIZE}x{TRAIN_SIZE}</b> (highlighted): below that the thin "
        "strokes start to break up, above it training time grows faster than the image does.</div>"
        f'{"".join(blocks)}'
        '<div class="cls"><div class="hdr"><span class="glyph">ՊԱՆԻՐ</span>'
        '<span class="meta">one real sample per letter, pasted side by side - each written by a '
        "different hand, which is exactly the problem the homework ends on</span></div>"
        f'{"".join(words)}</div>'
    )
    OUT.write_text(html, encoding="utf-8")
    log.info("wrote %s (%.2f MB)", OUT, OUT.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
