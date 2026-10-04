"""Build the static course browser: browser/data.js (+ slide frames in browser/frames/).

Run from the repo root (~15 s; the first run also extracts ~625 slide frames from
the slide PDFs, a few seconds more):

    ./ma/Scripts/python.exe learning/robot-learning/scripts/build_browser.py

Then open learning/robot-learning/browser/index.html. Embedded video playback needs
http, not file://, so for that serve the folder:

    python -m http.server 8765 -d learning/robot-learning/browser

Online: .github/workflows/publish.yml copies browser/ into the course site, at
https://hayktarkhanyan.github.io/python_math_ml_course/robot-learning/ - so commit
data.js and frames/ after rebuilding.

Inputs: the week folders (transcript_*.md, slides.md, slides.pdf), homework/,
README.md and the hand-written scripts/browser_content.json. index.html, app.js and
style.css are hand-written and only read data.js.

Lectures without YouTube chapters get a contents list estimated by aligning slide
sections to transcript minutes (TF-IDF similarity + monotonic dynamic programming),
so those times are approximate.
"""

import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]  # learning/robot-learning
REPO = ROOT.parents[1]
OUT = ROOT / "browser"
HW_REPO = "mees-robot-learning-course/ethz-course-2026"

ALIGN_LAMBDA = 0.3  # weight of the "slides progress roughly linearly in time" prior
WEB_WIDTH = 800  # slide frames served by the site; full-size copies stay in _raw/frames (DECISIONS #69)

PARA_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\(https://youtu\.be/[\w-]{11}\?t=(\d+)\) (.*)$")
CHAP_RE = re.compile(r"^- \[\d\d:\d\d:\d\d\]\(https://youtu\.be/[\w-]{11}\?t=(\d+)\) (.*)$")
HEAD_RE = re.compile(r"^## (.*?) \(frames? (\d+)(?:-(\d+))?\)\s*$")
SEC_RE = re.compile(r'<section id="([^"]*)" class="level2">\s*<h2>(.*?)</h2>(.*?)</section>', re.S)
TAG_RE = re.compile(r"<[^>]+>")

(REPO / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPO / "logs" / "build_browser.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def pandoc(md: str, *args: str) -> str:
    res = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "html5", "--mathml", "--wrap=none", *args],
        input=md, capture_output=True, text=True, encoding="utf-8",
    )
    if res.returncode != 0:
        raise RuntimeError(f"pandoc failed: {res.stderr}")
    for line in res.stderr.splitlines():
        if line.strip():
            log.warning(f"pandoc: {line}")
    return res.stdout


def plain(html: str) -> str:
    html = re.sub(r"<annotation[^>]*>.*?</annotation>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", html)).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()


# ---------- transcripts ----------

def parse_transcript(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").split("\n")
    meta, section, desc, chapters, paras = {}, None, [], [], []
    for ln in lines[1:]:
        if ln.startswith("## "):
            section = ln[3:].strip()
            continue
        if section is None and ln.startswith("- ") and ": " in ln:
            key, val = ln[2:].split(": ", 1)
            meta[key] = val
        elif section == "Chapters" and (m := CHAP_RE.match(ln)):
            chapters.append({"t": int(m.group(1)), "title": m.group(2)})
        elif section == "Video description":
            desc.append(ln)
        elif section == "Transcript" and (m := PARA_RE.match(ln)):
            paras.append({"t": int(m.group(4)), "text": m.group(5)})
    if not paras:
        raise RuntimeError(f"{path}: no transcript paragraphs parsed")
    vid = re.search(r"v=([\w-]{11})", meta["Video"]).group(1)
    return {
        "id": vid,
        "title": lines[0][2:].strip(),
        "speaker": meta["Speaker"],
        "duration": meta["Duration"].split(",")[0],
        "description": "\n".join(desc).strip(),
        "chapters": chapters,
        "paragraphs": paras,
    }


# ---------- slides ----------

def math_mask(line: str) -> list:
    """True for every character of line that sits inside $...$ or $$...$$ (delimiters included)."""
    mask, in_math, k = [False] * len(line), False, 0
    while k < len(line):
        if line[k] == "$":
            step = 2 if line.startswith("$$", k) else 1  # $$ is one delimiter, not two
            for q in range(k, min(k + step, len(line))):
                mask[q] = True
            in_math = not in_math
            k += step
            continue
        mask[k] = in_math
        k += 1
    return mask


def matching_bracket(line: str, start: int, mask: list) -> int:
    """Index of the ']' closing line[start] == '[', ignoring brackets inside math; -1 if none."""
    depth = 0
    for k in range(start, len(line)):
        if mask[k]:
            continue
        if line[k] == "[":
            depth += 1
        elif line[k] == "]":
            depth -= 1
            if depth == 0:
                return k
    return -1


def own_word_spans(line: str) -> list:
    """(start, end) of transcriber brackets: [ + letter ... ], outside math, not a [link](url)."""
    mask, spans, i = math_mask(line), [], 0
    while i < len(line):
        if line[i] == "[" and not mask[i] and line[i + 1:i + 2].isalpha():
            j = matching_bracket(line, i, mask)
            if j != -1 and line[j + 1:j + 2] != "(":
                spans.append((i, j + 1))
                i = j + 1
                continue
        i += 1
    return spans


def mark_own_words(line: str) -> str:
    """Wrap transcriber brackets ([Figure: ...], [Note: ...], ...) in styled spans.

    Only groups starting with a letter count; [-1, +1] or [156, 55] are slide text.
    """
    out, prev = [], 0
    for a, b in own_word_spans(line):
        inner = line[a:b]
        cls = "own note" if inner.startswith("[Note:") else "own"
        out.append(line[prev:a] + f'<span class="{cls}">{inner}</span>')
        prev = b
    out.append(line[prev:])
    return "".join(out)


def strip_own_words(text: str) -> str:
    out = []
    for line in text.split("\n"):
        prev, buf = 0, []
        for a, b in own_word_spans(line):
            buf.append(line[prev:a])
            prev = b
        buf.append(line[prev:])
        out.append("".join(buf))
    return "\n".join(out)


def in_display_block(lines: list) -> list:
    """True for lines inside (or opening/closing) a multi-line $$ ... $$ block."""
    flags, inside = [], False
    for ln in lines:
        opens_or_closes = ln.count("$$") % 2 == 1
        flags.append(inside or opens_or_closes)
        if opens_or_closes:
            inside = not inside
    return flags


def parse_slides(path: Path) -> dict:
    md = path.read_text(encoding="utf-8")
    lines = md.split("\n")
    heads, sec_md, cur = [], [], None
    for ln in lines:
        if ln.startswith("## "):
            m = HEAD_RE.match(ln)
            if not m:
                raise RuntimeError(f"{path}: heading without frame range: {ln}")
            heads.append(m)
            cur = []
            sec_md.append(cur)
        elif cur is not None:
            cur.append(ln)

    marked, in_code = [], False
    for ln, in_math_block in zip(lines, in_display_block(lines)):
        if ln.startswith("```"):
            in_code = not in_code
        keep = in_code or in_math_block or ln.startswith("# ")
        marked.append(ln if keep else mark_own_words(ln))
    html = pandoc("\n".join(marked), "--section-divs")

    about = re.search(r"</h1>(.*?)<section", html, re.S)
    found = SEC_RE.findall(html)
    if len(found) != len(heads):
        raise RuntimeError(f"{path}: {len(found)} html sections vs {len(heads)} headings")

    sections = []
    for (sid, h2, body), m, body_md in zip(found, heads, sec_md):
        a, b = int(m.group(2)), int(m.group(3) or m.group(2))
        num = re.match(r"Slide (\d+)", m.group(1))
        title_html = re.sub(r"\s*\(frames? [\d-]+\)\s*$", "", h2).strip()
        sections.append({
            "id": sid,
            "num": int(num.group(1)) if num else None,
            "kind": "title" if m.group(1).startswith("Title") else "end" if m.group(1).startswith("End") else "slide",
            "title_html": title_html,
            "title": plain(title_html),
            "frames": [a, b],
            "html": body.strip(),
            "text": plain(body),
            "_align_text": strip_own_words(m.group(1) + "\n" + "\n".join(body_md)),
        })
    return {"about_html": about.group(1).strip() if about else "", "sections": sections}


def extract_notes(sections: list) -> list:
    notes = []
    for s in sections:
        pos = 0
        while (k := s["html"].find('<span class="own note">', pos)) != -1:
            depth, j = 0, k
            while j < len(s["html"]):
                if s["html"].startswith("<span", j):
                    depth += 1
                elif s["html"].startswith("</span>", j):
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = s["html"][k + len('<span class="own note">'):j]
            notes.append({"slide": s["id"], "label": s["title"], "html": re.sub(r"^\[Note:\s*|\]$", "", inner)})
            pos = j
    return notes


# ---------- alignment ----------

def tex_to_words(text: str) -> str:
    text = re.sub(r"\\(text|mathbf|mathcal|mathbb|operatorname)\b", " ", text)
    text = re.sub(r"\\([A-Za-z]+)", r" \1 ", text)
    return re.sub(r"[{}_^$|&\\]", " ", text)


def align_slides(sections: list, paras: list, lam: float = ALIGN_LAMBDA) -> list:
    """Start time (s) for each section, monotonic in slide order."""
    idx = [i for i, s in enumerate(sections) if s["kind"] == "slide"]
    docs_s = [tex_to_words((sections[i]["title"] + " ") * 3 + sections[i]["_align_text"]) for i in idx]
    docs_p = [p["text"] + " " + (paras[j + 1]["text"] if j + 1 < len(paras) else "") for j, p in enumerate(paras)]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]{2,}\b")
    x = vec.fit_transform(docs_s + docs_p)
    sim = (x[: len(docs_s)] @ x[len(docs_s):].T).toarray()
    n, m = sim.shape
    rows = np.arange(n)[:, None] / max(n - 1, 1)
    cols = np.arange(m)[None, :] / max(m - 1, 1)
    score = sim - lam * np.abs(rows - cols)

    dp = np.empty((n, m))
    back = np.zeros((n, m), dtype=int)
    dp[0] = score[0]
    for i in range(1, n):
        run_arg = np.zeros(m, dtype=int)
        best = -np.inf
        for j in range(m):
            if dp[i - 1, j] > best:
                best, arg = dp[i - 1, j], j
            run_arg[j] = arg
        dp[i] = score[i] + dp[i - 1][run_arg]
        back[i] = run_arg
    j = int(np.argmax(dp[-1]))
    path = [0] * n
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]

    times = [None] * len(sections)
    for k, i in enumerate(idx):
        times[i] = paras[path[k]]["t"]
    for i, s in enumerate(sections):
        if s["kind"] == "title":
            times[i] = 0
    return times


# ---------- frames, homework, readme ----------

def extract_frames(week_dir: Path, n_frames: int) -> None:
    """Full-size frames go to _raw/frames (gitignored); WEB_WIDTH copies to browser/frames (committed, served)."""
    cache = ROOT / "_raw" / "frames" / week_dir.name
    if not (cache.exists() and len(list(cache.glob("p-*.jpg"))) == n_frames):
        pdf = week_dir / "slides.pdf"
        if not pdf.exists():
            raise RuntimeError(f"{pdf} missing - frames cannot be extracted")
        cache.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(["pdfimages", "-j", str(pdf), str(cache / "p")], capture_output=True, text=True)
        got = sorted(cache.glob("p-*"))
        if res.returncode != 0 or len(got) != n_frames or any(p.suffix != ".jpg" for p in got):
            raise RuntimeError(f"{week_dir.name}: frame extraction gave {len(got)} files (expected {n_frames} .jpg): {res.stderr}")
        log.info(f"{week_dir.name}: extracted {n_frames} full-size frames")

    dest = OUT / "frames" / week_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    made = 0
    for src in sorted(cache.glob("p-*.jpg")):
        out = dest / src.name
        if out.exists():
            with Image.open(out) as im:
                if im.width == WEB_WIDTH:
                    continue
        with Image.open(src) as im:
            size = (WEB_WIDTH, round(im.height * WEB_WIDTH / im.width))
            im.convert("RGB").resize(size, Image.LANCZOS).save(out, "JPEG", quality=78, optimize=True)
        made += 1
    stale = {p.name for p in dest.glob("p-*")} - {p.name for p in cache.glob("p-*.jpg")}
    if stale:
        raise RuntimeError(f"{dest}: files with no source frame: {sorted(stale)[:5]}")
    if made:
        log.info(f"{week_dir.name}: wrote {made} web frames ({WEB_WIDTH}px)")


def homework_pages(names: dict) -> list:
    pages = []
    for key, meta in names.items():
        path = ROOT / "homework" / f"{key}.md"
        folder = key.replace("_Installation_Guide", "")
        html = pandoc(path.read_text(encoding="utf-8"))
        html = re.sub(r'src="(?!https?:|data:)([^"]+)"',
                      lambda m: f'src="https://raw.githubusercontent.com/{HW_REPO}/main/{folder}/{m.group(1)}"', html)
        html = re.sub(r'href="(?!https?:|#|mailto:)([^"]+)"',
                      lambda m: f'href="https://github.com/{HW_REPO}/blob/main/{folder}/{m.group(1)}" target="_blank" rel="noopener"', html)
        pages.append({"key": key, "label": meta["label"], "due": meta["due"], "html": html,
                      "source": f"https://github.com/{HW_REPO}/tree/main/{folder}"})
    return pages


def readme_html() -> str:
    html = pandoc((ROOT / "README.md").read_text(encoding="utf-8"))
    html = re.sub(r'href="week(\d\d)_[^"]*"', r'href="#w\1"', html)
    return re.sub(r'href="homework/([^"]+)\.md"', r'href="#hw/\1"', html)


# ---------- main ----------

def main() -> None:
    t0 = time.time()
    content = json.loads((ROOT / "scripts" / "browser_content.json").read_text(encoding="utf-8"))
    week_dirs = sorted(p for p in ROOT.glob("week*") if p.is_dir())
    weeks = []
    for i, wd in enumerate(week_dirs, 1):
        key = wd.name[4:6]
        info = content["weeks"][key]
        lecture = parse_transcript(wd / "transcript_lecture.md")
        guests = [parse_transcript(p) for p in sorted(wd.glob("transcript_guest_*.md"))]
        slides = parse_slides(wd / "slides.md")
        secs = slides["sections"]
        extract_frames(wd, max(s["frames"][1] for s in secs))

        times = align_slides(secs, lecture["paragraphs"])
        for s, t in zip(secs, times):
            s["t"] = t
            del s["_align_text"]

        if lecture["chapters"]:
            lecture["toc"], lecture["toc_source"] = lecture["chapters"], "chapters from YouTube"
        else:
            lecture["toc"] = [{"t": s["t"], "title": s["title"], "slide": s["id"]}
                              for s in secs if s["kind"] == "slide"]
            lecture["toc_source"] = "estimated from slides (approx.)"
        for g in guests:
            g["toc"], g["toc_source"] = g["chapters"], "chapters from YouTube"
            g["affiliation"] = content["guest_affiliations"].get(g["id"], "")
            if not g["toc"]:
                raise RuntimeError(f"guest talk {g['id']} has no chapters")

        weeks.append({
            "key": key, "dir": wd.name, "date": info["date"], "topic": info["topic"],
            "summary": info["summary"], "key_ideas": info["key_ideas"], "papers": info["papers"],
            "homework": info["homework"], "lecture": lecture, "guests": guests,
            "slides": slides, "notes": extract_notes(secs),
        })
        el = time.time() - t0
        log.info(f"{i}/{len(week_dirs)} {wd.name}: {len(secs)} slide sections, {len(guests)} guest talk(s), "
                 f"{el:.0f}s elapsed, ~{el / i * (len(week_dirs) - i):.0f}s left")

    for key, info in content["weeks"].items():
        if not any(w["key"] == key for w in weeks):
            weeks.append({"key": key, "dir": None, "date": info["date"], "topic": info["topic"],
                          "summary": info["summary"], "key_ideas": [], "papers": [], "homework": [],
                          "lecture": None, "guests": [], "slides": None, "notes": [], "placeholder": True})

    data = {"built": time.strftime("%Y-%m-%d %H:%M"), "weeks": weeks,
            "homework": homework_pages(content["homework"]), "about_html": readme_html()}
    out = OUT / "data.js"
    out.write_text("window.COURSE = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
