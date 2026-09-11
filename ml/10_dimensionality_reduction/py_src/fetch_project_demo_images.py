"""Fetch the Project 3 demo photo set from Wikimedia Commons.

Instructor-side, run once (like fetch_1000g_genotypes.py). Builds the folder the
reference walkthrough treats as "my photos": ~10 web images per category, flat in
``data/demo_photos/`` with the category as the filename prefix (that prefix is the
self-annotation students are told to do by hand -- here it comes for free). The
folder then goes through the same given code the students use:

    python embed_my_photos.py data/demo_photos      ->  data/demo_photos_clip.npz

Categories mix things CLIP should separate (a mountain, a dance, an instrument)
with one deliberately messy pair: khorovats vs shawarma, both close-up grilled
meat, expected to smear together.

Each category is a list of Commons search queries, merged in order. Obvious
non-photos (maps, flags, logos, diagrams) are skipped by filename; the final
visual curation is manual. NOTE: the committed folder IS that curated version
(duplicates, logos, archival b/w and off-topic hits removed, a few Sevanavank
shots added by hand) -- re-running this script refetches an uncurated superset,
so don't re-run it over the committed folder. ``_sources.json`` maps every kept
file back to its Commons title.

Run:  ./ma/Scripts/python.exe ml/10_dimensionality_reduction/py_src/fetch_project_demo_images.py
"""
from __future__ import annotations

import io
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

CHAPTER = Path(__file__).resolve().parents[1]
OUT_DIR = CHAPTER / "data" / "demo_photos"

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia requires a descriptive User-Agent; the default urllib one gets 403s.
UA = "python-math-ml-course dataset builder (educational; github.com/HaykTarkhanyan)"

WIDTH = 800             # Commons renders a scaled file for us -- no full-res downloads
PER_CAT = 12            # over-fetch; manual visual prune trims to 5-10
MIN_PER_CAT = 5         # fewer than this after download -> something is wrong, raise

CATEGORIES: dict[str, list[str]] = {
    "khustup":     ["Khustup"],
    "shawarma":    ["shawarma"],
    "cheese":      ["cheese wheel", "cheese"],
    "potato":      ["potatoes", "potato tubers"],
    "folkdance":   ["Armenian folk dance", "Kochari dance"],
    "duduk":       ["duduk instrument", "duduk"],
    "sevan":       ["Lake Sevan"],
    "khachkar":    ["khachkar"],
    "pomegranate": ["pomegranate fruit", "pomegranate"],
    "cascade":     ["Cascade Yerevan"],
    "khorovats":   ["khorovats", "shashlik skewers"],
}

# filename substrings that mean "not a photo" on Commons
_JUNK = ("map", "flag", "coat_of_arms", "logo", "diagram", "chart", "icon",
         "banner", "stamp", "seal_of", "emblem")


def _api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _search_files(query: str, limit: int) -> list[dict]:
    """Commons file search -> [{title, thumburl, mime, width, height}, ...]."""
    data = _api({
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|mime|size", "iiurlwidth": WIDTH,
    })
    pages = data.get("query", {}).get("pages", {})
    # generator results come keyed by pageid with an "index" for search rank
    hits = sorted(pages.values(), key=lambda p: p.get("index", 0))
    out = []
    for p in hits:
        info = p.get("imageinfo", [{}])[0]
        if not info.get("thumburl"):
            continue
        out.append({"title": p["title"], "thumburl": info["thumburl"],
                    "mime": info.get("mime", ""),
                    "width": info.get("width", 0), "height": info.get("height", 0)})
    return out


def _usable(hit: dict) -> bool:
    if hit["mime"] not in ("image/jpeg", "image/png", "image/webp"):
        return False
    if hit["width"] < 500 or hit["height"] < 400:        # thumbnails / tiny crops
        return False
    name = hit["title"].lower().replace(" ", "_")
    return not any(j in name for j in _JUNK)


def _download_jpeg(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")     # PNG/WebP -> one format
    img.save(dest, "JPEG", quality=88)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for cat, queries in CATEGORIES.items():
        seen, kept = set(), 0
        for q in queries:
            if kept >= PER_CAT:
                break
            for hit in _search_files(q, limit=40):
                if kept >= PER_CAT:
                    break
                if hit["title"] in seen or not _usable(hit):
                    continue
                seen.add(hit["title"])
                dest = OUT_DIR / f"{cat}_{kept:02d}.jpg"
                try:
                    _download_jpeg(hit["thumburl"], dest)
                except Exception as e:          # one bad file: log loudly, keep going
                    logging.warning("SKIP %s (%s): %s", hit["title"], cat, e)
                    continue
                manifest[dest.name] = hit["title"]
                kept += 1
        logging.info("%-12s %2d images", cat, kept)
        if kept < MIN_PER_CAT:
            raise SystemExit(f"only {kept} images for '{cat}' -- fix the queries")

    (OUT_DIR / "_sources.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT_DIR.glob("*.jpg"))
    logging.info("done: %d images, %.1f MB, sources in _sources.json",
                 len(manifest), total / 1e6)


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler("logs/fetch_project_demo_images.log")])
    main()
