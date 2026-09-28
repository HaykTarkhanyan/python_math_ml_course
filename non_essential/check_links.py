"""Find broken links on the course website. Manual tool, not part of CI.

Two modes:

  source (default, offline, ~5 s)
      Reads every page listed in _quarto.yml and checks each relative link and each link into this
      repo on GitHub / Colab against the git-tracked files, case-sensitively (the site is built on
      Linux, so `01_Intro.ipynb` != `01_intro.ipynb` there even though Windows does not care).
      Also flags visible placeholder links (`[..]()`, `[..](ToDo)`) and unfilled template targets
      (`NN_topic`, `VIDEO_ID`, `](URL)`).

  live (~3-5 min, network-bound, 8 threads)
      Crawls the deployed site, then checks internal files and #anchors, links into this repo,
      YouTube videos (via the watch page, not oEmbed) and every other external URL.

Run from the repo root with the ma venv:
    ./ma/Scripts/python.exe non_essential/check_links.py            # source mode
    ./ma/Scripts/python.exe non_essential/check_links.py --live     # live mode

Exit code 1 when anything is broken. Full results: logs/check_links_<mode>.json.
"""
import argparse
import json
import logging
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urldefrag, urljoin, urlparse

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/check_links.log", encoding="utf-8")],
)
log = logging.getLogger("check_links")

SITE = "https://hayktarkhanyan.github.io/python_math_ml_course/"
REPO = "HaykTarkhanyan/python_math_ml_course"
OWN = re.compile(r"(?:github\.com/" + REPO + r"/(?:blob|tree|raw|edit)/([^/]+)/"
                 r"|colab\.research\.google\.com/github/" + REPO + r"/blob/([^/]+)/"
                 r"|raw\.githubusercontent\.com/" + REPO + r"/([^/]+)/)([^)\s\"'#?]+)", re.I)
MD_LINK = re.compile(r"\]\(<?([^)>]*?)>?(?:\s+\"[^\"]*\")?\)|(?:href|src)=\\?[\"']([^\"'\\]*)")
PLACEHOLDER = re.compile(r"\[[^\]]*\]\((?:ToDo)?\)")
TEMPLATE_TARGET = re.compile(r"NN_topic|NN_name|VIDEO_ID|\]\(URL\)")
YOUTUBE = re.compile(r"^https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch|shorts|embed|live)|youtu\.be/)", re.I)
# Sites that answer scripts with 401/403/429/999 whatever the URL: not evidence of a dead link.
BOT_BLOCK_CODES = {401, 403, 429, 999}


def git_lines(*args):
    """git output as lines; core.quotepath=off or non-ASCII paths come back octal-escaped."""
    return subprocess.run(["git", "-c", "core.quotepath=off", *args], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout.splitlines()


def tracked_files():
    files = set(git_lines("ls-files")) - set(git_lines("ls-files", "--deleted"))
    folders = {"/".join(f.split("/")[:i]) for f in files for i in range(1, f.count("/") + 1)}
    return files, folders


def site_pages():
    yml = Path("_quarto.yml").read_text(encoding="utf-8")
    return re.findall(r"^\s*-\s*(?:file:\s*)?(\S+\.(?:qmd|ipynb))", yml, re.M)


def own_repo_problem(url, files, folders):
    """None if a link into this repo resolves, else the reason."""
    m = OWN.search(url)
    branch = next(g for g in m.groups()[:3] if g is not None)
    path = unquote(m.group(4)).rstrip("/")
    if branch != "main":
        return f"points at branch '{branch}', not main"
    if path in files or path in folders:
        return None
    lower = {p.lower(): p for p in files | folders}
    if path.lower() in lower:
        return f"case mismatch, repo has '{lower[path.lower()]}'"
    return "path not in repo"


def check_source():
    files, folders = tracked_files()
    broken = []
    pages = site_pages()
    for page in pages:
        raw = Path(page).read_text(encoding="utf-8")
        if page.endswith(".ipynb"):
            raw = "\n".join("".join(c["source"]) for c in json.loads(raw)["cells"] if c["cell_type"] == "markdown")
        raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)   # hidden drafts do not render
        raw = re.sub(r"```.*?```", "", raw, flags=re.S)     # code fences
        raw = re.sub(r"`[^`\n]*`", "", raw)                 # inline code
        for m in PLACEHOLDER.finditer(raw):
            broken.append((page, m.group(0), "placeholder link (goes nowhere)"))
        for m in TEMPLATE_TARGET.finditer(raw):
            broken.append((page, m.group(0), "unfilled template placeholder"))
        base = PurePosixPath(page).parent
        for m in MD_LINK.finditer(raw):
            target = (m.group(1) if m.group(1) is not None else m.group(2)).strip()
            if OWN.search(target):
                problem = own_repo_problem(target, files, folders)
                if problem:
                    broken.append((page, target, problem))
                continue
            if not target or re.match(r"^(?:[a-z]+:|#)", target, re.I):   # http:, mailto:, attachment:, #anchor
                continue
            path = unquote(target.split("#")[0].split("?")[0])
            parts = []
            for part in (base / path).parts:
                if part == "..":
                    if not parts:
                        broken.append((page, target, "climbs above the site root"))
                        break
                    parts.pop()
                elif part != ".":
                    parts.append(part)
            else:
                resolved = "/".join(parts)
                if resolved not in files:
                    lower = {p.lower(): p for p in files}
                    why = (f"case mismatch, repo has '{lower[resolved.lower()]}'" if resolved.lower() in lower
                           else "folder, not a file (Pages has no directory listing)" if resolved in folders
                           else "file not in repo")
                    broken.append((page, target, why))
    return len(pages), broken, []


def check_live():
    import requests
    from bs4 import BeautifulSoup
    from tqdm import tqdm

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                          "(KHTML, like Gecko) Chrome/128.0 Safari/537.36",
                            "Accept-Language": "en-US,en;q=0.9"})
    session.cookies.set("CONSENT", "YES+1", domain=".youtube.com")

    # 1. crawl
    pages, ids, uses = {}, {}, defaultdict(set)
    queue, seen = deque([SITE]), {SITE}
    with tqdm(desc="crawl", unit="page") as bar, ThreadPoolExecutor(8) as ex:
        while queue:
            batch = [queue.popleft() for _ in range(min(8, len(queue)))]
            for url, r in zip(batch, ex.map(lambda u: session.get(u, timeout=30), batch)):
                bar.update(1)
                pages[url] = r.status_code
                if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                    continue
                r.encoding = "utf-8"
                soup = BeautifulSoup(r.text, "html.parser")
                ids[url] = {t["id"] for t in soup.find_all(id=True)}
                for tag, attr in (("a", "href"), ("img", "src"), ("iframe", "src"), ("source", "src"),
                                  ("video", "src"), ("embed", "src"), ("object", "data")):
                    for el in soup.find_all(tag):
                        raw = el.get(attr)
                        if not raw or re.match(r"^(?:javascript|mailto|data|tel):", raw):
                            continue
                        absolute = urljoin(url, raw)
                        uses[absolute].add(url.replace(SITE, "") or "index.html")
                        page_url = urldefrag(absolute)[0]
                        if page_url.startswith(SITE) and page_url.endswith((".html", "/")) and page_url not in seen:
                            seen.add(page_url)
                            queue.append(page_url)
    log.info(f"crawled {len(pages)} pages, {len(uses)} unique links")

    # 2. classify
    files, folders = tracked_files()
    broken, unverified, to_fetch = [], [], []
    for url in uses:
        page_url, frag = urldefrag(url)
        if page_url in pages:
            if pages[page_url] != 200:
                broken.append((url, f"HTTP {pages[page_url]}"))
            elif frag and page_url in ids and unquote(frag) not in ids[page_url]:
                broken.append((url, f"no element with id '{unquote(frag)}'"))
        elif OWN.search(url):
            problem = own_repo_problem(url, files, folders)
            if problem:
                broken.append((url, problem))
        elif urlparse(url).scheme in ("http", "https"):
            to_fetch.append(url)

    def fetch(url):
        try:
            if YOUTUBE.match(url):
                html = session.get(url, timeout=25).text
                m = re.search(r'"playabilityStatus":\{"status":"(\w+)"(?:,"reason":"([^"]*)")?', html)
                if m and m.group(1) != "OK":
                    return url, "broken", f"YouTube {m.group(1)}: {m.group(2) or ''}"
                return url, "ok", ""
            r = session.head(url, timeout=20, allow_redirects=True)
            if r.status_code >= 400 or r.status_code in (405, 501):
                r = session.get(url, timeout=25, allow_redirects=True, stream=True)
                r.close()
            if r.status_code in BOT_BLOCK_CODES:
                return url, "unverified", f"HTTP {r.status_code} (site blocks scripts; check in a browser)"
            if r.status_code >= 400:
                return url, "broken", f"HTTP {r.status_code}"
            asked, got = urlparse(url), urlparse(r.url)
            if got.path in ("", "/") and asked.path not in ("", "/") and asked.netloc != got.netloc:
                return url, "broken", f"redirects to a homepage ({r.url}): the target page is gone"
            return url, "ok", ""
        except requests.RequestException as e:
            return url, "broken", f"{type(e).__name__}: {str(e)[:120]}"

    with ThreadPoolExecutor(8) as ex:
        for url, status, why in tqdm(ex.map(fetch, to_fetch), total=len(to_fetch), desc="links", unit="url"):
            if status == "broken":
                broken.append((url, why))
            elif status == "unverified":
                unverified.append((url, why))
    broken = [(", ".join(sorted(uses[u])[:3]) + (" ..." if len(uses[u]) > 3 else ""), u.replace(SITE, "SITE/"), why)
              for u, why in broken]
    unverified = [(", ".join(sorted(uses[u])[:3]), u, why) for u, why in unverified]
    return len(pages), broken, unverified


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="crawl the deployed site instead of reading the sources")
    mode = "live" if ap.parse_args().live else "source"
    t0 = time.time()
    n_pages, broken, unverified = check_live() if mode == "live" else check_source()
    out = Path(f"logs/check_links_{mode}.json")
    out.write_text(json.dumps({"mode": mode, "pages": n_pages, "broken": broken, "unverified": unverified},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    for where, link, why in unverified:
        log.warning(f"UNVERIFIED  {where}: {link}  ({why})")
    for where, link, why in broken:
        log.error(f"BROKEN  {where}: {link}  ({why})")
    log.info(f"{mode} mode: {n_pages} pages, {len(broken)} broken, {len(unverified)} unverified, "
             f"{time.time() - t0:.0f} s. Details: {out}")
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
