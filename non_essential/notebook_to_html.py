"""Export an executed notebook to one standalone HTML page for reading in a browser: the JupyterLab
look (nbconvert's "lab" template), images embedded, a table of contents at the top, and every "#" and
"##" section collapsible (<details open>). It never runs the notebook: it shows the saved outputs.

Run (~5 s):
    ./ma/Scripts/python.exe non_essential/notebook_to_html.py llms/01_tokenization/02_armenian_tokens_solution.ipynb
Writes the .html next to the notebook. Fails if the page would still load anything from the internet.
"""
import argparse
import logging
import re
import sys
import time
from pathlib import Path

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^(#{1,2}) ")
CSS = """
nav.toc { border: 1px solid #ccc; border-radius: 6px; padding: 8px 18px; margin: 12px 0 20px; background: #fafafa; }
nav.toc ul { margin: 4px 0; padding-left: 18px; }
nav.toc li.l2 { margin-left: 18px; font-size: 0.95em; }
details.sec { margin: 6px 0; }
details.sec > summary { cursor: pointer; list-style-position: outside; }
details.sec > summary > h1, details.sec > summary > h2 { display: inline-block; margin: 8px 0; }
details.sec.l2 { margin-left: 4px; }
"""

log = logging.getLogger("notebook_to_html")


def split_at_headings(nb):
    """One heading per markdown cell, at its top, so sections can be cut between cells."""
    cells = []
    for c in nb.cells:
        if c.cell_type != "markdown":
            cells.append(c)
            continue
        chunks, cur, fence = [], [], False
        for line in c.source.splitlines():
            if line.lstrip().startswith("```"):
                fence = not fence
            if not fence and HEADING.match(line) and any(x.strip() for x in cur):
                chunks.append(cur)
                cur = []
            cur.append(line)
        chunks.append(cur)
        cells += [nbformat.v4.new_markdown_cell("\n".join(ch).strip("\n")) for ch in chunks if any(x.strip() for x in ch)]
    nb.cells = cells
    return nb


def sectioned(html):
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    cells = main.find_all("div", class_="jp-Cell", recursive=False)
    if not cells:
        raise ValueError("no top-level cells under <main>: the nbconvert template changed")
    root, stack, toc, title_seen = soup.new_tag("div"), [], [], False
    stack.append((0, root))
    for cell in cells:
        cell.extract()
        h = cell.find(["h1", "h2"]) if "jp-MarkdownCell" in cell["class"] else None
        rendered = cell.find(class_="jp-RenderedMarkdown")
        is_top = h is not None and rendered is not None and rendered.find(True) is h
        if is_top and not title_seen and h.name == "h1":       # the notebook's title stays a plain h1
            title_seen = True
            root.append(cell)
            root.append(soup.new_tag("nav", attrs={"class": "toc", "id": "toc"}))
            continue
        if not is_top:
            stack[-1][1].append(cell)
            continue
        level = int(h.name[1])
        while stack[-1][0] >= level:
            stack.pop()
        for a in h.find_all("a", class_="anchor-link"):
            a.decompose()
        sec_id = h.get("id") or f"sec-{len(toc)}"
        det = soup.new_tag("details", attrs={"class": f"sec l{level}", "open": "", "id": sec_id})
        summary = soup.new_tag("summary")
        h.extract()
        summary.append(h)
        det.append(summary)
        if rendered.get_text(strip=True):                      # the rest of the cell, if any
            det.append(cell)
        stack[-1][1].append(det)
        stack.append((level, det))
        toc.append((level, sec_id, h.get_text(strip=True)))
    nav = root.find("nav", id="toc")
    if nav is None:
        raise ValueError("the notebook has no '#' title cell to put the table of contents under")
    nav.append(BeautifulSoup("<b>Contents</b><ul>" + "".join(
        f'<li class="l{lv}"><a href="#{i}">{t}</a></li>' for lv, i, t in toc) + "</ul>", "html.parser"))
    for child in list(root.children):
        main.append(child.extract())
    style = soup.new_tag("style")
    style.string = CSS
    soup.head.append(style)
    return soup


def strip_remote(soup, has_math):
    """Drop the CDN scripts nbconvert adds (require.js always, MathJax when there is no math)."""
    for s in soup.find_all("script", src=True):
        if "require" in s["src"] or ("mathjax" in s["src"].lower() and not has_math):
            s.decompose()
    remote = [t for t in soup.find_all(["script", "link", "img"])
              if (t.get("src") or t.get("href") or "").startswith(("http://", "https://"))]
    if remote:
        raise ValueError(f"the page still loads from the internet: {[str(t)[:80] for t in remote]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    args = ap.parse_args()
    (REPO_ROOT / "logs").mkdir(exist_ok=True)
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(REPO_ROOT / "logs" / "notebook_to_html.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    t0 = time.perf_counter()
    nb = nbformat.read(args.notebook, as_version=4)
    # real LaTeX ($$...$$, \(...\), or $...\command...$), not a dollar amount like "$0.0005"
    has_math = any(re.search(r"\$\$|\\\(|\$[^$\n]*\\[A-Za-z]+[^$\n]*\$", c.source)
                   for c in nb.cells if c.cell_type == "markdown")
    body, _ = HTMLExporter(template_name="lab", embed_images=True).from_notebook_node(split_at_headings(nb))
    soup = sectioned(body)
    strip_remote(soup, has_math)
    out = args.notebook.with_suffix(".html")
    out.write_text(str(soup), encoding="utf-8")
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    sys.exit(main())
