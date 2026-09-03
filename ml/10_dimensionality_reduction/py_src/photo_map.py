"""Interactive hover-the-photo map -- GIVEN code for ch10 Project 3.

You are NOT asked to write this. Do the analysis (embeddings, clustering, the 2-D
projection, a normal matplotlib scatter) yourself; then hand your 2-D points and the
thumbnails to `make_photo_map` and get a self-contained HTML file where hovering a
point shows the photo. Works offline, one file, share it anywhere.

Inputs at a glance:

    xy     -- numpy array of shape (n, 2): YOUR 2-D coordinates, one row per photo,
              in the same order as the photos (the output of PCA / t-SNE / UMAP).
    jpegs  -- list of n JPEG byte-strings: the photo thumbnails, same order.
              You do not build this by hand -- unpack it from the npz that
              embed_my_photos.py wrote, with jpegs_from_npz (see below).

Usage in your notebook (own photos, after running embed_my_photos.py):

    import numpy as np
    from py_src.photo_map import jpegs_from_npz, make_photo_map

    d = np.load("my_photos_clip.npz")
    # ... your work: Z2 = 2-D projection of d["embeddings"], labels = your k-means ...
    make_photo_map(Z2, jpegs_from_npz(d), groups=labels,
                   hover_names=d["filenames"], out_html="photo_map.html")

The same call works on the chapter's imagenette_clip.npz -- both files use the
same thumb_blob / thumb_offsets thumbnail layout that jpegs_from_npz unpacks.

The hover mechanism (JPEG thumbnails as base64 data-URIs in `customdata`, a small
JS overlay that follows the cursor) is lifted from the chapter's solution notebook,
where it was browser-verified.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

_HOVER_JS = """
<div id="thumbbox" style="position:fixed;display:none;pointer-events:none;z-index:9999;
     border:2px solid #333;background:#fff;padding:3px;border-radius:4px;
     box-shadow:0 2px 10px rgba(0,0,0,.35)">
  <img id="thumbimg" width="112" height="112" style="display:block">
</div>
<script>
(function () {
  var gd = document.getElementById('photomap');
  var box = document.getElementById('thumbbox'), im = document.getElementById('thumbimg');
  gd.on('plotly_hover', function (ev) {
    var p = ev.points[0];
    if (!p.customdata) { return; }
    im.src = p.customdata[0];
    box.style.display = 'block';
    var x = ev.event.clientX + 18, yy = ev.event.clientY + 18;
    if (x + 130 > window.innerWidth)  { x = ev.event.clientX - 140; }
    if (yy + 130 > window.innerHeight) { yy = ev.event.clientY - 140; }
    box.style.left = x + 'px';
    box.style.top = yy + 'px';
  });
  gd.on('plotly_unhover', function () { box.style.display = 'none'; });
})();
</script>
"""


def jpegs_from_npz(d) -> list[bytes]:
    """Take the object returned by np.load(...) -- for a file written by
    embed_my_photos.py or the chapter's imagenette_clip.npz -- and unpack its
    packed thumbnails (thumb_blob + thumb_offsets) into a plain list of JPEG
    bytes, one entry per photo, in photo order. This list is what make_photo_map
    expects as its `jpegs` argument."""
    blob, off = d["thumb_blob"], d["thumb_offsets"]
    return [blob[off[i]:off[i + 1]].tobytes() for i in range(len(off) - 1)]


def make_photo_map(xy, jpegs, hover_names=None, groups=None, group_names=None,
                   title="Photo map (hover a point to see the photo)",
                   out_html="photo_map.html") -> Path:
    """Write a self-contained interactive photo map (hover a point -> see the photo).

    xy          : (n, 2) array -- your 2-D projection (PCA / t-SNE / UMAP scores),
                  one row per photo, same order as the photos.
    jpegs       : list of n JPEG byte-strings, same order -- NOT the npz itself:
                  get it with jpegs_from_npz(np.load("my_photos_clip.npz")).
    hover_names : optional n strings shown as the hover label (e.g. filenames).
    groups      : optional n integer labels (e.g. k-means clusters) -> one color
                  + legend entry per group.
    group_names : optional dict or sequence mapping group id -> legend text.
    """
    if hasattr(jpegs, "files") or isinstance(jpegs, dict):
        raise TypeError("jpegs must be a list of JPEG bytes, not the npz itself - "
                        "unpack it first: jpegs_from_npz(d)")
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (n, 2), got {xy.shape}")
    n = len(xy)
    if len(jpegs) != n:
        raise ValueError(f"{n} points but {len(jpegs)} thumbnails")
    if hover_names is None:
        hover_names = [""] * n
    if len(hover_names) != n:
        raise ValueError(f"{n} points but {len(hover_names)} hover_names")
    groups = np.zeros(n, dtype=int) if groups is None else np.asarray(groups)
    if len(groups) != n:
        raise ValueError(f"{n} points but {len(groups)} group labels")

    uris = ["data:image/jpeg;base64," + base64.b64encode(j).decode("ascii") for j in jpegs]

    fig = go.Figure()
    group_ids = np.unique(groups)
    for g in group_ids:
        m = np.where(groups == g)[0]
        if group_names is None:
            legend = f"cluster {g} ({len(m)})"
        elif isinstance(group_names, dict):
            legend = f"{group_names[g]} ({len(m)})"
        else:
            legend = f"{group_names[int(g)]} ({len(m)})"
        fig.add_trace(go.Scattergl(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=legend,
            marker=dict(size=7, opacity=0.85, line=dict(width=0)),
            customdata=[[uris[i], str(hover_names[i])] for i in m],
            hovertemplate="<b>%{customdata[1]}</b><extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white", width=1000, height=760,
        showlegend=len(group_ids) > 1,
    )

    html = fig.to_html(div_id="photomap", include_plotlyjs="inline", full_html=True)
    html = html.replace("</body>", _HOVER_JS + "</body>")
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logging.info("wrote %s (%.1f MB, self-contained)", out, out.stat().st_size / 1e6)
    return out


if __name__ == "__main__":
    # Quick self-test on YOUR npz, before you write any notebook code:
    #     python photo_map.py my_photos_clip.npz [out.html]
    # builds a PCA-2D map of the embeddings so you can check the pipeline works.
    # (In the project itself you make the 2-D projection and the clusters yourself.)
    import sys
    from sklearn.decomposition import PCA

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python photo_map.py my_photos_clip.npz [out.html]")
    d = np.load(sys.argv[1])
    Z2 = PCA(n_components=2, random_state=509).fit_transform(d["embeddings"].astype(np.float64))
    names = d["filenames"] if "filenames" in d.files else None
    out = sys.argv[2] if len(sys.argv) == 3 else "photo_map_quicklook.html"
    make_photo_map(Z2, jpegs_from_npz(d), hover_names=names,
                   title=f"PCA quick look: {len(Z2)} photos", out_html=out)
