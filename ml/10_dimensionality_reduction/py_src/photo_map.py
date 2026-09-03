"""Interactive hover-the-photo map -- GIVEN code for ch10 Project 3.

You are NOT asked to write this. Do the analysis (embeddings, clustering, the 2-D
projection, a normal matplotlib scatter) yourself; then hand your 2-D points and the
thumbnails to `make_photo_map` and get a self-contained HTML file where hovering a
point shows the photo. Works offline, one file, share it anywhere.

Usage in your notebook (chapter dataset):

    import numpy as np
    from py_src.photo_map import jpegs_from_npz, make_photo_map

    d = np.load("data/imagenette_clip.npz", allow_pickle=True)
    # ... your work: Z2 = 2-D projection (n, 2), labels = your k-means clusters ...
    make_photo_map(Z2, jpegs_from_npz(d), groups=labels,
                   hover_names=[str(d["class_names"][i]) for i in d["labels"]],
                   out_html="out/photo_map.html")

Same call works for your own photos: `embed_my_photos.py` writes an npz with the
same thumb_blob / thumb_offsets layout.

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
    """Unpack the (thumb_blob, thumb_offsets) JPEG store used by the chapter npz
    and by embed_my_photos.py into a plain list of JPEG bytes."""
    blob, off = d["thumb_blob"], d["thumb_offsets"]
    return [blob[off[i]:off[i + 1]].tobytes() for i in range(len(off) - 1)]


def make_photo_map(xy, jpegs, hover_names=None, groups=None, group_names=None,
                   title="Photo map (hover a point to see the photo)",
                   out_html="photo_map.html") -> Path:
    """Write a self-contained interactive photo map.

    xy          : (n, 2) array -- your 2-D projection (PCA / t-SNE / UMAP scores).
    jpegs       : list of n JPEG bytes (see jpegs_from_npz).
    hover_names : optional n strings shown as the hover label.
    groups      : optional n integer labels (e.g. k-means clusters) -> one color
                  + legend entry per group.
    group_names : optional dict or sequence mapping group id -> legend text.
    """
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
    # Demo / smoke test: PCA-2D of the chapter's CLIP embeddings, colored by true class.
    from sklearn.decomposition import PCA

    root = Path(__file__).resolve().parents[3]
    (root / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(root / "logs" / "photo_map.log", encoding="utf-8")],
    )
    chapter = Path(__file__).resolve().parents[1]
    d = np.load(chapter / "data" / "imagenette_clip.npz", allow_pickle=True)
    Z2 = PCA(n_components=2, random_state=509).fit_transform(d["embeddings"].astype(np.float64))
    names = [str(d["class_names"][i]) for i in d["labels"]]
    make_photo_map(Z2, jpegs_from_npz(d), hover_names=names, groups=d["labels"],
                   group_names=list(d["class_names"]),
                   title="PCA of 2000 CLIP embeddings (demo)",
                   out_html=chapter / "out" / "photo_map_demo.html")
