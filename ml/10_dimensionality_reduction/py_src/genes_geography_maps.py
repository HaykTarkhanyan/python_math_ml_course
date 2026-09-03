"""Map-vs-PCA figures for the "genes mirror geography" practical.

Two side-by-side figures, each pairing REAL geography (left) with the PCA of raw
genotype counts (right), so the room can compare the two pictures directly:

  out/gg_world_pca_map.png   -- 26 sampling sites on a world map | worldwide PCA
  out/gg_europe_pca_map.png  -- the European sites               | EUR-only refit

Population codes and descriptions are the official 1000 Genomes ones
(ftp.1000genomes.ebi.ac.uk/vol1/ftp/README_populations.md and
phase3/20131219.superpopulations.tsv). Site coordinates are approximate
(city-level); diaspora samples (ASW, MXL, GIH, STU, ITU, CEU) are plotted where
they were SAMPLED, not where their ancestry lies -- the world panel marks them with an
open symbol for exactly that reason.

Coastlines: Natural Earth 110m (public domain), fetched once into
~/.cache/natural_earth/.

Run:  ./ma/Scripts/python.exe ml/10_dimensionality_reduction/py_src/genes_geography_maps.py
"""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
if __name__ == "__main__":        # script run: headless. Imported by the practical
    matplotlib.use("Agg")         # notebook: leave its inline backend alone.
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SEED = 509
CHAPTER = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
NPZ = CHAPTER / "data" / "genomes_1000g_chr22.npz"
OUT = CHAPTER / "out"

COAST_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
             "master/geojson/ne_110m_coastline.geojson")
COAST_CACHE = Path.home() / ".cache" / "natural_earth" / "ne_110m_coastline.geojson"

SUPER_NAMES = {"AFR": "African", "AMR": "American", "EAS": "East Asian",
               "EUR": "European", "SAS": "South Asian"}
SUPER_COLORS = {"AFR": "tab:red", "EUR": "tab:blue", "EAS": "tab:green",
                "SAS": "tab:purple", "AMR": "tab:orange"}

# (lon, lat, sampled-in-diaspora?) per official population description; city-level.
SITES = {
    "CHB": (116.4, 39.9, False),   # Han Chinese in Beijing, China
    "JPT": (139.7, 35.7, False),   # Japanese in Tokyo, Japan
    "CHS": (112.9, 28.2, False),   # Han Chinese South
    "CDX": (100.8, 22.0, False),   # Chinese Dai in Xishuangbanna, China
    "KHV": (106.7, 10.8, False),   # Kinh in Ho Chi Minh City, Vietnam
    "CEU": (-111.9, 40.8, True),   # Utah residents with N/W European ancestry
    "TSI": (11.3, 43.8, False),    # Toscani in Italia
    "GBR": (-1.5, 53.0, False),    # British in England and Scotland
    "FIN": (24.9, 60.2, False),    # Finnish in Finland
    "IBS": (-3.7, 40.4, False),    # Iberian populations in Spain
    "YRI": (3.9, 7.4, False),      # Yoruba in Ibadan, Nigeria
    "LWK": (34.8, 0.6, False),     # Luhya in Webuye, Kenya
    "GWD": (-16.7, 13.4, False),   # Gambian in Western Division, The Gambia
    "MSL": (-13.2, 8.5, False),    # Mende in Sierra Leone
    "ESN": (6.3, 6.7, False),      # Esan in Nigeria
    "ASW": (-105.0, 35.0, True),   # African Ancestry in Southwest US
    "ACB": (-59.6, 13.1, False),   # African Caribbean in Barbados (sampled at home,
                                   # unlike ASW -- hence filled symbol despite admixture)
    "MXL": (-118.2, 34.1, True),   # Mexican Ancestry in Los Angeles
    "PUR": (-66.1, 18.4, False),   # Puerto Rican in Puerto Rico
    "CLM": (-75.6, 6.2, False),    # Colombian in Medellin, Colombia
    "PEL": (-77.0, -12.0, False),  # Peruvian in Lima, Peru
    "GIH": (-95.4, 29.8, True),    # Gujarati Indian in Houston, TX
    "PJL": (74.3, 31.5, False),    # Punjabi in Lahore, Pakistan
    "BEB": (90.4, 23.8, False),    # Bengali in Bangladesh
    "STU": (-0.8, 51.2, True),     # Sri Lankan Tamil in the UK
    "ITU": (0.9, 51.9, True),      # Indian Telugu in the UK
}

EUR_NAMES = {"CEU": "Utah, N/W-European ancestry", "GBR": "British (England & Scotland)",
             "FIN": "Finnish (Finland)", "IBS": "Iberian (Spain)", "TSI": "Toscani (Italy)"}


def coastlines() -> list[np.ndarray]:
    if not COAST_CACHE.exists():
        COAST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        logging.info("downloading %s", COAST_URL)
        urllib.request.urlretrieve(COAST_URL, COAST_CACHE)
    gj = json.loads(COAST_CACHE.read_text(encoding="utf-8"))
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        parts = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        lines.extend(np.asarray(p) for p in parts)
    return lines


def draw_map(ax, lines, extent):
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], color="0.75", lw=0.6, zorder=1)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.3)          # rough lat/lon aspect at mid latitudes
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.7")


def pca_scores(X, n=4):
    """Patterson-scaled PCA; same recipe as validate_genes_geography.py."""
    X = X.astype(np.float32)
    p = X.mean(axis=0) / 2.0
    keep = (p > 0) & (p < 1)
    if not keep.all():
        logging.info("  dropping %d SNPs monomorphic in this subset", int((~keep).sum()))
        X, p = X[:, keep], p[keep]
    Xc = (X - 2.0 * p) / np.sqrt(p * (1.0 - p))
    pca = PCA(n_components=n, random_state=SEED)
    return pca.fit_transform(Xc), pca.explained_variance_ratio_


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    X, pop, sup = d["genotypes"], d["pop"], d["super_pop"]
    missing = sorted(set(pop) - set(SITES))
    if missing:
        raise ValueError(f"no coordinates for populations: {missing}")
    lines = coastlines()
    OUT.mkdir(exist_ok=True)
    sup_of = {p_: sup[pop == p_][0] for p_ in set(pop)}

    # ---- world: sampling sites | worldwide PCA --------------------------------
    Z, evr = pca_scores(X)
    fig, (axm, axp) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    draw_map(axm, lines, (-135, 155, -40, 72))
    label_off = {"YRI": (-22, 2), "MSL": (-8, -11), "GBR": (-24, 2), "STU": (-26, -9)}
    for code, (lon, lat, diaspora) in SITES.items():
        c = SUPER_COLORS[sup_of[code]]
        axm.scatter(lon, lat, s=55, color="white" if diaspora else c, edgecolors=c,
                    linewidths=1.6, zorder=3)
        axm.annotate(code, (lon, lat), textcoords="offset points",
                     xytext=label_off.get(code, (4, 4)),
                     fontsize=6.5, color=c, fontweight="bold")
    axm.set_title("where the 26 samples come from\n(open symbol = diaspora, plotted at sampling site)",
                  fontsize=10)
    for s in sorted(SUPER_NAMES):
        m = sup == s
        axp.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.65, color=SUPER_COLORS[s],
                    linewidths=0, label=f"{s} — {SUPER_NAMES[s]} ({m.sum()})")
    axp.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); axp.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    axp.set_title("PCA of raw genotype counts (chr22, 16k SNPs)", fontsize=10)
    axp.legend(frameon=False, fontsize=8, loc="upper right")
    axp.spines[["top", "right"]].set_visible(False)
    fig.suptitle("1000 Genomes: geography (left) vs genetics (right)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "gg_world_pca_map.png", dpi=150); plt.close(fig)
    logging.info("wrote %s", OUT / "gg_world_pca_map.png")

    # ---- Europe: sites | EUR-only refit ---------------------------------------
    eur = sup == "EUR"
    Ze, evr_e = pca_scores(X[eur])
    pe = pop[eur]
    cmap = dict(zip(sorted(set(pe)), plt.cm.tab10.colors))
    fig, (axm, axp) = plt.subplots(1, 2, figsize=(12.6, 5.6),
                                   gridspec_kw={"width_ratios": [0.85, 1]})
    draw_map(axm, lines, (-13, 34, 34, 64))
    for code in sorted(set(pe)):
        lon, lat, diaspora = SITES[code]
        if diaspora:                                    # CEU: sampled in Utah
            axm.annotate("CEU — Utah residents with\nN/W-European ancestry\n(no European site to mark)",
                         (0.03, 0.03), xycoords="axes fraction", fontsize=8,
                         color=cmap[code],
                         bbox=dict(boxstyle="round", fc="white", ec=cmap[code]))
            continue
        axm.scatter(lon, lat, s=130, color=cmap[code], edgecolors="black", zorder=3)
        axm.annotate(f"{code} — {EUR_NAMES[code]}", (lon, lat), textcoords="offset points",
                     xytext=(8, 6), fontsize=8.5, color=cmap[code], fontweight="bold")
    axm.set_title("where the European samples live", fontsize=10)
    for i in range(len(Ze)):
        axp.text(Ze[i, 0], Ze[i, 1], pe[i], color=cmap[pe[i]], fontsize=4.5,
                 ha="center", va="center", alpha=0.7)
    for code in sorted(set(pe)):
        med = np.median(Ze[pe == code, :2], axis=0)
        axp.scatter(*med, s=170, color=cmap[code], edgecolors="black", zorder=5)
        axp.annotate(code, med, textcoords="offset points", xytext=(8, 8),
                     fontsize=10, fontweight="bold", color=cmap[code])
    axp.set_xlim(Ze[:, 0].min() * 1.15, Ze[:, 0].max() * 1.15)
    axp.set_ylim(Ze[:, 1].min() * 1.15, Ze[:, 1].max() * 1.15)
    axp.set_xlabel(f"PC1 ({evr_e[0]*100:.1f}%)"); axp.set_ylabel(f"PC2 ({evr_e[1]*100:.1f}%)")
    axp.set_title("Europeans only, PCA refit: north-south survives,\nSpain vs Italy does not (chr22 alone)",
                  fontsize=10)
    axp.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Europe: geography (left) vs genetics (right)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "gg_europe_pca_map.png", dpi=150); plt.close(fig)
    logging.info("wrote %s", OUT / "gg_europe_pca_map.png")


if __name__ == "__main__":
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / "genes_geography_maps.log", encoding="utf-8")],
    )
    main()
