"""Fetch and cache the Lake Sevan Sentinel-2 crop used by the land-cover practical.

Instructor-side, run once. Students never run this - they load the committed
`data/sevan_s2_crop.npz` with plain numpy.

What it does:
  1. Scores several candidate 20 km windows over Lake Sevan on ESA WorldCover class
     variety, and keeps the best one (>= MIN_CLASSES classes each over MIN_SHARE).
  2. Searches Earth Search (AWS Sentinel-2 L2A COGs, no account needed) for a
     cloud-free summer scene covering that window.
  3. Reads six bands into a 20 m grid, checks the crop itself is cloud-free via SCL.
  4. Reprojects the WorldCover labels onto the same grid with NEAREST neighbour.
  5. Writes the npz plus a quick-look PNG.

Run:  ./ma/Scripts/python.exe ml/09_clustering/py_src/fetch_sevan_scene.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform as warp_transform, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

# --- configuration -------------------------------------------------------------

CHAPTER = Path(__file__).resolve().parents[1]
OUT_NPZ = CHAPTER / "data" / "sevan_s2_crop.npz"
OUT_PNG = CHAPTER / "data" / "sevan_s2_quicklook.png"

PIXEL_M = 20  # native resolution of B11/B12; B02/B03/B04/B08 get averaged down to it
SIZE_PX = 1000  # 1000 * 20 m = 20 km square
BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
BAND_CODES = ["B02", "B03", "B04", "B08", "B11", "B12"]

# Candidate window centres (lat, lon) around Lake Sevan, scored on class variety.
CANDIDATES = {
    "sevan_town_peninsula": (40.550, 45.020),
    "east_shore_artanish": (40.420, 45.520),
    "gavar_west_shore": (40.350, 45.130),
    "tsapatagh_forest": (40.330, 45.430),
    "martuni_south": (40.140, 45.300),
}

MIN_CLASSES = 5  # at least this many WorldCover classes ...
MIN_SHARE = 0.02  # ... each covering at least this share of the crop
MAX_SCENE_CLOUD = 5.0  # percent, whole-scene, used to filter the STAC search
MAX_CROP_CLOUD = 0.01  # share of the crop allowed to be cloud/shadow per SCL
SEARCH_WINDOWS = ["2025-06-15/2025-09-20", "2024-06-15/2024-09-20", "2023-06-15/2023-09-20"]

STAC_URL = "https://earth-search.aws.element84.com/v1"
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
SCL_CLOUDY = (3, 8, 9, 10)  # shadow, cloud medium, cloud high, cirrus

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
)

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("logs") / "fetch_sevan_scene.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# --- grid ----------------------------------------------------------------------


def utm_crs_for(lon: float, lat: float) -> CRS:
    """EPSG code of the UTM zone holding this point (northern hemisphere only here)."""
    if lat < 0:
        raise ValueError(f"southern hemisphere not handled, got lat={lat}")
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg(32600 + zone)


def target_grid(lat: float, lon: float):
    """A SIZE_PX square grid at PIXEL_M, centred on (lat, lon), snapped to the S2 grid."""
    crs = utm_crs_for(lon, lat)
    (cx,), (cy,) = warp_transform(CRS.from_epsg(4326), crs, [lon], [lat])
    half = SIZE_PX * PIXEL_M / 2
    # Snap the top-left corner to a multiple of PIXEL_M so the 20 m bands need no resampling.
    x0 = np.floor((cx - half) / PIXEL_M) * PIXEL_M
    y1 = np.ceil((cy + half) / PIXEL_M) * PIXEL_M
    transform = from_origin(x0, y1, PIXEL_M, PIXEL_M)
    bounds = (x0, y1 - SIZE_PX * PIXEL_M, x0 + SIZE_PX * PIXEL_M, y1)
    return crs, transform, bounds


# --- ESA WorldCover ------------------------------------------------------------


def worldcover_tile_name(lat: float, lon: float) -> str:
    """WorldCover tiles are 3x3 degrees, named by their south-west corner."""
    tlat = int(np.floor(lat / 3) * 3)
    tlon = int(np.floor(lon / 3) * 3)
    ns = "N" if tlat >= 0 else "S"
    ew = "E" if tlon >= 0 else "W"
    return f"{ns}{abs(tlat):02d}{ew}{abs(tlon):03d}"


def read_worldcover(crs: CRS, transform, bounds) -> np.ndarray:
    """WorldCover labels resampled onto the target grid. NEAREST - the values are classes."""
    west, south, east, north = transform_bounds(crs, CRS.from_epsg(4326), *bounds)
    url = WORLDCOVER_URL.format(tile=worldcover_tile_name(south, west))
    dst = np.zeros((SIZE_PX, SIZE_PX), dtype=np.uint8)
    with rasterio.Env(**GDAL_ENV), rasterio.open(url) as src:
        pad = 0.01  # degrees of slack so reprojection has edge pixels to work with
        win = window_from_bounds(
            west - pad, south - pad, east + pad, north + pad, transform=src.transform
        )
        src_arr = src.read(1, window=win)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.window_transform(win),
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return dst


def class_shares(labels: np.ndarray) -> dict[int, float]:
    codes, counts = np.unique(labels, return_counts=True)
    return {int(c): float(n) / labels.size for c, n in zip(codes, counts) if c != 0}


def score_window(shares: dict[int, float]) -> tuple[int, float]:
    """(classes above MIN_SHARE, entropy) - more classes first, then more balanced."""
    kept = np.array([v for v in shares.values() if v >= MIN_SHARE])
    if kept.size == 0:
        return 0, 0.0
    p = kept / kept.sum()
    return int(kept.size), float(-(p * np.log(p)).sum())


def pick_window():
    log.info("Scoring %d candidate windows on WorldCover class variety", len(CANDIDATES))
    scored = []
    for name, (lat, lon) in CANDIDATES.items():
        crs, transform, bounds = target_grid(lat, lon)
        labels = read_worldcover(crs, transform, bounds)
        shares = class_shares(labels)
        n_classes, entropy = score_window(shares)
        pretty = ", ".join(
            f"{WORLDCOVER_CLASSES.get(c, c)} {v:.1%}"
            for c, v in sorted(shares.items(), key=lambda kv: -kv[1])
            if v >= 0.01
        )
        log.info("  %-22s %d classes >%.0f%%, entropy %.2f | %s",
                 name, n_classes, MIN_SHARE * 100, entropy, pretty)
        scored.append((n_classes, entropy, name, lat, lon, crs, transform, bounds, labels))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = scored[0]
    if best[0] < MIN_CLASSES:
        raise RuntimeError(
            f"no candidate window reaches {MIN_CLASSES} classes over {MIN_SHARE:.0%}; "
            f"best was {best[2]} with {best[0]}. Move the window, not the criterion."
        )
    log.info("Chosen window: %s (%d classes, entropy %.2f)", best[2], best[0], best[1])
    return best


# --- Sentinel-2 ----------------------------------------------------------------


def search_scenes(bounds, crs) -> list:
    west, south, east, north = transform_bounds(crs, CRS.from_epsg(4326), *bounds)
    client = Client.open(STAC_URL)
    items = []
    for window in SEARCH_WINDOWS:
        found = list(
            client.search(
                collections=["sentinel-2-l2a"],
                bbox=[west, south, east, north],
                datetime=window,
                query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD}},
                max_items=40,
            ).items()
        )
        log.info("  %s -> %d scenes under %.0f%% cloud", window, len(found), MAX_SCENE_CLOUD)
        items.extend(found)
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    return items


def read_band(url: str, crs: CRS, transform, bounds, resampling: Resampling) -> np.ndarray:
    with rasterio.Env(**GDAL_ENV), rasterio.open(url) as src:
        if src.crs != crs:
            raise RuntimeError(f"asset CRS {src.crs} != target {crs}")
        win = window_from_bounds(*bounds, transform=src.transform)
        return src.read(
            1, window=win, out_shape=(SIZE_PX, SIZE_PX), resampling=resampling, boundless=True,
            fill_value=0,
        )


def fetch_scene(item, crs, transform, bounds):
    """Six bands + SCL for one STAC item, or None if the crop is cloudy or clipped."""
    scl = read_band(item.assets["scl"].href, crs, transform, bounds, Resampling.nearest)
    cloudy = np.isin(scl, SCL_CLOUDY).mean()
    nodata = (scl == 0).mean()
    log.info("  %s  scene cloud %.1f%% | crop cloud %.2f%% | crop nodata %.2f%%",
             item.id, item.properties.get("eo:cloud_cover", float("nan")),
             cloudy * 100, nodata * 100)
    if nodata > 0.001:
        log.info("    rejected: crop falls outside the tile")
        return None
    if cloudy > MAX_CROP_CLOUD:
        log.info("    rejected: crop too cloudy")
        return None

    cube = np.zeros((SIZE_PX, SIZE_PX, len(BANDS)), dtype=np.uint16)
    for i, name in enumerate(BANDS):
        asset = item.assets[name]
        native = 10 if name in ("blue", "green", "red", "nir") else 20
        resampling = Resampling.average if native < PIXEL_M else Resampling.nearest
        cube[:, :, i] = read_band(asset.href, crs, transform, bounds, resampling)
        log.info("    %s (%s, %d m) ok", name, BAND_CODES[i], native)
    return cube, scl


def band_scale_offset(item) -> tuple[float, float]:
    """Reflectance = DN * scale + offset. Post-baseline-04.00 scenes carry offset -0.1."""
    raster = item.assets["blue"].extra_fields.get("raster:bands")
    if raster:
        band = raster[0]
        return float(band.get("scale", 1e-4)), float(band.get("offset", 0.0))
    log.warning("item %s carries no raster:bands metadata; assuming scale 1e-4, offset 0", item.id)
    return 1e-4, 0.0


# --- quicklook -----------------------------------------------------------------


def write_quicklook(cube: np.ndarray, labels: np.ndarray, scale: float, offset: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    refl = cube.astype(np.float32) * scale + offset
    def stretch(a):
        lo, hi = np.percentile(a, [2, 98])
        return np.clip((a - lo) / (hi - lo), 0, 1)

    true_colour = np.dstack([stretch(refl[:, :, 2]), stretch(refl[:, :, 1]), stretch(refl[:, :, 0])])
    false_colour = np.dstack([stretch(refl[:, :, 3]), stretch(refl[:, :, 2]), stretch(refl[:, :, 1])])

    codes = sorted(WORLDCOVER_CLASSES)
    lut = np.zeros(max(codes) + 2, dtype=np.uint8)
    for i, c in enumerate(codes):
        lut[c] = i
    cmap = ListedColormap(plt.get_cmap("tab20")(np.linspace(0, 1, len(codes))))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, img, title in zip(
        axes, [true_colour, false_colour, None], ["True colour", "False colour (NIR)", "ESA WorldCover"]
    ):
        if img is not None:
            ax.imshow(img)
        else:
            ax.imshow(lut[labels], cmap=cmap, norm=BoundaryNorm(np.arange(len(codes) + 1) - 0.5, len(codes)))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    log.info("Quicklook written to %s", OUT_PNG)


# --- main ----------------------------------------------------------------------


def main() -> None:
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    n_classes, entropy, name, lat, lon, crs, transform, bounds, labels = pick_window()

    log.info("Searching Earth Search for a cloud-free summer scene")
    items = search_scenes(bounds, crs)
    if not items:
        raise RuntimeError("no Sentinel-2 scenes matched the search")
    log.info("%d candidate scenes, trying them cleanest first", len(items))

    for item in items:
        result = fetch_scene(item, crs, transform, bounds)
        if result is not None:
            cube, scl = result
            break
    else:
        raise RuntimeError("every candidate scene was cloudy or clipped over the crop")

    scale, offset = band_scale_offset(item)
    shares = class_shares(labels)
    log.info("Scene %s (%s), reflectance = DN * %g + %g",
             item.id, item.properties["datetime"][:10], scale, offset)

    np.savez_compressed(
        OUT_NPZ,
        bands=cube,
        band_codes=np.array(BAND_CODES),
        band_names=np.array(BANDS),
        worldcover=labels,
        worldcover_codes=np.array(sorted(WORLDCOVER_CLASSES)),
        worldcover_names=np.array([WORLDCOVER_CLASSES[c] for c in sorted(WORLDCOVER_CLASSES)]),
        reflectance_scale=np.float32(scale),
        reflectance_offset=np.float32(offset),
        crs=str(crs),
        transform=np.array(transform.to_gdal()),
        bounds=np.array(bounds),
        pixel_m=np.int16(PIXEL_M),
        centre_latlon=np.array([lat, lon]),
        window_name=name,
        scene_id=item.id,
        scene_datetime=item.properties["datetime"],
        scene_cloud_cover=np.float32(item.properties.get("eo:cloud_cover", np.nan)),
    )
    size_mb = OUT_NPZ.stat().st_size / 1e6
    log.info("Wrote %s (%.1f MB)", OUT_NPZ, size_mb)
    if size_mb > 12:
        log.warning("npz is %.1f MB - over the ~10 MB budget, consider a smaller crop", size_mb)

    log.info("Class shares in the crop:")
    for code, share in sorted(shares.items(), key=lambda kv: -kv[1]):
        log.info("  %-26s %5.1f%%", WORLDCOVER_CLASSES.get(code, code), share * 100)

    write_quicklook(cube, labels, scale, offset)


if __name__ == "__main__":
    main()
