# Satellite data for a practical: no account, no GDAL on student machines

Operational notes from building `ml/09_clustering/py_src/fetch_sevan_scene.py`. Worth keeping
because the obvious paths (Copernicus login, download a whole tile, `pip install geopandas`) are
all worse, and none of this is guessable.

**Sentinel-2 needs no account.** The AWS `sentinel-2-l2a-cogs` bucket is free, unauthenticated
and *not* requester-pays, and it is searchable through the Earth Search STAC API at
`https://earth-search.aws.element84.com/v1`. Copernicus Data Space works too but wants a login.
`pystac_client.Client.open(...).search(collections=["sentinel-2-l2a"], bbox=..., datetime=...,
query={"eo:cloud_cover": {"lt": 5}})` returns items whose assets are plain HTTPS COG URLs.

**Never download a tile.** A full band is 10980x10980 uint16, ~240 MB. `rasterio` reads a
*window* straight over HTTPS:

```python
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"), \
     rasterio.open(asset_href) as src:
    win = rasterio.windows.from_bounds(*bounds, transform=src.transform)
    arr = src.read(1, window=win, out_shape=(1000, 1000), resampling=Resampling.average)
```

Six bands plus SCL for a 1000x1000 crop took ~11 s total. `GDAL_DISABLE_READDIR_ON_OPEN` is the
one that matters — without it GDAL lists the whole prefix on every open.

**Gotchas that cost real time if missed:**

- **The BOA offset is not optional.** Since processing baseline 04.00, L2A digital numbers carry
  `offset = -0.1`, so reflectance is `DN * 1e-4 - 0.1`, and dark water legitimately comes out
  slightly negative. Read `scale`/`offset` from the asset's `raster:bands` rather than assuming.
  Clip to 0 before computing band ratios or NDVI explodes where the denominator crosses zero.
- **Resample categorical rasters with NEAREST.** ESA WorldCover is EPSG:4326, Sentinel-2 is UTM,
  so the labels must be reprojected onto the S2 grid. Anything but nearest neighbour averages
  class *codes* and invents classes that do not exist (30 and 50 averaging to 40 = "cropland").
- **Verify the crop, not the scene.** `eo:cloud_cover` is for the whole 110 km tile. Read the SCL
  band for the actual window and reject on classes 3/8/9/10; also reject if SCL is 0 anywhere,
  which means the crop ran off the edge of the tile.
- **Snap the target grid to a multiple of the pixel size** so the 20 m bands need no resampling
  at all.

**Ship the crop, not the pipeline.** The `ma` venv had no geospatial stack, and requiring one on
every student machine adds a failure mode unrelated to the subject. The fetch script is
instructor-side; the committed `data/sevan_s2_crop.npz` (9.2 MB) loads with plain `np.load`.
See `DECISIONS.md` #16.

**Pick the window by criterion, not by eye.** The script scores five candidate centres on
WorldCover class variety (at least 5 classes over 2 % each, tie-broken on entropy) and refuses to
proceed if none passes. The chosen window was not the one that looked best on a map.
