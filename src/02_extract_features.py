"""
Step 2 - Extract features for the train and test cells.

Features (and only these - no land cover, roads or capacity, to avoid leakage):
  ghi       Global Solar Atlas GHI (kWh/m2/day)
  wind100   Global Wind Atlas mean wind speed at 100 m (m/s)
  elev      Copernicus DEM GLO-90 elevation (m)
  slope     slope from the same DEM (degrees)
  dist_grid distance from the cell centre to the nearest Gridfinder line (km)

GHI, wind100, elev, slope are sampled at each installation point and averaged per cell,
so a cell describes where the farms actually are, not the whole 5 km square.

Run from the project root:  python src/02_extract_features.py
"""
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

RAW, PROC = "data/raw/", "data/processed/"
LAEA = "EPSG:3035"         # ETRS89 Europe Lambert equal-area: metres, good for distances in Europe
BLOCK_M = 100_000          # 100 km spatial blocks for GroupKFold
DEM_CACHE = PROC + "dem_points.csv"   # so re-runs do not re-fetch the DEM over the network

# Copernicus DEM is read straight from the public AWS bucket (cloud-optimised GeoTIFFs):
# only the few pixels around each point are transferred, not the whole 3 GB of tiles.
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

cells = pd.read_csv(PROC + "cells.csv")
cells = cells[cells.split.isin(["train", "test"])]
pts = pd.read_csv(PROC + "points.csv").merge(cells[["cell_id"]], on="cell_id")
print(f"{len(cells)} cells, {len(pts)} installation points")


def sample(path, df):
    """Value of a raster at each point (nearest pixel)."""
    with rasterio.open(path) as r:
        vals = np.array([v[0] for v in r.sample(zip(df.lon, df.lat))], dtype=float)
        if r.nodata is not None:
            vals[vals == r.nodata] = np.nan
    return vals


pts["ghi"] = sample(RAW + "GSA_GHI_9arcsec.tif", pts)

# Wind: official Global Wind Atlas per-country GeoTIFFs (each is clipped to its country).
# Use the installation's own country file first; points right on a border may fall
# outside their country's mask, so fall back to any other file that has a value.
pts = pts.merge(cells[["cell_id", "country"]], on="cell_id")
wind = {}
for f in sorted(glob.glob(RAW + "gwa_ws100/*_wind-speed_100m.tif")):
    wind[os.path.basename(f)[:3]] = sample(f, pts)
stack = np.vstack(list(wind.values()))
own = np.array([wind[c][i] if c in wind else np.nan for i, c in enumerate(pts.country)])
pts["wind100"] = np.where(np.isnan(own), np.nanmax(stack, axis=0), own)


# ---- DEM: elevation + slope -------------------------------------------------
def tile_url(lat, lon):
    la, lo = int(np.floor(lat)), int(np.floor(lon))
    name = (f"Copernicus_DSM_COG_30_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")
    return f"/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"


def dem_for_tile(args):
    """Elevation and Horn slope for all points that fall in one 1x1 degree tile."""
    url, sub = args
    out = []
    try:
        r = rasterio.open(url)
    except Exception:            # no tile = open sea; those points get NaN and are dropped later
        return [(i, np.nan, np.nan) for i in sub.index]
    with r:
        for i, p in sub.iterrows():
            row, col = r.index(p.lon, p.lat)
            # 3x3 window centred on the point; clamp so it stays inside the tile
            row = min(max(row, 1), r.height - 2)
            col = min(max(col, 1), r.width - 2)
            z = r.read(1, window=Window(col - 1, row - 1, 3, 3)).astype(float)
            # pixel size in metres (DEM is in degrees; longitude spacing shrinks with latitude)
            dy = abs(r.res[1]) * 111_320
            dx = abs(r.res[0]) * 111_320 * np.cos(np.radians(p.lat))
            # Horn (1981) slope - the standard GIS slope formula
            dzdx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * dx)
            dzdy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * dy)
            out.append((i, z[1, 1], np.degrees(np.arctan(np.hypot(dzdx, dzdy)))))
    return out


if os.path.exists(DEM_CACHE):
    dem = pd.read_csv(DEM_CACHE, index_col=0)
else:
    groups = list(pts.groupby(pts.apply(lambda p: tile_url(p.lat, p.lon), axis=1)))
    print(f"Reading DEM from {len(groups)} tiles ...")
    with ThreadPoolExecutor(16) as ex:
        rows = [x for res in ex.map(dem_for_tile, groups) for x in res]
    dem = pd.DataFrame(rows, columns=["idx", "elev", "slope"]).set_index("idx").sort_index()
    dem.to_csv(DEM_CACHE)
pts[["elev", "slope"]] = dem[["elev", "slope"]]

# Average the point values per cell
feat = pts.groupby("cell_id")[["ghi", "wind100", "elev", "slope"]].mean().reset_index()
feat = cells.merge(feat, on="cell_id", how="left")

# ---- Distance to grid (cell centre -> nearest line) ---------------------------
xy = gpd.GeoDataFrame(feat[["cell_id"]], geometry=gpd.points_from_xy(feat.lon, feat.lat),
                      crs="EPSG:4326").to_crs(LAEA)
lon0, lat0, lon1, lat1 = feat.lon.min(), feat.lat.min(), feat.lon.max(), feat.lat.max()
grid = gpd.read_file(RAW + "grid.gpkg", bbox=(lon0 - 2, lat0 - 2, lon1 + 2, lat1 + 2))
print(f"{len(grid)} Gridfinder line segments in the region")
grid = grid[["geometry"]].to_crs(LAEA)
near = gpd.sjoin_nearest(xy, grid, distance_col="d").drop_duplicates("cell_id")
feat["dist_grid"] = feat.cell_id.map(near.set_index("cell_id").d / 1000).values

# Gridfinder has no lines on some islands (Canaries, small Mediterranean islands), so their
# "nearest line" is on another coast, 20-450 km away. That value is wrong, not a real feature.
# We checked: every cell above 20 km is such an island cell, so treat it as missing.
MAX_GRID_KM = 20
no_cov = feat.dist_grid > MAX_GRID_KM
print(f"Cells with no Gridfinder coverage (> {MAX_GRID_KM} km, islands): {int(no_cov.sum())}")
feat.loc[no_cov, "dist_grid"] = np.nan

# 100 km spatial blocks (for spatial CV in Step 3)
feat["block_id"] = ((xy.geometry.x // BLOCK_M).astype(int).astype(str) + "_"
                    + (xy.geometry.y // BLOCK_M).astype(int).astype(str)).values

# ---- Drop missing and save ---------------------------------------------------
cols = ["cell_id", "ghi", "wind100", "elev", "slope", "dist_grid", "label", "split", "block_id"]
missing = feat[cols].isna().any(axis=1)
print("\nCells dropped for missing features:", int(missing.sum()))
print(feat[missing].groupby(["split", "label"]).size().to_string() if missing.any() else "")
print("NaN count per feature:", feat[cols].isna().sum()[lambda s: s > 0].to_dict())
feat = feat[~missing]
feat[cols].to_csv(PROC + "features.csv", index=False)

print(f"\nWrote {PROC}features.csv: {len(feat)} cells")
print(pd.crosstab(feat.split, feat.label, margins=True).to_string())
print("\nFeature means by class (train):")
print(feat[feat.split == "train"].groupby("label")[cols[1:6]].mean().round(2).to_string())
