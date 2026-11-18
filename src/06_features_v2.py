"""
Step 6 (improvement round) - extended features for every cell.

Adds features that describe each site's *local context*, so the model can learn "windier than its
surroundings" instead of only absolute wind speed, plus two access/demand features:

  wind_rel, wind_pct   site wind speed minus the mean within 30 km; share of the 30 km area that is calmer
  ghi_rel,  ghi_pct    same for sunshine
  elev_rel, rugged     site elevation minus the mean within 2.5 km (hilltop/valley); std of elevation (ruggedness)
  dist_town            km to the nearest town or city (GHS settlement classes >= 21), a proxy for demand
  dist_water           km to the nearest sea or large lake (GHS class 10), a proxy for coastal sites

None of these is changed by building a farm (no leakage). Cells covered: Europe (train/test),
USA, China, India, and the locked final-test countries (JPN, TUR, CHL, AUS, ZAF). For the locked
countries this script also builds the 5 base features, exactly as in Step 2, but never scores them.

Run from the project root:  python src/06_features_v2.py
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer

RAW, PROC = "data/raw/", "data/processed/"
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
BASE = ["ghi", "wind100", "elev", "slope", "dist_grid"]
LOCK = json.load(open(PROC + "final_test_lock.json"))["countries"]
R_LOCAL_KM = 30          # neighbourhood for wind / sunshine context
R_TERRAIN_KM = 2.5       # neighbourhood for terrain context
FACTOR = 8               # read rasters at 1/8 resolution (~2 km) for the 30 km context

cells = pd.read_csv(PROC + "cells.csv")
pts_all = pd.read_csv(PROC + "points.csv")


# ---------------------------------------------------------------- base features
def sample(path, lon, lat):
    with rasterio.open(path) as r:
        v = np.array([x[0] for x in r.sample(zip(lon, lat))], dtype=float)
        if r.nodata is not None:
            v[v == r.nodata] = np.nan
    return v


def tile_url(lat, lon):
    la, lo = int(np.floor(lat)), int(np.floor(lon))
    name = (f"Copernicus_DSM_COG_30_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")
    return f"/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"


def dem_points(sub):
    """Elevation + Horn slope at points of one DEM tile (same method as Step 2)."""
    url, sub = sub
    try:
        r = rasterio.open(url)
    except Exception:
        return [(i, np.nan, np.nan) for i in sub.index]
    out = []
    with r:
        for i, p in sub.iterrows():
            row, col = r.index(p.lon, p.lat)
            row, col = min(max(row, 1), r.height - 2), min(max(col, 1), r.width - 2)
            z = r.read(1, window=Window(col - 1, row - 1, 3, 3)).astype(float)
            dy, dx = abs(r.res[1]) * 111_320, abs(r.res[0]) * 111_320 * np.cos(np.radians(p.lat))
            gx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * dx)
            gy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * dy)
            out.append((i, z[1, 1], np.degrees(np.arctan(np.hypot(gx, gy)))))
    return out


def base_features_locked():
    """Steps 1-2 base features for the locked countries (no scoring here)."""
    cache = PROC + "features_locked.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)
    c = cells[cells.country.isin(LOCK)].copy()
    p = pts_all.merge(c[["cell_id", "country"]], on="cell_id").reset_index(drop=True)
    p["ghi"] = sample(RAW + "GSA_GHI_9arcsec.tif", p.lon, p.lat)
    p["wind100"] = np.nan
    for iso in LOCK:
        m = p.country == iso
        p.loc[m, "wind100"] = sample(f"{RAW}gwa_ws100_other/{iso}_wind-speed_100m.tif", p.lon[m], p.lat[m])
    groups = list(p.groupby(p.apply(lambda q: tile_url(q.lat, q.lon), axis=1)))
    with ThreadPoolExecutor(16) as ex:
        rows = [x for res in ex.map(dem_points, groups) for x in res]
    dem = pd.DataFrame(rows, columns=["i", "elev", "slope"]).set_index("i")
    p[["elev", "slope"]] = dem.loc[p.index, ["elev", "slope"]].values
    f = c.merge(p.groupby("cell_id")[["ghi", "wind100", "elev", "slope"]].mean().reset_index(), on="cell_id")
    f["dist_grid"] = np.nan
    for iso in LOCK:
        m = f.country == iso
        g = f[m]
        crs = f"+proj=laea +lat_0={g.lat.median():.1f} +lon_0={g.lon.median():.1f} +units=m"
        xy = gpd.GeoDataFrame(g[["cell_id"]], geometry=gpd.points_from_xy(g.lon, g.lat), crs="EPSG:4326").to_crs(crs)
        grid = gpd.read_file(RAW + "grid.gpkg", bbox=(g.lon.min() - 1, g.lat.min() - 1, g.lon.max() + 1, g.lat.max() + 1))
        near = gpd.sjoin_nearest(xy, grid[["geometry"]].to_crs(crs), distance_col="d").drop_duplicates("cell_id")
        f.loc[m, "dist_grid"] = g.cell_id.map(near.set_index("cell_id").d / 1000).values
    f.loc[f.dist_grid > 20, "dist_grid"] = np.nan          # same no-coverage rule as Step 2
    f["split"] = "final_test"
    f.to_csv(cache, index=False)
    return f


# ------------------------------------------------------- all cells with base features
eu = pd.read_csv(PROC + "features.csv").merge(cells[["cell_id", "lon", "lat", "country"]], on="cell_id")
oth = pd.read_csv(PROC + "features_other.csv")
oth["split"] = "other"
lk = base_features_locked()
cols = ["cell_id", "lon", "lat", "country", "label", "split"] + BASE
df = pd.concat([eu[cols], oth[cols], lk[cols]], ignore_index=True)
print(f"{len(df)} cells: " + ", ".join(f"{k} {v}" for k, v in df.split.value_counts().items()))
for c in ["wind_local", "wind_rel", "wind_pct", "ghi_local", "ghi_rel", "ghi_pct", "dist_town", "dist_water"]:
    df[c] = np.nan


# ---------------------------------------------------------------- local context helpers
def coarse(path, bbox=None):
    """Read a raster (or a lon/lat box of it) at 1/FACTOR resolution, averaged; returns array + transform."""
    with rasterio.open(path) as r:
        win = from_bounds(*bbox, r.transform).round_offsets().round_lengths() if bbox else Window(0, 0, r.width, r.height)
        win = win.intersection(Window(0, 0, r.width, r.height))
        h, w = max(1, int(win.height // FACTOR)), max(1, int(win.width // FACTOR))
        a = r.read(1, window=win, out_shape=(h, w), resampling=Resampling.average, masked=True).astype(float).filled(np.nan)
        t = r.window_transform(win) * rasterio.Affine.scale(win.width / w, win.height / h)
    return a, t


def context(a, t, lon, lat, site, radius_km):
    """Mean of the neighbourhood, site minus mean, and share of the neighbourhood below the site value."""
    res_x, res_y = t.a, -t.e
    out = np.full((len(lon), 3), np.nan)
    for k, (x, y, s) in enumerate(zip(lon, lat, site)):
        col, row = ~t * (x, y)
        col, row = int(col), int(row)
        ry = int(np.ceil(radius_km / (res_y * 111.32)))
        rx = int(np.ceil(radius_km / (res_x * 111.32 * max(np.cos(np.radians(y)), 0.1))))
        r0, r1, c0, c1 = max(row - ry, 0), min(row + ry + 1, a.shape[0]), max(col - rx, 0), min(col + rx + 1, a.shape[1])
        if r0 >= r1 or c0 >= c1 or np.isnan(s):
            continue
        win = a[r0:r1, c0:c1]
        yy, xx = np.mgrid[r0:r1, c0:c1]
        d = np.hypot((yy - row) * res_y * 111.32, (xx - col) * res_x * 111.32 * np.cos(np.radians(y)))
        v = win[(d <= radius_km) & ~np.isnan(win)]
        if len(v) >= 5:
            out[k] = (v.mean(), s - v.mean(), (v < s).mean())
    return out


# Region boxes for the global sunshine raster (keeps memory small)
def bbox_of(g, pad=1.0):
    return (g.lon.min() - pad, g.lat.min() - pad, g.lon.max() + pad, g.lat.max() + pad)


groups = {"Europe": df[~df.country.isin(["USA", "CHN", "IND", *LOCK])]}
for iso in ["USA", "CHN", "IND", *LOCK]:
    groups[iso] = df[df.country == iso]

# -------- wind context: from each country's own Global Wind Atlas file
print("wind context ...")
for iso in df.country.unique():
    m = df.country == iso
    folder = "gwa_ws100" if os.path.exists(f"{RAW}gwa_ws100/{iso}_wind-speed_100m.tif") else "gwa_ws100_other"
    path = f"{RAW}{folder}/{iso}_wind-speed_100m.tif"
    if not os.path.exists(path):
        continue
    a, t = coarse(path)
    df.loc[m, ["wind_local", "wind_rel", "wind_pct"]] = context(a, t, df.lon[m], df.lat[m], df.wind100[m], R_LOCAL_KM)

# -------- sunshine context: global Global Solar Atlas file, one box per region
print("sunshine context ...")
for name, g in groups.items():
    for part in np.array_split(g.index, max(1, len(g) // 4000)):
        sub = df.loc[part]
        a, t = coarse(RAW + "GSA_GHI_9arcsec.tif", bbox_of(sub))
        df.loc[part, ["ghi_local", "ghi_rel", "ghi_pct"]] = context(a, t, sub.lon, sub.lat, sub.ghi, R_LOCAL_KM)


# -------- terrain context: Copernicus DEM window of +/- 2.5 km around each cell centre
def terrain(args):
    url, sub = args
    try:
        r = rasterio.open(url)
    except Exception:
        return [(i, np.nan, np.nan) for i in sub.index]
    out = []
    with r:
        for i, p in sub.iterrows():
            row, col = r.index(p.lon, p.lat)
            hy = int(np.ceil(R_TERRAIN_KM * 1000 / (abs(r.res[1]) * 111_320)))
            hx = int(np.ceil(R_TERRAIN_KM * 1000 / (abs(r.res[0]) * 111_320 * np.cos(np.radians(p.lat)))))
            w = Window(max(col - hx, 0), max(row - hy, 0), 2 * hx + 1, 2 * hy + 1).intersection(Window(0, 0, r.width, r.height))
            z = None
            for _ in range(3):                      # online reads can fail briefly; retry before giving up
                try:
                    z = r.read(1, window=w).astype(float)
                    break
                except Exception:
                    continue
            if z is None:
                out.append((i, np.nan, np.nan))
                continue
            z = z[z > -100]
            out.append((i, z.mean() if z.size else np.nan, z.std() if z.size else np.nan))
    return out


cache = PROC + "terrain_context.csv"
tc = pd.read_csv(cache).set_index("cell_id") if os.path.exists(cache) else pd.DataFrame(columns=["elev_mean5", "rugged"])
todo = df[~df.cell_id.isin(tc.index)]
if len(todo):
    print(f"terrain context for {len(todo)} cells ...")
    grp = list(todo.groupby(todo.apply(lambda q: tile_url(q.lat, q.lon), axis=1)))
    for start in range(0, len(grp), 100):            # save after every 100 tiles so progress is never lost
        with ThreadPoolExecutor(16) as ex:
            rows = [x for res in ex.map(terrain, grp[start:start + 100]) for x in res]
        new = pd.DataFrame(rows, columns=["i", "elev_mean5", "rugged"]).set_index("i")
        new.index = df.loc[new.index, "cell_id"].values
        tc = pd.concat([tc, new])
        tc.index.name = "cell_id"
        tc.to_csv(cache)
        print(f"  {min(start + 100, len(grp))}/{len(grp)} tiles")
df["elev_rel"] = df.elev - df.cell_id.map(tc.elev_mean5)
df["rugged"] = df.cell_id.map(tc.rugged)

# -------- distance to town and to water: GHS settlement model (1 km, Mollweide)
print("town / water distance ...")
to_moll = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
with rasterio.open(RAW + "ghs_2015.tif") as r:
    for name, g in groups.items():
        x, y = to_moll.transform(g.lon.values, g.lat.values)
        pad = 300_000                                   # 300 km margin so the nearest town/water is inside
        win = from_bounds(x.min() - pad, y.min() - pad, x.max() + pad, y.max() + pad, r.transform)
        win = win.round_offsets().round_lengths().intersection(Window(0, 0, r.width, r.height))
        a = r.read(1, window=win)
        t = r.window_transform(win)
        rows, cols_ = rasterio.transform.rowcol(t, x, y)
        rows, cols_ = np.clip(rows, 0, a.shape[0] - 1), np.clip(cols_, 0, a.shape[1] - 1)
        for feat, mask in [("dist_town", a >= 21), ("dist_water", a == 10)]:
            dist = distance_transform_edt(~mask) * abs(t.a) / 1000      # pixels -> km (1 km pixels)
            df.loc[g.index, feat] = np.minimum(dist[rows, cols_], 300)

NEW = ["wind_rel", "wind_pct", "ghi_rel", "ghi_pct", "elev_rel", "rugged", "dist_town", "dist_water"]
df.to_csv(PROC + "features_v2.csv", index=False)
print("missing values per new feature:", df[NEW].isna().sum().to_dict())
print(df[df.split != "final_test"].groupby("label")[NEW].median().round(2).to_string())
print("wrote", PROC + "features_v2.csv")
