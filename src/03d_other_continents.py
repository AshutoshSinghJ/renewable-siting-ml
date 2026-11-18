"""
Step 3d (extra check) - does the Europe-trained model work on other continents?

Takes the Random Forest exactly as trained in Step 3 (Europe without Iberia), with no retraining
and no tuning, and scores it on India, the USA and China. Features are built the same way as Step 2:
sampled at each installation and averaged per 5 km cell. The "I don't know" rules from Step 5
are applied with the same thresholds.

Run from the project root:  python src/03d_other_continents.py [IND] [USA] [CHN]
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import NearestNeighbors

RAW, PROC, OUT = "data/raw/", "data/processed/", "outputs/"
FEATURES = ["ghi", "wind100", "elev", "slope", "dist_grid"]
CLASSES = ["Hybrid", "Solar", "Wind"]
NAMES = {"IND": "India", "USA": "USA", "CHN": "China"}
COUNTRIES = sys.argv[1:] or ["IND", "USA", "CHN"]
MAX_GRID_KM = 20          # same "no grid coverage" rule as Step 2
DEM_CACHE = PROC + "dem_points_other.csv"
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

cells = pd.read_csv(PROC + "cells.csv")
cells = cells[cells.country.isin(COUNTRIES)]
pts = pd.read_csv(PROC + "points.csv").merge(cells[["cell_id", "country"]], on="cell_id").reset_index(drop=True)
print(f"{len(cells)} cells, {len(pts)} installation points in {COUNTRIES}")


def sample(path, df):
    with rasterio.open(path) as r:
        v = np.array([x[0] for x in r.sample(zip(df.lon, df.lat))], dtype=float)
        if r.nodata is not None:
            v[v == r.nodata] = np.nan
    return v


# ---- GHI (global file) and wind speed (official per-country Global Wind Atlas file) ----
pts["ghi"] = sample(RAW + "GSA_GHI_9arcsec.tif", pts)
pts["wind100"] = np.nan
for c in COUNTRIES:
    m = pts.country == c
    pts.loc[m, "wind100"] = sample(f"{RAW}gwa_ws100_other/{c}_wind-speed_100m.tif", pts[m])


# ---- elevation + slope from Copernicus DEM, read remotely (same method as Step 2) ----
def tile_url(lat, lon):
    la, lo = int(np.floor(lat)), int(np.floor(lon))
    name = (f"Copernicus_DSM_COG_30_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")
    return f"/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"


def dem_for_tile(args):
    url, sub = args
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
            dy = abs(r.res[1]) * 111_320
            dx = abs(r.res[0]) * 111_320 * np.cos(np.radians(p.lat))
            dzdx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * dx)
            dzdy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * dy)
            out.append((i, z[1, 1], np.degrees(np.arctan(np.hypot(dzdx, dzdy)))))
    return out


cache = pd.read_csv(DEM_CACHE) if os.path.exists(DEM_CACHE) else pd.DataFrame(columns=["lon", "lat", "elev", "slope"])
pts = pts.merge(cache, on=["lon", "lat"], how="left")
todo = pts[pts.elev.isna()]
if len(todo):
    groups = list(todo.groupby(todo.apply(lambda p: tile_url(p.lat, p.lon), axis=1)))
    print(f"Reading DEM for {len(todo)} points from {len(groups)} tiles ...")
    with ThreadPoolExecutor(16) as ex:
        rows = [x for res in ex.map(dem_for_tile, groups) for x in res]
    dem = pd.DataFrame(rows, columns=["idx", "elev", "slope"]).set_index("idx")
    pts.loc[dem.index, ["elev", "slope"]] = dem[["elev", "slope"]].values
    pd.concat([cache, pts.loc[dem.index, ["lon", "lat", "elev", "slope"]]]).drop_duplicates(["lon", "lat"]).to_csv(DEM_CACHE, index=False)

feat = pts.groupby("cell_id")[["ghi", "wind100", "elev", "slope"]].mean().reset_index()
feat = cells.merge(feat, on="cell_id", how="left")

# ---- distance to Gridfinder line, per country in a local equal-area projection (metres) ----
feat["dist_grid"] = np.nan
for c in COUNTRIES:
    m = feat.country == c
    f = feat[m]
    crs = f"+proj=laea +lat_0={f.lat.median():.1f} +lon_0={f.lon.median():.1f} +units=m"
    xy = gpd.GeoDataFrame(f[["cell_id"]], geometry=gpd.points_from_xy(f.lon, f.lat), crs="EPSG:4326").to_crs(crs)
    grid = gpd.read_file(RAW + "grid.gpkg", bbox=(f.lon.min() - 1, f.lat.min() - 1, f.lon.max() + 1, f.lat.max() + 1))
    print(f"{c}: {len(grid)} Gridfinder segments")
    near = gpd.sjoin_nearest(xy, grid[["geometry"]].to_crs(crs), distance_col="d").drop_duplicates("cell_id")
    feat.loc[m, "dist_grid"] = f.cell_id.map(near.set_index("cell_id").d / 1000).values
feat.loc[feat.dist_grid > MAX_GRID_KM, "dist_grid"] = np.nan

missing = feat[FEATURES].isna().any(axis=1)
print("Cells dropped for missing features:", feat[missing].groupby("country").size().to_dict(),
      "| NaN per feature:", feat[FEATURES].isna().sum()[lambda s: s > 0].to_dict())
feat = feat[~missing].copy()
feat.to_csv(PROC + "features_other.csv", index=False)

# ---- apply the Europe-trained models unchanged ----
rf = joblib.load(OUT + "rf_model.joblib")
eu = pd.read_csv(PROC + "features.csv")
tr = eu[eu.split == "train"]

# wind-speed rule: same threshold search as Step 3, on the same training data
ws = tr.wind100.to_numpy()
cands = np.quantile(ws, np.linspace(0.02, 0.98, 97))
t_rule = cands[int(np.argmax([f1_score(tr.label, np.where(ws >= t, "Wind", "Solar"), labels=CLASSES, average="macro",
                                       zero_division=0) for t in cands]))]

proba = rf.predict_proba(feat[FEATURES])
feat["rf_pred"] = rf.classes_[proba.argmax(axis=1)]
for i, c in enumerate(rf.classes_):
    feat[f"p_{c.lower()}"] = proba[:, i].round(3)
feat["rule_pred"] = np.where(feat.wind100 >= t_rule, "Wind", "Solar")

# "I don't know": same two rules and thresholds as Step 5
w = pd.read_csv(OUT + "feature_importance.csv", index_col="feature").loc[FEATURES, "rf_impurity_importance"]
mu, sd = tr[FEATURES].mean(), tr[FEATURES].std()
thr = pd.read_csv(OUT + "dont_know_summary.csv").aoa_threshold.iloc[0]
nn = NearestNeighbors(n_neighbors=1).fit(((tr[FEATURES] - mu) / sd * w).to_numpy())
feat["aoa_dist"] = nn.kneighbors(((feat[FEATURES] - mu) / sd * w).to_numpy())[0][:, 0]
feat["low_conf"] = proba.max(axis=1) < 0.5
feat["unfamiliar"] = feat.aoa_dist > thr
feat["idk"] = feat.low_conf | feat.unfamiliar

rows = []
for c in COUNTRIES:
    d = feat[feat.country == c]
    present = [k for k in CLASSES if (d.label == k).any()]
    rec = recall_score(d.label, d.rf_pred, labels=CLASSES, average=None, zero_division=0)
    conf = d[~d.idk]
    rows.append({
        "region": NAMES[c], "n_cells": len(d), **{f"n_{k}": int((d.label == k).sum()) for k in CLASSES},
        "majority_share": d.label.value_counts(normalize=True).iloc[0],
        "rf_accuracy": accuracy_score(d.label, d.rf_pred),
        "rf_macro_f1": f1_score(d.label, d.rf_pred, labels=present, average="macro", zero_division=0),
        "rule_accuracy": accuracy_score(d.label, d.rule_pred),
        **{f"rf_recall_{k}": (r if (d.label == k).any() else np.nan) for k, r in zip(CLASSES, rec)},
        "pct_I_dont_know": 100 * d.idk.mean(), "pct_unfamiliar": 100 * d.unfamiliar.mean(),
        "confident_accuracy": accuracy_score(conf.label, conf.rf_pred) if len(conf) else np.nan,
        "median_ghi": d.ghi.median(), "median_wind_solar": d[d.label == "Solar"].wind100.median(),
        "median_wind_wind": d[d.label == "Wind"].wind100.median(),
    })
res = pd.DataFrame(rows).round(3)
res.to_csv(OUT + "other_continents.csv", index=False)
feat[["cell_id", "country", "lon", "lat", "label", "rf_pred", "p_hybrid", "p_solar", "p_wind", "rule_pred",
      "low_conf", "unfamiliar", "idk"] + FEATURES].to_csv(OUT + "other_continents_predictions.csv", index=False)
pd.set_option("display.width", 220)
print(res.T.to_string())
print("Wrote outputs/other_continents.csv, outputs/other_continents_predictions.csv")
