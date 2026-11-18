"""
Step 10 (improvement round) - can a neural network learn local context by itself?

Instead of hand-made context features (wind_rel, wind_pct, ...), each site gets a small "patch":
an 11 x 11 grid of wind speed and sunshine values covering +/- 30 km around it (spacing 6 km),
expressed relative to the site's own value. A neural network (MLP) learns from the 242 patch values
plus the 5 base features. This is the deep-learning version of "windier than its surroundings".
(A convolutional network would be the natural next step; PyTorch is not installed here.)

Compared, with the same leave-one-region-out test over 11 regions:
  MLP on patches + base | MLP on the 13 hand-made features | Extra Trees on the 13 features (final model)

Run from the project root:  python src/10_patch_nn.py
"""
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, RAW = 42, "data/raw/"
CLASSES = ["Hybrid", "Solar", "Wind"]
BASE = ["ghi", "wind100", "elev", "slope", "dist_grid"]
HAND = open("outputs/v2_best_feature_set.txt").read().splitlines()[1].split(",")
N, STEP_KM, FACTOR = 11, 6.0, 16

o = pd.read_csv("outputs/v2_loro_predictions.csv")[["cell_id", "region"]]
df = pd.read_csv("data/processed/features_v2.csv").merge(o, on="cell_id")      # development cells only
off = (np.arange(N) - N // 2) * STEP_KM


def coarse(path, bbox=None):
    with rasterio.open(path) as r:
        win = from_bounds(*bbox, r.transform).round_offsets().round_lengths() if bbox else Window(0, 0, r.width, r.height)
        win = win.intersection(Window(0, 0, r.width, r.height))
        h, w = max(1, int(win.height // FACTOR)), max(1, int(win.width // FACTOR))
        a = r.read(1, window=win, out_shape=(h, w), resampling=Resampling.average, masked=True).astype(float).filled(np.nan)
        t = r.window_transform(win) * rasterio.Affine.scale(win.width / w, win.height / h)
    return a, t


def patches(a, t, lon, lat):
    """N x N samples around each site from a coarse raster (nearest pixel)."""
    lon, lat = np.asarray(lon), np.asarray(lat)
    dlat = off[:, None] / 111.32                                  # rows: north-south
    out = np.full((len(lon), N, N), np.nan)
    for i, (x, y) in enumerate(zip(lon, lat)):
        dlon = off[None, :] / (111.32 * max(np.cos(np.radians(y)), 0.1))
        xs, ys = x + dlon + 0 * dlat, y - dlat + 0 * dlon
        cols, rows = ~t * (xs, ys)
        rows, cols = rows.astype(int), cols.astype(int)
        ok = (rows >= 0) & (rows < a.shape[0]) & (cols >= 0) & (cols < a.shape[1])
        p = np.full((N, N), np.nan)
        p[ok] = a[rows[ok], cols[ok]]
        out[i] = p
    return out


cache = "data/processed/patches_dev.npz"
if os.path.exists(cache):
    z = np.load(cache)
    WP, GP = z["wind"], z["ghi"]
else:
    WP = np.full((len(df), N, N), np.nan)
    GP = np.full((len(df), N, N), np.nan)
    for iso, g in df.groupby("country"):
        folder = "gwa_ws100" if os.path.exists(f"{RAW}gwa_ws100/{iso}_wind-speed_100m.tif") else "gwa_ws100_other"
        path = f"{RAW}{folder}/{iso}_wind-speed_100m.tif"
        if os.path.exists(path):
            a, t = coarse(path)
            WP[g.index] = patches(a, t, g.lon, g.lat)
        a, t = coarse(RAW + "GSA_GHI_9arcsec.tif", (g.lon.min() - 1, g.lat.min() - 1, g.lon.max() + 1, g.lat.max() + 1))
        GP[g.index] = patches(a, t, g.lon, g.lat)
        print(f"  patches {iso}: {len(g)} cells")
    np.savez_compressed(cache, wind=WP, ghi=GP)

# patch values relative to the site itself; missing pixels (sea, outside the country file) -> 0 difference
Wr = np.nan_to_num(WP - df.wind100.values[:, None, None], nan=0.0).reshape(len(df), -1)
Gr = np.nan_to_num(GP - df.ghi.values[:, None, None], nan=0.0).reshape(len(df), -1)
Xpatch = np.hstack([df[BASE].values, Wr, Gr])
print(f"patch features: {Xpatch.shape[1]} per cell")


def mlp():
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=300, random_state=SEED))


def et():
    return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)


def loro(make, X):
    pred = np.empty(len(df), dtype=object)
    for r in df.region.unique():
        te = (df.region == r).values
        m = make().fit(X[~te], df.label.values[~te])
        pred[te] = m.predict(X[te])
    return pred


rows = []
for name, make, X in [("Neural network on raster patches + base (247 inputs)", mlp, Xpatch),
                      ("Neural network on 13 hand-made features", mlp, df[HAND].values),
                      ("Extra Trees on 13 hand-made features (final)", et, df[HAND].values)]:
    p = loro(make, X)
    per = {r: accuracy_score(df.label[df.region == r], p[(df.region == r).values]) for r in df.region.unique()}
    rows.append({"model": name, "accuracy": accuracy_score(df.label, p),
                 "macro_f1": f1_score(df.label, p, labels=CLASSES, average="macro"),
                 "mean_region_acc": np.mean(list(per.values())), "worst_region_acc": min(per.values()),
                 "british_isles": per["British Isles"], "india": per["India"]})
    print(rows[-1])
res = pd.DataFrame(rows).round(3)
res.to_csv("outputs/v2_patch_nn.csv", index=False)
print(res.to_string(index=False))
