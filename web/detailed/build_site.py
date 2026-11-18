"""
Build the project website: web/detailed/index.html (one self-contained file).

Reads the project outputs (cells, features, predictions, result tables) and the GADM country
borders from the Dunnett archive, packs them as JSON, and injects them into web/detailed/template.html.

Run from the project root:  python web/detailed/build_site.py
"""
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd

OUT = "outputs/"
GADM = "data/raw/gadm.gpkg"   # only needed if the border cache is missing
BBOX = (-32, 24, 70, 76)          # wider than Europe so the clipped edge stays off-screen

feat = pd.read_csv("data/processed/features.csv").merge(
    pd.read_csv("data/processed/cells.csv")[["cell_id", "lon", "lat", "country"]], on="cell_id")
pred = pd.read_csv(OUT + "test_predictions.csv").set_index("cell_id")

# ---- cells (column arrays keep the JSON small) ----
code = {"Solar": "S", "Wind": "W", "Hybrid": "H"}
feat = feat.sort_values("split")
t = feat.split == "test"
p = pred.reindex(feat.cell_id[t])
cells = {
    "lon": feat.lon.round(3).tolist(), "lat": feat.lat.round(3).tolist(),
    "lab": "".join(feat.label.map(code)), "split": "".join(np.where(t, "T", "R")),
    "cty": feat.country.tolist(),
    "ghi": feat.ghi.round(2).tolist(), "ws": feat.wind100.round(2).tolist(), "elev": feat.elev.round(0).astype(int).tolist(),
    "slope": feat.slope.round(1).tolist(), "grid": feat.dist_grid.round(2).tolist(),
    # test-only fields, aligned to the test rows in order
    "pred": "".join(p.rf_pred.map(code)),
    "final": "".join(np.where(p.final_pred == "I don't know", "?", p.rf_pred.map(code))),
    "lc": "".join(p.low_conf.map({True: "1", False: "0"})), "un": "".join(p.unfamiliar.map({True: "1", False: "0"})),
    "ps": p.p_solar.tolist(), "pw": p.p_wind.tolist(), "ph": p.p_hybrid.tolist(),
}

# ---- country borders (simplified) ----
# Simplified GADM borders are cached in web/detailed/borders.json (the full GADM file is ~500 MB).
CACHE = "web/detailed/borders.json"
if os.path.exists(CACHE):
    countries = json.load(open(CACHE))
else:
    g = gpd.read_file(GADM, layer="level0", bbox=BBOX)
    g = g.clip(gpd.GeoSeries.from_xy([BBOX[0], BBOX[2]], [BBOX[1], BBOX[3]]).total_bounds.tolist())
    g["geometry"] = g.geometry.simplify(0.04, preserve_topology=True)
    iso_col = [c for c in g.columns if c.upper() in ("GID_0", "ISO", "ISO3")][0]
    countries = []
    for iso, geom in zip(g[iso_col], g.geometry):
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        rings = []
        for poly in polys:
            if poly.area < 0.01:               # drop tiny islets (not visible at this scale)
                continue
            xy = np.round(np.asarray(poly.exterior.coords), 2)
            rings.append(xy.ravel().tolist())
        if rings:
            countries.append({"iso": iso, "rings": rings})
    json.dump(countries, open(CACHE, "w"), separators=(",", ":"))

# ---- result tables ----
def recs(f):
    return pd.read_csv(OUT + f).replace({np.nan: None}).to_dict("records")

data = {
    "cells": cells, "countries": countries,
    "results": recs("results.csv"), "diffs": recs("paired_differences.csv"), "cv": recs("cv_scores.csv"),
    "comp": recs("extra_model_comparison.csv"), "abl": recs("feature_ablation.csv"),
    "shap": recs("feature_importance.csv"), "idk": recs("dont_know_summary.csv"),
    "region": recs("region_holdout.csv"), "cont": recs("other_continents.csv"), "overfit": recs("overfit_check.csv"),
    "conf": pd.crosstab(p.index.map(feat.set_index("cell_id").label), p.rf_pred)
              .reindex(index=["Solar", "Wind", "Hybrid"], columns=["Solar", "Wind", "Hybrid"], fill_value=0).values.tolist(),
    "counts": feat.groupby(["split", "label"]).size().unstack().to_dict("index"),
}
js = json.dumps(data, separators=(",", ":"))
html = open("web/detailed/template.html", encoding="utf-8").read().replace("/*__DATA__*/null", js)
# standalone page (for any static host, e.g. GitHub Pages)
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n')
title_end = html.index("</style>") + len("</style>")
open("web/detailed/index.html", "w", encoding="utf-8").write(
    head + html[:title_end] + "\n</head>\n<body>\n" + html[title_end:] + "\n</body>\n</html>\n")
print(f"cells {len(feat)}, countries {len(countries)}, data {len(js) / 1e6:.2f} MB -> web/detailed/index.html")
