"""
Build the map-first project website: web/globe/index.html (standalone).

Packs the cells, the Iberia test predictions, the leave-one-region-out predictions, the result
tables and simplified world borders (GADM, from the Dunnett archive) into template.html.

Run from the project root:  python web/globe/build_site.py
"""
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.polygon import orient

OUT = "outputs/"
GADM = "data/raw/gadm.gpkg"   # only needed if the border cache is missing
code = {"Solar": "S", "Wind": "W", "Hybrid": "H"}

feat = pd.read_csv("data/processed/features.csv").merge(
    pd.read_csv("data/processed/cells.csv")[["cell_id", "lon", "lat", "country"]], on="cell_id")
test = pd.read_csv(OUT + "test_predictions.csv").set_index("cell_id")
loro = pd.read_csv(OUT + "region_holdout_predictions.csv").set_index("cell_id")
feat = feat.sort_values("split")                     # "test" rows first
t = feat.split == "test"
tp = test.reindex(feat.cell_id[t])
lp = loro.reindex(feat.cell_id)
regions = list(pd.read_csv(OUT + "region_holdout.csv").region)

cells = {
    "lon": feat.lon.round(3).tolist(), "lat": feat.lat.round(3).tolist(),
    "lab": "".join(feat.label.map(code)), "split": "".join(np.where(t, "T", "R")),
    "cty": feat.country.tolist(), "reg": [regions.index(r) for r in lp.region],
    "ghi": feat.ghi.round(2).tolist(), "ws": feat.wind100.round(2).tolist(), "elev": feat.elev.round(0).astype(int).tolist(),
    # Iberia test (aligned to the first n_test rows)
    "pred": "".join(tp.rf_pred.map(code)),
    "final": "".join(np.where(tp.final_pred == "I don't know", "?", tp.rf_pred.map(code))),
    "lc": "".join(tp.low_conf.map({True: "1", False: "0"})),
    # leave-one-region-out prediction for every cell
    "lpred": "".join(lp.rf_pred.map(code)),
    "lps": lp.p_solar.tolist(), "lpw": lp.p_wind.tolist(), "lph": lp.p_hybrid.tolist(),
}
# India and USA cells, appended after the European ones (Europe-trained model applied unchanged)
oc = pd.read_csv(OUT + "other_continents_predictions.csv")
cells["nEU"] = len(feat)
for k, v in {"lon": oc.lon.round(3), "lat": oc.lat.round(3), "cty": oc.country, "ghi": oc.ghi.round(2),
             "ws": oc.wind100.round(2), "elev": oc.elev.round(0).astype(int), "lps": oc.p_solar, "lpw": oc.p_wind,
             "lph": oc.p_hybrid}.items():
    cells[k] += v.tolist()
cells["reg"] += [-1] * len(oc)
cells["lab"] += "".join(oc.label.map(code))
cells["split"] += "O" * len(oc)
cells["lpred"] += "".join(oc.rf_pred.map(code))
cells["oidk"] = "".join(np.where(oc.idk, "1", "0"))

# World borders, simplified. Rings oriented clockwise, which is what d3's spherical paths expect.
# Reading the full GADM file takes minutes, so the simplified result is cached in web/globe/world.json.
WORLD_CACHE = "web/globe/world.json"
if os.path.exists(WORLD_CACHE):
    countries = json.load(open(WORLD_CACHE))
else:
    g = gpd.read_file(GADM, layer="level0")
    g["geometry"] = g.geometry.simplify(0.1, preserve_topology=True)
    countries = []
    for iso, geom in zip(g["GID_0"], g.geometry):
        if geom is None or geom.is_empty:
            continue
        polys = []
        for poly in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
            if poly.area < 0.05:
                continue
            ring = np.round(np.asarray(orient(poly, sign=-1.0).exterior.coords), 2).tolist()
            polys.append([ring])
        if polys:
            countries.append({"type": "Feature", "properties": {"iso": iso},
                              "geometry": {"type": "MultiPolygon", "coordinates": polys}})
    json.dump(countries, open(WORLD_CACHE, "w"), separators=(",", ":"))


def recs(f):
    return pd.read_csv(OUT + f).replace({np.nan: None}).to_dict("records")


data = {
    "cells": cells, "world": {"type": "FeatureCollection", "features": countries}, "regions": regions,
    "results": recs("results.csv"), "diffs": recs("paired_differences.csv"), "idk": recs("dont_know_summary.csv"),
    "region": recs("region_holdout.csv"), "overfit": recs("overfit_check.csv"), "shap": recs("feature_importance.csv"),
    "cont": recs("other_continents.csv"),
    "conf": pd.crosstab(tp.index.map(feat.set_index("cell_id").label), tp.rf_pred)
              .reindex(index=["Solar", "Wind", "Hybrid"], columns=["Solar", "Wind", "Hybrid"], fill_value=0).values.tolist(),
}
js = json.dumps(data, separators=(",", ":"))
html = open("web/globe/template.html", encoding="utf-8").read().replace("/*__DATA__*/null", js)
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>html,body{margin:0;height:100%}[hidden]{display:none!important}</style>\n')
cut = html.index("</style>") + len("</style>")
open("web/globe/index.html", "w", encoding="utf-8").write(
    head + html[:cut] + "\n</head>\n<body>\n" + html[cut:] + "\n</body>\n</html>\n")
print(f"cells {len(feat)}, countries {len(countries)}, data {len(js) / 1e6:.2f} MB -> web/globe/")
