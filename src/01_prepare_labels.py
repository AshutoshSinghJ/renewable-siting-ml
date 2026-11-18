"""
Step 1 - Prepare labels.

Turns Dunnett et al. (2020) solar and wind points into 5 km grid cells
labelled Solar / Wind / Hybrid, and assigns each cell to the training or
held-out test region.

Run from the project root:  python src/01_prepare_labels.py
"""
import sys
import geopandas as gpd
import numpy as np
import pandas as pd

RAW = "data/raw/"
OUT = "data/processed/cells.csv"

CELL_M = 5000                # 5 km cells (Hybrid definition depends on this - stated in report)
EQUAL_AREA = "ESRI:54012"    # World Eckert IV: equal-area, so every 5 km cell has the same area
MIN_MW = 1.0                 # utility-scale filter only; capacity is NEVER used as a feature
MIN_HYBRID = 300             # keep Hybrid as a class only if it has at least this many cells

# Region split (ISO3 codes). PROPOSED after looking at the counts - waiting for user confirmation.
# Test = Iberian Peninsula: one contiguous block that only touches the training region at the
# Pyrenees, has all 3 classes, and is sunnier/hotter than most of the training region,
# so it is a real "unseen region" test. Training = rest of Europe (>= 5,000 cells).
TEST_COUNTRIES = ["ESP", "PRT"]
EUROPE = ("ALB AND AUT BEL BGR BIH BLR CHE CYP CZE DEU DNK ESP EST FIN FRA GBR GRC HRV HUN IRL ISL "
          "ITA LIE LTU LUX LVA MDA MKD MLT MNE NLD NOR POL PRT ROU SRB SVK SVN SWE UKR XKO").split()
TRAIN_COUNTRIES = [c for c in EUROPE if c not in TEST_COUNTRIES]


def load(path, kind):
    g = gpd.read_file(path)
    n0 = len(g)
    # Dunnett's own recommended "farm" filter: not urban, not water.
    # `power` is only estimated for records that pass the authors' farm rules
    # (wind > 4 turbines, solar panel area > 1 ha), so NaN power = not a farm.
    g = g[(g["water"] != 1) & (g["urban"] != 1) & (g["power"] >= MIN_MW)].copy()
    print(f"{kind}: {n0} records -> {len(g)} after water/urban/>= {MIN_MW} MW filter")
    # Wind farms are MultiPoints (one point per turbine); use the farm's centroid.
    g = g.to_crs(EQUAL_AREA)
    g["geometry"] = g.geometry.centroid
    g["kind"] = kind
    g["GID_0"] = g["GID_0"].fillna("UNK")   # a few records have no country code
    return g[["GID_0", "kind", "geometry"]]


pts = pd.concat([load(RAW + "dunnett_solar_2020_WGS84.gpkg", "solar"),
                 load(RAW + "dunnett_wind_2020_WGS84.gpkg", "wind")], ignore_index=True)

# Snap each installation to a 5 km cell on a fixed equal-area grid.
pts["cx"] = np.floor(pts.geometry.x / CELL_M).astype(int)
pts["cy"] = np.floor(pts.geometry.y / CELL_M).astype(int)

cells = pts.groupby(["cx", "cy"]).agg(
    has_solar=("kind", lambda k: (k == "solar").any()),
    has_wind=("kind", lambda k: (k == "wind").any()),
    country=("GID_0", lambda c: c.mode().iloc[0]),   # majority country of the points in the cell
).reset_index()

# Only cells with an installation are used (no "empty land" negatives).
cells["label"] = np.select([cells.has_solar & cells.has_wind, cells.has_solar],
                           ["Hybrid", "Solar"], default="Wind")
cells["cell_id"] = "C_" + cells.cx.astype(str) + "_" + cells.cy.astype(str)

# Cell centre back to lon/lat for the output table.
centres = gpd.GeoSeries(gpd.points_from_xy((cells.cx + 0.5) * CELL_M, (cells.cy + 0.5) * CELL_M),
                        crs=EQUAL_AREA).to_crs("EPSG:4326")
cells["lon"], cells["lat"] = centres.x.round(5), centres.y.round(5)

# ---- Counts -------------------------------------------------------------
print("\nTotal labelled cells:", len(cells))
print(cells.label.value_counts().to_string())
tab = pd.crosstab(cells.country, cells.label)
tab["total"] = tab.sum(axis=1)
print("\nTop 15 countries:")
print(tab.sort_values("total", ascending=False).head(15).to_string())

n_hybrid = (cells.label == "Hybrid").sum()
print(f"\nHybrid cells: {n_hybrid} -> "
      + ("keep Hybrid as a class" if n_hybrid >= MIN_HYBRID else "too few: train Solar vs Wind only"))

# ---- Split --------------------------------------------------------------
cells["split"] = np.where(cells.country.isin(TRAIN_COUNTRIES), "train",
                          np.where(cells.country.isin(TEST_COUNTRIES), "test", "unused"))
if TRAIN_COUNTRIES:
    print("\nCells per split and class:")
    print(pd.crosstab(cells.split, cells.label, margins=True).to_string())

cells[["cell_id", "lon", "lat", "country", "label", "split"]].to_csv(OUT, index=False)

# Installation points per cell: Step 2 samples the rasters at these points (then averages per cell).
pts["cell_id"] = "C_" + pts["cx"].astype(str) + "_" + pts["cy"].astype(str)
ll = pts.geometry.to_crs("EPSG:4326")
pd.DataFrame({"cell_id": pts.cell_id, "kind": pts.kind,
              "lon": ll.x.round(6), "lat": ll.y.round(6)}).to_csv("data/processed/points.csv", index=False)
tab.sort_values("total", ascending=False).to_csv("outputs/step1_country_counts.csv")
print("\nWrote", OUT)
