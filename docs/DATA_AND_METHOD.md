# Data, Maps and Method: the complete technical guide

This document explains **every dataset, every map operation and every modelling step** of the
project, in the order they happen: *what we did → how it works → why (the justification) → the
evidence*. All numbers come from the files in `outputs/` and `data/processed/`.

**Contents**
1. [The pipeline at a glance](#1-the-pipeline-at-a-glance)
2. [All datasets](#2-all-datasets)
3. [Maps and coordinate systems](#3-maps-and-coordinate-systems)
4. [Step 1: from farm points to labelled cells](#4-step-1-from-farm-points-to-labelled-cells)
5. [Step 2: extracting the features](#5-step-2-extracting-the-features)
6. [Choosing training and test regions](#6-choosing-training-and-test-regions)
7. [Step 3: models, training and evaluation](#7-step-3-models-training-and-evaluation)
8. [Extra checks: proving the result](#8-extra-checks-proving-the-result)
9. [Step 4: explaining the model (SHAP)](#9-step-4-explaining-the-model-shap)
10. [Step 5: confidence and "I don't know"](#10-step-5-confidence-and-i-dont-know)
11. [Maps on the website](#11-maps-on-the-website)
12. [Summary: every decision and its justification](#12-summary-every-decision-and-its-justification)

---

## 1. The pipeline at a glance

```
Dunnett 2020 farm points ──▶ filter real farms ──▶ 5 km equal-area grid ──▶ label cells (Solar/Wind/Hybrid)
                                                                                   │
Global Solar Atlas ─┐                                                              ▼
Global Wind Atlas  ─┼──▶ sample at every farm ──▶ average per cell ──▶ 5 features per cell
Copernicus DEM     ─┤      (+ slope, + grid distance)
Gridfinder         ─┘                                                              │
                                                                                   ▼
        train on Europe (no Iberia) ──▶ spatial CV tuning ──▶ Random Forest + 3 baselines
                                                                                   │
            test once on Spain + Portugal ──▶ SHAP ──▶ "I don't know" ──▶ extra checks
                                                    (8 regions, USA, China, India, overfitting)
```

| Step | Script | Output |
|---|---|---|
| 1. Labels | `src/01_prepare_labels.py` | `data/processed/cells.csv`, `points.csv` |
| 2. Features | `src/02_extract_features.py` | `data/processed/features.csv` |
| 3. Models | `src/03_train_evaluate.py` | `outputs/results.csv`, `cv_scores.csv`, `rf_model.joblib`, confusion matrix |
| 3b–3e. Checks | `src/03b` … `src/03e` | model comparison, feature subsets, regions, continents, overfitting, relative features |
| 4. Explain | `src/04_explain.py` | SHAP plots, `feature_importance.csv` |
| 5. "I don't know" | `src/05_dont_know.py` | `dont_know_summary.csv`, `test_predictions.csv` |

---

## 2. All datasets

### 2.1 Labels: Dunnett et al. (2020), wind and solar farm locations

| Item | Detail |
|---|---|
| Citation | Dunnett, S., Sorichetta, A., Taylor, G., Eigenbrod, F. (2020). Harmonised global datasets of wind and solar farm locations and power. *Scientific Data* 7, 130 |
| Source | figshare, doi 10.6084/m9.figshare.11310269 (`global_wind_solar_2020.zip`) |
| Built from | OpenStreetMap, 2020 snapshot; harmonised and quality-checked by the authors |
| Files used | `global_solar_2020_WGS84.gpkg` (35,272 points), `global_wind_2020_WGS84.gpkg` (33,514 multipoints) |
| Coordinate system | WGS84 (longitude/latitude, EPSG:4326) |
| Columns | `GID_0` country code · `panels` / `turbines` count · `panel.area`, `landscape.area` · `water` (1 = in water) · `urban` (1 = in a city) · `power` (estimated MW) |

**How it works:** every solar panel or panel field, and every turbine, mapped by OpenStreetMap
volunteers is a record. A wind farm is a *multipoint*: one point per turbine.

**Justification:**
- **Real decisions:** every record is something that was actually built, so every label is a fact.
- **Peer-reviewed:** published in a Nature-group journal; capacity cross-checked against national
  statistics.
- **Both technologies in one harmonised format:** the model can't learn differences between two
  data sources instead of between sites.
- **Global:** this is what made the USA, China and India tests possible.
- **Free and reproducible.**

**Known limitations:** OpenStreetMap coverage is uneven (strong in Europe, North America and East
Asia; weak in India); a 2020 snapshot; capacity values are estimates.

### 2.2 Sunshine: Global Solar Atlas 2.0 (GHI)

| Item | Detail |
|---|---|
| Provider | Solargis, for the World Bank Group (ESMAP) |
| Variable | **GHI**, Global Horizontal Irradiation, long-term average, **kWh/m²/day** |
| Resolution | 9 arc-seconds (~250 m), world GeoTIFF |
| Coverage | 60°S to 65°N (nothing further north) |
| File | `data/raw/GSA_GHI_9arcsec.tif` |

**Justification:** GHI is the standard measure of solar resource: the sunlight that reaches a flat
surface. It's the first thing a solar developer checks.

### 2.3 Wind: Global Wind Atlas (wind speed at 100 m)

| Item | Detail |
|---|---|
| Provider | DTU Wind Energy, for the World Bank Group (ESMAP); described in Davis et al. (2023), *BAMS* |
| Variable | Mean wind speed at **100 m** above ground, **m/s** |
| Resolution | 250 m |
| Files | One official GeoTIFF per country, `data/raw/gwa_ws100/<ISO3>_wind-speed_100m.tif` (36 European countries) and `gwa_ws100_other/` (USA, China, India) |
| Download | `https://globalwindatlas.info/api/gis/country/<ISO3>/wind-speed/100` |

**Justification:** 100 m is roughly the hub height of modern turbines, so it's the wind the turbine
actually uses. Wind speed is the first thing a wind developer checks.

**Data-quality story:** the first global wind file we had on disk was **not genuine** Global Wind
Atlas data. Spot checks against the official India file showed the Gujarat coast at 4.7 m/s
instead of 6.6 m/s, and it put Bavaria (9.0 m/s) above the German North Sea coast (8.6 m/s). All
three "global" copies gave identical wrong values, so we replaced them with the official
per-country files. After the fix, Bavaria read 5.7 m/s and the coast 9.3 m/s, as expected.

### 2.4 Terrain: Copernicus DEM GLO-90

| Item | Detail |
|---|---|
| Provider | European Space Agency (from the TanDEM-X mission) |
| Variable | Elevation above sea level (m) |
| Resolution | 3 arc-seconds (~90 m), in 1°×1° tiles |
| Access | Read **online** from the public AWS bucket `copernicus-dem-90m` (cloud-optimised GeoTIFFs) |

**How we read it:** cloud-optimised GeoTIFFs let a program download only the few pixels it needs.
For each farm we read a 3×3 pixel window, about 0.1% of a tile. That avoided downloading
roughly 3.3 GB (700 tiles). Results are cached in `data/processed/dem_points.csv`.

**Justification:** elevation and slope are standard terrain factors. Solar needs flat land; wind
farms are often on ridges and hills.

### 2.5 Power grid: Gridfinder

| Item | Detail |
|---|---|
| Citation | Arderne, C., Zorn, C., Nicolas, C., Koks, E. E. (2020). Predictive mapping of the global power system using open data. *Scientific Data* 7, 19 |
| Source | zenodo record 3628142, file `grid.gpkg` (725 MB) |
| Content | Global map of power lines: OpenStreetMap lines plus lines predicted from night-time lights |

**Justification:** a farm must connect to the grid, and long connections are expensive, so grid
distance is a standard siting factor.

**Limitation found:** Gridfinder has no lines on some islands (Canary Islands, small Mediterranean
islands). Their "nearest line" was on another coast, 20–451 km away, so those cells were removed
(Section 5.4).

### 2.6 Borders: GADM (websites only)
GADM country borders, simplified, are used **only to draw maps on the websites**. The model never
uses them.

### 2.7 Datasets considered and rejected

| Dataset | Why not |
|---|---|
| Land cover (ESA WorldCover / Copernicus) | **Leakage:** 2020 imagery shows the farm itself |
| Roads (GRIP, OpenStreetMap) | **Leakage:** farms build their own access roads |
| Plant capacity (in Dunnett) | **Leakage:** size reveals the type; used only as the ≥ 1 MW filter |
| Global Energy Monitor trackers | Different size thresholds for solar (≥ 1 MW) and wind (≥ 10 MW); needs sign-up; kept only as a possible cross-check |
| National registers (UK REPD, US wind database) | One country each, different formats |
| Expert suitability maps (MCDA/AHP) | Opinions about where farms *could* go, not where they *are* |

---

## 3. Maps and coordinate systems

A coordinate system (CRS) decides how the round Earth is flattened into numbers. Different jobs
need different ones.

| CRS | Used for | Why |
|---|---|---|
| **WGS84** (EPSG:4326), longitude/latitude in degrees | Reading all inputs; sampling rasters; storing cell centres | All sources come in it; rasters are stored in it |
| **World Eckert IV** (ESRI:54012), metres, **equal-area** | Building the 5 km grid | Every cell covers the same land area anywhere on Earth, so cells in Norway and Spain are comparable |
| **ETRS89 / LAEA Europe** (EPSG:3035), metres, equal-area | Distance to grid in Europe; 100 km CV blocks | The European standard projection; accurate distances and areas across Europe |
| **Lambert azimuthal equal-area centred on each country** | Distance to grid in the USA, China and India | Same idea as EPSG:3035, centred on each country so distances stay accurate there |
| **Orthographic** (globe website) | Drawing the website map only | Display only; no calculation uses it |

**Why not do everything in degrees?** One degree of longitude is 111 km at the equator but only
about 56 km in Norway. Distances and areas in degrees would be wrong, and grid cells would shrink
towards the poles.

---

## 4. Step 1: from farm points to labelled cells

Script: `src/01_prepare_labels.py`

### 4.1 Filtering real farms

| Rule | How | Why |
|---|---|---|
| Drop `water == 1` | Removes offshore and on-water records | Offshore wind is out of scope |
| Drop `urban == 1` | Removes records inside cities | Mostly rooftop solar, out of scope |
| Keep `power ≥ 1 MW` | Removes small installations **and** records with no power value | Utility-scale only |

**Why "no power value" is dropped:** two-thirds of records had no `power`. The dataset README
explains that the authors estimate power **only for real farms** (wind with more than 4 turbines;
solar with more than 1 ha of panels, outside cities). So a missing value means "not a farm". This
is the authors' own recommendation, not our guess.

**Result:** 35,272 solar → **11,498**; 33,514 wind → **12,580**; total **24,078 farms**.

### 4.2 Building the grid
1. Each wind farm (many turbine points) is reduced to its **centroid**.
2. All points are projected to **Eckert IV** (metres).
3. Each point gets a cell index: `cx = floor(x / 5000)`, `cy = floor(y / 5000)`, which is a fixed
   5 km × 5 km grid.
4. Each cell gets a label: **Solar** (only solar), **Wind** (only wind) or **Hybrid** (both).
5. The cell's country = the most common country of its points.
6. The cell centre is converted back to longitude/latitude for storage.

**Why 5 km:** small enough to keep local conditions, large enough to group one farm's turbines.
**Why Hybrid this way:** simple and reproducible. An earlier sensitivity check (with a slightly
different filter) counted cells containing both technologies at different cell sizes: 33 at 1 km,
107 at 2 km, 409 at 5 km, 830 at 10 km. So Hybrid depends strongly on cell size, which is listed as a
limitation; 5 km is the smallest size with enough Hybrid examples to learn from.

**Result:** **18,705 cells** worldwide (Wind 9,609 · Solar 8,700 · Hybrid 396).
Biggest countries: Germany 3,054 · USA 2,342 · China 2,313 · Italy 1,644 · Spain 1,372 · UK 1,024.

**Output:** `cells.csv` (`cell_id, lon, lat, country, label, split`) and `points.csv` (each farm
point with its cell).

---

## 5. Step 2: extracting the features

Script: `src/02_extract_features.py`

### 5.1 Sampling at each farm
For every farm point in the training and test regions (13,457 points), the value of each raster is
read at the point's exact location (the pixel it falls in). **Why at farms, not across the whole
cell?** A 5 km cell can include valleys, villages and hills; the farm sits on one specific spot, and
that spot is what the developer chose.

### 5.2 How each feature is computed

**Sunshine (`ghi`):** pixel value from the Global Solar Atlas file.

**Wind speed (`wind100`):** pixel value from the farm's **own country** Global Wind Atlas file.
Country files are cut at the border, so a farm right on a border can fall outside its own file. In
that case we take the value from any neighbouring country's file that covers it.

**Elevation (`elev`):** the centre pixel of a 3×3 window read from the Copernicus DEM.

**Slope (`slope`):** Horn's (1981) method, the standard used by GIS software. With the 3×3 window

```
a b c
d e f        dz/dx = ((c + 2f + i) − (a + 2d + g)) / (8 · Δx)
g h i        dz/dy = ((g + 2h + i) − (a + 2b + c)) / (8 · Δy)
             slope = arctan( √( (dz/dx)² + (dz/dy)² ) )   in degrees
```

where Δx and Δy are the pixel sizes in metres: Δy = pixel height in degrees × 111,320 and
Δx = pixel width in degrees × 111,320 × cos(latitude). Longitude degrees shrink towards the poles,
hence the cosine.

**Distance to grid (`dist_grid`):** the cell centre and all Gridfinder lines in the region are
projected to EPSG:3035 (metres), and the nearest line is found with a spatial nearest-neighbour join.
The result is converted to km.

### 5.3 From points to cells
Sunshine, wind speed, elevation and slope are **averaged over all farm points in the cell**. Grid
distance is measured once from the cell centre. Each cell is also given a **`block_id`**: the 100 km
× 100 km block (in EPSG:3035) it lies in, used for spatial cross-validation.

### 5.4 Cleaning the features

| Dropped | Cells | Reason |
|---|---|---|
| Far north | 34 | Above ~65°N, no Global Solar Atlas data |
| Azores and Madeira | 13 | The official Portugal wind file doesn't cover them |
| Islands with no grid lines | 20 | Nearest Gridfinder line 20–451 km away on another coast (15 Canary Islands, 5 small Mediterranean islands). Every cell above 20 km was such an island, so > 20 km = no coverage |
| Kosovo | 1 | No Global Wind Atlas country file |
| **Total** | **68** | |

**Final dataset:** `features.csv`, **10,724 cells**: `cell_id, ghi, wind100, elev, slope,
dist_grid, label, split, block_id`.

### 5.5 What the features look like (medians)

| | Sunshine (kWh/m²/day) | Elevation (m) |
|---|---|---|
| Training region (Europe without Iberia) | 3.1 | 148 |
| Test region (Iberia) | 4.6 | 717 |

| Wind speed (median, m/s) | Solar cells | Wind cells |
|---|---|---|
| Training region | 5.7 | 7.4 |
| British Isles | 7.7 | 9.1 |
| USA | 5.7 | 8.1 |
| China | 5.3 | 7.7 |
| India | 5.5 | 6.4 |

These medians explain almost all the results later: the model works where wind farms are much
windier than solar farms.

### 5.6 Feature justification

| Feature | Factor it represents | Justification |
|---|---|---|
| Sunshine | Solar resource | Standard solar siting factor (Sachit et al. 2022) |
| Wind speed 100 m | Wind resource | Standard wind siting factor; hub height |
| Elevation | Terrain | Wind farms favour exposed high ground |
| Slope | Terrain | Solar needs flat land; construction cost |
| Distance to grid | Access | Connection cost |

---

## 6. Choosing training and test regions

| Region | Cells | Hybrid cells |
|---|---|---|
| **Europe** | **10,792** | **339** |
| USA | 2,342 | 1 |
| China | 2,313 | 40 |
| India | 777 | 3 |

**Training region:** Europe without Spain and Portugal, **9,176 cells** (Solar 5,320 · Wind 3,548
· Hybrid 308), 33 countries.
**Test region:** Spain + Portugal, **1,548 cells** (Solar 758 · Wind 765 · Hybrid 25).

**Justification:**
- Europe holds 58% of all cells and **339 of 396 Hybrid cells**. It's the only region where three
  classes can be trained and whole regions still held out. Our rule required at least 300 Hybrid
  cells.
- Iberia is one connected block touching training only at the Pyrenees, contains all three
  classes, and has a different climate. That makes it a fair but hard test.
- **No random split:** two cells 5 km apart are nearly identical, so a random split puts
  near-copies of training cells into the test set and inflates the score (Roberts et al. 2017).

---

## 7. Step 3: models, training and evaluation

Script: `src/03_train_evaluate.py` · features: `ghi, wind100, elev, slope, dist_grid` · seed 42.

### 7.1 The four models

| Model | How it works | Why it is included |
|---|---|---|
| **Majority class** | Always predicts the most common training class (Solar) | The minimum any useful model must beat |
| **Wind-speed rule** | Predict Wind if `wind100 ≥ t`, else Solar. `t` is chosen by trying 97 thresholds (the 2nd to 98th percentiles of training wind speed) and keeping the one with the best macro F1 on training data: **t = 6.60 m/s** | The key baseline: if the Random Forest can't beat this, it has only learned "windy = wind" |
| **Logistic regression** | Features standardised (z-scores), then a linear model with `class_weight="balanced"` | A simple, standard ML model; shows whether non-linearity is needed |
| **Random Forest** | 300 decision trees, each trained on a random sample of cells and features; their outputs are averaged | Main model |

### 7.2 How the Random Forest works
1. Each tree repeatedly splits the data with yes/no questions on one feature at a time ("is wind
   speed above 6.8 m/s?"), choosing the split that best separates the classes.
2. Each tree sees a random bootstrap sample of cells and, at each split, a random subset of
   features, so the trees differ from one another.
3. A cell's prediction from one tree = the class shares in the leaf it lands in.
4. The forest averages the shares over 300 trees → three probabilities (Solar, Wind, Hybrid).
5. `class_weight="balanced"` gives rare classes more weight (Hybrid is 3% of training cells).

### 7.3 Why the Random Forest (justification)

| Reason | Evidence / source |
|---|---|
| Standard, robust model for tabular data | Breiman (2001) |
| Captures non-linear interactions ("windy *and* not sunny") | Logistic regression (linear) scored lower in CV: 0.477 vs 0.543 |
| No feature scaling needed; robust to outliers | Property of tree models |
| Gives probabilities → enables "I don't know" | Section 10 |
| Explainable with SHAP | Section 9 |
| Used in published siting research | Sachit et al. (2022) |
| **No other standard model did better** | Gradient boosting 0.542, Extra Trees 0.537, k-NN 0.510, SVM 0.476 vs RF **0.543** (Section 8.1) |

### 7.4 Tuning with spatial cross-validation
- `GroupKFold(n_splits=5)` with `groups = block_id`: each fold holds out whole 100 km blocks, so
  validation mimics "a new place". Training region only.
- Grid: `max_depth ∈ {None, 10, 20}` × `min_samples_leaf ∈ {1, 5}`, scored by macro F1.
- CV results: 0.522–0.543. **Selected: max_depth = 10, min_samples_leaf = 1 (0.543).**
- The selected model is then fitted on the whole training region and **evaluated once** on Iberia.

| Model | Spatial CV macro F1 (training region) |
|---|---|
| Majority | 0.244 |
| Wind rule | 0.498 |
| Logistic regression | 0.477 |
| **Random Forest** | **0.543** |

### 7.5 Metrics, and why each

| Metric | Meaning | Why |
|---|---|---|
| Accuracy | Share of cells predicted correctly | Easy to understand, but hides the rare class |
| **Macro F1** | Average of the F1 scores of Solar, Wind and Hybrid, equally weighted | Solar + Wind are 97% of cells; macro F1 exposes poor Hybrid performance |
| Recall per class | Of all real Solar cells, how many were found | Shows which class fails |
| Precision (Hybrid) | Of all cells predicted Hybrid, how many were right | Catches a model that "finds" Hybrids by over-guessing |
| Confusion matrix | True class × predicted class | Shows exactly which mistakes happen |
| **Bootstrap 95% CI** | Re-sample the 1,548 test cells 2,000 times, recompute the metric, take the 2.5th–97.5th percentiles | Shows whether a difference is real or chance (Efron & Tibshirani 1993) |
| **Paired bootstrap difference** | Same resampled cells for both models | Fair comparison of two models |

### 7.6 Results on Iberia (`outputs/results.csv`)

| Model | Accuracy (95% CI) | Macro F1 (95% CI) | Recall Solar | Recall Wind | Recall Hybrid |
|---|---|---|---|---|---|
| Majority | 0.490 (0.465–0.516) | 0.219 | 1.000 | 0.000 | 0.00 |
| Wind rule | 0.882 (0.866–0.899) | 0.593 | 0.926 | 0.868 | 0.00 |
| Logistic regression | 0.844 (0.826–0.861) | 0.641 | 0.836 | 0.860 | 0.56 (precision 0.07) |
| **Random Forest** | **0.901 (0.886–0.915)** | 0.627 (0.599–0.675) | 0.918 | 0.911 | 0.04 |

- RF − wind rule: **+1.8 accuracy points (CI +0.5 to +3.2)**. Real, but small.
- RF − logistic regression: +5.7 accuracy points; macro F1 statistically tied. Logistic
  regression's Hybrid recall comes from guessing Hybrid 199 times with 14 correct.

**Confusion matrix (Random Forest, Iberia)**

| True ↓ / Predicted → | Solar | Wind | Hybrid |
|---|---|---|---|
| Solar | **696** | 59 | 3 |
| Wind | 65 | **697** | 3 |
| Hybrid | 6 | 18 | **1** |

---

## 8. Extra checks: proving the result

### 8.1 Other classifiers (`src/03b_extra_checks.py`, training-region CV only)

| Model | CV macro F1 |
|---|---|
| **Random Forest** | **0.543** |
| Gradient boosting (HistGradientBoosting) | 0.542 |
| Extra Trees | 0.537 |
| k-nearest neighbours (k = 15) | 0.510 |
| Logistic regression | 0.477 |
| SVM (RBF) | 0.476 |

**Conclusion:** the limit is the data, not the algorithm. RF is justified.

### 8.2 Feature subsets (what each feature group adds)

| Features | CV macro F1 |
|---|---|
| Wind speed only | 0.503 |
| + sunshine | 0.524 |
| + elevation, slope | 0.540 |
| + grid distance (all 5) | 0.543 |

**Conclusion:** wind speed does most of the work. Grid distance adds almost nothing, which also
shows it isn't leaking the answer.

### 8.3 Overfitting (`src/03c_region_holdout.py`)

| Random Forest scored on | Accuracy | Macro F1 |
|---|---|---|
| Training data (seen) | 0.812 | 0.701 |
| Unseen 100 km blocks (spatial CV) | 0.732 | 0.545 |
| Iberia (unseen region) | 0.901 | 0.627 |

- A forest with no depth limit scores 1.000 on its own training data (memorisation). Ours is
  limited to depth 10 and scores 0.812 → **no serious overfitting**.
- Iberia scores higher than CV because **Iberia is easier**: the wind rule also jumps from 0.74
  (training region) to 0.88 (Iberia).

### 8.4 Every European region held out in turn

Europe is split into 8 regions; each is held out, the model is trained on the other 7 with the same
settings (no retuning), and the held-out region is scored.

| Held-out region | Countries | RF accuracy | Wind rule |
|---|---|---|---|
| Italy | ITA | **0.920** | 0.875 |
| Balkans & Greece | GRC CYP BGR HRV BIH SRB MNE ALB XKO | **0.905** | 0.896 |
| Iberia | ESP PRT | **0.901** | 0.882 |
| Central & Eastern | POL CZE SVK HUN ROU UKR BLR SVN | 0.788 | 0.930 |
| France & Benelux | FRA BEL NLD LUX CHE | 0.768 | 0.801 |
| Germany & Austria | DEU AUT | 0.613 | 0.699 |
| Nordic & Baltic | DNK NOR SWE FIN EST LVA LTU | **0.396** | 0.712 |
| British Isles | GBR IRL | **0.350** | 0.349 |
| **All regions pooled** | | **0.712** | |

- The Iberia round reproduces Step 3 exactly (0.901), confirming the setup.
- **Why the British Isles fail:** their solar farms stand at 7.7 m/s, windier than the typical
  wind farm elsewhere in Europe (7.4 m/s). The model learned absolute wind speed, so it calls them
  Wind (Solar recall 0.006). A likely reason: UK siting followed land, subsidies and demand, not wind.

### 8.5 Other continents (`src/03d_other_continents.py`)

The Europe-trained model, applied **unchanged** (no retraining). Features built exactly as in
Step 2, with distance to grid measured in an equal-area projection centred on each country.

| | USA | China | India |
|---|---|---|---|
| Cells (Solar / Wind / Hybrid) | 2,326 (855 / 1,470 / 1) | 2,300 (407 / 1,853 / 40) | 777 (208 / 566 / 3) |
| Always guess most common | 0.632 | 0.806 | 0.728 |
| **Random Forest** | **0.896** | **0.839** | **0.533** |
| Wind rule | 0.927 | 0.823 | 0.541 |
| "I don't know" | 50% | 76% | **92%** |

### 8.6 Relative features experiment (`src/03e_relative_features.py`)

Each site's wind speed and sunshine minus its country's median (no labels used).

| | Current 5 features | + relative features |
|---|---|---|
| Europe, 8 regions pooled | 0.712 | 0.733 |
| British Isles | 0.350 | 0.384 |
| Italy | 0.920 | 0.890 |
| India | 0.533 | 0.546 |

Small gains in the north, losses in the south, macro F1 slightly down → **not adopted**.

---

## 9. Step 4: explaining the model (SHAP)

Script: `src/04_explain.py`

**How SHAP works:** for one prediction, SHAP measures how much each feature pushed each class
probability up or down compared with the average prediction. It's based on Shapley values from game
theory: a fair split of the "credit" between features (Lundberg & Lee 2017). **TreeExplainer**
computes them exactly and quickly for tree models (Lundberg et al. 2020). We used 1,000 randomly
chosen training cells (seed 42).

**Mean |SHAP| (average size of each feature's effect):**

| Feature | Hybrid | Solar | Wind | Average |
|---|---|---|---|---|
| Wind speed | 0.076 | 0.237 | 0.184 | **0.166** |
| Sunshine | 0.086 | 0.071 | 0.055 | **0.070** |
| Elevation | 0.037 | 0.032 | 0.036 | 0.035 |
| Slope | 0.036 | 0.030 | 0.024 | 0.030 |
| Grid distance | 0.023 | 0.021 | 0.026 | 0.023 |

**Interpretation:**
- **Solar:** low wind speed pushes strongly towards Solar; high sunshine adds to it.
- **Wind:** the mirror image, mainly high wind speed.
- **Hybrid:** windy but *not* sunny, because 213 of 308 training Hybrids are in Germany. Iberia is
  far sunnier, so the pattern doesn't transfer (1 of 25 correct).

---

## 10. Step 5: confidence and "I don't know"

Script: `src/05_dont_know.py`

### 10.1 Confidence
The Random Forest gives three probabilities per cell, e.g. Solar 0.20 · Wind 0.59 · Hybrid 0.21.
The **prediction** is the largest (Wind); the **confidence** is its value (0.59). With three classes
the top value is always at least 0.34.

### 10.2 Rule A: low confidence
**If the top probability < 0.5 → "I don't know".** Below 0.5 means no class has a majority: the
trees disagree.

### 10.3 Rule B: unfamiliar place (area of applicability, Meyer & Pebesma 2021)
1. **Standardise:** each feature becomes a z-score using the training mean and standard deviation,
   so metres, degrees and m/s are on one scale.
2. **Weight:** multiply each z-score by the Random Forest's feature importance (wind 0.41, sunshine
   0.23, elevation 0.14, slope 0.13, grid 0.10), so differences in important features count more.
3. **Distance:** for a new cell, the straight-line distance to the **nearest training cell** in this
   weighted 5-number space.
4. **Threshold:** for every training cell, its distance to the nearest training cell **in a different
   100 km block** (the same 5 CV folds). The **95th percentile = 0.141**.
   *Why other blocks?* Otherwise each cell's nearest neighbour is the cell next to it, distances are
   near zero, and the threshold would be uselessly small.
5. **If the distance > 0.141 → "I don't know"** ("this place is unlike anything I learned from").

### 10.4 Results

| Where | "I don't know" | Accuracy on answered cells | Verdict |
|---|---|---|---|
| Iberia | 40% (10% Rule A, 32% Rule B) | 0.938 (vs 0.901 overall) | Rule A works (those cells only 47% right); Rule B too cautious (those cells 97% right) |
| India | 92% | 0.770 | Rule B saved the model from a 53% result |
| China | 76% | 0.730 (vs 0.839 overall) | Rule B did not help |
| USA | 50% | 0.871 | Mixed |

**Confidence vs "I don't know":** they aren't the same. Most Iberian abstentions came from Rule B
while the model was confident. Rule B checks *where* the model is used, not how sure it feels.

---

## 11. Maps on the website

| | Globe website |
|---|---|
| Projection | Orthographic (a globe, as seen from space) |
| Library | D3.js 7.9.0 |
| Drawing | HTML canvas (17,189 dots) |
| Borders | GADM, simplified to ~0.1° (`world.json`) |
| Data | Packed into the page by `build_site.py` from the project CSVs |
| Page | One self-contained `index.html` (HTML structure, CSS look, JavaScript behaviour), about 1.8 MB, hosted on GitHub Pages |
| Charts | Results tab charts drawn as SVG with D3 scales |
| Design | Dark theme; Solar amber, Wind teal, Hybrid purple, wrong red, "I don't know" grey; fonts Chakra Petch, Inter, JetBrains Mono |
| Accessibility | Keyboard focus, ARIA labels, reduced-motion support, phone layouts below 860 px and 600 px |

**How the globe draws:** the visible hemisphere is tested for each cell (cells on the far side are
skipped); each dot is drawn twice, first a soft glow, then a solid core in its class colour; Hybrid
cells are drawn last so they stay visible. Dragging rotates the globe; scrolling or pinching zooms;
the "fly" animation moves smoothly between views.

**Views:** what was built · Iberia test (right/wrong) · every region tested · other continents ·
improved model · archetypes · final test · wind speed.

---

## 12. Improvement round: new data and methods

### 12.1 Locked final test
Five countries never used before were chosen to cover new climates and continents (Japan, Turkey,
Chile, Australia, South Africa) and locked in `data/processed/final_test_lock.json` before any
experiment. Their base features came from the same sources as before (official per-country Global
Wind Atlas files; Global Solar Atlas; Copernicus DEM; Gridfinder). 1,062 of 1,072 cells had all
features (10 had no grid line within 20 km).

### 12.2 New features (`src/06_features_v2.py`)
| Feature | Source | Map step | Why |
|---|---|---|---|
| `wind_pct` | Global Wind Atlas 100 m | Average to ~2 km (factor 8), take the 30 km window around the cell, share of land calmer than the site | Developers choose the windiest spot locally |
| `wind_rel` | same | site wind minus the 30 km mean | same, in m/s |
| `ghi_pct`, `ghi_rel` | Global Solar Atlas | same as wind | same idea for sunshine |
| `elev_rel` | Copernicus DEM (online COG) | site height minus mean height within 2.5 km | wind farms sit on ridges |
| `rugged` | same | spread (standard deviation) of heights within 2.5 km | rough vs flat land |
| `dist_town` | GHS-SMOD (1 km, Mollweide), classes 21–30 | distance transform to the nearest town or city pixel | solar is built near demand |
| `dist_water` | GHS-SMOD class 10 | distance transform to the nearest water pixel | coastal wind |

None is changed by building a farm (no leakage). Remote DEM reads were retried three times and cached
every 100 tiles (`data/processed/terrain_context.csv`).

### 12.3 Validation and models (`src/07`, `src/10`)
- Leave-one-region-out over 11 regions; score fixed in advance = mean accuracy across regions.
- Feature sets A (5 original) to D (all 13) and E (local only); D won (RF 0.780).
- Models on D: majority, wind rule, logistic regression, Random Forest, **Extra Trees (final:
  300 trees, min_samples_leaf 2, balanced classes, seed 42)**, gradient boosting, MLP; covariate-shift
  reweighting; and an MLP on raw 11 × 11 patches of wind, sunshine and elevation (6 km spacing).

### 12.4 Conformal prediction (`src/08`)
Cross-conformal per region: for each held-out region, the threshold comes from the other regions'
held-out predictions. Score = 1 − p(true class); threshold = 90th percentile (≈ 0.7, so a class is in
the set if p ≥ 0.3). Two variants: per class (Mondrian) and overall; overall chosen (coverage 0.895,
set size 1.27, 73% single answers). Rule B (area of applicability) was dropped.

### 12.5 Patterns (`src/09`)
- Archetypes: k-means (k = 4 by silhouette, 0.15) on standardised geography only, no labels.
- SHAP: TreeExplainer on a 100-tree copy of the final model, 80 cells per region.

### 12.6 Locked test (`src/11`)
Final model trained on all 11 regions, scored once; bootstrap CIs for differences.
Final 0.892, old RF 0.896, wind rule 0.905, so a tie; conformal coverage 0.943.

---

## 13. Summary: every decision and its justification

| Decision | Justification | Evidence |
|---|---|---|
| Only real farms, no "empty land" | Every label is a fact; empty ≠ unsuitable | Design principle |
| Dunnett 2020 dataset | Peer-reviewed, global, both technologies, free | *Scientific Data* 7:130 |
| ≥ 1 MW, not urban/water | Utility-scale farms; the authors' own filter | Dataset README |
| 5 km equal-area grid | Comparable units everywhere | Eckert IV |
| Hybrid = both in one cell | Simple, reproducible; 5 km gives enough examples | Earlier check: 33 (1 km) → 409 (5 km) → 830 (10 km) |
| Europe for training | 58% of cells, 339 of 396 Hybrids | Step 1 counts |
| Iberia for testing | Connected, all classes, different climate | Map and counts |
| Spatial CV, no random split | Nearby cells are near-copies | Roberts et al. 2017 |
| Five features | Resource + terrain + access | Sachit et al. 2022; feature-subset gains |
| No land cover / roads / capacity | Leakage | They are changed by the farm itself |
| Official wind data | First file was wrong | Gujarat 4.7 vs 6.6 m/s |
| Drop islands without grid | Wrong distances (20–451 km) | Max-value check |
| Random Forest | Best of 6 models, explainable, probabilistic | CV 0.543 vs ≤ 0.542 |
| Three baselines | Prove what the model adds | Wind rule 0.882 vs RF 0.901 |
| Macro F1 + recall + CIs | Rare class; real vs chance | Bootstrap CIs |
| SHAP | Show what drives each class | Wind 0.166, sunshine 0.070 |
| "I don't know" (2 rules) | Don't guess outside experience | Meyer & Pebesma 2021; India 92% |
| Extra regions and continents | One test region isn't enough | 71% pooled; USA 90, China 84, India 53 |
| Relative features rejected | No consistent gain | 0.712 → 0.733, macro F1 down |
| Locked final test first | Honest final number | 5 countries committed before experiments |
| Local-context features | "Windier than nearby" works where absolute wind fails | 11 regions 0.736 → 0.780 (RF) |
| Extra Trees final model | Best mean-region and worst-region accuracy | 0.794 vs RF 0.780, MLP 0.763 |
| Conformal "I don't know" | Promised error rate | 0.943 coverage on locked test |
| Rule B dropped | Flagged cells were more accurate | 0.93 vs 0.79 |

**One-sentence conclusion:** geography, mostly wind speed, separates solar from wind farms well
wherever wind farms stand in much windier places than solar farms (southern Europe, the USA, largely
China) and fails where they don't (the British Isles, India); Hybrid isn't learned, and the model can
say "I don't know" when it is out of its depth. Local context (wind compared with nearby land) and
Extra Trees lift held-out accuracy from 0.74 to 0.79, mostly in the hard northern regions; on five
locked, easy countries every model scores about 0.9.
