# Which Renewable Fits Here? The complete guide, from start to end

This is the **one document that explains the whole project**: why it exists, every dataset, every map
step, every model, every check, every mistake we found and fixed, the improvement round, the final
locked test, the websites and the deliverables. For each step it says **what we did, how it works,
why we did it that way, and what came out**. Every number comes from the files in `outputs/` and
`data/processed/`.

Read it in order once. After that, use the contents to jump to a part.

---

## Contents

**Part A: the idea**
1. [The project in one minute](#1-the-project-in-one-minute)
2. [Problem statement and scope](#2-problem-statement-and-scope)
3. [The rules we set ourselves](#3-the-rules-we-set-ourselves)

**Part B: the data**
4. [The labels: Dunnett et al. 2020](#4-the-labels-dunnett-et-al-2020)
5. [The features: four map sources](#5-the-features-four-map-sources)
6. [Datasets we rejected, and why](#6-datasets-we-rejected-and-why)
7. [Maps and coordinate systems](#7-maps-and-coordinate-systems)

**Part C: the original study (steps 1 to 15)**
8. [Step 1: from farm points to labelled 5 km cells](#8-step-1-from-farm-points-to-labelled-5-km-cells)
9. [Step 2: extracting the five features](#9-step-2-extracting-the-five-features)
10. [Choosing where to train and where to test](#10-choosing-where-to-train-and-where-to-test)
11. [Step 3: baselines, the Random Forest and spatial cross-validation](#11-step-3-baselines-the-random-forest-and-spatial-cross-validation)
12. [Result on Spain and Portugal](#12-result-on-spain-and-portugal)
13. [Extra checks: is the result real?](#13-extra-checks-is-the-result-real)
14. [Every European region held out in turn](#14-every-european-region-held-out-in-turn)
15. [Other continents: USA, China, India](#15-other-continents-usa-china-india)
16. [Step 4: SHAP, what drives the model](#16-step-4-shap-what-drives-the-model)
17. [Step 5: the first "I don't know"](#17-step-5-the-first-i-dont-know)
18. [First attempt at a fix: relative features](#18-first-attempt-at-a-fix-relative-features)
19. [Comparing with another approach](#19-comparing-with-another-approach)

**Part D: the improvement round (steps 16 to 20)**
20. [Why an improvement round, and its rules](#20-why-an-improvement-round-and-its-rules)
21. [Locking the final test](#21-locking-the-final-test)
22. [Eight new local-context features](#22-eight-new-local-context-features)
23. [Which features help?](#23-which-features-help)
24. [Which model? Trees, neural networks and reweighting](#24-which-model-trees-neural-networks-and-reweighting)
25. [Before and after, region by region](#25-before-and-after-region-by-region)
26. [What drives the improved model](#26-what-drives-the-improved-model)
27. [A better "I don't know": conformal prediction](#27-a-better-i-dont-know-conformal-prediction)
28. [Four types of place: siting archetypes](#28-four-types-of-place-siting-archetypes)
29. [The locked final test](#29-the-locked-final-test)
30. [Overfitting of the final model](#30-overfitting-of-the-final-model)

**Part E: how it ended**
31. [Final answers to the research question](#31-final-answers-to-the-research-question)
32. [Using the model on a new location: the assumption](#32-using-the-model-on-a-new-location-the-assumption)
33. [Limitations](#33-limitations)
34. [Problems we found and fixed](#34-problems-we-found-and-fixed)
35. [The website](#35-the-website)
36. [Report, slides, documents and GitHub](#36-report-slides-documents-and-github)
37. [How to reproduce everything](#37-how-to-reproduce-everything)
38. [Every decision and its justification](#38-every-decision-and-its-justification)
39. [References and why each was used](#39-references-and-why-each-was-used)
40. [Glossary](#40-glossary)
41. [The whole story in 20 lines](#41-the-whole-story-in-20-lines)

---

# Part A: the idea

## 1. The project in one minute

We asked a simple question: **if you only know the geography of a place (how sunny, how windy, how
high, how steep, how far from the power grid), can you tell whether the renewable farm built there
is Solar, Wind, or both (Hybrid)?**

We took about 24,000 real solar and wind farms from OpenStreetMap, put them on a 5 km grid, and
described each grid cell with map data. We trained models on Europe and tested them on regions they
had **never seen**: first Spain and Portugal (90% correct), then every European region in turn,
then the USA, China and India. It works where wind farms stand in much windier places than solar
farms, and fails where they don't (the British Isles, India).

Then we ran an **improvement round**: we locked five untouched countries as a final test, added
*local context* (is this site windier than the land around it?), switched to Extra Trees and added
a properly measured "I don't know" (conformal prediction). Held-out accuracy on 11 regions rose from
**73.6% to 79.4%**, the hardest region (British Isles) from **35% to 54%**. On the five locked
countries every model, even a one-line wind rule, scored about **89 to 91%**, because those
countries are easy; the "I don't know" sets kept the true answer **94%** of the time.

## 2. Problem statement and scope

> Given a place where an onshore, utility-scale renewable farm already exists, can a model trained
> on geographic factors predict whether it is **Solar, Wind or Hybrid**, including in a region it has
> never seen, and say **"I don't know"** when it is unsure?

**In scope**
- Onshore, utility-scale farms (at least 1 MW), solar and wind.
- Three classes: Solar, Wind, Hybrid (both in the same 5 km cell).
- Geography only: resource, terrain, grid access (and later, local context).
- Testing on whole regions the model never saw.

**Out of scope, on purpose**
- Offshore wind, rooftop solar (different problems).
- Hydro, biomass and other technologies.
- **Suitability.** The model never says *whether* a place should be developed. It answers "if
  something is built here, which type fits?". It has never seen empty land, so it cannot judge a
  forest, a city or a lake.

**Why this question is interesting:** planners and students usually build suitability maps from
expert weights (MCDA/AHP). Real farms are a record of real decisions made by developers who studied
the site. Learning from those decisions is data-driven, and testing on unseen regions tells us
whether the lessons travel.

## 3. The rules we set ourselves

These came from the project brief and never changed:

1. **Simple, readable code**: one script per step, numbered in order.
2. **No random train/test splits.** Two cells 5 km apart are nearly identical; a random split puts
   near-copies of training cells into the test and inflates the score (Roberts et al. 2017).
3. **No tuning on the test set.** The test region is used once, at the end.
4. **No leaky features**: no land cover, roads or capacity (Section 6).
5. **Always compare with simple baselines**, so we know what the model really adds.
6. **Report failures honestly** and explain them.
7. **Ask when unclear** (for example, the train/test split was confirmed before modelling).
8. Fixed random seed (`random_state = 42`) so every run is reproducible.

---

# Part B: the data

## 4. The labels: Dunnett et al. 2020

| Item | Detail |
|---|---|
| Citation | Dunnett, S., Sorichetta, A., Taylor, G., Eigenbrod, F. (2020). Harmonised global datasets of wind and solar farm locations and power. *Scientific Data* 7, 130 |
| Source | figshare, doi 10.6084/m9.figshare.11310269 (`global_wind_solar_2020.zip`) |
| Built from | OpenStreetMap (2020 snapshot), harmonised and quality-checked by the authors |
| Files | `global_solar_2020_WGS84.gpkg` (35,272 records), `global_wind_2020_WGS84.gpkg` (33,514 records) |
| Total | **68,786 records** |
| Useful columns | `GID_0` country · `panels` / `turbines` · `panel.area`, `landscape.area` · `water` (1 = on water) · `urban` (1 = in a city) · `power` (estimated MW) |

**How it works:** OpenStreetMap volunteers map solar panels and wind turbines. A wind farm is a
*multipoint* (one point per turbine). The authors cleaned the data and estimated power for real farms.

**Why we chose it**
- **Every label is a fact**: each record is something that was actually built.
- **Peer-reviewed** (Nature-group journal); capacity was cross-checked against national statistics.
- **Both technologies in one harmonised format**, so the model learns differences between *sites*,
  not between two data sources.
- **Global**, which made the USA, China, India and the five locked countries possible.
- **Free and reproducible.**

**Known weaknesses:** OpenStreetMap coverage is uneven (strong in Europe, North America and East
Asia; weak in India and Africa); it is a 2020 snapshot; capacities are estimates.

## 5. The features: four map sources

| Feature | Source | What it measures | Resolution | Why |
|---|---|---|---|---|
| `ghi` (kWh/m²/day) | **Global Solar Atlas 2.0** (Solargis for the World Bank, ESMAP) | Global horizontal irradiation, long-term average | 9 arc-seconds (~250 m), 60°S to 65°N | The standard solar resource measure; the first thing a solar developer checks |
| `wind100` (m/s) | **Global Wind Atlas** (DTU Wind Energy for the World Bank; Davis et al. 2023) | Mean wind speed at 100 m | 250 m, one official file per country | 100 m is about the hub height of modern turbines; the first thing a wind developer checks |
| `elev` (m), `slope` (°) | **Copernicus DEM GLO-90** (ESA, TanDEM-X) | Height above sea level; steepness | 3 arc-seconds (~90 m), 1°×1° tiles | Solar needs flat land; wind farms like exposed high ground |
| `dist_grid` (km) | **Gridfinder** (Arderne et al. 2020) | Distance to the nearest power line | Vector lines (OSM + lines predicted from night-time lights) | A farm must connect; long connections are expensive |

Added in the improvement round (Section 22):

| Feature | Source | Why |
|---|---|---|
| `dist_town`, `dist_water` | **GHS Settlement Model (GHS-SMOD)**, European Commission JRC, 1 km grid | Solar is built near demand; wind often near coasts |
| `wind_pct`, `wind_rel`, `ghi_pct`, `ghi_rel`, `elev_rel`, `rugged` | Same Wind/Solar Atlas and DEM, read around each site | Local context: "windier / higher than nearby" |

**Borders (GADM)** were used only to draw the website maps. The model never used them.

**How the DEM was read without downloading it:** the Copernicus DEM is stored on AWS as
*cloud-optimised GeoTIFFs*. A program can request just the few pixels it needs over the internet
(`/vsicurl/`). For each farm we read a 3×3 pixel window, about 0.1% of a tile, which avoided
downloading about 3.3 GB (700 tiles).

## 6. Datasets we rejected, and why

| Dataset | Why not |
|---|---|
| Land cover (ESA WorldCover, Copernicus) | **Leakage**: 2020 imagery shows the farm itself (a solar farm *is* "built-up / bare" land) |
| Roads (GRIP, OpenStreetMap) | **Leakage**: farms build their own access roads |
| Plant capacity (in Dunnett) | **Leakage**: size reveals the type; used only as the at-least-1 MW filter |
| Global Energy Monitor trackers | Different size thresholds for solar (1 MW) and wind (10 MW); needs sign-up; kept as a possible cross-check |
| National registers (UK REPD, US wind database) | One country each, all different formats |
| Expert suitability maps (MCDA/AHP) | Opinions about where farms *could* go, not where they *are* |
| "Empty land" as a negative class | Empty does not mean unsuitable (it may be protected, private or simply not yet built) |

**Leakage** means a feature that contains the answer because of the thing we are predicting. A
model with leaky features scores very high in testing and is useless for a new site.

## 7. Maps and coordinate systems

A coordinate reference system (CRS) decides how the round Earth becomes numbers. Different jobs need
different ones.

| CRS | Used for | Why |
|---|---|---|
| **WGS84** (EPSG:4326), degrees | Reading all inputs, sampling rasters, storing cell centres | Every source comes in it |
| **World Eckert IV** (ESRI:54012), metres, equal-area | Building the 5 km grid | Every cell covers the same area anywhere on Earth, so Norway and Spain cells are comparable |
| **ETRS89 / LAEA Europe** (EPSG:3035), metres, equal-area | Distance to grid in Europe; 100 km cross-validation blocks | The European standard; accurate distances and areas |
| **Lambert azimuthal equal-area centred on each country** | Distance to grid in the USA, China, India and the locked countries | Same idea as EPSG:3035, centred where it is needed |
| **Mollweide** (equal-area) | GHS-SMOD towns and water (improvement round) | The settlement map is published in it; 1 km pixels make distances simple |
| **Orthographic** (globe website) | Drawing the website map | Display only |

**Why not work in degrees?** One degree of longitude is 111 km at the equator but about 56 km in
Norway. Distances and areas in degrees would be wrong, and grid cells would shrink towards the poles.

---

# Part C: the original study (steps 1 to 15)

| Step | Script | Output |
|---|---|---|
| 1. Labels | `src/01_prepare_labels.py` | `data/processed/cells.csv`, `points.csv` |
| 2. Features | `src/02_extract_features.py` | `data/processed/features.csv` |
| 3. Models | `src/03_train_evaluate.py` | `outputs/results.csv`, `cv_scores.csv`, `rf_model.joblib`, confusion matrix |
| 3b. Other models, subsets | `src/03b_extra_checks.py` | `extra_model_comparison.csv`, `feature_ablation.csv` |
| 3c. Regions, overfitting | `src/03c_region_holdout.py` | `region_holdout.csv`, `overfit_check.csv` |
| 3d. Other continents | `src/03d_other_continents.py` | `other_continents.csv` |
| 3e. Relative features | `src/03e_relative_features.py` | `relative_features_experiment.csv` |
| 4. Explain | `src/04_explain.py` | SHAP plots, `feature_importance.csv` |
| 5. "I don't know" | `src/05_dont_know.py` | `dont_know_summary.csv`, `test_predictions.csv` |

## 8. Step 1: from farm points to labelled 5 km cells

### 8.1 Keeping only real farms

| Rule | How | Why |
|---|---|---|
| Drop `water == 1` | Removes offshore and floating records | Offshore wind is out of scope |
| Drop `urban == 1` | Removes records inside cities | Mostly rooftop solar, out of scope |
| Keep `power ≥ 1 MW` | Removes small installations **and** records with no power value | Utility-scale only |

**Why "no power value" is dropped:** two-thirds of records had no power. The dataset's README says
the authors estimated power **only for real farms** (wind: more than 4 turbines; solar: more than
1 ha of panels, outside cities). A missing value therefore means "not a farm". This is the authors'
own filter, not our guess.

**Result:** solar 35,272 → **11,498**; wind 33,514 → **12,580**; total **24,078 farms**.

### 8.2 Building the grid
1. Each wind farm (many turbine points) is reduced to its **centroid**.
2. All points are projected to **Eckert IV** (metres).
3. Each point gets a cell: `cx = floor(x / 5000)`, `cy = floor(y / 5000)`, a fixed 5 km × 5 km grid.
4. Each cell is labelled **Solar** (only solar), **Wind** (only wind) or **Hybrid** (both).
5. The cell's country is the most common country of its points.

**Why 5 km:** small enough to keep local conditions, large enough to group one farm's turbines.

**Why Hybrid is defined this way:** it is simple and reproducible. A sensitivity check counted cells
with both technologies at several sizes: **33 at 1 km, 107 at 2 km, 409 at 5 km, 830 at 10 km.** So
Hybrid depends strongly on cell size (a stated limitation); 5 km is the smallest size with enough
examples (the brief required at least 300 Hybrid cells to keep it as a class).

**Result:** **18,705 labelled cells** worldwide (Wind 9,609 · Solar 8,700 · Hybrid 396). Largest
countries: Germany 3,054 · USA 2,342 · China 2,313 · Italy 1,644 · Spain 1,372 · UK 1,024.

## 9. Step 2: extracting the five features

### 9.1 Sampling at each farm, then averaging per cell
For every farm point in the study regions (13,457 points in Europe) we read each raster at the
point's exact pixel, then **averaged per cell**.

**Why at the farm and not over the whole cell?** A 5 km cell contains valleys, villages and hills.
The farm sits on one specific spot, and that spot is what the developer chose.

### 9.2 How each feature is computed
- **Sunshine (`ghi`)**: the pixel value from the Global Solar Atlas.
- **Wind (`wind100`)**: the pixel value from the farm's **own country** Global Wind Atlas file. Files
  are cut at borders, so a farm right on a border may fall outside; then a neighbour's file is used.
- **Elevation (`elev`)**: the centre of a 3×3 DEM window.
- **Slope (`slope`)**: Horn's (1981) method, the GIS standard:

```
a b c        dz/dx = ((c + 2f + i) − (a + 2d + g)) / (8 · Δx)
d e f        dz/dy = ((g + 2h + i) − (a + 2b + c)) / (8 · Δy)
g h i        slope = arctan( √((dz/dx)² + (dz/dy)²) )   in degrees
```
  with Δy = pixel height in degrees × 111,320 m and Δx = pixel width × 111,320 × cos(latitude)
  (longitude degrees shrink towards the poles).
- **Distance to grid (`dist_grid`)**: cell centre and Gridfinder lines projected to EPSG:3035; the
  nearest line found with a spatial nearest-neighbour join; converted to km.
- Each cell also gets a **`block_id`**: the 100 km × 100 km block it lies in, for spatial cross-validation.

### 9.3 Cleaning

| Dropped | Cells | Reason |
|---|---|---|
| Far north | 34 | Above ~65°N, no Global Solar Atlas data |
| Azores and Madeira | 13 | Not covered by the official Portugal wind file |
| Islands without grid lines | 20 | Nearest Gridfinder line 20–451 km away on another coast (15 Canary Islands, 5 small Mediterranean islands); every cell above 20 km was such an island, so >20 km = "no coverage" |
| Kosovo | 1 | No Global Wind Atlas country file |
| **Total** | **68** | |

**Final European dataset:** `features.csv`, **10,724 cells**.

### 9.4 What the data looks like (medians)

| | Sunshine (kWh/m²/day) | Elevation (m) |
|---|---|---|
| Training region (Europe without Iberia) | 3.1 | 148 |
| Test region (Iberia) | 4.6 | 717 |

| Wind speed (m/s) | Solar cells | Wind cells |
|---|---|---|
| Training region | 5.7 | 7.4 |
| British Isles | **7.7** | 9.1 |
| USA | 5.7 | 8.1 |
| China | 5.3 | 7.7 |
| India | 5.5 | 6.4 |

These medians explain almost every later result: **the model works where wind farms are much windier
than solar farms.** British solar farms (7.7 m/s) are windier than the typical European wind farm
(7.4 m/s), and Indian wind farms (6.4 m/s) are barely windier than Indian solar farms.

## 10. Choosing where to train and where to test

| Region | Cells | Hybrid cells |
|---|---|---|
| **Europe** | **10,792** | **339** |
| USA | 2,342 | 1 |
| China | 2,313 | 40 |
| India | 777 | 3 |

- **Training:** Europe without Spain and Portugal, **9,176 cells** (Solar 5,320 · Wind 3,548 · Hybrid
  308), 33 countries.
- **Test:** Spain + Portugal, **1,548 cells** (Solar 758 · Wind 765 · Hybrid 25).

**Why Europe:** 58% of all cells and **339 of the 396 Hybrid cells**. It is the only place where all
three classes can be trained *and* whole regions held out.

**Why Iberia:** one connected block touching the training region only at the Pyrenees, it contains
all three classes, and its climate is different (sunnier, higher). That is a fair but hard test.

**Why no random split:** spatial autocorrelation. Nearby cells are near-copies, so random splits
measure memory, not generalisation (Roberts et al. 2017).

## 11. Step 3: baselines, the Random Forest and spatial cross-validation

### 11.1 The four models

| Model | How it works | Why it is there |
|---|---|---|
| **Majority class** | Always predicts the most common training class (Solar) | The minimum any model must beat |
| **Wind-speed rule** | Wind if `wind100 ≥ t`, else Solar. `t` chosen from 97 candidate thresholds (2nd–98th percentile) by best macro F1 on training data: **t = 6.60 m/s** | The key baseline: if a model cannot beat this, it has only learned "windy = wind" |
| **Logistic regression** | Standardised features, linear model, balanced class weights | Simple standard ML; shows whether non-linearity is needed |
| **Random Forest** | 300 decision trees on random samples; outputs averaged | Main model |

### 11.2 How a Random Forest works
1. A decision tree asks yes/no questions on one feature at a time ("is wind above 6.8 m/s?"),
   choosing the split that separates the classes best.
2. Each tree sees a random **bootstrap** sample of cells and, at each split, a random subset of
   features, so trees differ.
3. A tree's answer is the class shares in the leaf the cell lands in.
4. The forest averages 300 trees → three probabilities (Solar, Wind, Hybrid).
5. `class_weight="balanced"` gives rare classes more weight (Hybrid is 3% of training cells).

### 11.3 Why the Random Forest
- Standard, robust model for tabular data (Breiman 2001).
- Captures interactions such as "windy *and* not sunny" (logistic regression scored lower in CV:
  0.477 vs 0.543).
- No scaling needed; robust to outliers.
- Gives probabilities, which makes "I don't know" possible.
- Explainable with SHAP.
- Used in published siting research (Sachit et al. 2022).
- **No other standard model did better** (Section 13.1).

### 11.4 Tuning with spatial cross-validation
- `GroupKFold(5)` with `groups = block_id`: each fold holds out whole 100 km blocks, mimicking "a new
  place". Training region only; Iberia never touched.
- Grid: `max_depth ∈ {None, 10, 20}` × `min_samples_leaf ∈ {1, 5}`, scored by macro F1.
- **Selected: max_depth = 10, min_samples_leaf = 1 (CV macro F1 0.543).**
- Fitted on the full training region and **evaluated once** on Iberia.

| Model | Spatial CV macro F1 |
|---|---|
| Majority | 0.244 |
| Wind rule | 0.498 |
| Logistic regression | 0.477 |
| **Random Forest** | **0.543** |

### 11.5 Metrics, and why each

| Metric | Meaning | Why |
|---|---|---|
| Accuracy | Share of cells right | Easy to read, but hides the rare class |
| **Macro F1** | Average F1 of Solar, Wind, Hybrid, equally weighted | Exposes poor Hybrid performance |
| Recall per class | Of the real Solar cells, how many found | Shows which class fails |
| Precision (Hybrid) | Of the predicted Hybrids, how many right | Catches over-guessing |
| Confusion matrix | True × predicted | Shows which mistakes happen |
| **Bootstrap 95% CI** | Resample test cells 2,000 times; 2.5th–97.5th percentiles | Real difference or chance? (Efron & Tibshirani 1993) |
| Paired bootstrap difference | Same resamples for two models | Fair model comparison |

## 12. Result on Spain and Portugal

| Model | Accuracy (95% CI) | Macro F1 | Recall Solar | Recall Wind | Recall Hybrid |
|---|---|---|---|---|---|
| Majority | 0.490 | 0.219 | 1.000 | 0.000 | 0.00 |
| Wind rule | 0.882 (0.866–0.899) | 0.593 | 0.926 | 0.868 | 0.00 |
| Logistic regression | 0.844 | 0.641 | 0.836 | 0.860 | 0.56 (precision 0.07) |
| **Random Forest** | **0.901 (0.886–0.915)** | 0.627 | 0.918 | 0.911 | 0.04 (1 of 25) |

- Random Forest beats the wind rule by **+1.8 points (CI +0.5 to +3.2)**: real but small.
- Logistic regression's Hybrid "recall" came from guessing Hybrid 199 times with only 14 right.

**Confusion matrix (Random Forest, Iberia)**

| True ↓ / Predicted → | Solar | Wind | Hybrid |
|---|---|---|---|
| Solar | **696** | 59 | 3 |
| Wind | 65 | **697** | 3 |
| Hybrid | 6 | 18 | **1** |

## 13. Extra checks: is the result real?

### 13.1 Other classifiers (training-region spatial CV)

| Model | CV macro F1 |
|---|---|
| **Random Forest** | **0.543** |
| Gradient boosting | 0.542 |
| Extra Trees | 0.537 |
| k-nearest neighbours (k = 15) | 0.510 |
| Logistic regression | 0.477 |
| SVM (RBF) | 0.476 |

The limit was the **data**, not the algorithm.

### 13.2 Feature subsets

| Features | CV macro F1 |
|---|---|
| Wind only | 0.503 |
| + sunshine | 0.524 |
| + elevation, slope | 0.540 |
| + grid distance (all 5) | 0.543 |

Wind does most of the work. Grid distance adds little, which also shows it doesn't leak the answer.

### 13.3 Overfitting

| Random Forest scored on | Accuracy | Macro F1 |
|---|---|---|
| Training data (seen) | 0.812 | 0.701 |
| Unseen 100 km blocks | 0.732 | 0.545 |
| Iberia | 0.901 | 0.627 |

A forest with no depth limit scores 1.000 on its training data (memorisation); ours (depth 10) scores
0.812 → **no serious overfitting**. Iberia scores higher than CV because it is easier: the wind rule
also jumps from 0.74 (training region) to 0.88 (Iberia).

## 14. Every European region held out in turn

Europe split into 8 regions; each held out, model trained on the other 7 with the same settings.

| Held-out region | Countries | RF | Wind rule |
|---|---|---|---|
| Italy | ITA | **0.920** | 0.875 |
| Balkans & Greece | GRC CYP BGR HRV BIH SRB MNE ALB XKO | **0.905** | 0.896 |
| Iberia | ESP PRT | **0.901** | 0.882 |
| Central & Eastern | POL CZE SVK HUN ROU UKR BLR SVN | 0.788 | 0.930 |
| France & Benelux | FRA BEL NLD LUX CHE | 0.768 | 0.801 |
| Germany & Austria | DEU AUT | 0.613 | 0.699 |
| Nordic & Baltic | DNK NOR SWE FIN EST LVA LTU | **0.396** | 0.712 |
| British Isles | GBR IRL | **0.350** | 0.349 |
| **Pooled** | | **0.712** | |

**Why the British Isles fail:** their solar farms stand at 7.7 m/s, windier than the typical
European wind farm. The model learned *absolute* wind speed, so it calls them Wind (Solar recall
0.006). UK siting followed land, subsidies and demand, not wind. **Lesson: one test region is not
enough; Iberia alone made the model look better than it is.**

## 15. Other continents: USA, China, India

The Europe-trained model applied **unchanged**.

| | USA | China | India |
|---|---|---|---|
| Cells (Solar / Wind / Hybrid) | 2,326 (855 / 1,470 / 1) | 2,300 (407 / 1,853 / 40) | 777 (208 / 566 / 3) |
| Always guess most common | 0.632 | 0.806 | 0.728 |
| **Random Forest** | **0.896** | **0.839** | **0.533** |
| Wind rule | 0.927 | 0.823 | 0.541 |
| "I don't know" | 50% | 76% | **92%** |

- **USA works**: wind farms are much windier than solar farms there (8.1 vs 5.7 m/s).
- **China partly**: 84%, but only 3 points above always guessing Wind (81% of cells are wind). Hybrid
  0 of 40.
- **India fails**: Indian wind farms (6.4 m/s) are calmer than European ones. But the model said
  "I don't know" for 92% of Indian cells: it knew it was out of its depth.

## 16. Step 4: SHAP, what drives the model

**How SHAP works:** for each prediction it splits the "credit" fairly between features (Shapley
values from game theory, Lundberg & Lee 2017). TreeExplainer computes this exactly for tree models
(Lundberg et al. 2020). We used 1,000 random training cells.

| Feature | Mean \|SHAP\| (average over classes) |
|---|---|
| Wind speed | **0.166** |
| Sunshine | 0.070 |
| Elevation | 0.035 |
| Slope | 0.030 |
| Grid distance | 0.023 |

- Low wind pushes towards Solar; high wind towards Wind. Sunshine adds to Solar.
- **Hybrid = "windy but not sunny"**, because 213 of 308 training Hybrids are in Germany. Iberia is
  far sunnier, so that pattern can't transfer (1 of 25 correct).
- Wind speed carried **about half** of the model's reasoning, which is the root of the northern failures.

## 17. Step 5: the first "I don't know"

**Confidence** = the highest of the three probabilities (at least 0.34 with three classes).

- **Rule A, low confidence:** top probability < 0.5 → "I don't know" (the trees disagree).
- **Rule B, unfamiliar place** (area of applicability, Meyer & Pebesma 2021): standardise features,
  weight them by RF importance (wind 0.41, sunshine 0.23, elevation 0.14, slope 0.13, grid 0.10),
  measure the distance to the nearest training cell; the threshold is the 95th percentile of the same
  distance for training cells measured against *other* 100 km blocks (**0.141**). Above it → "I don't know".

| Where | "I don't know" | Accuracy on answered cells | Verdict |
|---|---|---|---|
| Iberia | 40% (10% A, 32% B) | 0.938 (vs 0.901) | Rule A works (those cells ~47% right); Rule B too cautious (those cells 97% right) |
| India | 92% | 0.770 | Rule B saved a 53% result |
| China | 76% | 0.730 (vs 0.839) | Rule B did not help |
| USA | 50% | 0.871 | Mixed |

*Later replaced by conformal prediction (Section 27); Rule B was dropped.*

## 18. First attempt at a fix: relative features

Each site's wind and sunshine minus its **country's median** (`src/03e`).

| | 5 features | + relative |
|---|---|---|
| 8 regions pooled | 0.712 | 0.733 |
| British Isles | 0.350 | 0.384 |
| Italy | 0.920 | 0.890 |
| India | 0.533 | 0.546 |

Small gains in the north, losses in the south, macro F1 slightly down → **not adopted**. The lesson
pointed the way: a country is too big; the comparison should be with the **land around the site**.
That became the improvement round.

## 19. Comparing with another approach

A different project predicted "farm vs random land" using land cover and scored very high. Its top
solar feature was land cover, which shows the farm itself: **leakage**. Our design gives lower but
trustworthy numbers, because every feature exists before the farm.

---

# Part D: the improvement round (steps 16 to 20)

## 20. Why an improvement round, and its rules

**Goal:** better generalisation to unseen regions, less dependence on absolute wind speed, more
meaningful features, a better "I don't know", and a check whether deep learning helps.

**Rules, fixed before any experiment**
1. **Lock a final test first** (Section 21) and never look at it until the end.
2. **Development = leave-one-region-out (LORO) over 11 regions**: the 8 European regions + USA, China,
   India (16,127 cells: Wind 8,202 · Solar 7,548 · Hybrid 377). Each region is predicted by a model
   trained on the other 10.
3. **Selection score fixed in advance: mean accuracy across regions** (rewards working everywhere,
   not just in the biggest regions); macro F1 as tie-breaker.
4. No change after seeing the locked test.

**Why LORO over 11 regions instead of the single Iberia test?** Iberia turned out to be easy. Eleven
held-out regions, including the hard ones, give a much more honest picture.

## 21. Locking the final test

- **Countries:** Japan, Turkey, Chile, Australia, South Africa: new continents and climates, never used.
- **Cells:** 1,072 (Solar 684 · Wind 381 · Hybrid 7). 1,062 had all features (10 had no grid line
  within 20 km).
- Locked in `data/processed/final_test_lock.json` and **committed to git before any result**
  (commit "Lock untouched final test countries"), so anyone can check the order.
- Their base features came from the same sources (official per-country wind files for JPN, TUR, CHL,
  AUS, ZAF).

## 22. Eight new local-context features

Script: `src/06_features_v2.py` → `data/processed/features_v2.csv` (17,199 cells, no missing values).

| Feature | How it is computed | Why |
|---|---|---|
| `wind_pct` | Wind Atlas averaged to ~2 km (factor 8); within a **30 km** window, the share of land calmer than the site | Developers pick the windiest spot **locally**; works even where all land is windy |
| `wind_rel` | Site wind minus the 30 km mean (m/s) | Same idea in m/s |
| `ghi_pct`, `ghi_rel` | Same for sunshine | Same idea for solar |
| `elev_rel` | Site height minus the mean height within **2.5 km** (Copernicus DEM) | Wind farms sit on ridges and hilltops |
| `rugged` | Standard deviation of heights within 2.5 km | Rough vs flat land |
| `dist_town` | GHS-SMOD 1 km: distance transform to the nearest town or city pixel (classes 21–30) | Solar is built near demand |
| `dist_water` | GHS-SMOD class 10: distance to the nearest sea or large lake | Coastal wind |

**Distance transform** (scipy `distance_transform_edt`): for every pixel, the straight-line distance
to the nearest "town" (or "water") pixel, computed for the whole map at once.

**Leakage check:** none of these changes when a farm is built (wind, sun, terrain, towns, sea).

**What the features show (medians, development data)**

| | Solar | Wind | Hybrid |
|---|---|---|---|
| `wind_pct` (local wind percentile) | 0.44 | **0.84** | 0.67 |
| `elev_rel` (m above surroundings) | −1.8 | **+15.3** | +2.9 |
| `dist_town` (km) | **5.4** | 12.2 | 7.1 |

Wind farms sit in the **windiest 16% of their surroundings**, on raised ground, away from towns.
Solar farms sit at an average spot, slightly low, close to towns.

**Engineering details:** the remote DEM reads sometimes failed, so each read is retried 3 times and
results are cached every 100 tiles (`terrain_context.csv`).

## 23. Which features help?

Random Forest, LORO over 11 regions (`src/07_global_experiments.py`):

| Feature set | n | Accuracy | Macro F1 | Mean region | Worst region |
|---|---|---|---|---|---|
| A. original 5 | 5 | 0.736 | 0.536 | 0.706 | 0.354 (British Isles) |
| B. + local wind/sun | 9 | 0.766 | 0.547 | 0.737 | 0.440 (British Isles) |
| C. + terrain context | 11 | 0.774 | 0.553 | 0.743 | 0.442 (Nordic & Baltic) |
| **D. + town/water (all 13)** | 13 | **0.780** | 0.552 | **0.756** | **0.494** (British Isles) |
| E. D without absolute wind/sun | 11 | 0.724 | 0.516 | 0.712 | 0.448 (Nordic & Baltic) |

- **Every group helps**; set D is chosen.
- Set E (no absolute wind) is worse overall, but lifts the British Isles to 0.754: in windy countries
  "windier than nearby" matters more than "windy".

## 24. Which model? Trees, neural networks and reweighting

All on the 13 features, LORO over 11 regions (`src/07`, `src/10`):

| Model | Accuracy | Macro F1 | Mean region | Worst region |
|---|---|---|---|---|
| Majority class | 0.419 | 0.220 | 0.443 | 0.147 |
| Wind-speed rule | 0.776 | 0.522 | 0.761 | 0.349 |
| Logistic regression | 0.670 | 0.530 | 0.634 | 0.230 |
| Random Forest | 0.780 | 0.552 | 0.756 | 0.494 |
| **Extra Trees (final)** | **0.794** | 0.536 | **0.781** | **0.536** |
| Gradient boosting | 0.744 | 0.553 | 0.718 | 0.428 |
| Neural network (MLP) | 0.763 | 0.516 | 0.752 | 0.492 |
| Neural network on raw map patches | 0.738 | 0.506 | 0.721 | 0.434 |
| RF + covariate-shift reweighting | 0.779 | 0.548 | 0.755 | 0.534 |

**Extra Trees (extremely randomised trees, Geurts et al. 2006)** is like a Random Forest, but at each
split it picks the cut-point **at random** instead of searching for the best one, then averages many
such trees. The extra randomness makes it less likely to learn region-specific details. Settings:
300 trees, `min_samples_leaf = 2`, balanced class weights, seed 42. It won on the pre-set score
(mean region 0.781) and on the worst region (0.536), so it became the **final model**.

**Neural networks (did deep learning help? No.)**
- **MLP** on the 13 features: 0.763.
- **Patch network**: instead of hand-made features, an MLP was given raw **11 × 11 patches** of wind,
  sunshine and elevation around each site (6 km spacing, 247 inputs) to learn local context by
  itself. It scored 0.738. It did learn some context (British Isles 0.434 vs 0.354 before), but trees
  on good features win on ~16,000 tabular rows.

**Covariate-shift reweighting (Shimodaira 2000):** training cells weighted to resemble the held-out
region. No gain (0.779 vs 0.780): the problem is that the *relationship* between geography and
technology differs between regions, not just the mix of places. Not adopted.

**Why the wind rule is still close (0.776):** wind remains the main signal; the gains come from the
hard regions where the rule fails.

**Option A (what we chose for the final story):** keep the original study as the main story, add the
improvement round as the final chapter, **Extra Trees as the final model**, and present the
depth-limited Random Forest (0.780) as the simpler alternative.

## 25. Before and after, region by region

Original 5-feature RF vs final model, both trained on the other 10 regions:

| Held-out region | Before | After | Change (points) |
|---|---|---|---|
| British Isles | 0.354 | **0.536** | **+18** |
| India | 0.607 | 0.611 | +0 |
| Nordic & Baltic | 0.364 | **0.643** | **+28** |
| Germany & Austria | 0.602 | 0.691 | +9 |
| France & Benelux | 0.635 | 0.733 | +10 |
| China | 0.849 | 0.822 | −3 |
| Central & Eastern | 0.752 | **0.890** | **+14** |
| Iberia | 0.893 | 0.907 | +1 |
| Balkans & Greece | 0.905 | 0.909 | +0 |
| Italy | 0.899 | 0.920 | +2 |
| USA | 0.904 | 0.926 | +2 |
| **All 11, pooled** | **0.736** | **0.794** | **+6** |
| Mean across regions | 0.706 | 0.781 | +8 |
| Worst region | 0.354 | 0.536 | +18 |

(The "before" numbers differ slightly from Section 14 because here each model is trained on 10
regions including the USA, China and India, not on 7 European regions.)

**The gain is exactly where the original model failed** (north and centre of Europe); the easy south
barely changes; China drops 3 points; India stays hard.

## 26. What drives the improved model

SHAP (TreeExplainer on a 100-tree copy of the final model, 80 cells per region; `src/09_patterns.py`):

| Feature | Share of total importance |
|---|---|
| Wind speed (absolute) | **23%** |
| Local wind percentile (new) | 17% |
| Sunshine | 10% |
| Wind vs local mean (new) | 7% |
| Distance to town (new) | 7% |
| Height vs surroundings (new) | 6% |
| Elevation | 5% |
| Distance to grid | 5% |
| Slope, ruggedness, distance to water, sunshine percentile, sunshine vs local mean | 3–4% each |

- **Absolute wind fell from about half of the reasoning to 23%.**
- In the British Isles the **local wind percentile** is the most important feature.
- A dependence plot by region (`outputs/v2_shap_wind_dependence.png`) shows the same wind speed pushes
  towards Wind by different amounts in different regions.

## 27. A better "I don't know": conformal prediction

Script: `src/08_conformal.py`.

**Problem with the old rules:** the 0.5 threshold was arbitrary, and Rule B flagged cells that were
mostly right. We wanted an "I don't know" with a **promised error rate**.

**How split conformal prediction works (Angelopoulos & Bates 2021)**
1. Take held-out predictions (from LORO). For each cell, the **score** = 1 − probability the model
   gave to the **true** class (a big score means the model was surprised).
2. The threshold = the **90th percentile** of these scores (target error α = 0.10). Here it is
   **0.70**.
3. For a new place, every class whose score is below the threshold enters the **answer set**, which
   here means **every class with probability ≥ 0.30**.
4. One class in the set = a confident answer. Two classes (e.g. "Solar or Wind") = **"I don't know"**.
5. Guarantee: the true class is in the set about 90% of the time, **if new places resemble the
   calibration places**.

**Cross-conformal per region:** for each held-out region the threshold is computed from the *other*
regions' predictions, so the region being scored never sets its own threshold.

| Calibration | Coverage (true answer in set) | Average set size | Single answer | Right when single |
|---|---|---|---|---|
| Per class (Mondrian) | 0.888 | 1.98 | 21% | 0.935 |
| **Overall (chosen)** | **0.895** | **1.27** | **73%** | **0.859** |

The overall version was chosen: the same coverage with far more useful single answers.

**Where it is weaker:** coverage drops where regions differ most (British Isles 0.72, India 0.79),
because the guarantee assumes similar places.

**Rule B dropped:** cells it flagged as unfamiliar were **more** accurate (0.928) than the rest
(0.787), so it did not identify errors.

## 28. Four types of place: siting archetypes

K-means clustering on standardised geography only (no labels), k = 4 chosen by silhouette score
(0.15, so the groups overlap):

| Archetype | Sun | Wind (m/s) | Elevation (m) | Built | Model accuracy | Main regions |
|---|---|---|---|---|---|---|
| Calm, sunny lowland | 3.9 | 5.4 | 289 | 82% Solar | 85% | Germany, Italy, Central Europe |
| Windy, very sunny upland | 4.7 | 7.3 | 716 | 71% Wind | 84% | USA, China, Iberia |
| Very windy, sunny mountain | 4.3 | 8.0 | 894 | 94% Wind | 94% | China, Iberia, USA |
| **Windy, cloudy lowland** | 3.0 | 7.3 | 64 | **44% Solar, 52% Wind** | **64%** | Germany, British Isles, France |

**Meaning:** the hard place is the windy, cloudy lowland, where both technologies really are built
side by side. No geographic model can fully separate them there; other factors (land, subsidies,
grid) decide.

## 29. The locked final test

Script: `src/11_final_test.py`. Final model trained on **all 11 development regions**, then the five
locked countries scored **once**. Bootstrap CIs for the differences.

| Model | Accuracy | Macro F1 | Japan | Turkey | Chile | Australia | South Africa |
|---|---|---|---|---|---|---|---|
| Majority class | 0.354 | 0.174 | 0.22 | 0.33 | 0.20 | 0.79 | 0.42 |
| Wind rule (≥ 6.47 m/s) | **0.905** | 0.601 | 0.90 | 0.91 | 0.95 | 0.87 | 0.84 |
| Old model (RF, 5 features, Europe only) | 0.896 | 0.595 | 0.91 | 0.87 | 0.93 | 0.95 | 0.81 |
| RF, 5 features, all 11 regions | 0.902 | 0.600 | 0.91 | 0.89 | 0.95 | 0.87 | 0.93 |
| **Final: Extra Trees, 13 features** | **0.892** | 0.592 | 0.89 | 0.88 | **0.96** | 0.89 | 0.85 |

**Confusion matrix (final model)**

| True ↓ / Predicted → | Hybrid | Solar | Wind |
|---|---|---|---|
| Hybrid | 0 | 2 | 5 |
| Solar | 0 | **604** | 75 |
| Wind | 0 | 33 | **343** |

**How to read it (honestly)**
- **Every sensible model ties at about 0.89–0.91.** Final − old: −0.005 (95% CI −0.020 to +0.009);
  final − wind rule: −0.014 (CI −0.029 to +0.001). Both intervals include zero: the differences are
  chance.
- **Why a tie:** the locked countries are **easy**. The wind rule scores 0.906 there, against 0.776 in
  development. In all five, wind farms stand in much windier places than solar farms. The improvement
  targets windy, cloudy lowlands, which these countries don't contain.
- **The model generalises:** 89% on five countries on four continents it never saw.
- **Conformal sets carried over:** true answer in the set for **94.3%** of cells (target 90%); **85%**
  got a single answer, right **93.6%** of the time; **15%** got "Solar or Wind". Average set size 1.15.
- **Hybrid:** 0 of 7.
- **Nothing was changed after this test.**

## 30. Overfitting of the final model

| Scored on | Accuracy |
|---|---|
| Training data (seen) | 0.995 |
| 11 held-out regions | 0.794 |
| Locked countries | 0.892 |

Extra Trees grows deep trees with random splits and averages them; fitting the training data almost
perfectly is how the method works, and the averaging is what controls overfitting. What matters is
unseen data, where it was the best of the models tested. The depth-limited Random Forest (0.780
held-out, about 0.8 on training data) is the simpler alternative with a smaller train/test gap.

---

# Part E: how it ended

## 31. Final answers to the research question

| Question | Answer |
|---|---|
| Can geography tell Solar from Wind? | **Yes, where wind farms stand in much windier places than solar farms** (southern Europe 90–92%, USA 90–93%, five locked countries 89%). Local context lifts unseen regions from 74% to 79% (British Isles 35% → 54%). |
| Does it work in regions it never saw? | Yes in most, no in windy, cloudy lowlands and India. The hard archetype stays at 64%. |
| Hybrid? | **Not yet**: too few and too concentrated examples (mostly Germany); 1 of 25 in Iberia, 0 of 7 in the locked test. |
| Can it say "I don't know"? | **Yes, with conformal prediction**: "Solar or Wind" when unsure; true answer in the set for 94% of locked cells (target 90%). |
| What drives it? | Wind speed first, but now shared with local wind percentile, sunshine, distance to town and height above surroundings (wind 51% → 23%). |
| Does deep learning help? | Not on this data: trees on good features beat an MLP and a patch network. |

## 32. Using the model on a new location: the assumption

The model answers "given a farm exists here, which type is it?". To use it on a **new** place we need:

> **Assumption:** existing utility-scale farms reflect well-informed siting decisions. Under this
> assumption, the model recommends which type (Solar, Wind or Hybrid) fits a location *if it is
> developed*, and answers "I don't know" when unsure.

- **Why reasonable:** developers study wind, sun, terrain and grid before building, so real farms carry
  expert judgement (the same idea as Sachit et al. 2022).
- **Why only partly true:** land prices, subsidies, permits and grid connections also matter (UK solar
  farms sit in windy places). Accuracy ranges from 35% to 92% by region (54% to 93% after the
  improvement round).
- **How to use it:** "which type fits *if* built", never "build here"; respect "I don't know" (an
  answer set with two types); trust it most where it was shown to work.

## 33. Limitations

1. **Labels:** OpenStreetMap coverage is uneven; a 2020 snapshot; capacities estimated.
2. **Hybrid** depends on the 5 km cell (co-location, not necessarily one project) and has few,
   concentrated examples (308 training, mostly Germany).
3. **Wind still dominates**; windy, cloudy lowlands stay hardest (64%).
4. **Type, not suitability**: the model never saw unsuitable land.
5. **Conformal guarantee assumes similar places** (British Isles coverage 0.72).
6. **Features are site-level averages**; grid distance uses the cell centre; Gridfinder is partly predicted.
7. **The locked countries were easy**, so they cannot show the improvement; a harder untouched test
   (e.g. windy, cloudy countries) would be the next step.
8. **The final Extra Trees model fits training data almost perfectly** (0.995); the depth-limited RF is
   the simpler alternative.

**Next steps:** a convolutional network on terrain and wind maps with more data, harder untouched
test regions, more Hybrid examples from other continents, a label check with Global Energy Monitor.

## 34. Problems we found and fixed

| Problem | How we found it | Fix | Lesson |
|---|---|---|---|
| **Fake wind data**: the global wind file on disk was not genuine Global Wind Atlas data | Spot checks: Gujarat coast 4.7 vs official 6.6 m/s; Bavaria (9.0) windier than the North Sea coast (8.6) | Replaced with official per-country files (Bavaria 5.7, coast 9.3 after the fix) | Always check inputs against known values |
| **Islands with no grid lines** (Canaries, small islands) | Max-value check on grid distance: 20–451 km | Dropped 20 cells; >20 km = no coverage | Look at extremes |
| Missing power values | Two-thirds of records | Used the authors' rule: missing = not a farm | Read the dataset documentation |
| Iberia looked too good | Every region held out | Reported 71% pooled and the northern failure | One test region isn't enough |
| Relative (country) features didn't help | Experiment | Not adopted; led to local context | Report what doesn't work |
| Rule B flagged correct cells | Accuracy of flagged cells (97% Iberia; 93% vs 79% in LORO) | Replaced by conformal prediction | Check every rule against data |
| Remote DEM reads crashed | `RasterioIOError` during feature building | Retry ×3, cache every 100 tiles | Make long downloads resumable |
| Neural network training error with text labels | Library error (`early_stopping`) | Turned early stopping off | Small fixes, documented |
| SHAP too slow on the final model | Hours of compute | 100-tree copy, 80 cells per region | Explain a faithful copy |
| Locked test integrity | Risk of tuning after peeking | Locked and committed before experiments | Order of operations matters |
| Re-run check | Full pipeline from scratch | Byte-identical outputs (seed 42) | Reproducibility is a result |

## 35. The website

One website, built from the project CSVs by `build_site.py`, which packs the data into one
`index.html` (no server needed).

**Globe website**: https://AshutoshSinghJ.github.io/renewable-siting-ml/
- **Map**: a 3-D globe (D3.js 7.9.0, orthographic projection, HTML canvas, 17,189 dots). Views:
  - *What was built*: every cell coloured Solar / Wind / Hybrid.
  - *Iberia test*: right/wrong on Spain and Portugal.
  - *Every region tested*: accuracy of each held-out region.
  - *Other continents*: USA, China, India.
  - *Improved model*: right/wrong of the final model on all 11 held-out regions.
  - *Archetypes*: the four types of place.
  - *Final test*: the five locked countries, outlined, with conformal answers.
  - *Wind speed*: the main feature.
- Hovering a cell shows its class, prediction, confidence and the conformal answer.
- **Results** tab: all charts and tables, including the improvement round (before/after chart, models,
  SHAP, conformal, archetypes, locked test, overfitting).
- **About** tab: the 8-step method (Data, Features, Train, Test, Improve, Retrain, Unsure, Final),
  validation vs test, features, limitations, references.
- How it draws: only the visible hemisphere is drawn; each dot gets a glow then a solid core; drag to
  rotate, scroll to zoom, buttons "fly" between views.

**Hosting:** GitHub Pages serves the `gh-pages` branch.

### 35.1 What the website is made of (UI/UX)

**Building blocks**

| Part | What it is | Why |
|---|---|---|
| `build_site.py` (Python) | Reads the result CSVs and the simplified GADM borders, packs them as JSON and writes it into `template.html` | The website shows exactly the numbers in the report; nothing is typed by hand |
| `template.html` | HTML (structure: top bar, 3 tabs, globe, floating panels), CSS (look: colours, fonts, phone layouts) and JavaScript (behaviour) | One file, no framework, easy to read |
| `index.html` | The finished page, about 1.8 MB, with the data inside | Opens without a server; GitHub Pages serves it as it is |
| D3.js 7.9.0 | Orthographic projection (longitude/latitude to screen), graticule, borders, scales, animation timer | Standard library for maps and charts |
| HTML canvas | Draws all 17,189 dots on every frame | SVG with that many elements would be slow |
| SVG (D3) | The small charts on the Results tab | Sharp text and lines for a few dozen marks |

Main JavaScript pieces: `DATA` (the packed results), `state` (current view, rotation, zoom),
`setMode()` (switch view), `colorOf()` (the colour of a dot in this view), `draw()` (redraw the globe),
`fly()` (animate the camera), `pick()` (find the dot under the pointer, within 9 px) and `showTip()`
(the tooltip).

**Visual design (UI)**
- **Dark theme**, so the bright dots stand out against the Earth.
- **One colour per class everywhere** (website, slides, report): Solar amber `#F6B03E`, Wind teal
  `#36D0BC`, Hybrid purple `#B794F6`, wrong red `#FF5C5C`, "I don't know" grey `#9AB0B9`.
- **Fonts:** Chakra Petch for headings, Inter for text, JetBrains Mono for numbers (digits line up).
- **Layout:** the map fills the screen; small floating panels hold the title, the key numbers, the legend
  with a clickable region list, the zoom buttons (+, −, EU) and the 8 view buttons.
- **Drawing details:** each dot is drawn twice (a soft glow, then a solid core); Hybrid dots are drawn
  last so the rare class is not hidden; the test countries are outlined with a red dashed line.

**User experience (UX)**
- **Map first:** the visitor sees the data immediately; detail lives in the Results and About tabs.
- **One view = one question:** "Where are the real farms?", "Which predictions were wrong?", "Did the new
  model help?" and so on, with plain-language button names.
- **Interaction:** drag to rotate, scroll or pinch to zoom; click a region in the list and the camera
  flies there; hover or tap a dot to see its wind, sunshine, elevation, class probabilities, whether the
  model was right, and its conformal answer.
- **Links:** each tab has its own address (`#map`, `#results`, `#about`).

**Accessibility and phones**
- Visible keyboard focus outlines; ARIA labels on the tabs, buttons and globe; `aria-pressed` on view buttons.
- Camera flights are skipped when the system asks for reduced motion.
- Separate layouts below 860 px and 600 px; dot size scales with the screen; tested at 375 px width.

## 36. Report, slides, documents and GitHub

| Deliverable | Where | Content |
|---|---|---|
| Report | `outputs/REPORT_NOTES.pdf` / `.docx` / `.md` (20 pages) | Sections 1–7 original study, Section 8 improvement round, reproduce, references |
| Presentation | `outputs/Project_Presentation.pptx` (28 slides, speaker notes) | Phase by phase: 1–2 title and roadmap, 3–7 problem and data, 8–10 models and first tests, 11–14 improvement and locked test, 15–17 explanations and uncertainty, 18–23 website (UI/UX, build, views, archetypes, live demo), 24–28 results, limitations, future work, conclusion, references |
| This guide | `docs/COMPLETE_GUIDE.md` | Everything, start to end |
| Journey | `docs/JOURNEY.md` | 20 steps: did, happened, learnt |
| Data and method | `docs/DATA_AND_METHOD.md` | Technical details of every dataset and step |
| Team guide | `docs/TEAM_GUIDE.md` | Who did what, and what each person must know |
| Mind map | `docs/MINDMAP.md` | One-page overview |
| GitHub repo | https://github.com/AshutoshSinghJ/renewable-siting-ml | Code, processed data, results; tags `v1.0-final`, `v1.0.1` |

**Team** (guide: Dr. N Sangita Acharya): Ashutosh Singh (project start, design and dataset);
Astitva Tripathi (models and training); Divyanshu Puri (validation, locked final test, GitHub);
Keshav Sudarshan (SHAP, presentation); Aryan Singh ("I don't know", conformal prediction, result
checks, report); Madhusudan (archetypes, figures and maps, website UI/UX design and build). Details: `docs/CONTRIBUTIONS.md`.

## 37. How to reproduce everything

```bash
pip install -r requirements.txt

# original study (steps 3-5 run from data/processed/, no downloads needed)
python src/03_train_evaluate.py
python src/03b_extra_checks.py
python src/03c_region_holdout.py
python src/04_explain.py
python src/05_dont_know.py

# improvement round (06 and 10 need the raw rasters)
python src/06_features_v2.py
python src/07_global_experiments.py
python src/08_conformal.py
python src/09_patterns.py
python src/10_patch_nn.py
python src/11_final_test.py

# websites
python web/globe/build_site.py
```

Steps 1, 2, 3d, 3e, 6 and 10 need the raw downloads listed in `data/raw/README.md` (~25 GB, not in
git). The 183 MB trained final model is rebuilt by `src/11`. All runs use seed 42.

## 38. Every decision and its justification

| Decision | Justification | Evidence |
|---|---|---|
| Real farms only, no "empty land" | Every label is a fact; empty ≠ unsuitable | Design principle |
| Dunnett 2020 | Peer-reviewed, global, both technologies, free | *Scientific Data* 7:130 |
| ≥ 1 MW, not urban/water | Utility-scale; the authors' own filter | Dataset README |
| 5 km equal-area grid | Comparable cells everywhere | Eckert IV |
| Hybrid = both in a cell | Simple, reproducible, enough examples | 33 / 107 / 409 / 830 at 1 / 2 / 5 / 10 km |
| Europe to train | 58% of cells, 339 of 396 Hybrids | Step 1 counts |
| Iberia to test | Connected, all classes, different climate | Map, counts |
| Spatial CV, never random splits | Nearby cells are near-copies | Roberts et al. 2017 |
| Five original features | Resource, terrain, access | Sachit et al. 2022; subset gains |
| No land cover / roads / capacity | Leakage | Changed by the farm |
| Official wind files | First file was wrong | Gujarat 4.7 vs 6.6 m/s |
| Drop islands without grid | Wrong distances | 20–451 km |
| Random Forest (original) | Best of 6, explainable, probabilistic | CV 0.543 |
| Three baselines | Show what the model adds | Wind rule 0.882 vs RF 0.901 |
| Macro F1, recall, bootstrap CIs | Rare class; chance vs real | CIs reported |
| SHAP | What drives each class | Wind 0.166 |
| Region and continent tests | One test region isn't enough | 71%; USA 90, China 84, India 53 |
| Relative (country) features rejected | No consistent gain | 0.712 → 0.733, macro F1 down |
| **Lock final test first** | Honest final number | Committed before experiments |
| **LORO over 11 regions** | Covers hard and easy places | 16,127 cells |
| **Mean region accuracy as the score** | Reward working everywhere | Fixed in advance |
| **Local-context features** | "Windier than nearby" | 0.736 → 0.780 (RF) |
| **Extra Trees final** | Best mean and worst region | 0.794; worst 0.536 |
| **No deep learning** | Did not beat trees | MLP 0.763, patches 0.738 |
| **No reweighting** | No gain | 0.779 vs 0.780 |
| **Conformal "I don't know"** | Promised error rate | 0.895 dev, 0.943 locked |
| **Rule B dropped** | Flagged cells more accurate | 0.93 vs 0.79 |
| **Archetypes** | Explain where it is hard | Windy, cloudy lowland 64% |

## 39. References and why each was used

| Reference | Used for |
|---|---|
| Dunnett, S. et al. (2020). Harmonised global datasets of wind and solar farm locations and power. *Scientific Data* 7, 130 | The labels and the real-farm filter |
| Breiman, L. (2001). Random forests. *Machine Learning* 45, 5–32 | Main original model |
| Geurts, P., Ernst, D., Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning* 63, 3–42 | Final model |
| Sachit, M. S. et al. (2022). *ISPRS International Journal of Geo-Information* 11(8), 422 | Precedent: RF + SHAP on real plants; feature choice; existing plants as evidence |
| Roberts, D. R. et al. (2017). Cross-validation strategies for data with spatial structure. *Ecography* 40, 913–929 | Why random splits cheat; spatial block CV |
| Meyer, H., Pebesma, E. (2021). Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution* 12, 1620–1633 | Original Rule B |
| Angelopoulos, A. N., Bates, S. (2021). A gentle introduction to conformal prediction. arXiv:2107.07511 | Conformal "I don't know" |
| Shimodaira, H. (2000). Improving predictive inference under covariate shift. *J. Statistical Planning and Inference* 90, 227–244 | Reweighting experiment |
| Lundberg, S., Lee, S.-I. (2017), NeurIPS; Lundberg, S. et al. (2020), *Nature Machine Intelligence* | SHAP and TreeExplainer |
| Arderne, C. et al. (2020). Predictive mapping of the global power system using open data. *Scientific Data* 7, 19 | Gridfinder (distance to grid) |
| Davis, N. N. et al. (2023). The Global Wind Atlas. *BAMS* | Wind speed |
| Global Solar Atlas 2.0 (Solargis / World Bank ESMAP) | Sunshine |
| Copernicus DEM GLO-90 (ESA) | Elevation, slope, terrain context |
| GHS Settlement Model, GHS-SMOD (European Commission JRC) | Distance to town and to water |
| Horn, B. K. P. (1981). Hill shading and the reflectance map. *Proc. IEEE* 69, 14–47 | Slope formula |
| Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap* | Confidence intervals |
| Pedregosa, F. et al. (2011). Scikit-learn. *JMLR* 12, 2825–2830 | All models |
| GADM | Website borders only |

## 40. Glossary

| Term | Meaning |
|---|---|
| Cell | A 5 km × 5 km square of the equal-area grid |
| Label | The true class of a cell: Solar, Wind or Hybrid |
| Feature | A number describing a cell (wind speed, sunshine, …) |
| Baseline | A very simple model used for comparison |
| Spatial cross-validation | Validation that holds out whole areas, not random cells |
| LORO | Leave-one-region-out: each region predicted by a model trained on the others |
| Leakage | A feature that reveals the answer because of the thing being predicted |
| Macro F1 | Average F1 over classes, equally weighted |
| Bootstrap CI | Range of a metric from resampling the test cells |
| SHAP | Fair split of a prediction between features |
| Extra Trees | Random Forest with random split points |
| Conformal prediction | Answer sets with a target coverage (here 90%) |
| Coverage | Share of cells whose true class is in the answer set |
| Archetype | A group of similar places found by clustering |
| Overfitting | Learning training details that don't transfer to new data |
| Covariate shift | New places have a different mix of features than training places |

## 41. The whole story in 20 lines

1. Question: can geography tell Solar from Wind (and Hybrid) at real farms, in unseen regions?
2. Data: 68,786 OpenStreetMap records → 24,078 real farms → 18,705 cells of 5 km.
3. Features: sunshine, wind at 100 m, elevation, slope, distance to grid. No leaky features.
4. Found and replaced a fake wind file; removed islands without grid data.
5. Trained on Europe, tuned with 100 km spatial blocks, tested once on Spain + Portugal.
6. Random Forest 90%, wind rule 88%: real but small gain.
7. Every European region held out: 71%; south 90–92%, British Isles 35%.
8. USA 90%, China 84%, India 53% (and 92% "I don't know" in India).
9. SHAP: wind speed is about half of the reasoning; Hybrid = "windy, not sunny" (Germany).
10. Relative (country) features: no consistent gain.
11. Improvement round: locked Japan, Turkey, Chile, Australia, South Africa first.
12. Developed on 11 held-out regions with a pre-set score.
13. Added local context: wind farms sit in the windiest 16% of their surroundings, 15 m higher.
14. Extra Trees won (0.794); neural networks didn't beat trees; reweighting didn't help.
15. British Isles 35% → 54%, Nordic 36% → 64%; wind share 51% → 23%.
16. Conformal prediction gives "Solar or Wind" when unsure, with 90% coverage.
17. Four archetypes; the windy, cloudy lowland is the hard one (64%).
18. Locked test, scored once: every model ties at ~0.9 (easy countries); conformal coverage 94%.
19. Hybrid is still not learned.
20. Lessons: test on unseen regions, compare with baselines, check inputs, lock the test, report failures.
