> **Note:** this is the original brief written at the start of the project. The work later went further:
> an improvement round (local-context features, Extra Trees, conformal "I don't know", a locked final
> test on five countries). See `docs/COMPLETE_GUIDE.md` and the report, Section 8.

# Project: Predicting Renewable Installation Type from Geography (ML)

Major project (5 credits). Build it in one night. **Keep everything simple, explainable and standard.** Prefer plain Python scripts over frameworks. Do not add features, models or steps that are not listed here.

---

## 1. Problem statement

> Given a location where an onshore renewable energy installation exists, can a machine learning model trained on geographic and environmental factors (solar resource, wind resource, terrain, grid proximity) correctly predict whether it is a **solar, wind, or hybrid** site — including in a geographic region it has never seen during training — and say **"I don't know"** when it is unsure or the location is unlike anything it was trained on?

### In scope
- Onshore, utility-scale solar and wind installations.
- Classes: **Solar, Wind, Hybrid** (+ "I don't know" output).
- Validation on a **geographically held-out region** using real, known installations.

### Out of scope
- Hydro, offshore wind, rooftop solar, economics, policy.
- Odisha / any application to new areas (optional demo only, at the very end).
- Any claim that a location is "unsuitable". The model answers *"if something is built here, which type fits?"*, not *"should anything be built here?"*

---

## 2. Key decisions and WHY (keep these in the report)

| Decision | Why |
|---|---|
| Train only on real installations (no "empty land" negatives) | Every label is an observed fact. "No plant" ≠ "unsuitable" (professor's concern). |
| One model, 3 classes (Solar / Wind / Hybrid) | Same features and pipeline for all; simple to explain. |
| Hybrid = a 5 km grid cell containing both solar and wind installations | Consistent, reproducible definition. State in report that it depends on cell size. |
| Features: GHI, wind speed 100 m, elevation, slope, distance to transmission line | Core resource + terrain + grid factors used in published siting studies (Sachit et al. 2022). |
| **Do NOT use land cover or distance to roads as features** | The farm itself changes land cover (2020 imagery shows the farm) and brings access roads → data leakage. |
| **Capacity used only as a filter (≥1 MW), never as a feature** | Keeps utility-scale only. As a feature, the model would guess type from plant size, not geography → leakage. |
| Spatial split: whole held-out test region + spatial block CV | Nearby sites are similar; random splits inflate scores (Roberts et al. 2017, Ecography). |
| Baselines: majority class, simple wind-speed rule, logistic regression | Proves Random Forest learned more than "windy = wind". |
| Main model: Random Forest, `class_weight="balanced"` | Standard for tabular geospatial data, robust, explainable with SHAP (used in Sachit 2022, Wu 2026). |
| Metrics: macro F1, per-class recall, confusion matrix (accuracy secondary) | Solar will dominate counts; accuracy alone hides poor Wind/Hybrid performance. |
| SHAP for explanation | Shows which factor drives each class. |
| "I don't know" = low confidence OR unfamiliar location | Model should not guess outside what it learned (Meyer & Pebesma 2021, area of applicability). |

---

## 3. Dataset

### Primary (training + testing): Dunnett et al. 2020
- Paper: Dunnett, S., Sorichetta, A., Taylor, G., Eigenbrod, F. *Harmonised global datasets of wind and solar farm locations and power.* Scientific Data 7, 130 (2020). https://doi.org/10.1038/s41597-020-0469-8
- Data: https://doi.org/10.6084/m9.figshare.11310269
- Built from OpenStreetMap, peer-reviewed, capacity cross-checked against national statistics. Wind records already exclude urban/water and need >4 turbines.
- Local files the user already has (check which exist in the repo / data folder):
  - `global_wind_solar_2020.zip` (original Dunnett data)
  - `dunnett_solar_wind_5km_labels.gpkg` / `.csv` (user's earlier 5 km derived labels)
- Known fields in the original data include: `x_id`, `GID_0` (ISO3 country), `turbines`, `panels`, `power` (estimated MW), `water`, `urban`. **Inspect the actual columns first; do not assume.**
- Limitations (state in report): OSM coverage is uneven (dominated by Europe, North America, East Asia); 2020 snapshot; capacity values are first-order estimates.

### Independent label check (optional): Global Energy Monitor
- Global Solar Power Tracker (≥1 MW) and Global Wind Power Tracker (≥10 MW). Free download after email form; the user must download manually.
- Used ONLY to cross-check labels in the test region, never for training.

### Feature layers (download only for the chosen countries)
| Feature | Source |
|---|---|
| GHI | Global Solar Atlas (globalsolaratlas.info) GeoTIFF |
| Wind speed at 100 m | Global Wind Atlas (globalwindatlas.info) GeoTIFF |
| Elevation (+ slope derived) | SRTM or Copernicus DEM (90 m is enough) |
| Distance to transmission line | Gridfinder (Arderne et al. 2020, Scientific Data) or OSM power lines |

If network downloads are blocked, **stop and tell the user exactly which files to download and where to put them** (`data/raw/`).

---

## 4. Pipeline (6 steps, one script each)

Project layout:
```
data/raw/        # downloaded inputs (never modified)
data/processed/  # cells.csv, features.csv
src/             # 01_..06_ scripts
outputs/         # tables (csv), figures (png), report notes
```
Tech: Python, pandas, geopandas, rasterio, numpy, scikit-learn, shap, matplotlib. Fixed `random_state=42` everywhere. Use an equal-area or projected CRS for distances.

### Step 1 — `01_prepare_labels.py` (STOP after this and show the user the counts)
- Load Dunnett solar + wind points. Drop records with `water` or `urban` flagged, and estimated capacity < 1 MW (if `power` is available).
- Build a 5 km grid; label each cell: Solar only / Wind only / Both = Hybrid. Cells with nothing are NOT used.
- Print counts per class, total and per country (top 15).
- **Region rule:** choose a training region with ≥ 5,000 labelled cells and a neighbouring held-out test region that has all classes present. Propose a split and wait for user confirmation.
- **Hybrid rule:** keep Hybrid as a class only if it has ≥ 300 cells. Otherwise train Solar vs Wind only and derive Hybrid later (both probabilities ≥ 0.4).
- Output: `data/processed/cells.csv` — `cell_id, lon, lat, country, label, split`.

### Step 2 — `02_extract_features.py`
- Sample GHI, wind100, elevation, slope at each installation point; average per cell.
- Distance from each cell to the nearest transmission line (km).
- Drop cells with missing features; report how many dropped.
- Output: `data/processed/features.csv` — `cell_id, ghi, wind100, elev, slope, dist_grid, label, split, block_id` (block_id = 100 km spatial block, for CV).

### Step 3 — `03_train_evaluate.py`
- Tuning: `GroupKFold(n_splits=5)` on the training region, groups = `block_id`. Small grid only (RF: n_estimators 300, max_depth [None, 10, 20], min_samples_leaf [1, 5]).
- Models: (a) majority class, (b) wind-speed rule (threshold chosen on training data), (c) logistic regression (scaled features, balanced), (d) Random Forest (balanced).
- Fit final models on the full training region; evaluate ONCE on the held-out test region. **The test region must not be touched before this.**
- Output: `outputs/results.csv` (accuracy, macro F1, per-class recall for all models), `outputs/confusion_rf.png`, `outputs/cv_scores.csv`.

### Step 4 — `04_explain.py`
- SHAP (TreeExplainer) on the Random Forest: summary plot per class + mean |SHAP| bar chart.
- Output: `outputs/shap_*.png`, `outputs/feature_importance.csv`.

### Step 5 — `05_dont_know.py`
- Rule A (low confidence): max class probability < 0.5 → "I don't know".
- Rule B (unfamiliar location): standardise features, weight by RF importance; for each test cell compute distance to nearest training cell. Threshold = 95th percentile of the same distance computed for training cells against other CV folds. Above threshold → "I don't know".
- Report: % of test cells labelled "I don't know", and accuracy / macro F1 on the remaining confident predictions vs on all.
- Output: `outputs/dont_know_summary.csv`, `outputs/test_predictions.csv`.

### Step 6 (optional) — `06_label_check.py`
- If the user provides GEM files: match test-region Dunnett sites to GEM sites within 2 km. Report % matched and % technology agreement.
- Output: `outputs/label_check.csv`.

---

## 5. Working rules
- Keep code short and readable, with comments explaining the WHY.
- Never use a random train/test split. Never tune on the test region.
- Never add land cover, roads, or capacity as features.
- Stop and ask the user after Step 1 (region + Hybrid decision) and whenever a data file is missing.
- At the end, write `outputs/REPORT_NOTES.md`: problem statement, dataset + limitations, method table (Section 2), results table, confusion matrix, SHAP findings, "I don't know" results, limitations.

## 6. Limitations to state honestly
- OSM-based coverage is incomplete and regionally uneven.
- Data is a 2020 snapshot; capacity is estimated.
- Hybrid depends on the 5 km cell definition.
- The model predicts technology type given an installation exists; it does not judge whether a location should be developed.
- If the simple wind-speed rule performs close to Random Forest, report that honestly — it is a valid finding.

## 7. Key references
- Dunnett et al. 2020, Scientific Data — dataset.
- Sachit et al. 2022, ISPRS Int. J. Geo-Inf. 11(8):422 — RF + SHAP on >55,000 real wind/solar plants (closest prior work).
- Roberts et al. 2017, Ecography — spatial cross-validation. https://doi.org/10.1111/ecog.02881
- Meyer & Pebesma 2021, Methods in Ecology and Evolution 12:1620–1633 — area of applicability ("I don't know").
- Wu et al. 2026 — comparison of RF, gradient boosting, MaxEnt, logistic regression for US solar/wind siting.
