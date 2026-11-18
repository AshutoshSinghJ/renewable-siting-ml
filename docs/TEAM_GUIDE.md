# Team Guide: who did what, and everything each person must know

**Project:** Which Renewable Fits Here? Predicting Solar, Wind or Hybrid sites from geography

**Guide:** Dr. N Sangita Acharya

| Member | Role | Presents slides |
|---|---|---|
| **Ashutosh Singh** | Project start, design and **dataset** (labels, features, local-context features) | 2–5, 16 |
| **Astitva Tripathi** | Models and training (baselines, Random Forest, model comparison, Extra Trees, neural networks) | 7, 8, 10, 18 |
| **Divyanshu Puri** | Validation and testing (regions, continents, locked final test) and GitHub deployment | 11–13, 20 |
| **Keshav Sudarshan** | Explainability (SHAP) and the presentation | 1, 6, 14, 24 |
| **Aryan Singh** | "I don't know" (confidence rules, conformal prediction), result checks, the project report and references | 15, 19, 22, 23, 25 |
| **Madhusudan** | Visualisation: siting archetypes, all figures and maps, the globe website | 9, 17, 21 + live demo |

Ashutosh owns the start, design and dataset; each of the other five owns one technical part and one deliverable, with a similar share of slides. The parts
below explain each topic in depth; the owner of each topic is in the table above. The exact split of every script and file is in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

Everyone reads **Part 0** first. Then read your own part carefully, and skim the others so you can
hand questions to the right person in the viva.

---

## Part 0: what EVERYONE must know

### The story in 60 seconds
> "We asked whether geography alone can tell a solar farm from a wind farm. We took 24,000 real
> farms from OpenStreetMap (Dunnett et al. 2020), put them on a 5 km grid, and described each cell
> with five features: sunshine, wind speed, elevation, slope and distance to the grid. We trained a
> Random Forest on Europe and tested it on Spain and Portugal, which it never saw: 90% accuracy.
> Then we tested every European region in turn and three other countries. It works where wind
> farms are much windier than solar farms (southern Europe 90–92%, USA 90%, China 84%) and fails
> where they are not (British Isles 35%, India 53%), because it relies on absolute wind speed.
> Hybrid sites were not learned. The model can also say 'I don't know', and in India it did so for
> 92% of cells.
>
> Then we ran an improvement round: we locked five untouched countries first, added local context
> (is this site windier than the land around it?) and switched to Extra Trees. On 11 held-out regions
> accuracy rose from 74% to 79%, the British Isles from 35% to 54%. On the locked countries every
> model tied at about 90%, because they are easy, and conformal 'I don't know' sets kept the true
> answer 94% of the time."

### Problem statement
> Given the location of an existing onshore, utility-scale renewable installation, can a model
> trained only on geographic factors (sunshine, wind speed at 100 m, elevation, slope, distance to
> the grid) predict whether it is **Solar, Wind or Hybrid**, including in regions it has never seen,
> and say **"I don't know"** when it is unsure or the location is unlike its training data?

**Application, with a stated assumption:** *if* existing farms reflect well-informed siting
decisions, the model can recommend which type fits a new location *if it is developed*. It never
says "build here".

### Numbers everyone must remember

| What | Number |
|---|---|
| Raw records → real farms | 68,786 → 24,078 |
| Labelled 5 km cells (world) | 18,705 (Wind 9,609 · Solar 8,700 · Hybrid 396) |
| Cells used in Europe | 10,724 (train 9,176 · test 1,548) |
| **Iberia test accuracy** | **0.901** (95% CI 0.886–0.915), macro F1 0.627 |
| Wind-speed rule on Iberia | 0.882 (RF only +1.8 points better) |
| Solar / Wind / Hybrid recall (Iberia) | 92% / 91% / 1 of 25 |
| Overfitting check | train 0.81 · unseen blocks 0.73 · Iberia 0.90 |
| 8 European regions held out | 71% overall; south 90–92%; British Isles 35%; Nordic 40% |
| USA / China / India | 90% / 84% / 53% |
| "I don't know" | Iberia 40%, India 92%, China 76% |
| Most important feature | Wind speed (SHAP 0.166), then sunshine (0.070) |
| **Improvement round, 11 held-out regions** | **0.736 → 0.794** (British Isles 0.354 → 0.536) |
| Final model | Extra Trees, 13 features; wind's share of SHAP 51% → 23% |
| **Locked final test (JPN, TUR, CHL, AUS, ZAF)** | final 0.892 · wind rule 0.905 · old RF 0.896 (a tie) |
| Conformal "I don't know" on locked test | true answer in set 94.3% (target 90%) |

### Who answers what in the viva

| If asked about… | Answered by |
|---|---|
| Why this problem, design decisions, dataset, filters, grid, features, leakage, data errors, local-context features | Ashutosh Singh |
| Baselines, Random Forest, tuning, metrics, Iberia result, model comparison, Extra Trees, neural networks | Astitva Tripathi |
| Overfitting, regions, other continents, locked final test, GitHub, reproducibility | Divyanshu Puri |
| SHAP, what drives the model, why Hybrid fails, presentation flow | Keshav Sudarshan |
| "I don't know", Rule A/B, conformal prediction, how the results were checked, limitations, references | Aryan Singh |
| Archetypes, figures and maps, the globe website, before/after map | Madhusudan |

### Links
- Repository: https://github.com/AshutoshSinghJ/renewable-siting-ml
- Live website: https://ashutoshsinghj.github.io/renewable-siting-ml/

---

## Part 1 — Ashutosh Singh: initiation, project design and dataset

### 1.1 How the project started
1. **Starting idea:** solar and wind farms are built in different kinds of places. Can a machine
   learn that difference from geography alone?
2. **Earlier work** (an Odisha decision-support system using MCDA/AHP suitability maps) showed the
   weakness of opinion-based weights. So this project learns from **real decisions**: farms that
   were actually built.
3. **The key design decision:** train **only on real installations**, never on "empty land".
   - Empty land isn't proof of "unsuitable" (it may be protected, expensive, privately owned, or
     never proposed).
   - Every label is then a fact.
   - The cost: the model answers *"which type fits if something is built?"*, never *"should
     anything be built?"*.

### 1.2 Design decisions and why

| Decision | Why |
|---|---|
| 3 classes: Solar, Wind, Hybrid | One model, one pipeline; Hybrid is interesting for planning |
| Hybrid = solar and wind farms in the same 5 km cell | Simple, reproducible; weakness: depends on cell size |
| 5 km grid in an equal-area projection (World Eckert IV, ESRI:54012) | Turns points into comparable units; every cell has the same area |
| Keep Hybrid only if ≥ 300 cells | Fewer would be too few to learn from (we had 396) |
| Train on Europe | 10,792 of 18,705 cells and **339 of 396 Hybrids** are in Europe |
| Test on Spain + Portugal (Iberia) | Connected block, all 3 classes, touches training only at the Pyrenees, different climate (sunnier, higher) |
| No random train/test split | Neighbouring cells are near-copies; random splits inflate scores (Roberts et al. 2017) |
| Five features: sunshine, wind 100 m, elevation, slope, grid distance | Resource + terrain + access, the standard siting factors (Sachit et al. 2022) |
| No land cover, roads or capacity as features | **Leakage**: the farm itself changes land cover, brings roads, and its size reveals the type |
| `random_state = 42` everywhere | Reproducible results |

### 1.3 The dataset: Dunnett et al. (2020)
- **Citation:** Dunnett, S., Sorichetta, A., Taylor, G., Eigenbrod, F. (2020). Harmonised global
  datasets of wind and solar farm locations and power. *Scientific Data* 7, 130.
- **Download:** figshare, doi 10.6084/m9.figshare.11310269, file `global_wind_solar_2020.zip`.
- **Built from:** OpenStreetMap (2020 snapshot), peer-reviewed, capacity cross-checked against
  national statistics.
- **Files used:** `global_solar_2020_WGS84.gpkg` (35,272 points) and `global_wind_2020_WGS84.gpkg`
  (33,514 multipoints, one point per turbine).
- **Columns:** `GID_0` (country code), `panels` / `turbines`, `panel.area`, `landscape.area`,
  `water` (1 = in water), `urban` (1 = in a city), `power` (estimated MW).

**Why this dataset:** free, peer-reviewed, real installations, solar and wind in one harmonised
format, global (which made the USA/China/India tests possible).

**Its limits (say them yourself):** OpenStreetMap coverage is uneven (strong in Europe, North
America, East Asia; weak in India), it's a 2020 snapshot, and capacity values are estimates.

### 1.4 Cleaning: `src/01_prepare_labels.py`
1. Drop records flagged **water** or **urban** (offshore turbines, rooftop panels).
2. Keep **power ≥ 1 MW**. Two-thirds of records had no power value. The dataset README explains
   that the authors only estimate power for real farms (wind > 4 turbines; solar > 1 ha of panels),
   so "no power" means "not a farm". This is the authors' own recommended filter.
   → **11,498 solar + 12,580 wind = 24,078 farms.**
3. Wind farms are multipoints, so each farm uses its **centroid**.
4. Project to Eckert IV and snap to a **5,000 m grid**: `cell = floor(x / 5000), floor(y / 5000)`.
5. Label each cell: Solar only / Wind only / **Hybrid** (both).
6. Result: **18,705 cells** (Wind 9,609, Solar 8,700, Hybrid 396).
7. Split: train = Europe without ESP/PRT (33 countries), test = ESP + PRT.
8. Outputs: `data/processed/cells.csv` (`cell_id, lon, lat, country, label, split`) and
   `data/processed/points.csv` (every farm point with its cell).

Top countries: Germany 3,054 cells · USA 2,342 · China 2,313 · Italy 1,644 · Spain 1,372 · UK 1,024.

### 1.5 Features: `src/02_extract_features.py`

| Feature | Source | How it is computed |
|---|---|---|
| `ghi` (kWh/m²/day) | Global Solar Atlas 2.0, 9 arc-sec (~250 m) | Raster value at each farm point |
| `wind100` (m/s) | Global Wind Atlas, official per-country GeoTIFFs, 250 m | Value at each farm, from the farm's own country file |
| `elev` (m) | Copernicus DEM GLO-90 | Read online from AWS (cloud-optimised GeoTIFF), only the pixels needed |
| `slope` (°) | Same DEM | Horn (1981) formula on a 3×3 pixel window |
| `dist_grid` (km) | Gridfinder (Arderne et al. 2020) | Cell centre to the nearest line, in metres (EPSG:3035 for Europe) |

- Values are **averaged per cell** (a cell can contain several farms).
- `block_id` = 100 km block, used for spatial cross-validation.
- **Dropped cells (68):** 34 far-north (no sunshine data above ~65°N), 13 Azores/Madeira (no wind
  file), 20 islands with no Gridfinder lines (distance > 20 km, e.g. the Canaries at 117–451 km),
  1 Kosovo (no wind file).
- Output: `data/processed/features.csv` → **10,724 cells**.

### 1.6 Data problems you found and fixed (good viva material)
1. **Fake wind raster:** the first wind file did not match the official Global Wind Atlas (Gujarat
   coast 4.7 vs official 6.6 m/s; Bavaria windier than the North Sea coast). Replaced with
   official per-country files. *Lesson: check inputs against known values.*
2. **Island grid distances:** one cell was "451 km" from the grid (Canary Islands, no lines in
   Gridfinder). Every cell above 20 km was such an island, so they were dropped. *Lesson: look at
   the extremes, not just averages.*

### 1.7 Viva questions for Ashutosh
- **Why this dataset?** The only free, peer-reviewed, global dataset with real solar and wind farms
  in one format; every label is a fact.
- **Why not use land without farms as negatives?** "No farm" isn't "unsuitable"; that label would
  be false. It's why we predict type, not suitability.
- **Why Europe?** Most data (58% of cells) and 339 of 396 Hybrids; the only region where three
  classes and held-out regions were possible.
- **Why Iberia as the test?** Connected, all three classes, different climate, a fair but hard test.
- **Why 5 km cells?** Small enough to keep local conditions, large enough to group one farm's
  turbines; Hybrid depends on it (a limitation).
- **Why not land cover?** The 2020 imagery shows the farm itself, so it would leak the answer.
- **Why is capacity only a filter?** As a feature, plant size would give the type away.

---

## Part 2 — Models (Astitva Tripathi), validation and GitHub (Divyanshu Puri), SHAP (Keshav Sudarshan), "I don't know" (Aryan Singh)

Suggested focus: **Astitva** leads on model selection, training and evaluation (2A); **Divyanshu**
leads on GitHub, deployment and reproducibility (2B). Both should be able to explain both.

### 2A. Model selection and training (`src/03` to `src/05`)

#### The four models (`src/03_train_evaluate.py`)

| Model | What it does | Why it is there |
|---|---|---|
| Majority class | Always predicts the most common class | The floor any model must beat |
| **Wind-speed rule** | "Wind if wind speed ≥ t, else Solar"; t = **6.60 m/s**, chosen on training data | The key baseline: did ML learn more than "windy = wind"? |
| Logistic regression | Linear model, scaled features, `class_weight="balanced"` | Simple statistical model; tests whether non-linearity is needed |
| **Random Forest** (main) | 300 trees, `class_weight="balanced"` | Main model |

#### Why the Random Forest
1. Tabular data (5 features, ~9,000 rows): tree ensembles are the standard choice.
2. Captures non-linear interactions ("windy *and* not sunny → Hybrid").
3. No scaling needed; robust to outliers.
4. `class_weight="balanced"` stops it ignoring the rare Hybrid class (3% of cells).
5. Gives probabilities, which made "I don't know" possible.
6. Explainable with SHAP; used in published siting work (Sachit et al. 2022); Breiman (2001).
7. **Checked, not assumed:** gradient boosting 0.542, Extra Trees 0.537, k-NN 0.510, SVM 0.476,
   logistic regression 0.477 vs RF **0.543** (CV macro F1, `src/03b_extra_checks.py`).

#### Training and tuning
- **Validation:** `GroupKFold(n_splits=5)` with `groups = block_id` (100 km blocks), inside the
  training region only.
- **Small grid:** `max_depth ∈ {None, 10, 20}`, `min_samples_leaf ∈ {1, 5}`, scored by macro F1.
- **Selected:** `max_depth = 10`, `min_samples_leaf = 1`, CV macro F1 **0.543** (the six settings
  ranged 0.522–0.543).
- **Final fit** on the whole training region, then **one** evaluation on Iberia.
- Saved model: `outputs/rf_model.joblib`.

#### Metrics and why
- **Macro F1:** averages F1 over the three classes equally; accuracy alone hides poor Hybrid results.
- **Per-class recall:** of all real Solar cells, how many were found.
- **Confusion matrix:** what each true class was predicted as.
- **Bootstrap 95% CIs** (2,000 resamples of test cells): shows whether differences are real.

#### Results on Iberia (`outputs/results.csv`)

| Model | Accuracy | Macro F1 | Recall Hybrid |
|---|---|---|---|
| Majority | 0.490 | 0.219 | 0 |
| Wind rule | 0.882 | 0.593 | 0 |
| Logistic regression | 0.844 | 0.641 | 0.56 (but only 7% of its Hybrid guesses correct) |
| **Random Forest** | **0.901** | 0.627 | 0.04 |

RF − wind rule: +1.8 accuracy points, CI +0.5 to +3.2 → real but small.

Confusion matrix (RF, Iberia): Solar 696 right / 59 → Wind; Wind 697 right / 65 → Solar; Hybrid
1 right, 18 → Wind, 6 → Solar.

#### Extra checks
- **`03b_extra_checks.py`:** other classifiers (above) and feature subsets: wind only 0.503 → +sun
  0.524 → +terrain 0.540 → +grid 0.543. Wind does most of the work; grid distance adds almost
  nothing, so it isn't leaking.
- **`03c_region_holdout.py`:** each of 8 European regions held out in turn (same settings, no
  retuning): Italy 92%, Balkans 91%, Iberia 90% (reproduces Step 3 exactly), Central & Eastern 79%,
  France & Benelux 77%, Germany & Austria 61%, Nordic & Baltic 40%, British Isles 35%; pooled 71%.
  Also the overfitting check: 0.812 / 0.732 / 0.901.
- **`03d_other_continents.py`:** Europe-trained model applied unchanged: USA 90%, China 84%
  (always-Wind gives 81%), India 53%.
- **`03e_relative_features.py`:** wind and sun minus the country median. Pooled 71% → 73%, but
  macro F1 down and the south worse, so it was not adopted.

**Why the north fails:** UK solar farms stand at 7.7 m/s, windier than typical European wind farms
(7.4 m/s). The model learned absolute wind speed, so it calls them Wind.

#### Explanation: `src/04_explain.py`
- SHAP with TreeExplainer (Lundberg & Lee 2017; Lundberg et al. 2020) on 1,000 training cells.
- Mean |SHAP|: **wind 0.166**, sunshine 0.070, elevation 0.035, slope 0.030, grid 0.023.
- Hybrid learned as "windy, not sunny", because 213 of 308 training Hybrids are in Germany.

#### Confidence and "I don't know": `src/05_dont_know.py`
- **Confidence:** each of the 300 trees ends in a leaf with class shares; the forest averages them
  into 3 probabilities that add up to 1. The top one is the prediction; its value is the confidence.
- **Rule A (low confidence):** top probability < 0.5, so no class has a majority.
- **Rule B (unfamiliar place, Meyer & Pebesma 2021):** z-score the 5 features, weight by RF
  importance (wind 0.41, sun 0.23, elevation 0.14, slope 0.13, grid 0.10), measure the distance to
  the nearest training cell. Threshold **0.141** = 95th percentile of the same distance for training
  cells measured against *other* CV blocks. Above → "I don't know".
- **Iberia:** 40% abstained (10% low confidence, 32% unfamiliar); accuracy on the answered cells
  0.938; low-confidence cells only 47% right (rule works); unfamiliar cells 97% right (too cautious).
- Confidence ≠ accuracy: most Iberian "I don't know" answers came from Rule B while the model was
  confident.

#### Viva questions for Astitva Tripathi
- Why RF and not deep learning? · Why these baselines? · What is macro F1? · Is it overfitting?
  · Why is the test higher than validation? (Iberia is easier; the wind rule jumps too.) · Is it just
  "windy = wind"? (Mostly, and we show it.) · How does "I don't know" work?

### 2B. GitHub and web deployment

#### Repository structure
```
renewable-siting-ml/
├── src/              pipeline 01 → 11 (+ 03b–03e checks)
├── data/processed/   cells and features (steps 3–5 run without downloads)
├── data/raw/         not in git (~25 GB); README lists every download
├── outputs/          results, figures, model, report, slides
├── web/globe/        website: template.html, build_site.py, world.json, index.html
├── docs/             guides, contributions, journey, mind map, project brief
├── tools/            md2docx.js (report → Word), make_map.py
└── README.md · requirements.txt · .gitignore · LICENSE (MIT)
```

#### What was removed or kept out, and why
- Raw rasters and Dunnett files (~25 GB): too large; `.gitignore` excludes `data/raw/*`, `*.tif`, `*.gpkg`.
- Temporary preview copies, caches, and old stage folders: not needed.
- The website border cache (`world.json`) is **kept**, so the site rebuilds without the 500 MB GADM file.

#### Branches and how GitHub Pages works
- **`main`:** code, data, results, documents.
- **`gh-pages`:** the published website (`index.html`) and `.nojekyll`.
- GitHub Pages serves `gh-pages` automatically (Settings → Pages → Deploy from branch → gh-pages /
  root). Each push to `gh-pages` updates the live site within ~30 seconds.

#### How to update the live site
```bash
python web/globe/build_site.py          # rebuild after changing the template or results
git add -A && git commit -m "Update website" && git push origin main

git worktree add ../pages gh-pages      # publish to GitHub Pages
cp web/globe/index.html ../pages/index.html
cd ../pages && git add -A && git commit -m "Publish website" && git push origin gh-pages
cd - && git worktree remove ../pages
```

#### Reproducibility
- `pip install -r requirements.txt` (pandas, geopandas, pyogrio, shapely, rasterio, numpy,
  scikit-learn, shap, matplotlib, joblib; Python 3.11).
- Steps 3–5 and 7–9 run straight from `data/processed/`. Steps 1, 2, 3d, 3e, 6, 10 and 11 need `data/raw/` (see its README).
- A full re-run from scratch reproduced every output byte-for-byte (seed 42).

#### Viva questions for Divyanshu Puri
- How is the site hosted? (GitHub Pages from the `gh-pages` branch.)
- Why aren't the raw files on GitHub? (25 GB; download links in `data/raw/README.md`.)
- How can someone reproduce the results? (requirements, processed data, scripts in order.)
- What happens when you push? (Pages rebuilds and the live site updates in ~30 s.)

#### Improvement round scripts
| Script | What it does | Output |
|---|---|---|
| `06_features_v2.py` | 8 local-context features + base features for the locked countries (Ashutosh) | `features_v2.csv` |
| `07_global_experiments.py` | Feature sets A–E and 7 models, leave-one-region-out over 11 regions (Astitva) | `v2_feature_sets.csv`, `v2_models.csv`, `v2_loro_predictions.csv` |
| `08_conformal.py` | Conformal answer sets, "I don't know" (Aryan) | `v2_conformal_*.csv/json` |
| `09_patterns.py` | Archetypes (Madhusudan) and SHAP by region (Keshav) | `v2_archetypes*.csv/png`, `v2_shap_*.csv/png` |
| `10_patch_nn.py` | Neural network on raw 11 × 11 map patches (Astitva) | `v2_patch_nn.csv` |
| `11_final_test.py` | Locked test, scored once (Divyanshu) | `v2_final_test*.csv`, `v2_overfit_check.csv` |

Key results: 11 held-out regions 0.736 → 0.794 (Extra Trees, final); locked test final 0.892 vs
wind rule 0.905 (a tie); conformal coverage 0.943. The 183 MB trained model and the patch cache are
not in git (`.gitignore`): rerun `src/11` and `src/10` to rebuild them. The website has three views
for this round (Improved model, Archetypes, Final test).

---

## Part 3 — Madhusudan: visualisation and the website

### 3.1 The website

| | Globe website |
|---|---|
| Live link | https://ashutoshsinghj.github.io/renewable-siting-ml/ |
| Inspiration | Kaspersky Cybermap: dark, glowing points |
| Structure | Top bar with **3 tabs** (Map · Results · About) |
| Map | 3D **globe** (orthographic projection) |

The map comes first; the detail lives in short Results and About tabs.

Madhusudan also owns the **siting archetypes** (k-means on geography, `src/09`) and **all figures and maps**
used in the report and slides (train/test map, confusion matrix, region charts, archetype map).

### 3.2 Design decisions

| Decision | Why |
|---|---|
| Dark "control room" theme | Glowing dots read best on dark; matches the map-first, Kaspersky-like feel |
| One colour per class everywhere: **Solar amber, Wind teal, Hybrid purple, wrong red, "I don't know" grey** | The same colour means the same thing on every chart, map and slide |
| Fonts: Chakra Petch / Space Grotesk (headings), Inter (text), JetBrains Mono (numbers) | Technical feel; monospaced digits line up in stats |
| Numbers in floating corner panels that change per view | Like a dashboard: the key number for each view is always visible |
| Hybrid dots drawn last | They're rare and would otherwise be hidden under Solar and Wind |
| Plain-language labels ("What was built", "Model on Iberia") | Non-experts understand them without reading the report |

### 3.3 Map views
1. **What was built:** every cell in its true class colour.
2. **Iberia test:** flies to Spain/Portugal; right = class colour, wrong = red.
3. **Every region tested:** each cell predicted by a model that never saw its region; a clickable
   region list flies the globe there (the British Isles show red).
4. **Other continents:** USA, China, India; grey = "I don't know".
5. **Improved model:** right/wrong of the final model on all 11 held-out regions, with the change per region.
6. **Archetypes:** the four types of place.
7. **Final test:** the five locked countries, outlined, with their conformal answers.
8. **Wind speed:** colour ramp from calm to windy.

Interactions: drag to rotate, scroll/pinch to zoom, +/−/EU buttons, hover or tap a dot for a
tooltip (features, class probabilities, right/wrong, the improved model's answer and its conformal set).

### 3.4 How it works technically
- **Data flow:** `build_site.py` reads the project CSVs → packs cells, predictions, results and
  borders into JSON → injects it into `template.html` → writes a single self-contained `index.html`.
- **Drawing:** HTML `<canvas>` (fast enough for ~17,000 dots), D3.js 7.9.0 for projections.
- **Globe:** `d3.geoOrthographic`; dots on the far side are hidden with a hemisphere test; each dot is
  drawn twice: a soft additive glow, then a solid core (pure additive glow turned dense areas white).
- **Camera:** smooth "fly" animations between views; jumps instantly if the user prefers reduced
  motion or the page is in the background.
- **Hover:** finds the nearest visible dot within 9 pixels.
- **Routing:** `#map`, `#results`, `#about` in the URL, so each tab can be linked directly.
- **Charts** (Results tab): D3-drawn SVG; the "I don't know" bar is computed live from the cell data.
- **Share version:** split into `index.html`, `css/style.css`, `js/app.js`, `js/data.js`,
  `lib/d3.min.js`; opens by double-clicking, no server needed.

### 3.5 Responsiveness and accessibility
- Works on phones: panels stack, dots shrink with screen size, no sideways scrolling (tested at
  375 px width).
- Respects "reduced motion" settings; visible keyboard focus; text alternatives on the map canvas.
- Tested in the browser at desktop and phone sizes with no console errors.

### 3.6 Viva questions for Madhusudan
- Why a dark theme and glowing dots? · Why these colours? · How does the globe
  draw 17,000 points smoothly? (canvas, not SVG) · What happens when I hover a dot? · Does it work
  on a phone? · How is the data inside the page? (packed as JSON by `build_site.py`)

---

## Part 4 — Report (Aryan Singh) and presentation (Keshav Sudarshan)

Suggested focus: **Aryan Singh** leads the report and references (4A); **Keshav** leads the slides
and presentation flow (4B). Both should know the whole storyline.

### 4A. The report (`outputs/REPORT_NOTES.pdf`, `.docx`, `.md`, 20 pages)

| Section | What it contains |
|---|---|
| 1. Problem statement | The question, scope, the short answer |
| 2. Data | Dunnett dataset, filtering, region split table, features table, data-quality notes, dropped cells |
| 3. Method and key decisions | Decision → why table; selected RF settings |
| 4. Results | CV table, Iberia table with CIs, paired differences, confusion matrix + figure |
| 4.1 Extra checks | Other classifiers; feature subsets |
| 4.2 Overfitting | Train / CV / Iberia table |
| 4.3 Other regions | 8-region table; why the British Isles fail; relative-feature experiment |
| 4.4 Other continents | USA / China / India table and explanation |
| 5. SHAP findings | Importance values, 4 SHAP figures, interpretation per class |
| 6. "I don't know" | Rules A and B, results, what worked |
| 7. Limitations | Data coverage, Hybrid definition, absolute wind speed, type not suitability |
| 7.1 Conclusion | Findings in bullets, next steps |
| 7.2 New locations | The assumption, why reasonable, why only partly true, how to use |
| 8. Reproduce | Commands in order |
| 9. References | Table: reference → what we used it for (papers + data sources) |

**Where every number comes from:** `outputs/*.csv` (results, cv_scores, region_holdout,
other_continents, overfit_check, dont_know_summary, feature_importance, relative_features_experiment).

**How it is built:** written in Markdown (`REPORT_NOTES.md`) → converted to Word with
`tools/md2docx.js` → exported to PDF with Microsoft Word.

**References (and why each):**

| Reference | Used for |
|---|---|
| Dunnett et al. 2020, *Scientific Data* 7:130 | All labels |
| Breiman 2001, *Machine Learning* 45:5–32 | Random Forest |
| Sachit et al. 2022, *ISPRS IJGI* 11(8):422 | RF + SHAP precedent; features; existing plants as evidence |
| Roberts et al. 2017, *Ecography* 40:913–929 | Spatial cross-validation |
| Meyer & Pebesma 2021, *Methods Ecol. Evol.* 12:1620–1633 | "I don't know" Rule B |
| Lundberg & Lee 2017 (NeurIPS); Lundberg et al. 2020 (*Nat. Mach. Intell.*) | SHAP |
| Arderne et al. 2020, *Scientific Data* 7:19 | Gridfinder grid lines |
| Horn 1981, *Proc. IEEE* 69:14–47 | Slope formula |
| Efron & Tibshirani 1993; Pedregosa et al. 2011 | Bootstrap; scikit-learn |
| Global Solar Atlas, Global Wind Atlas (Davis et al. 2023), Copernicus DEM, GADM | Data layers |

"Wu et al. 2026" from the original brief was **removed**: not used and not verifiable.

### 4B. The presentation (`outputs/Project_Presentation.pptx`, 25 slides)

| # | Slide | Its one message |
|---|---|---|
| 1 | Which renewable fits here? | Title |
| 2 | The question | Problem + the four possible answers; type, not "build here" |
| 3 | Data: real installations from OpenStreetMap | 68,786 → 24,078 → 18,705 → 10,724 |
| 4 | Train on Europe, test on an unseen region | Map + split table + why Iberia |
| 5 | Five features, and three we left out | Features vs leakage |
| 6 | Pipeline: six simple scripts | Steps + validation rules |
| 7 | Three baselines and one main model | Why each model; metrics |
| 8 | Result: 90% accuracy on the unseen region | Chart; +1.8 points over the wind rule |
| 9 | Solar and Wind transfer well; Hybrid does not | Confusion matrix |
| 10 | Is the Random Forest the right choice? | Other classifiers; feature subsets |
| 11 | Is the Random Forest overfitting? | 0.81 / 0.73 / 0.90; Iberia is easier |
| 12 | Does it hold in other regions? Only in the south | 8 regions; 71%; why the UK fails |
| 13 | Other continents: USA works, China partly, India fails | 90 / 84 / 53 |
| 14 | What drives the predictions (SHAP) | Wind first, sun second; why Hybrid fails |
| 15 | Saying "I don't know" | 40%; Rule A works, Rule B cautious |
| 16 | Improvement round: teaching the model local context | Locked test first; 8 new features |
| 17 | Before and after | 11 held-out regions 0.736 → 0.794; British Isles +18 |
| 18 | Which model, and what drives it now | Extra Trees wins; NNs don't; wind share 51% → 23% |
| 19 | Better "I don't know" and four types of place | Conformal sets; archetypes |
| 20 | Locked final test | All models tie ~0.9; conformal 94% coverage |
| 21 | Using the model on a new location | The assumption and how to use it |
| 22 | How we checked the results | Fake wind data, islands, reproducible, no leakage |
| 23 | Limitations | Honest limits |
| 24 | What we found | Answers + next steps |
| 25 | References | Reference → why table |

- Every slide has **speaker notes** (in PowerPoint: View → Notes).
- **Timing:** ~1 minute per slide → 22–25 minutes; keep slides 8, 12, 17 and 20 for the main emphasis.
- **Design:** Cambria headings, Calibri text; same class colours as the websites; native,
  editable charts on slides 8, 10, 11, 12.
- Pictures of every slide are in the share zip (`photos/slides/`).

### 4C. How to present (suggested speaking order)

| Slides | Speaker |
|---|---|
| 1 (title, opening) | Keshav Sudarshan |
| 2–5 (question, data, split, features) | Ashutosh Singh |
| 6 (pipeline) | Keshav Sudarshan |
| 7–8 (models, Iberia result) | Astitva Tripathi |
| 9 (confusion matrix: Solar and Wind transfer, Hybrid does not) | Madhusudan |
| 10 (is Random Forest the right choice?) | Astitva Tripathi |
| 11–13 (overfitting, regions, other continents) | Divyanshu Puri |
| 14 (SHAP) | Keshav Sudarshan |
| 15 ("I don't know") | Aryan Singh |
| 16 (improvement round: local-context features) | Ashutosh Singh |
| 17 (before and after) | Madhusudan |
| 18 (which model and what drives it) | Astitva Tripathi |
| 19 (conformal "I don't know" and archetypes) | Aryan Singh |
| 20 (locked final test) | Divyanshu Puri |
| 21 (using the model on a new location) + live website demo | Madhusudan |
| 22 (how we checked the results) | Aryan Singh |
| 23 (limitations) | Aryan Singh |
| 24 (what we found, closing) | Keshav Sudarshan |
| 25 (references) | Aryan Singh |

### 4D. Viva questions for Aryan Singh and Keshav Sudarshan
- What's the main finding in one sentence? · What are the limitations? · Why these references? ·
  What does the model *not* claim? (It never says "build here".) · What would you do next?
  (Relative/local wind context, more Hybrid examples, calibrated threshold, label check with Global
  Energy Monitor.)

---

## Part 5 — Lessons the whole team learnt
1. Random splits cheat on map data; hold out whole regions.
2. Always compare with simple baselines; the wind rule got 88% on its own.
3. Check inputs against known values (the fake wind file).
4. One test region isn't enough (Iberia 90%, all regions 71%).
5. High scores can come from leakage; avoid features the target itself changes.
6. Explain failures (UK, India) instead of hiding them.
7. "I don't know" is valuable exactly where the model is out of its depth.
