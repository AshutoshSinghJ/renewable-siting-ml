# Report notes: Predicting renewable installation type from geography

## 1. Problem statement

Given a location where an onshore, utility-scale renewable installation exists, can a model
trained on geographic and environmental factors (solar resource, wind resource, terrain, grid
proximity) predict whether it is a **Solar, Wind or Hybrid** site, including in a region it has
never seen during training, and say **"I don't know"** when it is unsure or the location is unlike
anything it was trained on?

The model answers *"if something is built here, which type fits?"*. It does **not** judge whether
a location should be developed, and it makes no claim that any location is "unsuitable".

**Short answer:** partly. Solar vs Wind transfers well to sunny southern regions (90–92% accuracy
on Iberia, Italy, and the Balkans and Greece) but fails in windy northern regions (35% on the
British Isles, 40% on the Nordic and Baltic countries); across all eight European regions held out
in turn the model is right 71% of the time. Applied unchanged to other continents, it scores 90%
on the USA, 84% on China (barely above always guessing Wind, 81%) and only 53% on India, where it
correctly answers "I don't know" for 92% of cells.
An improvement round (Section 8) added local-context features and wider training: on 11 held-out
regions accuracy rose from 0.736 to 0.794 and the worst region from 0.354 to 0.536; on five locked,
untouched countries every model ties at about 0.89-0.91 because those countries are easy.
Hybrid is not learned (too few, geographically concentrated examples). "I don't know" partly works
(the low-confidence rule works; the unfamiliar-location rule is too cautious in Iberia but catches
India).

## 2. Data

**Labels: Dunnett et al. (2020), Scientific Data 7:130** (OpenStreetMap-based, 2020 snapshot).
- 35,272 solar and 33,514 wind records. Kept only records that are not flagged urban or water and
  have estimated capacity ≥ 1 MW. Capacity is only estimated for records that pass the
  authors' own "farm" filter (wind > 4 turbines; solar panel area > 1 ha), so missing capacity
  means "not a farm" and those records were dropped. This leaves 11,498 solar and 12,580 wind
  installations.
- 5 km grid in World Eckert IV (equal-area). Cell = Solar only / Wind only / **Hybrid = both in
  the same 5 km cell**. Empty cells are not used.
- Global: 18,705 cells (Wind 9,609, Solar 8,700, Hybrid 396). Hybrid ≥ 300, so it stays a class.

**Region split (confirmed by user)**
| Split | Region | Hybrid | Solar | Wind | Total |
|---|---|---|---|---|---|
| Train | Europe except Spain and Portugal (33 countries) | 308 | 5,320 | 3,548 | 9,176 |
| Test (held out) | Spain + Portugal | 25 | 758 | 765 | 1,548 |

68 cells were dropped for missing or invalid features:
- 34 northern wind cells above the solar atlas's limit (~65°N);
- 13 on the Azores/Madeira (not in the official Portugal wind file);
- 20 island cells with no Gridfinder lines (15 Canary Islands, 5 small Mediterranean islands),
  where the "nearest line" was on another coast 20–450 km away;
- 1 in Kosovo (no wind file available).

**Features**
| Feature | Source | How |
|---|---|---|
| `ghi` | Global Solar Atlas v2 GHI, 9 arc-sec | Sampled at each installation, averaged per cell |
| `wind100` | Global Wind Atlas, mean wind speed 100 m, official per-country GeoTIFFs (250 m) | Same |
| `elev` | Copernicus DEM GLO-90 | Same |
| `slope` | Horn slope from the same DEM (3×3 window) | Same |
| `dist_grid` | Gridfinder (Arderne et al. 2020): nearest grid line, mostly medium/high voltage | Cell centre to nearest line, km, in EPSG:3035 |

**Data-quality notes**
- The global wind-speed rasters found in the old `01_raw` folder (`AUTHENTIC_/VERIFIED_/GWA_v4.0_*WS100*`)
  did not match the official Global Wind Atlas. For example, they gave the Gujarat coast 4.7 m/s
  against the official 6.6 m/s, and ranked Bavaria windier than the German North Sea coast. They
  were replaced with official per-country downloads.
- Gridfinder has no lines on some islands; those cells are excluded (see above). Every cell more
  than 20 km from a line was such an island cell.

**Dataset limitations:** OSM coverage is uneven (dominated by Europe, North America and East Asia);
the data is a 2020 snapshot; capacity values are first-order estimates.

## 3. Method and key decisions

| Decision | Why |
|---|---|
| Train only on real installations (no "empty land" negatives) | Every label is an observed fact. "No plant" ≠ "unsuitable". |
| One model, 3 classes (Solar / Wind / Hybrid) | Same features and pipeline for all; simple to explain. |
| Hybrid = a 5 km cell containing both solar and wind | Consistent, reproducible. Depends on cell size. |
| Features: GHI, wind speed 100 m, elevation, slope, distance to grid | Core resource, terrain and grid factors from published siting studies (Sachit et al. 2022). |
| No land cover or road distance | The farm itself changes land cover and brings access roads (leakage). |
| Capacity only as a filter (≥ 1 MW), never a feature | Utility-scale only; as a feature the model would guess type from plant size (leakage). |
| Spatial split: whole held-out region + spatial block CV (100 km blocks, GroupKFold k=5) | Nearby sites are similar; random splits inflate scores (Roberts et al. 2017). |
| Baselines: majority class, wind-speed rule, logistic regression | Shows whether RF learned more than "windy = wind". |
| Random Forest, `class_weight="balanced"`, 300 trees, small grid (max_depth, min_samples_leaf) | Standard for tabular geospatial data, explainable with SHAP. |
| Metrics: macro F1, per-class recall, confusion matrix (accuracy secondary), bootstrap 95% CIs | Accuracy hides poor performance on the rare Hybrid class; CIs show what is real vs noise. |
| SHAP (TreeExplainer) | Shows which factor drives each class. |
| "I don't know" = low confidence OR unfamiliar location | Model should not guess outside what it learned (Meyer & Pebesma 2021). |

`random_state=42` everywhere. The test region was used once, for the final evaluation only. The
full pipeline was re-run from scratch and reproduced every output file exactly.

**Selected Random Forest:** `max_depth=10`, `min_samples_leaf=1`, 300 trees. The six grid settings
scored 0.522–0.543 in CV. The selected setting's CV score is slightly optimistic, because the same
folds were used to choose it, but the spread between settings is small.

## 4. Results

**Spatial CV, training region only (macro F1, mean ± sd over 5 block folds)**
| Model | CV macro F1 |
|---|---|
| Majority class | 0.244 ± 0.015 |
| Wind-speed rule | 0.498 ± 0.017 |
| Logistic regression | 0.477 ± 0.036 |
| **Random Forest** | **0.543 ± 0.023** |

**Held-out test region: Spain + Portugal (n = 1,548), with 95% bootstrap CIs (2,000 resamples)**
| Model | Accuracy (95% CI) | Macro F1 (95% CI) | Recall Hybrid | Recall Solar | Recall Wind | Precision Hybrid |
|---|---|---|---|---|---|---|
| Majority class | 0.490 (0.465–0.516) | 0.219 (0.212–0.227) | 0.00 | 1.000 | 0.000 | – |
| Wind-speed rule (Wind if ≥ 6.60 m/s) | 0.882 (0.866–0.899) | 0.593 (0.582–0.603) | 0.00 | 0.926 | 0.868 | – |
| Logistic regression | 0.844 (0.826–0.861) | 0.641 (0.618–0.666) | 0.56 | 0.836 | 0.860 | 0.07 |
| **Random Forest** | **0.901 (0.886–0.915)** | 0.627 (0.599–0.675) | 0.04 | 0.918 | 0.911 | 0.14 |

**Paired bootstrap differences (same resampled cells for both models)**
| Comparison | Accuracy diff (95% CI) | Macro F1 diff (95% CI) |
|---|---|---|
| RF − wind-speed rule | +0.018 (+0.005 to +0.032) | +0.034 (+0.006 to +0.082) |
| RF − logistic regression | +0.057 (+0.043 to +0.071) | −0.014 (−0.051 to +0.035) |

**Confusion matrix, RF, test region:**

| True \ Pred | Hybrid | Solar | Wind |
|---|---|---|---|
| Hybrid | 1 | 6 | 18 |
| Solar | 3 | 696 | 59 |
| Wind | 3 | 65 | 697 |

![Random Forest confusion matrix on the test region](confusion_rf.png)

**Interpretation**
- Solar vs Wind transfers well to Iberia: 92% / 91% recall with RF. Section 4.3 shows this does
  not hold for every region.
- **RF beats the simple wind-speed rule, but only slightly.** It gains +1.8 accuracy points and
  +0.034 macro F1, and both confidence intervals are above zero, so the gain is real but small.
  Most of what the model knows is "windy = wind". This is reported as a valid finding.
- **RF vs logistic regression:** RF is clearly more accurate (+5.7 points). Their macro F1 is
  statistically tied. LR's higher Hybrid recall (0.56) comes from over-predicting Hybrid: it
  labels 199 test cells Hybrid and only 14 are correct (precision 0.07).
- **Hybrid is not really learned by any model** (RF: 1 of 25). There are only 25 test Hybrid
  cells, so Hybrid numbers are noisy, which is why the RF macro F1 CI is wide.

### 4.1 Extra checks: is the Random Forest the right choice? (training-region CV only)

These checks use the same spatial CV and never touch the test region (`src/03b_extra_checks.py`).

**Other standard classifiers**
| Model | CV macro F1 | CV accuracy |
|---|---|---|
| **Random Forest (main model)** | **0.543 ± 0.023** | 0.732 |
| Gradient boosting (HistGradientBoosting) | 0.542 ± 0.023 | 0.763 |
| Extra Trees | 0.537 ± 0.023 | 0.738 |
| k-nearest neighbours (k=15) | 0.510 ± 0.009 | 0.763 |
| Logistic regression | 0.477 ± 0.036 | 0.604 |
| SVM (RBF) | 0.476 ± 0.027 | 0.576 |

Tree ensembles are tied at the top, and gradient boosting gives no gain. The limit is the data
(few, concentrated Hybrid examples and a climate shift), not the choice of algorithm.

**Feature subsets (Random Forest)**
| Features | CV macro F1 |
|---|---|
| wind100 only | 0.503 |
| + ghi | 0.524 |
| + terrain (elev, slope) | 0.540 |
| + dist_grid (all 5) | 0.543 |

Wind speed alone gives most of the performance. GHI and terrain add a little each, and grid
distance adds almost nothing. That also means grid distance is not leaking information (for
example, through a farm's own connection line appearing in the grid map).

### 4.2 Is the Random Forest overfitting? (`outputs/overfit_check.csv`)

| Random Forest scored on | Accuracy | Macro F1 |
|---|---|---|
| Training data (seen) | 0.812 | 0.701 |
| Spatial CV: unseen 100 km blocks of the training region | 0.732 | 0.545 |
| Iberia test region (unseen) | 0.901 | 0.627 |

- **No heavy overfitting.** A Random Forest without a depth limit scores 1.000 on its own training
  data, which is what memorisation looks like. The selected model (max depth 10) scores 0.812, and
  drops only to 0.732 on unseen blocks.
- **Iberia scores higher than CV because Iberia is an easier region,** not because of leakage. The
  simple wind-speed rule shows the same jump: 0.742 in the training region, 0.882 in Iberia. In
  Iberia, wind farms stand on windy ridges and solar farms on calm, sunny plains, so wind speed
  separates them cleanly.

### 4.3 Does the result hold in other regions? (`outputs/region_holdout.csv`)

One test region can be lucky or unlucky, so Europe was split into 8 regions and each was held out
in turn: the model was trained on the other 7 and scored on the held-out one
(`src/03c_region_holdout.py`). Nothing was tuned: the Random Forest uses the settings chosen in
Step 3, and the wind-speed rule's threshold is refit on each training part. The Iberia round uses
the same training data as the main experiment and reproduces its result exactly (0.901).

| Held-out region | Cells | Hybrid | RF accuracy | RF macro F1 | Rule accuracy | RF recall Solar | RF recall Wind |
|---|---|---|---|---|---|---|---|
| Italy | 1,642 | 34 | **0.920** | 0.573 | 0.875 | 0.977 | 0.726 |
| Balkans & Greece | 547 | 6 | **0.905** | 0.595 | 0.896 | 0.927 | 0.885 |
| Iberia | 1,548 | 25 | **0.901** | 0.627 | 0.882 | 0.918 | 0.911 |
| Central & Eastern | 1,033 | 2 | 0.788 | 0.497 | 0.930 | 0.856 | 0.522 |
| France & Benelux | 1,163 | 15 | 0.768 | 0.553 | 0.801 | 0.621 | 0.845 |
| Germany & Austria | 3,133 | 215 | 0.613 | 0.472 | 0.699 | 0.719 | 0.568 |
| Nordic & Baltic | 538 | 17 | **0.396** | 0.307 | 0.712 | 0.307 | 0.413 |
| British Isles | 1,120 | 19 | **0.350** | 0.205 | 0.349 | 0.006 | 0.987 |

All 8 regions pooled (every cell predicted by a model that never saw its region): accuracy 0.712,
macro F1 0.523.

- **The result depends strongly on the region.** Sunny southern regions score 90–92%; windy
  northern regions fail. Iberia, the planned test region, turned out to be easier than average.
- **Why the British Isles fail:** solar farms in the UK and Ireland stand at a median wind speed
  of 7.7 m/s, windier than the typical wind farm elsewhere in Europe (7.4 m/s). The model learned
  "windy means wind farm" in absolute terms, so it labels almost every British solar farm as wind
  (Solar recall 0.006).
- **The simple wind-speed rule beats the Random Forest in four regions** (Central & Eastern,
  France & Benelux, Germany & Austria, Nordic & Baltic). The Random Forest wins clearly only in the
  south. Its extra flexibility helps where the training data looks like the test region and hurts
  where it does not.
- **Tried fix: relative features** (`src/03e_relative_features.py`). Adding each site's wind
  speed and sunshine minus its country's median raised the pooled accuracy only from 0.712 to
  0.733 (British Isles 0.35 → 0.38, but Italy 0.92 → 0.89) and lowered macro F1 slightly, so the
  main model keeps its 5 features. A better fix needs local context, such as wind speed compared
  with the surrounding 50 km of land (future work).

### 4.4 Other continents: USA, China and India (`outputs/other_continents.csv`)

The Random Forest trained in Step 3 (Europe without Iberia) was applied **unchanged** to the USA,
China and India: no retraining, no tuning. Features were built exactly as in Step 2 (official
per-country Global Wind Atlas files, the same Global Solar Atlas, Copernicus DEM and Gridfinder
layers), and the "I don't know" rules used the same thresholds as Step 5
(`src/03d_other_continents.py`). Cells with no Gridfinder line within 20 km or a missing value were
dropped (USA 16, China 13).

| | USA | China | India |
|---|---|---|---|
| Cells (Solar / Wind / Hybrid) | 2,326 (855 / 1,470 / 1) | 2,300 (407 / 1,853 / 40) | 777 (208 / 566 / 3) |
| Always guessing the most common class | 0.632 | 0.806 | 0.728 |
| **Random Forest accuracy** | **0.896** | **0.839** | **0.533** |
| Wind-speed rule accuracy | 0.927 | 0.823 | 0.541 |
| RF recall Solar / Wind | 0.844 / 0.927 | 0.737 / 0.880 | 1.000 / 0.364 |
| RF recall Hybrid | 0 of 1 | 0 of 40 | 0 of 3 |
| "I don't know" (share of cells) | 49.6% | 76.0% | **92.1%** |
| Accuracy on the cells it answers | 0.871 | 0.730 | 0.770 |
| Median wind speed: solar / wind farms (m/s) | 5.7 / 8.1 | 5.3 / 7.7 | 5.5 / 6.4 |

- **USA: the model transfers.** US wind farms sit in very windy places (median 8.1 m/s, mostly the
  Great Plains) and solar farms in calm, sunny ones, so wind speed separates them as cleanly as in
  southern Europe. As in Europe, the simple wind-speed rule does slightly better (0.927).
- **India: the model fails, but it knows.** Indian wind farms sit at a median 6.4 m/s, below the
  roughly 6.6 m/s that counts as "windy" in the European training data, so almost every Indian
  site is predicted Solar (Wind recall 0.364). The result, 0.533, is worse than always guessing
  Wind (0.728). But India is far sunnier than any training cell (median GHI 5.4 kWh/m²/day), so
  the unfamiliar-location rule flags 92% of Indian cells as "I don't know". In Iberia that rule
  looked over-cautious; India is the situation it is designed for.
- **China: reasonable, but mostly wind speed and class balance.** Chinese wind farms are much
  windier than solar farms (7.7 vs 5.3 m/s), so 0.839 accuracy follows. But 81% of Chinese cells
  are Wind, so always guessing Wind already scores 0.806, and the wind rule 0.823: the Random
  Forest adds only 1.6 points. Hybrid: 0 of 40, the largest Hybrid test outside Europe, which
  confirms Hybrid is not learned. Here the "I don't know" rule does not help: it abstains on 76%
  of cells, yet the answered cells are less accurate (0.730) than all cells together.
- Together with Section 4.3, this confirms that the model's main limitation is absolute wind
  speed: it works where wind farms are much windier than solar farms (southern Europe, the USA,
  largely China) and fails where they are not (British Isles, India).
- Macro F1 is not compared for the USA and India because they have only 1–3 Hybrid cells; China's
  RF macro F1 is 0.514.

## 5. SHAP findings (`outputs/shap_*.png`, `outputs/feature_importance.csv`)

Mean |SHAP|, averaged over classes: **wind100 0.166** > ghi 0.070 > elev 0.035 > slope 0.030 > dist_grid 0.023.

![Mean |SHAP| per feature and class](shap_bar.png)

![SHAP summary for Solar](shap_summary_solar.png)

![SHAP summary for Wind](shap_summary_wind.png)

![SHAP summary for Hybrid](shap_summary_hybrid.png)

- **Solar:** low wind speed is the strongest push towards Solar; high GHI adds to it.
- **Wind:** the mirror image, mainly high wind speed.
- **Hybrid:** medium-to-high wind and *low* GHI raise P(Hybrid). In training, Hybrids are
  concentrated in northern Germany and Denmark (213 of 308 training Hybrids are in Germany). The
  model therefore learned "Hybrid = windy, not very sunny", which does not hold in sunny Iberia
  (test GHI median 4.6 vs train 3.1 kWh/m²/day). This explains the very low Hybrid recall.
- Terrain and grid distance contribute, but much less than the two resource variables. This agrees
  with the feature-subset check.

## 6. "I don't know" (`outputs/dont_know_summary.csv`, `outputs/test_predictions.csv`)

*Original study. In the improvement round (Section 8.8) both rules were replaced by conformal
prediction sets, and Rule B was dropped.*

- Rule A: max class probability < 0.5.
- Rule B: weighted (RF importance) standardised feature distance to the nearest training cell is
  above 0.141, the 95th percentile of the same distance for training cells measured against other
  spatial CV folds.

| Subset | Cells | Accuracy | Macro F1 |
|---|---|---|---|
| All test cells | 1,548 | 0.901 | 0.627 |
| Confident only | 925 (59.8%) | 0.938 | 0.626 |

- "I don't know" for **40.2%** of test cells: 10.2% low confidence, 32.1% unfamiliar (some overlap).
- By true class: Hybrid 64%, Wind 56%, Solar 24%.
- **Rule A works as intended:** low-confidence cells would have been only ~45–56% correct.
- **Rule B is too cautious here:** the 465 cells flagged only as unfamiliar would have been
  97% correct. Iberia's GHI and elevation lie mostly outside the training range (test median
  elevation 717 m vs 148 m), so many cells look unfamiliar. But the decision relies mainly on wind
  speed, which transfers well. Rule B correctly warns that the test region is outside the training
  conditions; in this case that did not translate into wrong predictions.

## 7. Limitations

- OSM-based labels are incomplete and regionally uneven; the data is a 2020 snapshot; capacity is
  estimated.
- Hybrid depends on the 5 km cell definition: co-location, not necessarily a joint project. Only
  308 training and 25 test Hybrid cells, mostly from one country (Germany).
- Training and test climates differ strongly (Iberia is sunnier and higher), which is the point of
  the test, but it limits how much transfer can be expected for Hybrid.
- Features are averaged over the installations in a cell; distance to grid uses the cell centre.
  Gridfinder is partly predicted, covers mostly medium/high-voltage lines, and misses some islands
  (those cells are excluded).
- The model predicts technology type *given* an installation exists. It does not judge whether
  a location should be developed.
- The simple wind-speed rule performs close to the Random Forest on Iberia (+1.8 accuracy points)
  and beats it in four of the eight held-out regions.
- **The model does not transfer to windy northern regions** (35% on the British Isles, 40% on the
  Nordic and Baltic countries) or to India (53%), because it relies on absolute wind speed. It
  does transfer to the USA (90%) and partly to China (84%, close to always guessing Wind).
  The improvement round (Section 8) partly fixes this: British Isles 54%, Nordic & Baltic 64%.
- The unfamiliar-location rule is a rough safety net: it caught India, but in China the cells it
  answered were less accurate than average.
- The Rule B threshold (95th percentile) is a convention, not tuned; it flagged many cells the
  model got right (Rule B was dropped in Section 8).

## 7.1 Conclusion

- Geography separates solar from wind farms well in sunny southern Europe (90–92% on three
  different held-out regions), mostly through wind speed.
- The same model fails in windy northern Europe, where solar farms are built in windy places too.
  Across all eight held-out regions it is right 71% of the time.
- On other continents it works for the USA (90%), partly for China (84%, only 3 points above always
  guessing Wind) and fails for India (53%), where Indian wind farms are less windy than European
  ones; there it answers "I don't know" for 92% of cells.
- Hybrid sites are not learned with this data.
- The model can say "I don't know": the low-confidence rule works; the unfamiliar-location rule is
  too cautious in Iberia but correctly catches India.
- Improvement round: local-context features raised held-out accuracy from 0.736 to 0.794 (British
  Isles +18 points); conformal sets kept 94% coverage on five locked countries, where all models tie.
- Next steps: a convolutional network on terrain and wind maps, harder untouched test regions, more Hybrid examples from other continents,
  and a label cross-check with Global Energy Monitor.

## 7.2 Using the model on new locations: an explicit assumption

The model is trained only on places where farms were actually built. On its own it therefore
answers *"given a farm exists here, which type is it?"*. To use it on a **new** location, one
further assumption is needed:

> **Assumption:** existing utility-scale installations reflect well-informed siting decisions.
> Under this assumption, the model recommends which technology (Solar, Wind or Hybrid) best fits
> a location *if it is developed*, and answers "I don't know" when the location is unlike the
> places it learned from or the evidence is unclear.

**Why the assumption is reasonable:** developers assess wind and solar resource, terrain and grid
access before building, so real installations carry that expert information. The same idea, using
existing plants as evidence of suitable sites, underlies Sachit et al. (2022).

**Why it is only partly true (evidence from this project):**
- Siting also follows land prices, subsidies, permits and existing grid connections. UK solar
  farms stand in windy places (median 7.7 m/s) for those reasons, not because windy land suits solar.
- Accuracy depends strongly on the region (35% British Isles to 92% Italy; 54% to 93% after the
  improvement round), so the recommendation
  is only trustworthy where the model was shown to work.
- OpenStreetMap covers some countries much better than others, so "expert decisions" are mostly
  European, American and Chinese ones.

**How to use it correctly:**
1. Read the output as "which type fits *if* something is built here", never "build here". The
   model never saw unsuitable land, so it cannot flag a protected forest, a city or a lake.
2. Respect "I don't know": if the answer set holds more than one type (Section 8.8), do not use
   the prediction.
3. Trust it most in regions resembling those where it scored well (southern Europe, the USA, and
   the five locked countries).

## 8. Improvement round: better generalisation

The original model relied on **absolute** wind speed and failed where solar farms also stand in
windy places (British Isles 35%, India 53%). The improvement round tried to fix that, under strict
rules so the result stays honest.

### 8.1 Rules: a locked test set first
- **Before** any experiment, five countries never used before were **locked** as a final test:
  Japan, Turkey, Chile, Australia, South Africa (1,072 cells: 684 Solar, 381 Wind, 7 Hybrid;
  `data/processed/final_test_lock.json`, committed to git before any result).
- Every choice (features, model, thresholds) was made on **leave-one-region-out over 11 regions**:
  the 8 European regions + USA, China and India (16,127 cells). Each region is predicted by a model
  trained on the other 10.
- Selection score fixed in advance: **mean accuracy across regions** (rewards working everywhere,
  not just in the largest regions), with macro F1 as tie-breaker.
- The locked countries were scored **once**, at the end. Nothing was changed after seeing them.

### 8.2 New features (`src/06_features_v2.py`)
| Feature | How | Why |
|---|---|---|
| `wind_pct`, `wind_rel` | Share of land within 30 km that is calmer than the site; site minus the 30 km mean (Global Wind Atlas at ~2 km) | Developers pick the windiest spot **locally**; works even where all land is windy |
| `ghi_pct`, `ghi_rel` | Same for sunshine | Same idea for solar |
| `elev_rel`, `rugged` | Site height minus the mean within 2.5 km; spread of heights (Copernicus DEM) | Wind farms sit on ridges and hilltops |
| `dist_town` | km to the nearest town or city (GHS settlement map, classes 21-30) | Solar is built near demand |
| `dist_water` | km to the nearest sea or large lake (GHS class 10) | Coastal wind sites |

None of these is changed by building a farm, so there is no leakage. Medians on the development
data: wind farms sit at the **84th percentile** of local wind (solar 44th), **15 m above** the
surrounding terrain (solar 2 m below), and **12 km** from a town (solar 5 km).

### 8.3 Which features help? (Random Forest, 11 held-out regions)
| Feature set | Accuracy | Macro F1 | Mean region accuracy | Worst region |
|---|---|---|---|---|
| A. base 5 (current) | 0.736 | 0.536 | 0.706 | 0.354 (British Isles) |
| B. base + local wind/sun context | 0.766 | 0.547 | 0.737 | 0.440 (British Isles) |
| C. B + terrain context | 0.774 | 0.553 | 0.743 | 0.442 (Nordic & Baltic) |
| D. C + town/water distance (all 13) | 0.780 | 0.552 | 0.756 | 0.494 (British Isles) |
| E. D without absolute wind/sun | 0.724 | 0.516 | 0.712 | 0.448 (Nordic & Baltic) |

Every group helps. Removing absolute wind and sunshine (set E) hurts overall but lifts the British
Isles to 0.754: in windy countries "windier than nearby" is more useful than "windy".

### 8.4 Which model? (13 features, 11 held-out regions; `src/07`, `src/10`)
| Model | Accuracy | Macro F1 | Mean region accuracy | Worst region |
|---|---|---|---|---|
| Majority class | 0.419 | 0.220 | 0.443 | 0.147 |
| Wind-speed rule | 0.776 | 0.522 | 0.761 | 0.349 |
| Logistic regression | 0.670 | 0.530 | 0.634 | 0.230 |
| Random Forest | 0.780 | 0.552 | 0.756 | 0.494 |
| Extra Trees **(final)** | 0.794 | 0.536 | 0.781 | 0.536 |
| Gradient boosting | 0.744 | 0.553 | 0.718 | 0.428 |
| Neural network (MLP) | 0.763 | 0.516 | 0.752 | 0.492 |
| Neural network on raw map patches (247 inputs) | 0.738 | 0.506 | 0.721 | 0.434 |

- **Extra Trees** (extremely randomised trees) wins on the pre-set score and on the worst region, and
  becomes the final model. It is a more randomised Random Forest, designed to overfit less.
- **Neural networks did not beat it.** An MLP on the 13 features scored 0.763; an MLP that learns
  local context by itself from 11 x 11 raw map patches (247 inputs) scored 0.738. It did learn some
  context (British Isles 0.434 vs 0.354), but tree models on good features win on ~16,000 tabular rows.
- **Covariate-shift reweighting** (training cells weighted to resemble the held-out region) gave no
  gain (0.779 vs 0.780) and was not adopted.
- The one-line wind rule is still close on accuracy (0.776): wind remains the main signal.

### 8.5 Before and after, by held-out region
| Region | Original 5-feature RF | Improved model | Change (points) |
|---|---|---|---|
| British Isles | 0.354 | 0.536 | +18 |
| India | 0.607 | 0.611 | +0 |
| Nordic & Baltic | 0.364 | 0.643 | +28 |
| Germany & Austria | 0.602 | 0.691 | +9 |
| France & Benelux | 0.635 | 0.733 | +10 |
| China | 0.849 | 0.822 | -3 |
| Central & Eastern | 0.752 | 0.890 | +14 |
| Iberia | 0.893 | 0.907 | +1 |
| Balkans & Greece | 0.905 | 0.909 | +0 |
| Italy | 0.899 | 0.920 | +2 |
| USA | 0.904 | 0.926 | +2 |

Overall: **0.736 → 0.794**; mean across regions 0.706 → 0.781; worst region **0.354 → 0.536**.

### 8.6 What drives the improved model (SHAP, `src/09_patterns.py`)
| Feature | Share of total SHAP importance |
|---|---|
| Wind speed (absolute) | 23% |
| Local wind percentile (new) | 17% |
| Sunshine | 10% |
| Wind vs local mean (new) | 7% |
| Distance to town (new) | 7% |
| Height vs surroundings (new) | 6% |
| Elevation | 5% |
| Distance to grid | 5% |

Absolute wind speed fell from about **half** of the model's reasoning to **23%**. In the British
Isles the local wind percentile is the most important feature.

![How wind speed pushes the prediction towards Wind, by region](v2_shap_wind_dependence.png)

### 8.7 Siting archetypes
K-means on geography only (no labels), k = 4 chosen by silhouette (0.15, so the groups overlap):

| Archetype | Built | Model accuracy | Main regions |
|---|---|---|---|
| Calm, sunny lowland | 82% Solar | 85% | Germany, Italy, Central Europe |
| Windy, very sunny upland | 71% Wind | 84% | USA, China, Iberia |
| Very windy, sunny mountain | 94% Wind | 94% | China, Iberia, USA |
| **Windy, cloudy lowland** | **44% Solar, 52% Wind** | **64%** | Germany, British Isles, France |

The hardest place is the windy, cloudy lowland, where both technologies really are built side by side.

![Siting archetypes](v2_archetypes_map.png)

### 8.8 A better "I don't know": conformal prediction (`src/08_conformal.py`)
Split conformal prediction turns probabilities into answer **sets** with a target error rate: the
threshold is the 90th percentile of (1 - probability of the true class) on held-out-region
predictions, and every class within it enters the set. A set like Solar-or-Wind means "unsure".

| Calibration | True answer in set | Avg set size | Single answer | Right when single |
|---|---|---|---|---|
| Per class (Mondrian) | 0.888 | 1.98 | 21% | 0.935 |
| **Overall (chosen)** | **0.895** | **1.27** | **73%** | **0.859** |

Coverage drops where regions differ most (British Isles 0.72, India 0.79): the guarantee assumes new
places resemble the calibration places. The old "unfamiliar place" rule (Rule B) was dropped: cells it
flagged were **more** accurate (0.93) than the rest (0.79), so it did not identify errors.

### 8.9 Locked final test (`src/11_final_test.py`, scored once)
1,062 of 1,072 locked cells had all features (10 lacked a grid line within 20 km).

| Model | Accuracy | Macro F1 | Japan | Turkey | Chile | Australia | South Africa |
|---|---|---|---|---|---|---|---|
| Majority class | 0.354 | 0.174 | 0.22 | 0.33 | 0.20 | 0.79 | 0.42 |
| Wind-speed rule (>= 6.47 m/s) | 0.905 | 0.601 | 0.90 | 0.91 | 0.95 | 0.87 | 0.84 |
| Old model: RF, 5 features, Europe only | 0.896 | 0.595 | 0.91 | 0.87 | 0.93 | 0.95 | 0.81 |
| RF, 5 features, all development regions | 0.902 | 0.600 | 0.91 | 0.89 | 0.95 | 0.87 | 0.93 |
| Final: Extra Trees, 13 features, all regions | 0.892 | 0.592 | 0.89 | 0.88 | 0.96 | 0.89 | 0.85 |

- **Every sensible model ties at about 0.89-0.91.** Final vs original: -0.005 (95% CI -0.020 to
  +0.009); final vs wind rule: -0.014 (CI -0.029 to +0.001). The differences are within chance.
- **Why:** the locked countries are "easy". The wind rule scores 0.906 there against 0.776 on the
  development regions: wind farms stand in much windier places than solar farms in all five. The
  improvement targets the hard regions (windy, cloudy lowlands), which these countries do not contain.
- **Conformal sets carried over:** the true answer was in the set for **94%** of locked cells (target
  90%); 85% got a single answer, right 94% of the time; 15% got "Solar or Wind".
- Hybrid: 0 of 7 correct.

### 8.10 Overfitting of the final model
| Scored on | Accuracy |
|---|---|
| Training data (seen) | 0.995 |
| 11 held-out regions | 0.794 |
| Locked countries | 0.892 |

Extra Trees fits its training data almost perfectly; that is how the method works (deep trees with
random splits, then averaging). What matters is unseen data, where it was the best of six models. The
depth-limited Random Forest (0.780 held-out, about 0.8 on training data) is the simpler alternative
with a smaller gap.

### 8.11 What the improvement round shows
1. Local context helps exactly where absolute wind fails: British Isles +18 and Nordic & Baltic +28
   points (held-out regions).
2. Wind's dominance fell from about half to under a quarter of the model's reasoning.
3. On easy regions every model, even a one-line rule, scores about 90%; the gain appears only where
   the problem is genuinely hard.
4. Conformal prediction gives a usable, measured "I don't know" that held up on untouched countries.
5. Deep learning did not beat trees on this tabular problem.

## 9. Reproduce

```
python src/01_prepare_labels.py
python src/02_extract_features.py   # reads Copernicus DEM from AWS (cached in data/processed/dem_points.csv)
python src/03_train_evaluate.py
python src/03b_extra_checks.py      # optional: model comparison + feature subsets (training region only)
python src/03c_region_holdout.py    # optional: each of 8 regions held out in turn + overfitting check
python src/03d_other_continents.py  # optional: Europe-trained model on the USA, China and India (needs their GWA files)
python src/03e_relative_features.py # optional experiment: relative wind/sun features (not adopted)
python src/04_explain.py
python src/05_dont_know.py

# improvement round
python src/06_features_v2.py        # new local-context features (+ base features for the locked countries)
python src/07_global_experiments.py # feature sets and 7 models on 11 held-out regions
python src/08_conformal.py           # conformal prediction sets
python src/09_patterns.py            # siting archetypes and SHAP by region
python src/10_patch_nn.py            # neural network on raw map patches
python src/11_final_test.py          # one-time locked final test
```

## 10. References and what we used them for

**Papers**

| Reference | What we used it for |
|---|---|
| Dunnett, S., Sorichetta, A., Taylor, G. & Eigenbrod, F. (2020). Harmonised global datasets of wind and solar farm locations and power. *Scientific Data* 7, 130. | All labels: solar and wind farm locations, plus the authors' recommended "real farm" filter |
| Breiman, L. (2001). Random Forests. *Machine Learning* 45(1), 5–32. | The main model (Random Forest) |
| Sachit, M. S., Shafri, H. Z. M., Abdullah, A. F. & Rafie, A. S. M. (2022). Global spatial suitability mapping of wind and solar systems using an explainable AI-based approach. *ISPRS Int. J. Geo-Inf.* 11(8), 422. | Precedent for Random Forest + SHAP on real wind and solar plants; choice of resource, terrain and grid features; idea of existing plants as evidence of good sites (Section 7.2) |
| Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40, 913–929. | Why random splits overestimate accuracy on map data; spatial block cross-validation and a held-out test region |
| Meyer, H. & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution* 12, 1620–1633. | Rule B of "I don't know": weighted feature-space distance to the training data and a cross-validation-based threshold |
| Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* 30. | SHAP values (Section 5) |
| Lundberg, S. M. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* 2, 56–67. | TreeExplainer, the fast exact SHAP method for tree models |
| Arderne, C., Zorn, C., Nicolas, C. & Koks, E. E. (2020). Predictive mapping of the global power system using open data. *Scientific Data* 7, 19. | Gridfinder power-line map, used for the distance-to-grid feature |
| Horn, B. K. P. (1981). Hill shading and the reflectance map. *Proceedings of the IEEE* 69(1), 14–47. | The standard slope formula used to compute slope from elevation |
| Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall. | Bootstrap 95% confidence intervals for the test results |
| Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research* 12, 2825–2830. | Software for all models, cross-validation and metrics |

**Data sources**

| Source | What we used it for |
|---|---|
| Global Solar Atlas 2.0 (Solargis, World Bank Group / ESMAP), https://globalsolaratlas.info | Sunshine (GHI) feature |
| Global Wind Atlas (DTU Wind Energy, World Bank Group / ESMAP), https://globalwindatlas.info; described in Davis, N. N. et al. (2023), *Bulletin of the American Meteorological Society* 104(8), E1507–E1525 | Wind speed at 100 m, official per-country files |
| Copernicus DEM GLO-90 (European Space Agency), via https://registry.opendata.aws/copernicus-dem/ | Elevation and slope |
| GADM, Database of Global Administrative Areas, https://gadm.org | Country borders on the website maps only (not used by the model) |
