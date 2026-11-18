# Viva Guide: Which Renewable Fits Here?

Read this once before the viva. Section 1 is the story, section 2 the numbers to remember,
sections 3–6 the "why" behind every choice, section 7 likely questions with short answers.

---

## 1. The story in 60 seconds

> "I asked whether geography alone can tell a solar farm from a wind farm. I used 24,000 real
> farms from OpenStreetMap, put them on a 5 km grid, and described each cell with five
> features: sunshine, wind speed, elevation, slope and distance to the grid. I trained a Random
> Forest on Europe and tested it on Spain and Portugal, a region it never saw: 90% accuracy.
> Then I tested every European region in turn and two other continents. It works where wind
> farms are much windier than solar farms (southern Europe 90–92%, USA 90%) and fails where they
> are not (British Isles 35%, India 53%), because it relies on absolute wind speed. China (84%)
> sits in between: it works, but mostly because 81% of Chinese sites are wind farms. Hybrid sites
> were not learned. The model can also say 'I don't know', and in India it did so for 92% of
> cells, which is exactly when it should."

---

## 2. Numbers to remember

| What | Number |
|---|---|
| Raw records (Dunnett 2020) | 68,786 |
| Real farms after filtering | 24,078 |
| Labelled 5 km cells (world) | 18,705 |
| Cells used in Europe | 10,724 (train 9,176 · test 1,548) |
| Classes in Europe | Solar 6,078 · Wind 4,313 · Hybrid 333 |
| **Iberia test accuracy (RF)** | **0.901** (95% CI 0.886–0.915) |
| Iberia macro F1 | 0.627 |
| Wind-speed rule on Iberia | 0.882 (RF is +1.8 points better, CI +0.5 to +3.2) |
| Solar / Wind / Hybrid recall | 92% / 91% / 1 of 25 |
| Overfitting check | train 0.81 · unseen blocks 0.73 · Iberia 0.90 |
| 8 regions held out, pooled | **71%** (south 90–92%, British Isles 35%, Nordic 40%) |
| USA / China / India (Europe model, unchanged) | **90% / 84% / 53%** (China: always-Wind guess = 81%) |
| "I don't know" on Iberia | 40% of cells; accuracy on the rest 0.938 |
| "I don't know" on India | 92% of cells |
| Most important feature (SHAP) | Wind speed 0.166, then sunshine 0.070 |

---

## 3. How it started and what changed along the way

1. **Brief:** predict Solar / Wind / Hybrid from geography, test on an unseen region, add "I don't know".
2. **Labels first (Step 1):** built 5 km cells, counted classes, chose the split (Europe → Iberia). Hybrid had 396 cells (≥ 300), so it stayed a class.
3. **Problem found: fake wind data.** The wind raster already on disk didn't match the official Global Wind Atlas (Gujarat coast 4.7 vs 6.6 m/s; Bavaria windier than the North Sea coast). Replaced with official per-country files. *Lesson: always sanity-check inputs against known values.*
4. **Features (Step 2):** elevation read online from the Copernicus DEM (3 GB of tiles avoided), grid distance from Gridfinder.
5. **Models (Step 3), SHAP (Step 4), "I don't know" (Step 5).**
6. **Verification pass:** full re-run gave identical outputs; found **island cells with no grid lines** (Canaries: 117–450 km "distance"), removed them; added bootstrap confidence intervals.
7. **Extra checks:** 5 other classifiers and feature subsets (to justify RF), an overfitting check.
8. **"Is Iberia lucky?"** Held out each of 8 regions → discovered the north fails. Investigated: UK solar farms are windier than most European wind farms.
9. **Other continents:** USA works, China works mostly through class balance (Hybrid 0 of 40), India fails but the model flags it as unfamiliar.
10. **Tried relative features** (wind and sun compared with the country median): no consistent gain, so kept the 5 features.

---

## 4. Every design decision and why

| Decision | Why |
|---|---|
| Use only real installations, no "empty land" | Every label is a fact. "No farm here" doesn't mean "unsuitable" (policy, land price, ownership). So the model answers *which type*, never *whether to build*. |
| Dunnett et al. 2020 dataset | Peer-reviewed, global, built from OpenStreetMap, capacity cross-checked against national statistics. |
| Filter: not urban, not water, ≥ 1 MW | Keeps utility-scale farms only. This is the dataset authors' own recommended filter. |
| 5 km grid, equal-area projection | Turns points into comparable units; equal-area means every cell has the same size. Hybrid = both types in one cell (simple, reproducible). |
| Five features | Resource (sun, wind), terrain (elevation, slope), access (grid): the standard siting factors in published studies (Sachit et al. 2022). |
| No land cover, roads or capacity | **Leakage**: the farm changes land cover, brings its own roads, and size would reveal the type. |
| Test on a whole held-out region | Nearby cells are near-copies; a random split would inflate scores (Roberts et al. 2017). |
| Iberia as test region | Connected block, all three classes, touches training only at the Pyrenees, different climate. |
| Spatial block CV (100 km blocks, 5 folds) | Validation that mimics "new place", used only to tune. |
| Baselines (majority, wind rule, logistic regression) | To prove RF learned more than "windy = wind". |
| Random Forest | Standard for tabular geodata, handles non-linear effects, robust, explainable with SHAP. |
| `class_weight="balanced"` | Hybrid is rare (3%); without weights the model ignores it. |
| Macro F1 + per-class recall, not just accuracy | Accuracy hides a failing rare class. |
| Bootstrap 95% CIs | Shows whether differences between models are real or noise. |
| SHAP | Explains which feature drives each class. |
| "I don't know" = low confidence OR unfamiliar place | A model should not guess outside what it learned (Meyer & Pebesma 2021). |
| `random_state=42` everywhere | Reproducible results. |

---

## 5. Key concepts in one line each

- **Data leakage:** the input secretly contains the answer, so test scores look better than real use.
- **Spatial autocorrelation:** nearby places are similar, so random splits cheat.
- **Validation vs test:** validation chooses settings (spatial CV inside training); test is scored once at the end (Iberia).
- **Overfitting:** great on training data, poor on new data. Ours: 0.81 → 0.73, not severe.
- **Macro F1:** average F1 over classes, each class weighted equally.
- **Recall:** of all real Solar cells, how many we found.
- **Precision:** of all cells we called Hybrid, how many really were.
- **Random Forest:** many decision trees on random subsets; they vote.
- **SHAP:** how much each feature pushes a single prediction up or down; averaged = importance.
- **Area of applicability:** the region of feature space the model has seen; outside it, predictions are unreliable.

---

## 6. Problems solved (good viva material)

| Problem | How it was found | Fix |
|---|---|---|
| Fake wind raster | Spot-checked values against an official file | Official Global Wind Atlas country files |
| Island cells with no grid lines | Max distance 451 km looked impossible | Treat > 20 km as missing; 20 cells dropped |
| Azores/Madeira, far north | Missing wind / sunshine values | Dropped (48 cells), reported |
| 3 GB of elevation tiles | Too big to download | Read only needed pixels online (cloud-optimised GeoTIFF) |
| "Is 90% too good?" | Compared train / CV / test | Iberia is easy; the wind rule also gets 88% there |
| "Is Iberia lucky?" | Held out every region | Found the north fails, explained why |

---

## 7. Likely viva questions

**Q: Why not just use accuracy?**
Solar and Wind are 97% of cells. A model that never predicts Hybrid still looks accurate. Macro F1 and per-class recall expose that.

**Q: Why Random Forest and not deep learning?**
Five tabular features and ~9,000 rows; tree ensembles are the standard and explainable. I tested gradient boosting, Extra Trees, k-NN, SVM and logistic regression: none beat RF (0.543 CV macro F1).

**Q: Is the model overfitting?**
No serious overfitting: 0.81 on training data, 0.73 on unseen blocks. An unlimited forest scores 1.00 on training data; ours is depth-limited.

**Q: Why is the test (90%) higher than validation (73%)?**
Iberia is an easier region: wind and solar farms are far apart in wind speed. Even the one-line wind rule gets 88% there versus 74% in training.

**Q: Then is your result just "windy = wind"?**
Mostly, and I say so. Wind speed alone gives 0.503 CV macro F1; all five features 0.543. RF beats the wind rule on Iberia by 1.8 points (statistically real, but small), and the wind rule beats RF in four regions.

**Q: Why does it fail in the UK?**
UK solar farms stand at 7.7 m/s, windier than typical European wind farms (7.4). The model learned an absolute threshold, so it calls British solar farms Wind.

**Q: Why does it fail in India?**
Indian wind farms are at 6.4 m/s, below Europe's "windy". But India is much sunnier than training, so the unfamiliar-place rule flags 92% of cells as "I don't know". The safety net works.

**Q: Why does Hybrid fail?**
Only 308 training examples, 213 of them in Germany. The model learned "Hybrid = windy, not sunny", which doesn't hold in sunny Iberia.

**Q: What about China?**
84%, but 81% of Chinese sites are wind farms, so always guessing Wind gets 81%; the wind rule 82%. None of China's 40 Hybrid cells were found, which confirms Hybrid isn't learned. The "I don't know" rule abstained on 76% but didn't pick the right cells there, so it's a rough safety net.

**Q: How would you improve it?**
Relative wind speed (each site compared with its surrounding region), more Hybrid examples from other continents, a calibrated unfamiliar-place threshold, label cross-check with Global Energy Monitor.

**Q: Why didn't you use land cover? It's an obvious factor.**
The 2020 imagery already shows the farm itself (panels, cleared land), so it would leak the answer.

**Q: Why no negative samples (land without farms)?**
"No farm" doesn't mean "unsuitable". Labelling it as such would teach the model false facts. The model answers which type fits, not whether to build.

**Q: Why 5 km cells?**
Small enough to keep local conditions, large enough to group a farm's turbines/panels. Hybrid depends on cell size; that is listed as a limitation.

**Q: How do you know your results are reproducible?**
Fixed seed 42; the entire pipeline re-ran from scratch with byte-identical outputs.

**Q: What does "I don't know" achieve on Iberia?**
40% abstained; accuracy on the remaining cells rose from 0.901 to 0.938. Low-confidence cells were only 47% right (rule works); unfamiliar-only cells were 97% right (rule too cautious there).

**Q: Can this be used to decide where to build in Odisha?**
No. It predicts the type for a site where something will be built; it never judges suitability. And India results show it must not be trusted there without retraining.

**Q: Can the model recommend what to build at a new location?**
Only under a stated assumption: *if* existing farms reflect well-informed siting decisions, the model recommends which type fits a location *if it is developed*, and says "I don't know" when out of its depth. The assumption is reasonable (developers study the resource; Sachit et al. 2022 uses the same idea) but only partly true: subsidies and land prices also matter (UK solar farms sit in windy places), and accuracy varies from 35% to 92% by region. It never says "build here", because it never saw unsuitable land. (Report section 7.2, slide 16.)

**Q: How does the model's confidence work?**
Each of the 300 trees ends in a leaf holding training cells (e.g. 70% Wind, 30% Solar). The forest averages those shares, giving three probabilities that add up to 1. The top one is the prediction; its value is the confidence. Below 0.5 means no class has a majority, so Rule A says "I don't know".

**Q: What is your biggest limitation?**
Reliance on absolute wind speed, which doesn't transfer to regions where wind farms aren't much windier than solar farms.

---

## 8. Limitations (say them before you're asked)

- OpenStreetMap coverage is uneven; data is a 2020 snapshot.
- Few Hybrid examples, mostly from Germany; Hybrid = co-location in 5 km.
- Absolute wind speed doesn't transfer (UK 35%, India 53%).
- Predicts type given a farm exists, not suitability.

## 9. The five questions your teacher is most likely to ask

**1. State the problem clearly.**
"Given a place where a solar or wind farm already exists, predict from five geographic features
(sunshine, wind speed, elevation, slope, distance to the grid) whether it is Solar, Wind or Hybrid;
test it on a region it has never seen; and let it say 'I don't know' when it is unsure or the
place is unlike anything it learned from."
Input = 5 numbers per 5 km cell. Output = Solar / Wind / Hybrid / I don't know.

**2. Why this dataset (Dunnett et al. 2020)?**
- Peer-reviewed (Scientific Data, Nature group), global, free, reproducible.
- Real, observed farms (24,000 after filtering), so every label is a fact, not an opinion.
- Capacity cross-checked against national statistics; the authors give a recommended "farm" filter.
- Covers solar and wind in one harmonised format, so both classes come from the same source.
- Alternatives: national registers (one country only), Global Energy Monitor (needs sign-up, used
  for cross-checks), or expert suitability maps (MCDA), which are opinions, not observations.

**3. What is the advantage of using ML here?**
- A rule like "wind ≥ 6.6 m/s" uses one feature; ML combines all five and finds non-linear
  interactions (e.g. "windy *and* not sunny" for Hybrid).
- It learns from thousands of real decisions instead of hand-set weights (MCDA/AHP).
- It gives a probability, so it can say "I don't know".
- It is explainable (SHAP shows which feature drove each prediction).
- Honest caveat: ML beat the simple rule by only 1.8 points on Iberia, and lost to it in four
  regions. Proving that with baselines is itself part of doing ML properly.

**4. What did you learn?**
- Random splits cheat on map data; held-out regions give the honest score.
- Always sanity-check inputs (the first wind file was wrong).
- One test region is not enough: Iberia said 90%, all regions together say 71%.
- A model can be "accurate" for the wrong reason (absolute wind speed); explaining *why* it fails
  (UK, India) matters more than a high number.
- Leakage hides in obvious features (land cover, roads, capacity).
- "I don't know" is useful exactly where the model is out of its depth (India 92%).

**5. Is this the method used in research papers?**
Yes, each part is standard:
- Random Forest + SHAP on real wind/solar plants: Sachit et al. 2022 (ISPRS IJGI).
- Spatial / block cross-validation: Roberts et al. 2017 (Ecography).
- "I don't know" = area of applicability: Meyer & Pebesma 2021 (Methods in Ecology and Evolution).
- Dunnett et al. 2020 dataset is used in global siting and land-use studies.
- Baselines + confidence intervals + held-out region are standard good practice in applied ML.

---

## 10. Our project vs the friend's (Devin AI) report

| | Friend's project | Our project |
|---|---|---|
| Question | Where in Odisha *resembles* places with farms? (yes/no per technology) | Given a farm exists, *which type* fits? (Solar / Wind / Hybrid) |
| Negatives | Random "background" land points | None: only real farms |
| Models | Two separate yes/no models | One 3-class model + 3 baselines |
| Features | 11, **including land cover** | 5; land cover, roads, capacity excluded |
| Test | USA + China held out | Iberia, every European region, USA, China, India |
| Metrics | AUC, PR-AUC, Boyce | Macro F1, per-class recall, confusion matrix, 95% CIs |
| Extra | Odisha map, MCDA comparison | "I don't know", overfitting check, failure analysis |

**Is ours going right? Yes.** It answers a narrower question but with fewer hidden assumptions:
- Their top solar feature is **land cover**. WorldCover 2021 imagery shows the solar farms
  themselves, so the model can partly "see" the answer: the leakage we deliberately avoided.
- Random background points as "non-farm" can make AUC look high just because farms differ from
  random land (deserts, forests, mountains), not because the model understands siting.
- Ours never claims a place is suitable, so it can't be wrong in that way.
Their strength is a direct, useful Odisha application. Don't copy their design; if asked, say
"we chose a classification design so every label is an observed fact and no feature can contain
the farm itself."

---

## 11. Tried and rejected: relative features (`src/03e_relative_features.py`)

To help the model learn more than raw wind speed, we tried **relative wind and sunshine**: each
site's value minus the median of all farm sites in the same country.

| | Current 5 features | + relative wind & sun | relative only |
|---|---|---|---|
| Europe, 8 regions pooled | 0.712 | 0.733 | 0.625 |
| British Isles | 0.350 | 0.384 | 0.611 |
| Italy | 0.920 | 0.890 | 0.651 |
| USA | 0.896 | 0.879 | 0.807 |
| India | 0.533 | 0.546 | 0.551 |

Small gains in some regions, losses in others, macro F1 slightly down, India unchanged, so the
main model keeps the 5 features. Viva line: "We tried region-relative features; they helped the
UK a little but hurt the south, so the real fix needs better local context (e.g. wind compared
with the surrounding 50 km of land), which is future work."

---

## 12. Team of six: who did what

| Person | Role | Owns | Should be able to explain |
|---|---|---|---|
| 1 | **Data & labels** | `01_prepare_labels.py`, Dunnett filtering, 5 km grid, train/test split | Why Dunnett, the ≥ 1 MW filter, Hybrid definition, why Iberia |
| 2 | **Features & GIS** | `02_extract_features.py`, rasters, DEM, Gridfinder; caught the fake wind file and island cells | The 5 features, leakage, data-quality fixes |
| 3 | **Modelling & validation** | `03_*`, `04_explain.py`, `05_dont_know.py`: models, spatial CV, region and continent tests, SHAP, "I don't know" | Baselines, macro F1, overfitting, why UK/India fail |
| 4 | **Web UI** | `website/`, `website2/`: globe map, results and about views | How the map is drawn, what each view shows |
| 5 | **GitHub & deployment** | Repository, README, `requirements.txt`, Vercel/GitHub Pages deploy, reproducibility re-run | How to run the pipeline, how the site is hosted |
| 6 | **Report & presentation** | `REPORT_NOTES` (PDF/Word), the 19-slide PPT, viva guide | The story, numbers table, limitations, references |

Everyone should know section 1 (the story) and section 2 (the numbers).

---

## 13. References and why we used each

| Reference | Why we used it |
|---|---|
| Dunnett et al. 2020, *Scientific Data* | The dataset: all farm locations, plus the "real farm" filter |
| Breiman 2001, *Machine Learning* | Random Forest, the main model |
| Sachit et al. 2022, *ISPRS IJGI* | Precedent: RF + SHAP on real plants; feature choice; existing plants as evidence of good sites |
| Roberts et al. 2017, *Ecography* | Why random splits cheat on maps; spatial block CV |
| Meyer & Pebesma 2021, *Methods Ecol. Evol.* | "I don't know" Rule B (area of applicability) |
| Lundberg & Lee 2017 (NeurIPS); Lundberg et al. 2020 (*Nat. Mach. Intell.*) | SHAP and TreeExplainer |
| Arderne et al. 2020, *Scientific Data* | Gridfinder power lines (distance to grid) |
| Horn 1981, *Proc. IEEE* | Slope formula |
| Efron & Tibshirani 1993; Pedregosa et al. 2011 | Bootstrap confidence intervals; scikit-learn |
| Global Solar Atlas, Global Wind Atlas (Davis et al. 2023), Copernicus DEM, GADM | Sunshine, wind speed, elevation/slope, website borders |

If asked about "Wu et al. 2026" (it was in the original brief): we did not use it and could not verify it, so it is not cited.
