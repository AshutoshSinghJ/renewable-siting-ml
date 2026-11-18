# Contributions: who did what (6 members)

**Project:** Which Renewable Fits Here? Predicting Solar, Wind and Hybrid installations from geography

**Guide:** Dr. N Sangita Acharya, School of Computer Science and Engineering, KIIT

**How the work is divided.** Ashutosh Singh started the project and owns its design and the whole
dataset. The rest of the work is split evenly: each of the other five members owns **one technical
part of the pipeline** (with its scripts and results) **and one deliverable** (website, slides,
report or repository). Every script and file belongs to exactly one person, so each member can
explain their own part in full in the viva.

| # | Member | Technical part (scripts) | Deliverable | Presents |
|---|---|---|---|---|
| 1 | **Ashutosh Singh** | Project start and design; **dataset**: labels, features, local-context features (`src/01`, `02`, `06`) | `docs/PROJECT_BRIEF.md`, `docs/DATA_AND_METHOD.md` | Slides 1–5, 16 |
| 2 | **Astitva Tripathi** | **Models and training**: baselines, Random Forest, tuning, model comparison, Extra Trees, neural networks (`src/03`, `03b`, `07`, `10`) | Model results tables and charts | Slides 6–10, 18 |
| 3 | **Divyanshu Puri** | **Validation and testing**: overfitting, every region held out, other continents, relative features, locked final test (`src/03c`, `03d`, `03e`, `11`) | **GitHub repository and GitHub Pages** | Slides 11–13, 17, 20, 22 |
| 4 | **Keshav Sudarshan** | **Explainability and patterns**: SHAP, SHAP by region, siting archetypes (`src/04`, `09`) | **Presentation** (25 slides, speaker notes), viva guide | Slides 14, 24 + opening and closing |
| 5 | **Aryan Singh** | **Uncertainty ("I don't know")**: confidence and unfamiliar-place rules, conformal prediction (`src/05`, `08`) | **Project report** (both formats) and references | Slides 15, 19, 23, 25 |
| 6 | **Madhusudan** | **Visualisation**: website data packing and map views (`web/*/build_site.py`, `template.html`) | **Websites** (globe and detailed), screenshots | Slide 21 + live website demo |

---

## 1. Ashutosh Singh: project start, design and dataset

**Work done**
- **Started the project:** learn from **real farms** instead of expert-weighted suitability maps
  (MCDA/AHP); never treat empty land as "unsuitable"; the model answers "which type fits *if* built".
- **Design decisions:** three classes (Solar / Wind / Hybrid); 5 km equal-area grid (World Eckert IV);
  train on Europe (339 of 396 Hybrids), test on Spain + Portugal; no random splits; no leaky features
  (land cover, roads, capacity); fixed seed; one script per step.
- **Labels (`src/01`):** chose the Dunnett et al. (2020) dataset; applied the real-farm filter
  (≥ 1 MW, not urban, not on water): 68,786 records → **24,078 farms** → **18,705 cells**; cell-size
  check for Hybrid (33 / 107 / 409 / 830 at 1 / 2 / 5 / 10 km).
- **Features (`src/02`):** Global Solar Atlas, Global Wind Atlas (100 m), Copernicus DEM (elevation,
  Horn slope, read online), Gridfinder (distance to grid).
- **Data quality:** found and replaced the **fake wind raster** (Gujarat 4.7 vs 6.6 m/s); removed
  **island cells without grid lines**; 68 cells dropped in total → 10,724 European cells.
- **Improvement-round data (`src/06`):** the **8 local-context features** (wind / sunshine percentile and
  difference within 30 km, height vs surroundings, ruggedness, distance to town and to water from
  GHS-SMOD) and the base features for the five **locked countries**.

**Must know:** why this problem and dataset, every design decision, the filters, the grid, each feature
and its source, coordinate systems, leakage, the two data problems, what the new features show
(wind farms at the 85th percentile of local wind, 16.5 m above their surroundings, 12 km from towns).

---

## 2. Astitva Tripathi: models and training

**Work done**
- **Baselines and main model (`src/03`):** majority class, **wind-speed rule** (threshold 6.60 m/s
  chosen by macro F1), logistic regression, **Random Forest** (300 trees, balanced classes).
- **Tuning** with spatial cross-validation (GroupKFold over 100 km blocks); one-time **Iberia test:
  RF 0.901** vs wind rule 0.882 (+1.8 points, CI +0.5 to +3.2); bootstrap confidence intervals.
- **Model comparison (`src/03b`):** six classifiers (RF best, CV macro F1 0.543) and feature subsets
  (wind alone 0.503 → all five 0.543).
- **Improvement round (`src/07`, `src/10`):** five feature sets and seven models on 11 held-out regions
  → **Extra Trees selected** (0.794; worst region 0.536); MLP (0.763); **neural network on raw map
  patches** (0.738); covariate-shift reweighting (no gain).

**Must know:** how RF and Extra Trees work, why baselines, spatial CV, the metrics (accuracy, macro F1,
recall, CIs), the Iberia results and confusion matrix, why Extra Trees, why not deep learning.

---

## 3. Divyanshu Puri: validation, testing and GitHub

**Work done**
- **Overfitting check (`src/03c`):** train 0.81 · unseen blocks 0.73 · Iberia 0.90.
- **Every European region held out (`src/03c`):** 71% pooled; south 90–92%; **British Isles 35%**,
  Nordic 40%; explained why (UK solar farms are windier than most European wind farms).
- **Other continents (`src/03d`):** Europe model unchanged on USA / China / India: **90 / 84 / 53%**.
- **Relative features (`src/03e`):** tested and rejected (no consistent gain).
- **Before/after of the improvement round** on 11 regions: 0.736 → **0.794**; British Isles 0.354 → 0.536.
- **Locked final test (`src/11`):** five countries scored once: final 0.892, wind rule 0.905, old RF
  0.896 (a tie, explained); overfitting of the final model (0.995 / 0.794 / 0.892).
- **GitHub:** repository structure, `.gitignore` (25 GB raw data and 183 MB model kept out), README,
  `requirements.txt`, full reproducibility re-run, `gh-pages` branch, releases `v1.0-final`, `v1.0.1`.

**Must know:** why one test region is not enough, why the north and India fail, why the locked test is
a tie, how the site is hosted, how to reproduce the results.

---

## 4. Keshav Sudarshan: explainability, patterns and presentation

**Work done**
- **SHAP (`src/04`):** wind speed 0.166, sunshine 0.070, elevation 0.035, slope 0.030, grid 0.023;
  Hybrid learned as "windy, not sunny" (213 of 308 training Hybrids in Germany).
- **SHAP by region and archetypes (`src/09`):** wind's share of the final model fell from ~51% to **23%**;
  local wind percentile most important in the British Isles; **four siting archetypes** (k-means,
  k = 4); the hard one is the windy, cloudy lowland (64%).
- **Presentation:** the 25-slide deck with one message per slide, native charts and speaker notes;
  speaking order and timing; **viva guide**, project journey and mind map.

**Must know:** how SHAP works, what drives each class, why Hybrid fails, the four archetypes, the flow
of the whole presentation.

---

## 5. Aryan Singh: "I don't know", report and references

**Work done**
- **First "I don't know" (`src/05`):** Rule A (confidence < 0.5) and Rule B (unfamiliar place, area of
  applicability); Iberia 40% abstained (0.901 → 0.938 on the rest), India 92%.
- **Conformal prediction (`src/08`):** answer sets with a 90% target; overall variant chosen (coverage
  0.895, 73% single answers); **Rule B dropped** (flagged cells were more accurate, 0.93 vs 0.79);
  94.3% coverage on the locked countries.
- **Report:** the project report (`REPORT_NOTES`, 20 pages) and the KIIT-format report (43 pages);
  **references** and why each is used; **limitations**; the **assumption** for using the model on new
  locations; checking every number against the result files.

**Must know:** how confidence, Rule A, Rule B and conformal sets work, the main finding in one sentence,
the limitations and the assumption, why each reference.

---

## 6. Madhusudan: visualisation and websites

**Work done**
- Designed both websites: the main **globe site** (Map · Results · About) and the detailed single page.
- Visual system: dark theme; one colour per class everywhere (Solar amber, Wind teal, Hybrid purple,
  wrong red, "I don't know" grey); fonts; floating number panels; plain-language labels.
- Map views: What was built, Iberia test, Every region tested, Other continents, **Improved model**,
  **Archetypes**, **Final test**, Wind speed; hover tooltips with prediction and conformal answer.
- `build_site.py`: packs the result files as JSON into one self-contained page (D3.js 7.9.0 on an
  HTML canvas, ~17,000 dots, orthographic globe, fly animations).
- Phone layout, keyboard focus, reduced-motion support; website screenshots for the report.

**Must know:** why two sites, why these colours, how the globe draws so many points, what each view
shows, how the data gets into the page.

---

## Viva: who answers what

| If asked about… | Answered by |
|---|---|
| Why this problem, design decisions, dataset, filters, grid, features, leakage, data errors, local-context features | Ashutosh Singh |
| Baselines, Random Forest, tuning, metrics, Iberia result, model comparison, Extra Trees, neural networks | Astitva Tripathi |
| Overfitting, regions, other continents, before/after, locked final test, GitHub, reproducibility | Divyanshu Puri |
| SHAP, what drives the model, Hybrid failure, archetypes, presentation flow | Keshav Sudarshan |
| "I don't know", Rule A/B, conformal prediction, report, limitations, references | Aryan Singh |
| Websites, map, views, colours, interactions | Madhusudan |
