# Which Renewable Fits Here?

**Predicting whether a renewable energy site is Solar, Wind or Hybrid from geography alone,
tested on regions the model has never seen, with an option to answer "I don't know".**

Minor project · Machine learning · Geospatial · 2026

🌍 **Live demo:** https://AshutoshSinghJ.github.io/renewable-siting-ml/ (globe) ·
[detailed page](https://AshutoshSinghJ.github.io/renewable-siting-ml/detailed/)

---

## Problem statement

Given a place where an onshore, utility-scale renewable farm already exists, can a model trained on
geographic factors (sunshine, wind speed, elevation, slope, distance to the grid) predict whether it
is **Solar, Wind or Hybrid**, including in a region it has never seen, and say **"I don't know"**
when it is unsure?

The model answers *"if something is built here, which type fits?"*. It never judges whether a place
*should* be developed.

## Results at a glance

| Tested on (never seen in training) | Random Forest accuracy | One-line wind rule |
|---|---|---|
| Spain + Portugal (planned test) | **90%** | 88% |
| Italy · Balkans & Greece | 92% · 91% | 88% · 90% |
| All 8 European regions held out in turn | **71%** | |
| British Isles · Nordic & Baltic | 35% · 40% | 35% · 71% |
| USA · China · India (Europe-trained model, unchanged) | **90% · 84% · 53%** | 93% · 82% · 54% |

- **Works** where wind farms stand in much windier places than solar farms; **fails** where they don't (UK solar farms are windier than most European wind farms).
- **Hybrid** (solar + wind in one 5 km cell) is not learned: 1 of 25 on Iberia, 0 of 40 in China.
- **"I don't know"**: 40% of Iberian cells, and 92% of Indian cells, where the model would otherwise fail.
- No serious overfitting: 0.81 on training data, 0.73 on unseen blocks.

Full details: [report (PDF)](outputs/REPORT_NOTES.pdf) · [slides](outputs/Project_Presentation.pptx) ·
[project journey](docs/JOURNEY.md) · [mind map](docs/MINDMAP.md) · [viva guide](docs/VIVA_GUIDE.md)

## How it works

```
Dunnett et al. 2020          5 km grid cells            5 features per cell
(68,786 OSM records)  ──▶   Solar / Wind / Hybrid  ──▶  sunshine, wind 100 m,   ──▶  Random Forest  ──▶  Solar / Wind / Hybrid
filter: farms ≥ 1 MW         (10,724 in Europe)          elevation, slope, grid        + 3 baselines        or "I don't know"
```

1. **Labels** (`src/01`): keep real farms (≥ 1 MW, not urban or water), snap to a 5 km equal-area grid, label each cell, and split: train on Europe, test on Spain + Portugal.
2. **Features** (`src/02`): sample the Global Solar Atlas, Global Wind Atlas, Copernicus DEM and Gridfinder at every farm, then average per cell. Land cover, roads and capacity are left out on purpose (the farm itself changes them).
3. **Models** (`src/03`): majority class, wind-speed rule, logistic regression, Random Forest. Tuned with spatial cross-validation (100 km blocks); the test region is used once.
4. **Extra checks** (`src/03b`–`03e`): five other classifiers, feature subsets, overfitting, every region held out, other continents, relative features.
5. **Explain** (`src/04`): SHAP shows what drives each class (wind speed first, then sunshine).
6. **I don't know** (`src/05`): abstain on low confidence or on unfamiliar places.

## Repository structure

```
renewable-siting-ml/
├── src/                   pipeline, one script per step (01 → 05)
├── data/
│   ├── processed/         labelled cells and features (ready to use)
│   └── raw/               large downloads, not in git (see data/raw/README.md)
├── outputs/               results (CSV), figures (PNG), trained model, report, slides
├── web/
│   ├── globe/             map-first website (Map · Results · About)
│   └── detailed/          single-page website with all details
├── docs/
│   ├── JOURNEY.md         what we did, what happened, what we learnt
│   ├── VIVA_GUIDE.md      viva preparation and likely questions
│   ├── MINDMAP.md         one-page mind map of the project
│   ├── DATA_AND_METHOD.md every dataset, map step and model step, with justification
│   └── PROJECT_BRIEF.md   original brief and design decisions
└── tools/                 helpers: report → Word, split map figure
```

## Run it

```bash
pip install -r requirements.txt

# Steps 3–5 run from data/processed/ (no downloads needed)
python src/03_train_evaluate.py
python src/03b_extra_checks.py
python src/03c_region_holdout.py
python src/04_explain.py
python src/05_dont_know.py

# Rebuild the websites
python web/globe/build_site.py
python web/detailed/build_site.py
```

Steps 1, 2, 3d and 3e need the raw data listed in [data/raw/README.md](data/raw/README.md).
All runs use `random_state=42`; a full re-run reproduces every output exactly.

## Team

| Member | Role |
|---|---|
| 1 | Data & labels: Dunnett filtering, 5 km grid, region split |
| 2 | Features & GIS: rasters, DEM, grid distance, data-quality fixes |
| 3 | Modelling & validation: models, spatial CV, region/continent tests, SHAP, "I don't know" |
| 4 | Web UI: globe and detailed websites |
| 5 | GitHub & deployment: repository, reproducibility, GitHub Pages |
| 6 | Report & presentation: report, slides, viva guide |

## References

- Dunnett, S. et al. (2020). Harmonised global datasets of wind and solar farm locations and power. *Scientific Data* 7, 130.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with spatial structure. *Ecography* 40, 913–929.
- Meyer, H. & Pebesma, E. (2021). Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution* 12, 1620–1633.
- Sachit, M. S. et al. (2022). *ISPRS International Journal of Geo-Information* 11(8), 422.
- Arderne, C. et al. (2020). Predictive mapping of the global power system using open data. *Scientific Data* 7, 19.

Data: Global Solar Atlas (Solargis / World Bank), Global Wind Atlas (DTU / World Bank), Copernicus DEM GLO-90, Gridfinder, GADM borders.
