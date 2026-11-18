# Contributions: who did what (6 members)

**Project:** Which Renewable Fits Here? Predicting Solar, Wind and Hybrid installations from geography

**Guide:** Dr. N Sangita Acharya, School of Computer Science and Engineering, KIIT

Every part of the project belongs to exactly one person, so each member can explain their own part in
full and pass other questions to the right person. Everyone must still know the story and the key
numbers (see [TEAM_GUIDE.md](TEAM_GUIDE.md), Part 0).

| # | Member | Role | Owns (scripts / files) | Presents |
|---|---|---|---|---|
| 1 | **Ashutosh Singh** | Project lead: idea, design and **dataset** | `src/01`, `src/02`, `src/06`; `data/`; `docs/DATA_AND_METHOD.md`, `docs/PROJECT_BRIEF.md` | Slides 1–5 |
| 2 | **Astitva Tripathi** | **Model selection and training** | `src/03`, `src/03b`–`03e`, `src/07`, `src/10` | Slides 6–13, 16–18 |
| 3 | **Divyanshu Puri** | **Explainability, "I don't know", final test, GitHub** | `src/04`, `src/05`, `src/08`, `src/09`, `src/11`; repository, GitHub Pages | Slides 14–15, 19–20, 22 + GitHub demo |
| 4 | **Madhusudan** | **UI / UX and websites** | `web/globe/`, `web/detailed/`, `build_site.py`, screenshots | Slide 21 + live website demo |
| 5 | **Aryan Singh** | **Project report and references** | `outputs/REPORT_NOTES.*`, `outputs/Project_Report_KIIT.*`, `docs/COMPLETE_GUIDE.md` | Slides 23 and 25 |
| 6 | **Keshav Sudarshan** | **Presentation and viva preparation** | `outputs/Project_Presentation.pptx`, `docs/VIVA_GUIDE.md`, `docs/JOURNEY.md`, `docs/MINDMAP.md`, `docs/TEAM_GUIDE.md` | Slide 24 + opening and flow |

---

## 1. Ashutosh Singh: project lead, design and dataset

**Work done**
- Started the project: learn from **real farms** instead of expert-weighted suitability maps (MCDA/AHP);
  never treat empty land as "unsuitable".
- Chose the **Dunnett et al. (2020)** dataset and applied the authors' real-farm filter
  (≥ 1 MW, not urban, not on water): 68,786 records → **24,078 farms**.
- Designed the **5 km equal-area grid** (World Eckert IV) and the Solar / Wind / Hybrid labels; ran the
  cell-size check (33 / 107 / 409 / 830 Hybrid cells at 1 / 2 / 5 / 10 km) → **18,705 cells**.
- Chose **Europe for training** (339 of 396 Hybrids) and **Spain + Portugal for testing**; ruled out random splits.
- Built the five features (`src/02`): Global Solar Atlas, Global Wind Atlas (100 m), Copernicus DEM
  (elevation, Horn slope, read online), Gridfinder (distance to grid).
- Found and fixed the data problems: the **fake wind raster** (Gujarat 4.7 vs 6.6 m/s) replaced with
  official country files; **island cells without grid lines** removed (68 cells dropped in total → 10,724).
- Excluded land cover, roads and capacity (**leakage**).
- Improvement round data (`src/06`): the **8 local-context features** (wind/sun percentile and
  difference within 30 km, height vs surroundings, ruggedness, distance to town and water from GHS-SMOD)
  and the base features for the five **locked countries**.

**Must know:** why this dataset, why Europe/Iberia, the filters, the grid, every feature and its source,
the coordinate systems, leakage, the two data problems, what the new features show (wind farms at the
85th percentile of local wind, 16.5 m above surroundings, 12 km from towns).

---

## 2. Astitva Tripathi: model selection and training

**Work done**
- Baselines and main model (`src/03`): majority class, **wind-speed rule** (6.60 m/s), logistic
  regression, **Random Forest** (300 trees, balanced classes).
- Tuning with **spatial cross-validation** (GroupKFold over 100 km blocks); one-time Iberia test:
  **RF 0.901** vs wind rule 0.882 (+1.8 points, CI +0.5 to +3.2).
- Extra checks (`src/03b`–`03e`): six classifiers (RF best, 0.543), feature subsets, overfitting
  (0.81 / 0.73 / 0.90), **every European region held out** (71%; British Isles 35%), **USA / China /
  India** (90 / 84 / 53%), relative features (rejected).
- Improvement round (`src/07`, `src/10`): **leave-one-region-out over 11 regions** with the pre-set score,
  five feature sets, seven models → **Extra Trees selected (0.794; worst region 0.536)**; neural network
  (MLP, 0.763) and **patch neural network** on raw maps (0.738); covariate-shift reweighting (no gain).

**Must know:** how RF and Extra Trees work, why spatial CV, metrics (accuracy, macro F1, recall, CIs),
why baselines matter, overfitting, why the north fails, why Extra Trees, why not deep learning.

---

## 3. Divyanshu Puri: explainability, "I don't know", final test and GitHub

**Work done**
- **SHAP** explanations (`src/04`): wind speed 0.166, sunshine 0.070; Hybrid = "windy, not sunny" (Germany).
- First **"I don't know"** (`src/05`): Rule A (confidence < 0.5) and Rule B (unfamiliar place);
  Iberia 40% abstained (accuracy 0.901 → 0.938), India 92%.
- **Conformal prediction** (`src/08`): answer sets with 90% target coverage; overall variant chosen
  (coverage 0.895, 73% single answers); **Rule B dropped** (flagged cells 0.93 vs 0.79 accurate).
- **Archetypes and SHAP by region** (`src/09`): four types of place; wind's share fell from ~51% to 23%.
- **Locked final test** (`src/11`): five countries scored once: final 0.892, wind rule 0.905, old RF
  0.896 (a tie); conformal coverage **0.943**; overfitting check (0.995 / 0.794 / 0.892).
- **GitHub**: repository structure, `.gitignore` (25 GB raw data and 183 MB model kept out), README,
  `requirements.txt`, reproducibility re-run, `gh-pages` branch, tags `v1.0-final`, `v1.0.1`.

**Must know:** how SHAP works, the two old rules and why Rule B was dropped, how conformal sets work,
why the locked test is a tie, how the site is hosted, how to reproduce the results.

---

## 4. Madhusudan: UI / UX and websites

**Work done**
- Designed both websites: the main **globe site** (Map · Results · About) and the detailed single page.
- Visual system: dark "control room" theme; one colour per class everywhere (Solar amber, Wind teal,
  Hybrid purple, wrong red, "I don't know" grey); fonts; floating number panels; plain-language labels.
- Map views: What was built, Iberia test, Every region tested, Other continents, **Improved model**,
  **Archetypes**, **Final test**, Wind speed; hover tooltips with prediction and conformal answer.
- Technical: D3.js 7.9.0 on an HTML canvas (~17,000 dots), orthographic globe, fly animations,
  `build_site.py` packing results as JSON into one self-contained page.
- Phone layout, keyboard focus, reduced-motion support; website screenshots for the report.

**Must know:** why two sites, why these colours, how the globe draws so many points, what each view
shows, how data gets into the page, that the numbers come straight from the result files.

---

## 5. Aryan Singh: project report and references

**Work done**
- Wrote and structured the **project report** (`REPORT_NOTES`, 20 pages) and the **KIIT-format report**
  (43 pages): abstract, literature review, requirements, conclusion, future scope.
- Compiled the **references** and why each is used (dataset, models, spatial CV, area of applicability,
  conformal prediction, SHAP, data sources).
- Wrote the **limitations** and the **assumption** for using the model on new locations
  ("which type fits *if* built", never "build here").
- Checked every number in the report against the result files.

**Must know:** the main finding in one sentence, the limitations, the assumption, why each reference.

---

## 6. Keshav Sudarshan: presentation and viva preparation

**Work done**
- Built the **25-slide presentation** with one message per slide, native charts and speaker notes:
  original study (1–15), improvement round (16–20), use, checks, limitations, findings, references.
- Planned the **speaking order** (table above) and timing (~1 minute per slide).
- Prepared the **viva guide** (likely questions and answers), the **project journey** (20 steps) and the
  **mind map**.

**Must know:** the flow of the whole story, the key message of every slide, the numbers to remember,
the five questions the teacher is most likely to ask.

---

## Viva: who answers what

| If asked about… | Answered by |
|---|---|
| Why this problem, dataset, filters, grid, features, leakage, data errors, local-context features | Ashutosh Singh |
| Models, baselines, spatial CV, metrics, regions, continents, Extra Trees, neural networks, overfitting | Astitva Tripathi |
| SHAP, "I don't know", conformal prediction, archetypes, locked final test, GitHub, reproducibility | Divyanshu Puri |
| Website design, map, views, interactions, colours | Madhusudan |
| Report, references, limitations, assumption, conclusion | Aryan Singh |
| Overall story, slide messages, summary, what we learnt | Keshav Sudarshan |
