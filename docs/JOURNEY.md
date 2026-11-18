# Project Journey: what we did, what happened, what we learnt

The project in the order it actually happened. Each step: **what we did → what happened → what we learnt.**

---

### 1. Defining the question

- **Did:** Started from a simple idea: solar and wind farms are built in different kinds of places. Asked: from geography alone, can a model tell whether a site is Solar, Wind or Hybrid?
- **Happened:** Had to decide what "negative" data means. Land without a farm isn't "unsuitable"; it might be protected, expensive or simply never proposed.
- **Learnt:** Train only on places where something *was* built. The model answers "which type fits here?", never "should anything be built here?". Every label is then a fact, not a guess.

### 2. Getting the labels

- **Did:** Used Dunnett et al. (2020), a peer-reviewed global dataset of 68,786 solar and wind records from OpenStreetMap.
- **Happened:** Many records were single rooftop panels or lone turbines, and two-thirds had no capacity value.
- **Learnt:** Read the dataset's own documentation. The authors only estimate capacity for real farms (wind > 4 turbines, solar > 1 ha), so filtering on "≥ 1 MW, not urban, not water" left 24,078 genuine farms. Their recommendation, not our guess.

### 3. Turning points into cells

- **Did:** Snapped farms to a 5 km equal-area grid; each cell is Solar only, Wind only, or Hybrid (both).
- **Happened:** 18,705 cells worldwide, but 339 of the 396 Hybrid cells were in Europe.
- **Learnt:** The data decides the study area. We used Europe, and Hybrid stayed a class because it passed our 300-cell rule.

### 4. Choosing how to test

- **Did:** Held out a whole region (Spain + Portugal, "Iberia") instead of a random split; tuned only with spatial cross-validation on 100 km blocks.
- **Happened:** Neighbouring cells are almost identical, so a random split would let the model "see" the test set.
- **Learnt:** On map data, random splits cheat. Hold out whole regions, and touch the test region only once (Roberts et al. 2017).

### 5. Choosing features, and what to leave out

- **Did:** Five features: sunshine, wind speed at 100 m, elevation, slope, distance to the grid.
- **Happened:** Land cover, roads and capacity looked useful, but each is *changed by the farm itself*.
- **Learnt:** That's data leakage. A 2020 satellite image already shows the solar panels. Leaving these out makes the score lower but honest.

### 6. The fake wind data

- **Did:** Extracted wind speed from a global wind raster that was already on disk.
- **Happened:** Wind speed was almost identical for every class. Spot checks showed Bavaria windier than the North Sea coast, and the Gujarat coast at 4.7 m/s against the official 6.6 m/s. The file wasn't real Global Wind Atlas data.
- **Learnt:** Always sanity-check inputs against known values. We replaced it with official per-country files, and wind speed immediately separated the classes.

### 7. Building and comparing models

- **Did:** Trained a Random Forest and three baselines: always-the-majority, a one-line wind rule ("Wind if ≥ 6.6 m/s") and logistic regression.
- **Happened:** On Iberia the Random Forest got 90% accuracy, but the one-line wind rule got 88%.
- **Learnt:** Baselines are essential. Without the wind rule, 90% looks brilliant; with it, we know most of the skill is "windy = wind". Bootstrap confidence intervals showed the 1.8-point gain is real but small.

### 8. Explaining the model

- **Did:** Used SHAP to see which features drive each class.
- **Happened:** Wind speed dominated. Hybrid was learned as "windy but not sunny", because 213 of the 308 training Hybrids are in Germany.
- **Learnt:** That explained why Hybrid failed in sunny Iberia (1 of 25 correct). Explanation tools show *why* a model fails, not just that it does.

### 9. Saying "I don't know"

- **Did:** Added two abstain rules: low confidence (top probability < 0.5) and unfamiliar place (far from all training cells).
- **Happened:** 40% of Iberia abstained. Low-confidence cells were only 47% right, so that rule works. "Unfamiliar" cells were 97% right, so that rule is too cautious there.
- **Learnt:** A safety net needs testing too; one rule earned its place, the other was over-careful.

### 10. Checking everything again

- **Did:** Re-ran the whole pipeline from scratch and audited splits, features and numbers.
- **Happened:** Outputs were byte-identical. But one cell was "451 km from the grid": the Canary Islands have no lines in Gridfinder.
- **Learnt:** Look at extremes, not just averages. We dropped 20 island cells without grid data.

### 11. "Is 90% too good? Is it overfitting?"

- **Did:** Compared accuracy on training data, unseen blocks and Iberia: 0.81, 0.73, 0.90.
- **Happened:** No serious overfitting, but the test scored *higher* than validation.
- **Learnt:** Iberia is an easy region: there, wind farms are on windy ridges and solar farms on calm plains. Even the wind rule jumps from 74% to 88%.

### 12. "Was Iberia lucky?"

- **Did:** Held out each of 8 European regions in turn.
- **Happened:** Southern regions scored 90–92%, but the British Isles 35% and Nordic & Baltic 40%; 71% overall.
- **Learnt:** One test region is not enough. The reason: UK solar farms sit at 7.7 m/s, windier than most European wind farms. The model learned *absolute* wind speed, which doesn't travel.

### 13. Other continents

- **Did:** Applied the Europe-trained model, unchanged, to the USA, China and India.
- **Happened:** USA 90%, China 84% (always guessing Wind gets 81%), India 53%. But in India the model said "I don't know" for 92% of cells.
- **Learnt:** The model works wherever wind farms are much windier than solar farms. And the unfamiliar-place rule, over-cautious in Iberia, did its job in India.

### 14. Trying to fix it

- **Did:** Added relative wind and sunshine (each site compared with its country's median).
- **Happened:** Small gains in the north, losses in the south; overall 71% → 73%, macro F1 slightly down.
- **Learnt:** Not every idea works; reporting that is part of the result. A real fix needs local context, e.g. wind compared with the surrounding 50 km of land.

### 15. Comparing with another approach

- **Did:** Compared with a project that predicts "farm vs random land" using land cover.
- **Happened:** Its top solar feature was land cover, which shows the farms themselves.
- **Learnt:** High scores can come from leakage. Our design gives lower but trustworthy numbers.

---

## The project in one paragraph

Geography, mostly wind speed, separates solar from wind farms well wherever wind farms stand in
much windier places than solar farms (southern Europe 90–92%, USA 90%, China 84%), and fails
where they don't (British Isles 35%, India 53%). Hybrid sites aren't learned with this data. The
model can say "I don't know", which helped exactly where it failed most (India). The biggest
lessons: test on whole unseen regions, compare against simple baselines, check inputs, and
explain failures instead of hiding them.
