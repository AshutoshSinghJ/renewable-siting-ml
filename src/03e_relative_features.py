"""
Step 3e (experiment) - do relative features help where absolute wind speed fails?

rel_x = x minus the median x of all renewable cells in the same country (no labels used).
Compares the current 5 features with (B) 5 + relative wind and sun, and (C) relative instead of
absolute, on the 8-region hold-out and on the USA and India. Result: no consistent gain, so the
main pipeline keeps the 5 features.

Run from the project root:  python src/03e_relative_features.py
"""
# rel_x = x minus the median x of all renewable cells in the same country (uses no labels).

import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

BASE = ["ghi", "wind100", "elev", "slope", "dist_grid"]
C3 = ["Hybrid", "Solar", "Wind"]
cells = pd.read_csv("data/processed/cells.csv")[["cell_id", "country"]]
eu = pd.read_csv("data/processed/features.csv").merge(cells, on="cell_id")
oth = pd.read_csv("data/processed/features_other.csv")
eu["region"] = None
REG = {"Iberia": "ESP PRT", "France & Benelux": "FRA BEL NLD LUX CHE", "Germany & Austria": "DEU AUT",
       "British Isles": "GBR IRL", "Italy": "ITA", "Nordic & Baltic": "DNK NOR SWE FIN EST LVA LTU",
       "Central & Eastern": "POL CZE SVK HUN ROU UKR BLR SVN", "Balkans & Greece": "GRC CYP BGR HRV BIH SRB MNE ALB XKO"}
for r, cs in REG.items(): eu.loc[eu.country.isin(cs.split()), "region"] = r
allc = pd.concat([eu, oth], ignore_index=True)
for f in ["wind100", "ghi"]:
    allc["rel_" + f] = allc[f] - allc.groupby("country")[f].transform("median")
eu, oth = allc[allc.region.notna()], allc[allc.region.isna()]

rf0 = joblib.load("outputs/rf_model.joblib")
SETS = {"A. current 5 features": BASE,
        "B. + relative wind & sun": BASE + ["rel_wind100", "rel_ghi"],
        "C. relative wind & sun instead of absolute": ["rel_ghi", "rel_wind100", "elev", "slope", "dist_grid"]}
rows = []
for name, F in SETS.items():
    row = {"features": name}
    pooled_y, pooled_p = [], []
    for r in REG:
        tr, te = eu[eu.region != r], eu[eu.region == r]
        p = clone(rf0).fit(tr[F], tr.label).predict(te[F])
        row[r] = round(accuracy_score(te.label, p), 3); pooled_y += list(te.label); pooled_p += list(p)
    row["Europe pooled"] = round(accuracy_score(pooled_y, pooled_p), 3)
    row["Europe pooled macroF1"] = round(f1_score(pooled_y, pooled_p, labels=C3, average="macro"), 3)
    tr = eu[eu.region != "Iberia"]
    m = clone(rf0).fit(tr[F], tr.label)
    for c, n in [("USA", "USA"), ("IND", "India")]:
        te = oth[oth.country == c]; row[n] = round(accuracy_score(te.label, m.predict(te[F])), 3)
    rows.append(row)
out = pd.DataFrame(rows).set_index("features").T
out.to_csv("outputs/relative_features_experiment.csv")
print(out.to_string())
