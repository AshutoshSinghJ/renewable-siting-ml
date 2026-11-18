"""
Step 3c (extra check) - does the result hold in other regions, not just Iberia?

Leave-one-region-out: Europe is split into 8 regions. For each region, the model is trained on
the other 7 and scored on the held-out one. Nothing is tuned here: the Random Forest uses exactly
the settings Step 3 selected, and the wind-speed rule's threshold is refit on each training part.
The Iberia round uses the same training set as the main experiment, so it must reproduce Step 3.

Run from the project root:  python src/03c_region_holdout.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score

FEATURES = ["ghi", "wind100", "elev", "slope", "dist_grid"]
CLASSES = ["Hybrid", "Solar", "Wind"]
REGIONS = {
    "Iberia": "ESP PRT",
    "France & Benelux": "FRA BEL NLD LUX CHE",
    "Germany & Austria": "DEU AUT",
    "British Isles": "GBR IRL",
    "Italy": "ITA",
    "Nordic & Baltic": "DNK NOR SWE FIN EST LVA LTU",
    "Central & Eastern": "POL CZE SVK HUN ROU UKR BLR SVN",
    "Balkans & Greece": "GRC CYP BGR HRV BIH SRB MNE ALB XKO",
}
COUNTRY_TO_REGION = {c: r for r, cs in REGIONS.items() for c in cs.split()}

df = pd.read_csv("data/processed/features.csv").merge(
    pd.read_csv("data/processed/cells.csv")[["cell_id", "country"]], on="cell_id")
df["region"] = df.country.map(COUNTRY_TO_REGION)
assert df.region.notna().all(), df[df.region.isna()].country.unique()
rf_selected = joblib.load("outputs/rf_model.joblib")


def wind_rule(train):
    """Same baseline as Step 3: Wind if wind100 >= t, else Solar; t chosen on the training part."""
    ws = train.wind100.to_numpy()
    cands = np.quantile(ws, np.linspace(0.02, 0.98, 97))
    scores = [f1_score(train.label, np.where(ws >= t, "Wind", "Solar"), labels=CLASSES, average="macro",
                       zero_division=0) for t in cands]
    return cands[int(np.argmax(scores))]


rows, preds = [], []
for region in REGIONS:
    tr, te = df[df.region != region], df[df.region == region]
    rf = clone(rf_selected).fit(tr[FEATURES], tr.label)
    p_rf = rf.predict(te[FEATURES])
    proba = rf.predict_proba(te[FEATURES])
    t = wind_rule(tr)
    p_rule = np.where(te.wind100 >= t, "Wind", "Solar")
    # macro F1 over the classes that actually occur in this region (some regions have no Hybrid)
    present = [c for c in CLASSES if (te.label == c).any()]
    rec = recall_score(te.label, p_rf, labels=CLASSES, average=None, zero_division=0)
    rows.append({
        "region": region, "n_cells": len(te),
        **{f"n_{c}": int((te.label == c).sum()) for c in CLASSES},
        "rf_accuracy": accuracy_score(te.label, p_rf),
        "rf_macro_f1": f1_score(te.label, p_rf, labels=present, average="macro", zero_division=0),
        "rule_accuracy": accuracy_score(te.label, p_rule),
        "rule_macro_f1": f1_score(te.label, p_rule, labels=present, average="macro", zero_division=0),
        **{f"rf_recall_{c}": (r if (te.label == c).any() else np.nan) for c, r in zip(CLASSES, rec)},
        "rule_threshold": t,
    })
    preds.append(pd.DataFrame({"cell_id": te.cell_id, "region": region, "label": te.label, "rf_pred": p_rf,
                               **{f"p_{c.lower()}": proba[:, i].round(3) for i, c in enumerate(rf.classes_)}}))

res = pd.DataFrame(rows)
res["rf_minus_rule_accuracy"] = res.rf_accuracy - res.rule_accuracy
res = res.round(3)
res.to_csv("outputs/region_holdout.csv", index=False)
pd.concat(preds).to_csv("outputs/region_holdout_predictions.csv", index=False)

pd.set_option("display.width", 200)
print(res[["region", "n_cells", "n_Hybrid", "rf_accuracy", "rf_macro_f1", "rule_accuracy", "rule_macro_f1",
           "rf_recall_Solar", "rf_recall_Wind", "rf_recall_Hybrid"]].to_string(index=False))
allp = pd.concat(preds)
print(f"\nAll regions pooled (every cell predicted by a model that never saw its region): "
      f"accuracy {accuracy_score(allp.label, allp.rf_pred):.3f}, "
      f"macro F1 {f1_score(allp.label, allp.rf_pred, labels=CLASSES, average='macro'):.3f}")
print("Wrote outputs/region_holdout.csv, outputs/region_holdout_predictions.csv")

# ---- Overfitting check for the main model: seen data vs unseen blocks vs unseen region ----
from sklearn.model_selection import GroupKFold, cross_val_predict
main_tr, main_te = df[df.split == "train"], df[df.split == "test"]
cv_pred = cross_val_predict(clone(rf_selected), main_tr[FEATURES], main_tr.label, groups=main_tr.block_id,
                            cv=GroupKFold(n_splits=5))
over = pd.DataFrame([
    {"scored_on": "training data (seen)", "y": main_tr.label, "p": rf_selected.predict(main_tr[FEATURES])},
    {"scored_on": "spatial CV, unseen 100 km blocks", "y": main_tr.label, "p": cv_pred},
    {"scored_on": "Iberia test region (unseen)", "y": main_te.label, "p": rf_selected.predict(main_te[FEATURES])},
])
over["accuracy"] = [accuracy_score(r.y, r.p) for r in over.itertuples()]
over["macro_f1"] = [f1_score(r.y, r.p, labels=CLASSES, average="macro") for r in over.itertuples()]
over = over[["scored_on", "accuracy", "macro_f1"]].round(3)
over.to_csv("outputs/overfit_check.csv", index=False)
print("\nOverfitting check (Random Forest):\n", over.to_string(index=False))
