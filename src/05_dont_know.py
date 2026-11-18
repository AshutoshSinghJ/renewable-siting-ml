"""
Step 5 - "I don't know".

The model abstains on a test cell if EITHER
  Rule A (low confidence):   max class probability < 0.5
  Rule B (unfamiliar place): the cell is far, in feature space, from every training cell
                             (area of applicability idea, Meyer & Pebesma 2021).

Rule B details:
- standardise features with the TRAINING mean/std, multiply by RF feature importance
  (important features count more when judging "similar"),
- distance of each test cell to its nearest training cell,
- threshold = 95th percentile of the same distance for training cells, where each training
  cell is compared only to training cells in OTHER spatial CV folds (so the threshold
  reflects "new region" distances, not a cell's own neighbours).

Run from the project root:  python src/05_dont_know.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

FEATURES = ["ghi", "wind100", "elev", "slope", "dist_grid"]
CLASSES = ["Hybrid", "Solar", "Wind"]
MIN_PROB = 0.5
IDK = "I don't know"

rf = joblib.load("outputs/rf_model.joblib")
df = pd.read_csv("data/processed/features.csv")
train, test = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)

# ---- Rule A: low confidence ---------------------------------------------------
proba = rf.predict_proba(test[FEATURES])
pred = rf.classes_[proba.argmax(axis=1)]
low_conf = proba.max(axis=1) < MIN_PROB

# ---- Rule B: unfamiliar location (weighted feature-space distance) ------------
w = pd.read_csv("outputs/feature_importance.csv", index_col="feature").loc[FEATURES, "rf_impurity_importance"]
mu, sd = train[FEATURES].mean(), train[FEATURES].std()
Z_tr = ((train[FEATURES] - mu) / sd * w).to_numpy()
Z_te = ((test[FEATURES] - mu) / sd * w).to_numpy()

# Threshold from training cells vs other folds (same GroupKFold blocks as Step 3)
cv_dist = np.empty(len(train))
for fit_idx, held_idx in GroupKFold(n_splits=5).split(Z_tr, groups=train.block_id):
    nn = NearestNeighbors(n_neighbors=1).fit(Z_tr[fit_idx])
    cv_dist[held_idx] = nn.kneighbors(Z_tr[held_idx])[0][:, 0]
threshold = np.percentile(cv_dist, 95)

test_dist = NearestNeighbors(n_neighbors=1).fit(Z_tr).kneighbors(Z_te)[0][:, 0]
unfamiliar = test_dist > threshold

final = np.where(low_conf | unfamiliar, IDK, pred)

# ---- Outputs -------------------------------------------------------------------
out = test[["cell_id", "label"]].copy()
out[[f"p_{c.lower()}" for c in rf.classes_]] = proba.round(3)
out["rf_pred"], out["low_conf"], out["aoa_dist"], out["unfamiliar"], out["final_pred"] = \
    pred, low_conf, test_dist.round(4), unfamiliar, final
out.to_csv("outputs/test_predictions.csv", index=False)


def scores(mask):
    y, p = test.label[mask], pred[mask]
    return {"n_cells": int(mask.sum()), "accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)}


conf = ~(low_conf | unfamiliar)
summary = pd.DataFrame([
    {"subset": "all test cells", **scores(np.ones(len(test), bool))},
    {"subset": "confident only (after I don't know)", **scores(conf)},
]).round(3)
summary["pct_I_dont_know"] = [0.0, round(100 * (~conf).mean(), 1)]
summary["pct_low_confidence"] = [np.nan, round(100 * low_conf.mean(), 1)]
summary["pct_unfamiliar"] = [np.nan, round(100 * unfamiliar.mean(), 1)]
summary["aoa_threshold"] = round(threshold, 4)
summary.to_csv("outputs/dont_know_summary.csv", index=False)

print(f"Rule B threshold (95th pct of train-vs-other-fold distance): {threshold:.4f}")
print(summary.to_string(index=False))
print("\nI don't know rate by true class (%):")
print((pd.crosstab(test.label, final == IDK, normalize="index")[True] * 100).round(1).to_string())
print("\nWrote outputs/dont_know_summary.csv, outputs/test_predictions.csv")
