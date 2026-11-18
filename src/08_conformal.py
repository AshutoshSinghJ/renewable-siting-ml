"""
Step 8 (improvement round) - conformal prediction: answers with a guaranteed error rate.

Instead of one label, the model returns a *set* of labels: {Solar}, {Wind}, {Solar, Wind}, ...
Split conformal prediction (class-conditional, "Mondrian") guarantees that the true label is in
the set for about (1 - ALPHA) of new cells *of each class*, provided new cells resemble the
calibration cells. A set with more than one label is an honest "I'm not sure between these".

Calibration data: the leave-one-region-out probabilities from Step 7 (each cell predicted by a
model that never saw its region), so the guarantee is about *new regions*, not neighbouring cells.

Checked by "cross-conformal": calibrate on 10 regions, measure coverage on the 11th.
Also recalibrates Rule B ("unfamiliar place") with region-level distances.

Run from the project root:  python src/08_conformal.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ALPHA = 0.10                                   # target: true label in the set for 90% of cells
CLASSES = ["Hybrid", "Solar", "Wind"]
P = ["p_hybrid", "p_solar", "p_wind"]

o = pd.read_csv("outputs/v2_loro_predictions.csv")
probs = o[P].to_numpy()
y_idx = o.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


MONDRIAN = True                                # True: per-class guarantee; False: one overall guarantee


def thresholds(idx):
    """Conformal threshold(s) on the score 1 - p(true class): per class (Mondrian) or one for all."""
    if not MONDRIAN:
        s = 1 - probs[idx, y_idx[idx]]
        level = min(1.0, np.ceil((len(s) + 1) * (1 - ALPHA)) / len(s))
        qq = float(np.quantile(s, level, method="higher"))
        return {k: qq for k in range(3)}
    q = {}
    for k in range(3):
        s = 1 - probs[idx][y_idx[idx] == k, k]
        n = len(s)
        if n == 0:
            q[k] = 1.0
            continue
        level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
        q[k] = float(np.quantile(s, level, method="higher"))
    return q


def sets(idx, q):
    return np.stack([1 - probs[idx, k] <= q[k] for k in range(3)], axis=1)


def cross_conformal():
    rows = []
    for r in o.region.unique():
        cal, test = np.where(o.region != r)[0], np.where(o.region == r)[0]
        S = sets(test, thresholds(cal))
        covered = S[np.arange(len(test)), y_idx[test]]
        size = S.sum(1)
        single = size == 1
        single_right = (S[single].argmax(1) == y_idx[test][single]).mean() if single.any() else np.nan
        rows.append({"region": r, "cells": len(test), "coverage": covered.mean(), "avg_set_size": size.mean(),
                     "single_label_share": single.mean(), "accuracy_when_single": single_right,
                     "empty_sets": (size == 0).mean()})
    cc = pd.DataFrame(rows)
    tot = {"region": "ALL (pooled)", "cells": cc.cells.sum()}
    for c in ["coverage", "avg_set_size", "single_label_share", "accuracy_when_single", "empty_sets"]:
        tot[c] = np.average(cc[c].fillna(0), weights=cc.cells)
    return pd.concat([cc, pd.DataFrame([tot])]).round(3)


summary = []
for variant in [True, False]:
    MONDRIAN = variant
    tag = "per_class" if variant else "overall"
    cc = cross_conformal()
    cc.to_csv(f"outputs/v2_conformal_by_region_{tag}.csv", index=False)
    print(f"\nCross-conformal ({tag}), target coverage {1 - ALPHA:.0%}:\n", cc.to_string(index=False))
    summary.append({"variant": tag, **cc.iloc[-1].drop(["region", "cells"]).to_dict()})
    q_all = thresholds(np.arange(len(o)))           # final thresholds from all development regions
    json.dump({"alpha": ALPHA, "variant": tag, "thresholds_score_1_minus_p": {CLASSES[k]: v for k, v in q_all.items()}},
              open(f"outputs/v2_conformal_thresholds_{tag}.json", "w"), indent=2)
    print("final thresholds:", {CLASSES[k]: round(v, 3) for k, v in q_all.items()})
pd.DataFrame(summary).to_csv("outputs/v2_conformal_summary.csv", index=False)
print("\n", pd.DataFrame(summary).to_string(index=False))

# ---------------------------------------------------------------- Rule B recalibrated per region
best = open("outputs/v2_best_feature_set.txt").read().splitlines()[1].split(",")
f = pd.read_csv("data/processed/features_v2.csv").set_index("cell_id").loc[o.cell_id, best].reset_index(drop=True)
Z = ((f - f.mean()) / f.std()).to_numpy()
dist = np.empty(len(o))
for r in o.region.unique():
    tr, te = np.where(o.region != r)[0], np.where(o.region == r)[0]
    dist[te] = NearestNeighbors(n_neighbors=1).fit(Z[tr]).kneighbors(Z[te])[0][:, 0]
thr = float(np.percentile(dist, 95))
o["region_distance"] = dist
o["unfamiliar"] = dist > thr
ok = o.pred == o.label
rb = pd.DataFrame([
    {"group": "flagged unfamiliar", "share": o.unfamiliar.mean(), "accuracy": ok[o.unfamiliar].mean()},
    {"group": "not flagged", "share": 1 - o.unfamiliar.mean(), "accuracy": ok[~o.unfamiliar].mean()},
]).round(3)
rb.to_csv("outputs/v2_rule_b.csv", index=False)
json.dump({"features": best, "mean": f.mean().to_dict(), "std": f.std().to_dict(), "threshold": thr},
          open("outputs/v2_rule_b.json", "w"), indent=2)
print(f"\nRule B (threshold = 95th percentile of new-region distances = {thr:.3f}):\n", rb.to_string(index=False))
