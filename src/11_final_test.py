"""
Step 11 (improvement round) - the one-time final test on the locked countries.

The locked countries (Japan, Turkey, Chile, Australia, South Africa; see
data/processed/final_test_lock.json) were chosen before the improvement round and never used for any
choice. Every model below is trained on all development regions (Europe, USA, China, India) and
scored on them exactly once:

  old model      Random Forest, 5 base features, trained on Europe only (the original project)
  baselines      majority class, wind-speed rule (trained on all development regions)
  RF base 5      Random Forest, 5 base features, all development regions (effect of wider training)
  final model    Extra Trees, 13 features, all development regions (effect of new features too)

Plus conformal prediction sets (overall variant, alpha = 0.10) for the final model.

Run from the project root:  python src/11_final_test.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

SEED = 42
CLASSES = ["Hybrid", "Solar", "Wind"]
BASE = ["ghi", "wind100", "elev", "slope", "dist_grid"]
FEATS = open("outputs/v2_best_feature_set.txt").read().splitlines()[1].split(",")
lock = json.load(open("data/processed/final_test_lock.json"))

f = pd.read_csv("data/processed/features_v2.csv")
dev = f[f.split != "final_test"].dropna(subset=FEATS)
test_all = f[f.split == "final_test"]
test = test_all.dropna(subset=FEATS)
print(f"development cells {len(dev)} | locked test cells {len(test)} of {len(test_all)} "
      f"({len(test_all) - len(test)} dropped: missing feature, e.g. no grid line within 20 km)")


def wind_rule(train):
    w = train.wind100.to_numpy()
    cands = np.quantile(w, np.linspace(0.02, 0.98, 97))
    sc = [f1_score(train.label, np.where(w >= t, "Wind", "Solar"), labels=CLASSES, average="macro", zero_division=0) for t in cands]
    return cands[int(np.argmax(sc))]


final = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)
final.fit(dev[FEATS], dev.label)
rf_base = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1)
rf_base.fit(dev[BASE], dev.label)
old = joblib.load("outputs/rf_model.joblib")
t_rule = wind_rule(dev)

preds = {
    "Majority class": DummyClassifier(strategy="most_frequent").fit(dev[BASE], dev.label).predict(test[BASE]),
    f"Wind-speed rule (>= {t_rule:.2f} m/s)": np.where(test.wind100 >= t_rule, "Wind", "Solar"),
    "Old model: RF, 5 features, Europe only": old.predict(test[BASE]),
    "RF, 5 features, all development regions": rf_base.predict(test[BASE]),
    "Final: Extra Trees, 13 features, all regions": final.predict(test[FEATS]),
}
rows = []
for name, p in preds.items():
    per = {c: accuracy_score(test.label[test.country == c], p[(test.country == c).values]) for c in lock["countries"]}
    rows.append({"model": name, "accuracy": accuracy_score(test.label, p),
                 "macro_f1": f1_score(test.label, p, labels=CLASSES, average="macro", zero_division=0),
                 "balanced_acc": balanced_accuracy_score(test.label, p),
                 **{lock["countries"][c]: v for c, v in per.items()}})
res = pd.DataFrame(rows).round(3)
res.to_csv("outputs/v2_final_test.csv", index=False)
pd.set_option("display.width", 250)
print("\nLOCKED FINAL TEST (scored once):\n", res.to_string(index=False))

# confusion matrix of the final model
p = preds["Final: Extra Trees, 13 features, all regions"]
cm = pd.DataFrame(confusion_matrix(test.label, p, labels=CLASSES), index=CLASSES, columns=CLASSES)
cm.to_csv("outputs/v2_final_test_confusion.csv")
print("\nFinal model confusion (rows = truth):\n", cm.to_string())

# overfitting check for the final model: seen data vs unseen regions (Step 7) vs locked test
loro = pd.read_csv("outputs/v2_models.csv").set_index("model").loc["Extra Trees", "accuracy"]
over = pd.DataFrame([{"scored_on": "training data (seen)", "accuracy": accuracy_score(dev.label, final.predict(dev[FEATS]))},
                     {"scored_on": "held-out regions (11-region test)", "accuracy": loro},
                     {"scored_on": "locked final countries", "accuracy": accuracy_score(test.label, p)}]).round(3)
over.to_csv("outputs/v2_overfit_check.csv", index=False)
print("\nOverfitting check (final model):\n", over.to_string(index=False))

# conformal prediction sets on the locked countries
q = json.load(open("outputs/v2_conformal_thresholds_overall.json"))["thresholds_score_1_minus_p"]
proba = pd.DataFrame(final.predict_proba(test[FEATS]), columns=final.classes_, index=test.index)
S = np.stack([1 - proba[c].values <= q[c] for c in CLASSES], axis=1)
yi = test.label.map({c: i for i, c in enumerate(CLASSES)}).values
size = S.sum(1)
single = size == 1
conf = {"coverage": S[np.arange(len(test)), yi].mean(), "avg_set_size": size.mean(), "single_label_share": single.mean(),
        "accuracy_when_single": (S[single].argmax(1) == yi[single]).mean(), "share_unsure": (size > 1).mean()}
pd.DataFrame([conf]).round(3).to_csv("outputs/v2_final_test_conformal.csv", index=False)
print("\nConformal sets on locked countries (target coverage 90%):", {k: round(v, 3) for k, v in conf.items()})

out = test[["cell_id", "country", "lon", "lat", "label"]].copy()
out["pred"] = p
out[["p_hybrid", "p_solar", "p_wind"]] = proba[CLASSES].round(4).values
out["answer"] = ["/".join(c for c, s in zip(CLASSES, row) if s) or "none" for row in S]
out.to_csv("outputs/v2_final_test_predictions.csv", index=False)
joblib.dump(final, "outputs/final_model_extra_trees.joblib")
print("\nWrote outputs/v2_final_test*.csv, v2_overfit_check.csv, final_model_extra_trees.joblib")
