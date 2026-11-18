"""
Step 7 (improvement round) - do the new features, wider training and other models generalise better?

Evaluation: leave-one-region-out over 11 regions (8 European regions + USA, China, India).
For each region, train on the other 10, predict the held-out one; pool all held-out predictions.
The locked final-test countries are excluded here entirely (see data/processed/final_test_lock.json).

Main scores (all on held-out regions):
  accuracy, macro F1 (3 classes), balanced accuracy, mean of per-region accuracies, worst region.

Run from the project root:  python src/07_global_experiments.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CLASSES = ["Hybrid", "Solar", "Wind"]
BASE = ["ghi", "wind100", "elev", "slope", "dist_grid"]
LOCAL = ["wind_rel", "wind_pct", "ghi_rel", "ghi_pct"]
TERRAIN = ["elev_rel", "rugged"]
ACCESS = ["dist_town", "dist_water"]
FEATURE_SETS = {
    "A. base 5 (current)": BASE,
    "B. base + local wind/sun context": BASE + LOCAL,
    "C. B + terrain context": BASE + LOCAL + TERRAIN,
    "D. C + town/water distance (all 13)": BASE + LOCAL + TERRAIN + ACCESS,
    "E. D without absolute wind/sun": ["elev", "slope", "dist_grid"] + LOCAL + TERRAIN + ACCESS,
}
REGIONS = {
    "Iberia": "ESP PRT", "France & Benelux": "FRA BEL NLD LUX CHE", "Germany & Austria": "DEU AUT",
    "British Isles": "GBR IRL", "Italy": "ITA", "Nordic & Baltic": "DNK NOR SWE FIN EST LVA LTU",
    "Central & Eastern": "POL CZE SVK HUN ROU UKR BLR SVN", "Balkans & Greece": "GRC CYP BGR HRV BIH SRB MNE ALB XKO",
    "USA": "USA", "China": "CHN", "India": "IND",
}
C2R = {c: r for r, cs in REGIONS.items() for c in cs.split()}

df = pd.read_csv("data/processed/features_v2.csv")
df = df[df.split != "final_test"].copy()                     # never touch the locked countries here
df["region"] = df.country.map(C2R)
ALL = BASE + LOCAL + TERRAIN + ACCESS
n0 = len(df)
df = df.dropna(subset=ALL).reset_index(drop=True)            # same cells for every comparison
print(f"{len(df)} of {n0} development cells have all 13 features; regions: {df.region.nunique()}")


def rf():
    return RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=1, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def loro(make_model, feats, data=df, weight_fn=None):
    """Leave-one-region-out predictions (and probabilities) for every cell."""
    pred = pd.Series(index=data.index, dtype=object)
    proba = pd.DataFrame(index=data.index, columns=CLASSES, dtype=float)
    for r in data.region.unique():
        tr, te = data[data.region != r], data[data.region == r]
        m = make_model()
        if weight_fn is None:
            m.fit(tr[feats], tr.label)
        else:
            m.fit(tr[feats], tr.label, sample_weight=weight_fn(tr[feats], te[feats]))
        pred[te.index] = m.predict(te[feats])
        if hasattr(m, "predict_proba"):
            proba.loc[te.index, list(m.classes_)] = m.predict_proba(te[feats])
    return pred, proba.fillna(0.0)


def scores(pred, data=df):
    y = data.label
    per = {r: accuracy_score(y[g.index], pred[g.index]) for r, g in data.groupby("region")}
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro"),
            "balanced_acc": balanced_accuracy_score(y, pred), "mean_region_acc": np.mean(list(per.values())),
            "worst_region_acc": min(per.values()), "worst_region": min(per, key=per.get)}, per


# ---------------------------------------------------------------- 1. feature sets (Random Forest)
rows, per_region, keep = [], {}, {}
for name, feats in FEATURE_SETS.items():
    pred, proba = loro(rf, feats)
    s, per = scores(pred)
    rows.append({"feature_set": name, "n_features": len(feats), **s})
    per_region[name] = per
    keep[name] = (pred, proba)
    print(f"{name:40s} acc {s['accuracy']:.3f}  macroF1 {s['macro_f1']:.3f}  mean-region {s['mean_region_acc']:.3f}  worst {s['worst_region']} {s['worst_region_acc']:.3f}")
fs = pd.DataFrame(rows).round(3)
fs.to_csv("outputs/v2_feature_sets.csv", index=False)
pr = pd.DataFrame(per_region).round(3)
pr.to_csv("outputs/v2_feature_sets_by_region.csv", index_label="region")
print("\nAccuracy by held-out region:\n", pr.to_string())

# pick the best feature set by mean per-region accuracy (rewards working everywhere, not just big regions)
best = fs.sort_values(["mean_region_acc", "macro_f1"], ascending=False).iloc[0].feature_set
BEST = FEATURE_SETS[best]
print(f"\nBest feature set: {best}")

# ---------------------------------------------------------------- 2. models on the best feature set
ws = None


class WindRule:
    """Baseline: Wind if wind100 >= t else Solar, t tuned on the training regions (as in Step 3)."""
    classes_ = np.array(CLASSES)

    def fit(self, X, y, sample_weight=None):
        w = X["wind100"].to_numpy()
        cands = np.quantile(w, np.linspace(0.02, 0.98, 97))
        sc = [f1_score(y, np.where(w >= t, "Wind", "Solar"), labels=CLASSES, average="macro", zero_division=0) for t in cands]
        self.t = cands[int(np.argmax(sc))]
        return self

    def predict(self, X):
        return np.where(X["wind100"].to_numpy() >= self.t, "Wind", "Solar")


MODELS = {
    "Majority class": lambda: __import__("sklearn.dummy", fromlist=["DummyClassifier"]).DummyClassifier(strategy="most_frequent"),
    "Wind-speed rule": WindRule,
    "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)),
    "Random Forest": rf,
    "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(class_weight="balanced", max_depth=6, random_state=SEED),
    # early_stopping is off: in scikit-learn 1.8 it fails with text class labels
    "Neural network (MLP)": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                                                    max_iter=400, random_state=SEED)),
}
rows, model_out = [], {}
for name, mk in MODELS.items():
    feats = BEST if name != "Wind-speed rule" else ["wind100"]
    pred, proba = loro(mk, feats)
    model_out[name] = (pred, proba)
    s, _ = scores(pred)
    rows.append({"model": name, **s})
    print(f"{name:25s} acc {s['accuracy']:.3f}  macroF1 {s['macro_f1']:.3f}  mean-region {s['mean_region_acc']:.3f}  worst {s['worst_region_acc']:.3f}")
md = pd.DataFrame(rows).round(3)
md.to_csv("outputs/v2_models.csv", index=False)
# final model = the learned model with the best mean per-region accuracy (same rule as for feature sets)
FINAL = md[~md.model.isin(["Majority class", "Wind-speed rule"])].sort_values(["mean_region_acc", "macro_f1"], ascending=False).iloc[0].model
print(f"\nFinal model (best mean per-region accuracy): {FINAL}")


# ---------------------------------------------------------------- 3. covariate-shift reweighting
def shift_weights(Xtr, Xte):
    """Weight training cells by how much they resemble the held-out region (uses its features, never its labels)."""
    X = pd.concat([Xtr, Xte])
    d = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X, d)
    p = clf.predict_proba(Xtr)[:, 1]
    w = p / (1 - p) * (len(Xtr) / len(Xte))
    return np.clip(w / w.mean(), 0.1, 10)


pred, _ = loro(rf, BEST, weight_fn=shift_weights)
s, per_rw = scores(pred)
rw = pd.DataFrame([{"variant": "Random Forest, best features", **scores(keep[best][0])[0]},
                   {"variant": "+ covariate-shift reweighting", **s}]).round(3)
rw.to_csv("outputs/v2_reweighting.csv", index=False)
print("\nReweighting:\n", rw.to_string(index=False))

# ---------------------------------------------------------------- 4. save held-out predictions of the final model
pred, proba = model_out[FINAL]
out = df[["cell_id", "region", "country", "label"]].copy()
out["pred"] = pred
out[["p_hybrid", "p_solar", "p_wind"]] = proba[["Hybrid", "Solar", "Wind"]].round(4).values
out["pred_base"] = keep["A. base 5 (current)"][0]
out.to_csv("outputs/v2_loro_predictions.csv", index=False)
open("outputs/v2_best_feature_set.txt", "w").write(best + "\n" + ",".join(BEST) + "\n" + FINAL + "\n")
print("\nWrote outputs/v2_feature_sets.csv, v2_feature_sets_by_region.csv, v2_models.csv, v2_reweighting.csv, v2_loro_predictions.csv")
