"""
Step 3 - Train and evaluate.

Models: (a) majority class, (b) wind-speed rule, (c) logistic regression, (d) Random Forest.
- All tuning / CV happens inside the TRAINING region only, with GroupKFold over 100 km
  spatial blocks (nearby cells are similar; random folds would inflate the scores).
- Every model is then fit on the full training region and evaluated ONCE on the held-out
  test region (Spain + Portugal).

Run from the project root:  python src/03_train_evaluate.py
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["ghi", "wind100", "elev", "slope", "dist_grid"]
CLASSES = ["Hybrid", "Solar", "Wind"]

df = pd.read_csv("data/processed/features.csv")
train, test = df[df.split == "train"], df[df.split == "test"]
X_tr, y_tr, groups = train[FEATURES], train.label, train.block_id
cv = GroupKFold(n_splits=5)   # folds = groups of whole 100 km blocks


class WindRule(BaseEstimator, ClassifierMixin):
    """Baseline: 'Wind if wind100 >= t, else Solar'. t is picked on training data only.
    It never predicts Hybrid - that is the point: can RF beat 'windy = wind'?"""

    def fit(self, X, y):
        self.classes_ = np.array(CLASSES)
        ws = X["wind100"].to_numpy()
        cands = np.quantile(ws, np.linspace(0.02, 0.98, 97))
        scores = [f1_score(y, np.where(ws >= t, "Wind", "Solar"), average="macro",
                           labels=CLASSES, zero_division=0) for t in cands]
        self.t_ = cands[int(np.argmax(scores))]
        return self

    def predict(self, X):
        return np.where(X["wind100"].to_numpy() >= self.t_, "Wind", "Solar")


models = {
    "majority_class": DummyClassifier(strategy="most_frequent"),
    "wind_speed_rule": WindRule(),
    # scaled features + balanced weights so the rare Hybrid class is not ignored
    "logistic_regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(class_weight="balanced", max_iter=2000)),
}

# Random Forest: small grid, tuned with spatial block CV on the training region only
rf_search = GridSearchCV(
    RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
    param_grid={"max_depth": [None, 10, 20], "min_samples_leaf": [1, 5]},
    scoring="f1_macro", cv=cv, n_jobs=1)
rf_search.fit(X_tr, y_tr, groups=groups)
print("RF best params:", rf_search.best_params_, "CV macro F1:", round(rf_search.best_score_, 3))
models["random_forest"] = rf_search.best_estimator_

# ---- Spatial CV scores (training region only) -------------------------------
cv_rows = []
for name, m in models.items():
    s = cross_val_score(m, X_tr, y_tr, groups=groups, cv=cv, scoring="f1_macro")
    cv_rows.append({"model": name, "cv_macro_f1_mean": s.mean(), "cv_macro_f1_std": s.std(),
                    **{f"fold{i+1}": v for i, v in enumerate(s)}})
cv_df = pd.DataFrame(cv_rows).round(3)
grid_df = pd.DataFrame(rf_search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
cv_df.to_csv("outputs/cv_scores.csv", index=False)
grid_df.round(3).to_csv("outputs/cv_rf_grid.csv", index=False)
print("\nSpatial CV (train region):\n", cv_df[["model", "cv_macro_f1_mean", "cv_macro_f1_std"]].to_string(index=False))

# ---- Final fit on full training region, ONE evaluation on the test region ----
def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


# Bootstrap the test cells (2,000 resamples) to get 95% confidence intervals:
# shows whether differences between models are bigger than sampling noise.
y_te = test.label.to_numpy()
rng = np.random.default_rng(SEED)
boot = [rng.integers(0, len(y_te), len(y_te)) for _ in range(2000)]

rows, preds = [], {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred = preds[name] = m.predict(test[FEATURES])
    rec = recall_score(y_te, pred, labels=CLASSES, average=None, zero_division=0)
    acc_b = [accuracy_score(y_te[i], pred[i]) for i in boot]
    f1_b = [macro_f1(y_te[i], pred[i]) for i in boot]
    rows.append({"model": name,
                 "accuracy": accuracy_score(y_te, pred),
                 "accuracy_ci_low": np.percentile(acc_b, 2.5), "accuracy_ci_high": np.percentile(acc_b, 97.5),
                 "macro_f1": macro_f1(y_te, pred),
                 "macro_f1_ci_low": np.percentile(f1_b, 2.5), "macro_f1_ci_high": np.percentile(f1_b, 97.5),
                 **{f"recall_{c}": r for c, r in zip(CLASSES, rec)},
                 # precision for Hybrid: catches a model that "finds" Hybrids by over-predicting them
                 "precision_Hybrid": ((pred == "Hybrid") & (y_te == "Hybrid")).sum() / max(1, (pred == "Hybrid").sum())})
res = pd.DataFrame(rows).round(3)
res.to_csv("outputs/results.csv", index=False)

# Paired bootstrap: same resampled cells for both models, so the difference is a fair comparison
diff_rows = []
for a, b in [("random_forest", "wind_speed_rule"), ("random_forest", "logistic_regression")]:
    for metric, fn in [("accuracy", accuracy_score), ("macro_f1", macro_f1)]:
        d = np.array([fn(y_te[i], preds[a][i]) - fn(y_te[i], preds[b][i]) for i in boot])
        diff_rows.append({"comparison": f"{a} - {b}", "metric": metric, "diff": fn(y_te, preds[a]) - fn(y_te, preds[b]),
                          "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
diffs = pd.DataFrame(diff_rows).round(3)
diffs.to_csv("outputs/paired_differences.csv", index=False)
print(f"\nWind rule threshold: {models['wind_speed_rule'].t_:.2f} m/s")
print("\nHeld-out test region (Spain + Portugal), n =", len(test))
print(res.to_string(index=False))
print("\nPaired bootstrap differences (95% CI):\n", diffs.to_string(index=False))

# Confusion matrix for the Random Forest (counts + row-normalised recall)
rf = models["random_forest"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, norm, title in [(ax[0], None, "Counts"), (ax[1], "true", "Row-normalised (recall)")]:
    ConfusionMatrixDisplay.from_predictions(test.label, rf.predict(test[FEATURES]), labels=CLASSES,
                                            normalize=norm, cmap="Blues", ax=a, colorbar=False,
                                            values_format=".2f" if norm else "d")
    a.set_title(f"Random Forest - test region\n{title}")
plt.tight_layout()
plt.savefig("outputs/confusion_rf.png", dpi=150)

joblib.dump(rf, "outputs/rf_model.joblib")   # reused by Steps 4 and 5
print("\nWrote outputs/results.csv, outputs/paired_differences.csv, outputs/cv_scores.csv, outputs/confusion_rf.png")
