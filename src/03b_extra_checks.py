"""
Step 3b (extra checks, not part of the main pipeline) - justify the model choice.

1. Model comparison: would another standard classifier beat the Random Forest?
2. Feature subsets: how much does each feature group add beyond "windy = wind"?

TRAINING REGION ONLY, same 5-fold spatial GroupKFold (100 km blocks) as Step 3.
The held-out test region is NOT used here, so these checks cannot leak into the test result.

Run from the project root:  python src/03b_extra_checks.py
"""
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
FEATURES = ["ghi", "wind100", "elev", "slope", "dist_grid"]
df = pd.read_csv("data/processed/features.csv")
train = df[df.split == "train"]
cv = GroupKFold(n_splits=5)


RF_SELECTED = joblib.load("outputs/rf_model.joblib")   # the Random Forest Step 3 selected


def rf():
    # an unfitted copy with exactly the settings Step 3 selected
    return clone(RF_SELECTED)


def score(model, cols):
    s = cross_validate(model, train[cols], train.label, groups=train.block_id, cv=cv,
                       scoring=["f1_macro", "accuracy", "balanced_accuracy"])
    return {"cv_macro_f1": s["test_f1_macro"].mean(), "cv_macro_f1_std": s["test_f1_macro"].std(),
            "cv_accuracy": s["test_accuracy"].mean(), "cv_balanced_accuracy": s["test_balanced_accuracy"].mean()}


# 1. Other standard classifiers (default settings, class balancing where available)
models = {
    "Random Forest (main model)": rf(),
    "Gradient boosting (HistGradientBoosting)": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                        random_state=SEED, n_jobs=-1),
    "k-nearest neighbours (k=15, scaled)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "SVM RBF (scaled, balanced)": make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=SEED)),
    "Logistic regression (scaled, balanced)": make_pipeline(StandardScaler(),
                                                            LogisticRegression(class_weight="balanced", max_iter=2000)),
}
comp = pd.DataFrame([{"model": n, **score(m, FEATURES)} for n, m in models.items()]).round(3)
comp.to_csv("outputs/extra_model_comparison.csv", index=False)
print("Model comparison (spatial CV, training region):\n", comp.to_string(index=False))

# 2. Feature subsets with the Random Forest
subsets = {
    "wind100 only": ["wind100"],
    "wind100 + ghi": ["wind100", "ghi"],
    "wind100 + ghi + terrain (elev, slope)": ["wind100", "ghi", "elev", "slope"],
    "all 5 features": FEATURES,
}
abl = pd.DataFrame([{"features": n, **score(rf(), c)} for n, c in subsets.items()]).round(3)
abl.to_csv("outputs/feature_ablation.csv", index=False)
print("\nFeature subsets, Random Forest (spatial CV, training region):\n", abl.to_string(index=False))
print("\nWrote outputs/extra_model_comparison.csv, outputs/feature_ablation.csv")
