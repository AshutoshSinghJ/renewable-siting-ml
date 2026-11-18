"""
Step 4 - Explain the Random Forest with SHAP (TreeExplainer).

Explains the model on a fixed random sample of TRAINING cells: SHAP here describes what the
model learned, so it does not need the test region (and a sample keeps TreeExplainer fast).

Run from the project root:  python src/04_explain.py
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

FEATURES = ["ghi", "wind100", "elev", "slope", "dist_grid"]
N_SAMPLE = 1000

rf = joblib.load("outputs/rf_model.joblib")
df = pd.read_csv("data/processed/features.csv")
X = df[df.split == "train"][FEATURES].sample(N_SAMPLE, random_state=42)

sv = shap.TreeExplainer(rf)(X).values          # shape: (cells, features, classes)
classes = list(rf.classes_)

# Beeswarm summary plot per class: direction + size of each feature's effect
for k, c in enumerate(classes):
    plt.figure()
    shap.summary_plot(sv[:, :, k], X, show=False)
    plt.title(f"SHAP - effect on P({c})")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_summary_{c.lower()}.png", dpi=150)
    plt.close()

# Mean |SHAP| per feature and class (global importance)
imp = pd.DataFrame(np.abs(sv).mean(axis=0), index=FEATURES, columns=classes)
imp["mean_abs_shap_all"] = imp[classes].mean(axis=1)
imp["rf_impurity_importance"] = rf.feature_importances_     # used as weights in Step 5
imp = imp.sort_values("mean_abs_shap_all", ascending=False).round(4)
imp.to_csv("outputs/feature_importance.csv", index_label="feature")

ax = imp[classes].plot.barh(figsize=(7, 4))
ax.invert_yaxis()
ax.set_xlabel("mean |SHAP value| (change in class probability)")
ax.set_title("Feature importance per class (Random Forest)")
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150)

print(imp.to_string())
print("\nWrote outputs/shap_summary_*.png, outputs/shap_bar.png, outputs/feature_importance.csv")
