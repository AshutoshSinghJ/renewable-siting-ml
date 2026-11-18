"""
Step 9 (improvement round) - meaningful patterns.

1. Siting archetypes: k-means clustering of all development sites on their geography only (no
   labels), then describe each cluster and what was actually built there. This shows *types of
   place* (e.g. windy coastal lowlands, sunny high plateaus) and which technology each attracts.
2. SHAP by region: how the final Random Forest uses each feature, overall and per region, and the
   wind-speed dependence curve (where the model switches from Solar to Wind).

Run from the project root:  python src/09_patterns.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
best = open("outputs/v2_best_feature_set.txt").read().splitlines()[1].split(",")
o = pd.read_csv("outputs/v2_loro_predictions.csv")
f = pd.read_csv("data/processed/features_v2.csv").set_index("cell_id")
df = o.merge(f[best + ["lon", "lat"]].reset_index(), on="cell_id")

# ---------------------------------------------------------------- 1. archetypes
CLUSTER_FEATS = ["ghi", "wind100", "elev", "slope", "dist_grid", "wind_rel", "ghi_rel", "rugged", "dist_town", "dist_water"]
CLUSTER_FEATS = [c for c in CLUSTER_FEATS if c in df.columns]
X = df[CLUSTER_FEATS].copy()
for c in ["dist_grid", "dist_town", "dist_water", "slope", "rugged"]:
    if c in X:
        X[c] = np.log1p(X[c].clip(lower=0))                 # distances and slopes are very skewed
Z = StandardScaler().fit_transform(X)
sil = {}
for k in range(4, 9):
    lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)
    sil[k] = silhouette_score(Z, lab, sample_size=5000, random_state=SEED)
K = max(sil, key=sil.get)
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
df["archetype"] = km.labels_
print("silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "-> k =", K)

prof = df.groupby("archetype")[["ghi", "wind100", "elev", "slope", "dist_town", "dist_water"]].median().round(2)
mix = pd.crosstab(df.archetype, df.label, normalize="index").round(3)
acc = df.assign(ok=df.pred == df.label).groupby("archetype").ok.mean().round(3)
where = df.groupby("archetype").region.agg(lambda s: ", ".join(f"{r} {v:.0%}" for r, v in s.value_counts(normalize=True).head(3).items()))
arch = prof.join(mix).join(acc.rename("model_accuracy")).join(df.archetype.value_counts().rename("cells")).join(where.rename("main_regions"))


def name(r):
    """Plain-language name from the cluster's median geography."""
    wind = "very windy" if r.wind100 >= 8 else "windy" if r.wind100 >= 6.6 else "calm"
    sun = "very sunny" if r.ghi >= 4.5 else "sunny" if r.ghi >= 3.5 else "cloudy"
    land = "mountain" if r.elev >= 700 else "upland" if r.elev >= 300 else "coastal" if r.dist_water <= 5 else "lowland"
    return f"{wind}, {sun} {land}"


arch.insert(0, "name", arch.apply(name, axis=1))
arch.to_csv("outputs/v2_archetypes.csv", index_label="archetype")
df[["cell_id", "archetype"]].to_csv("outputs/v2_archetype_cells.csv", index=False)
pd.set_option("display.width", 250)
print(arch.to_string())

fig, ax = plt.subplots(figsize=(9, 5.2))
colors = plt.cm.tab10(np.arange(K))
for k in range(K):
    d = df[df.archetype == k]
    ax.scatter(d.lon, d.lat, s=2, color=colors[k], label=f"{k}: {arch.loc[k, 'name']}")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Siting archetypes (clusters of geography, no labels used)")
ax.legend(markerscale=5, fontsize=8, loc="lower left", frameon=False)
plt.tight_layout(); plt.savefig("outputs/v2_archetypes_map.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 2. SHAP: final model on all development regions
# The final model chosen in Step 7 is Extra Trees. Its trees are deep, so exact SHAP is slow: we explain
# a 100-tree copy (same settings) on a balanced sample of 80 cells per region.
rf = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)
rf.fit(df[best], df.label)
samp = pd.concat([g.sample(min(len(g), 80), random_state=SEED) for _, g in df.groupby("region")])
sv = shap.TreeExplainer(rf)(samp[best]).values               # cells x features x classes
classes = list(rf.classes_)
mean_abs = pd.DataFrame(np.abs(sv).mean(axis=0), index=best, columns=classes)
mean_abs["all"] = mean_abs.mean(axis=1)
mean_abs = mean_abs.sort_values("all", ascending=False).round(4)
mean_abs["share_of_total"] = (mean_abs["all"] / mean_abs["all"].sum()).round(3)
mean_abs.to_csv("outputs/v2_shap_importance.csv", index_label="feature")
print("\nSHAP importance (final model):\n", mean_abs.to_string())

by_region = pd.DataFrame({r: np.abs(sv[(samp.region == r).values]).mean(axis=(0, 2)) for r in samp.region.unique()}, index=best)
by_region = (by_region / by_region.sum()).round(3)
by_region.to_csv("outputs/v2_shap_by_region.csv", index_label="feature")
print("\nShare of SHAP importance by region:\n", by_region.to_string())

wi = classes.index("Wind")
fig, ax = plt.subplots(figsize=(7, 4.2))
for grp, col in [(["British Isles"], "#e15759"), (["India"], "#f28e2b"), (["Iberia", "Italy", "Balkans & Greece"], "#4e79a7"), (["USA"], "#59a14f")]:
    m = samp.region.isin(grp).values
    ax.scatter(samp.wind100[m], sv[m, best.index("wind100"), wi], s=8, alpha=.6, color=col, label=" + ".join(grp))
ax.axhline(0, color="#999", lw=.8)
ax.set_xlabel("Wind speed at 100 m (m/s)"); ax.set_ylabel("SHAP value for 'Wind'")
ax.set_title("How wind speed pushes the prediction towards Wind")
ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig("outputs/v2_shap_wind_dependence.png", dpi=150); plt.close()
print("\nWrote outputs/v2_archetypes*.csv/png, v2_shap_importance.csv, v2_shap_by_region.csv, v2_shap_wind_dependence.png")
