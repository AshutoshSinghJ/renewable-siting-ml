# Map of the labelled 5 km cells used in the project: train (Europe) vs held-out test (Iberia), by class.
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
f = pd.read_csv("data/processed/features.csv").merge(pd.read_csv("data/processed/cells.csv")[["cell_id","lon","lat"]])
COL = {"Solar": "#E8A33D", "Wind": "#2A9D8F", "Hybrid": "#7B5EA7"}
fig, ax = plt.subplots(figsize=(7.2, 6.4))
for split, alpha in [("train", 0.45), ("test", 0.9)]:
    for lab in ["Solar", "Wind", "Hybrid"]:
        d = f[(f.split == split) & (f.label == lab)]
        ax.scatter(d.lon, d.lat, s=6 if lab != "Hybrid" else 14, c=COL[lab], alpha=alpha, lw=0,
                   label=lab if split == "train" else None, zorder=3 if lab == "Hybrid" else 2)
t = f[f.split == "test"]
ax.add_patch(plt.Rectangle((t.lon.min() - .6, t.lat.min() - .6), t.lon.max() - t.lon.min() + 1.2,
                           t.lat.max() - t.lat.min() + 1.2, fill=False, ec="#C0392B", lw=1.6, ls="--"))
ax.text(t.lon.min() - .4, t.lat.max() + 0.9, "Held-out test region\n(Spain + Portugal)", color="#C0392B", fontsize=10, weight="bold")
ax.text(14, 66.5, "Training region\n(rest of Europe)", color="#333333", fontsize=10, weight="bold")
ax.set_xlim(-11, 33); ax.set_ylim(35, 70.5); ax.set_aspect(1.4)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
leg = ax.legend(loc="lower right", markerscale=3, frameon=False, title="Cell label")
for h in leg.legend_handles: h.set_alpha(1)
plt.tight_layout(); plt.savefig("outputs/map_split.png", dpi=170)
print("ok")
