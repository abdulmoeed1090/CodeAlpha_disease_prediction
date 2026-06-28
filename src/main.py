
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import DATASETS
from train_eval import run_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

all_results = []
all_confusions = {}

for dataset_name, loader in DATASETS.items():
    print(f"\n{'='*60}\n{dataset_name.upper()}\n{'='*60}")
    X, y, feature_names = loader()
    print(f"Shape: {X.shape}, disease rate: {y.mean():.3f}")

    results_df, confusions = run_dataset(dataset_name, X, y, feature_names)
    print(results_df.drop(columns=["dataset"]).round(3).to_string(index=False))

    all_results.append(results_df)
    all_confusions[dataset_name] = confusions

# ---------- Combined comparison table ----------
combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

print(f"\n{'='*60}\nCOMBINED COMPARISON (all datasets x all models)\n{'='*60}")
print(combined.round(3).to_string(index=False))

# ---------- Plot: accuracy comparison grouped by dataset ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, dataset_name in zip(axes, DATASETS.keys()):
    subset = combined[combined["dataset"] == dataset_name].sort_values("accuracy", ascending=True)
    ax.barh(subset["model"], subset["accuracy"], color="#3B7DD8")
    ax.set_xlim(0, 1)
    ax.set_title(dataset_name.replace("_", " ").title())
    ax.set_xlabel("Accuracy")
    for i, v in enumerate(subset["accuracy"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=130)
print(f"\nSaved comparison table and plot to {RESULTS_DIR}/")

# ---------- Best model per dataset ----------
print(f"\n{'='*60}\nBEST MODEL PER DATASET (by ROC-AUC)\n{'='*60}")
best = combined.loc[combined.groupby("dataset")["roc_auc"].idxmax()]
print(best[["dataset", "model", "roc_auc", "accuracy", "f1"]].round(3).to_string(index=False))
