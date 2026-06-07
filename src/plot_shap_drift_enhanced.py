"""
plot_shap_drift_enhanced.py
Generates three visualisations from temporal SHAP drift data:
1. Faceted line plots (small panels) – optional, may move to supplement.
2. Bar plot of change in SHAP importance (pre‑2016 vs. post‑2016).
3. Single‑panel line plot for the top 8 features (recommended for main paper).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Load SHAP drift data
# ------------------------------------------------------------
shap_df = pd.read_csv('../data/temporal_shap_drift.csv', index_col=0)
print("SHAP drift data shape:", shap_df.shape)
print("Election years:", shap_df.index.tolist())

# ------------------------------------------------------------
# 1. Faceted line plots (small, multi‑panel)
# ------------------------------------------------------------
features_of_interest = [
    'dynasty_x_ira', 'ira_share', 'delta_pubwelf', 'dynasty_x_delta_pubwelf',
    'delta_health', 'delta_educ', 'enc_gol', 'local_rev_pc'
]
features_of_interest = [f for f in features_of_interest if f in shap_df.columns]
print("Features for faceted plot:", features_of_interest)

fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten()

for idx, feat in enumerate(features_of_interest):
    ax = axes[idx]
    ax.plot(shap_df.index, shap_df[feat], marker='o', color='steelblue', linewidth=2)
    ax.axvline(x=2016, color='red', linestyle='--', alpha=0.7)
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel('Election year')
    ax.set_ylabel('Mean |SHAP|')
    ax.grid(True, alpha=0.3)

# Hide any unused subplots
for idx in range(len(features_of_interest), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('../data/shap_drift_faceted.png', dpi=300)
plt.close()
print("Faceted plot saved to ../data/shap_drift_faceted.png")

# ------------------------------------------------------------
# 2. Bar plot of pre‑ vs post‑2016 change
# ------------------------------------------------------------
pre_years = [y for y in shap_df.index if y < 2016]
post_years = [y for y in shap_df.index if y >= 2016]

pre_mean = shap_df.loc[pre_years].mean()
post_mean = shap_df.loc[post_years].mean()
change = post_mean - pre_mean
pct_change = (change / pre_mean.abs()) * 100

top_change = change.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in top_change]
top_change.plot(kind='bar', color=colors)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.title('Change in SHAP Importance: Post‑2016 minus Pre‑2016', fontsize=14)
plt.ylabel('Change in Mean |SHAP|', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../data/shap_change_pre_post.png', dpi=300)
plt.close()
print("Pre‑post change bar plot saved to ../data/shap_change_pre_post.png")

# ------------------------------------------------------------
# 3. Single‑panel line plot for top 8 features (publication ready)
# ------------------------------------------------------------
mean_shap = shap_df.mean().sort_values(ascending=False)
top_features = mean_shap.head(8).index.tolist()
print("Top 8 features for line plot:", top_features)

plt.figure(figsize=(8, 5))
for feat in top_features:
    plt.plot(shap_df.index, shap_df[feat], marker='o', linewidth=2, markersize=6, label=feat)

plt.axvline(x=2016, color='red', linestyle='--', alpha=0.7, label='2016 election')
plt.xlabel('Election year', fontsize=12)
plt.ylabel('Mean |SHAP|', fontsize=12)
plt.title('Temporal SHAP Drift (Top 8 Features)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/shap_drift_lineplot.png', dpi=300)
plt.close()
print("Single‑panel line plot saved to ../data/shap_drift_lineplot.png")

print("All plots saved to ../data/")