"""
plot_enhanced_drift.py
Creates faceted line plots and pre/post‑2016 comparison bar plots with error bars.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ------------------------------------------------------------
# Load SHAP drift data
# ------------------------------------------------------------
shap_df = pd.read_csv('../data/temporal_shap_drift.csv', index_col=0)
print("SHAP drift data shape:", shap_df.shape)
print("Election years:", shap_df.index.tolist())

# ------------------------------------------------------------
# Define features to plot (based on drift test results)
# ------------------------------------------------------------
features_to_plot = [
    'dynasty_x_ira',
    'ira_share',
    'delta_pubwelf',
    'dynasty_x_delta_pubwelf',
    'delta_health',
    'delta_educ',
    'enc_gol',
    'local_rev_pc',
    'dynasty',
    'dynasty_x_delta_educ',
    'dynasty_x_delta_health'
]

# Keep only those present in the data
features_to_plot = [f for f in features_to_plot if f in shap_df.columns]
print("\nPlotting features:", features_to_plot)

# ------------------------------------------------------------
# 1. Faceted line plots (2 rows, 3 columns per facet)
# ------------------------------------------------------------
n_features = len(features_to_plot)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, feat in enumerate(features_to_plot):
    ax = axes[idx]
    ax.plot(shap_df.index, shap_df[feat], marker='o', linestyle='-', 
            color='steelblue', linewidth=2, markersize=8)
    ax.axvline(x=2016, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xlabel('Election year', fontsize=9)
    ax.set_ylabel('Mean |SHAP|', fontsize=9)
    ax.grid(True, alpha=0.3)
    # Set y-axis to start at 0 for easier comparison
    ax.set_ylim(bottom=0)

# Hide any unused subplots
for idx in range(len(features_to_plot), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('../data/shap_drift_faceted.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved faceted line plot: ../data/shap_drift_faceted.png")

# ------------------------------------------------------------
# 2. Pre- vs post-2016 comparison with error bars (bootstrap CIs)
# ------------------------------------------------------------
pre_years = [y for y in shap_df.index if y < 2016]
post_years = [y for y in shap_df.index if y >= 2016]

# Function to compute bootstrap confidence interval for mean difference
def bootstrap_ci(pre_vals, post_vals, n_bootstrap=10000, ci=95):
    np.random.seed(42)
    diff_means = []
    n_pre = len(pre_vals)
    n_post = len(post_vals)
    for _ in range(n_bootstrap):
        pre_sample = np.random.choice(pre_vals, size=n_pre, replace=True)
        post_sample = np.random.choice(post_vals, size=n_post, replace=True)
        diff_means.append(post_sample.mean() - pre_sample.mean())
    lower = np.percentile(diff_means, (100 - ci) / 2)
    upper = np.percentile(diff_means, 100 - (100 - ci) / 2)
    return lower, upper

# Prepare data for bar plot
pre_means = []
post_means = []
ci_lower = []
ci_upper = []
feat_labels = []

for feat in features_to_plot:
    pre_vals = shap_df.loc[pre_years, feat].dropna().values
    post_vals = shap_df.loc[post_years, feat].dropna().values
    if len(pre_vals) == 0 or len(post_vals) == 0:
        continue
    pre_means.append(pre_vals.mean())
    post_means.append(post_vals.mean())
    lower, upper = bootstrap_ci(pre_vals, post_vals)
    ci_lower.append(lower)
    ci_upper.append(upper)
    feat_labels.append(feat)

# Sort by absolute difference for better visual
diffs = np.array(post_means) - np.array(pre_means)
sorted_idx = np.argsort(diffs)[::-1]  # descending
pre_means = np.array(pre_means)[sorted_idx]
post_means = np.array(post_means)[sorted_idx]
ci_lower = np.array(ci_lower)[sorted_idx]
ci_upper = np.array(ci_upper)[sorted_idx]
feat_labels = np.array(feat_labels)[sorted_idx]

# Bar plot
x = np.arange(len(feat_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, pre_means, width, label='Pre‑2016', color='steelblue', alpha=0.7)
bars2 = ax.bar(x + width/2, post_means, width, label='Post‑2016', color='darkorange', alpha=0.7)

# Add error bars for the difference (or you could add separate error bars for each mean)
# Here we show the 95% CI of the difference as a visual guide
ax.errorbar(x, post_means - pre_means, 
            yerr=[post_means - pre_means - ci_lower, ci_upper - (post_means - pre_means)],
            fmt='none', ecolor='black', capsize=3, alpha=0.6)

ax.set_ylabel('Mean |SHAP|', fontsize=12)
ax.set_title('SHAP Importance Before and After 2016', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=10)
ax.legend()
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../data/shap_pre_post_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved pre‑post comparison bar plot: ../data/shap_pre_post_comparison.png")

# ------------------------------------------------------------
# 3. Heatmap of SHAP values over time (optional)
# ------------------------------------------------------------
# Normalise each feature to [0,1] to highlight relative changes
shap_norm = (shap_df - shap_df.min()) / (shap_df.max() - shap_df.min())
plt.figure(figsize=(10, 8))
sns.heatmap(shap_norm.T, cmap='viridis', cbar_kws={'label': 'Normalised SHAP'})
plt.xlabel('Election year')
plt.ylabel('Feature')
plt.title('Temporal SHAP Drift Heatmap (Normalised)')
plt.tight_layout()
plt.savefig('../data/shap_drift_heatmap_normalised.png', dpi=300)
plt.close()
print("Saved normalised heatmap: ../data/shap_drift_heatmap_normalised.png")

# ------------------------------------------------------------
# 4. Save a summary table of pre/post means and change
# ------------------------------------------------------------
summary = pd.DataFrame({
    'feature': feat_labels,
    'mean_pre': pre_means,
    'mean_post': post_means,
    'diff': post_means - pre_means,
    'pct_change': ((post_means - pre_means) / np.abs(pre_means) * 100),
    'ci_diff_lower': ci_lower,
    'ci_diff_upper': ci_upper
})
summary = summary.sort_values('diff', ascending=False)
summary.to_csv('../data/pre_post_summary.csv', index=False)
print("\nSaved pre/post summary table: ../data/pre_post_summary.csv")
print("\nSummary of changes:")
print(summary.to_string(index=False))