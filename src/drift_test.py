"""
drift_tests.py
Statistical tests for temporal SHAP drift: pre-2016 vs post-2016.
"""

import pandas as pd
import numpy as np
from scipy import stats

# ------------------------------------------------------------
# Load SHAP drift data
# ------------------------------------------------------------
shap_df = pd.read_csv('../data/temporal_shap_drift.csv', index_col=0)
print("SHAP drift data shape:", shap_df.shape)
print("Election years:", shap_df.index.tolist())

# Define pre- and post-2016 periods
pre_years = [y for y in shap_df.index if y < 2016]
post_years = [y for y in shap_df.index if y >= 2016]

print(f"\nPre-2016 years: {pre_years}")
print(f"Post-2016 years: {post_years}")

# ------------------------------------------------------------
# Features to test (focus on interactions and top main effects)
# ------------------------------------------------------------
features_to_test = [
    'dynasty_x_ira',           # key interaction from ICUNSSI paper
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

# Keep only those that exist in the dataframe
features_to_test = [f for f in features_to_test if f in shap_df.columns]
print(f"\nTesting {len(features_to_test)} features: {features_to_test}")

# ------------------------------------------------------------
# Function to compute permutation test p-value
# ------------------------------------------------------------
def permutation_test(pre_vals, post_vals, n_permutations=10000, seed=42):
    """Two-sided permutation test for difference in means."""
    np.random.seed(seed)
    observed_diff = post_vals.mean() - pre_vals.mean()
    combined = np.concatenate([pre_vals, post_vals])
    n_pre = len(pre_vals)
    n_post = len(post_vals)
    count = 0
    for _ in range(n_permutations):
        permuted = np.random.permutation(combined)
        perm_pre = permuted[:n_pre]
        perm_post = permuted[n_pre:]
        perm_diff = perm_post.mean() - perm_pre.mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count / n_permutations

# ------------------------------------------------------------
# Run tests
# ------------------------------------------------------------
results = []

for feat in features_to_test:
    pre_vals = shap_df.loc[pre_years, feat].dropna().values
    post_vals = shap_df.loc[post_years, feat].dropna().values
    
    if len(pre_vals) < 2 or len(post_vals) < 2:
        print(f"Skipping {feat}: insufficient data (pre_n={len(pre_vals)}, post_n={len(post_vals)})")
        continue
    
    # Basic statistics
    mean_pre = pre_vals.mean()
    mean_post = post_vals.mean()
    std_pre = pre_vals.std()
    std_post = post_vals.std()
    
    # Percentage change (handling zero or near-zero pre-mean)
    if abs(mean_pre) < 1e-6:
        pct_change = np.nan
    else:
        pct_change = (mean_post - mean_pre) / abs(mean_pre) * 100
    
    # Welch's t-test (unequal variance)
    t_stat, p_val_t = stats.ttest_ind(pre_vals, post_vals, equal_var=False)
    
    # Mann-Whitney U test (non-parametric, robust for small samples)
    u_stat, p_val_mw = stats.mannwhitneyu(pre_vals, post_vals, alternative='two-sided')
    
    # Permutation test
    p_val_perm = permutation_test(pre_vals, post_vals)
    
    results.append({
        'feature': feat,
        'mean_pre': mean_pre,
        'mean_post': mean_post,
        'pct_change': pct_change,
        't_stat': t_stat,
        'p_value_t': p_val_t,
        'p_value_mw': p_val_mw,
        'p_value_perm': p_val_perm,
        'significant_05': p_val_t < 0.05,
        'n_pre': len(pre_vals),
        'n_post': len(post_vals)
    })

# ------------------------------------------------------------
# Create DataFrame and save
# ------------------------------------------------------------
results_df = pd.DataFrame(results)
# Sort by absolute percentage change (most drift first)
results_df['abs_pct_change'] = results_df['pct_change'].abs()
results_df = results_df.sort_values('abs_pct_change', ascending=False).drop(columns='abs_pct_change')

# Save to CSV
results_df.to_csv('../data/shap_drift_stats.csv', index=False)
print("\n" + "="*70)
print("STATISTICAL TEST RESULTS (pre-2016 vs post-2016)")
print("="*70)
print(results_df[['feature', 'mean_pre', 'mean_post', 'pct_change', 'p_value_t', 'significant_05']].to_string(index=False))

# ------------------------------------------------------------
# Optional: Print a summary for LaTeX table
# ------------------------------------------------------------
print("\n" + "="*70)
print("LATEX TABLE READY (copy-paste into your paper)")
print("="*70)
for _, row in results_df.iterrows():
    sig_star = "*" if row['significant_05'] else ""
    print(f"{row['feature']} & {row['mean_pre']:.4f} & {row['mean_post']:.4f} & {row['pct_change']:.1f}\% & {row['p_value_t']:.3f}{sig_star} \\\\")