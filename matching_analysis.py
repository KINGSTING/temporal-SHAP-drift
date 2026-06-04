import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

df = pd.read_csv('full_panel_all_sectors.csv')

# Define outcome columns (all post spending except possibly totals)
outcome_cols = [c for c in df.columns if c.endswith('_post3') and c not in ['totexp_mn_post3']]
# Keep totals for reference but separate
outcome_cols = [c for c in outcome_cols if 'totexp' not in c]  # remove total exp for now

# Feature columns for matching
feature_cols = ['ira_share', 'local_rev_pc', 'enc_gol', 'income_class', 'region', 'election_year']

# Separate treated and control
treated = df[df['dynasty'] == 1].copy()
control = df[df['dynasty'] == 0].copy()
print(f"Treated (dynasty): {len(treated)}")
print(f"Control (non-dynasty): {len(control)}")

# Preprocess features: one‑hot encode region and income_class
X_treated = pd.get_dummies(treated[feature_cols], columns=['region', 'income_class'], drop_first=True)
X_control = pd.get_dummies(control[feature_cols], columns=['region', 'income_class'], drop_first=True)

# Align columns
all_cols = X_treated.columns.union(X_control.columns)
X_treated = X_treated.reindex(columns=all_cols, fill_value=0)
X_control = X_control.reindex(columns=all_cols, fill_value=0)

# Normalize
scaler = StandardScaler()
X_treated_scaled = scaler.fit_transform(X_treated)
X_control_scaled = scaler.transform(X_control)

# Nearest neighbor matching (1:1)
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(X_control_scaled)
distances, indices = nn.kneighbors(X_treated_scaled)

matched_control = control.iloc[indices.flatten()].copy()
matched_treated = treated.copy()

# Now compare outcomes for each sector
results = []
for outcome in outcome_cols:
    treated_vals = matched_treated[outcome].values
    control_vals = matched_control[outcome].values
    diff = treated_vals - control_vals
    t_stat, p_val = ttest_ind(treated_vals, control_vals)
    cohen_d = (np.mean(treated_vals) - np.mean(control_vals)) / np.std(np.concatenate([treated_vals, control_vals]))
    results.append({
        'sector': outcome.replace('_post3', '').replace('_mn', ''),
        'dynasty_mean': np.mean(treated_vals),
        'non_dynasty_mean': np.mean(control_vals),
        'difference': diff.mean(),
        'p_value': p_val,
        'cohens_d': cohen_d
    })

# Apply FDR correction for multiple comparisons
p_vals = [r['p_value'] for r in results]
rejected, p_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
for i, r in enumerate(results):
    r['p_corrected'] = p_corrected[i]
    r['significant'] = rejected[i]

# Print results
print("\n" + "="*80)
print("Matching Results: Dynasty vs. Non-Dynasty Spending by Sector")
print("="*80)
for r in results:
    print(f"\n{r['sector'].upper()}:")
    print(f"  Dynasty mean: {r['dynasty_mean']:.2f}, Non-dynasty mean: {r['non_dynasty_mean']:.2f}")
    print(f"  Difference: {r['difference']:.2f}")
    print(f"  Raw p-value: {r['p_value']:.4f}, FDR-corrected p: {r['p_corrected']:.4f}")
    print(f"  Cohen's d: {r['cohens_d']:.2f}")
    if r['significant']:
        if r['difference'] < 0:
            print("  → Dynasties significantly REDUCE this spending.")
        else:
            print("  → Dynasties significantly INCREASE this spending.")
    else:
        print("  → No significant difference.")

# Save results
pd.DataFrame(results).to_csv('sector_effects.csv', index=False)
print("\nResults saved to sector_effects.csv")