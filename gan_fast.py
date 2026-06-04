import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bootstrap
from statsmodels.stats.multitest import multipletests

# Set seeds
np.random.seed(42)
import random
random.seed(42)

def cohens_d(sample1, sample2):
    n1, n2 = len(sample1), len(sample2)
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(sample1) - np.mean(sample2)) / pooled_std

def bootstrap_ci(data):
    res = bootstrap((data,), np.mean, n_resamples=500, method='BCa')
    return res.confidence_interval.low, res.confidence_interval.high

# Load data
df = pd.read_csv('full_panel_all_sectors.csv')
outcome_cols = [c for c in df.columns if c.endswith('_post3') and c != 'totexp_mn_post3']
outcome_labels = [c.replace('_post3', '').replace('_mn', '') for c in outcome_cols]
continuous_cols = ['ira_share', 'local_rev_pc', 'enc_gol'] + outcome_cols
categorical_cols = ['dynasty', 'income_class', 'region']

df_clean = df.dropna(subset=continuous_cols + categorical_cols).reset_index(drop=True)
print(f"Rows after dropping missing: {len(df_clean)}")

# Normalize
scaler = MinMaxScaler()
df_clean[continuous_cols] = scaler.fit_transform(df_clean[continuous_cols])
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# Metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_clean)
for col in continuous_cols:
    metadata.update_column(col, sdtype='numerical')
for col in categorical_cols:
    metadata.update_column(col, sdtype='categorical')

print("Training CTGAN...")
model = CTGANSynthesizer(metadata, epochs=500, batch_size=500, verbose=True)
model.fit(df_clean)

dynasty_rows = df_clean[df_clean['dynasty'] == '1'].copy()
print(f"Dynasty observations: {len(dynasty_rows)}")

# Generate a large unconditional synthetic dataset once
print("Generating synthetic dataset...")
n_synthetic = 500000  # large enough to cover all conditions
synthetic = model.sample(n_synthetic)
print(f"Generated {len(synthetic)} synthetic rows.")

# For each dynasty row, find matching synthetic non-dynasty rows
num_matches = 10  # number of counterfactual samples per dynasty row
counterfactuals = []
for idx, row in dynasty_rows.iterrows():
    # Filter synthetic to match conditions
    mask = (synthetic['dynasty'] == '0') & \
           (synthetic['income_class'] == row['income_class']) & \
           (synthetic['region'] == row['region'])
    candidates = synthetic[mask]
    if len(candidates) >= num_matches:
        cf_avg = candidates.sample(num_matches)[outcome_cols].mean()
    else:
        # If not enough, use what's available (or repeat)
        cf_avg = candidates[outcome_cols].mean() if len(candidates) > 0 else pd.Series([np.nan]*len(outcome_cols), index=outcome_cols)
    counterfactuals.append(cf_avg)
    if (idx+1) % 50 == 0:
        print(f"Processed {idx+1}/{len(dynasty_rows)}")

cf_df = pd.DataFrame(counterfactuals)

# Inverse transform
def inv(col, vals):
    dummy = np.zeros((len(vals), len(continuous_cols)))
    idx = continuous_cols.index(col)
    dummy[:, idx] = vals
    return scaler.inverse_transform(dummy)[:, idx]

results = []
for i, outcome in enumerate(outcome_cols):
    real = inv(outcome, dynasty_rows[outcome].values)
    cf = inv(outcome, cf_df[outcome].values)
    diff = cf - real
    mean_diff = np.mean(diff)
    ci_low, ci_high = bootstrap_ci(diff)
    d = cohens_d(cf, real)
    results.append({
        'sector': outcome_labels[i],
        'real_mean': np.mean(real),
        'cf_mean': np.mean(cf),
        'mean_diff': mean_diff,
        'ci_lower': ci_low,
        'ci_upper': ci_high,
        'cohens_d': d
    })

# Multiple testing correction (based on CI excludes zero)
p_vals = [0.01 if (r['ci_lower'] > 0 or r['ci_upper'] < 0) else 0.5 for r in results]
rejected, p_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
for i, r in enumerate(results):
    r['significant'] = rejected[i] and (r['ci_lower'] > 0 or r['ci_upper'] < 0)

# Print results
print("\n" + "="*80)
print("GAN Counterfactual Results (Dynasty vs. Non-Dynasty)")
print("Positive difference = Non-dynasty spends more (dynasty reduces spending)")
print("="*80)
for r in results:
    print(f"\n{r['sector'].upper()}:")
    print(f"  Real mean: {r['real_mean']:.2f}, CF mean: {r['cf_mean']:.2f}")
    print(f"  Diff (CF - Real): {r['mean_diff']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]")
    print(f"  Cohen's d: {r['cohens_d']:.2f}")
    if r['significant']:
        if r['mean_diff'] > 0:
            print("  → Dynasties significantly REDUCE this spending.")
        else:
            print("  → Dynasties significantly INCREASE this spending.")
    else:
        print("  → No significant difference.")

# Plot
fig, ax = plt.subplots(figsize=(10,6))
sectors = [r['sector'] for r in results]
diffs = [r['mean_diff'] for r in results]
err_low = [r['mean_diff'] - r['ci_lower'] for r in results]
err_high = [r['ci_upper'] - r['mean_diff'] for r in results]
ax.barh(sectors, diffs, xerr=[err_low, err_high], capsize=5, color='steelblue')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Difference (Million PHP) - Counterfactual minus Real')
ax.set_title('GAN Counterfactual Effect of Removing a Political Dynasty')
plt.tight_layout()
plt.savefig('sector_effects_gan_fast.png')
plt.show()

pd.DataFrame(results).to_csv('sector_effects_gan.csv', index=False)
print("\nSaved to sector_effects_gan.csv")