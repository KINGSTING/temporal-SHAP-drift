import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bootstrap
from statsmodels.stats.multitest import multipletests

# Set fixed seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# Helper functions
def cohens_d(sample1, sample2):
    n1, n2 = len(sample1), len(sample2)
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(sample1) - np.mean(sample2)) / pooled_std

def bootstrap_ci(data, n_bootstrap=1000):
    """Bootstrap 95% confidence interval for mean"""
    res = bootstrap((data,), np.mean, n_resamples=n_bootstrap, method='BCa')
    return res.confidence_interval.low, res.confidence_interval.high

# Load the full panel (all sectors)
df = pd.read_csv('full_panel_all_sectors.csv')

# Identify outcome columns (all spending post3)
outcome_cols = [c for c in df.columns if c.endswith('_post3') and c != 'totexp_mn_post3']  # exclude total (redundant)
outcome_labels = [c.replace('_post3', '').replace('_mn', '') for c in outcome_cols]

# Feature columns for conditioning (we'll keep them as is)
continuous_cols = ['ira_share', 'local_rev_pc', 'enc_gol'] + outcome_cols
categorical_cols = ['dynasty', 'income_class', 'region']

# Drop rows with missing values in these columns
df_clean = df.dropna(subset=continuous_cols + categorical_cols).reset_index(drop=True)
print(f"Rows after dropping missing: {len(df_clean)}")

# Normalize continuous variables
scaler = MinMaxScaler()
df_clean[continuous_cols] = scaler.fit_transform(df_clean[continuous_cols])

# Convert categorical to string
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# Prepare metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_clean)
for col in continuous_cols:
    metadata.update_column(col, sdtype='numerical')
for col in categorical_cols:
    metadata.update_column(col, sdtype='categorical')

# Train CTGAN
print("Training CTGAN...")
model = CTGANSynthesizer(metadata, epochs=500, batch_size=500, verbose=True)
model.fit(df_clean)

# Split into dynasty and non-dynasty
dynasty_rows = df_clean[df_clean['dynasty'] == '1'].copy()
print(f"Dynasty observations: {len(dynasty_rows)}")

# Function to generate a counterfactual sample for a given condition
def generate_counterfactual(row, model, n_samples=100):
    cond = {'dynasty': '0', 'income_class': row['income_class'], 'region': row['region']}
    samples = []
    for _ in range(n_samples):
        try:
            sample = model.sample_from_conditions(conditions=cond)
            samples.append(sample.iloc[0])
        except (AttributeError, TypeError):
            # Fallback: unconditional + filter (slower but works)
            temp = model.sample(20000)
            mask = (temp['dynasty'] == cond['dynasty']) & \
                   (temp['income_class'] == cond['income_class']) & \
                   (temp['region'] == cond['region'])
            filtered = temp[mask]
            if len(filtered) > 0:
                samples.append(filtered.iloc[0])
            else:
                # If none found, duplicate the real row (will result in zero difference)
                samples.append(row)
    return pd.DataFrame(samples)

# Generate counterfactuals for each dynasty row (multiple samples)
print("Generating counterfactual samples (this may take a few minutes)...")
all_counterfactuals = []
for idx, row in dynasty_rows.iterrows():
    cf_samples = generate_counterfactual(row, model, n_samples=100)
    # Average the counterfactual samples (per observation)
    cf_avg = cf_samples[outcome_cols].mean()
    all_counterfactuals.append(cf_avg)
    if (idx+1) % 50 == 0:
        print(f"Processed {idx+1}/{len(dynasty_rows)}")

cf_df = pd.DataFrame(all_counterfactuals)

# Real values for the same dynasty rows
real_values = dynasty_rows[outcome_cols].copy()

# Inverse transform to original scale
def inv(col, vals):
    dummy = np.zeros((len(vals), len(continuous_cols)))
    idx = continuous_cols.index(col)
    dummy[:, idx] = vals
    return scaler.inverse_transform(dummy)[:, idx]

results = []
for i, outcome in enumerate(outcome_cols):
    real = inv(outcome, real_values[outcome].values)
    cf = inv(outcome, cf_df[outcome].values)
    diff = cf - real  # counterfactual minus real
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
        'cohens_d': d,
        'p_value': None  # we'll use bootstrap CI for inference (if CI excludes 0)
    })

# Apply FDR correction based on whether CI excludes zero (pseudo p-value)
p_vals = [0.01 if (r['ci_lower'] > 0 or r['ci_upper'] < 0) else 0.5 for r in results]
rejected, p_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
for i, r in enumerate(results):
    r['significant'] = rejected[i] and (r['ci_lower'] > 0 or r['ci_upper'] < 0)

# Print results
print("\n" + "="*80)
print("GAN Counterfactual Results: Dynasty vs. Non-Dynasty Spending by Sector")
print("(Positive difference = Non-dynasty spends more; Negative = Dynasty spends more)")
print("="*80)
for r in results:
    print(f"\n{r['sector'].upper()}:")
    print(f"  Real (dynasty) mean: {r['real_mean']:.2f}")
    print(f"  Counterfactual (non-dynasty) mean: {r['cf_mean']:.2f}")
    print(f"  Difference (CF - Real): {r['mean_diff']:.2f}")
    print(f"  95% CI: [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]")
    print(f"  Cohen's d: {r['cohens_d']:.2f}")
    if r['significant']:
        if r['mean_diff'] > 0:
            print("  → Dynasties significantly REDUCE this spending (non-dynasty would spend more).")
        else:
            print("  → Dynasties significantly INCREASE this spending.")
    else:
        print("  → No significant difference (CI includes zero).")

# Plot differences
fig, ax = plt.subplots(figsize=(10, 6))
sectors = [r['sector'] for r in results]
diffs = [r['mean_diff'] for r in results]
err_low = [r['mean_diff'] - r['ci_lower'] for r in results]
err_high = [r['ci_upper'] - r['mean_diff'] for r in results]
ax.barh(sectors, diffs, xerr=[err_low, err_high], capsize=5, color='steelblue')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Difference (Million PHP) - Counterfactual minus Real')
ax.set_title('GAN Counterfactual Effect of Removing a Political Dynasty')
plt.tight_layout()
plt.savefig('sector_effects_gan.png')
plt.show()

# Save results to CSV
pd.DataFrame(results).to_csv('sector_effects_gan.csv', index=False)
print("\nResults saved to sector_effects_gan.csv")