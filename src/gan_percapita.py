import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bootstrap
from statsmodels.stats.multitest import multipletests

# Set random seeds
np.random.seed(42)
import random
random.seed(42)

# Helper functions
def cohens_d(s1, s2):
    n1, n2 = len(s1), len(s2)
    var1, var2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2)/(n1+n2-2))
    return (np.mean(s1)-np.mean(s2))/pooled if pooled else 0

def bootstrap_ci(data):
    res = bootstrap((data,), np.mean, n_resamples=1000, method='BCa')
    return res.confidence_interval

# Load per‑capita panel
df = pd.read_csv('/home/jemarjohn/Documents/Research/mayors-slack-off/data/full_panel_per_capita.csv')

# Identify per‑capita outcome columns (end with '_percap')
outcome_cols = [c for c in df.columns if c.endswith('_percap')]
outcome_labels = [c.replace('_percap', '').replace('_mn', '') for c in outcome_cols]

# Confounders and categoricals
cont_cols = ['ira_share', 'local_rev_pc', 'enc_gol'] + outcome_cols
cat_cols = ['dynasty', 'income_class', 'region']

# Drop missing
df_clean = df.dropna(subset=cont_cols+cat_cols).reset_index(drop=True)
print(f"Rows after dropping missing: {len(df_clean)}")

# Normalize continuous
scaler = MinMaxScaler()
df_clean[cont_cols] = scaler.fit_transform(df_clean[cont_cols])
for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# Metadata and model
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_clean)
for col in cont_cols:
    metadata.update_column(col, sdtype='numerical')
for col in cat_cols:
    metadata.update_column(col, sdtype='categorical')

model = CTGANSynthesizer(metadata, epochs=500, batch_size=500, verbose=True)
model.fit(df_clean)

# Dynasty rows
dynasty_rows = df_clean[df_clean['dynasty'] == '1'].copy()
print(f"Dynasty rows: {len(dynasty_rows)}")

# Generate large synthetic dataset once
print("Generating synthetic data...")
synth = model.sample(200000)
print(f"Synthetic rows: {len(synth)}")

# For each dynasty row, find matching counterfactuals (dynasty=0, same income_class, region)
counterfactuals = []
for idx, row in dynasty_rows.iterrows():
    cond = (synth['dynasty'] == '0') & \
           (synth['income_class'] == row['income_class']) & \
           (synth['region'] == row['region'])
    matches = synth[cond]
    if len(matches) >= 20:
        # Take only outcome columns, then sample and average
        cf = matches[outcome_cols].sample(20, random_state=42).mean()
    elif len(matches) > 0:
        cf = matches[outcome_cols].mean()
    else:
        cf = row[outcome_cols]   # fallback (no change)
    counterfactuals.append(cf)
    if (idx+1) % 50 == 0:
        print(f"Processed {idx+1}/{len(dynasty_rows)}")

cf_df = pd.DataFrame(counterfactuals)

# Real values
real_vals = dynasty_rows[outcome_cols]

# Inverse transform and compute differences
def inv(col, vals):
    dummy = np.zeros((len(vals), len(cont_cols)))
    idx = cont_cols.index(col)
    dummy[:, idx] = vals
    return scaler.inverse_transform(dummy)[:, idx]

results = []
for i, col in enumerate(outcome_cols):
    real = inv(col, real_vals[col].values)
    cf = inv(col, cf_df[col].values)
    diff = cf - real
    mean_diff = np.mean(diff)
    ci = bootstrap_ci(diff)
    d = cohens_d(cf, real)
    results.append({
        'sector': outcome_labels[i],
        'real_mean': np.mean(real),
        'cf_mean': np.mean(cf),
        'diff': mean_diff,
        'ci_low': ci.low,
        'ci_high': ci.high,
        'cohens_d': d
    })

# FDR correction based on CI excluding zero
p_vals = [0.01 if (r['ci_low'] > 0 or r['ci_high'] < 0) else 0.5 for r in results]
rejected, _, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
for i, r in enumerate(results):
    r['significant'] = rejected[i] and (r['ci_low'] > 0 or r['ci_high'] < 0)

# Print results
print("\n" + "="*80)
print("GAN Counterfactual Results (Per Capita Spending)")
print("Positive difference = Non‑dynasty spends more per person (dynasty reduces per capita spending)")
print("="*80)
for r in results:
    print(f"\n{r['sector'].upper()}:")
    print(f"  Real (dynasty) mean: {r['real_mean']:.2f} PHP")
    print(f"  Counterfactual (non‑dynasty) mean: {r['cf_mean']:.2f} PHP")
    print(f"  Difference (CF - Real): {r['diff']:.2f}")
    print(f"  95% CI: [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
    print(f"  Cohen's d: {r['cohens_d']:.2f}")
    if r['significant']:
        if r['diff'] > 0:
            print("  → Dynasties significantly REDUCE per capita spending.")
        else:
            print("  → Dynasties significantly INCREASE per capita spending.")
    else:
        print("  → No significant difference.")

# Horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))
sectors = [r['sector'] for r in results]
diffs = [r['diff'] for r in results]
err_low = [r['diff'] - r['ci_low'] for r in results]
err_high = [r['ci_high'] - r['diff'] for r in results]
ax.barh(sectors, diffs, xerr=[err_low, err_high], capsize=5, color='steelblue')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Difference (PHP per person) – Counterfactual minus Real')
ax.set_title('Per Capita GAN Counterfactual Effect of Removing Political Dynasty')
plt.tight_layout()
plt.savefig('sector_percapita_gan.png')
plt.show()

# Save results
pd.DataFrame(results).to_csv('sector_percapita_gan.csv', index=False)
print("\nResults saved to sector_percapita_gan.csv and figure saved to sector_percapita_gan.png")