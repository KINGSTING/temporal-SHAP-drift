import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import MinMaxScaler

# Helper: Cohen's d
def cohens_d(sample1, sample2):
    n1, n2 = len(sample1), len(sample2)
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(sample1) - np.mean(sample2)) / pooled_std

# Load data
df = pd.read_csv('full_panel_gan.csv')
df_clean = df.dropna().reset_index(drop=True)
print(f"Rows after dropping missing: {len(df_clean)}")

# Define columns
continuous_cols = ['health_post3', 'educ_post3', 'pubwelf_post3',
                   'ira_share', 'local_rev_pc', 'enc_gol']
categorical_cols = ['dynasty', 'income_class', 'region']

# Normalize continuous
scaler = MinMaxScaler()
df_clean[continuous_cols] = scaler.fit_transform(df_clean[continuous_cols])

# Convert categorical to string
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# Create metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_clean)
# Override column types to ensure correct (optional)
for col in continuous_cols:
    metadata.update_column(col, sdtype='numerical')
for col in categorical_cols:
    metadata.update_column(col, sdtype='categorical')

# Train CTGAN
model = CTGANSynthesizer(metadata, epochs=300, batch_size=500, verbose=True)
model.fit(df_clean)

# Select target LGU
target_lgu = 'LANAO DEL NORTE'
if target_lgu not in df_clean['LGU_clean'].values:
    target_lgu = df_clean[df_clean['dynasty'] == '1']['LGU_clean'].iloc[0]
real_row = df_clean[df_clean['LGU_clean'] == target_lgu].iloc[0]
real_income = real_row['income_class']
real_region = real_row['region']
print(f"Target: {target_lgu}, income={real_income}, region={real_region}")

# Generate synthetic data
num_gen = 200000
synthetic = model.sample(num_gen)
filtered = synthetic[(synthetic['dynasty'] == '0') & 
                     (synthetic['income_class'] == real_income) & 
                     (synthetic['region'] == real_region)]
print(f"Counterfactual samples: {len(filtered)}")
if len(filtered) == 0:
    raise ValueError("No counterfactual samples; increase num_gen")
filtered = filtered.iloc[:10000]

# Real data for that LGU
real_data = df_clean[df_clean['LGU_clean'] == target_lgu]
real_health = real_data['health_post3'].values
real_educ = real_data['educ_post3'].values
real_welfare = real_data['pubwelf_post3'].values

# Inverse transform function
def inv(col, vals):
    dummy = np.zeros((len(vals), len(continuous_cols)))
    idx = continuous_cols.index(col)
    dummy[:, idx] = vals
    return scaler.inverse_transform(dummy)[:, idx]

real_h = inv('health_post3', real_health)
real_e = inv('educ_post3', real_educ)
real_w = inv('pubwelf_post3', real_welfare)
cf_h = inv('health_post3', filtered['health_post3'].values)
cf_e = inv('educ_post3', filtered['educ_post3'].values)
cf_w = inv('pubwelf_post3', filtered['pubwelf_post3'].values)

# KS test and effect size
print("\n" + "="*60)
print("Kolmogorov‑Smirnov Test (Real Dynasty vs. Counterfactual No Dynasty)")
print("="*60)

ks_h, p_h = ks_2samp(real_h, cf_h)
d_h = cohens_d(real_h, cf_h)
print(f"Health spending: KS = {ks_h:.4f}, p = {p_h:.6f}, Cohen's d = {d_h:.2f}")

ks_e, p_e = ks_2samp(real_e, cf_e)
d_e = cohens_d(real_e, cf_e)
print(f"Education spending: KS = {ks_e:.4f}, p = {p_e:.6f}, Cohen's d = {d_e:.2f}")

ks_w, p_w = ks_2samp(real_w, cf_w)
d_w = cohens_d(real_w, cf_w)
print(f"Public welfare: KS = {ks_w:.4f}, p = {p_w:.6f}, Cohen's d = {d_w:.2f}")

print("\nInterpretation: p < 0.05 indicates distributions are significantly different.")
print("Cohen's d: 0.2 = small, 0.5 = medium, 0.8 = large effect.")

print("\n=== Counterfactual Comparison (Means) ===")
print(f"Health: Real mean = {real_h.mean():.2f}, CF mean = {cf_h.mean():.2f}, Diff = {cf_h.mean() - real_h.mean():.2f}")
print(f"Education: Real mean = {real_e.mean():.2f}, CF mean = {cf_e.mean():.2f}, Diff = {cf_e.mean() - real_e.mean():.2f}")
print(f"Welfare: Real mean = {real_w.mean():.2f}, CF mean = {cf_w.mean():.2f}, Diff = {cf_w.mean() - real_w.mean():.2f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(real_h, bins=30, alpha=0.5, label='Real (Dynasty)')
axes[0].hist(cf_h, bins=30, alpha=0.5, label='Counterfactual (No Dynasty)')
axes[0].set_title('Health Spending')
axes[0].legend()
axes[1].hist(real_e, bins=30, alpha=0.5, label='Real')
axes[1].hist(cf_e, bins=30, alpha=0.5, label='Counterfactual')
axes[1].set_title('Education Spending')
axes[1].legend()
axes[2].hist(real_w, bins=30, alpha=0.5, label='Real')
axes[2].hist(cf_w, bins=30, alpha=0.5, label='Counterfactual')
axes[2].set_title('Public Welfare Spending')
axes[2].legend()
plt.tight_layout()
plt.savefig('counterfactual_comparison_with_ks.png')
plt.show()