import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bootstrap

np.random.seed(42)

# Load per-capita panel
df = pd.read_csv('/home/jemarjohn/Documents/Research/mayors-slack-off/data/full_panel_per_capita.csv')
outcome_cols = [c for c in df.columns if c.endswith('_percap')]
cont_confounders = ['ira_share', 'local_rev_pc', 'enc_gol']
cat_confounders = ['income_class', 'region']

# Discretize continuous confounders into quantile bins (5 bins each)
for col in cont_confounders:
    df[col + '_bin'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
    # Convert to string for categorical
    df[col + '_bin'] = df[col + '_bin'].astype(str)

# New categorical columns for conditioning
condition_cols = cat_confounders + [c + '_bin' for c in cont_confounders]

# Keep needed data
keep_cols = condition_cols + ['dynasty'] + outcome_cols
df_clean = df[keep_cols].dropna().reset_index(drop=True)
print(f"Total rows: {len(df_clean)}")

# Normalize outcomes for GAN (outcomes are continuous)
cont_outcomes = outcome_cols
cont_all = cont_outcomes  # only outcomes are continuous now; confounders are binned => categorical
cat_all = condition_cols + ['dynasty']

scaler_gan = MinMaxScaler()
df_gan = df_clean.copy()
df_gan[cont_outcomes] = scaler_gan.fit_transform(df_gan[cont_outcomes])
for col in cat_all:
    df_gan[col] = df_gan[col].astype(str).str.strip()

# Metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_gan)
for col in cont_outcomes:
    metadata.update_column(col, sdtype='numerical')
for col in cat_all:
    metadata.update_column(col, sdtype='categorical')

# Train CTGAN
model = CTGANSynthesizer(metadata, epochs=500, batch_size=500, verbose=True)
model.fit(df_gan)

# Generate synthetic data
synth = model.sample(500000)
print(f"Synthetic rows: {len(synth)}")

# Ensure categorical columns are strings
for col in cat_all:
    synth[col] = synth[col].astype(str).str.strip()

# Separate treated (real dynasty)
treated = df_clean[df_clean['dynasty'] == '1'].copy()
print(f"Treated rows: {len(treated)}")

# For each treated row, find synthetic non-dynasty row with exact match on condition_cols
counterfactuals = []
for idx, row in treated.iterrows():
    # Filter synthetic: dynasty=0 and all condition columns match
    mask = (synth['dynasty'] == '0')
    for col in condition_cols:
        mask = mask & (synth[col] == str(row[col]))
    matches = synth[mask]
    if len(matches) > 0:
        best = matches.iloc[0]  # take the first match
        # Inverse transform outcomes
        # Build a dummy row for all continuous outcomes
        dummy = np.zeros((1, len(cont_outcomes)))
        for i, col in enumerate(cont_outcomes):
            dummy[0, i] = best[col]
        orig_vals = scaler_gan.inverse_transform(dummy)[0]
        outcomes_orig = {col: orig_vals[i] for i, col in enumerate(cont_outcomes)}
        cf_row = pd.Series(outcomes_orig)
    else:
        # fallback: use treated row's outcomes
        cf_row = row[cont_outcomes]
    counterfactuals.append(cf_row)

cf_df = pd.DataFrame(counterfactuals)
real_vals = treated[cont_outcomes].reset_index(drop=True)
cf_vals = cf_df.reset_index(drop=True)

def bootstrap_ci(data):
    res = bootstrap((data,), np.mean, n_resamples=1000, method='BCa')
    return res.confidence_interval

print("\n" + "="*80)
print("GAN Counterfactual Results (Binned Continuous Confounders)")
print("Positive diff = Counterfactual > Real (dynasty reduces spending)")
print("="*80)
for col in cont_outcomes:
    diff = cf_vals[col].values - real_vals[col].values
    mean_diff = np.mean(diff)
    ci = bootstrap_ci(diff)
    sector = col.replace('_percap', '').replace('_mn', '')
    print(f"{sector.upper()}: diff = {mean_diff:.2f}, 95% CI = [{ci.low:.2f}, {ci.high:.2f}]")
    if ci.low <= 0 <= ci.high:
        print("  → Not significant")
    else:
        print("  → Significant")

# Placebo test: permute dynasty within condition_cols groups
df_placebo = df_clean.copy()
df_placebo['dynasty_perm'] = df_placebo.groupby(condition_cols)['dynasty'].transform(
    lambda x: x.sample(frac=1, random_state=42).values
)
treated_placebo = df_placebo[df_placebo['dynasty_perm'] == 1].copy()
print(f"\nPlacebo treated rows: {len(treated_placebo)}")

cf_placebo = []
for idx, row in treated_placebo.iterrows():
    mask = (synth['dynasty'] == '0')
    for col in condition_cols:
        mask = mask & (synth[col] == str(row[col]))
    matches = synth[mask]
    if len(matches) > 0:
        best = matches.iloc[0]
        dummy = np.zeros((1, len(cont_outcomes)))
        for i, col in enumerate(cont_outcomes):
            dummy[0, i] = best[col]
        orig_vals = scaler_gan.inverse_transform(dummy)[0]
        outcomes_orig = {col: orig_vals[i] for i, col in enumerate(cont_outcomes)}
        cf_row = pd.Series(outcomes_orig)
    else:
        cf_row = row[cont_outcomes]
    cf_placebo.append(cf_row)

cf_placebo_df = pd.DataFrame(cf_placebo)
real_placebo_vals = treated_placebo[cont_outcomes].reset_index(drop=True)
cf_placebo_vals = cf_placebo_df.reset_index(drop=True)

print("\n" + "="*80)
print("PLACEBO TEST (Binned Confounders)")
print("All CIs should contain zero")
print("="*80)
for col in cont_outcomes:
    diff = cf_placebo_vals[col].values - real_placebo_vals[col].values
    mean_diff = np.mean(diff)
    ci = bootstrap_ci(diff)
    sector = col.replace('_percap', '').replace('_mn', '')
    print(f"{sector.upper()}: diff = {mean_diff:.2f}, 95% CI = [{ci.low:.2f}, {ci.high:.2f}]")
    if ci.low <= 0 <= ci.high:
        print("  → PASS")
    else:
        print("  → FAIL")