import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import bootstrap

np.random.seed(42)

# Helper: bootstrap confidence interval
def bootstrap_ci(data):
    res = bootstrap((data,), np.mean, n_resamples=1000, method='BCa')
    return res.confidence_interval

# Load and prepare data
df = pd.read_csv('/home/jemarjohn/Documents/Research/mayors-slack-off/data/full_panel_per_capita.csv')
outcome_cols = [c for c in df.columns if c.endswith('_percap')]
cont_confounders = ['ira_share', 'local_rev_pc', 'enc_gol']
cat_confounders = ['income_class', 'region']
all_confounders = cont_confounders + cat_confounders

keep_cols = all_confounders + ['dynasty'] + outcome_cols
df_clean = df[keep_cols].dropna().reset_index(drop=True)
print(f"Total rows: {len(df_clean)}")

# Standardize continuous confounders for distance
scaler_dist = StandardScaler()
df_clean[cont_confounders] = scaler_dist.fit_transform(df_clean[cont_confounders])

treated = df_clean[df_clean['dynasty'] == 1].copy()
print(f"Treated rows: {len(treated)}")

# Train CTGAN
cont_cols = cont_confounders + outcome_cols
cat_cols = cat_confounders + ['dynasty']

scaler_gan = MinMaxScaler()
df_gan = df_clean.copy()
df_gan[cont_cols] = scaler_gan.fit_transform(df_gan[cont_cols])
for col in cat_cols:
    df_gan[col] = df_gan[col].astype(str).str.strip()

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_gan)
for col in cont_cols:
    metadata.update_column(col, sdtype='numerical')
for col in cat_cols:
    metadata.update_column(col, sdtype='categorical')

model = CTGANSynthesizer(metadata, epochs=500, batch_size=500, verbose=True)
model.fit(df_gan)

# Generate synthetic data
synth = model.sample(500000)
print(f"Synthetic rows: {len(synth)}")

# Prepare synthetic continuous confounders in original scale and normalized
def inv_continuous(df_sub, cols, scaler, all_cont_cols):
    dummy = np.zeros((len(df_sub), len(all_cont_cols)))
    for i, col in enumerate(cols):
        idx = all_cont_cols.index(col)
        dummy[:, idx] = df_sub[col].values
    orig = scaler.inverse_transform(dummy)
    return pd.DataFrame(orig[:, [all_cont_cols.index(c) for c in cols]], columns=cols)

synth_conf_orig = inv_continuous(synth, cont_confounders, scaler_gan, cont_cols)
synth_conf_norm = scaler_dist.transform(synth_conf_orig)
for i, col in enumerate(cont_confounders):
    synth[col + '_norm'] = synth_conf_norm[:, i]

for col in cat_confounders:
    synth[col] = synth[col].astype(str).str.strip()

# Main counterfactual matching
counterfactuals = []
for idx, row in treated.iterrows():
    mask = (synth['dynasty'] == '0') & \
           (synth['income_class'] == str(row['income_class'])) & \
           (synth['region'] == row['region'])
    candidates = synth[mask]
    if len(candidates) > 0:
        cont_vals = row[cont_confounders].values.astype(float)
        cand_vals = candidates[[c + '_norm' for c in cont_confounders]].values.astype(float)
        dists = cdist([cont_vals], cand_vals, metric='euclidean')[0]
        best_idx = np.argmin(dists)
        best = candidates.iloc[best_idx]
        # Inverse transform outcomes from GAN scale to original
        # Build a dummy row for all continuous columns
        dummy_row = np.zeros((1, len(cont_cols)))
        for i_c, col in enumerate(cont_confounders):
            col_idx = cont_cols.index(col)
            dummy_row[0, col_idx] = best[col]   # scalar
        for i_c, col in enumerate(outcome_cols):
            col_idx = cont_cols.index(col)
            dummy_row[0, col_idx] = best[col]   # scalar
        orig_vals = scaler_gan.inverse_transform(dummy_row)[0]
        outcomes_orig = {col: orig_vals[cont_cols.index(col)] for col in outcome_cols}
        cf_row = pd.Series(outcomes_orig)
    else:
        cf_row = row[outcome_cols]
    counterfactuals.append(cf_row)
    if (idx+1) % 50 == 0:
        print(f"Processed {idx+1}/{len(treated)}")

cf_df = pd.DataFrame(counterfactuals)
real_vals = treated[outcome_cols].reset_index(drop=True)
cf_vals = cf_df.reset_index(drop=True)

print("\n" + "="*80)
print("MAIN RESULTS: Corrected GAN Counterfactual")
print("Positive diff = Counterfactual > Real (dynasty reduces spending)")
print("="*80)
for col in outcome_cols:
    diff = cf_vals[col].values - real_vals[col].values
    mean_diff = np.mean(diff)
    ci = bootstrap_ci(diff)
    sector = col.replace('_percap', '').replace('_mn', '')
    print(f"{sector.upper()}: diff = {mean_diff:.2f}, 95% CI = [{ci.low:.2f}, {ci.high:.2f}]")
    if ci.low <= 0 <= ci.high:
        print("  → Not significant")
    else:
        print("  → Significant")

# Placebo test: permute dynasty within groups
df_placebo = df_clean.copy()
df_placebo['dynasty_perm'] = df_placebo.groupby(['income_class', 'region'])['dynasty'].transform(
    lambda x: x.sample(frac=1, random_state=42).values
)
treated_placebo = df_placebo[df_placebo['dynasty_perm'] == 1].copy()
print(f"\nPlacebo treated rows: {len(treated_placebo)}")

cf_placebo = []
for idx, row in treated_placebo.iterrows():
    mask = (synth['dynasty'] == '0') & \
           (synth['income_class'] == str(row['income_class'])) & \
           (synth['region'] == row['region'])
    candidates = synth[mask]
    if len(candidates) > 0:
        cont_vals = row[cont_confounders].values.astype(float)
        cand_vals = candidates[[c + '_norm' for c in cont_confounders]].values.astype(float)
        dists = cdist([cont_vals], cand_vals, metric='euclidean')[0]
        best_idx = np.argmin(dists)
        best = candidates.iloc[best_idx]
        dummy_row = np.zeros((1, len(cont_cols)))
        for i_c, col in enumerate(cont_confounders):
            col_idx = cont_cols.index(col)
            dummy_row[0, col_idx] = best[col]
        for i_c, col in enumerate(outcome_cols):
            col_idx = cont_cols.index(col)
            dummy_row[0, col_idx] = best[col]
        orig_vals = scaler_gan.inverse_transform(dummy_row)[0]
        outcomes_orig = {col: orig_vals[cont_cols.index(col)] for col in outcome_cols}
        cf_row = pd.Series(outcomes_orig)
    else:
        cf_row = row[outcome_cols]
    cf_placebo.append(cf_row)
    if (idx+1) % 50 == 0:
        print(f"Placebo processed {idx+1}/{len(treated_placebo)}")

cf_placebo_df = pd.DataFrame(cf_placebo)
real_placebo_vals = treated_placebo[outcome_cols].reset_index(drop=True)
cf_placebo_vals = cf_placebo_df.reset_index(drop=True)

print("\n" + "="*80)
print("PLACEBO TEST (Randomly Permuted Dynasty)")
print("Expect all 95% CIs to contain zero")
print("="*80)
for col in outcome_cols:
    diff = cf_placebo_vals[col].values - real_placebo_vals[col].values
    mean_diff = np.mean(diff)
    ci = bootstrap_ci(diff)
    sector = col.replace('_percap', '').replace('_mn', '')
    print(f"{sector.upper()}: diff = {mean_diff:.2f}, 95% CI = [{ci.low:.2f}, {ci.high:.2f}]")
    if ci.low <= 0 <= ci.high:
        print("  → PASS")
    else:
        print("  → FAIL")