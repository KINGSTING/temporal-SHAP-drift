import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# Load per-capita panel
# ------------------------------------------------------------
df = pd.read_csv('/home/jemarjohn/Documents/Research/mayors-slack-off/data/full_panel_per_capita.csv')

# Keep needed columns: confounders + treatment + conditioning variables
confounders = ['ira_share', 'local_rev_pc', 'enc_gol']
condition_cols = ['dynasty', 'income_class', 'region']
all_needed = confounders + condition_cols + ['LGU_clean', 'election_year']  # keep identifiers for reference
df_clean = df[all_needed].dropna().reset_index(drop=True)
print(f"Total rows: {len(df_clean)}")

# Separate treated (dynasty=1) and control (dynasty=0) for later comparison
treated_real = df_clean[df_clean['dynasty'] == 1].copy()
print(f"Real dynasty rows: {len(treated_real)}")

# ------------------------------------------------------------
# Prepare data for CTGAN (normalize continuous, convert categorical to string)
# ------------------------------------------------------------
cont_cols = confounders
cat_cols = condition_cols

# Normalize continuous
scaler = MinMaxScaler()
df_clean[cont_cols] = scaler.fit_transform(df_clean[cont_cols])
for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# Metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_clean)
for col in cont_cols:
    metadata.update_column(col, sdtype='numerical')
for col in cat_cols:
    metadata.update_column(col, sdtype='categorical')

# Train CTGAN
model = CTGANSynthesizer(metadata, epochs=500, batch_size=500, verbose=True)
model.fit(df_clean)

# ------------------------------------------------------------
# Generate large synthetic dataset once
# ------------------------------------------------------------
synth = model.sample(300000)
print(f"Synthetic rows: {len(synth)}")

# ------------------------------------------------------------
# For each real dynasty row, find matching synthetic counterfactual
# Condition: dynasty='0', same income_class, same region
# ------------------------------------------------------------
counterfactual_rows = []
for idx, row in treated_real.iterrows():
    cond = (synth['dynasty'] == '0') & \
           (synth['income_class'] == row['income_class']) & \
           (synth['region'] == row['region'])
    matches = synth[cond]
    if len(matches) > 0:
        # Take the first match (or average? We need a single row per real dynasty row for balance)
        cf = matches.iloc[0].copy()
    else:
        # If no match, fallback to the real row itself (will produce zero difference)
        cf = row.copy()
        cf['dynasty'] = '0'   # force dynasty=0
    counterfactual_rows.append(cf)

cf_df = pd.DataFrame(counterfactual_rows)
print(f"Counterfactual rows: {len(cf_df)}")

# ------------------------------------------------------------
# Inverse transform continuous columns back to original scale
# ------------------------------------------------------------
def inv_cont(col, vals):
    dummy = np.zeros((len(vals), len(cont_cols)))
    idx = cont_cols.index(col)
    dummy[:, idx] = vals
    return scaler.inverse_transform(dummy)[:, idx]

for col in confounders:
    cf_df[col] = inv_cont(col, cf_df[col].values)
    treated_real[col] = inv_cont(col, treated_real[col].values)

# ------------------------------------------------------------
# Build balance table
# ------------------------------------------------------------
balance = []
for col in confounders + ['income_class', 'region']:
    # For continuous, compute mean, sd, smd
    if col in confounders:
        mean_t = treated_real[col].mean()
        sd_t = treated_real[col].std()
        mean_cf = cf_df[col].mean()
        sd_cf = cf_df[col].std()
        smd = (mean_t - mean_cf) / np.sqrt((sd_t**2 + sd_cf**2)/2)
        balance.append({
            'Variable': col,
            'Real Dynasty Mean': f"{mean_t:.3f}",
            'Real Dynasty SD': f"{sd_t:.3f}",
            'Synthetic CF Mean': f"{mean_cf:.3f}",
            'Synthetic CF SD': f"{sd_cf:.3f}",
            'SMD': f"{smd:.3f}"
        })
    else:
        # For categorical, we show proportion of each category (simplified)
        # For simplicity, we'll just show the distribution as frequency table
        # Here we compute proportion of each level for region and income_class
        # We'll append a separate entry for each level
        pass

# For categorical, we compute proportion of each category
for col in ['income_class', 'region']:
    levels = sorted(set(treated_real[col].unique()) | set(cf_df[col].unique()))
    for lev in levels:
        prop_t = (treated_real[col] == lev).mean()
        prop_cf = (cf_df[col] == lev).mean()
        smd = (prop_t - prop_cf) / np.sqrt((prop_t*(1-prop_t) + prop_cf*(1-prop_cf))/2) if (prop_t>0 and prop_cf>0) else np.nan
        balance.append({
            'Variable': f"{col}={lev}",
            'Real Dynasty Mean': f"{prop_t:.3f}",
            'Real Dynasty SD': "",
            'Synthetic CF Mean': f"{prop_cf:.3f}",
            'Synthetic CF SD': "",
            'SMD': f"{smd:.3f}" if not np.isnan(smd) else ""
        })

balance_df = pd.DataFrame(balance)

# ------------------------------------------------------------
# Print and save
# ------------------------------------------------------------
print("\n" + "="*80)
print("Balance Table: Real Dynasty vs. Synthetic Counterfactual (Conditioned on income_class & region)")
print("="*80)
print(balance_df.to_string(index=False))

balance_df.to_csv('balance_table.csv', index=False)
print("\nBalance table saved to balance_table.csv")

# Also save the counterfactual rows for further analysis
cf_df.to_csv('synthetic_counterfactuals.csv', index=False)
print("Synthetic counterfactual rows saved to synthetic_counterfactuals.csv")