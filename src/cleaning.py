import pandas as pd
import numpy as np
import re

def standardize_lgu(name):
    if pd.isna(name):
        return ''
    name = str(name).strip().upper()
    name = re.sub(r'\s+PROVINCE$', '', name)
    name = re.sub(r'\s+CITY$', '', name)
    name = re.sub(r'^CITY OF\s+', '', name)
    name = re.sub(r'\*', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def extract_surname(name):
    if pd.isna(name):
        return ''
    parts = str(name).split(',')
    if len(parts) > 1:
        return parts[0].strip().upper()
    else:
        return name.strip().upper().split()[-1]

# Load fiscal data
fiscal_df = pd.read_excel('/home/jemarjohn/Documents/Research/mayors-slack-off/data/fiscal_data.xlsx')
fiscal_df['LGU_clean'] = fiscal_df['LGU name'].apply(standardize_lgu)

# Rename columns
fiscal_df.rename(columns={
    'year': 'fiscal_year',
    'election year': 'election_year',
    "incumbent governor's name": 'incumbent_name',
    'Health Expenditures in millions': 'health_mn',
    'expenditures for education in millions': 'educ_mn',
    'public welfare expenditures in millions': 'pubwelf_mn',
    'Social Services Expenditures': 'socserv_mn',
    'Economic Development Expenditures': 'econdev_mn',
    'Labor Expenditures in millions': 'labor_mn',
    'Housing Expenditures in millions': 'housing_mn',
    'government expenditures in millions': 'gov_mn',
    'Total Expenditures in millions': 'totexp_mn',
    "LGU's Income in millions": 'income_mn',
    'Total Tax Collection (Tax Collection) in millions': 'tax_mn',
    'total external sources in millions': 'ext_mn',
    'Internal Revenue Collection (IRA) in millions': 'ira_mn',
    'total local sources in millions': 'local_rev_mn',
    'share of IRA over total income': 'ira_share',
    'effective number of candidates (Golosov)': 'enc_gol',
    'region': 'region',
    'Income class': 'income_class'
}, inplace=True, errors='ignore')

# Convert numeric
num_cols = ['fiscal_year', 'election_year', 'health_mn', 'educ_mn', 'pubwelf_mn',
            'socserv_mn', 'econdev_mn', 'labor_mn', 'housing_mn', 'gov_mn', 'totexp_mn',
            'income_mn', 'tax_mn', 'ext_mn', 'ira_mn', 'local_rev_mn', 'ira_share', 'enc_gol']
for col in num_cols:
    if col in fiscal_df.columns:
        fiscal_df[col] = pd.to_numeric(fiscal_df[col], errors='coerce')

if 'income_class' not in fiscal_df.columns:
    fiscal_df['income_class'] = 3

keep_cols = ['LGU_clean', 'fiscal_year', 'election_year', 'incumbent_name',
             'health_mn', 'educ_mn', 'pubwelf_mn', 'socserv_mn', 'econdev_mn',
             'labor_mn', 'housing_mn', 'gov_mn', 'totexp_mn',
             'income_mn', 'tax_mn', 'ext_mn', 'ira_mn', 'local_rev_mn',
             'ira_share', 'enc_gol', 'region', 'income_class']
fiscal_df = fiscal_df[[c for c in keep_cols if c in fiscal_df.columns]]
fiscal_df = fiscal_df.reset_index(drop=True)

# Election cycles
cycle_start = fiscal_df[fiscal_df['fiscal_year'] == fiscal_df['election_year']].copy()
print(f"Election cycles: {len(cycle_start)}")

# Load election data
election_df = pd.read_excel('/home/jemarjohn/Documents/Research/mayors-slack-off/data/election_data.xlsx')
election_df = election_df[election_df['position'].str.lower().str.contains('governor')].copy()
election_df['LGU_clean'] = election_df['city'].apply(standardize_lgu)
election_df['vote_share'] = election_df['votes'] / election_df['total']
election_df['margin'] = election_df['vote_share'] - 0.5
election_df['won'] = (election_df['margin'] > 0).astype(int)
election_df['candidate_clean'] = election_df['candidate'].str.upper().str.strip()
cycle_start['incumbent_clean'] = cycle_start['incumbent_name'].str.upper().str.strip()

# Merge
merged = cycle_start.merge(
    election_df[['LGU_clean', 'year', 'candidate_clean', 'vote_share', 'margin', 'won']],
    left_on=['LGU_clean', 'election_year', 'incumbent_clean'],
    right_on=['LGU_clean', 'year', 'candidate_clean'],
    how='inner'
)
merged.rename(columns={'year': 'election_year'}, inplace=True)
merged['reelected'] = merged['won']
merged = merged.loc[:, ~merged.columns.duplicated()]
print(f"Matched cycles: {len(merged)}")

# Compute post-election spending averages
fiscal_lgu = fiscal_df['LGU_clean'].values
fiscal_year = fiscal_df['fiscal_year'].values
spending_cols = ['health_mn', 'educ_mn', 'pubwelf_mn', 'socserv_mn', 'econdev_mn',
                 'labor_mn', 'housing_mn', 'gov_mn', 'totexp_mn']

for col in spending_cols:
    if col not in fiscal_df.columns:
        continue
    arr = fiscal_df[col].values
    post_vals = []
    for row in merged.itertuples():
        lgu = row.LGU_clean
        elec_yr = int(row.election_year)
        mask = (fiscal_lgu == lgu) & (fiscal_year > elec_yr) & (fiscal_year <= elec_yr + 3)
        if np.any(mask):
            post_vals.append(np.nanmean(arr[mask]))
        else:
            post_vals.append(np.nan)
    merged[f'{col}_post3'] = post_vals

# Also compute pre-election spending (t-3 to t-1) for each sector (as confounders)
for col in spending_cols:
    if col not in fiscal_df.columns:
        continue
    arr = fiscal_df[col].values
    pre_vals = []
    for row in merged.itertuples():
        lgu = row.LGU_clean
        elec_yr = int(row.election_year)
        mask = (fiscal_lgu == lgu) & (fiscal_year >= elec_yr - 3) & (fiscal_year < elec_yr)
        if np.any(mask):
            pre_vals.append(np.nanmean(arr[mask]))
        else:
            pre_vals.append(np.nan)
    merged[f'{col}_pre3'] = pre_vals

# Add incumbent years in office (approximate: count previous terms)
merged = merged.sort_values(['LGU_clean', 'election_year'])
merged['incumbent_terms'] = merged.groupby('LGU_clean')['incumbent_name'].transform(
    lambda x: (x == x.shift(1)).cumsum()
)
merged['incumbent_terms'] = merged['incumbent_terms'].fillna(0).astype(int)

# Dynasty indicator
merged['prev_incumbent'] = merged.groupby('LGU_clean')['incumbent_name'].shift(1)
merged['surname_inc'] = merged['incumbent_name'].apply(extract_surname)
merged['surname_prev'] = merged['prev_incumbent'].apply(extract_surname)
merged['dynasty'] = (merged['surname_inc'] == merged['surname_prev']).astype(int)

# Save raw panel with extra features
output_cols = ['LGU_clean', 'election_year', 'incumbent_name', 'dynasty', 'incumbent_terms',
               'ira_share', 'local_rev_mn', 'enc_gol', 'income_class', 'region']
# Add post spending
for col in spending_cols:
    if f'{col}_post3' in merged.columns:
        output_cols.append(f'{col}_post3')
# Add pre spending as confounders
for col in spending_cols:
    if f'{col}_pre3' in merged.columns:
        output_cols.append(f'{col}_pre3')

final_df = merged[output_cols].copy()
final_df.to_csv('full_panel_all_sectors_enhanced.csv', index=False)
print(f"Saved enhanced panel with {len(final_df)} rows")