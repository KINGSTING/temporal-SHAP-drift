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
    name = str(name).strip().upper()
    if ',' in name:
        return name.split(',')[0].strip()
    parts = name.split()
    return parts[-1] if parts else ''

def standardize_region(name):
    if pd.isna(name):
        return ''
    name = str(name).strip().upper()
    if 'MIMAROPA' in name:
        return 'MIMAROPA'
    return name

# Load data
input_file = '/home/jemarjohn/Documents/Research/temporal-SHAP-drift/data/fiscal+electoral_data_July 2025.xlsx'
df = pd.read_excel(input_file)
print("Raw data shape:", df.shape)

df['LGU_clean'] = df['lgu'].apply(standardize_lgu)

# Rename columns
rename_dict = {
    'year': 'fiscal_year',
    'elecyr': 'election_year',
    'incumbent': 'incumbent_name',
    'pubwelf': 'pubwelf_mn',
    'educexp': 'educ_mn',
    'healthexp': 'health_mn',
    'govexp': 'gov_mn',
    'socservexp': 'socserv_mn',
    'econdevexp': 'econdev_mn',
    'laborexp': 'labor_mn',
    'housingexp': 'housing_mn',
    'totexp': 'totexp_mn',
    'lgusincome': 'income_mn',
    'tottax': 'tax_mn',
    'totexsrc': 'ext_mn',
    'ira': 'ira_mn',
    'totlocsrc': 'local_rev_mn',
    'region': 'region',
    'lgutype': 'lgutype',
    'party': 'party',
    'sex': 'sex',
    'votes': 'votes',
    'no_cand': 'no_cand',
    'totvot': 'total_votes',
    'ENC_gol': 'enc_gol',
    'incumbent_terms': 'incumbent_terms'
}
df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)

# Convert numeric columns
num_cols = ['fiscal_year', 'election_year', 'health_mn', 'educ_mn', 'pubwelf_mn',
            'socserv_mn', 'econdev_mn', 'labor_mn', 'housing_mn', 'gov_mn', 'totexp_mn',
            'income_mn', 'tax_mn', 'ext_mn', 'ira_mn', 'local_rev_mn', 'enc_gol']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute IRA share
if 'ira_share' not in df.columns:
    if 'ira_mn' in df.columns and 'income_mn' in df.columns:
        income_safe = df['income_mn'].replace(0, np.nan)
        df['ira_share'] = df['ira_mn'] / income_safe
    else:
        df['ira_share'] = np.nan

if 'income_class' not in df.columns:
    df['income_class'] = 3
else:
    df['income_class'] = pd.to_numeric(df['income_class'], errors='coerce').fillna(3).astype(int)

# Keep rows with valid election_year and fiscal_year
df = df[df['election_year'].notna() & df['fiscal_year'].notna()].copy()
print(f"After cleaning: {len(df)} rows")

# Election cycles
cycle_df = df[df['fiscal_year'] == df['election_year']].copy()
print(f"Election cycles found: {len(cycle_df)}")
print("Election years present:", sorted(cycle_df['election_year'].unique()))

# Pre/post spending
spending_cols = ['health_mn', 'educ_mn', 'pubwelf_mn', 'socserv_mn', 'econdev_mn',
                 'labor_mn', 'housing_mn', 'gov_mn', 'totexp_mn']
spending_cols = [c for c in spending_cols if c in df.columns]

# Create a mapping from (LGU, election_year) to pre and post means
pre_dict = {}
post_dict = {}
for lgu in cycle_df['LGU_clean'].unique():
    lgu_fiscal = df[df['LGU_clean'] == lgu][['fiscal_year'] + spending_cols].sort_values('fiscal_year')
    if lgu_fiscal.empty:
        continue
    elec_years = cycle_df[cycle_df['LGU_clean'] == lgu]['election_year'].values
    for elec_yr in elec_years:
        pre_mask = (lgu_fiscal['fiscal_year'] >= elec_yr - 3) & (lgu_fiscal['fiscal_year'] < elec_yr)
        post_mask = (lgu_fiscal['fiscal_year'] > elec_yr) & (lgu_fiscal['fiscal_year'] <= elec_yr + 3)
        pre_mean = lgu_fiscal.loc[pre_mask, spending_cols].mean()
        post_mean = lgu_fiscal.loc[post_mask, spending_cols].mean()
        pre_dict[(lgu, elec_yr)] = pre_mean
        post_dict[(lgu, elec_yr)] = post_mean

# Convert to DataFrame for merging
pre_df = pd.DataFrame([(k[0], k[1]) + tuple(v) for k, v in pre_dict.items()],
                      columns=['LGU_clean', 'election_year'] + [f'{col}_pre3' for col in spending_cols])
post_df = pd.DataFrame([(k[0], k[1]) + tuple(v) for k, v in post_dict.items()],
                       columns=['LGU_clean', 'election_year'] + [f'{col}_post3' for col in spending_cols])

# Merge with cycle_df
cycle_df = cycle_df.merge(pre_df, on=['LGU_clean', 'election_year'], how='left')
cycle_df = cycle_df.merge(post_df, on=['LGU_clean', 'election_year'], how='left')

# Dynasty indicator
cycle_df = cycle_df.sort_values(['LGU_clean', 'election_year'])
cycle_df['incumbent_name_str'] = cycle_df['incumbent_name'].astype(str).str.upper().str.strip()
cycle_df['prev_incumbent'] = cycle_df.groupby('LGU_clean')['incumbent_name_str'].shift(1)
cycle_df['surname_inc'] = cycle_df['incumbent_name_str'].apply(extract_surname)
cycle_df['surname_prev'] = cycle_df['prev_incumbent'].apply(extract_surname)
cycle_df['dynasty'] = (cycle_df['surname_inc'] == cycle_df['surname_prev']).astype(int)

if 'incumbent_terms' not in cycle_df.columns:
    cycle_df['incumbent_terms'] = cycle_df.groupby('LGU_clean')['incumbent_name_str'].transform(
        lambda x: (x == x.shift(1)).cumsum()
    ).fillna(0).astype(int)
else:
    cycle_df['incumbent_terms'] = pd.to_numeric(cycle_df['incumbent_terms'], errors='coerce').fillna(0).astype(int)

# Final columns
output_cols = ['LGU_clean', 'election_year', 'incumbent_name', 'dynasty', 'incumbent_terms',
               'ira_share', 'local_rev_mn', 'enc_gol', 'income_class', 'region']
for col in spending_cols:
    output_cols.append(f'{col}_pre3')
    output_cols.append(f'{col}_post3')
output_cols = [c for c in output_cols if c in cycle_df.columns]
final_df = cycle_df[output_cols].copy()

final_df['region'] = final_df['region'].apply(standardize_region)

# Save
output_path = 'full_panel_all_sectors_enhanced.csv'
final_df.to_csv(output_path, index=False)
print(f"Saved enhanced panel with {len(final_df)} rows to {output_path}")
print("Election years in final dataset:", sorted(final_df['election_year'].unique()))