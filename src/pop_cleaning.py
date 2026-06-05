import pandas as pd
import numpy as np
import re
from scipy.interpolate import interp1d

def clean_lgu_name(name):
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

# ------------------------------------------------------------
# 1. Load 1995-2020 population from 2024_T1_1.xlsx
# ------------------------------------------------------------
file_1995_2020 = '../data/2024_T1_1.xlsx'   # adjust path
df1 = pd.read_excel(file_1995_2020, header=None)

# Find start of table
start_row = df1[df1[0] == 'Philippines'].index[0]
# Columns: 0=name, 1=1995, 2=?,3=2000,4=?,5=2007,6=?,7=2010,8=?,9=2015,10=?,11=2020
col_indices = [0, 1, 3, 5, 7, 9, 11]
years1 = [1995, 2000, 2007, 2010, 2015, 2020]
pop1 = df1.iloc[start_row:, col_indices].copy()
pop1.columns = ['LGU_raw'] + years1
pop1 = pop1.dropna(subset=['LGU_raw'])
pop1 = pop1[~pop1['LGU_raw'].str.contains('Region|Note|Source|Continued|Table|Land area|Density|Homeless|Embassies', na=False, case=False)]
pop1['LGU_raw'] = pop1['LGU_raw'].apply(lambda x: re.sub(r'^\.+', '', str(x)).strip())
pop1['LGU_clean'] = pop1['LGU_raw'].apply(clean_lgu_name)
for yr in years1:
    pop1[yr] = pd.to_numeric(pop1[yr], errors='coerce')
pop1 = pop1[['LGU_clean'] + years1].drop_duplicates('LGU_clean')
print(f"1995-2020: {len(pop1)} LGUs")

# ------------------------------------------------------------
# 2. Load 2000-2024 population from 2025_T1_1.xlsx
# ------------------------------------------------------------
file_2000_2024 = '../data/2025_T1_1.xlsx'
df2 = pd.read_excel(file_2000_2024, header=None)
start_row2 = df2[df2[0] == 'Philippines'].index[0]
# Columns: 0=name, 1=2000, 2=?,3=2007,4=?,5=2010,6=?,7=2015,8=?,9=2020,10=?,11=2024
col_indices2 = [0, 1, 3, 5, 7, 9, 11]
years2 = [2000, 2007, 2010, 2015, 2020, 2024]
pop2 = df2.iloc[start_row2:, col_indices2].copy()
pop2.columns = ['LGU_raw'] + years2
pop2 = pop2.dropna(subset=['LGU_raw'])
pop2 = pop2[~pop2['LGU_raw'].str.contains('Region|Note|Source|Continued|Table|Land area|Density|Homeless|Embassies', na=False, case=False)]
pop2['LGU_raw'] = pop2['LGU_raw'].apply(lambda x: re.sub(r'^\.+', '', str(x)).strip())
pop2['LGU_clean'] = pop2['LGU_raw'].apply(clean_lgu_name)
for yr in years2:
    pop2[yr] = pd.to_numeric(pop2[yr], errors='coerce')
pop2 = pop2[['LGU_clean'] + years2].drop_duplicates('LGU_clean')
print(f"2000-2024: {len(pop2)} LGUs")

# ------------------------------------------------------------
# 3. Combine: use pop2 for years 2000-2024, and pop1 for 1995 only (and fill 1995 from pop1)
# ------------------------------------------------------------
pop_combined = pop2.merge(pop1[['LGU_clean', 1995]], on='LGU_clean', how='left')
# Also add 1990? We don't have 1990. We'll extrapolate backward from 1995.

years_all = [1995, 2000, 2007, 2010, 2015, 2020, 2024]
pop_data = pop_combined[['LGU_clean'] + years_all].copy()

# ------------------------------------------------------------
# 4. Interpolate yearly population 1992-2022 (extrapolate back to 1992 using 1995 value and rate)
# ------------------------------------------------------------
def interpolate_row(row):
    known = [(yr, row[yr]) for yr in years_all if pd.notna(row[yr])]
    if len(known) < 2:
        return pd.Series([np.nan]*31, index=range(1992,2023))
    yrs, pops = zip(*known)
    # For years before the first known year, extrapolate using the first two known points.
    f = interp1d(yrs, pops, kind='linear', fill_value='extrapolate')
    year_range = np.arange(1992, 2023)
    pop_est = f(year_range)
    pop_est = np.maximum(pop_est, 0)
    return pd.Series(pop_est, index=year_range)

pop_interp = pop_data.set_index('LGU_clean').apply(interpolate_row, axis=1)
pop_long = pop_interp.stack().reset_index()
pop_long.columns = ['LGU_clean', 'year', 'population']
pop_long['year'] = pop_long['year'].astype(int)
print(f"Interpolated: {pop_long['LGU_clean'].nunique()} LGUs, years {pop_long['year'].min()}-{pop_long['year'].max()}")

# ------------------------------------------------------------
# 5. Merge with spending panel (full_panel_all_sectors.csv)
# ------------------------------------------------------------
spending = pd.read_csv('../data/full_panel_all_sectors.csv')
spending['LGU_clean'] = spending['LGU_clean'].apply(clean_lgu_name)

# Rename local revenue if needed
if 'local_rev_mn' in spending.columns and 'local_rev_pc' not in spending.columns:
    spending.rename(columns={'local_rev_mn': 'local_rev_pc'}, inplace=True)

pop_long.rename(columns={'year': 'election_year'}, inplace=True)
merged = spending.merge(pop_long, on=['LGU_clean', 'election_year'], how='left')

# ------------------------------------------------------------
# 6. Compute per capita spending
# ------------------------------------------------------------
spending_cols = [c for c in merged.columns if c.endswith('_post3')]
for col in spending_cols:
    percap_col = col.replace('_post3', '_percap')
    merged[percap_col] = (merged[col] * 1_000_000) / merged['population']

keep = ['LGU_clean', 'election_year', 'dynasty', 'ira_share', 'local_rev_pc', 'enc_gol', 'income_class', 'region']
keep += [col.replace('_post3', '_percap') for col in spending_cols]

percap_df = merged[keep].dropna(subset=[c for c in keep if '_percap' in c])
print(f"Per capita panel shape: {percap_df.shape}")

percap_df.to_csv('../data/full_panel_per_capita_enhanced.csv', index=False)
print("Saved full_panel_per_capita_enhanced.csv")