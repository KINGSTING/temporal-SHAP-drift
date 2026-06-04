import pandas as pd
import numpy as np
import re
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# Helper: standardize LGU names
# ------------------------------------------------------------
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
# 1. Load 1990-2015 population from Excel
# ------------------------------------------------------------
excel_file = '/home/jemarjohn/Documents/Research/mayors-slack-off/data/1990-2015_popcen.xlsx'
df_old = pd.read_excel(excel_file, header=None)

start_row = df_old[df_old[0] == 'Philippines'].index[0]
col_indices = [0, 1, 2, 3, 5, 7, 9]
pop_table = df_old.iloc[start_row:, col_indices].copy()
pop_table.columns = ['LGU_raw', 1990, 1995, 2000, 2007, 2010, 2015]

pop_table = pop_table.dropna(subset=['LGU_raw'])
pop_table = pop_table[~pop_table['LGU_raw'].str.contains('Region|Note|Source|Continued|Table|Land area|Density|Homeless|Embassies', na=False, case=False)]
pop_table['LGU_raw'] = pop_table['LGU_raw'].apply(lambda x: re.sub(r'^\.+', '', str(x)).strip())
pop_table['LGU_clean'] = pop_table['LGU_raw'].apply(clean_lgu_name)

for yr in [1990, 1995, 2000, 2007, 2010, 2015]:
    pop_table[yr] = pd.to_numeric(pop_table[yr], errors='coerce')

pop_old = pop_table[['LGU_clean'] + [1990,1995,2000,2007,2010,2015]].drop_duplicates('LGU_clean')
print(f"1990-2015: {len(pop_old)} LGUs")

# ------------------------------------------------------------
# 2. Load 2020 population from CSV
# ------------------------------------------------------------
csv_file = '/home/jemarjohn/Documents/Research/mayors-slack-off/data/2020_popcen.csv'
df_2020 = pd.read_csv(csv_file, sep=';', skiprows=1, header=None, encoding='latin-1')
if df_2020.shape[1] == 1:
    df_2020 = pd.read_csv(csv_file, skiprows=1, header=None, encoding='latin-1')
df_2020.columns = ['LGU_raw', 'pop2020']
df_2020 = df_2020.dropna(subset=['LGU_raw'])
region_keywords = 'PHILIPPINES|REGION|NCR|CAR|MIMAROPA|ARMM|BARMM|CALABARZON|BICOL|ILOCOS|CAGAYAN|CENTRAL|EASTERN|WESTERN|ZAMBOANGA|DAVAO|SOCCSKSARGEN|CARAGA|BANGSAMORO'
df_2020 = df_2020[~df_2020['LGU_raw'].str.contains(region_keywords, na=False, case=False)]
df_2020['LGU_clean'] = df_2020['LGU_raw'].apply(clean_lgu_name)
df_2020['pop2020'] = pd.to_numeric(df_2020['pop2020'], errors='coerce')
df_2020 = df_2020.dropna(subset=['pop2020']).drop_duplicates('LGU_clean')
print(f"2020: {len(df_2020)} LGUs")

# ------------------------------------------------------------
# 3. Combine and interpolate
# ------------------------------------------------------------
pop_combined = pop_old.merge(df_2020[['LGU_clean', 'pop2020']], on='LGU_clean', how='outer')
pop_combined[2020] = pop_combined['pop2020']
years = [1990, 1995, 2000, 2007, 2010, 2015, 2020]
pop_data = pop_combined[['LGU_clean'] + years].copy()

def interpolate_row(row):
    known = [(yr, row[yr]) for yr in years if pd.notna(row[yr])]
    if len(known) < 2:
        return pd.Series([np.nan]*31, index=range(1992,2023))
    yrs, pops = zip(*known)
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
# 4. Load spending panel (enhanced)
# ------------------------------------------------------------
spending = pd.read_csv('/home/jemarjohn/Documents/Research/mayors-slack-off/data/full_panel_all_sectors.csv')
spending['LGU_clean'] = spending['LGU_clean'].apply(clean_lgu_name)

# Rename local revenue column if needed
if 'local_rev_mn' in spending.columns and 'local_rev_pc' not in spending.columns:
    spending.rename(columns={'local_rev_mn': 'local_rev_pc'}, inplace=True)
print("Spending columns:", spending.columns.tolist())

pop_long.rename(columns={'year': 'election_year'}, inplace=True)
merged = spending.merge(pop_long, on=['LGU_clean', 'election_year'], how='left')

# ------------------------------------------------------------
# 5. Compute per capita spending
# ------------------------------------------------------------
spending_cols = [c for c in merged.columns if c.endswith('_post3')]
for col in spending_cols:
    percap_col = col.replace('_post3', '_percap')
    merged[percap_col] = (merged[col] * 1_000_000) / merged['population']

keep = ['LGU_clean', 'election_year', 'dynasty', 'ira_share', 'local_rev_pc', 'enc_gol', 'income_class', 'region']
keep += [col.replace('_post3', '_percap') for col in spending_cols]

# Ensure all keep columns exist
missing = [c for c in keep if c not in merged.columns]
if missing:
    print("Missing columns:", missing)
    # Fallback: try to use local_rev_pc if present, else skip
    keep = [c for c in keep if c in merged.columns]

percap_df = merged[keep].dropna(subset=[c for c in keep if '_percap' in c])
print(f"Per capita panel shape: {percap_df.shape}")

percap_df.to_csv('/home/jemarjohn/Documents/Research/mayors-slack-off/data/full_panel_per_capita.csv', index=False)
print("Saved full_panel_per_capita.csv")