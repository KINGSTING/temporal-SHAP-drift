import pandas as pd
import numpy as np
import re
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# Helper: clean LGU names
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
# Load spending panel (full_panel_all_sectors.csv)
# ------------------------------------------------------------
spending = pd.read_csv('../data/full_panel_all_sectors.csv')
print("Spending panel shape:", spending.shape)

health_pre = 'health_mn_pre3'
health_post = 'health_mn_post3'
# For IRA share, we already have the variable 'ira_share' in the panel
# It is the share of IRA over total income in the election year.
# But for post-election, we need the average over t+1..t+3? Or just the value at the end of term?
# To be consistent, we compute the average IRA share over the three years after the election.
# However, the panel has annual ira_share values. We'll compute the post-election average.

# Also include public welfare if desired, but we'll focus on health and IRA share.

# ------------------------------------------------------------
# Load population data (interpolated) – needed for per-capita health
# ------------------------------------------------------------
def load_population():
    df1 = pd.read_excel('../data/2024_T1_1.xlsx', header=None)
    start1 = df1[df1[0] == 'Philippines'].index[0]
    cols1 = [0, 1, 3, 5, 7, 9, 11]
    years1 = [1995, 2000, 2007, 2010, 2015, 2020]
    pop1 = df1.iloc[start1:, cols1].copy()
    pop1.columns = ['LGU_raw'] + years1
    pop1 = pop1.dropna(subset=['LGU_raw'])
    pop1['LGU_raw'] = pop1['LGU_raw'].apply(lambda x: re.sub(r'^\.+', '', str(x)).strip())
    pop1['LGU_clean'] = pop1['LGU_raw'].apply(clean_lgu_name)
    for y in years1:
        pop1[y] = pd.to_numeric(pop1[y], errors='coerce')
    pop1 = pop1[['LGU_clean'] + years1].drop_duplicates('LGU_clean')
    
    df2 = pd.read_excel('../data/2025_T1_1.xlsx', header=None)
    start2 = df2[df2[0] == 'Philippines'].index[0]
    cols2 = [0, 1, 3, 5, 7, 9, 11]
    years2 = [2000, 2007, 2010, 2015, 2020, 2024]
    pop2 = df2.iloc[start2:, cols2].copy()
    pop2.columns = ['LGU_raw'] + years2
    pop2 = pop2.dropna(subset=['LGU_raw'])
    pop2 = pop2[~pop2['LGU_raw'].str.contains('Region|Note|Source|Continued|Table|Land area|Density|Homeless|Embassies', na=False, case=False)]
    pop2['LGU_raw'] = pop2['LGU_raw'].apply(lambda x: re.sub(r'^\.+', '', str(x)).strip())
    pop2['LGU_clean'] = pop2['LGU_raw'].apply(clean_lgu_name)
    for y in years2:
        pop2[y] = pd.to_numeric(pop2[y], errors='coerce')
    pop2 = pop2[['LGU_clean'] + years2].drop_duplicates('LGU_clean')
    
    combined = pop2.merge(pop1[['LGU_clean', 1995]], on='LGU_clean', how='left')
    years_all = [1995, 2000, 2007, 2010, 2015, 2020, 2024]
    
    def interp_row(row):
        known = [(yr, row[yr]) for yr in years_all if pd.notna(row[yr])]
        if len(known) < 2:
            return pd.Series([np.nan]*31, index=range(1992,2023))
        yrs, pops = zip(*known)
        f = interp1d(yrs, pops, kind='linear', fill_value='extrapolate')
        year_range = np.arange(1992, 2023)
        pop_est = f(year_range)
        pop_est = np.maximum(pop_est, 0)
        return pd.Series(pop_est, index=year_range)
    
    pop_interp = combined.set_index('LGU_clean')[years_all].apply(interp_row, axis=1)
    pop_long = pop_interp.stack().reset_index()
    pop_long.columns = ['LGU_clean', 'year', 'population']
    pop_long['year'] = pop_long['year'].astype(int)
    return pop_long

print("Loading population...")
pop_long = load_population()
print(f"Population data: {pop_long['LGU_clean'].nunique()} LGUs, years {pop_long['year'].min()}-{pop_long['year'].max()}")

# ------------------------------------------------------------
# Merge population and compute per capita health growth, and post-election IRA average
# ------------------------------------------------------------
spending['LGU_clean'] = spending['LGU_clean'].apply(clean_lgu_name)
pop_long.rename(columns={'year': 'election_year'}, inplace=True)
spending = spending.merge(pop_long, on=['LGU_clean', 'election_year'], how='left')

# Health per capita
spending['health_pre_pc'] = (spending[health_pre] * 1_000_000) / spending['population']
spending['health_post_pc'] = (spending[health_post] * 1_000_000) / spending['population']
spending = spending.dropna(subset=['health_pre_pc', 'health_post_pc'])
spending['delta_health'] = (spending['health_post_pc'] - spending['health_pre_pc']) / spending['health_pre_pc']
spending['delta_health'] = spending['delta_health'].clip(-0.9, 5)

# For IRA share, we need to compute the average IRA share in the three years after the election.
# The spending panel already has 'ira_share' for each fiscal year.
# We'll group by LGU and election year and compute the mean of ira_share over fiscal_year > election_year and <= election_year+3.
# First, we need the fiscal DataFrame with ira_share.
# But we already have the original fiscal_df? We'll recompute from the spending panel? Actually spending panel has only one row per election cycle, not annual.
# We need the annual fiscal data to compute post-election IRA average.
# Therefore, we reload the fiscal_df and compute the average.

fiscal_df = pd.read_excel('../data/fiscal_data.xlsx')
fiscal_df['LGU_clean'] = fiscal_df['LGU name'].apply(clean_lgu_name)
# Keep needed columns
fiscal_df = fiscal_df[['LGU_clean', 'year', 'election year', 'share of IRA over total income']].copy()
fiscal_df.rename(columns={'year': 'fiscal_year', 'election year': 'election_year', 'share of IRA over total income': 'ira_share'}, inplace=True)
for col in ['fiscal_year', 'election_year', 'ira_share']:
    fiscal_df[col] = pd.to_numeric(fiscal_df[col], errors='coerce')
fiscal_df = fiscal_df.dropna()

# For each election cycle (LGU, election_year), compute average ira_share in fiscal_year > election_year and <= election_year+3
post_ira = []
for idx, row in spending.iterrows():
    lgu = row['LGU_clean']
    elec_yr = row['election_year']
    mask = (fiscal_df['LGU_clean'] == lgu) & (fiscal_df['fiscal_year'] > elec_yr) & (fiscal_df['fiscal_year'] <= elec_yr + 3)
    if mask.any():
        post_ira.append(fiscal_df.loc[mask, 'ira_share'].mean())
    else:
        post_ira.append(np.nan)
spending['ira_post'] = post_ira

# Also pre-election IRA average (for possible control, not used in RDD)
pre_ira = []
for idx, row in spending.iterrows():
    lgu = row['LGU_clean']
    elec_yr = row['election_year']
    mask = (fiscal_df['LGU_clean'] == lgu) & (fiscal_df['fiscal_year'] >= elec_yr - 3) & (fiscal_df['fiscal_year'] < elec_yr)
    if mask.any():
        pre_ira.append(fiscal_df.loc[mask, 'ira_share'].mean())
    else:
        pre_ira.append(np.nan)
spending['ira_pre'] = pre_ira

# ------------------------------------------------------------
# Load election data to get incumbent vote share (margin)
# ------------------------------------------------------------
election = pd.read_excel('../data/election_data.xlsx')
election['LGU_clean'] = election['city'].apply(clean_lgu_name)
election = election[election['position'].str.lower().str.contains('governor')]
election['vote_share'] = election['votes'] / election['total']
election['margin'] = election['vote_share'] - 0.5
election['candidate_clean'] = election['candidate'].str.upper().str.strip()

# Merge incumbent margin into spending (using incumbent_name and election_year)
spending['incumbent_clean'] = spending['incumbent_name'].str.upper().str.strip()
margin_df = election[['LGU_clean', 'year', 'candidate_clean', 'margin']].copy()
margin_df.rename(columns={'year': 'election_year', 'candidate_clean': 'incumbent_clean'}, inplace=True)
spending = spending.merge(margin_df, on=['LGU_clean', 'election_year', 'incumbent_clean'], how='left')
spending = spending.dropna(subset=['margin'])

# Create win indicator
spending['win'] = (spending['margin'] > 0).astype(int)

# Keep only rows with valid outcomes
rdd_df = spending[['LGU_clean', 'election_year', 'margin', 'win', 'delta_health', 'ira_post']].dropna()
print(f"RDD dataset size: {len(rdd_df)}")

# ------------------------------------------------------------
# RDD estimation function (general)
# ------------------------------------------------------------
def rdd_estimate(df, outcome, running, bandwidth):
    df_sub = df[np.abs(df[running]) <= bandwidth].copy()
    df_sub['running_centered'] = df_sub[running]
    X = df_sub[['win', 'running_centered']].copy()
    X['win_x_running'] = df_sub['win'] * df_sub['running_centered']
    X = sm.add_constant(X)
    y = df_sub[outcome]
    model = OLS(y, X).fit(cov_type='HC1')
    te = model.params['win']
    se = model.bse['win']
    return te, se, model, df_sub

bandwidths = [0.03, 0.05, 0.07]
outcomes = ['delta_health', 'ira_post']
outcome_names = ['Health spending growth', 'IRA share (post-election)']

for out_name, out_col in zip(outcome_names, outcomes):
    print(f"\n=== {out_name} ===")
    results = {}
    for bw in bandwidths:
        te, se, model, df_sub = rdd_estimate(rdd_df, out_col, 'margin', bw)
        results[bw] = {'te': te, 'se': se, 'model': model, 'df': df_sub}
        print(f"Bandwidth {bw*100}%: ATE = {te:.4f}, SE = {se:.4f}, 95% CI = [{te-1.96*se:.4f}, {te+1.96*se:.4f}]")
    
    # Plot for the 5% bandwidth
    bw_main = 0.05
    te_main, se_main, model_main, df_main = results[bw_main]['te'], results[bw_main]['se'], results[bw_main]['model'], results[bw_main]['df']
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df_main['margin'], df_main[out_col], alpha=0.3, color='gray', label='Observations')
    margin_grid = np.linspace(-bw_main, bw_main, 100)
    X_left = pd.DataFrame({'const': 1, 'win': 0, 'running_centered': margin_grid, 'win_x_running': 0})
    y_left = model_main.predict(X_left)
    X_right = pd.DataFrame({'const': 1, 'win': 1, 'running_centered': margin_grid, 'win_x_running': margin_grid})
    y_right = model_main.predict(X_right)
    plt.plot(margin_grid, y_left, 'b-', linewidth=2, label='Losers (fitted)')
    plt.plot(margin_grid, y_right, 'r-', linewidth=2, label='Winners (fitted)')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.xlabel('Margin of victory')
    plt.ylabel(out_name)
    plt.title(f'RDD: Effect of Winning on {out_name}')
    plt.legend()
    plt.savefig(f'../data/rdd_plot_{out_col}.png', dpi=300)
    plt.close()
    print(f"Plot saved to ../data/rdd_plot_{out_col}.png")
    
    # Sensitivity plot
    bws = list(results.keys())
    tes = [results[bw]['te'] for bw in bws]
    cis = [1.96 * results[bw]['se'] for bw in bws]
    plt.figure(figsize=(6, 4))
    plt.errorbar(bws, tes, yerr=cis, fmt='o-', capsize=5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Bandwidth')
    plt.ylabel('Estimated treatment effect')
    plt.title(f'Sensitivity: {out_name}')
    plt.savefig(f'../data/rdd_sensitivity_{out_col}.png', dpi=300)
    plt.close()
    print(f"Sensitivity plot saved to ../data/rdd_sensitivity_{out_col}.png")
    
    # Save results to CSV
    results_df = pd.DataFrame([{
        'outcome': out_name,
        'bandwidth': bw,
        'ate': results[bw]['te'],
        'se': results[bw]['se'],
        'ci_lower': results[bw]['te'] - 1.96*results[bw]['se'],
        'ci_upper': results[bw]['te'] + 1.96*results[bw]['se']
    } for bw in bandwidths])
    results_df.to_csv(f'../data/rdd_results_{out_col}.csv', index=False)
    print(f"Results saved to ../data/rdd_results_{out_col}.csv")

print("\nAll RDD analyses completed.")