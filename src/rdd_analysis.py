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
# Load spending panel
# ------------------------------------------------------------
spending = pd.read_csv('../data/full_panel_all_sectors.csv')
print("Spending panel shape:", spending.shape)

health_pre = 'health_mn_pre3'
health_post = 'health_mn_post3'

# ------------------------------------------------------------
# Load population data (interpolated)
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
# Merge population and compute per capita health growth
# ------------------------------------------------------------
spending['LGU_clean'] = spending['LGU_clean'].apply(clean_lgu_name)
pop_long.rename(columns={'year': 'election_year'}, inplace=True)
spending = spending.merge(pop_long, on=['LGU_clean', 'election_year'], how='left')

spending['health_pre_pc'] = (spending[health_pre] * 1_000_000) / spending['population']
spending['health_post_pc'] = (spending[health_post] * 1_000_000) / spending['population']
spending = spending.dropna(subset=['health_pre_pc', 'health_post_pc'])

spending['delta_health'] = (spending['health_post_pc'] - spending['health_pre_pc']) / spending['health_pre_pc']
spending['delta_health'] = spending['delta_health'].clip(-0.9, 5)

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

# For RDD, we need the outcome after the election. We already have delta_health.
rdd_df = spending[['LGU_clean', 'election_year', 'margin', 'win', 'delta_health']].dropna()
print(f"RDD dataset size: {len(rdd_df)}")

# ------------------------------------------------------------
# RDD estimation using local linear regression (fixed bandwidth)
# ------------------------------------------------------------
def rdd_estimate(df, outcome, running, bandwidth, kernel='triangular'):
    """
    Estimate RDD using local linear regression with given bandwidth.
    Returns: treatment effect, standard error, and model, and subset.
    """
    df_sub = df[np.abs(df[running]) <= bandwidth].copy()
    # Create polynomial terms
    df_sub['running_centered'] = df_sub[running]
    # For local linear, we include interaction with treatment
    X = df_sub[['win', 'running_centered']].copy()
    X['win_x_running'] = df_sub['win'] * df_sub['running_centered']
    X = sm.add_constant(X)
    y = df_sub[outcome]
    model = OLS(y, X).fit(cov_type='HC1')
    te = model.params['win']
    se = model.bse['win']
    return te, se, model, df_sub

bandwidths = [0.03, 0.05, 0.07]  # 3%, 5%, 7%
results = {}
for bw in bandwidths:
    te, se, model, df_sub = rdd_estimate(rdd_df, 'delta_health', 'margin', bw)
    results[bw] = {'te': te, 'se': se, 'model': model, 'df': df_sub}
    print(f"Bandwidth {bw*100}%: ATE = {te:.4f}, SE = {se:.4f}, 95% CI = [{te-1.96*se:.4f}, {te+1.96*se:.4f}]")

# Use the 5% bandwidth for plotting
bw_main = 0.05
te_main, se_main, model_main, df_main = results[bw_main]['te'], results[bw_main]['se'], results[bw_main]['model'], results[bw_main]['df']

# ------------------------------------------------------------
# Plot: outcome vs margin with fitted lines
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df_main['margin'], df_main['delta_health'], alpha=0.3, color='gray', label='Observations')

# Generate prediction lines
margin_grid = np.linspace(-bw_main, bw_main, 100)
# For left side (win=0)
X_left = pd.DataFrame({
    'const': 1,
    'win': 0,
    'running_centered': margin_grid,
    'win_x_running': 0
})
y_left = model_main.predict(X_left)
# For right side (win=1)
X_right = pd.DataFrame({
    'const': 1,
    'win': 1,
    'running_centered': margin_grid,
    'win_x_running': margin_grid
})
y_right = model_main.predict(X_right)

plt.plot(margin_grid, y_left, 'b-', linewidth=2, label='Losers (fitted)')
plt.plot(margin_grid, y_right, 'r-', linewidth=2, label='Winners (fitted)')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel('Margin of victory (vote share - 0.5)')
plt.ylabel('Health spending growth (Δ health per capita)')
plt.title('Regression Discontinuity: Effect of Winning on Health Spending Growth')
plt.legend()
plt.savefig('../data/rdd_plot.png', dpi=300)
plt.close()
print("RDD plot saved to ../data/rdd_plot.png")

# ------------------------------------------------------------
# Sensitivity plot
# ------------------------------------------------------------
bws = list(results.keys())
tes = [results[bw]['te'] for bw in bws]
cis = [1.96 * results[bw]['se'] for bw in bws]

plt.figure(figsize=(6, 4))
plt.errorbar(bws, tes, yerr=cis, fmt='o-', capsize=5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Bandwidth')
plt.ylabel('Estimated treatment effect')
plt.title('Sensitivity of RDD estimate to bandwidth choice')
plt.savefig('../data/rdd_sensitivity.png', dpi=300)
plt.close()
print("Sensitivity plot saved to ../data/rdd_sensitivity.png")

# Save results
results_df = pd.DataFrame([{
    'bandwidth': bw,
    'ate': results[bw]['te'],
    'se': results[bw]['se'],
    'ci_lower': results[bw]['te'] - 1.96*results[bw]['se'],
    'ci_upper': results[bw]['te'] + 1.96*results[bw]['se']
} for bw in bandwidths])
results_df.to_csv('../data/rdd_results.csv', index=False)

print("\nRDD analysis completed. Results saved to ../data/rdd_results.csv")