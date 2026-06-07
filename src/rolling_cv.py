import pandas as pd
import numpy as np
import re
from scipy.interpolate import interp1d
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

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
educ_pre = 'educ_mn_pre3'
educ_post = 'educ_mn_post3'
pubwelf_pre = 'pubwelf_mn_pre3'
pubwelf_post = 'pubwelf_mn_post3'

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
# Merge population and compute per capita growth rates
# ------------------------------------------------------------
spending['LGU_clean'] = spending['LGU_clean'].apply(clean_lgu_name)
pop_long.rename(columns={'year': 'election_year'}, inplace=True)
spending = spending.merge(pop_long, on=['LGU_clean', 'election_year'], how='left')

# Health per capita
spending['health_pre_pc'] = (spending[health_pre] * 1_000_000) / spending['population']
spending['health_post_pc'] = (spending[health_post] * 1_000_000) / spending['population']
# Education per capita
spending['educ_pre_pc'] = (spending[educ_pre] * 1_000_000) / spending['population']
spending['educ_post_pc'] = (spending[educ_post] * 1_000_000) / spending['population']
# Public welfare per capita
spending['pubwelf_pre_pc'] = (spending[pubwelf_pre] * 1_000_000) / spending['population']
spending['pubwelf_post_pc'] = (spending[pubwelf_post] * 1_000_000) / spending['population']

spending = spending.dropna(subset=['health_pre_pc', 'health_post_pc', 'educ_pre_pc', 'educ_post_pc', 'pubwelf_pre_pc', 'pubwelf_post_pc'])

# Growth rates
spending['delta_health'] = (spending['health_post_pc'] - spending['health_pre_pc']) / spending['health_pre_pc']
spending['delta_educ'] = (spending['educ_post_pc'] - spending['educ_pre_pc']) / spending['educ_pre_pc']
spending['delta_pubwelf'] = (spending['pubwelf_post_pc'] - spending['pubwelf_pre_pc']) / spending['pubwelf_pre_pc']
spending['delta_health'] = spending['delta_health'].clip(-0.9, 5)
spending['delta_educ'] = spending['delta_educ'].clip(-0.9, 5)
spending['delta_pubwelf'] = spending['delta_pubwelf'].clip(-0.9, 5)

# ------------------------------------------------------------
# Compute re‑election target using election data
# ------------------------------------------------------------
election = pd.read_excel('../data/election_data.xlsx')
election['LGU_clean'] = election['city'].apply(clean_lgu_name)
election = election[election['position'].str.lower().str.contains('governor')]
election['vote_share'] = election['votes'] / election['total']
election['won'] = (election['vote_share'] > 0.5).astype(int)
election['candidate_clean'] = election['candidate'].str.upper().str.strip()

winners = election[election['won'] == 1][['LGU_clean', 'year', 'candidate_clean']]
winners.rename(columns={'year': 'election_year', 'candidate_clean': 'winner_name'}, inplace=True)

spending = spending.merge(winners, on=['LGU_clean', 'election_year'], how='left')
next_winners = winners.copy()
next_winners['election_year'] = next_winners['election_year'] - 3
next_winners.rename(columns={'winner_name': 'next_winner'}, inplace=True)
spending = spending.merge(next_winners, on=['LGU_clean', 'election_year'], how='left')

spending['reelected'] = ((spending['incumbent_name'].str.upper().str.strip() == spending['next_winner']) & spending['next_winner'].notna()).astype(int)
spending = spending.dropna(subset=['reelected', 'next_winner'])
print(f"Spending after adding reelected: {len(spending)} rows")

# ------------------------------------------------------------
# Prepare feature dataset (including interactions)
# ------------------------------------------------------------
df_model = spending[['LGU_clean', 'election_year', 'delta_health', 'delta_educ', 'delta_pubwelf',
                     'local_rev_mn', 'enc_gol', 'dynasty', 'region', 'reelected', 'ira_share']].copy()
df_model.rename(columns={'reelected': 'won', 'local_rev_mn': 'local_rev_pc'}, inplace=True)
df_model = df_model.drop_duplicates(subset=['LGU_clean', 'election_year'])

prev_margin = election[['LGU_clean', 'year', 'vote_share']].copy()
prev_margin.rename(columns={'year': 'election_year', 'vote_share': 'prev_margin'}, inplace=True)
df_model = df_model.merge(prev_margin, on=['LGU_clean', 'election_year'], how='left')
df_model['prev_margin'] = df_model['prev_margin'].fillna(0.5)

# Create interaction terms
df_model['dynasty_x_delta_health'] = df_model['dynasty'] * df_model['delta_health']
df_model['dynasty_x_delta_educ'] = df_model['dynasty'] * df_model['delta_educ']
df_model['dynasty_x_delta_pubwelf'] = df_model['dynasty'] * df_model['delta_pubwelf']
df_model['dynasty_x_ira'] = df_model['dynasty'] * df_model['ira_share']

df_model = df_model.dropna(subset=['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 'prev_margin', 'won', 'ira_share'])
print(f"Final dataset: {df_model.shape[0]} observations")
print(df_model['won'].value_counts())

# ------------------------------------------------------------
# Feature engineering (one‑hot encode region, include interactions)
# ------------------------------------------------------------
feature_cols = ['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 
                'dynasty', 'prev_margin', 'ira_share',
                'dynasty_x_delta_health', 'dynasty_x_delta_educ', 'dynasty_x_delta_pubwelf', 'dynasty_x_ira',
                'region']

X_raw = df_model[feature_cols].copy()
y = df_model['won'].values

# One-hot encode region
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X_raw[['region']])
region_dummies = encoder.get_feature_names_out(['region'])

# Separate continuous features (excluding region)
X_cont = X_raw.drop(columns=['region']).values
cont_feature_names = [col for col in feature_cols if col != 'region']

# Combined feature names
all_feature_names = cont_feature_names + list(region_dummies)
X = np.hstack([X_cont, X_cat])

# ------------------------------------------------------------
# Rolling‑window temporal validation with SHAP (including interactions)
# ------------------------------------------------------------
years = sorted(df_model['election_year'].unique())
results = []
shap_records = []

for i, test_year in enumerate(years):
    if test_year <= years[0]:
        continue
    train_mask = df_model['election_year'] < test_year
    test_mask = df_model['election_year'] == test_year
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        continue
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              eval_metric='logloss')
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results.append((test_year, acc))
    print(f"Test year {test_year}: accuracy = {acc:.3f} (n_train={train_mask.sum()}, n_test={test_mask.sum()})")
    
    # SHAP computation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    yearly_shap = pd.DataFrame([mean_abs_shap], columns=all_feature_names, index=[test_year])
    shap_records.append(yearly_shap)

# ------------------------------------------------------------
# Plot and save rolling accuracy (with corrected y-axis)
# ------------------------------------------------------------
years_plot, acc_plot = zip(*results)
plt.figure(figsize=(8, 5))
plt.plot(years_plot, acc_plot, 'o-', color='steelblue', linewidth=2, markersize=8)
plt.axhline(y=0.5, color='red', linestyle='--', label='Random guess (0.5)')
plt.axvline(x=2016, color='orange', linestyle='--', label='2016 election')
plt.xlabel('Test election year')
plt.ylabel('Accuracy')
plt.title('Rolling‑window temporal validation (with interactions)')
plt.ylim(0, 1)                     # Full y-axis range – honest scaling
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../data/rolling_cv.png', dpi=300)
plt.close()
print("Rolling CV plot saved to ../data/rolling_cv.png")

# ------------------------------------------------------------
# Save temporal SHAP drift results
# ------------------------------------------------------------
if shap_records:
    shap_drift_df = pd.concat(shap_records)
    shap_drift_df.to_csv('../data/temporal_shap_drift.csv')
    print(f"Temporal SHAP drift saved to ../data/temporal_shap_drift.csv (shape: {shap_drift_df.shape})")
else:
    print("No SHAP records were generated.")