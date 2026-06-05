import pandas as pd
import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float_

import re
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Helper: clean LGU names
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
# 1. Load spending panel (full_panel_all_sectors.csv)
# ------------------------------------------------------------
spending = pd.read_csv('../data/full_panel_all_sectors.csv')
print("Spending panel shape:", spending.shape)

health_pre = 'health_mn_pre3'
health_post = 'health_mn_post3'
educ_pre = 'educ_mn_pre3'
educ_post = 'educ_mn_post3'

# ------------------------------------------------------------
# 2. Load population data (interpolated)
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
# 3. Merge population and compute per capita growth rates
# ------------------------------------------------------------
spending['LGU_clean'] = spending['LGU_clean'].apply(clean_lgu_name)
pop_long.rename(columns={'year': 'election_year'}, inplace=True)
spending = spending.merge(pop_long, on=['LGU_clean', 'election_year'], how='left')

spending['health_pre_pc'] = (spending[health_pre] * 1_000_000) / spending['population']
spending['health_post_pc'] = (spending[health_post] * 1_000_000) / spending['population']
spending['educ_pre_pc'] = (spending[educ_pre] * 1_000_000) / spending['population']
spending['educ_post_pc'] = (spending[educ_post] * 1_000_000) / spending['population']
spending = spending.dropna(subset=['health_pre_pc', 'health_post_pc', 'educ_pre_pc', 'educ_post_pc'])

spending['delta_health'] = (spending['health_post_pc'] - spending['health_pre_pc']) / spending['health_pre_pc']
spending['delta_educ'] = (spending['educ_post_pc'] - spending['educ_pre_pc']) / spending['educ_pre_pc']
spending['delta_health'] = spending['delta_health'].clip(-0.9, 5)
spending['delta_educ'] = spending['delta_educ'].clip(-0.9, 5)

# ------------------------------------------------------------
# 4. Compute re‑election target using election data
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
# 5. Prepare feature dataset
# ------------------------------------------------------------
df_model = spending[['LGU_clean', 'election_year', 'delta_health', 'delta_educ',
                     'local_rev_mn', 'enc_gol', 'dynasty', 'income_class', 'region', 'reelected']].copy()
df_model.rename(columns={'reelected': 'won', 'local_rev_mn': 'local_rev_pc'}, inplace=True)
df_model = df_model.drop_duplicates(subset=['LGU_clean', 'election_year'])

prev_margin = election[['LGU_clean', 'year', 'vote_share']].copy()
prev_margin.rename(columns={'year': 'election_year', 'vote_share': 'prev_margin'}, inplace=True)
df_model = df_model.merge(prev_margin, on=['LGU_clean', 'election_year'], how='left')
df_model['prev_margin'] = df_model['prev_margin'].fillna(0.5)

df_model = df_model.dropna(subset=['delta_health', 'delta_educ', 'local_rev_pc', 'enc_gol', 'prev_margin', 'won'])
print(f"Final dataset: {df_model.shape[0]} observations")
print(df_model['won'].value_counts())

# ------------------------------------------------------------
# 6. Feature engineering
# ------------------------------------------------------------
feature_cols = ['delta_health', 'delta_educ', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin', 'income_class', 'region']
X_raw = df_model[feature_cols].copy()
y = df_model['won'].values

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X_raw[['income_class', 'region']])
X_cont = X_raw[['delta_health', 'delta_educ', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin']].values
X = np.hstack([X_cont, X_cat])
feature_names = X_raw.columns.tolist()[:-2] + list(encoder.get_feature_names_out(['income_class', 'region']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------
# 7. Train XGBoost (compatible with SHAP)
# ------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 8. SHAP explanation (only bar plot and dependence plots)
# ------------------------------------------------------------
print("Computing SHAP values...")
background = X_train[:200]
explainer = shap.TreeExplainer(model, background, feature_perturbation='interventional')
shap_values = explainer.shap_values(X_test)

# Take positive class (class 1)
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
else:
    shap_values_pos = shap_values

# 1. SHAP bar plot (mean absolute SHAP)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_pos, X_test, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('../data/shap_bar.png', bbox_inches='tight', dpi=300)
plt.close()
print("SHAP bar plot saved.")

# 2. Dependence plot for delta_educ
try:
    idx_educ = feature_names.index('delta_educ')
    plt.figure()
    shap.dependence_plot(idx_educ, shap_values_pos, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('../data/shap_dependence_educ.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("SHAP dependence plot for delta_educ saved.")
except ValueError:
    print("Feature 'delta_educ' not found in feature_names.")

# 3. Dependence plot for delta_health
try:
    idx_health = feature_names.index('delta_health')
    plt.figure()
    shap.dependence_plot(idx_health, shap_values_pos, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('../data/shap_dependence_health.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("SHAP dependence plot for delta_health saved.")
except ValueError:
    print("Feature 'delta_health' not found in feature_names.")

print("All SHAP plots saved to ../data/")