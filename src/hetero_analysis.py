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
import warnings
warnings.filterwarnings('ignore')

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
pubwelf_pre = 'pubwelf_mn_pre3'
pubwelf_post = 'pubwelf_mn_post3'

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
# 3. Merge population and compute per capita growth rates
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
# 5. Prepare feature dataset (includes ira_share)
# ------------------------------------------------------------
df_model = spending[['LGU_clean', 'election_year', 'delta_health', 'delta_educ', 'delta_pubwelf',
                     'local_rev_mn', 'enc_gol', 'dynasty', 'income_class', 'region', 'reelected', 'ira_share']].copy()
df_model.rename(columns={'reelected': 'won', 'local_rev_mn': 'local_rev_pc'}, inplace=True)
df_model = df_model.drop_duplicates(subset=['LGU_clean', 'election_year'])

prev_margin = election[['LGU_clean', 'year', 'vote_share']].copy()
prev_margin.rename(columns={'year': 'election_year', 'vote_share': 'prev_margin'}, inplace=True)
df_model = df_model.merge(prev_margin, on=['LGU_clean', 'election_year'], how='left')
df_model['prev_margin'] = df_model['prev_margin'].fillna(0.5)

df_model = df_model.dropna(subset=['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 'prev_margin', 'won', 'ira_share'])
print(f"Final dataset: {df_model.shape[0]} observations")
print(df_model['won'].value_counts())

# Print income class distribution
print("\nIncome class distribution:")
income_dist = df_model['income_class'].value_counts().sort_index()
print(income_dist)
income_dist.to_csv('../data/income_class_distribution.csv')

# ------------------------------------------------------------
# 6. Feature engineering with interaction terms (including new features)
# ------------------------------------------------------------
df_model['dynasty_x_delta_educ'] = df_model['dynasty'] * df_model['delta_educ']
df_model['dynasty_x_delta_health'] = df_model['dynasty'] * df_model['delta_health']
df_model['dynasty_x_delta_pubwelf'] = df_model['dynasty'] * df_model['delta_pubwelf']
df_model['dynasty_x_ira'] = df_model['dynasty'] * df_model['ira_share']

feature_cols = ['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin',
                'ira_share', 'income_class', 'region',
                'dynasty_x_delta_educ', 'dynasty_x_delta_health', 'dynasty_x_delta_pubwelf', 'dynasty_x_ira']
X_raw = df_model[feature_cols].copy()
y = df_model['won'].values

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X_raw[['income_class', 'region']])
X_cont = X_raw[['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin', 'ira_share',
                'dynasty_x_delta_educ', 'dynasty_x_delta_health', 'dynasty_x_delta_pubwelf', 'dynasty_x_ira']].values
X = np.hstack([X_cont, X_cat])
feature_names = (['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin', 'ira_share',
                  'dynasty_x_delta_educ', 'dynasty_x_delta_health', 'dynasty_x_delta_pubwelf', 'dynasty_x_ira'] +
                 list(encoder.get_feature_names_out(['income_class', 'region'])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost
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
# SHAP analysis (disable additivity check)
# ------------------------------------------------------------
print("Computing SHAP values...")
background = X_train[:200]
explainer = shap.TreeExplainer(model, background, feature_perturbation='interventional')
shap_values = explainer.shap_values(X_test, check_additivity=False)
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
else:
    shap_values_pos = shap_values

# Feature importance (gain)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv('../data/feature_importance_with_interactions.csv', index=False)
print("\nFeature importance (with interactions) saved to ../data/feature_importance_with_interactions.csv")
print(importance_df.head(15))

# Mean absolute SHAP
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'mean_shap': np.abs(shap_values_pos).mean(axis=0)
}).sort_values('mean_shap', ascending=False)
shap_importance.to_csv('../data/mean_shap_values.csv', index=False)
print("\nMean |SHAP| values saved to ../data/mean_shap_values.csv")
print(shap_importance.head(15))

# Interaction term SHAP values (for dynasty × health, etc.)
try:
    idx_dyn_health = feature_names.index('dynasty_x_delta_health')
    print(f"\nMean SHAP for dynasty_x_delta_health: {np.mean(shap_values_pos[:, idx_dyn_health]):.4f}")
except ValueError:
    print("dynasty_x_delta_health not found")

try:
    idx_dyn_ira = feature_names.index('dynasty_x_ira')
    print(f"Mean SHAP for dynasty_x_ira: {np.mean(shap_values_pos[:, idx_dyn_ira]):.4f}")
except ValueError:
    print("dynasty_x_ira not found")

# ------------------------------------------------------------
# Subgroup analysis by income class (extract SHAP for key features)
# ------------------------------------------------------------
subgroup_results = []
for inc_class in sorted(df_model['income_class'].unique()):
    subset = df_model[df_model['income_class'] == inc_class]
    n = len(subset)
    if n >= 30:
        X_sub = subset[feature_cols]
        X_cat_sub = encoder.transform(X_sub[['income_class', 'region']])
        X_cont_sub = X_sub[['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin', 'ira_share',
                            'dynasty_x_delta_educ', 'dynasty_x_delta_health', 'dynasty_x_delta_pubwelf', 'dynasty_x_ira']].values
        X_sub_enc = np.hstack([X_cont_sub, X_cat_sub])
        shap_sub = explainer.shap_values(X_sub_enc, check_additivity=False)
        if isinstance(shap_sub, list):
            shap_sub_pos = shap_sub[1]
        else:
            shap_sub_pos = shap_sub
        idx_educ = feature_names.index('delta_educ')
        idx_health = feature_names.index('delta_health')
        idx_pubwelf = feature_names.index('delta_pubwelf')
        idx_dyn = feature_names.index('dynasty')
        idx_ira = feature_names.index('ira_share')
        subgroup_results.append({
            'income_class': inc_class,
            'n': n,
            'mean_shap_delta_educ': np.mean(shap_sub_pos[:, idx_educ]),
            'mean_shap_delta_health': np.mean(shap_sub_pos[:, idx_health]),
            'mean_shap_delta_pubwelf': np.mean(shap_sub_pos[:, idx_pubwelf]),
            'mean_shap_dynasty': np.mean(shap_sub_pos[:, idx_dyn]),
            'mean_shap_ira_share': np.mean(shap_sub_pos[:, idx_ira])
        })
subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv('../data/subgroup_shap_by_income.csv', index=False)
print("\nSubgroup SHAP results saved to ../data/subgroup_shap_by_income.csv")
print(subgroup_df)

print("\nHeterogeneity analysis completed. All outputs saved to ../data/")