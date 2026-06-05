import pandas as pd
import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float_

import re
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
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

# Pre/post columns for health, education, public welfare
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
# 5. Prepare feature dataset (include ira_share and delta_pubwelf)
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

# ------------------------------------------------------------
# 6. Feature engineering (one‑hot encode categoricals)
# ------------------------------------------------------------
feature_cols = ['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin', 'ira_share', 'income_class', 'region']
X_raw = df_model[feature_cols].copy()
y = df_model['won'].values

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X_raw[['income_class', 'region']])
X_cont = X_raw[['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin', 'ira_share']].values
X = np.hstack([X_cont, X_cat])
feature_names = ['delta_health', 'delta_educ', 'delta_pubwelf', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin', 'ira_share'] + list(encoder.get_feature_names_out(['income_class', 'region']))

# ------------------------------------------------------------
# 7. Manual Stratified 5‑fold cross‑validation for XGBoost
# ------------------------------------------------------------
print("\n=== Stratified 5‑fold CV for XGBoost ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

acc_scores = []
auc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_val)
    y_proba = xgb_model.predict_proba(X_val)[:, 1]
    acc_scores.append(accuracy_score(y_val, y_pred))
    auc_scores.append(roc_auc_score(y_val, y_proba))
    prec_scores.append(precision_score(y_val, y_pred))
    rec_scores.append(recall_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred))

print(f"Accuracy: mean = {np.mean(acc_scores):.3f} (+/- {np.std(acc_scores):.3f})")
print(f"AUC: mean = {np.mean(auc_scores):.3f} (+/- {np.std(auc_scores):.3f})")
print(f"Precision: mean = {np.mean(prec_scores):.3f}")
print(f"Recall: mean = {np.mean(rec_scores):.3f}")
print(f"F1: mean = {np.mean(f1_scores):.3f}")

# ------------------------------------------------------------
# 8. Baseline models on the same folds (manual loop)
# ------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
}

print("\n=== Comparison of Models (5‑fold CV) ===")
for name, model in models.items():
    acc_list = []
    auc_list = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)
        y_proba = model_clone.predict_proba(X_val)[:, 1]
        acc_list.append(accuracy_score(y_val, y_pred))
        auc_list.append(roc_auc_score(y_val, y_proba))
    print(f"{name}: Accuracy = {np.mean(acc_list):.3f} (+/- {np.std(acc_list):.3f}), AUC = {np.mean(auc_list):.3f} (+/- {np.std(auc_list):.3f})")

# ------------------------------------------------------------
# 9. Calibration curve for XGBoost (using full training and a test split)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
xgb_model.fit(X_train, y_train)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration curve')
plt.legend()
plt.tight_layout()
plt.savefig('../data/calibration_curve.png', dpi=300)
plt.close()
print("Calibration curve saved to ../data/calibration_curve.png")

# ------------------------------------------------------------
# 10. Save cross‑validation results to CSV
# ------------------------------------------------------------
cv_results = pd.DataFrame({
    'fold': range(1, 6),
    'accuracy': acc_scores,
    'auc': auc_scores,
    'precision': prec_scores,
    'recall': rec_scores,
    'f1': f1_scores
})
cv_results.to_csv('../data/xgboost_cv_results.csv', index=False)
print("CV results saved to ../data/xgboost_cv_results.csv")

print("\nAll evaluation completed.")