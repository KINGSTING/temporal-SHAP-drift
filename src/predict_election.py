import pandas as pd
import numpy as np
import re
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay, precision_recall_fscore_support
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Helper: clean LGU names (same as before)
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
# Merge population and compute per capita growth rates
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
# Prepare feature dataset
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
# Feature engineering
# ------------------------------------------------------------
feature_cols = ['delta_health', 'delta_educ', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin', 'income_class', 'region']
X_raw = df_model[feature_cols].copy()
y = df_model['won'].values

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X_raw[['income_class', 'region']])
X_cont = X_raw[['delta_health', 'delta_educ', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin']].values
X = np.hstack([X_cont, X_cat])
feature_names = X_raw.columns.tolist()[:-2] + list(encoder.get_feature_names_out(['income_class', 'region']))

# ------------------------------------------------------------
# Random split (stratified) – main model
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Random split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

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

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Main Model (Random Split) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# ROC curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title('ROC Curve (XGBoost)')
plt.savefig('../data/roc_curve.png', dpi=300)
plt.close()
print("ROC curve saved.")

# Correlation heatmap
numeric_features = ['delta_health', 'delta_educ', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin', 'income_class']
corr_matrix = df_model[numeric_features + ['won']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('Correlation with Re‑election (won)')
plt.tight_layout()
plt.savefig('../data/correlation_heatmap.png', dpi=300)
plt.close()
print("Correlation heatmap saved.")

# Feature importance bar plot
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)
plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance (XGBoost)')
plt.tight_layout()
plt.savefig('../data/feature_importance_bar.png', bbox_inches='tight')
plt.close()
importance_df.to_csv('../data/feature_importance.csv', index=False)
print("Feature importance bar plot saved.")

# Partial dependence plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
PartialDependenceDisplay.from_estimator(model, X_train, features=[feature_names.index('delta_educ')], ax=ax[0], kind='average')
ax[0].set_title('Partial Dependence: Education Growth')
PartialDependenceDisplay.from_estimator(model, X_train, features=[feature_names.index('delta_health')], ax=ax[1], kind='average')
ax[1].set_title('Partial Dependence: Health Growth')
plt.tight_layout()
plt.savefig('../data/pdp_plots.png', dpi=300)
plt.close()
print("Partial dependence plots saved.")

# Temporal split (robustness check)
df_model['election_year'] = df_model['election_year'].astype(int)
train_df = df_model[df_model['election_year'] <= 2016]
test_df = df_model[df_model['election_year'] > 2016]
if len(train_df) > 0 and len(test_df) > 0:
    X_train_t_cont = train_df[feature_cols][['delta_health', 'delta_educ', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin']].values
    X_train_t_cat = encoder.transform(train_df[feature_cols][['income_class', 'region']])
    X_train_t = np.hstack([X_train_t_cont, X_train_t_cat])
    X_test_t_cont = test_df[feature_cols][['delta_health', 'delta_educ', 'local_rev_pc', 'enc_gol', 'dynasty', 'prev_margin']].values
    X_test_t_cat = encoder.transform(test_df[feature_cols][['income_class', 'region']])
    X_test_t = np.hstack([X_test_t_cont, X_test_t_cat])
    y_train_t = train_df['won'].values
    y_test_t = test_df['won'].values
    model_t = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    model_t.fit(X_train_t, y_train_t)
    y_pred_t = model_t.predict(X_test_t)
    print("\n=== Temporal Split (Robustness) ===")
    print(f"Accuracy: {accuracy_score(y_test_t, y_pred_t):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test_t, model_t.predict_proba(X_test_t)[:,1]):.3f}")
else:
    print("Temporal split skipped (insufficient data).")

print("\nAll outputs saved to ../data/")