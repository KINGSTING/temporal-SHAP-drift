"""
change_point_detection.py
Uses the ruptures library to detect change points in SHAP importance time series.
Focuses on key features from the drift analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

# ------------------------------------------------------------
# Load SHAP drift data
# ------------------------------------------------------------
shap_df = pd.read_csv('../data/temporal_shap_drift.csv', index_col=0)
print("SHAP data shape:", shap_df.shape)
years = shap_df.index.values
print("Election years:", years)

# ------------------------------------------------------------
# Features to analyse (key features showing large drift)
# ------------------------------------------------------------
features = [
    'dynasty_x_ira',
    'ira_share',
    'delta_health',
    'dynasty_x_delta_educ',
    'dynasty_x_delta_pubwelf',
    'local_rev_pc',
    'enc_gol'
]
# Keep only those present
features = [f for f in features if f in shap_df.columns]
print("Analysing features:", features)

# ------------------------------------------------------------
# Parameters for change point detection
# ------------------------------------------------------------
# We have only 7 data points, so we limit the number of change points to 1 (or at most 2)
# Using PELT (Pruned Exact Linear Time) with a small penalty to allow a single break.
# Alternatively, use binary segmentation with number of breakpoints = 1.

results = {}

for feat in features:
    series = shap_df[feat].values
    # Reshape for ruptures (need 2D array: (n_samples, 1))
    signal = series.reshape(-1, 1)
    
    # Method 1: Binary segmentation with 1 breakpoint
    algo = rpt.Binseg(model="l2").fit(signal)
    breakpoints = algo.predict(n_bkps=1)  # returns indices of breakpoints (last index is end)
    # breakpoints are given as [bk1, bk2, ... , n_samples]
    # The actual break is at index breakpoints[0] (0-based)
    bk_idx = breakpoints[0] - 1  # convert to 0-based index of the last point before break
    bk_year = years[bk_idx] if bk_idx < len(years) else years[-1]
    
    # Method 2: PELT with a penalty (suitable for short series)
    # penalty can be adjusted; here we use a heuristic based on variance
    # For very short series, we expect 0 or 1 break. We'll set penalty high to avoid overfitting.
    penalty = 0.5 * np.std(series) ** 2 * np.log(len(series))
    algo_pelt = rpt.Pelt(model="l2", min_size=2, jump=1).fit(signal)
    breakpoints_pelt = algo_pelt.predict(pen=penalty)
    # breakpoints_pelt is a list; last element is n_samples; we take the first internal break
    pelt_breaks = [b for b in breakpoints_pelt if b < len(series)]
    pelt_year = years[pelt_breaks[0] - 1] if pelt_breaks else None
    
    results[feat] = {
        'binary_seg_break_index': bk_idx,
        'binary_seg_break_year': bk_year,
        'pelt_breaks': breakpoints_pelt,
        'pelt_break_years': [years[b-1] for b in pelt_breaks] if pelt_breaks else []
    }
    
    print(f"\n{feat}:")
    print(f"  Binary segmentation (1 break) at year: {bk_year} (index {bk_idx})")
    if pelt_breaks:
        print(f"  PELT break year(s): {[years[b-1] for b in pelt_breaks]}")
    else:
        print("  PELT found no break with the chosen penalty.")

# ------------------------------------------------------------
# Plot each feature's time series with detected breakpoints
# ------------------------------------------------------------
fig, axes = plt.subplots(len(features), 1, figsize=(10, 3 * len(features)))
if len(features) == 1:
    axes = [axes]

for idx, feat in enumerate(features):
    ax = axes[idx]
    ax.plot(years, shap_df[feat], marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
    # Mark the binary segmentation break
    if results[feat]['binary_seg_break_year'] is not None:
        ax.axvline(x=results[feat]['binary_seg_break_year'], color='red', linestyle='--', alpha=0.7, label='Binary segmentation break')
    # Optionally mark PELT breaks
    pelt_years = results[feat]['pelt_break_years']
    for py in pelt_years:
        ax.axvline(x=py, color='orange', linestyle=':', alpha=0.7, label='PELT break' if py == pelt_years[0] else '')
    ax.set_title(feat, fontsize=12)
    ax.set_xlabel('Election year')
    ax.set_ylabel('Mean |SHAP|')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('../data/change_point_detection.png', dpi=300)
plt.close()
print("\nChange point detection plot saved to ../data/change_point_detection.png")

# ------------------------------------------------------------
# Summary of break years
# ------------------------------------------------------------
summary_df = pd.DataFrame([
    {'feature': feat,
     'binary_break_year': results[feat]['binary_seg_break_year'],
     'pelt_break_years': str(results[feat]['pelt_break_years'])}
    for feat in features
])
summary_df.to_csv('../data/change_point_summary.csv', index=False)
print("\nSummary saved to ../data/change_point_summary.csv")
print(summary_df.to_string(index=False))