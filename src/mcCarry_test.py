import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

def mccrary_test(data, cutoff=0, binwidth=None, n_bins=None):
    # ... (same function as before, unchanged) ...
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    if binwidth is None:
        if n_bins is None:
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            n_bins = int(2 * iqr / (len(data) ** (1/3)))
            n_bins = max(20, n_bins)
        binwidth = (data.max() - data.min()) / n_bins
    
    bins = np.arange(data.min(), data.max() + binwidth, binwidth)
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_idx = np.digitize(cutoff, bin_edges) - 1
    if bin_idx < 0 or bin_idx >= len(counts):
        raise ValueError("Cutoff not within bin range")
    
    left_counts = counts[:bin_idx+1]
    right_counts = counts[bin_idx+1:]
    left_centers = bin_centers[:bin_idx+1]
    right_centers = bin_centers[bin_idx+1:]
    
    if len(left_counts) >= 2:
        X_left = left_centers.reshape(-1, 1)
        y_left = np.log(left_counts + 0.5)
        model_left = LinearRegression().fit(X_left, y_left)
        log_density_left = model_left.predict([[cutoff]])[0]
    else:
        log_density_left = np.log(left_counts[-1] + 0.5)
    
    if len(right_counts) >= 2:
        X_right = right_centers.reshape(-1, 1)
        y_right = np.log(right_counts + 0.5)
        model_right = LinearRegression().fit(X_right, y_right)
        log_density_right = model_right.predict([[cutoff]])[0]
    else:
        log_density_right = np.log(right_counts[0] + 0.5)
    
    theta = log_density_left - log_density_right
    
    n_bootstrap = 499
    boot_theta = []
    np.random.seed(12345)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_counts, _ = np.histogram(sample, bins=bins)
        boot_bin_idx = np.digitize(cutoff, bins) - 1
        if boot_bin_idx < 0 or boot_bin_idx >= len(boot_counts):
            continue
        left_b = boot_counts[:boot_bin_idx+1]
        right_b = boot_counts[boot_bin_idx+1:]
        if len(left_b) < 2 or len(right_b) < 2:
            continue
        log_left_b = np.log(left_b + 0.5)
        log_right_b = np.log(right_b + 0.5)
        X_left_b = bin_centers[:boot_bin_idx+1].reshape(-1, 1)
        X_right_b = bin_centers[boot_bin_idx+1:].reshape(-1, 1)
        try:
            m_left = LinearRegression().fit(X_left_b, log_left_b)
            m_right = LinearRegression().fit(X_right_b, log_right_b)
            theta_b = m_left.predict([[cutoff]])[0] - m_right.predict([[cutoff]])[0]
            boot_theta.append(theta_b)
        except:
            continue
    
    se = np.std(boot_theta) if boot_theta else np.nan
    z = theta / se if se > 0 else np.nan
    p_value = 2 * (1 - norm.cdf(np.abs(z))) if not np.isnan(z) else np.nan
    
    plt.figure(figsize=(8,5))
    plt.hist(data, bins=bins, density=False, alpha=0.5, edgecolor='black')
    plt.axvline(cutoff, color='red', linestyle='--', linewidth=2, label='Cutoff')
    plt.xlabel('Vote margin (vote share - 0.5)')
    plt.ylabel('Frequency')
    plt.title('McCrary Density Test')
    plt.legend()
    plt.tight_layout()
    
    return {'theta': theta, 'se': se, 'z': z, 'p_value': p_value, 'plot': plt.gcf()}


# ========== CORRECTED SCRIPT ==========
file_path = "/home/jemarjohn/Documents/Research/mayors-slack-off/data/fiscal+electoral_data_July 2025.xlsx"
df = pd.read_excel(file_path)

print("Columns available:", df.columns.tolist())
print(f"Number of rows: {len(df)}")

# Check if a margin column already exists (e.g., 'prev_margin', 'margin')
margin_col = None
for col in ['prev_margin', 'margin', 'vote_margin', 'win_margin']:
    if col in df.columns:
        margin_col = col
        break

if margin_col:
    margin = df[margin_col].dropna()
    print(f"Using existing margin column '{margin_col}'")
else:
    # Compute margin from votes and total votes
    if 'votes' in df.columns and 'totvot' in df.columns:
        # Ensure numeric
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
        df['totvot'] = pd.to_numeric(df['totvot'], errors='coerce')
        # Compute vote share
        df['vote_share'] = df['votes'] / df['totvot']
        # Keep only valid shares between 0 and 1
        df = df[(df['vote_share'] >= 0) & (df['vote_share'] <= 1)]
        margin = df['vote_share'] - 0.5
        print(f"Computed margin from 'votes' and 'totvot'. Valid rows: {len(margin)}")
    else:
        raise ValueError("No margin column and no 'votes'/'totvot' columns to compute margin.")

# Drop missing
margin = margin.dropna()
print(f"Margin range: [{margin.min():.4f}, {margin.max():.4f}]")

# Optionally filter to plausible range [-0.5, 0.5]
margin = margin[(margin >= -0.5) & (margin <= 0.5)]
print(f"After filtering to [-0.5, 0.5]: {len(margin)} observations")

# Run McCrary test
result = mccrary_test(margin, cutoff=0)

print("\n=== McCrary Density Test Results ===")
print(f"θ (log difference in densities): {result['theta']:.4f}")
print(f"Bootstrap standard error: {result['se']:.4f}")
print(f"Z-statistic: {result['z']:.4f}")
print(f"P-value: {result['p_value']:.4f}")

# Save plot
result['plot'].savefig("/home/jemarjohn/Documents/Research/mayors-slack-off/src/mccrary_plot.png", dpi=300)
print("Plot saved as 'mccrary_plot.png' in src/ folder.")