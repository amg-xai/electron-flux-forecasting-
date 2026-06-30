import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load GRASP data
grasp = pd.read_csv(os.path.join(base, 'data', 'grasp_clean.csv'), index_col='time', parse_dates=True)

# Resample GRASP to hourly to match GOES/OMNI resolution
grasp_hourly = grasp.resample('1h').mean()
grasp_hourly['log_grasp_flux'] = np.log10(grasp_hourly['electron_flux'].clip(lower=0.1))

# Load our GOES-based features (has actual log_flux from GOES-15)
features = pd.read_csv(os.path.join(base, 'data', 'features.csv'), index_col='time', parse_dates=True)

# Merge on overlapping timestamps
merged = grasp_hourly.join(features[['log_flux', 'Vsw', 'Bz_GSM', 'Kp', 'Dst']], how='inner')
merged = merged.dropna()

print(f"Overlapping hours: {len(merged)}")
print(f"Date range: {merged.index[0]} to {merged.index[-1]}")

# Correlation between GOES-15 flux and GRASP flux
corr, pval = pearsonr(merged['log_flux'], merged['log_grasp_flux'])
print(f"\nCorrelation (GOES-15 vs GRASP log flux): r={corr:.3f}, p={pval:.2e}")

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('GOES-15 vs GRASP/GSAT-19 Electron Flux Comparison', fontsize=14, fontweight='bold')

ax1 = axes[0]
ax1.plot(merged.index, merged['log_flux'], color='cyan', linewidth=1, label='GOES-15 (>2 MeV)')
ax1_twin = ax1.twinx()
ax1_twin.plot(merged.index, merged['log_grasp_flux'], color='orange', linewidth=1, label='GRASP/GSAT-19', alpha=0.7)
ax1.set_ylabel('log10 GOES Flux', color='cyan')
ax1_twin.set_ylabel('log10 GRASP Flux', color='orange')
ax1.set_title(f'Flux Comparison (r={corr:.3f})')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(merged.index, merged['Vsw'], color='lime', linewidth=1)
ax2.set_ylabel('Vsw (km/s)')
ax2.set_title('Solar Wind Speed')
ax2.grid(alpha=0.3)

ax3 = axes[2]
ax3.plot(merged.index, merged['Bz_GSM'], color='magenta', linewidth=1)
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax3.set_ylabel('Bz GSM (nT)')
ax3.set_title('IMF Bz')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30)
ax3.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(base, 'notebooks', 'grasp_validation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to {out_path}")

# Save merged data
merged.to_csv(os.path.join(base, 'data', 'grasp_goes_merged.csv'))
print("Merged validation data saved")