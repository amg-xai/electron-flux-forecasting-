import pandas as pd
import numpy as np
import torch
import pickle
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ElectronFluxLSTM
from src.features import get_feature_sets

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")

# Load trained regression model
features_1h, features_6h, features_12h = get_feature_sets()
scalers, features = {}, {}
for h, fl in zip(['1h','6h','12h'], [features_1h, features_6h, features_12h]):
    with open(os.path.join(MODEL_DIR, f'scaler_{h}.pkl'), 'rb') as f:
        scalers[h] = pickle.load(f)
    features[h] = fl

model = ElectronFluxLSTM(
    size_1h=len(features_1h), size_6h=len(features_6h), size_12h=len(features_12h)
)
model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, 'best_model.pt'), map_location='cpu', weights_only=True
))
model.eval()

# Load full feature dataset (has solar wind inputs for the GRASP overlap period)
df = pd.read_csv(os.path.join(BASE, 'data', 'features.csv'), index_col='time', parse_dates=True)

# Load GRASP actual flux
grasp = pd.read_csv(os.path.join(BASE, 'data', 'grasp_clean.csv'), index_col='time', parse_dates=True)
grasp_hourly = grasp.resample('1h').mean()
grasp_hourly['log_grasp_flux'] = np.log10(grasp_hourly['electron_flux'].clip(lower=0.1))

# Run model predictions over the GRASP overlap window
seq_len = 72
overlap_start = grasp_hourly.index.min()
overlap_end   = grasp_hourly.index.max()

# Need 72h of history before overlap_start for the model
model_input_start = overlap_start - pd.Timedelta(hours=seq_len)
df_window = df[model_input_start:overlap_end].copy()

print(f"Running model predictions for {len(df_window)} hours...")

preds_1h = np.full(len(df_window), np.nan)

x1_all = torch.tensor(scalers['1h'].transform(df_window[features['1h']].values), dtype=torch.float32)
x6_all = torch.tensor(scalers['6h'].transform(df_window[features['6h']].values), dtype=torch.float32)
x12_all = torch.tensor(scalers['12h'].transform(df_window[features['12h']].values), dtype=torch.float32)

with torch.no_grad():
    for i in range(seq_len, len(df_window)):
        x1  = x1_all[i-seq_len:i].unsqueeze(0)
        x6  = x6_all[i-seq_len:i].unsqueeze(0)
        x12 = x12_all[i-seq_len:i].unsqueeze(0)
        p1, p6, p12 = model(x1, x6, x12)
        preds_1h[i] = float(p1[0])

df_window['model_pred_1h'] = preds_1h

# Align model predictions (trained on GOES-15 scale) with GRASP actual flux
merged = df_window[['model_pred_1h']].join(grasp_hourly[['log_grasp_flux']], how='inner').dropna()

print(f"\nOverlapping hours for true model-vs-GRASP validation: {len(merged)}")

corr, pval = pearsonr(merged['model_pred_1h'], merged['log_grasp_flux'])
print(f"Correlation: model's 1h-ahead GOES-trained predictions vs actual GRASP flux")
print(f"r={corr:.3f}, p={pval:.2e}")

# Plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(merged.index, merged['model_pred_1h'], color='cyan', label='Model prediction (GOES-trained)', linewidth=1)
ax2 = ax.twinx()
ax2.plot(merged.index, merged['log_grasp_flux'], color='orange', label='GRASP actual flux', linewidth=1, alpha=0.7)
ax.set_ylabel('Model predicted log10 flux', color='cyan')
ax2.set_ylabel('GRASP log10 flux', color='orange')
ax.set_title(f'Model Predictions vs Independent GRASP Measurements (r={corr:.3f})')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
plt.tight_layout()

out_path = os.path.join(BASE, 'notebooks', 'grasp_model_validation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to {out_path}")