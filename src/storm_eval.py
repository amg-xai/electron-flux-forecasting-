import pandas as pd
import numpy as np
import torch
import pickle
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ElectronFluxLSTM

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "data", "features.csv")

# ── Load model ─────────────────────────────────────────────────────────────────
def load_model():
    import pickle
    scalers = {}
    features = {}
    for horizon in ['1h', '6h', '12h']:
        with open(os.path.join(MODEL_DIR, f'scaler_{horizon}.pkl'), 'rb') as f:
            scalers[horizon] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, f'features_{horizon}.pkl'), 'rb') as f:
            features[horizon] = pickle.load(f)
    
    model = ElectronFluxLSTM(
        size_1h=len(features['1h']),
        size_6h=len(features['6h']),
        size_12h=len(features['12h']),
    )
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'best_model.pt'),
        map_location='cpu', weights_only=True
    ))
    model.eval()
    return model, scalers, features

# ── Rolling prediction over a time window ──────────────────────────────────────
def predict_timeseries(model, scalers, features, df, seq_len=72):
    X_1h  = scalers['1h'].transform(df[features['1h']].values)
    X_6h  = scalers['6h'].transform(df[features['6h']].values)
    X_12h = scalers['12h'].transform(df[features['12h']].values)

    preds_1h  = np.full(len(df), np.nan)
    preds_6h  = np.full(len(df), np.nan)
    preds_12h = np.full(len(df), np.nan)

    t1  = torch.tensor(X_1h,  dtype=torch.float32)
    t6  = torch.tensor(X_6h,  dtype=torch.float32)
    t12 = torch.tensor(X_12h, dtype=torch.float32)

    with torch.no_grad():
        for i in range(seq_len, len(df)):
            x1  = t1[i-seq_len:i].unsqueeze(0)
            x6  = t6[i-seq_len:i].unsqueeze(0)
            x12 = t12[i-seq_len:i].unsqueeze(0)
            p1, p6, p12 = model(x1, x6, x12)
            preds_1h[i]  = float(p1[0])
            preds_6h[i]  = float(p6[0])
            preds_12h[i] = float(p12[0])

    df = df.copy()
    df['pred_1h']  = preds_1h
    df['pred_6h']  = preds_6h
    df['pred_12h'] = preds_12h
    return df

# ── Storm events definition ────────────────────────────────────────────────────
STORM_EVENTS = [
    {
        'name': 'Oct 2015 Storm',
        'start': '2015-10-04',
        'end':   '2015-10-14',
        'peak':  '2015-10-08',
    },
    {
        'name': 'Sep-Oct 2016 Storm',
        'start': '2016-09-24',
        'end':   '2016-10-04',
        'peak':  '2016-09-30',
    },
    {
        'name': 'Oct 2016 Storm',
        'start': '2016-10-20',
        'end':   '2016-10-31',
        'peak':  '2016-10-26',
    },
    {
        'name': 'Apr 2017 Storm',
        'start': '2017-04-18',
        'end':   '2017-04-28',
        'peak':  '2017-04-23',
    },
    {
        'name': 'Oct 2017 Storm',
        'start': '2017-10-09',
        'end':   '2017-10-18',
        'peak':  '2017-10-14',
    },
]

ALERT_THRESHOLD = 3.5  # log10 flux — HIGH level

# ── Analyze single storm ───────────────────────────────────────────────────────
def analyze_storm(df_pred, storm):
    name  = storm['name']
    start = storm['start']
    end   = storm['end']

    window = df_pred[start:end].copy()
    window = window.dropna(subset=['pred_1h', 'pred_6h', 'pred_12h'])

    actual   = window['log_flux'].values
    pred_1h  = window['pred_1h'].values
    pred_6h  = window['pred_6h'].values
    pred_12h = window['pred_12h'].values
    times    = window.index

    # RMSE per horizon
    mask     = ~np.isnan(actual)
    rmse_1h  = np.sqrt(mean_squared_error(actual[mask], pred_1h[mask]))
    rmse_6h  = np.sqrt(mean_squared_error(actual[mask], pred_6h[mask]))
    rmse_12h = np.sqrt(mean_squared_error(actual[mask], pred_12h[mask]))

    # Find peak time first
    peak_idx  = np.argmax(actual)
    peak_time = times[peak_idx]
    peak_flux = actual[peak_idx]

    # Find alert crossings BEFORE peak only
    actual_alert_time   = None
    pred_6h_alert_time  = None
    pred_12h_alert_time = None

    for i, t in enumerate(times):
        if t >= peak_time:
            break
        if actual[i] >= ALERT_THRESHOLD and actual_alert_time is None:
            actual_alert_time = t
        if pred_6h[i] >= ALERT_THRESHOLD and pred_6h_alert_time is None:
            pred_6h_alert_time = t
        if pred_12h[i] >= ALERT_THRESHOLD and pred_12h_alert_time is None:
            pred_12h_alert_time = t

    lead_6h  = (actual_alert_time - pred_6h_alert_time).total_seconds() / 3600 \
               if (actual_alert_time and pred_6h_alert_time) else None
    lead_12h = (actual_alert_time - pred_12h_alert_time).total_seconds() / 3600 \
               if (actual_alert_time and pred_12h_alert_time) else None

    print(f"\n{'='*60}")
    print(f"Storm: {name}")
    print(f"Period: {start} to {end}")
    print(f"Peak flux: {peak_flux:.3f} log10 ({10**peak_flux:.0f} e/cm²/s/sr) at {peak_time}")
    print(f"RMSE  — 1h: {rmse_1h:.4f} | 6h: {rmse_6h:.4f} | 12h: {rmse_12h:.4f}")
    print(f"Actual first alert:   {actual_alert_time}")
    print(f"6h pred first alert:  {pred_6h_alert_time}  | Lead: {f'{lead_6h:.1f}h' if lead_6h is not None else 'N/A'}")
    print(f"12h pred first alert: {pred_12h_alert_time} | Lead: {f'{lead_12h:.1f}h' if lead_12h is not None else 'N/A'}")

    return {
        'name': name,
        'peak_flux': peak_flux,
        'rmse_1h': rmse_1h,
        'rmse_6h': rmse_6h,
        'rmse_12h': rmse_12h,
        'actual_alert': actual_alert_time,
        'pred_6h_alert': pred_6h_alert_time,
        'pred_12h_alert': pred_12h_alert_time,
        'lead_6h': lead_6h,
        'lead_12h': lead_12h,
        'window': window,
        'times': times,
        'actual': actual,
        'pred_1h': pred_1h,
        'pred_6h': pred_6h,
        'pred_12h': pred_12h,
    }
# ── Plot storm ─────────────────────────────────────────────────────────────────
def plot_storm(result, out_dir):
    name     = result['name']
    times    = result['times']
    actual   = result['actual']
    pred_1h  = result['pred_1h']
    pred_6h  = result['pred_6h']
    pred_12h = result['pred_12h']
    window   = result['window']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Storm Analysis: {name}', fontsize=14, fontweight='bold')

    # Panel 1: Flux actual vs predicted
    ax1 = axes[0]
    ax1.plot(times, actual,   color='cyan',   linewidth=2,   label='Actual flux',    zorder=5)
    ax1.plot(times, pred_1h,  color='lime',   linewidth=1.5, label='1h forecast',    linestyle='--')
    ax1.plot(times, pred_6h,  color='yellow', linewidth=1.5, label='6h forecast',    linestyle='--')
    ax1.plot(times, pred_12h, color='orange', linewidth=1.5, label='12h forecast',   linestyle='--')
    ax1.axhline(y=ALERT_THRESHOLD, color='red', linestyle='--', alpha=0.8, label=f'Alert threshold ({ALERT_THRESHOLD})')
    ax1.axhline(y=4.0, color='darkred', linestyle='--', alpha=0.8, label='Severe threshold (4.0)')

    # Mark alert times
    if result['actual_alert']:
        ax1.axvline(x=result['actual_alert'], color='cyan', linestyle=':', linewidth=2, label='Actual alert')
    if result['pred_6h_alert']:
        ax1.axvline(x=result['pred_6h_alert'], color='yellow', linestyle=':', linewidth=2, label='6h pred alert')
    if result['pred_12h_alert']:
        ax1.axvline(x=result['pred_12h_alert'], color='orange', linestyle=':', linewidth=2, label='12h pred alert')

    ax1.set_ylabel('log₁₀ Flux (>2 MeV)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('Electron Flux: Actual vs Predicted')
    ax1.grid(alpha=0.3)

    # Panel 2: Solar wind speed
    ax2 = axes[1]
    ax2.plot(times, window['Vsw'].values, color='lime', linewidth=1.5)
    ax2.axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='500 km/s')
    ax2.axhline(y=600, color='red',    linestyle='--', alpha=0.5, label='600 km/s')
    ax2.set_ylabel('Solar Wind Speed (km/s)')
    ax2.legend(fontsize=8)
    ax2.set_title('Solar Wind Speed (Vsw)')
    ax2.grid(alpha=0.3)

    # Panel 3: IMF Bz
    ax3 = axes[2]
    ax3.plot(times, window['Bz_GSM'].values, color='magenta', linewidth=1.5)
    ax3.axhline(y=0,   color='white',  linestyle='-',  alpha=0.3)
    ax3.axhline(y=-10, color='red',    linestyle='--', alpha=0.5, label='Bz=-10nT')
    ax3.fill_between(times, window['Bz_GSM'].values, 0,
                     where=window['Bz_GSM'].values < 0,
                     color='red', alpha=0.2, label='Southward Bz')
    ax3.set_ylabel('Bz GSM (nT)')
    ax3.legend(fontsize=8)
    ax3.set_title('IMF Bz (Storm Driver)')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fname = name.replace(' ', '_').replace('/', '_') + '.png'
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {fpath}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, index_col='time', parse_dates=True)

    print("Loading model...")
    model, scalers, features = load_model()

    print("Running predictions over full dataset (this takes ~2 min)...")
    df_pred = predict_timeseries(model, scalers, features, df)

    # Save predictions
    pred_path = os.path.join(BASE, "data", "predictions.csv")
    df_pred[['log_flux','pred_1h','pred_6h','pred_12h']].to_csv(pred_path)
    print(f"Predictions saved to {pred_path}")

    # Analyze each storm
    out_dir = os.path.join(BASE, "notebooks", "storm_plots")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for storm in STORM_EVENTS:
        result = analyze_storm(df_pred, storm)
        plot_storm(result, out_dir)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("STORM EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Storm':<25} {'Peak':>6} {'RMSE_1h':>8} {'RMSE_6h':>8} {'RMSE_12h':>9} {'Lead_6h':>8} {'Lead_12h':>9}")
    print("-"*75)
    for r in results:
        lead_6h  = f"{r['lead_6h']:.1f}h"  if r['lead_6h']  else "N/A"
        lead_12h = f"{r['lead_12h']:.1f}h" if r['lead_12h'] else "N/A"
        print(f"{r['name']:<25} {r['peak_flux']:>6.3f} {r['rmse_1h']:>8.4f} {r['rmse_6h']:>8.4f} {r['rmse_12h']:>9.4f} {lead_6h:>8} {lead_12h:>9}")

    # Overall lead time stats
    lead_6h_vals  = [r['lead_6h']  for r in results if r['lead_6h']  is not None]
    lead_12h_vals = [r['lead_12h'] for r in results if r['lead_12h'] is not None]

    if lead_6h_vals:
        print(f"\nAverage 6h forecast lead time:  {np.mean(lead_6h_vals):.1f} hours")
    if lead_12h_vals:
        print(f"Average 12h forecast lead time: {np.mean(lead_12h_vals):.1f} hours")