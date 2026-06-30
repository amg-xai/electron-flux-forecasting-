import pandas as pd
import numpy as np
import torch
import pickle
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classifier import StormClassifier, CLASSIFIER_FEATURES

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "data", "features.csv")
ALERT_THRESHOLD = 3.5

STORM_EVENTS = [
    {'name': 'Oct 2015 Storm',      'start': '2015-10-04', 'end': '2015-10-14'},
    {'name': 'Sep-Oct 2016 Storm',  'start': '2016-09-24', 'end': '2016-10-04'},
    {'name': 'Oct 2016 Storm',      'start': '2016-10-20', 'end': '2016-10-31'},
    {'name': 'Apr 2017 Storm',      'start': '2017-04-18', 'end': '2017-04-28'},
    {'name': 'Oct 2017 Storm',      'start': '2017-10-09', 'end': '2017-10-18'},
]


def load_classifier():
    with open(os.path.join(MODEL_DIR, 'classifier_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    model = StormClassifier(input_size=len(CLASSIFIER_FEATURES))
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'classifier_best.pt'),
        map_location='cpu', weights_only=True
    ))
    model.eval()
    return model, scaler


def predict_timeseries(model, scaler, df, seq_len=72):
    X = scaler.transform(df[CLASSIFIER_FEATURES].values)
    t = torch.tensor(X, dtype=torch.float32)

    probs_6h  = np.full(len(df), np.nan)
    probs_12h = np.full(len(df), np.nan)

    with torch.no_grad():
        for i in range(seq_len, len(df)):
            x = t[i-seq_len:i].unsqueeze(0)
            p6, p12 = model(x)
            probs_6h[i]  = float(p6[0])
            probs_12h[i] = float(p12[0])

    df = df.copy()
    df['prob_6h']  = probs_6h
    df['prob_12h'] = probs_12h
    return df


def find_sustained_alert(times, probs, peak_time, threshold=0.5, min_consecutive=3):
    """Find first time probability crosses threshold AND stays above it for min_consecutive hours."""
    above = probs >= threshold
    for i in range(len(times) - min_consecutive):
        if times[i] >= peak_time:
            break
        if all(above[i:i+min_consecutive]):
            return times[i]
    return None


def analyze_storm_classifier(df_pred, storm, prob_threshold=0.5):
    name  = storm['name']
    start = storm['start']
    end   = storm['end']

    window = df_pred[start:end].copy()
    window = window.dropna(subset=['prob_6h', 'prob_12h'])

    actual   = window['log_flux'].values
    prob_6h  = window['prob_6h'].values
    prob_12h = window['prob_12h'].values
    times    = window.index

    peak_idx  = np.argmax(actual)
    peak_time = times[peak_idx]

    # Actual first threshold crossing before peak
    actual_alert_time = None
    for i, t in enumerate(times):
        if t >= peak_time:
            break
        if actual[i] >= ALERT_THRESHOLD and actual_alert_time is None:
            actual_alert_time = t

    # Classifier alert — requires 3 consecutive hours above threshold (filters noise spikes)
    clf_6h_alert  = find_sustained_alert(times, prob_6h,  peak_time, prob_threshold, min_consecutive=6)
    clf_12h_alert = find_sustained_alert(times, prob_12h, peak_time, prob_threshold, min_consecutive=6)

    lead_6h  = (actual_alert_time - clf_6h_alert).total_seconds() / 3600 \
               if (actual_alert_time and clf_6h_alert) else None
    lead_12h = (actual_alert_time - clf_12h_alert).total_seconds() / 3600 \
               if (actual_alert_time and clf_12h_alert) else None

    print(f"\n{'='*60}")
    print(f"Storm: {name}")
    print(f"Actual first threshold crossing: {actual_alert_time}")
    print(f"Classifier 6h alert:  {clf_6h_alert}  | Lead: {f'{lead_6h:.1f}h' if lead_6h is not None else 'N/A'}")
    print(f"Classifier 12h alert: {clf_12h_alert} | Lead: {f'{lead_12h:.1f}h' if lead_12h is not None else 'N/A'}")

    return {
        'name': name, 'lead_6h': lead_6h, 'lead_12h': lead_12h,
        'times': times, 'actual': actual, 'prob_6h': prob_6h, 'prob_12h': prob_12h,
        'actual_alert': actual_alert_time, 'clf_6h_alert': clf_6h_alert, 'clf_12h_alert': clf_12h_alert,
    }


def plot_storm_classifier(result, out_dir):
    name  = result['name']
    times = result['times']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'Early Warning Classifier: {name}', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(times, result['actual'], color='cyan', linewidth=2, label='Actual flux')
    ax1.axhline(y=ALERT_THRESHOLD, color='red', linestyle='--', alpha=0.7, label='Alert threshold')
    if result['actual_alert']:
        ax1.axvline(x=result['actual_alert'], color='cyan', linestyle=':', linewidth=2, label='Actual storm onset')
    if result['clf_6h_alert']:
        ax1.axvline(x=result['clf_6h_alert'], color='yellow', linestyle=':', linewidth=2, label='6h classifier alert')
    if result['clf_12h_alert']:
        ax1.axvline(x=result['clf_12h_alert'], color='orange', linestyle=':', linewidth=2, label='12h classifier alert')
    ax1.set_ylabel('log₁₀ Flux')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_title('Electron Flux')
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(times, result['prob_6h'],  color='yellow', linewidth=1.5, label='P(storm in 6h)')
    ax2.plot(times, result['prob_12h'], color='orange', linewidth=1.5, label='P(storm in 12h)')
    ax2.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='Decision threshold')
    if result['actual_alert']:
        ax2.axvline(x=result['actual_alert'], color='cyan', linestyle=':', linewidth=2)
    ax2.set_ylabel('Storm Probability')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    ax2.set_title('Early Warning Classifier Output')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fname = name.replace(' ', '_').replace('/', '_') + '_classifier.png'
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {fpath}")


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, index_col='time', parse_dates=True)

    print("Loading classifier...")
    model, scaler = load_classifier()

    print("Running predictions (this takes ~2 min)...")
    df_pred = predict_timeseries(model, scaler, df)

    out_dir = os.path.join(BASE, "notebooks", "storm_plots")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for storm in STORM_EVENTS:
        result = analyze_storm_classifier(df_pred, storm)
        plot_storm_classifier(result, out_dir)
        results.append(result)

    print(f"\n{'='*60}")
    print("CLASSIFIER LEAD TIME SUMMARY")
    print(f"{'='*60}")
    print(f"{'Storm':<25} {'Lead_6h':>10} {'Lead_12h':>10}")
    print("-"*47)
    for r in results:
        l6  = f"{r['lead_6h']:.1f}h"  if r['lead_6h']  is not None else "N/A"
        l12 = f"{r['lead_12h']:.1f}h" if r['lead_12h'] is not None else "N/A"
        print(f"{r['name']:<25} {l6:>10} {l12:>10}")

    lead_6h_vals  = [r['lead_6h']  for r in results if r['lead_6h']  is not None]
    lead_12h_vals = [r['lead_12h'] for r in results if r['lead_12h'] is not None]
    if lead_6h_vals:
        print(f"\nAverage 6h classifier lead time:  {np.mean(lead_6h_vals):.1f} hours")
    if lead_12h_vals:
        print(f"Average 12h classifier lead time: {np.mean(lead_12h_vals):.1f} hours")